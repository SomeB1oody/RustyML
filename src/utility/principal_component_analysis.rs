use super::*;

/// Threshold for determining whether to use parallel processing in PCA computations
/// When n_components >= this threshold, parallel processing is used for variance calculations
const PCA_PARALLEL_THRESHOLD: usize = 64;

/// PCA structure for implementing Principal Component Analysis
///
/// This structure provides functionality for dimensionality reduction using PCA.
/// It allows fitting a model to data, transforming data into principal component space,
/// and retrieving various statistics about the decomposition.
///
/// # Fields
///
/// - `n_components` - Number of principal components to keep in the model
/// - `components` - Principal axes in feature space, representing the directions of maximum variance Shape is (n_components, n_features)
/// - `mean` - Mean of each feature in the training data, used for centering
/// - `explained_variance` - Amount of variance explained by each component
/// - `explained_variance_ratio` - Percentage of variance explained by each component
/// - `singular_values` - Singular values corresponding to each component
///
/// # Examples
/// ```rust
/// use ndarray::{array, Array2};
/// use rustyml::utility::principal_component_analysis::PCA;
///
/// // Create some sample data (3 samples, 4 features)
/// let data = Array2::from_shape_vec((3, 4), vec![
///     1.0, 2.0, 3.0, 4.0,
///     2.0, 3.0, 4.0, 5.0,
///     3.0, 4.0, 5.0, 6.0
/// ]).unwrap();
///
/// // Create a PCA model with 2 components
/// let mut pca = PCA::new(2).expect("Failed to create PCA model");
///
/// // Fit the model to the data
/// pca.fit(&data).expect("Failed to fit PCA model");
///
/// // Transform the data to the principal component space
/// let transformed = pca.transform(&data).expect("Failed to transform data");
/// println!("Transformed data:\n{:?}", transformed);
///
/// // Get the explained variance ratio
/// let variance_ratio = pca.get_explained_variance_ratio().expect("Model not fitted");
/// println!("Explained variance ratio: {:?}", variance_ratio);
///
/// // Transform back to original space
/// let reconstructed = pca.inverse_transform(&transformed).expect("Failed to inverse transform");
/// println!("Reconstructed data:\n{:?}", reconstructed);
/// ```
///
/// # Common use cases
///
/// - Dimensionality reduction: Reduce high-dimensional data to a lower-dimensional space
/// - Data visualization: Project data to 2 or 3 dimensions for visualization
/// - Feature extraction: Extract the most important features from the dataset
/// - Noise filtering: Remove noise by discarding components with low variance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    singular_values: Option<Array1<f64>>,
}

impl Default for PCA {
    /// Default implementation for PCA
    ///
    /// Creates a new PCA instance with default values.
    ///
    /// # Default Values
    ///
    /// - `n_components` - 2 (common for visualization purposes)
    /// - `components` - None (computed during fitting)
    /// - `mean` - None (computed during fitting)
    /// - `explained_variance` - None (computed during fitting)
    /// - `explained_variance_ratio` - None (computed during fitting)
    /// - `singular_values` - None (computed during fitting)
    fn default() -> Self {
        // Default to 2 components which is common for visualization purposes
        Self {
            n_components: 2,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
        }
    }
}

impl PCA {
    /// Creates a new PCA instance with validation
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of principal components to keep (must be > 0)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new PCA instance with the specified number of components, or an error if validation fails
    ///
    /// # Errors
    ///
    /// Returns `ModelError::InputValidationError` if `n_components` is 0
    pub fn new(n_components: usize) -> Result<Self, ModelError> {
        if n_components == 0 {
            return Err(ModelError::InputValidationError(
                "n_components must be greater than 0".to_string(),
            ));
        }

        Ok(PCA {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
        })
    }

    get_field_as_ref!(get_components, components, Option<&Array2<f64>>);
    get_field_as_ref!(get_mean, mean, Option<&Array1<f64>>);
    get_field_as_ref!(
        get_explained_variance,
        explained_variance,
        Option<&Array1<f64>>
    );
    get_field_as_ref!(
        get_explained_variance_ratio,
        explained_variance_ratio,
        Option<&Array1<f64>>
    );
    get_field_as_ref!(get_singular_values, singular_values, Option<&Array1<f64>>);
    get_field!(get_n_components, n_components, usize);

    /// Validates input data for NaN and infinite values
    fn validate_input_data<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<(), ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Check if input data is empty
        if x.nrows() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data is empty".to_string(),
            ));
        }

        // Check for NaN or infinite values using iterator
        // More efficient than collecting and parallelizing for most use cases
        if let Some(((i, j), _)) = x
            .indexed_iter()
            .find(|&(_, &val)| val.is_nan() || val.is_infinite())
        {
            return Err(ModelError::InputValidationError(format!(
                "Input data contains NaN or infinite value at position [{}, {}]",
                i, j
            )));
        }

        Ok(())
    }

    //// Fits the PCA model
    ///
    /// # Parameters
    ///
    /// - `x` - The input data matrix, where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - The instance itself if successful
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - returns if input data is empty, contains NaN/infinite values, or feature count is 0
    /// - `ModelError::ProcessingError` - returns if internal computation (SVD, mean) fails
    ///
    /// # Performance
    ///
    /// Parallel processing is used for variance and singular value calculations when `n_components >= 64`.
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Validate input data
        self.validate_input_data(x)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Check if number of features is valid
        if n_features == 0 {
            return Err(ModelError::InputValidationError(
                "Number of features must be greater than 0".to_string(),
            ));
        }

        // Calculate feature means in parallel using ndarray's built-in method
        let mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| ModelError::ProcessingError("Failed to compute mean".to_string()))?;

        // Center the data more efficiently using broadcasting
        let x_centered = x - &mean;

        // Use SVD for more stable computation
        let x_slice = x_centered.as_slice().ok_or_else(|| {
            ModelError::ProcessingError("Failed to convert x_centered to slice".to_string())
        })?;
        let x_mat = nalgebra::DMatrix::from_row_slice(n_samples, n_features, x_slice);

        // Compute SVD
        let svd = nalgebra::SVD::new(x_mat, true, true);
        let s_vals = svd.singular_values;
        let m = s_vals.len();

        // Limit components to available singular values
        let n_components = self.n_components.min(m);

        // Get principal components from V^T
        let v_t = svd
            .v_t
            .ok_or_else(|| ModelError::ProcessingError("SVD did not compute V^T".to_string()))?;
        let components = Array2::from_shape_fn((n_components, n_features), |(i, j)| v_t.row(i)[j]);

        // Compute explained variance and singular values
        // Use parallel processing only when n_components exceeds threshold
        let variance_and_singular: Vec<(f64, f64)> = if n_components >= PCA_PARALLEL_THRESHOLD {
            (0..n_components)
                .into_par_iter()
                .map(|i| {
                    let s_val = s_vals[i];
                    let exp_var = (s_val * s_val) / ((n_samples - 1) as f64);
                    (exp_var, s_val)
                })
                .collect()
        } else {
            (0..n_components)
                .map(|i| {
                    let s_val = s_vals[i];
                    let exp_var = (s_val * s_val) / ((n_samples - 1) as f64);
                    (exp_var, s_val)
                })
                .collect()
        };

        let (explained_variance, singular_values): (Vec<f64>, Vec<f64>) =
            variance_and_singular.into_iter().unzip();

        let explained_variance = Array1::from(explained_variance);
        let singular_values = Array1::from(singular_values);

        // Compute total variance from already calculated values
        let total_variance: f64 = explained_variance.sum();

        // Calculate explained variance ratio
        let explained_variance_ratio = explained_variance.mapv(|v| v / total_variance);

        // Store results
        self.components = Some(components);
        self.mean = Some(mean);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.singular_values = Some(singular_values);

        Ok(self)
    }

    /// Transforms data into principal component space
    ///
    /// # Parameters
    ///
    /// - `x` - The input data matrix to transform
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - The transformed data in PC space
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - returns if the model has not been fitted yet
    /// - `ModelError::InputValidationError` - returns if input is invalid or feature dimension doesn't match
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        let components = self.components.as_ref().ok_or(ModelError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(ModelError::NotFitted)?;

        // Validate input data
        self.validate_input_data(x)?;

        // Check feature dimension match
        if x.ncols() != mean.len() {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data: expected {}, got {}",
                mean.len(),
                x.ncols()
            )));
        }

        // Center data using broadcasting (more efficient than manual subtraction)
        let x_centered = x - mean;

        // Direct matrix multiplication for transformation
        let transformed = x_centered.dot(&components.t());

        Ok(transformed)
    }

    /// Fits the model and transforms the data
    ///
    /// # Parameters
    ///
    /// - `x` - The input data matrix
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - The transformed data if successful
    ///
    /// # Errors
    ///
    /// - `ModelError` - returns if fitting or transformation fails
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Transforms data from principal component space back to original feature space
    ///
    /// # Parameters
    ///
    /// - `x` - The input data matrix in principal component space
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - The reconstructed data in original space
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - returns if the model has not been fitted yet
    /// - `ModelError::InputValidationError` - returns if input is invalid or component count doesn't match
    pub fn inverse_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        let components = self.components.as_ref().ok_or(ModelError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(ModelError::NotFitted)?;

        // Validate input data
        self.validate_input_data(x)?;

        // Check component dimension match
        if x.ncols() != components.nrows() {
            return Err(ModelError::InputValidationError(format!(
                "Number of components does not match model: expected {}, got {}",
                components.nrows(),
                x.ncols()
            )));
        }

        // Transform back to original feature space using broadcasting
        let x_restored = x.dot(components) + mean;

        Ok(x_restored)
    }

    model_save_and_load_methods!(PCA);
}
