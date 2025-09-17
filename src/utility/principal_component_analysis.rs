use crate::ModelError;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::error::Error;

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
/// let mut pca = PCA::new(2);
///
/// // Fit the model to the data
/// pca.fit(data.view()).expect("Failed to fit PCA model");
///
/// // Transform the data to the principal component space
/// let transformed = pca.transform(data.view()).expect("Failed to transform data");
/// println!("Transformed data:\n{:?}", transformed);
///
/// // Get the explained variance ratio
/// let variance_ratio = pca.get_explained_variance_ratio().expect("Model not fitted");
/// println!("Explained variance ratio: {:?}", variance_ratio);
///
/// // Transform back to original space
/// let reconstructed = pca.inverse_transform(transformed.view()).expect("Failed to inverse transform");
/// println!("Reconstructed data:\n{:?}", reconstructed);
/// ```
///
/// # Common use cases
///
/// - Dimensionality reduction: Reduce high-dimensional data to a lower-dimensional space
/// - Data visualization: Project data to 2 or 3 dimensions for visualization
/// - Feature extraction: Extract the most important features from the dataset
/// - Noise filtering: Remove noise by discarding components with low variance
#[derive(Debug, Clone)]
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    singular_values: Option<Array1<f64>>,
}

/// Default implementation for PCA
impl Default for PCA {
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
    /// * `n_components` - Number of principal components to keep (must be > 0)
    ///
    /// # Returns
    ///
    /// * `PCA` - A new PCA instance with the specified number of components
    ///
    /// # Panics
    ///
    /// Panics if n_components is 0
    pub fn new(n_components: usize) -> Self {
        if n_components == 0 {
            panic!("Number of components must be positive");
        }

        PCA {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
        }
    }

    /// Gets the components matrix
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - The components matrix if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_components(&self) -> Result<&Array2<f64>, ModelError> {
        self.components.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Gets the explained variance
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The explained variance array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_explained_variance(&self) -> Result<&Array1<f64>, ModelError> {
        self.explained_variance
            .as_ref()
            .ok_or(ModelError::NotFitted)
    }

    /// Gets the explained variance ratio
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The explained variance ratio array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_explained_variance_ratio(&self) -> Result<&Array1<f64>, ModelError> {
        self.explained_variance_ratio
            .as_ref()
            .ok_or(ModelError::NotFitted)
    }

    /// Gets the singular values
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The singular values array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_singular_values(&self) -> Result<&Array1<f64>, ModelError> {
        self.singular_values.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Validates input data for NaN and infinite values
    fn validate_input_data(&self, x: ArrayView2<f64>) -> Result<(), ModelError> {
        // Check if input data is empty
        if x.nrows() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data is empty".to_string(),
            ));
        }

        // Parallel validation for better performance on large datasets
        let validation_results: Vec<Option<(usize, usize)>> = x
            .indexed_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|((i, j), &val)| {
                if val.is_nan() || val.is_infinite() {
                    Some((i, j))
                } else {
                    None
                }
            })
            .collect();

        // Find first error position
        if let Some(Some((i, j))) = validation_results.into_iter().find(|x| x.is_some()) {
            return Err(ModelError::InputValidationError(format!(
                "Input data contains NaN or infinite value at position [{}, {}]",
                i, j
            )));
        }

        Ok(())
    }

    /// Fits the PCA model
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix, where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - The instance
    /// - `Err(Box<dyn std::error::Error>)` - If something goes wrong
    ///
    /// # Implementation Details
    ///
    /// - Computes the mean of each feature
    /// - Centers the data by subtracting the mean
    /// - Computes SVD directly instead of eigendecomposition
    /// - Sorts components by explained variance
    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<&mut Self, Box<dyn Error>> {
        // Validate input data
        self.validate_input_data(x)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Calculate feature means in parallel using ndarray's built-in method
        let mean = x.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;

        // Center the data more efficiently using broadcasting
        let x_centered = &x - &mean;

        // Use SVD for more stable computation
        let x_slice = x_centered
            .as_slice()
            .ok_or("Failed to convert x_centered to slice")?;
        let x_mat = nalgebra::DMatrix::from_row_slice(n_samples, n_features, x_slice);

        // Compute SVD
        let svd = nalgebra::SVD::new(x_mat, true, true);
        let s_vals = svd.singular_values;
        let m = s_vals.len();

        // Limit components to available singular values
        let n_components = self.n_components.min(m);

        // Get principal components from V^T
        let v_t = svd.v_t.ok_or("SVD did not compute V^T")?;
        let components = Array2::from_shape_fn((n_components, n_features), |(i, j)| v_t.row(i)[j]);

        // Compute explained variance and singular values in parallel
        let variance_and_singular: Vec<(f64, f64)> = (0..n_components)
            .into_par_iter()
            .map(|i| {
                let s_val = s_vals[i];
                let exp_var = (s_val * s_val) / ((n_samples - 1) as f64);
                (exp_var, s_val)
            })
            .collect();

        let (explained_variance, singular_values): (Vec<f64>, Vec<f64>) =
            variance_and_singular.into_iter().unzip();

        let explained_variance = Array1::from(explained_variance);
        let singular_values = Array1::from(singular_values);

        // Compute total variance more efficiently
        let total_variance = s_vals.iter().map(|&s| s * s).sum::<f64>() / ((n_samples - 1) as f64);

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
    /// * `x` - The input data matrix to transform
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - The transformed data if successful
    /// - `Err(Box<dyn Error>)` - If something goes wrong while processing
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let components = self.components.as_ref().ok_or(ModelError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(ModelError::NotFitted)?;

        // Center data using broadcasting (more efficient than manual subtraction)
        let x_centered = &x - mean;

        // Direct matrix multiplication for transformation
        let transformed = x_centered.dot(&components.t());

        Ok(transformed)
    }

    /// Fits the model and transforms the data
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - The transformed data if successful
    /// - `Err(Box<dyn Error>)` - If something goes wrong while processing
    pub fn fit_transform(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Transforms data from principal component space back to original feature space
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix in principal component space
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - The reconstructed data in original space
    /// - `Err(Box<dyn Error>)` - If something goes wrong while processing
    pub fn inverse_transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let components = self.components.as_ref().ok_or("PCA model not fitted yet")?;
        let mean = self.mean.as_ref().ok_or("PCA model not fitted yet")?;

        // Transform back to original feature space using broadcasting
        let x_restored = x.dot(components) + mean;

        Ok(x_restored)
    }
}
