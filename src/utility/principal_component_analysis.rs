use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::error::Error;
use crate::ModelError;
use rayon::prelude::*;

/// # PCA structure for implementing Principal Component Analysis
///
/// This structure provides functionality for dimensionality reduction using PCA.
/// It allows fitting a model to data, transforming data into principal component space,
/// and retrieving various statistics about the decomposition.
///
/// ## Fields
///
/// - `n_components` - Number of principal components to keep in the model
/// - `components` - Principal axes in feature space, representing the directions of maximum variance Shape is (n_components, n_features)
/// - `mean` - Mean of each feature in the training data, used for centering
/// - `explained_variance` - Amount of variance explained by each component
/// - `explained_variance_ratio` - Percentage of variance explained by each component
/// - `singular_values` - Singular values corresponding to each component
///
/// ## Examples
///
/// ```
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
    /// Creates a new PCA instance
    ///
    /// # Parameters
    ///
    /// * `n_components` - Number of principal components to keep
    ///
    /// # Returns
    ///
    /// * `Self` - A new PCA instance with the specified number of components
    pub fn new(n_components: usize) -> Self {
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
        match self.components.as_ref() {
            Some(components) => Ok(components),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the explained variance
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The explained variance array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_explained_variance(&self) -> Result<&Array1<f64>, ModelError> {
        match self.explained_variance.as_ref() {
            Some(explained_variance) => Ok(explained_variance),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the explained variance ratio
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The explained variance ratio array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_explained_variance_ratio(&self) -> Result<&Array1<f64>, ModelError> {
        match self.explained_variance_ratio.as_ref() {
            Some(explained_variance_ratio) => Ok(explained_variance_ratio),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the singular values
    ///
    /// # Returns
    ///
    /// * `Ok(&Array1<f64>)` - The singular values array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_singular_values(&self) -> Result<&Array1<f64>, ModelError> {
        match self.singular_values.as_ref() {
            Some(singular_values) => Ok(singular_values),
            None => Err(ModelError::NotFitted),
        }
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
    /// - Computes the covariance matrix
    /// - Calculates eigenvalues and eigenvectors
    /// - Sorts components by explained variance
    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<&mut Self, Box<dyn Error>> {
        use crate::machine_learning::preliminary_check;
        preliminary_check(x, None)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.n_components == 0 {
            return Err(Box::new(ModelError::InputValidationError(
                "Number of components must be positive.".to_string(),
            )));
        }

        // Calculate feature means in parallel
        let mean: Array1<f64> = (0..n_features)
            .into_par_iter()
            .map(|i| x.column(i).mean().unwrap_or(0.0))
            .collect::<Vec<f64>>()
            .into();

        // Center the data in parallel
        let mut x_centered = x.to_owned();
        x_centered
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for j in 0..n_features {
                    row[j] -= mean[j];
                }
            });

        // Use SVD for computation (no need to compute covariance matrix first)
        // Convert ndarray data to nalgebra DMatrix
        let x_slice = x_centered
            .as_slice()
            .ok_or("Failed to convert x_centered to slice")?;
        let x_mat = nalgebra::DMatrix::from_row_slice(n_samples, n_features, x_slice);

        // Compute SVD, requesting singular values and right singular vectors
        let svd = nalgebra::SVD::new(x_mat, true, true);
        let s_vals = svd.singular_values; // Length is min(n_samples, n_features)
        let m = s_vals.len();
        // Number of principal components to keep: not exceeding the available singular values
        let n_components = self.n_components.min(m);

        // Get the right singular vector matrix V^T as principal components. Each row of V^T is a principal component
        let v_t = svd.v_t.ok_or("SVD did not compute V^T")?;
        let components = Array2::from_shape_fn((n_components, n_features), |(i, j)| {
            v_t.row(i)[j]
        });

        // Calculate explained variance using singular values: eigenvalue = (singular_value^2) / (n_samples - 1)
        let mut explained_variance = Array1::<f64>::zeros(n_components);
        let mut singular_values = Array1::<f64>::zeros(n_components);
        let values: Vec<(usize, f64, f64)> = (0..n_components)
            .into_par_iter()
            .map(|i| {
                let s_val = s_vals[i];
                let exp_var = (s_val * s_val) / ((n_samples - 1) as f64);
                (i, exp_var, s_val)
            })
            .collect();

        for (i, exp_var, s_val) in values {
            explained_variance[i] = exp_var;
            singular_values[i] = s_val;
        }


        // Total variance: calculated using all singular values (note: when n_features > n_samples, remaining features have variance of 0)
        let s_vals_vec: Vec<f64> = s_vals.iter().cloned().collect();
        let total_variance: f64 = s_vals_vec.par_iter()
            .map(|&s| s * s)
            .sum::<f64>() / ((n_samples - 1) as f64);

        // Calculate explained variance ratio
        let explained_variance_ratio = explained_variance.map(|v| v / total_variance);

        // Save results to the struct
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

        // Use ndarray's vectorized operations for centering
        // This creates a view instead of cloning the entire array
        let x_centered = x.view().outer_iter()
            .into_par_iter()
            .map(|row| {
                // Subtract the mean from each row
                let mut centered_row = row.to_owned();
                for (i, &m) in mean.iter().enumerate() {
                    centered_row[i] -= m;
                }
                centered_row
            })
            .collect::<Vec<_>>();

        // Convert the vector collection back to Array2
        let x_centered = Array2::from_shape_vec(
            (x.nrows(), x.ncols()),
            x_centered.into_iter().flat_map(|row| row.into_iter().collect::<Vec<_>>()).collect()
        )?;

        // Transform to principal component space
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

        // Transform back to original feature space
        let mut x_restored = x.dot(components);
        let n_features = mean.len();

        x_restored
            .axis_chunks_iter_mut(Axis(0), 1)
            .into_par_iter()
            .for_each(|mut chunk| {
                for j in 0..n_features {
                    chunk[[0, j]] += mean[j];
                }
            });

        Ok(x_restored)
    }
}