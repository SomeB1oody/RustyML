use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use crate::ModelError;
use rayon::prelude::*;

/// # Linear Discriminant Analysis (LDA)
///
/// A classifier and dimensionality reduction technique that projects data onto a lower-dimensional space while maintaining class separability.
///
/// ## Fields
/// * `classes` - Array of unique class labels from training data
/// * `priors` - Prior probabilities for each class
/// * `means` - Mean vectors for each class
/// * `cov_inv` - Inverse of the common covariance matrix
/// * `projection` - Projection matrix for dimensionality reduction
///
/// ## Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::linear_discriminant_analysis::LDA;
///
/// // Create feature matrix and class labels
/// let x = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 5.0, 5.0, 5.5, 4.5, 6.0, 5.0]).unwrap();
/// let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// // Create and fit LDA model
/// let mut lda = LDA::new();
/// lda.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let x_new = Array2::from_shape_vec((2, 2), vec![1.2, 2.2, 5.2, 4.8]).unwrap();
/// let predictions = lda.predict(x_new.view()).unwrap();
///
/// // Transform data to lower dimension
/// let x_transformed = lda.transform(x.view(), 1).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LDA {
    /// Array of classes (e.g., class labels represented by i32)
    classes: Option<Array1<i32>>,
    /// Prior probabilities for each class
    priors: Option<Array1<f64>>,
    /// Mean vectors for each class (each row corresponds to a class, shape: (n_classes, n_features))
    means: Option<Array2<f64>>,
    /// Inverse of the common covariance matrix (based on within-class scatter, shape: (n_features, n_features))
    cov_inv: Option<Array2<f64>>,
    /// Projection matrix for dimensionality reduction, each column is a projection vector (max dimension is n_classes - 1)
    projection: Option<Array2<f64>>,
}

/// Default implementation for LDA
impl Default for LDA {
    /// Creates a new LDA instance with default settings
    ///
    /// Returns a new LDA instance with all fields set to None
    fn default() -> Self {
        Self::new()
    }
}

impl LDA {
    /// Creates a new LDA instance
    ///
    /// Returns a new LDA instance with all fields set to None
    pub fn new() -> Self {
        LDA {
            classes: None,
            priors: None,
            means: None,
            cov_inv: None,
            projection: None,
        }
    }

    /// Returns the unique class labels from the training data
    ///
    /// Returns an error if the model has not been fitted
    ///
    /// # Returns
    /// - `Ok(&Array1<i32>)` - Array of class labels
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn get_classes(&self) -> Result<&Array1<i32>, ModelError> {
        self.classes.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Returns the prior probabilities for each class
    ///
    /// Returns an error if the model has not been fitted
    ///
    /// # Returns
    /// - `Ok(&Array1<f64>)` - Array of prior probabilities
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn get_priors(&self) -> Result<&Array1<f64>, ModelError> {
        self.priors.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Returns the mean vectors for each class
    ///
    /// Returns an error if the model has not been fitted
    ///
    /// # Returns
    /// - `Ok(&Array2<f64>)` - Matrix of mean vectors
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn get_means(&self) -> Result<&Array2<f64>, ModelError> {
        self.means.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Returns the inverse of the common covariance matrix
    ///
    /// Returns an error if the model has not been fitted
    ///
    /// # Returns
    /// - `Ok(&Array2<f64>)` - Inverse covariance matrix
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn get_cov_inv(&self) -> Result<&Array2<f64>, ModelError> {
        self.cov_inv.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Returns the projection matrix for dimensionality reduction
    ///
    /// Returns an error if the model has not been fitted
    ///
    /// # Returns
    /// - `Ok(&Array2<f64>)` - Projection matrix
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn get_projection(&self) -> Result<&Array2<f64>, ModelError> {
        self.projection.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Fits the LDA model using training data
    ///
    /// This method calculates both classification parameters and the projection matrix for
    /// dimensionality reduction.
    ///
    /// # Parameters
    /// * `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    /// * `y` - Class labels corresponding to each sample, shape: (n_samples,)
    ///
    /// # Returns
    /// - `Ok(&mut Self)` - Reference to self
    /// - `Err(Box<dyn std::error::Error>>)` - If something goes wrong
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<&mut Self, Box<dyn std::error::Error>> {
        // Input validation
        if x.nrows() != y.len() {
            return Err(Box::new(ModelError::InputValidationError(
                format!("x.nrows() {} != y.len() {}", x.nrows(), y.len())
            )));
        }
        if x.is_empty() || y.len() == 0 {
            return Err(Box::new(ModelError::InputValidationError(
                "Input array is empty".to_string()
            )));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Extract unique class labels from y
        let mut classes_vec: Vec<i32> = y.iter().copied().collect();
        classes_vec.sort();
        classes_vec.dedup();
        if classes_vec.len() < 2 {
            return Err(Box::new(ModelError::InputValidationError(
                "At least two distinct classes are required".to_string()
            )));
        }
        let classes_arr = Array1::from_vec(classes_vec);
        self.classes = Some(classes_arr);

        let n_classes = self.classes.as_ref().unwrap().len();
        let classes = self.classes.as_ref().unwrap();

        // Calculate prior probabilities and means for each class in parallel
        let class_stats: Vec<(usize, f64, Array1<f64>, Vec<usize>)> = classes.iter().enumerate()
            .map(|(i, &class)| {
                let indices: Vec<usize> = y.indexed_iter()
                    .filter(|&(_, &val)| val == class)
                    .map(|(idx, _)| idx)
                    .collect();
                let n_class = indices.len();
                let prior = n_class as f64 / n_samples as f64;
                let class_data = x.select(Axis(0), &indices);
                let mean_row = class_data.mean_axis(Axis(0))
                    .expect("Error computing class mean");
                (i, prior, mean_row, indices)
            })
            .collect();

        // Extract prior probabilities and means
        let mut priors_vec = Vec::with_capacity(n_classes);
        let mut means_mat = Array2::<f64>::zeros((n_classes, n_features));
        let mut class_indices = Vec::with_capacity(n_classes);
        for (i, prior, mean_row, indices) in class_stats {
            priors_vec.push(prior);
            means_mat.row_mut(i).assign(&mean_row);
            class_indices.push((i, indices));
        }

        self.priors = Some(Array1::from_vec(priors_vec));
        self.means = Some(means_mat);

        // Calculate within-class scatter matrix (Sw) in parallel
        let sw_parts: Vec<Array2<f64>> = class_indices.par_iter()
            .map(|(i, indices)| {
                let class_data = x.select(Axis(0), indices);
                let class_mean = &self.means.as_ref().unwrap().row(*i);
                class_data.outer_iter()
                    .fold(Array2::<f64>::zeros((n_features, n_features)), |acc, row| {
                        let diff = &row - class_mean;
                        let diff_col = diff.insert_axis(Axis(1));
                        acc + diff_col.dot(&diff_col.t())
                    })
            })
            .collect();

        let sw = sw_parts.into_iter()
            .fold(Array2::<f64>::zeros((n_features, n_features)), |acc, matrix| acc + matrix);

        // Estimate covariance matrix: cov = Sw / (n_samples - n_classes)
        let cov = sw / ((n_samples - n_classes) as f64);

        // Convert cov from ndarray to nalgebra DMatrix and calculate its inverse
        let cov_slice = cov.as_slice().ok_or("Failed to convert cov to slice")?;
        let cov_mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, cov_slice);
        let cov_inv_mat = cov_mat.try_inverse().ok_or("Failed to invert covariance matrix")?;
        let cov_inv_arr = Array2::from_shape_vec((n_features, n_features), cov_inv_mat.as_slice().to_vec())?;
        self.cov_inv = Some(cov_inv_arr);

        // Calculate overall mean of x
        let overall_mean = x.mean_axis(Axis(0))
            .ok_or(ModelError::ProcessingError("Error computing overall mean".to_string()))?;

        // Calculate between-class scatter matrix (Sb) in parallel
        let sb_parts: Vec<Array2<f64>> = classes.iter().enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&(i, &class)| {
                let count = y.iter().filter(|&&val| val == class).count() as f64;
                let diff = &self.means.as_ref().unwrap().row(i) - &overall_mean;
                let diff_col = diff.insert_axis(Axis(1));
                count * diff_col.dot(&diff_col.t())
            })
            .collect();
        let sb = sb_parts.into_iter()
            .fold(Array2::<f64>::zeros((n_features, n_features)), |acc, matrix| acc + matrix);

        // Solve generalized eigenvalue problem: calculate a_mat = cov_inv * Sb
        let cov_inv = self.cov_inv.as_ref().unwrap();
        let a_mat = cov_inv.dot(&sb);
        // Symmetrize a_mat to ensure symmetric eigendecomposition: a_sym = (a_mat + a_mat^T)/2
        let a_sym = (a_mat.clone() + a_mat.t()) * 0.5;
        let a_slice = a_sym.as_slice().ok_or("Failed to convert a_sym to slice")?;
        let a_dmat = nalgebra::DMatrix::from_row_slice(n_features, n_features, a_slice);
        let sym_eigen = nalgebra::SymmetricEigen::new(a_dmat);

        // Convert eigenvalues and eigenvectors from nalgebra to ndarray format
        let eigenvalues = Array1::from_vec(sym_eigen.eigenvalues.as_slice().to_vec());
        let eigenvectors = Array2::from_shape_vec((n_features, n_features), sym_eigen.eigenvectors.as_slice().to_vec())?;

        // Sort indices by eigenvalues in descending order
        let mut eig_pairs: Vec<(usize, f64)> = eigenvalues.iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        eig_pairs.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // LDA's maximum dimensionality is n_classes - 1
        let max_components = if n_classes > 1 { n_classes - 1 } else { 1 };
        let mut w = Array2::<f64>::zeros((n_features, max_components));
        for (j, &(i, _)) in eig_pairs.iter().take(max_components).enumerate() {
            // Use each column i of eigenvectors as a projection vector
            let vec = eigenvectors.slice(s![.., i]).to_owned();
            w.column_mut(j).assign(&vec);
        }
        self.projection = Some(w);

        println!("LDA model training finished");
        Ok(self)
    }

    /// Predicts class labels for new samples using the trained model
    ///
    /// # Parameters
    /// * `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    ///
    /// # Returns
    /// - `Ok(Array1<i32>)` - Array of predicted class labels
    /// - `Err(ModelError::InputValidationError)` - If input does not match the expectation
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<i32>, ModelError> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError("Input array is empty".to_string()));
        }
        if self.classes.is_none() || self.means.is_none() || self.cov_inv.is_none() {
            return Err(ModelError::NotFitted);
        }

        let classes = self.classes.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let cov_inv = self.cov_inv.as_ref().unwrap();
        let priors = self.priors.as_ref().unwrap();
        let n_classes = classes.len();

        // Use Rayon's parallel iteration
        let predictions: Vec<i32> = x.outer_iter()
            .into_par_iter()  // Convert to parallel iterator
            .map(|row| {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_class = classes[0];
                for j in 0..n_classes {
                    let score = self.discriminant_score(&row.to_owned(), &means.row(j).to_owned(), priors[j], cov_inv);
                    if score > best_score {
                        best_score = score;
                        best_class = classes[j];
                    }
                }
                best_class
            })
            .collect();

        // Convert results back to ndarray's Array1
        Ok(Array1::from(predictions))
    }

    /// Transforms data using the trained projection matrix for dimensionality reduction
    ///
    /// # Parameters
    /// * `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    /// * `n_components` - Number of dimensions after reduction (must be in \[1, n_classes - 1\])
    ///
    /// # Returns
    /// - `Ok(Array2<f64>)` - Transformed data matrix
    /// - `Err(ModelError::InputValidationError)` - If input does not match expectation
    pub fn transform(&self, x: ArrayView2<f64>, n_components: usize) -> Result<Array2<f64>, ModelError> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError("Input array is empty".to_string()));
        }
        let proj = self.projection.as_ref().ok_or(ModelError::NotFitted)?;
        let total_components = proj.ncols();
        if n_components == 0 || n_components > total_components {
            return Err(ModelError::InputValidationError(
                format!("n_components should be in range [1, {}], got {}", total_components, n_components)
            ));
        }
        let w_reduced = proj.slice(s![.., 0..n_components]).to_owned();
        Ok(x.dot(&w_reduced))
    }

    /// Fits the model and transforms the data in one step
    ///
    /// # Parameters
    /// * `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    /// * `y` - Class labels corresponding to each sample, shape: (n_samples,)
    /// * `n_components` - Number of dimensions after reduction
    ///
    /// # Returns
    /// - `Ok(Array2<f64>)` - Transformed data matrix
    /// - `Err(Box<dyn std::error::Error>>)` - If something goes wrong
    pub fn fit_transform(&mut self, x: ArrayView2<f64>, y: ArrayView1<i32>, n_components: usize) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        self.fit(x, y)?;
        Ok(self.transform(x, n_components)?)
    }

    /// Calculates the discriminant score for classification
    ///
    /// # Parameters
    /// * `x` - Feature vector of a sample, shape: (n_features,)
    /// * `mean` - Mean vector of a class, shape: (n_features,)
    /// * `prior` - Prior probability of a class
    /// * `cov_inv` - Inverse of the common covariance matrix
    ///
    /// # Returns
    /// * `f64` - Discriminant score
    fn discriminant_score(
        &self,
        x: &Array1<f64>,
        mean: &Array1<f64>,
        prior: f64,
        cov_inv: &Array2<f64>,
    ) -> f64 {
        let term1 = x.dot(&cov_inv.dot(mean));
        let term2 = mean.dot(&cov_inv.dot(mean));

        term1 - 0.5 * term2 + prior.ln()
    }
}