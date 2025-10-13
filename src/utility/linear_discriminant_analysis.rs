use super::*;

/// Threshold for using parallel computation in LDA.
/// When the number of samples is below this threshold, sequential computation is used.
const LDA_PARALLEL_THRESHOLD: usize = 100;

/// Linear Discriminant Analysis (LDA)
///
/// A classifier and dimensionality reduction technique that projects data onto a lower-dimensional space while maintaining class separability.
///
/// # Fields
///
/// - `classes` - Array of unique class labels from training data
/// - `priors` - Prior probabilities for each class
/// - `means` - Mean vectors for each class
/// - `cov_inv` - Inverse of the common covariance matrix
/// - `projection` - Projection matrix for dimensionality reduction
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::linear_discriminant_analysis::LDA;
///
/// // Create feature matrix and class labels
/// let x = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 5.0, 5.0, 5.5, 4.5, 6.0, 5.0]).unwrap();
/// let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// // Create and fit LDA model
/// let mut lda = LDA::new().unwrap();
/// lda.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let x_new = Array2::from_shape_vec((2, 2), vec![1.2, 2.2, 5.2, 4.8]).unwrap();
/// let predictions = lda.predict(x_new.view()).unwrap();
///
/// // Transform data to lower dimension
/// let x_transformed = lda.transform(x.view(), 1).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDA {
    classes: Option<Array1<i32>>,
    priors: Option<Array1<f64>>,
    means: Option<Array2<f64>>,
    cov_inv: Option<Array2<f64>>,
    projection: Option<Array2<f64>>,
}

/// Default implementation for LDA
impl Default for LDA {
    fn default() -> Self {
        Self::new().expect("LDA::new() should never fail")
    }
}

impl LDA {
    /// Creates a new LDA instance
    ///
    /// # Returns
    ///
    /// * `Ok(LDA)` - A new LDA instance with all fields set to None
    pub fn new() -> Result<Self, ModelError> {
        Ok(LDA {
            classes: None,
            priors: None,
            means: None,
            cov_inv: None,
            projection: None,
        })
    }

    get_field_as_ref!(get_classes, classes, Option<&Array1<i32>>);
    get_field_as_ref!(get_priors, priors, Option<&Array1<f64>>);
    get_field_as_ref!(get_means, means, Option<&Array2<f64>>);
    get_field_as_ref!(get_cov_inv, cov_inv, Option<&Array2<f64>>);
    get_field_as_ref!(get_projection, projection, Option<&Array2<f64>>);

    /// Fits the LDA model using training data
    ///
    /// This method calculates both classification parameters and the projection matrix for
    /// dimensionality reduction.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    /// - `y` - Class labels corresponding to each sample, shape: (n_samples,)
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - Reference to self
    /// - `Err(ModelError::InputValidationError)` - If input validation fails
    /// - `Err(ModelError::ProcessingError)` - If matrix operations fail
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<&mut Self, ModelError> {
        // Input validation
        if x.nrows() != y.len() {
            return Err(ModelError::InputValidationError(format!(
                "x.nrows() {} != y.len() {}",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() || y.len() == 0 {
            return Err(ModelError::InputValidationError(
                "Input array is empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Create progress bar for training stages
        let progress_bar = ProgressBar::new(5);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Stage: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message("Validating input and extracting classes");

        // Check minimum requirements for LDA
        if n_features == 0 {
            return Err(ModelError::InputValidationError(
                "Number of features must be greater than 0".to_string(),
            ));
        }

        // Extract unique class labels from y - use HashSet for better performance
        let mut classes_set = AHashSet::new();
        for &label in y.iter() {
            classes_set.insert(label);
        }
        if classes_set.len() < 2 {
            return Err(ModelError::InputValidationError(
                "At least two distinct classes are required".to_string(),
            ));
        }
        let mut classes_vec: Vec<i32> = classes_set.into_iter().collect();
        classes_vec.sort_unstable(); // sort_unstable is faster for i32

        // Check if we have enough samples for reliable covariance estimation
        if n_samples <= n_features {
            return Err(ModelError::InputValidationError(format!(
                "Number of samples ({}) must be greater than number of features ({}) for stable covariance estimation",
                n_samples, n_features
            )));
        }

        let classes_arr = Array1::from_vec(classes_vec);
        self.classes = Some(classes_arr);
        let n_classes = self.classes.as_ref().unwrap().len();
        let classes = self.classes.as_ref().unwrap();

        progress_bar.inc(1);
        progress_bar.set_message("Computing class statistics and scatter matrices");

        // Pre-allocate class indices map for better performance
        let mut class_indices_map = std::collections::HashMap::new();
        for &class in classes.iter() {
            class_indices_map.insert(class, Vec::new());
        }

        // Single pass to collect indices for all classes
        for (idx, &class) in y.iter().enumerate() {
            if let Some(indices) = class_indices_map.get_mut(&class) {
                indices.push(idx);
            }
        }

        // Ensure each class has enough samples
        for (&class, indices) in &class_indices_map {
            if indices.len() < 2 {
                return Err(ModelError::InputValidationError(format!(
                    "Class {} has only {} sample(s). Each class must have at least 2 samples",
                    class,
                    indices.len()
                )));
            }
        }

        // Pre-allocate arrays for better memory efficiency
        let mut priors_vec = Vec::with_capacity(n_classes);
        let mut means_mat = Array2::<f64>::zeros((n_classes, n_features));
        let mut sw = Array2::<f64>::zeros((n_features, n_features));

        // Calculate overall mean first - will be used for between-class scatter
        let overall_mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            ModelError::ProcessingError("Error computing overall mean".to_string())
        })?;
        let mut sb = Array2::<f64>::zeros((n_features, n_features));

        // Process each class - use parallel computation for large datasets
        let process_class = |&(class_idx, &class): &(usize, &i32)| {
            let indices = &class_indices_map[&class];
            let n_class = indices.len();
            let prior = n_class as f64 / n_samples as f64;

            let class_data = x.select(Axis(0), indices);
            let class_mean = class_data
                .mean_axis(Axis(0))
                .expect("Error computing class mean");

            // Calculate within-class scatter for this class
            let mut class_sw = Array2::<f64>::zeros((n_features, n_features));
            for row in class_data.outer_iter() {
                let diff = &row - &class_mean;
                let diff_col = diff.insert_axis(Axis(1));
                class_sw += &diff_col.dot(&diff_col.t());
            }

            // Calculate between-class scatter contribution for this class
            let mean_diff = &class_mean - &overall_mean;
            let mean_diff_col = mean_diff.insert_axis(Axis(1));
            let class_sb = mean_diff_col.dot(&mean_diff_col.t()) * (n_class as f64);

            (class_idx, prior, class_mean, class_sw, class_sb)
        };

        let class_pairs: Vec<_> = classes.iter().enumerate().collect();
        let class_results: Vec<_> = if n_samples >= LDA_PARALLEL_THRESHOLD {
            class_pairs.par_iter().map(process_class).collect()
        } else {
            class_pairs.iter().map(process_class).collect()
        };

        // Aggregate results from parallel computation
        for (class_idx, prior, class_mean, class_sw, class_sb) in class_results {
            priors_vec.push(prior);
            means_mat.row_mut(class_idx).assign(&class_mean);
            sw += &class_sw;
            sb += &class_sb;
        }

        self.priors = Some(Array1::from_vec(priors_vec));
        self.means = Some(means_mat);

        progress_bar.inc(1);
        progress_bar.set_message("Inverting covariance matrix");

        // Estimate covariance matrix with better regularization strategy
        let cov = sw / ((n_samples - n_classes) as f64);

        // Adaptive regularization based on condition number estimation
        let trace = cov.diag().sum();
        let regularization = (trace / n_features as f64) * 1e-6; // Scale with data magnitude
        let regularized_cov = cov + Array2::<f64>::eye(n_features) * regularization;

        // Use more numerically stable matrix inversion
        let cov_slice = regularized_cov.as_slice().ok_or_else(|| {
            ModelError::ProcessingError("Failed to convert covariance matrix to slice".to_string())
        })?;
        let cov_mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, cov_slice);

        // Use SVD for more stable inversion
        let svd = nalgebra::linalg::SVD::new(cov_mat.clone(), true, true);
        let max_singular_value = svd.singular_values.max();
        let tolerance = 1e-12 * max_singular_value; // Scale tolerance by largest singular value

        let cov_inv_mat = svd.pseudo_inverse(tolerance).or_else(|_| {
            Err(ModelError::ProcessingError(
                "Covariance matrix is singular and cannot be inverted. Try using more samples or reducing dimensionality".to_string()
            ))
        })?;

        let cov_inv_arr =
            Array2::from_shape_vec((n_features, n_features), cov_inv_mat.as_slice().to_vec())
                .map_err(|e| {
                    ModelError::ProcessingError(format!(
                        "Failed to create inverse covariance array: {}",
                        e
                    ))
                })?;
        self.cov_inv = Some(cov_inv_arr);

        progress_bar.inc(1);
        progress_bar.set_message("Computing projection matrix via eigendecomposition");

        // Solve generalized eigenvalue problem using more stable approach
        let cov_inv = self.cov_inv.as_ref().unwrap();
        let a_mat = cov_inv.dot(&sb);

        // Use SVD for eigendecomposition for better numerical stability
        let a_slice = a_mat.as_slice().ok_or_else(|| {
            ModelError::ProcessingError(
                "Failed to convert matrix for eigendecomposition".to_string(),
            )
        })?;
        let a_dmat = nalgebra::DMatrix::from_row_slice(n_features, n_features, a_slice);

        // Use SVD instead of symmetric eigendecomposition for better stability
        let svd = nalgebra::linalg::SVD::new(a_dmat, true, true);

        let (eigenvalues, eigenvectors) = if let (Some(u), Some(_)) = (svd.u, svd.v_t) {
            let singular_values = svd.singular_values;
            (
                Array1::from_vec(singular_values.as_slice().to_vec()),
                Array2::from_shape_vec((n_features, n_features), u.as_slice().to_vec()).map_err(
                    |e| {
                        ModelError::ProcessingError(format!(
                            "Failed to create eigenvectors array: {}",
                            e
                        ))
                    },
                )?,
            )
        } else {
            return Err(ModelError::ProcessingError(
                "SVD decomposition failed".to_string(),
            ));
        };

        // Sort indices by eigenvalues in descending order - use unstable sort for better performance
        let mut eig_pairs: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        eig_pairs
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // LDA's maximum dimensionality is min(n_classes - 1, n_features)
        let max_components = (n_classes - 1).min(n_features);
        let mut w = Array2::<f64>::zeros((n_features, max_components));
        let mut component_idx = 0;
        for &(i, eigenval) in eig_pairs.iter().take(max_components) {
            // Filter out very small eigenvalues for numerical stability
            if eigenval.abs() > 1e-10 {
                let vec = eigenvectors.slice(s![.., i]);
                // Normalize eigenvector for better numerical properties
                let norm = vec.dot(&vec).sqrt();
                if norm > 1e-12 {
                    w.column_mut(component_idx).assign(&(&vec / norm));
                    component_idx += 1;
                }
            }
        }

        self.projection = Some(w);

        progress_bar.inc(1);
        progress_bar.finish_with_message("Completed");

        println!(
            "\nLDA training completed: {} samples, {} features, {} classes, max components: {}",
            n_samples, n_features, n_classes, component_idx
        );

        Ok(self)
    }

    /// Predicts class labels for new samples using the trained model
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<i32>)` - Array of predicted class labels
    /// - `Err(ModelError::InputValidationError)` - If input does not match the expectation
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<i32>, ModelError> {
        // Check if model has been fitted
        if self.classes.is_none() || self.means.is_none() || self.cov_inv.is_none() {
            return Err(ModelError::NotFitted);
        }

        // Check for empty input data
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot predict on empty dataset".to_string(),
            ));
        }

        let classes = self.classes.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let cov_inv = self.cov_inv.as_ref().unwrap();
        let priors = self.priors.as_ref().unwrap();
        let n_classes = classes.len();

        // Check feature dimension match
        let n_features = means.ncols();
        if x.ncols() != n_features {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                n_features
            )));
        }

        // Check for invalid values in input data
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        // Predict class for each sample
        let predict_sample = |row: ArrayView1<f64>| {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_class = classes[0];
            for j in 0..n_classes {
                let diff = &row - &means.row(j);
                let mahalanobis_term = diff.dot(&cov_inv.dot(&diff));
                let prior_term = if priors[j] > 0.0 {
                    priors[j].ln()
                } else {
                    f64::NEG_INFINITY
                };
                let score = -0.5 * mahalanobis_term + prior_term;

                if score > best_score {
                    best_score = score;
                    best_class = classes[j];
                }
            }
            best_class
        };

        let predictions: Vec<i32> = if x.nrows() >= LDA_PARALLEL_THRESHOLD {
            x.outer_iter().into_par_iter().map(predict_sample).collect()
        } else {
            x.outer_iter().map(predict_sample).collect()
        };

        Ok(Array1::from(predictions))
    }

    /// Transforms data using the trained projection matrix for dimensionality reduction
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    /// - `n_components` - Number of dimensions after reduction (must be in \[1, n_classes - 1\])
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - Transformed data matrix
    /// - `Err(ModelError::InputValidationError)` - If input does not match expectation
    /// - `Err(ModelError::NotFitted)` - If not fitted
    pub fn transform(
        &self,
        x: ArrayView2<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>, ModelError> {
        // Check if model has been fitted
        let proj = self.projection.as_ref().ok_or(ModelError::NotFitted)?;

        // Check for empty input data
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot transform empty dataset".to_string(),
            ));
        }

        // Check feature dimension match
        let n_features = proj.nrows();
        if x.ncols() != n_features {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                n_features
            )));
        }

        // Check for invalid values in input data
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        // Check n_components range
        let total_components = proj.ncols();
        if n_components == 0 || n_components > total_components {
            return Err(ModelError::InputValidationError(format!(
                "n_components should be in range [1, {}], got {}",
                total_components, n_components
            )));
        }

        let w_reduced = proj.slice(s![.., 0..n_components]).to_owned();
        Ok(x.dot(&w_reduced))
    }

    /// Fits the model and transforms the data in one step
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample, shape: (n_samples, n_features)
    /// - `y` - Class labels corresponding to each sample, shape: (n_samples,)
    /// - `n_components` - Number of dimensions after reduction
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - Transformed data matrix
    /// - `Err(Box<dyn std::error::Error>>)` - If something goes wrong
    pub fn fit_transform(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        n_components: usize,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        self.fit(x, y)?;
        Ok(self.transform(x, n_components)?)
    }

    model_save_and_load_methods!(LDA);
}
