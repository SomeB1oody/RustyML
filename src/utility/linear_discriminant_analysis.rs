use crate::error::ModelError;
use crate::{Deserialize, Serialize};
use ahash::{AHashMap, AHashSet};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2, s};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Solver options for Linear Discriminant Analysis.
///
/// Selects the numerical method used to compute the inverse covariance or
/// discriminant matrix during fitting.
///
/// # Variants
///
/// - `SVD` - Uses singular value decomposition for stable pseudo-inverse computation
/// - `Eigen` - Uses symmetric eigen decomposition for covariance inversion
/// - `LSQR` - Solves linear systems with SVD-based least-squares
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Solver {
    SVD,
    Eigen,
    LSQR,
}

/// Shrinkage strategy for covariance estimation.
///
/// Controls how the covariance matrix is regularized to improve numerical stability.
///
/// # Variants
///
/// - `Auto` - Uses an automatic shrinkage factor based on sample and feature counts
/// - `Manual` - Uses an explicit shrinkage factor in the range [0, 1]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Shrinkage {
    Auto,
    Manual(f64),
}

/// Threshold for switching to parallel computation in LDA.
/// Uses sequential computation at or below this sample count.
const LDA_PRARALLEL_THRESHOLD: usize = 500;

/// Linear Discriminant Analysis (LDA) model.
///
/// Provides supervised dimensionality reduction and classification by projecting
/// samples onto a lower-dimensional space that maximizes class separability.
///
/// # Fields
///
/// - `n_components` - Number of components to keep after dimensionality reduction
/// - `solver` - Solver strategy for LDA computations
/// - `shrinkage` - Optional shrinkage strategy for covariance estimation
/// - `classes` - Array of unique class labels from training data
/// - `priors` - Prior probabilities for each class
/// - `means` - Mean vectors for each class
/// - `cov_inv` - Inverse of the common covariance matrix
/// - `projection` - Projection matrix for dimensionality reduction
///
/// # Examples
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::linear_discriminant_analysis::{LDA, Shrinkage, Solver};
///
/// let x = Array2::from_shape_vec(
///     (6, 2),
///     vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 5.0, 5.0, 5.5, 4.5, 6.0, 5.0],
/// ).unwrap();
/// let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// let mut lda = LDA::new(1, Some(Solver::SVD), Some(Shrinkage::Manual(0.1))).unwrap();
/// lda.fit(&x, &y).unwrap();
/// let _predictions = lda.predict(&x).unwrap();
/// let _x_transformed = lda.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDA {
    n_components: usize,
    solver: Solver,
    shrinkage: Option<Shrinkage>,
    classes: Option<Array1<i32>>,
    priors: Option<Array1<f64>>,
    means: Option<Array2<f64>>,
    cov_inv: Option<Array2<f64>>,
    projection: Option<Array2<f64>>,
}

/// Default LDA configuration.
///
/// Provides a reasonable starting point for most datasets.
///
/// # Default Values
///
/// - `n_components` - 2
/// - `solver` - `Solver::SVD`
/// - `shrinkage` - `None`
impl Default for LDA {
    fn default() -> Self {
        Self::new(2, None, None).expect("Default LDA parameters should be valid")
    }
}

impl LDA {
    /// Creates a new LDA instance with validated hyperparameters.
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of components to keep (must be > 0)
    /// - `solver` - Optional solver choice (defaults to `Solver::SVD`)
    /// - `shrinkage` - Optional shrinkage strategy for covariance estimation
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new LDA instance or validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `n_components` is zero or shrinkage is out of range
    pub fn new(
        n_components: usize,
        solver: Option<Solver>,
        shrinkage: Option<Shrinkage>,
    ) -> Result<Self, ModelError> {
        if n_components == 0 {
            return Err(ModelError::InputValidationError(
                "n_components must be greater than 0".to_string(),
            ));
        }

        if let Some(Shrinkage::Manual(alpha)) = shrinkage {
            if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                return Err(ModelError::InputValidationError(format!(
                    "shrinkage Manual(alpha) must be in [0, 1], got {}",
                    alpha
                )));
            }
        }

        Ok(Self {
            n_components,
            solver: solver.unwrap_or(Solver::SVD),
            shrinkage,
            classes: None,
            priors: None,
            means: None,
            cov_inv: None,
            projection: None,
        })
    }

    // Getters
    get_field!(get_n_components, n_components, usize);
    get_field!(get_solver, solver, Solver);
    get_field!(get_shrinkage, shrinkage, Option<Shrinkage>);
    get_field_as_ref!(get_classes, classes, Option<&Array1<i32>>);
    get_field_as_ref!(get_priors, priors, Option<&Array1<f64>>);
    get_field_as_ref!(get_means, means, Option<&Array2<f64>>);
    get_field_as_ref!(get_cov_inv, cov_inv, Option<&Array2<f64>>);
    get_field_as_ref!(get_projection, projection, Option<&Array2<f64>>);

    /// Fits the LDA model using training data.
    ///
    /// Estimates class statistics, covariance, and projection matrices from labeled samples.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    /// - `y` - Class labels aligned with the rows of `x`
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - Mutable reference to self for chaining
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If inputs are empty, shapes mismatch, or contain invalid values
    /// - `ModelError::ProcessingError` - If numerical computation fails during fitting
    ///
    /// # Performance
    ///
    /// Runs sequentially when `x.nrows()` is at or below `LDA_PRARALLEL_THRESHOLD`.
    /// Uses Rayon parallelism when `x.nrows()` is above `LDA_PRARALLEL_THRESHOLD`.
    pub fn fit<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<&mut Self, ModelError>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = i32>,
    {
        if x.nrows() != y.len() {
            return Err(ModelError::InputValidationError(format!(
                "x.nrows() {} != y.len() {}",
                x.nrows(),
                y.len()
            )));
        }

        if x.is_empty() || y.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input array is empty".to_string(),
            ));
        }

        if x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Number of features must be greater than 0".to_string(),
            ));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        // Decide execution mode from sample count
        let use_parallel = n_samples > LDA_PRARALLEL_THRESHOLD;

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                5,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input and extracting classes");
            Some(pb)
        };

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
        classes_vec.sort_unstable();
        self.classes = Some(Array1::from_vec(classes_vec));
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        if n_samples <= n_classes {
            return Err(ModelError::InputValidationError(format!(
                "Number of samples ({}) must be greater than number of classes ({})",
                n_samples, n_classes
            )));
        }

        let max_components = (n_classes - 1).min(n_features);
        if self.n_components > max_components {
            return Err(ModelError::InputValidationError(format!(
                "n_components should be <= {}, got {}",
                max_components, self.n_components
            )));
        }

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing class statistics and scatter matrices");
        }

        // Group row indices by class label
        let mut class_indices_map: AHashMap<i32, Vec<usize>> = AHashMap::with_capacity(n_classes);
        for &class in classes.iter() {
            class_indices_map.insert(class, Vec::new());
        }
        for (idx, &class) in y.iter().enumerate() {
            if let Some(indices) = class_indices_map.get_mut(&class) {
                indices.push(idx);
            }
        }
        for (&class, indices) in &class_indices_map {
            if indices.len() < 2 {
                return Err(ModelError::InputValidationError(format!(
                    "Class {} has only {} sample(s). Each class must have at least 2 samples",
                    class,
                    indices.len()
                )));
            }
        }

        // Compute the overall mean for between-class scatter
        let overall_mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            ModelError::ProcessingError("Error computing overall mean".to_string())
        })?;

        let class_pairs: Vec<_> = classes.iter().enumerate().collect();
        let class_results: Vec<_> = if use_parallel {
            // Compute per-class stats in parallel
            let x_owned = x.to_owned();
            class_pairs
                .par_iter()
                .map(|&(class_idx, &class)| {
                    let indices = &class_indices_map[&class];
                    let (prior, class_mean, class_sw, class_sb) =
                        Self::compute_class_stats(&x_owned, indices, &overall_mean, n_samples);
                    (class_idx, prior, class_mean, class_sw, class_sb)
                })
                .collect()
        } else {
            // Compute per-class stats sequentially
            class_pairs
                .iter()
                .map(|&(class_idx, &class)| {
                    let indices = &class_indices_map[&class];
                    let (prior, class_mean, class_sw, class_sb) =
                        Self::compute_class_stats(x, indices, &overall_mean, n_samples);
                    (class_idx, prior, class_mean, class_sw, class_sb)
                })
                .collect()
        };

        let mut priors_vec = Vec::with_capacity(n_classes);
        let mut means_mat = Array2::<f64>::zeros((n_classes, n_features));
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        let mut sb = Array2::<f64>::zeros((n_features, n_features));

        // Aggregate priors, class means, and scatter matrices
        for (class_idx, prior, class_mean, class_sw, class_sb) in class_results {
            priors_vec.push(prior);
            means_mat.row_mut(class_idx).assign(&class_mean);
            sw += &class_sw;
            sb += &class_sb;
        }

        self.priors = Some(Array1::from_vec(priors_vec));
        self.means = Some(means_mat);

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Applying shrinkage and inverting covariance matrix");
        }

        // Estimate and stabilize the shared covariance
        let mut cov = sw / ((n_samples - n_classes) as f64);
        cov = self.apply_shrinkage(&cov, n_samples, n_features);
        self.regularize_covariance(&mut cov);

        // Compute inverse covariance for linear scoring
        let cov_inv = self.compute_cov_inv(&cov)?;
        self.cov_inv = Some(cov_inv);

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing projection matrix");
        }

        // Build the discriminant projection
        let solver_matrix = self.compute_solver_matrix(&cov, sb, self.cov_inv.as_ref().unwrap())?;
        let projection = self.compute_projection(&solver_matrix, self.n_components)?;
        self.projection = Some(projection);

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Finalizing model state");
        }

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.finish_with_message("Completed");
        }

        Ok(self)
    }

    /// Predicts class labels for new samples using the trained model.
    ///
    /// Applies the learned class means and shared covariance to compute linear scores.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, ModelError>` - Predicted class labels
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted
    /// - `ModelError::InputValidationError` - If inputs are empty, mismatched, or contain invalid values
    ///
    /// # Performance
    ///
    /// Uses parallel prediction when `x.nrows()` is above `LDA_PRARALLEL_THRESHOLD` (500).
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        let classes = self.classes.as_ref().ok_or(ModelError::NotFitted)?;
        let means = self.means.as_ref().ok_or(ModelError::NotFitted)?;
        let cov_inv = self.cov_inv.as_ref().ok_or(ModelError::NotFitted)?;
        let priors = self.priors.as_ref().ok_or(ModelError::NotFitted)?;

        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot predict on empty dataset".to_string(),
            ));
        }

        let n_features = means.ncols();
        if x.ncols() != n_features {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                n_features
            )));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                x.nrows() as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Scoring samples");
            pb
        };

        let n_classes = classes.len();
        let mut coefficients = Array2::<f64>::zeros((n_classes, n_features));
        let mut intercepts = Array1::<f64>::zeros(n_classes);

        for j in 0..n_classes {
            // Build linear discriminant coefficients per class
            let mean = means.row(j).to_owned();
            let coef = cov_inv.dot(&mean);
            coefficients.row_mut(j).assign(&coef);
            let prior_term = if priors[j] > 0.0 {
                priors[j].ln()
            } else {
                f64::NEG_INFINITY
            };
            intercepts[j] = -0.5 * mean.dot(&coef) + prior_term;
        }

        let predict_sample = |row: ArrayView1<f64>| {
            // Score each class and keep the best label
            let mut best_score = f64::NEG_INFINITY;
            let mut best_class = classes[0];
            for j in 0..n_classes {
                let score = row.dot(&coefficients.row(j)) + intercepts[j];
                if score > best_score {
                    best_score = score;
                    best_class = classes[j];
                }
            }
            best_class
        };

        let predictions: Vec<i32> = if x.nrows() > LDA_PRARALLEL_THRESHOLD {
            // Use parallel scoring for large batches
            let x_owned = x.to_owned();
            #[cfg(feature = "show_progress")]
            let pb = progress_bar.clone();
            x_owned
                .outer_iter()
                .into_par_iter()
                .map(|row| {
                    let pred = predict_sample(row);
                    #[cfg(feature = "show_progress")]
                    pb.inc(1);
                    pred
                })
                .collect()
        } else {
            // Use sequential scoring for smaller batches
            x.outer_iter()
                .map(|row| {
                    let pred = predict_sample(row);
                    #[cfg(feature = "show_progress")]
                    progress_bar.inc(1);
                    pred
                })
                .collect()
        };

        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Completed");
        Ok(Array1::from(predictions))
    }

    /// Transforms data using the trained projection matrix.
    ///
    /// Projects samples onto the learned discriminant components.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted
    /// - `ModelError::InputValidationError` - If inputs are empty, mismatched, or contain invalid values
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.transform_internal(x)
    }

    /// Fits the model and transforms the data in one step.
    ///
    /// Convenience method that trains the model and returns the projected data.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    /// - `y` - Class labels aligned with the rows of `x`
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If inputs are empty, shapes mismatch, or contain invalid values
    /// - `ModelError::ProcessingError` - If numerical computation fails during fitting
    ///
    /// # Performance
    ///
    /// Runs sequentially when `x.nrows()` is at or below `LDA_PRARALLEL_THRESHOLD`.
    /// Uses Rayon parallelism when `x.nrows()` is above `LDA_PRARALLEL_THRESHOLD`.
    pub fn fit_transform<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<Array2<f64>, ModelError>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = i32>,
    {
        // Fit the model with adaptive parallelism
        self.fit(x, y)?;
        // Project the input using the fitted components
        self.transform_internal(x)
    }

    /// Transforms input data using the fitted projection
    fn transform_internal<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        let projection = self.projection.as_ref().ok_or(ModelError::NotFitted)?;

        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot transform empty dataset".to_string(),
            ));
        }

        if x.ncols() != projection.nrows() {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                projection.nrows()
            )));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                2,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input");
            pb
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Applying projection");
        }

        let transformed = x.dot(projection);

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(transformed)
    }

    /// Computes per-class statistics and scatter matrices
    fn compute_class_stats<S>(
        x: &ArrayBase<S, Ix2>,
        indices: &[usize],
        overall_mean: &Array1<f64>,
        n_samples: usize,
    ) -> (f64, Array1<f64>, Array2<f64>, Array2<f64>)
    where
        S: Data<Elem = f64>,
    {
        let n_features = x.ncols();
        let n_class = indices.len();
        let prior = n_class as f64 / n_samples as f64;

        let class_data = x.select(Axis(0), indices);
        let class_mean = class_data
            .mean_axis(Axis(0))
            .expect("Error computing class mean");

        let mut class_sw = Array2::<f64>::zeros((n_features, n_features));
        for row in class_data.outer_iter() {
            // Accumulate within-class scatter
            let diff = &row - &class_mean;
            let diff_col = diff.insert_axis(Axis(1));
            class_sw += &diff_col.dot(&diff_col.t());
        }

        // Between-class scatter from mean shift
        let mean_diff = &class_mean - overall_mean;
        let mean_diff_col = mean_diff.insert_axis(Axis(1));
        let class_sb = mean_diff_col.dot(&mean_diff_col.t()) * (n_class as f64);

        (prior, class_mean, class_sw, class_sb)
    }

    /// Applies the configured shrinkage to the covariance matrix
    fn apply_shrinkage(
        &self,
        cov: &Array2<f64>,
        n_samples: usize,
        n_features: usize,
    ) -> Array2<f64> {
        let alpha = match self.shrinkage {
            None => return cov.clone(),
            Some(Shrinkage::Manual(alpha)) => alpha,
            Some(Shrinkage::Auto) => {
                // Simple Ledoit-Wolf style shrinkage heuristic
                let denom = (n_samples + n_features) as f64;
                if denom > 0.0 {
                    (n_features as f64 / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
        };

        if alpha <= 0.0 {
            return cov.clone();
        }

        // Shrink toward a scaled identity matrix
        let mut shrunk = cov.mapv(|v| v * (1.0 - alpha));
        let mu = cov.diag().sum() / n_features as f64;
        shrunk += &(Array2::<f64>::eye(n_features) * (alpha * mu));
        shrunk
    }

    /// Adds a small diagonal regularization term to covariance
    fn regularize_covariance(&self, cov: &mut Array2<f64>) {
        let n_features = cov.ncols().max(1);
        let trace = cov.diag().sum();
        let avg_var = if trace.is_finite() && trace > 0.0 {
            trace / n_features as f64
        } else {
            1.0
        };
        let regularization = avg_var * 1e-6;
        *cov += &(Array2::<f64>::eye(n_features) * regularization);
    }

    /// Computes the inverse covariance matrix using the chosen solver
    fn compute_cov_inv(&self, cov: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        let n_features = cov.ncols();
        let cov_slice = cov.as_slice().ok_or_else(|| {
            ModelError::ProcessingError("Failed to convert covariance matrix to slice".to_string())
        })?;
        let cov_mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, cov_slice);

        let cov_inv_mat = match self.solver {
            Solver::Eigen => {
                // Invert via eigen decomposition with tolerance
                let eig = nalgebra::linalg::SymmetricEigen::new(cov_mat);
                let mut inv_vals = eig.eigenvalues.clone();
                let max_eval = inv_vals.iter().cloned().fold(0.0_f64, f64::max);
                let tol = (1e-12 * max_eval).max(1e-12);
                for i in 0..inv_vals.len() {
                    let val = inv_vals[i];
                    inv_vals[i] = if val.abs() > tol { 1.0 / val } else { 0.0 };
                }
                let inv_diag = nalgebra::DMatrix::from_diagonal(&inv_vals);
                &eig.eigenvectors * inv_diag * eig.eigenvectors.transpose()
            }
            Solver::LSQR => {
                // Solve linear system for inverse using SVD
                let svd = nalgebra::linalg::SVD::new(cov_mat.clone(), true, true);
                let max_sv = svd.singular_values.max();
                let tol = (1e-12 * max_sv).max(1e-12);
                let identity = nalgebra::DMatrix::<f64>::identity(n_features, n_features);
                svd.solve(&identity, tol).map_err(|_| {
                    ModelError::ProcessingError(
                        "LSQR solver failed to compute covariance inverse".to_string(),
                    )
                })?
            }
            Solver::SVD => {
                // Use pseudo-inverse for numerical stability
                let svd = nalgebra::linalg::SVD::new(cov_mat, true, true);
                let max_sv = svd.singular_values.max();
                let tol = (1e-12 * max_sv).max(1e-12);
                svd.pseudo_inverse(tol).map_err(|_| {
                    ModelError::ProcessingError(
                        "Covariance matrix is singular and cannot be inverted".to_string(),
                    )
                })?
            }
        };

        Array2::from_shape_vec((n_features, n_features), cov_inv_mat.as_slice().to_vec()).map_err(
            |e| ModelError::ProcessingError(format!("Failed to build inverse covariance: {}", e)),
        )
    }

    /// Builds the solver matrix used to derive the projection
    fn compute_solver_matrix(
        &self,
        cov: &Array2<f64>,
        sb: Array2<f64>,
        cov_inv: &Array2<f64>,
    ) -> Result<Array2<f64>, ModelError> {
        match self.solver {
            Solver::LSQR => {
                // Solve cov * M = sb for the discriminant matrix
                let n_features = cov.ncols();
                let cov_slice = cov.as_slice().ok_or_else(|| {
                    ModelError::ProcessingError(
                        "Failed to convert covariance matrix to slice".to_string(),
                    )
                })?;
                let sb_slice = sb.as_slice().ok_or_else(|| {
                    ModelError::ProcessingError(
                        "Failed to convert between-class matrix to slice".to_string(),
                    )
                })?;

                let cov_mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, cov_slice);
                let sb_mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, sb_slice);
                let svd = nalgebra::linalg::SVD::new(cov_mat, true, true);
                let max_sv = svd.singular_values.max();
                let tol = (1e-12 * max_sv).max(1e-12);
                let solved = svd.solve(&sb_mat, tol).map_err(|_| {
                    ModelError::ProcessingError(
                        "LSQR solver failed to compute discriminant matrix".to_string(),
                    )
                })?;

                Array2::from_shape_vec((n_features, n_features), solved.as_slice().to_vec())
                    .map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to build discriminant matrix: {}",
                            e
                        ))
                    })
            }
            _ => Ok(cov_inv.dot(&sb)),
        }
    }

    /// Computes the projection matrix from the solver output
    fn compute_projection(
        &self,
        solver_matrix: &Array2<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>, ModelError> {
        let n_features = solver_matrix.nrows();
        let (eigenvalues, eigenvectors) = match self.solver {
            Solver::Eigen => {
                // Use symmetric eigen decomposition for stability
                let sym_matrix = (solver_matrix + &solver_matrix.t()) * 0.5;
                let slice = sym_matrix.as_slice().ok_or_else(|| {
                    ModelError::ProcessingError(
                        "Failed to convert symmetric matrix to slice".to_string(),
                    )
                })?;
                let mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, slice);
                let eig = nalgebra::linalg::SymmetricEigen::new(mat);
                (
                    Array1::from_vec(eig.eigenvalues.as_slice().to_vec()),
                    Array2::from_shape_vec(
                        (n_features, n_features),
                        eig.eigenvectors.as_slice().to_vec(),
                    )
                    .map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to build eigenvector matrix: {}",
                            e
                        ))
                    })?,
                )
            }
            Solver::SVD | Solver::LSQR => {
                // Use SVD to obtain principal directions
                let slice = solver_matrix.as_slice().ok_or_else(|| {
                    ModelError::ProcessingError(
                        "Failed to convert solver matrix to slice".to_string(),
                    )
                })?;
                let mat = nalgebra::DMatrix::from_row_slice(n_features, n_features, slice);
                let svd = nalgebra::linalg::SVD::new(mat, true, true);
                let u = svd.u.ok_or_else(|| {
                    ModelError::ProcessingError("SVD did not compute U matrix".to_string())
                })?;
                (
                    Array1::from_vec(svd.singular_values.as_slice().to_vec()),
                    Array2::from_shape_vec((n_features, n_features), u.as_slice().to_vec())
                        .map_err(|e| {
                            ModelError::ProcessingError(format!(
                                "Failed to build eigenvector matrix: {}",
                                e
                            ))
                        })?,
                )
            }
        };

        let mut eig_pairs: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        eig_pairs
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top components and normalize basis vectors
        let mut w = Array2::<f64>::zeros((n_features, n_components));
        for (component_idx, (i, _)) in eig_pairs.iter().take(n_components).enumerate() {
            let vec = eigenvectors.slice(s![.., *i]);
            let norm = vec.dot(&vec).sqrt();
            if norm <= 1e-12 {
                return Err(ModelError::ProcessingError(
                    "Eigenvector norm too small for stable projection".to_string(),
                ));
            }
            w.column_mut(component_idx).assign(&(&vec / norm));
        }

        Ok(w)
    }

    model_save_and_load_methods!(LDA);
}
