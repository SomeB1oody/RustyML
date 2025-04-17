use super::*;
use ndarray::s;

/// Mean Squared Error loss function
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Creates a new instance of MeanSquaredError
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanSquaredError {
    /// Computes the Mean Squared Error between predicted and true values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values
    /// - `y_pred` - Tensor with predicted values
    ///
    /// # Returns
    ///
    /// `f32` - Average of squared differences between predictions and ground truth
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Calculate the difference between predictions and ground truth
        let diff = y_pred - y_true;

        // Calculate the squared difference
        let squared_diff = &diff.mapv(|x| x * x);

        // Calculate the mean (sum divided by number of elements)
        let n = squared_diff.len() as f32;
        squared_diff.sum() / n
    }

    /// Computes the gradient of Mean Squared Error with respect to predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values
    /// - `y_pred` - Tensor with predicted values
    ///
    /// # Returns
    ///
    /// * `Tensor` - Gradient tensor for backpropagation
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Calculate the difference between predictions and ground truth
        let diff = y_pred - y_true;

        // Gradient is 2 times the difference divided by sample count
        let n = diff.len() as f32;
        diff.mapv(|x| 2.0 * x / n)
    }
}

/// Mean Absolute Error loss function
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Creates a new instance of MeanAbsoluteError
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanAbsoluteError {
    /// Computes the Mean Absolute Error between predicted and true values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values
    /// - `y_pred` - Tensor with predicted values
    ///
    /// # Returns
    ///
    /// * `f32` - Average of absolute differences between predictions and ground truth
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        let diff = y_pred - y_true;
        let abs_diff = diff.mapv(|x| x.abs());
        abs_diff.sum() / (y_true.len() as f32)
    }

    /// Computes the gradient of Mean Absolute Error with respect to predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values
    /// - `y_pred` - Tensor with predicted values
    ///
    /// # Returns
    ///
    /// * `Tensor` - Gradient tensor for backpropagation
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let n = y_true.len() as f32;
        (y_pred - y_true).mapv(|x| {
            if x > 0.0 {
                1.0 / n
            } else if x < 0.0 {
                -1.0 / n
            } else {
                0.0
            }
        })
    }
}

/// Binary Cross Entropy loss function for binary classification
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    /// Creates a new instance of BinaryCrossEntropy
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for BinaryCrossEntropy {
    /// Computes the Binary Cross Entropy loss between predicted and true values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values (0 or 1)
    /// - `y_pred` - Tensor with predicted probabilities in range \[0,1\]
    ///
    /// # Returns
    ///
    /// * `f32` - Binary cross entropy loss value
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Ensure predictions are in range (0,1) to avoid numerical issues
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Binary cross entropy formula: -1/n * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        let losses = y_true.mapv(|y_t| y_t).to_owned() * &y_pred_clipped.mapv(|y_p| y_p.ln())
            + (1.0 - y_true).mapv(|y_t| y_t) * &(1.0 - &y_pred_clipped).mapv(|y_p| y_p.ln());

        // Calculate average loss (with negative sign)
        let n = losses.len() as f32;
        -losses.sum() / n
    }

    /// Computes the gradient of Binary Cross Entropy with respect to predictions
    ///
    /// # Arguments
    ///
    /// - `y_true` - Tensor with ground truth values (0 or 1)
    /// - `y_pred` - Tensor with predicted probabilities in range \[0,1\]
    ///
    /// # Returns
    ///
    /// * `Tensor` - Gradient tensor for backpropagation
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in range (0,1) to avoid numerical issues
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Binary cross entropy gradient: -y_true/y_pred + (1-y_true)/(1-y_pred)
        let grad = -y_true / &y_pred_clipped + (1.0 - y_true) / (1.0 - &y_pred_clipped);

        // Divide by sample count to get average gradient
        let n = grad.len() as f32;
        grad / n
    }
}

/// Categorical Cross Entropy loss function for multi-class classification
pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    /// Creates a new instance of CategoricalCrossEntropy
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for CategoricalCrossEntropy {
    /// Computes the Categorical Cross Entropy loss between predicted and true values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values in one-hot encoding
    /// - `y_pred` - Tensor with predicted probabilities for each class
    ///
    /// # Returns
    ///
    /// * `f32` - Categorical cross entropy loss value
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Ensure predictions are in a numerically stable range to avoid log(0) issues
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Calculate multi-class cross entropy: -Σ[y_true * log(y_pred)]
        // Here y_true must be one-hot encoded
        let losses = y_true * &y_pred_clipped.mapv(|y_p| y_p.ln());

        // Calculate average loss (with negative sign)
        let n = y_true.shape()[0] as f32; // Assume first dimension is sample count
        -losses.sum() / n
    }

    /// Computes the gradient of Categorical Cross Entropy with respect to predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values in one-hot encoding
    /// - `y_pred` - Tensor with predicted probabilities for each class
    ///
    /// # Returns
    ///
    /// * `Tensor` - Gradient tensor for backpropagation
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in a numerically stable range
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Multi-class cross entropy gradient is -y_true / y_pred
        let grad = -y_true / &y_pred_clipped;

        // Divide by sample count to get average gradient
        let n = y_true.shape()[0] as f32; // Assume first dimension is sample count
        grad / n
    }
}

/// Sparse Categorical Cross Entropy loss function for multi-class classification
/// where true labels are integers instead of one-hot vectors
pub struct SparseCategoricalCrossEntropy;

impl SparseCategoricalCrossEntropy {
    /// Creates a new instance of SparseCategoricalCrossEntropy
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for SparseCategoricalCrossEntropy {
    /// Computes the Sparse Categorical Cross Entropy loss between predicted values
    /// and class indices
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with class indices as ground truth
    /// - `y_pred` - Tensor with predicted probabilities for each class
    ///
    /// # Returns
    ///
    /// * `f32` - Sparse categorical cross entropy loss value
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Ensure predictions are in a numerically stable range to avoid log(0) issues
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Assume y_true contains class indices (integer values)
        // We need to get the probability of the true class for each sample
        let mut total_loss = 0.0;
        let batch_size = y_true.shape()[0];

        for i in 0..batch_size {
            // Get the true class index for the current sample
            let class_idx = y_true.slice(s![i, ..]).iter().next().unwrap().round() as usize;

            // First save the slice view, then extract value from it
            let slice = y_pred_clipped.slice(s![i, class_idx]);
            let predicted_prob = slice.iter().next().unwrap();

            // Accumulate cross entropy loss: -log(predicted_prob)
            total_loss -= predicted_prob.ln();
        }

        // Return average loss
        total_loss / batch_size as f32
    }

    /// Computes the gradient of Sparse Categorical Cross Entropy with respect to predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with class indices as ground truth
    /// - `y_pred` - Tensor with predicted probabilities for each class
    ///
    /// # Returns
    ///
    /// * `Tensor` - Gradient tensor for backpropagation
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in a numerically stable range
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Create a gradient tensor with the same shape as y_pred, initialized to 0
        let mut grad = y_pred.clone();
        grad.fill(0.0);

        let batch_size = y_true.shape()[0];

        for i in 0..batch_size {
            // Get the true class index for the current sample
            let class_idx = y_true.slice(s![i, ..]).iter().next().unwrap().round() as usize;

            // First save the slice view, then extract value from it
            let slice = y_pred_clipped.slice(s![i, class_idx]);
            let predicted_prob = slice.iter().next().unwrap();

            // Modify the gradient at the corresponding position
            let mut view = grad.slice_mut(s![i, class_idx]);
            *view.iter_mut().next().unwrap() = -1.0 / predicted_prob;
        }

        // Return average gradient
        grad / batch_size as f32
    }
}
