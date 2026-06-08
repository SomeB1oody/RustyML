use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::loss_function::{clip_probabilities, validate_same_shape};
use crate::neural_network::neural_network_trait::LossFunction;

/// Categorical Cross Entropy loss function for multi-class classification
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::loss_function::*;
/// use rustyml::neural_network::neural_network_trait::LossFunction;
/// use ndarray::array;
///
/// // Create the loss function
/// let loss_fn = CategoricalCrossEntropy::new();
///
/// // Ground truth labels (one-hot encoded)
/// // Class 0: [1, 0, 0], Class 1: [0, 1, 0], Class 2: [0, 0, 1]
/// let y_true = array![
///     [1.0f32, 0.0f32, 0.0f32],  // Sample 1: belongs to class 0
///     [0.0f32, 1.0f32, 0.0f32],  // Sample 2: belongs to class 1
///     [0.0f32, 0.0f32, 1.0f32]   // Sample 3: belongs to class 2
/// ].into_dyn();
///
/// // Predicted probabilities (softmax output)
/// let y_pred = array![
///     [0.8f32, 0.1f32, 0.1f32],  // Sample 1: high confidence for class 0
///     [0.2f32, 0.7f32, 0.1f32],  // Sample 2: good confidence for class 1
///     [0.1f32, 0.2f32, 0.7f32]   // Sample 3: good confidence for class 2
/// ].into_dyn();
///
/// // Compute loss
/// let loss = loss_fn.compute_loss(&y_true, &y_pred).unwrap();
/// println!("Categorical Cross Entropy Loss: {:.4}", loss);
///
/// // Compute gradient for backpropagation
/// let gradient = loss_fn.compute_grad(&y_true, &y_pred).unwrap();
/// println!("Gradient shape: {:?}", gradient.shape());
/// ```
pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    /// Creates a new instance of CategoricalCrossEntropy
    ///
    /// # Returns
    ///
    /// - `CategoricalCrossEntropy` - Returns a unit-like struct `CategoricalCrossEntropy`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CategoricalCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates that one-hot targets and predictions are non-empty and shape-compatible.
///
/// Categorical cross entropy is an element-wise product reduced over the batch, so the two
/// tensors must share the same shape, and the batch axis must be non-empty (it is the divisor).
fn validate_shapes(y_true: &Tensor, y_pred: &Tensor) -> Result<(), Error> {
    if y_true.is_empty() {
        return Err(Error::empty_input(
            "CategoricalCrossEntropy expects non-empty y_true",
        ));
    }
    validate_same_shape(y_true, y_pred)
}

impl LossFunction for CategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error> {
        validate_shapes(y_true, y_pred)?;

        // Ensure predictions are in a numerically stable range to avoid log(0) issues
        let y_pred_clipped = clip_probabilities(y_pred);

        // Calculate multi-class cross entropy: -Σ[y_true * log(y_pred)]
        // Here y_true must be one-hot encoded
        let losses = y_true * &y_pred_clipped.mapv(|y_p| y_p.ln());

        // Calculate average loss (with negative sign)
        let n = y_true.shape()[0] as f32; // Assume first dimension is sample count
        Ok(-losses.sum() / n)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        validate_shapes(y_true, y_pred)?;

        // Ensure predictions are in a numerically stable range
        let y_pred_clipped = clip_probabilities(y_pred);

        // Multi-class cross entropy gradient is -y_true / y_pred
        let grad = -y_true / &y_pred_clipped;

        // Divide by sample count to get average gradient
        let n = y_true.shape()[0] as f32; // Assume first dimension is sample count
        Ok(grad / n)
    }
}
