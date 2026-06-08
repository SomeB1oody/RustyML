use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::{clip_probabilities, validate_same_shape};
use crate::neural_network::traits::Loss;

/// Binary Cross Entropy loss function for binary classification
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::losses::*;
/// use rustyml::neural_network::Tensor;
/// use rustyml::neural_network::traits::Loss;
/// use ndarray::array;
///
/// // Create a binary cross entropy loss function
/// let bce_loss = BinaryCrossEntropy::new();
///
/// // True binary labels (0 or 1)
/// let y_true = Tensor::from(array![[0.0f32, 1.0f32, 1.0f32, 0.0f32]].into_dyn());
///
/// // Predicted probabilities (between 0 and 1)
/// let y_pred = Tensor::from(array![[0.1f32, 0.9f32, 0.8f32, 0.2f32]].into_dyn());
///
/// // Compute the loss
/// let loss = bce_loss.compute_loss(&y_true, &y_pred).unwrap();
/// println!("Binary Cross Entropy Loss: {:.4}", loss);
///
/// // Compute gradients for backpropagation
/// let gradients = bce_loss.compute_grad(&y_true, &y_pred).unwrap();
/// println!("Gradients: {:?}", gradients);
/// ```
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    /// Creates a new instance of BinaryCrossEntropy
    ///
    /// # Returns
    ///
    /// - `BinaryCrossEntropy` - Returns a unit-like struct `BinaryCrossEntropy`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for BinaryCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for BinaryCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error> {
        validate_same_shape(y_true, y_pred)?;

        // Ensure predictions are in range (0,1) to avoid numerical issues
        let y_pred_clipped = clip_probabilities(y_pred);

        // Binary cross entropy formula: -1/n * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        let log_pred = y_pred_clipped.mapv(|y_p| y_p.ln());
        let log_one_minus_pred = (1.0 - &y_pred_clipped).mapv(|y_p| y_p.ln());
        let losses = y_true * &log_pred + (1.0 - y_true) * &log_one_minus_pred;

        // Calculate average loss (with negative sign)
        let n = losses.len() as f32;
        Ok(-losses.sum() / n)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        validate_same_shape(y_true, y_pred)?;

        // Ensure predictions are in range (0,1) to avoid numerical issues
        let y_pred_clipped = clip_probabilities(y_pred);

        // Binary cross entropy gradient: -y_true/y_pred + (1-y_true)/(1-y_pred)
        let grad = -y_true / &y_pred_clipped + (1.0 - y_true) / (1.0 - &y_pred_clipped);

        // Divide by element count to get average gradient
        let n = grad.len() as f32;
        Ok(grad / n)
    }
}
