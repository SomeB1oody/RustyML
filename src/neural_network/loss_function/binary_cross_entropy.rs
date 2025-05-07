use crate::neural_network::{LossFunction, Tensor};

/// Binary Cross Entropy loss function for binary classification
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    /// Creates a new instance of BinaryCrossEntropy
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for BinaryCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Ensure predictions are in range (0,1) to avoid numerical issues
        let mut y_pred_clipped = y_pred.clone();
        y_pred_clipped.par_mapv_inplace(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Binary cross entropy formula: -1/n * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        let losses = y_true.mapv(|y_t| y_t).to_owned() * &y_pred_clipped.mapv(|y_p| y_p.ln())
            + (1.0 - y_true).mapv(|y_t| y_t) * &(1.0 - &y_pred_clipped).mapv(|y_p| y_p.ln());

        // Calculate average loss (with negative sign)
        let n = losses.len() as f32;
        -losses.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in range (0,1) to avoid numerical issues
        let mut y_pred_clipped = y_pred.clone();
        y_pred_clipped.par_mapv_inplace(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Binary cross entropy gradient: -y_true/y_pred + (1-y_true)/(1-y_pred)
        let grad = -y_true / &y_pred_clipped + (1.0 - y_true) / (1.0 - &y_pred_clipped);

        // Divide by sample count to get average gradient
        let n = grad.len() as f32;
        grad / n
    }
}
