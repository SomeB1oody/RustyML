use crate::neural_network::{LossFunction, Tensor};

/// Mean Absolute Error loss function
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Creates a new instance of MeanAbsoluteError
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanAbsoluteError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        let mut diff = y_pred - y_true;
        diff.par_mapv_inplace(|x| x.abs());
        diff.sum() / (y_true.len() as f32)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let n = y_true.len() as f32;
        let mut result = y_pred - y_true;
        result.par_mapv_inplace(|x| {
            if x > 0.0 {
                1.0 / n
            } else if x < 0.0 {
                -1.0 / n
            } else {
                0.0
            }
        });
        result
    }
}
