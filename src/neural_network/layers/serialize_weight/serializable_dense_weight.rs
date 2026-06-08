use crate::error::Error;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of Dense layer weights.
///
/// # Fields
///
/// - `weight` - Weight matrix (input_dim, output_dim)
/// - `bias` - Bias matrix (1, output_dim)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDenseWeight {
    pub weight: Array2<f32>,
    pub bias: Array2<f32>,
}

impl ApplyWeights<Dense> for SerializableDenseWeight {
    fn apply_to_layer(&self, layer: &mut Dense) -> Result<(), Error> {
        layer.set_weights(self.weight.clone(), self.bias.clone())?;
        Ok(())
    }
}
