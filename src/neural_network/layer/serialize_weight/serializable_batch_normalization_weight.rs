use super::*;

/// Serializable representation of BatchNormalization layer weights.
///
/// # Fields
///
/// - `gamma` - Scale parameter values flattened into a vector
/// - `beta` - Shift parameter values flattened into a vector
/// - `running_mean` - Running mean values flattened into a vector
/// - `running_var` - Running variance values flattened into a vector
/// - `shape` - Original parameter shape used to rebuild arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableBatchNormalizationWeight {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub shape: Vec<usize>,
}

impl ApplyWeights<BatchNormalization> for SerializableBatchNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut BatchNormalization) -> Result<(), IoError> {
        let gamma =
            ArrayD::from_shape_vec(self.shape.as_slice(), self.gamma.clone()).map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
        let beta =
            ArrayD::from_shape_vec(self.shape.as_slice(), self.beta.clone()).map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
        let running_mean = ArrayD::from_shape_vec(self.shape.as_slice(), self.running_mean.clone())
            .map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
        let running_var = ArrayD::from_shape_vec(self.shape.as_slice(), self.running_var.clone())
            .map_err(|e| {
            IoError::StdIoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            ))
        })?;
        layer.set_weights(gamma, beta, running_mean, running_var);
        Ok(())
    }
}
