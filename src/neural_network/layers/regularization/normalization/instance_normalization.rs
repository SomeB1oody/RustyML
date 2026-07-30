//! Instance Normalization layer that normalizes each sample and channel independently

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::{InstanceNormalizationLayerWeight, LayerWeight};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape;
use crate::neural_network::layers::regularization::normalization::{
    group_norm_backward_core, group_norm_forward_core,
};
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_input_shape, validate_input_shape_not_empty, validate_min_input_ndim,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use std::borrow::Cow;

/// Instance Normalization layer for neural networks
///
/// Normalizes each sample and channel independently, which is useful for
/// style transfer and generative models
///
/// Instance normalization is group normalization with 1 group per channel, so it shares
/// `group_norm_forward_core` / `group_norm_backward_core` with `num_groups` set to the channel
/// count
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array3;
///
/// // Create an InstanceNormalization layer for input shape [batch, spatial, channels]
/// let mut in_layer = InstanceNormalization::new(vec![4, 32, 3], 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array3::ones((4, 32, 3)).into_dyn();
///
/// // During training, normalizes each channel of each sample independently
/// let output = in_layer.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct InstanceNormalization {
    /// Small constant for numerical stability in normalization
    epsilon: f32,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Scale parameter (trainable)
    gamma: Tensor,
    /// Shift parameter (trainable)
    beta: Tensor,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Normalized input, cached for the backward pass
    x_normalized: Option<Tensor>,
    /// Per-instance `1 / sqrt(var + epsilon)` from the forward pass, cached for the backward pass
    inv_std: Option<Tensor>,
    /// Gradient for the gamma parameter
    grad_gamma: Option<Tensor>,
    /// Gradient for the beta parameter
    grad_beta: Option<Tensor>,
}

impl InstanceNormalization {
    /// Creates a new InstanceNormalization layer
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New InstanceNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `input_shape` is empty
    /// - `Error::InvalidParameter` - If `epsilon` is not positive or not finite
    pub fn new(input_shape: Vec<usize>, epsilon: f32) -> Result<Self, Error> {
        validate_input_shape_not_empty(&input_shape)?;
        validate_epsilon(epsilon)?;

        // Parameters have the shape of the channel dimension
        let param_shape = if input_shape.len() > 1 {
            vec![input_shape[input_shape.len() - 1]]
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(InstanceNormalization {
            epsilon,
            input_shape,
            gamma: Tensor::ones(param_shape_ndarray),
            beta: Tensor::zeros(param_shape_ndarray),
            training: true,
            x_normalized: None,
            inv_std: None,
            grad_gamma: None,
            grad_beta: None,
        })
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the InstanceNormalization layer
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable)
    /// - `beta` - Shift parameter (trainable)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `gamma` or `beta` does not match the
    ///   existing weight shape
    pub fn set_weights(&mut self, gamma: Tensor, beta: Tensor) -> Result<(), Error> {
        validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}

impl Layer for InstanceNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Instance normalization")?;
        // 1 group per channel makes group normalization equal to instance normalization
        let num_channels = input.shape()[input.ndim() - 1];

        let (output, x_normalized, inv_std) =
            group_norm_forward_core(input, num_channels, &self.gamma, &self.beta, self.epsilon);

        self.x_normalized = Some(x_normalized);
        self.inv_std = Some(inv_std);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    ///
    /// # Errors
    ///
    /// Returns an error if the input shape or dimensionality is invalid
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Instance normalization")?;
        let num_channels = input.shape()[input.ndim() - 1];

        let (output, _x_normalized, _inv_std) =
            group_norm_forward_core(input, num_channels, &self.gamma, &self.beta, self.epsilon);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass the gradient through unchanged
            return Ok(grad_output.clone());
        }

        // The channel axis is last, matching the cached forward intermediates
        let num_channels = grad_output.shape()[grad_output.ndim() - 1];

        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("InstanceNormalization"))?;
        let inv_std = self
            .inv_std
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("InstanceNormalization"))?;

        let (grad_input, grad_gamma, grad_beta) = group_norm_backward_core(
            grad_output,
            x_normalized,
            inv_std,
            num_channels,
            &self.gamma,
        );

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "InstanceNormalization"
    }

    fn output_shape(&self) -> String {
        normalization_layer_output_shape!(self)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.gamma.len() + self.beta.len())
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            gamma,
            beta,
            grad_gamma,
            grad_beta,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(grad_a), Some(grad_b)) = (grad_gamma.as_ref(), grad_beta.as_ref()) {
            params.push(ParamGrad::no_decay(
                gamma.as_slice_mut().expect("gamma must be contiguous"),
                grad_a.as_slice().expect("grad_gamma must be contiguous"),
            ));
            params.push(ParamGrad::no_decay(
                beta.as_slice_mut().expect("beta must be contiguous"),
                grad_b.as_slice().expect("grad_beta must be contiguous"),
            ));
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::InstanceNormalization(InstanceNormalizationLayerWeight {
            gamma: Cow::Borrowed(&self.gamma),
            beta: Cow::Borrowed(&self.beta),
        })
    }

    mode_dependent_layer_trait!();
}
