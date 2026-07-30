//! Group Normalization layer: divides channels into groups and normalizes within each
//! group per sample, independent of batch size

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::{GroupNormalizationLayerWeight, LayerWeight};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape;
use crate::neural_network::layers::regularization::normalization::{
    group_norm_backward_core, group_norm_forward_core,
};
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_input_shape, validate_input_shape_not_empty,
    validate_min_input_ndim, validate_num_groups, validate_num_groups_positive,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use std::borrow::Cow;

/// Group Normalization layer for neural networks
///
/// Divides channels into groups and normalizes within each group per sample, reducing
/// dependence on batch size. Channel divisibility is validated on every `forward` or `predict`
/// call
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array3;
///
/// // Create a GroupNormalization layer for input shape [batch, spatial, channels]
/// // with 4 groups dividing 8 channels
/// let mut gn_layer = GroupNormalization::new(vec![4, 32, 8], 4, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array3::ones((4, 32, 8)).into_dyn();
///
/// // During training, normalizes within each group independently
/// let output = gn_layer.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct GroupNormalization {
    /// Number of groups to divide channels into
    num_groups: usize,
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

impl GroupNormalization {
    /// Creates a new GroupNormalization layer
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor
    /// - `num_groups` - Number of groups to divide channels into
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New GroupNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `input_shape` is empty
    /// - `Error::InvalidParameter` - If `num_groups` is 0
    /// - `Error::InvalidParameter` - If `epsilon` is not positive or not finite
    pub fn new(input_shape: Vec<usize>, num_groups: usize, epsilon: f32) -> Result<Self, Error> {
        validate_input_shape_not_empty(&input_shape)?;
        validate_num_groups_positive(num_groups)?;
        validate_epsilon(epsilon)?;

        // Parameters have the shape of the channel dimension
        let param_shape = if input_shape.len() > 1 {
            let num_channels = input_shape[input_shape.len() - 1];
            // Divisibility is checked in forward() instead, matching the other layers' error
            // pattern
            vec![num_channels]
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(GroupNormalization {
            num_groups,
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

    /// Sets the weights for the GroupNormalization layer
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable)
    /// - `beta` - Shift parameter (trainable)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `gamma` or `beta` does not match the
    ///   stored parameter shape
    pub fn set_weights(&mut self, gamma: Tensor, beta: Tensor) -> Result<(), Error> {
        validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}

impl Layer for GroupNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Group normalization")?;

        validate_num_groups(input.shape()[input.ndim() - 1], self.num_groups)?;

        let (output, x_normalized, inv_std) = group_norm_forward_core(
            input,
            self.num_groups,
            &self.gamma,
            &self.beta,
            self.epsilon,
        );

        // Cache the intermediates for the backward pass
        self.x_normalized = Some(x_normalized);
        self.inv_std = Some(inv_std);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Group normalization")?;

        validate_num_groups(input.shape()[input.ndim() - 1], self.num_groups)?;

        let (output, _x_normalized, _inv_std) = group_norm_forward_core(
            input,
            self.num_groups,
            &self.gamma,
            &self.beta,
            self.epsilon,
        );

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("GroupNormalization"))?;
        let inv_std = self
            .inv_std
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("GroupNormalization"))?;

        let (grad_input, grad_gamma, grad_beta) = group_norm_backward_core(
            grad_output,
            x_normalized,
            inv_std,
            self.num_groups,
            &self.gamma,
        );

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GroupNormalization"
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
        LayerWeight::GroupNormalization(GroupNormalizationLayerWeight {
            gamma: Cow::Borrowed(&self.gamma),
            beta: Cow::Borrowed(&self.beta),
        })
    }

    mode_dependent_layer_trait!();
}
