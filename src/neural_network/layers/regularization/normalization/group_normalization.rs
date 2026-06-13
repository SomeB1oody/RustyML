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
    from_channels_first, group_norm_backward_core, group_norm_forward_core, to_channels_first,
};
use crate::neural_network::layers::regularization::validation::{
    validate_channel_axis, validate_epsilon, validate_input_shape, validate_input_shape_not_empty,
    validate_min_input_ndim, validate_num_groups, validate_num_groups_positive,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use std::borrow::Cow;

/// Group Normalization layer for neural networks
///
/// Divides channels into groups and normalizes within each group per sample,
/// reducing dependence on batch size. Channel divisibility is validated during
/// the forward pass
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array3;
///
/// // Create a GroupNormalization layer for input shape [batch, channels, spatial]
/// // with 4 groups dividing 8 channels
/// let mut gn_layer = GroupNormalization::new(vec![4, 8, 32], 4, 1, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array3::ones((4, 8, 32)).into_dyn();
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
    /// Axis representing channels (typically 1 for [batch, channels, spatial...])
    channel_axis: usize,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Scale parameter (trainable)
    gamma: Tensor,
    /// Shift parameter (trainable)
    beta: Tensor,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Normalized input (channels-first), cached for the backward pass
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
    /// - `channel_axis` - Axis representing channels for inputs like [batch, channels, ...]
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
    /// - `Error::InvalidParameter` - If `epsilon` is not positive
    /// - `Error::InvalidParameter` - If `channel_axis` is out of bounds or is 0 (batch axis)
    pub fn new(
        input_shape: Vec<usize>,
        num_groups: usize,
        channel_axis: usize,
        epsilon: f32,
    ) -> Result<Self, Error> {
        validate_input_shape_not_empty(&input_shape)?;
        validate_num_groups_positive(num_groups)?;
        validate_epsilon(epsilon)?;
        validate_channel_axis(channel_axis, input_shape.len())?;

        // Any channel position works: forward/backward permute the channel axis to position 1 and run
        // a layout-equivariant channels-first core, so both NCHW and NHWC are supported

        // Parameters have the shape of the channel dimension
        let param_shape = if input_shape.len() > channel_axis {
            let num_channels = input_shape[channel_axis];
            // Divisibility is checked in forward() instead, matching the other layers' error pattern
            vec![num_channels]
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(GroupNormalization {
            num_groups,
            epsilon,
            channel_axis,
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
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `gamma` or `beta` does not match the stored parameter shape
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
        validate_channel_axis(self.channel_axis, input.ndim())?;

        // Permute to channels-first so the layout-equivariant core handles any channel position
        // (borrows when channel_axis == 1)
        let cf_input = to_channels_first(input, self.channel_axis);
        let input_cf = cf_input.as_ref();
        validate_num_groups(input_cf.shape()[1], self.num_groups)?;

        let (output, x_normalized, inv_std) = group_norm_forward_core(
            input_cf,
            self.num_groups,
            &self.gamma,
            &self.beta,
            self.epsilon,
        );

        // Cache the channels-first intermediates for the backward pass
        self.x_normalized = Some(x_normalized);
        self.inv_std = Some(inv_std);

        Ok(from_channels_first(output, self.channel_axis))
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Group normalization")?;
        validate_channel_axis(self.channel_axis, input.ndim())?;

        // See `forward`: permute channel to axis 1, run the channels-first core, permute back
        let cf_input = to_channels_first(input, self.channel_axis);
        let input_cf = cf_input.as_ref();
        validate_num_groups(input_cf.shape()[1], self.num_groups)?;

        let (output, _x_normalized, _inv_std) = group_norm_forward_core(
            input_cf,
            self.num_groups,
            &self.gamma,
            &self.beta,
            self.epsilon,
        );

        Ok(from_channels_first(output, self.channel_axis))
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        // Channels-first matches the cached intermediates; the input-gradient is permuted back below
        let cf_grad = to_channels_first(grad_output, self.channel_axis);
        let grad_cf = cf_grad.as_ref();

        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("GroupNormalization"))?;
        let inv_std = self
            .inv_std
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("GroupNormalization"))?;

        let (grad_input, grad_gamma, grad_beta) =
            group_norm_backward_core(grad_cf, x_normalized, inv_std, self.num_groups, &self.gamma);

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        Ok(from_channels_first(grad_input, self.channel_axis))
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
            params.push(ParamGrad {
                value: gamma.as_slice_mut().expect("gamma must be contiguous"),
                grad: grad_a.as_slice().expect("grad_gamma must be contiguous"),
            });
            params.push(ParamGrad {
                value: beta.as_slice_mut().expect("beta must be contiguous"),
                grad: grad_b.as_slice().expect("grad_beta must be contiguous"),
            });
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
