//! Instance Normalization layer that normalizes each sample and channel independently

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape;
use crate::neural_network::layers::regularization::normalization::{
    group_norm_backward_core, group_norm_forward_core,
};
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_input_shape, validate_input_shape_not_empty, validate_min_input_ndim,
};
use crate::neural_network::layers::validation::{validate_optional_weight, validate_weight_shape};
use crate::neural_network::traits::{Layer, ParamGrad};

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
    ///
    /// The array stays allocated and holds every element at 1 when `scale` is false. A scale of
    /// 1 changes no value, so the forward pass reads it and gives the same result that dropping
    /// the multiply gives. `weights` hides the array and `parameters` never yields it
    gamma: Tensor,
    /// Shift parameter (trainable)
    ///
    /// The array stays allocated and holds every element at 0 when `center` is false. A shift
    /// of 0 changes every value except a negative zero, which it turns into a positive zero.
    /// `weights` hides the array and `parameters` never yields it
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
    /// Whether the layer adds the shift `beta`
    center: bool,
    /// Whether the layer applies the scale `gamma`
    scale: bool,
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
            center: true,
            scale: true,
        })
    }

    mode_dependent_layer_set_training!();

    /// Sets whether the layer adds the shift `beta` (defaults to `true`)
    ///
    /// With `center` set to false the layer holds no `beta`: `param_count` counts none for it,
    /// `parameters` yields none for it, and a checkpoint of the layer holds no
    /// `<position>.beta` path. The normalized value passes through unshifted
    ///
    /// # Parameters
    ///
    /// - `center` - `true` to add `beta`, `false` to leave it out
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        if !center {
            // Put the array back at the identity shift and drop any gradient a previous
            // backward pass left, so nothing the layer no longer holds can reach a result
            self.beta = Tensor::zeros(self.beta.shape());
            self.grad_beta = None;
        }
        self
    }

    /// Sets whether the layer applies the scale `gamma` (defaults to `true`)
    ///
    /// With `scale` set to false the layer holds no `gamma`: `param_count` counts none for it,
    /// `parameters` yields none for it, and a checkpoint of the layer holds no
    /// `<position>.gamma` path. `beta` keeps its own name and its own optimizer state, because
    /// a checkpoint and an optimizer both address an array by name and never by position
    ///
    /// # Parameters
    ///
    /// - `scale` - `true` to apply `gamma`, `false` to leave it out
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        if !scale {
            // Put the array back at the identity scale and drop any gradient a previous
            // backward pass left, so nothing the layer no longer holds can reach a result
            self.gamma = Tensor::ones(self.gamma.shape());
            self.grad_gamma = None;
        }
        self
    }

    /// Sets the weights for the InstanceNormalization layer
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable), or `None` for a layer built with
    ///   [`with_scale(false)`](Self::with_scale)
    /// - `beta` - Shift parameter (trainable), or `None` for a layer built with
    ///   [`with_center(false)`](Self::with_center)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `gamma` or `beta` does not match the
    ///   existing weight shape
    pub fn set_weights(
        &mut self,
        gamma: impl Into<Option<Tensor>>,
        beta: impl Into<Option<Tensor>>,
    ) -> Result<(), Error> {
        let gamma = validate_optional_weight("gamma", "scale", self.scale, gamma.into())?;
        let beta = validate_optional_weight("beta", "center", self.center, beta.into())?;
        if let Some(gamma) = gamma.as_ref() {
            validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        }
        if let Some(beta) = beta.as_ref() {
            validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        }
        if let Some(gamma) = gamma {
            self.gamma = gamma;
        }
        if let Some(beta) = beta {
            self.beta = beta;
        }
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

        // An array the layer does not hold keeps no gradient, so `parameters` yields
        // none for it and no optimizer state is ever keyed on it
        self.grad_gamma = self.scale.then_some(grad_gamma);
        self.grad_beta = self.center.then_some(grad_beta);

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "InstanceNormalization"
    }

    fn output_shape(&self) -> String {
        normalization_layer_output_shape!(self)
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so dropping
        // `gamma` or `beta` corrects the count with no second formula to keep in step
        let gamma = if self.scale { self.gamma.len() } else { 0 };
        let beta = if self.center { self.beta.len() } else { 0 };
        ParamCounts::trainable(gamma + beta)
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
        // Each tensor is pushed on its own, so a tensor without a gradient holds back no other
        if let Some(grad) = grad_gamma.as_ref() {
            params.push(ParamGrad::no_decay(
                "gamma",
                gamma.as_slice_mut().expect("gamma must be contiguous"),
                grad.as_slice().expect("grad_gamma must be contiguous"),
            ));
        }
        if let Some(grad) = grad_beta.as_ref() {
            params.push(ParamGrad::no_decay(
                "beta",
                beta.as_slice_mut().expect("beta must be contiguous"),
                grad.as_slice().expect("grad_beta must be contiguous"),
            ));
        }
        params
    }

    named_weight_layer_functions!(
        trainable "gamma" => gamma if scale,
        trainable "beta" => beta if center,
    );

    mode_dependent_layer_trait!();
}
