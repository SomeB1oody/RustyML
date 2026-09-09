//! Group Normalization layer: splits the channel axis into groups and normalizes each group
//! over the spatial axes and its own channels, per sample. The result has no dependence on
//! batch size

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::built_layer_shape_functions;
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape_function;
use crate::neural_network::layers::regularization::normalization::{
    group_norm_backward_core, group_norm_forward_core,
};
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_min_input_ndim, validate_num_groups, validate_num_groups_positive,
};
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Group Normalization layer for neural networks
///
/// Splits the channel axis into `num_groups` groups. For each sample, it computes 1 mean and
/// 1 variance per group, over the spatial axes and the channels inside that group. The group
/// never reaches across the batch axis. Channel divisibility is validated on every forward pass
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::UnaryLayer;
/// use ndarray::Array3;
///
/// // Create a GroupNormalization layer for input shape [batch, spatial, channels]
/// // with 4 groups dividing 8 channels
/// let mut gn_layer = GroupNormalization::new(4, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array3::ones((4, 32, 8)).into_dyn();
///
/// // A training pass normalizes within each group independently
/// let mut ctx = Ctx::training();
/// let output = gn_layer.forward_mut(&input, &mut ctx).unwrap();
/// ```
#[derive(Debug)]
pub struct GroupNormalization {
    /// Number of groups to divide channels into
    num_groups: usize,
    /// Small constant for numerical stability in normalization
    epsilon: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
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
    /// Whether the layer adds the shift `beta`
    center: bool,
    /// Whether the layer applies the scale `gamma`
    scale: bool,
}

impl GroupNormalization {
    /// Creates a new GroupNormalization layer
    ///
    /// # Parameters
    ///
    /// - `num_groups` - Number of groups to divide channels into
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New GroupNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `num_groups` is 0
    /// - `Error::InvalidParameter` - If `epsilon` is not positive or not finite
    pub fn new(num_groups: usize, epsilon: f32) -> Result<Self, Error> {
        validate_num_groups_positive(num_groups)?;
        validate_epsilon(epsilon)?;

        Ok(GroupNormalization {
            num_groups,
            epsilon,
            built: None,
            gamma: Tensor::ones([0].as_slice()),
            beta: Tensor::zeros([0].as_slice()),
            center: true,
            scale: true,
        })
    }

    /// Sets whether the layer adds the shift `beta` (defaults to `true`)
    ///
    /// With `center` set to false, the layer holds no `beta`. `param_count` counts none for it,
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
            // Put the array back at the identity shift, so an array the layer no longer holds
            // reaches no result
            self.beta = Tensor::zeros(self.beta.shape());
        }
        self
    }

    /// Sets whether the layer applies the scale `gamma` (defaults to `true`)
    ///
    /// With `scale` set to false, the layer holds no `gamma`. `param_count` counts none for it,
    /// `parameters` yields none for it, and a checkpoint of the layer holds no
    /// `<position>.gamma` path. `beta` keeps its own name and its own optimizer state. A
    /// checkpoint and an optimizer both address an array by name and never by position
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
            // Put the array back at the identity scale, so an array the layer no longer holds
            // reaches no result
            self.gamma = Tensor::ones(self.gamma.shape());
        }
        self
    }

    /// Sets the weights for the GroupNormalization layer
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
    ///   stored parameter shape
    pub fn set_weights(
        &mut self,
        gamma: impl Into<Option<Tensor>>,
        beta: impl Into<Option<Tensor>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("GroupNormalization"));
        }
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

/// What the forward pass of [`GroupNormalization`] parks for its backward pass
struct GroupNormalizationCache {
    /// The normalized input, before the scale and the shift
    x_normalized: Tensor,
    /// Per-instance `1 / sqrt(var + epsilon)`, 1 value per batch item and group
    inv_std: Tensor,
}

impl LayerBase for GroupNormalization {
    fn layer_type(&self) -> &str {
        "GroupNormalization"
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so dropping
        // `gamma` or `beta` corrects the count with no second formula to keep in step
        let gamma = if self.scale { self.gamma.len() } else { 0 };
        let beta = if self.center { self.beta.len() } else { 0 };
        ParamCounts::trainable(gamma + beta)
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let Self {
            gamma,
            beta,
            center,
            scale,
            ..
        } = self;
        let mut params = Vec::new();
        // Each tensor is pushed on its own, so a tensor the layer drops holds back no other
        if *scale {
            params.push(ParamRef::no_decay(
                "gamma",
                gamma.as_slice_mut().expect("gamma must be contiguous"),
            ));
        }
        if *center {
            params.push(ParamRef::no_decay(
                "beta",
                beta.as_slice_mut().expect("beta must be contiguous"),
            ));
        }
        params
    }

    built_layer_shape_functions!();

    named_weight_layer_functions!(
        trainable "gamma" => gamma if scale,
        trainable "beta" => beta if center,
    );
}

impl UnaryLayer for GroupNormalization {
    /// Allocates the per-channel arrays from the trailing axis of the input
    ///
    /// The channel axis is the last axis. An input of rank 1 has no channel axis, so the
    /// arrays hold 1 element that every position shares
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "GroupNormalization", input)? else {
            return Ok(());
        };
        input.check_min_rank("GroupNormalization", 1)?;
        let channels = match built.axes()[built.rank() - 1] {
            _ if built.rank() == 1 => 1,
            Some(extent) => extent,
            None => {
                return Err(Error::invalid_input(format!(
                    "GroupNormalization needs a fixed extent on the channel axis, and the shape {built} \
                     leaves that axis free"
                )));
            }
        };
        validate_num_groups(channels, self.num_groups)?;
        self.gamma = Tensor::ones([channels].as_slice());
        self.beta = Tensor::zeros([channels].as_slice());
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "GroupNormalization", input.shape())?;
        validate_min_input_ndim(input.ndim(), 3, "Group normalization")?;

        validate_num_groups(input.shape()[input.ndim() - 1], self.num_groups)?;

        let (output, x_normalized, inv_std) = group_norm_forward_core(
            input,
            self.num_groups,
            &self.gamma,
            &self.beta,
            self.epsilon,
        );

        if ctx.is_training() {
            ctx.push_cache(
                "GroupNormalization",
                GroupNormalizationCache {
                    x_normalized,
                    inv_std,
                },
            );
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !ctx.is_training() {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        let cache: GroupNormalizationCache = ctx.pop_cache("GroupNormalization")?;

        let (grad_input, grad_gamma, grad_beta) = group_norm_backward_core(
            grad_output,
            &cache.x_normalized,
            &cache.inv_std,
            self.num_groups,
            &self.gamma,
        );

        // An array the layer does not hold gets no gradient, so the store holds none and no
        // optimizer state is ever keyed on it
        if self.scale {
            ctx.add_grad("gamma", grad_gamma)?;
        }
        if self.center {
            ctx.add_grad("beta", grad_beta)?;
        }

        Ok(grad_input)
    }

    normalization_layer_output_shape_function!("GroupNormalization");
}
