//! Batch normalization layer that normalizes each mini-batch per channel
//!
//! It normalizes over the batch axis for 2-D inputs, and over the batch and spatial axes for
//! rank > 2 (convolutional) inputs
//!
//! Both are the same pass. Under the crate's channels-last layout, the channel axis is
//! innermost. A `[batch, spatial..., channels]` buffer already *is* the `[M, C]` matrix the
//! per-channel folds read, with `M = batch * spatial`. Collapsing the leading axes is a
//! reinterpretation of the same bytes, not a reshape or a transpose. That is why 1 code path
//! serves every rank >= 2

use super::col_fold_parallel_min_elems;
use super::folds::{par_col_dot, par_col_sum, rows_per_block};
use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::built_layer_shape_functions;
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape_function;
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_momentum,
};
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Shape, StateSlot, Tensor};
use ndarray::Axis;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

tunable_gate! {
    /// Total-element count above which forward and backward switch from sequential to parallel
    ///
    /// Gates the centering, normalize, and gradient passes, each of which streams several
    /// arrays over the full tensor
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) BATCH_NORM_PARALLEL_THRESHOLD => batch_norm_parallel_threshold / set_batch_norm_parallel_threshold = 262_144
}

/// Batch Normalization layer for neural networks
///
/// Normalizes each mini-batch to keep activations centered and scaled, improving training
/// stability and speed
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::UnaryLayer;
/// use rustyml::neural_network::Ctx;
/// use ndarray::Array2;
///
/// // Create a BatchNormalization layer
/// let mut bn = BatchNormalization::new(0.99, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, normalizes the input
/// let mut ctx = Ctx::training();
/// let output = bn.forward_mut(&input, &mut ctx).unwrap();
/// ```
#[derive(Debug)]
pub struct BatchNormalization {
    /// Small constant for numerical stability in normalization
    epsilon: f32,
    /// Momentum for the moving average of mean and variance
    momentum: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Scale parameter (trainable)
    ///
    /// The array stays allocated and holds every element at 1 when `scale` is false. A scale of
    /// 1 changes no value, so the forward pass reads it and gives the same result that dropping
    /// the multiply gives. `weights` hides the array and `parameters_mut` never yields it
    gamma: Tensor,
    /// Shift parameter (trainable)
    ///
    /// The array stays allocated and holds every element at 0 when `center` is false. A shift
    /// of 0 changes every value except a negative zero, which it turns into a positive zero.
    /// `weights` hides the array and `parameters_mut` never yields it
    beta: Tensor,
    /// Running mean for inference
    moving_mean: Tensor,
    /// Running variance for inference
    moving_variance: Tensor,
    /// Whether the layer adds the shift `beta`
    center: bool,
    /// Whether the layer applies the scale `gamma`
    scale: bool,
}

impl BatchNormalization {
    /// Creates a new BatchNormalization layer
    ///
    /// # Parameters
    ///
    /// - `momentum` - Momentum for the moving average of mean and variance (typically 0.9 or 0.99)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New BatchNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `momentum` is not between 0.0 and 1.0
    /// - `Error::InvalidParameter` - If `epsilon` is not positive or not finite
    pub fn new(momentum: f32, epsilon: f32) -> Result<Self, Error> {
        validate_momentum(momentum)?;
        validate_epsilon(epsilon)?;

        Ok(BatchNormalization {
            epsilon,
            momentum,
            built: None,
            gamma: Tensor::ones([0].as_slice()),
            beta: Tensor::zeros([0].as_slice()),
            moving_mean: Tensor::zeros([0].as_slice()),
            moving_variance: Tensor::ones([0].as_slice()),
            center: true,
            scale: true,
        })
    }

    /// Sets whether the layer adds the shift `beta` (defaults to `true`)
    ///
    /// With `center` set to false, the layer holds no `beta`. `param_count` counts none for it,
    /// `parameters_mut` yields none for it, and a checkpoint of the layer holds no
    /// `<position>.beta` path. The moving mean and the moving variance stay, because they are
    /// state that the layer keeps and not parameters that an optimizer updates
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
            // Put the array back at the identity shift, so an array the layer does not hold
            // reaches no result
            self.beta = Tensor::zeros(self.beta.shape());
        }
        self
    }

    /// Sets whether the layer applies the scale `gamma` (defaults to `true`)
    ///
    /// With `scale` set to false, the layer holds no `gamma`. `param_count` counts none for it,
    /// `parameters_mut` yields none for it, and a checkpoint of the layer holds no
    /// `<position>.gamma` path. `beta` keeps its own name and its own optimizer state, because
    /// a checkpoint and an optimizer both address an array by name, never by position
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
            // Put the array back at the identity scale, so an array the layer does not hold
            // reaches no result
            self.gamma = Tensor::ones(self.gamma.shape());
        }
        self
    }

    /// Sets the weights for the BatchNormalization layer
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable), or `None` for a layer built with
    ///   [`with_scale(false)`](Self::with_scale)
    /// - `beta` - Shift parameter (trainable), or `None` for a layer built with
    ///   [`with_center(false)`](Self::with_center)
    /// - `moving_mean` - Running mean for inference
    /// - `moving_variance` - Running variance for inference
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   layer's expected shape
    pub fn set_weights(
        &mut self,
        gamma: impl Into<Option<Tensor>>,
        beta: impl Into<Option<Tensor>>,
        moving_mean: Tensor,
        moving_variance: Tensor,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("BatchNormalization"));
        }
        let gamma = validate_optional_weight("gamma", "scale", self.scale, gamma.into())?;
        let beta = validate_optional_weight("beta", "center", self.center, beta.into())?;
        if let Some(gamma) = gamma.as_ref() {
            validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        }
        if let Some(beta) = beta.as_ref() {
            validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        }
        validate_weight_shape("moving_mean", self.moving_mean.shape(), moving_mean.shape())?;
        validate_weight_shape(
            "moving_variance",
            self.moving_variance.shape(),
            moving_variance.shape(),
        )?;
        if let Some(gamma) = gamma {
            self.gamma = gamma;
        }
        if let Some(beta) = beta {
            self.beta = beta;
        }
        self.moving_mean = moving_mean;
        self.moving_variance = moving_variance;
        Ok(())
    }
}

/// What the forward pass of [`BatchNormalization`] parks for its backward pass
struct BatchNormalizationCache {
    /// Per-channel variance of the mini-batch
    batch_var: Tensor,
    /// The input after the subtraction of the mean and the divide by the standard deviation
    x_normalized: Tensor,
    /// The input after the subtraction of the mean
    x_centered: Tensor,
}

impl LayerBase for BatchNormalization {
    fn layer_type(&self) -> &str {
        "BatchNormalization"
    }

    fn param_count(&self) -> ParamCounts {
        // Running statistics are non-trainable: they move in the forward pass, not through the
        // optimizer. Read the arrays the layer holds, not the configuration, so dropping `gamma`
        // or `beta` corrects the count with no second formula to keep in step
        let gamma = if self.scale { self.gamma.len() } else { 0 };
        let beta = if self.center { self.beta.len() } else { 0 };
        ParamCounts::new(
            gamma + beta,
            self.moving_mean.len() + self.moving_variance.len(),
        )
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
        // Each array is pushed on its own, so an array the layer drops holds back no other
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
        non_trainable "moving_mean" => moving_mean,
        non_trainable "moving_variance" => moving_variance,
    );

    /// Takes back the running statistics that the training forward pass proposed
    ///
    /// The forward pass reads `&self`, so it writes the new running mean and the new running
    /// variance into the state channel of the context. This moves them into the layer
    fn apply_state(&mut self, state: &mut StateSlot<'_>) {
        if let Some(moving_mean) = state.take::<Tensor>("moving_mean") {
            self.moving_mean = moving_mean;
        }
        if let Some(moving_variance) = state.take::<Tensor>("moving_variance") {
            self.moving_variance = moving_variance;
        }
    }
}

impl UnaryLayer for BatchNormalization {
    /// Allocates the per-channel arrays from the trailing axis of the input
    ///
    /// The channel axis is the last axis. An input of rank 1 has no channel axis, so the
    /// arrays hold 1 element that every position shares
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "BatchNormalization", input)? else {
            return Ok(());
        };
        input.check_min_rank("BatchNormalization", 1)?;
        let channels = match built.axes()[built.rank() - 1] {
            _ if built.rank() == 1 => 1,
            Some(extent) => extent,
            None => {
                return Err(Error::invalid_input(format!(
                    "BatchNormalization needs a fixed extent on the channel axis, and the shape {built} \
                     leaves that axis free"
                )));
            }
        };
        self.gamma = Tensor::ones([channels].as_slice());
        self.beta = Tensor::zeros([channels].as_slice());
        self.moving_mean = Tensor::zeros([channels].as_slice());
        self.moving_variance = Tensor::ones([channels].as_slice());
        self.built = Some(built);
        Ok(())
    }

    /// Normalizes with the statistics of the mini-batch during training, and with the running
    /// statistics during inference
    ///
    /// A training pass proposes a new running mean and a new running variance through the state
    /// channel of the context. The layer takes them back in
    /// [`apply_state`](LayerBase::apply_state). An inference pass proposes nothing at all
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "BatchNormalization", input.shape())?;

        // The parallel passes below need a contiguous slice, so only a non-contiguous input
        // pays for a copy
        let owned;
        let input = if input.is_standard_layout() {
            input
        } else {
            owned = input.as_standard_layout().into_owned();
            &owned
        };

        if ctx.is_training() {
            let total_elements = input.len();
            // Every buffer derived from `input` below is either a fresh `Tensor::zeros` or an
            // owned ndarray result, so the rank >= 2 arms below can take the slice directly
            let use_col_fold = input.ndim() >= 2;
            let channels = if use_col_fold {
                input.shape()[input.ndim() - 1]
            } else {
                1
            };
            let m_rows = if use_col_fold {
                total_elements / channels.max(1)
            } else {
                input.shape()[0]
            };
            let col_stats_parallel = total_elements >= col_fold_parallel_min_elems();

            // Mean across the batch dimension (axis 0): a row-block deterministic fold, on
            // rayon above the column-stats gate
            let batch_mean = if use_col_fold {
                let s = input
                    .as_slice()
                    .expect("rank >= 2 batch-norm buffers are standard layout");
                par_col_sum(s, channels, col_stats_parallel, 1.0) / m_rows as f32
            } else {
                input.mean_axis(Axis(0)).unwrap()
            };

            let x_centered = if total_elements >= batch_norm_parallel_threshold() {
                let mut x_centered = Tensor::zeros(input.raw_dim());
                let mean_s = batch_mean.as_slice().unwrap();
                let feature_size = mean_s.len();
                let chunk = rows_per_block(feature_size) * feature_size;
                x_centered
                    .as_slice_mut()
                    .unwrap()
                    .par_chunks_mut(chunk)
                    .zip(input.as_slice().unwrap().par_chunks(chunk))
                    .for_each(|(out_c, in_c)| {
                        let rows = out_c
                            .chunks_exact_mut(feature_size)
                            .zip(in_c.chunks_exact(feature_size));
                        for (out_row, in_row) in rows {
                            for ((o, &v), &m) in out_row.iter_mut().zip(in_row).zip(mean_s.iter()) {
                                *o = v - m;
                            }
                        }
                    });
                x_centered
            } else {
                input - &batch_mean
            };

            // Per-channel variance of the centered data. The fused fold avoids the
            // squared-diff temp the serial form materializes
            let batch_var = if use_col_fold {
                let s = x_centered
                    .as_slice()
                    .expect("rank >= 2 batch-norm buffers are standard layout");
                par_col_dot(s, s, channels, col_stats_parallel, 1.0) / m_rows as f32
            } else {
                (&x_centered * &x_centered).mean_axis(Axis(0)).unwrap()
            };

            // Centering has to finish before this sweep starts, because the variance is folded
            // from `x_centered`
            let std_dev = (&batch_var + self.epsilon).mapv(|x| x.sqrt());
            let (x_normalized, output) = if total_elements >= batch_norm_parallel_threshold() {
                let mut x_normalized = Tensor::zeros(x_centered.raw_dim());
                let mut output = Tensor::zeros(x_centered.raw_dim());
                {
                    let std_s = std_dev.as_slice().unwrap();
                    let gamma_s = self.gamma.as_slice().unwrap();
                    let beta_s = self.beta.as_slice().unwrap();
                    let feature_size = std_s.len();
                    let chunk = rows_per_block(feature_size) * feature_size;
                    x_normalized
                        .as_slice_mut()
                        .unwrap()
                        .par_chunks_mut(chunk)
                        .zip(output.as_slice_mut().unwrap().par_chunks_mut(chunk))
                        .zip(x_centered.as_slice().unwrap().par_chunks(chunk))
                        .for_each(|((norm_c, out_c), cen_c)| {
                            let rows = norm_c
                                .chunks_exact_mut(feature_size)
                                .zip(out_c.chunks_exact_mut(feature_size))
                                .zip(cen_c.chunks_exact(feature_size));
                            for ((norm_row, out_row), cen_row) in rows {
                                for f in 0..feature_size {
                                    norm_row[f] = cen_row[f] / std_s[f];
                                    out_row[f] = norm_row[f] * gamma_s[f] + beta_s[f];
                                }
                            }
                        });
                }
                (x_normalized, output)
            } else {
                let x_normalized = &x_centered / &std_dev;
                let output = &x_normalized * &self.gamma + &self.beta;
                (x_normalized, output)
            };

            // The pass reads `&self`, so the new values go through the state channel. A second
            // call in the same pass reads back what the first call proposed, not the stale field
            let updated_mean = {
                let current = ctx
                    .state::<Tensor>("moving_mean")
                    .unwrap_or(&self.moving_mean);
                current * self.momentum + &batch_mean * (1.0 - self.momentum)
            };
            let updated_variance = {
                let current = ctx
                    .state::<Tensor>("moving_variance")
                    .unwrap_or(&self.moving_variance);
                current * self.momentum + &batch_var * (1.0 - self.momentum)
            };
            ctx.set_state("moving_mean", updated_mean);
            ctx.set_state("moving_variance", updated_variance);

            // Park what the backward pass needs
            ctx.push_cache(
                "BatchNormalization",
                BatchNormalizationCache {
                    batch_var,
                    x_normalized,
                    x_centered,
                },
            );

            Ok(output)
        } else {
            // The per-channel statistics are `[C]`, and the channel axis is innermost, so
            // ndarray's trailing-axis broadcast lines them up against an input of any rank
            let std_dev = (&self.moving_variance + self.epsilon).mapv(|x| x.sqrt());
            let x_normalized = (input - &self.moving_mean) / &std_dev;
            let output = &x_normalized * &self.gamma + &self.beta;

            Ok(output)
        }
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !ctx.is_training() {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        let cache: BatchNormalizationCache = ctx.pop_cache("BatchNormalization")?;
        let BatchNormalizationCache {
            batch_var,
            x_normalized,
            x_centered,
        } = cache;

        // As in `forward`: only a non-contiguous view needs the copy
        let owned;
        let grad_output = if grad_output.is_standard_layout() {
            grad_output
        } else {
            owned = grad_output.as_standard_layout().into_owned();
            &owned
        };
        let total_elements = grad_output.len();

        let channels = self.gamma.len();
        // `grad_output` is standard layout from above, and `x_normalized`, `x_centered`, and
        // `grad_x_normalized` are all owned arrays, so the rank >= 2 arms can take the slice
        // directly
        let use_col_fold = grad_output.ndim() >= 2;
        let batch_size = if use_col_fold {
            (total_elements / channels.max(1)) as f32
        } else {
            grad_output.shape()[0] as f32
        };
        let col_stats_parallel = total_elements >= col_fold_parallel_min_elems();

        // Fused row-block folds for gamma and beta (no [M, C] product temp), on rayon above
        // the column-stats gate
        let (grad_gamma, grad_beta) = if use_col_fold {
            let g = grad_output
                .as_slice()
                .expect("rank >= 2 batch-norm buffers are standard layout");
            let xn = x_normalized
                .as_slice()
                .expect("rank >= 2 batch-norm buffers are standard layout");
            (
                par_col_dot(g, xn, channels, col_stats_parallel, 1.0),
                par_col_sum(g, channels, col_stats_parallel, 1.0),
            )
        } else {
            (
                (grad_output * &x_normalized).sum_axis(Axis(0)),
                grad_output.sum_axis(Axis(0)),
            )
        };

        // An array the layer does not hold gets no gradient, so the store holds none and no
        // optimizer state is ever keyed on it
        if self.scale {
            ctx.add_grad("gamma", grad_gamma.into_dyn())?;
        }
        if self.center {
            ctx.add_grad("beta", grad_beta.into_dyn())?;
        }

        let grad_x_normalized = if total_elements >= batch_norm_parallel_threshold() {
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());
            let gamma_s = self.gamma.as_slice().unwrap();
            let feature_size = gamma_s.len();
            let chunk = rows_per_block(feature_size) * feature_size;
            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_chunks_mut(chunk)
                .zip(grad_output.as_slice().unwrap().par_chunks(chunk))
                .for_each(|(norm_c, out_c)| {
                    let rows = norm_c
                        .chunks_exact_mut(feature_size)
                        .zip(out_c.chunks_exact(feature_size));
                    for (norm_row, out_row) in rows {
                        for ((n, &g), &gam) in norm_row.iter_mut().zip(out_row).zip(gamma_s.iter())
                        {
                            *n = g * gam;
                        }
                    }
                });
            grad_x_norm
        } else {
            grad_output * &self.gamma
        };

        let std_dev = (&batch_var + self.epsilon).mapv(|x| x.sqrt());
        let inv_std = std_dev.mapv(|x| 1.0 / x);

        // The -0.5 / -1.0 scales are applied per term inside the folds, matching the serial
        // elementwise forms
        let grad_var_sum = if use_col_fold {
            let g = grad_x_normalized
                .as_slice()
                .expect("rank >= 2 batch-norm buffers are standard layout");
            let xc = x_centered
                .as_slice()
                .expect("rank >= 2 batch-norm buffers are standard layout");
            par_col_dot(g, xc, channels, col_stats_parallel, -0.5)
        } else {
            (&grad_x_normalized * &x_centered * -0.5).sum_axis(Axis(0))
        };
        let grad_var = grad_var_sum * &inv_std * &inv_std * &inv_std;

        let grad_mean_1_sum = if use_col_fold {
            let g = grad_x_normalized
                .as_slice()
                .expect("rank >= 2 batch-norm buffers are standard layout");
            par_col_sum(g, channels, col_stats_parallel, -1.0)
        } else {
            (&grad_x_normalized * -1.0).sum_axis(Axis(0))
        };
        let grad_mean_1 = grad_mean_1_sum * &inv_std;
        let x_centered_col_sum = if use_col_fold {
            let xc = x_centered
                .as_slice()
                .expect("rank >= 2 batch-norm buffers are standard layout");
            par_col_sum(xc, channels, col_stats_parallel, 1.0)
        } else {
            x_centered.sum_axis(Axis(0))
        };
        let grad_mean_2 = &grad_var * (x_centered_col_sum * -2.0 / batch_size);
        let grad_mean = grad_mean_1 + grad_mean_2;

        let grad_input = if total_elements >= batch_norm_parallel_threshold() {
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());
            let inv_std_s = inv_std.as_slice().unwrap();
            let grad_var_s = grad_var.as_slice().unwrap();
            let grad_mean_s = grad_mean.as_slice().unwrap();
            let feature_size = inv_std_s.len();
            let chunk = rows_per_block(feature_size) * feature_size;
            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_chunks_mut(chunk)
                .zip(grad_x_normalized.as_slice().unwrap().par_chunks(chunk))
                .zip(x_centered.as_slice().unwrap().par_chunks(chunk))
                .for_each(|((inp_c, norm_c), cen_c)| {
                    let rows = inp_c
                        .chunks_exact_mut(feature_size)
                        .zip(norm_c.chunks_exact(feature_size))
                        .zip(cen_c.chunks_exact(feature_size));
                    for ((inp_row, norm_row), cen_row) in rows {
                        for f in 0..feature_size {
                            // The 2nd term keeps the serial arm's association,
                            // `grad_var * ((x_centered * 2) / batch_size)`. Multiplying the
                            // product first rounds somewhere else and changes the last bit
                            inp_row[f] = norm_row[f] * inv_std_s[f]
                                + grad_var_s[f] * (cen_row[f] * 2.0 / batch_size)
                                + grad_mean_s[f] / batch_size;
                        }
                    }
                });
            grad_inp
        } else {
            &grad_x_normalized * &inv_std
                + &grad_var * (&x_centered * 2.0 / batch_size)
                + &grad_mean / batch_size
        };

        Ok(grad_input)
    }

    normalization_layer_output_shape_function!("BatchNormalization");
}

/// Unit tests for the batch-normalization layer and its column-fold kernels
#[cfg(test)]
mod tests {
    use super::super::folds::rows_per_block;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, IxDyn};

    fn test_matrix(m: usize, c: usize, salt: f32) -> Array2<f32> {
        Array2::from_shape_fn((m, c), |(i, j)| ((i * 31 + j * 17) as f32 * salt).sin())
    }

    /// The row-block fold must match a serial fold over the same blocks,
    /// including shapes where the block size does not divide the row count
    #[test]
    fn par_col_folds_match_serial_blocked_reference() {
        for &(m, c) in &[(517usize, 129usize), (4096, 64), (33, 3), (16384, 16)] {
            let a = test_matrix(m, c, 0.731);
            let b = test_matrix(m, c, 0.377);
            let block_rows = rows_per_block(c);

            for &scale in &[1.0f32, -0.5, -1.0] {
                // Hand-rolled serial reference with the same row-block grouping
                let mut ref_sum = vec![0.0f32; c];
                let mut ref_dot = vec![0.0f32; c];
                for block_start in (0..m).step_by(block_rows) {
                    let block_end = (block_start + block_rows).min(m);
                    let mut part_sum = vec![0.0f32; c];
                    let mut part_dot = vec![0.0f32; c];
                    for r in block_start..block_end {
                        for j in 0..c {
                            part_sum[j] += a[(r, j)] * scale;
                            part_dot[j] += a[(r, j)] * b[(r, j)] * scale;
                        }
                    }
                    for j in 0..c {
                        ref_sum[j] += part_sum[j];
                        ref_dot[j] += part_dot[j];
                    }
                }

                // Both flag values must match the reference: the flag is a pure
                // performance hint
                for parallel in [false, true] {
                    let col_sum = par_col_sum(a.as_slice().unwrap(), c, parallel, scale);
                    let col_dot = par_col_dot(
                        a.as_slice().unwrap(),
                        b.as_slice().unwrap(),
                        c,
                        parallel,
                        scale,
                    );
                    for j in 0..c {
                        assert_eq!(
                            col_sum[j], ref_sum[j],
                            "par_col_sum mismatch at [{m}x{c}] col {j} scale {scale} \
                             (parallel={parallel})"
                        );
                        assert_eq!(
                            col_dot[j], ref_dot[j],
                            "par_col_dot mismatch at [{m}x{c}] col {j} scale {scale} \
                             (parallel={parallel})"
                        );
                    }
                }
            }
        }
    }

    /// On integer-valued data, every per-channel sum is exact in f32. The row-block fold must
    /// therefore agree with ndarray's serial sum_axis exactly, regardless of grouping. This pins
    /// the fold against the serial path it replaces above the gate
    #[test]
    fn par_col_folds_exact_on_integer_data() {
        let (m, c) = (4096usize, 64usize);
        let a = Array2::from_shape_fn((m, c), |(i, j)| ((i * 7 + j * 13) % 9) as f32);
        let b = Array2::from_shape_fn((m, c), |(i, j)| ((i * 5 + j * 3) % 7) as f32);

        let serial_sum = a.sum_axis(Axis(0));
        let serial_dot = (&a * &b).sum_axis(Axis(0));
        for parallel in [false, true] {
            let col_sum = par_col_sum(a.as_slice().unwrap(), c, parallel, 1.0);
            assert_eq!(
                col_sum.as_slice().unwrap(),
                serial_sum.as_slice().unwrap(),
                "integer-data column sums must be exact and grouping-independent \
                 (parallel={parallel})"
            );

            let col_dot = par_col_dot(
                a.as_slice().unwrap(),
                b.as_slice().unwrap(),
                c,
                parallel,
                1.0,
            );
            assert_eq!(
                col_dot.as_slice().unwrap(),
                serial_dot.as_slice().unwrap(),
                "integer-data column dots must be exact and grouping-independent \
                 (parallel={parallel})"
            );
        }
    }

    /// A rank-4 input normalizes per channel over batch and every spatial position
    ///
    /// Channel 0 holds 1..4 and channel 1 holds 5..8 across the 4 spatial positions, so each
    /// channel has mean `2.5`/`6.5` and variance `1.25`. Deriving those by hand is what makes
    /// this a layout test. If the channel axis were read anywhere but last, the 2 channels
    /// would mix, and neither number would come out
    #[test]
    fn spatial_forward_normalizes_per_channel_hand_derived() {
        let mut layer = BatchNormalization::new(0.9, 1e-5).unwrap();
        // [1, 2, 2, 2] channels-last: each position holds [channel0, channel1]
        let x = Tensor::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
        )
        .unwrap();

        let mut ctx = Ctx::training();
        let out = layer.forward_mut(&x, &mut ctx).unwrap();
        assert_eq!(out.shape(), &[1, 2, 2, 2]);

        // Both channels share variance 1.25, so both use the same inverse standard deviation
        let inv = 1.0 / (1.25f32 + 1e-5).sqrt();
        let expected = [
            -1.5 * inv,
            -1.5 * inv,
            -0.5 * inv,
            -0.5 * inv,
            0.5 * inv,
            0.5 * inv,
            1.5 * inv,
            1.5 * inv,
        ];
        for (got, want) in out.iter().zip(expected) {
            assert_abs_diff_eq!(*got, want, epsilon = 1e-6);
        }
    }

    /// A rank-4 pass and the equivalent rank-2 pass agree bit for bit
    ///
    /// Under the channels-last layout `[B, H, W, C]` already *is* the `[B*H*W, C]` matrix the
    /// per-channel folds read. Collapsing the leading axes must therefore change nothing at all.
    /// This pins that the collapse is a reinterpretation and not a reduction that lost or
    /// reordered anything on the way
    #[test]
    fn spatial_pass_matches_the_equivalent_two_d_pass_bitwise() {
        let (b, h, w, c) = (2usize, 3usize, 4usize, 5usize);
        let flat: Vec<f32> = (0..b * h * w * c)
            .map(|i| (i % 17) as f32 * 0.25 - 2.0)
            .collect();

        let mut spatial = BatchNormalization::new(0.9, 1e-5).unwrap();
        let x4 = Tensor::from_shape_vec(IxDyn(&[b, h, w, c]), flat.clone()).unwrap();
        let mut spatial_ctx = Ctx::training();
        let out4 = spatial.forward_mut(&x4, &mut spatial_ctx).unwrap();
        // The running statistics live in the context until the layer takes them back
        spatial.apply_state(&mut spatial_ctx.state_slot(0));

        let mut folded = BatchNormalization::new(0.9, 1e-5).unwrap();
        let x2 = Tensor::from_shape_vec(IxDyn(&[b * h * w, c]), flat).unwrap();
        let mut folded_ctx = Ctx::training();
        let out2 = folded.forward_mut(&x2, &mut folded_ctx).unwrap();
        folded.apply_state(&mut folded_ctx.state_slot(0));

        assert_eq!(
            out4.iter().copied().collect::<Vec<f32>>(),
            out2.iter().copied().collect::<Vec<f32>>()
        );
        assert_eq!(spatial.moving_mean, folded.moving_mean);
        assert_eq!(spatial.moving_variance, folded.moving_variance);
    }
}
