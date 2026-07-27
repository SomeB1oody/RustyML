//! Batch normalization layer that normalizes each mini-batch per channel: over the batch axis for
//! 2-D inputs, and over the batch + spatial axes for rank > 2 (convolutional) inputs
//!
//! Both are the same pass. Under the crate's channels-last layout the channel axis is innermost, so
//! a `[batch, spatial..., channels]` buffer already *is* the `[M, C]` matrix the per-channel folds
//! read, with `M = batch * spatial`. Collapsing the leading axes is a reinterpretation of the same
//! bytes, not a reshape or a transpose, which is why one code path serves every rank >= 2

use super::folds::{par_col_dot, par_col_sum};
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::{BatchNormalizationLayerWeight, LayerWeight};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape;
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_input_shape, validate_input_shape_not_empty, validate_momentum,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::Axis;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::borrow::Cow;

tunable_gate! {
    /// Total-element count above which forward/backward switch from sequential to parallel
    ///
    /// The centering/variance/normalize passes stream several arrays like a fused optimizer step
    /// does, so the threshold is mapped from the multi-stream elementwise class (crossover bracket
    /// 256K-1M elements) rather than measured directly on this layer
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) BATCH_NORM_PARALLEL_THRESHOLD => batch_norm_parallel_threshold / set_batch_norm_parallel_threshold = 262_144
}

tunable_gate! {
    /// Element count (`M x C`) above which the per-channel statistics reductions of **2-D** inputs
    /// (mean, variance, and the backward sums) run as row-block deterministic folds
    ///
    /// Crossover bracket 64K-256K elements, 2.8-4.5x at 1-4M (C=64), 12x for narrow C. A
    /// channel-chunked alternative that would have preserved the serial accumulation order
    /// measured 0.3-0.9x everywhere and was rejected
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) BN_COL_STATS_PARALLEL_MIN_ELEMS => bn_col_stats_parallel_min_elems / set_bn_col_stats_parallel_min_elems = 262_144
}

tunable_gate! {
    /// Element count above which the per-channel statistics reductions of **rank >= 3** inputs
    /// (the plane folds over the native `[batch, *spatial, channels]` layout) run on rayon
    ///
    /// Crossover bracket 64K-256K elements (0.36x at 64K, 1.37x at 256K), 2.8-3.8x at 1M, 11.7x at
    /// the conv-scale 8.4M
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) BN_PLANE_STATS_PARALLEL_MIN_ELEMS => bn_plane_stats_parallel_min_elems / set_bn_plane_stats_parallel_min_elems = 262_144
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
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a BatchNormalization layer
/// let mut bn = BatchNormalization::new(vec![32, 128], 0.99, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, normalizes the input
/// let output = bn.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct BatchNormalization {
    /// Small constant for numerical stability in normalization
    epsilon: f32,
    /// Momentum for the moving average of mean and variance
    momentum: f32,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Scale parameter (trainable)
    gamma: Tensor,
    /// Shift parameter (trainable)
    beta: Tensor,
    /// Running mean for inference
    running_mean: Tensor,
    /// Running variance for inference
    running_var: Tensor,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Mean computed during forward pass (used in backward pass)
    batch_mean: Option<Tensor>,
    /// Variance computed during forward pass (used in backward pass)
    batch_var: Option<Tensor>,
    /// Normalized input (used in backward pass)
    x_normalized: Option<Tensor>,
    /// Centered input (used in backward pass)
    x_centered: Option<Tensor>,
    /// Gradient for gamma parameter
    grad_gamma: Option<Tensor>,
    /// Gradient for beta parameter
    grad_beta: Option<Tensor>,
}

impl BatchNormalization {
    /// Creates a new BatchNormalization layer
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor, with the **batch** as dimension 0 and the
    ///   **channel/feature** as the **last** dimension. The trainable `gamma`/`beta` (and the
    ///   running mean/variance) are per-channel, length `input_shape.last()`. For a 2-D
    ///   `[batch, features]` input this is standard per-feature BN; for a rank > 2
    ///   `[batch, *spatial, channels]` input the statistics reduce over batch **and** all spatial
    ///   positions (spatial BN, matching Keras), so there is one mean/variance/scale/shift per
    ///   channel. A 1-D `input_shape` (e.g. `vec![4]`) has no channel axis and yields scalar
    ///   (length-1) parameters broadcast over the whole input; pass `vec![batch, 4]` to mean
    ///   "4 features"
    /// - `momentum` - Momentum for the moving average of mean and variance (typically 0.9 or 0.99)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New BatchNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `input_shape` is empty
    /// - `Error::InvalidParameter` - If `momentum` is not between 0.0 and 1.0
    /// - `Error::InvalidParameter` - If `epsilon` is not positive
    pub fn new(input_shape: Vec<usize>, momentum: f32, epsilon: f32) -> Result<Self, Error> {
        validate_input_shape_not_empty(&input_shape)?;
        validate_momentum(momentum)?;
        validate_epsilon(epsilon)?;

        // Parameters are per-channel, and the channel axis is the trailing one
        let param_shape = if input_shape.len() > 1 {
            vec![input_shape[input_shape.len() - 1]]
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(BatchNormalization {
            epsilon,
            momentum,
            input_shape,
            gamma: Tensor::ones(param_shape_ndarray),
            beta: Tensor::zeros(param_shape_ndarray),
            running_mean: Tensor::zeros(param_shape_ndarray),
            running_var: Tensor::ones(param_shape_ndarray),
            training: true,
            batch_mean: None,
            batch_var: None,
            x_normalized: None,
            x_centered: None,
            grad_gamma: None,
            grad_beta: None,
        })
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the BatchNormalization layer
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable)
    /// - `beta` - Shift parameter (trainable)
    /// - `running_mean` - Running mean for inference
    /// - `running_var` - Running variance for inference
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   layer's expected shape
    pub fn set_weights(
        &mut self,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
    ) -> Result<(), Error> {
        validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        validate_weight_shape(
            "running_mean",
            self.running_mean.shape(),
            running_mean.shape(),
        )?;
        validate_weight_shape("running_var", self.running_var.shape(), running_var.shape())?;
        self.gamma = gamma;
        self.beta = beta;
        self.running_mean = running_mean;
        self.running_var = running_var;
        Ok(())
    }
}

impl Layer for BatchNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        // The parallel passes below need a contiguous slice. A standard-layout input already is
        // one, so only a non-contiguous view pays for a copy - `as_standard_layout().into_owned()`
        // would copy unconditionally, which on a conv-scale tensor costs more than the pass itself
        let owned;
        let input = if input.is_standard_layout() {
            input
        } else {
            owned = input.as_standard_layout().into_owned();
            &owned
        };

        if self.training {
            let total_elements = input.len();
            // Under the channels-last layout the channel axis is innermost, so a
            // `[batch, spatial..., channels]` buffer already *is* the `[M, C]` matrix the
            // per-channel folds want, with `M = batch * spatial`. No reshape and no transpose:
            // one path serves every rank >= 2, and the statistics of a rank > 2 input reduce
            // over batch and every spatial position exactly as spatial batch norm requires.
            // A 1-D input has no channel axis (see `new`) and keeps the serial ndarray path
            // with its 0-d statistics shapes
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
            let col_stats_parallel = total_elements >= bn_col_stats_parallel_min_elems();

            // Mean across the batch dimension (axis 0): a row-block deterministic fold, on
            // rayon above the column-stats gate
            let batch_mean = match input.as_slice() {
                Some(s) if use_col_fold => {
                    par_col_sum(s, channels, col_stats_parallel, 1.0) / m_rows as f32
                }
                _ => input.mean_axis(Axis(0)).unwrap(),
            };

            // Center the data
            //
            // The per-channel table is bound to a slice *outside* the loop. Calling
            // `as_slice().unwrap()` inside would re-read a dynamic-dimension array's heap-allocated
            // shape and re-run its standard-layout check on every element, and because the body
            // stores through a `&mut f32` the compiler cannot prove the call loop-invariant and
            // will not hoist it. Measured at roughly 2x on this pass
            let x_centered = if total_elements >= batch_norm_parallel_threshold() {
                let mut x_centered = Tensor::zeros(input.raw_dim());
                let mean_s = batch_mean.as_slice().unwrap();
                let feature_size = mean_s.len();
                x_centered
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .zip(input.as_slice().unwrap().par_iter())
                    .enumerate()
                    .for_each(|(i, (centered, &val))| {
                        *centered = val - mean_s[i % feature_size];
                    });
                x_centered
            } else {
                input - &batch_mean
            };

            // Per-channel variance of the centered data; the fused fold avoids the
            // squared-diff temp the serial form materializes
            let batch_var = match x_centered.as_slice() {
                Some(s) if use_col_fold => {
                    par_col_dot(s, s, channels, col_stats_parallel, 1.0) / m_rows as f32
                }
                _ => (&x_centered * &x_centered).mean_axis(Axis(0)).unwrap(),
            };

            // Normalize, then scale and shift - in one sweep
            //
            // Both outputs are needed (`x_normalized` is cached for the backward pass), but they
            // are the same walk over the same array, so writing them together saves a whole extra
            // read and write of a conv-scale tensor. Centering cannot join them: the variance is
            // folded from `x_centered`, so that pass has to finish first. Element for element this
            // is the same arithmetic in the same order as the two passes it replaces
            let std_dev = (&batch_var + self.epsilon).mapv(|x| x.sqrt());
            let (x_normalized, output) = if total_elements >= batch_norm_parallel_threshold() {
                let mut x_normalized = Tensor::zeros(x_centered.raw_dim());
                let mut output = Tensor::zeros(x_centered.raw_dim());
                {
                    let std_s = std_dev.as_slice().unwrap();
                    let gamma_s = self.gamma.as_slice().unwrap();
                    let beta_s = self.beta.as_slice().unwrap();
                    let feature_size = std_s.len();
                    x_normalized
                        .as_slice_mut()
                        .unwrap()
                        .par_iter_mut()
                        .zip(output.as_slice_mut().unwrap().par_iter_mut())
                        .zip(x_centered.as_slice().unwrap().par_iter())
                        .enumerate()
                        .for_each(|(i, ((norm, out), &centered))| {
                            let f = i % feature_size;
                            *norm = centered / std_s[f];
                            *out = *norm * gamma_s[f] + beta_s[f];
                        });
                }
                (x_normalized, output)
            } else {
                // Sequential normalize, scale and shift
                let x_normalized = &x_centered / &std_dev;
                let output = &x_normalized * &self.gamma + &self.beta;
                (x_normalized, output)
            };

            // Update running statistics
            self.running_mean =
                &self.running_mean * self.momentum + &batch_mean * (1.0 - self.momentum);
            self.running_var =
                &self.running_var * self.momentum + &batch_var * (1.0 - self.momentum);

            // Cache values for backward pass
            self.batch_mean = Some(batch_mean);
            self.batch_var = Some(batch_var);
            self.x_normalized = Some(x_normalized);
            self.x_centered = Some(x_centered);

            Ok(output)
        } else {
            // Inference mode: use running statistics
            let std_dev = (&self.running_var + self.epsilon).mapv(|x| x.sqrt());
            let x_normalized = (input - &self.running_mean) / &std_dev;
            let output = &x_normalized * &self.gamma + &self.beta;

            Ok(output)
        }
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        // The per-channel statistics are `[C]` and the channel axis is innermost, so ndarray's
        // trailing-axis broadcast lines them up against an input of any rank on its own
        let std_dev = (&self.running_var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = (input - &self.running_mean) / &std_dev;
        let output = &x_normalized * &self.gamma + &self.beta;

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        // As in `forward`: only a non-contiguous view needs the copy
        let owned;
        let grad_output = if grad_output.is_standard_layout() {
            grad_output
        } else {
            owned = grad_output.as_standard_layout().into_owned();
            &owned
        };
        let total_elements = grad_output.len();

        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("BatchNormalization"))?;

        let x_centered = self
            .x_centered
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("BatchNormalization"))?;

        let batch_var = self
            .batch_var
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("BatchNormalization"))?;

        let channels = self.gamma.len();
        // As in `forward`: any rank >= 2 gradient is already the `[M, C]` matrix the folds want.
        // `batch_size` is that `M` - batch times spatial - which is exactly the sample count the
        // per-channel statistics were taken over. 1-D (scalar-parameter) inputs keep the serial
        // ndarray path and its 0-d statistics shapes
        let use_col_fold = grad_output.ndim() >= 2;
        let batch_size = if use_col_fold {
            (total_elements / channels.max(1)) as f32
        } else {
            grad_output.shape()[0] as f32
        };
        let col_stats_parallel = total_elements >= bn_col_stats_parallel_min_elems();

        // Compute gradients for gamma and beta: fused row-block folds (no [M, C] product
        // temp), on rayon above the column-stats gate
        let (grad_gamma, grad_beta) = match (grad_output.as_slice(), x_normalized.as_slice()) {
            (Some(g), Some(xn)) if use_col_fold => (
                par_col_dot(g, xn, channels, col_stats_parallel, 1.0),
                par_col_sum(g, channels, col_stats_parallel, 1.0),
            ),
            _ => (
                (grad_output * x_normalized).sum_axis(Axis(0)),
                grad_output.sum_axis(Axis(0)),
            ),
        };

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        // Compute gradient with respect to normalized input
        let grad_x_normalized = if total_elements >= batch_norm_parallel_threshold() {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());
            let gamma_s = self.gamma.as_slice().unwrap();
            let feature_size = gamma_s.len();
            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    *g_norm = g_out * gamma_s[i % feature_size];
                });
            grad_x_norm
        } else {
            // Sequential computation
            grad_output * &self.gamma
        };

        // Compute gradient with respect to variance
        let std_dev = (batch_var + self.epsilon).mapv(|x| x.sqrt());
        let inv_std = std_dev.mapv(|x| 1.0 / x);

        // The -0.5 / -1.0 scales are applied per term inside the folds, matching the serial
        // elementwise forms
        let grad_var_sum = match (grad_x_normalized.as_slice(), x_centered.as_slice()) {
            (Some(g), Some(xc)) if use_col_fold => {
                par_col_dot(g, xc, channels, col_stats_parallel, -0.5)
            }
            _ => (&grad_x_normalized * x_centered * -0.5).sum_axis(Axis(0)),
        };
        let grad_var = grad_var_sum * &inv_std * &inv_std * &inv_std;

        // Compute gradient with respect to mean
        let grad_mean_1_sum = match grad_x_normalized.as_slice() {
            Some(g) if use_col_fold => par_col_sum(g, channels, col_stats_parallel, -1.0),
            _ => (&grad_x_normalized * -1.0).sum_axis(Axis(0)),
        };
        let grad_mean_1 = grad_mean_1_sum * &inv_std;
        let x_centered_col_sum = match x_centered.as_slice() {
            Some(xc) if use_col_fold => par_col_sum(xc, channels, col_stats_parallel, 1.0),
            _ => x_centered.sum_axis(Axis(0)),
        };
        let grad_mean_2 = &grad_var * (x_centered_col_sum * -2.0 / batch_size);
        let grad_mean = grad_mean_1 + grad_mean_2;

        // Compute gradient with respect to input
        let grad_input = if total_elements >= batch_norm_parallel_threshold() {
            // Parallel computation
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());
            // Three per-channel tables, so this pass paid the re-read three times per element
            let inv_std_s = inv_std.as_slice().unwrap();
            let grad_var_s = grad_var.as_slice().unwrap();
            let grad_mean_s = grad_mean.as_slice().unwrap();
            let feature_size = inv_std_s.len();
            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_x_normalized.as_slice().unwrap().par_iter())
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((g_inp, &g_norm), &x_cent))| {
                    let f = i % feature_size;
                    *g_inp = g_norm * inv_std_s[f]
                        + grad_var_s[f] * x_cent * 2.0 / batch_size
                        + grad_mean_s[f] / batch_size;
                });
            grad_inp
        } else {
            // Sequential computation
            &grad_x_normalized * &inv_std
                + &grad_var * (x_centered * 2.0 / batch_size)
                + &grad_mean / batch_size
        };

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "BatchNormalization"
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
        LayerWeight::BatchNormalization(BatchNormalizationLayerWeight {
            gamma: Cow::Borrowed(&self.gamma),
            beta: Cow::Borrowed(&self.beta),
            running_mean: Cow::Borrowed(&self.running_mean),
            running_var: Cow::Borrowed(&self.running_var),
        })
    }

    mode_dependent_layer_trait!();
}

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

    /// On integer-valued data every per-channel sum is exact in f32, so the row-block fold
    /// must agree with ndarray's serial sum_axis exactly regardless of grouping - this
    /// pins the fold against the serial path it replaces above the gate
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
    /// Channel 0 holds 1..4 and channel 1 holds 5..8 across the four spatial positions, so each
    /// channel has mean `2.5`/`6.5` and variance `1.25`. Deriving those by hand is what makes this
    /// a layout test: if the channel axis were read anywhere but last, the two channels would mix
    /// and neither number would come out
    #[test]
    fn spatial_forward_normalizes_per_channel_hand_derived() {
        let mut layer = BatchNormalization::new(vec![1, 2, 2, 2], 0.9, 1e-5).unwrap();
        // [1, 2, 2, 2] channels-last: each position holds [channel0, channel1]
        let x = Tensor::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
        )
        .unwrap();

        let out = layer.forward(&x).unwrap();
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
    /// per-channel folds read, so collapsing the leading axes must change nothing at all. This
    /// pins that the collapse is a reinterpretation and not a reduction that lost or reordered
    /// anything on the way
    #[test]
    fn spatial_pass_matches_the_equivalent_two_d_pass_bitwise() {
        let (b, h, w, c) = (2usize, 3usize, 4usize, 5usize);
        let flat: Vec<f32> = (0..b * h * w * c)
            .map(|i| (i % 17) as f32 * 0.25 - 2.0)
            .collect();

        let mut spatial = BatchNormalization::new(vec![b, h, w, c], 0.9, 1e-5).unwrap();
        let x4 = Tensor::from_shape_vec(IxDyn(&[b, h, w, c]), flat.clone()).unwrap();
        let out4 = spatial.forward(&x4).unwrap();

        let mut folded = BatchNormalization::new(vec![b * h * w, c], 0.9, 1e-5).unwrap();
        let x2 = Tensor::from_shape_vec(IxDyn(&[b * h * w, c]), flat).unwrap();
        let out2 = folded.forward(&x2).unwrap();

        assert_eq!(
            out4.iter().copied().collect::<Vec<f32>>(),
            out2.iter().copied().collect::<Vec<f32>>()
        );
        assert_eq!(spatial.running_mean, folded.running_mean);
        assert_eq!(spatial.running_var, folded.running_var);
    }
}
