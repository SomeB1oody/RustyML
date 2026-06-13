//! Batch normalization layer that normalizes each mini-batch per channel: over the batch axis for
//! 2-D inputs, and over the batch + spatial axes for rank > 2 (convolutional) inputs

use crate::error::Error;
use crate::math::reduction::DET_REDUCE_BLOCK;
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
use ndarray::{Array1, Axis, IxDyn};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSlice;

/// Total-element count above which forward/backward switch from sequential to parallel.
///
/// Mapped by analogy from the calibrated multi-stream (adam-like) elementwise class - the
/// centering/variance/normalize passes stream several arrays like a fused optimizer step does
/// (crossover bracket 256K-1M elements) - rather than measured directly on this layer.
/// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see benches/RESULTS.md
const BATCH_NORM_PARALLEL_THRESHOLD: usize = 262_144;

/// Element count (`M x C`) above which the per-channel statistics reductions (mean, variance,
/// and the backward sums) run as row-block deterministic folds.
///
/// Measured on the row-block fold itself (AMD Ryzen 9 9950X, 16C/32T, 32 rayon threads,
/// 2026-06-12; see benches/RESULTS.md "BatchNorm column stats, row-block fold"): crossover
/// bracket 64K-256K elements, 2.8-4.5x at 1-4M (C=64), 12x for narrow C. A channel-chunked
/// alternative that would have preserved the serial accumulation order bitwise measured
/// 0.3-0.9x everywhere and was rejected (negative-result section in the same report)
const BN_COL_STATS_PARALLEL_MIN_ELEMS: usize = 262_144;

/// Rows per block for the column-stats folds: whole rows, sized so one block holds about
/// [`DET_REDUCE_BLOCK`] elements. A function of the input shape only, so the (deterministic)
/// fold grouping never depends on scheduling
fn rows_per_block(c: usize) -> usize {
    (DET_REDUCE_BLOCK / c).max(1)
}

/// Folds one chunk of whole rows into a local per-channel accumulator (the serial kernel both
/// paths of the column folds share)
fn col_sum_chunk(chunk: &[f32], c: usize, scale: f32) -> Vec<f32> {
    let mut acc = vec![0.0f32; c];
    for row in chunk.chunks_exact(c) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v * scale;
        }
    }
    acc
}

/// The product-fold twin of [`col_sum_chunk`]
fn col_dot_chunk(chunk_a: &[f32], chunk_b: &[f32], c: usize, scale: f32) -> Vec<f32> {
    let mut acc = vec![0.0f32; c];
    for (row_a, row_b) in chunk_a.chunks_exact(c).zip(chunk_b.chunks_exact(c)) {
        for ((s, &va), &vb) in acc.iter_mut().zip(row_a).zip(row_b) {
            *s += va * vb * scale;
        }
    }
    acc
}

/// Merges per-block partial column sums in block order
fn merge_col_parts(parts: Vec<Vec<f32>>, c: usize) -> Tensor {
    let mut out = vec![0.0f32; c];
    for part in parts {
        for (o, p) in out.iter_mut().zip(part) {
            *o += p;
        }
    }
    Array1::from_vec(out).into_dyn()
}

/// Per-channel sums of scaled terms over a standard-layout `[M, C]` slice:
/// `out[j] = sum_r x[r, j] * scale`, computed as a row-block deterministic fold. Each block
/// streams whole rows into a local `[C]` accumulator and block partials merge in block order;
/// the `parallel` flag only decides whether the blocks run on rayon, never the result bits.
/// `scale` is applied per term, matching the serial `(x * scale).sum_axis(Axis(0))` form
fn par_col_sum(x: &[f32], c: usize, parallel: bool, scale: f32) -> Tensor {
    let block = rows_per_block(c) * c;
    let parts: Vec<Vec<f32>> = if parallel {
        x.par_chunks(block)
            .map(|chunk| col_sum_chunk(chunk, c, scale))
            .collect()
    } else {
        x.chunks(block)
            .map(|chunk| col_sum_chunk(chunk, c, scale))
            .collect()
    };
    merge_col_parts(parts, c)
}

/// Per-channel sums of scaled products over two standard-layout `[M, C]` slices:
/// `out[j] = sum_r a[r, j] * b[r, j] * scale`, as the same row-block deterministic fold as
/// [`par_col_sum`] (same flag semantics). Fusing the product into the fold avoids
/// materializing the `[M, C]` temp the serial `(a * b * scale).sum_axis(Axis(0))` form
/// requires
fn par_col_dot(a: &[f32], b: &[f32], c: usize, parallel: bool, scale: f32) -> Tensor {
    let block = rows_per_block(c) * c;
    let parts: Vec<Vec<f32>> = if parallel {
        a.par_chunks(block)
            .zip(b.par_chunks(block))
            .map(|(ca, cb)| col_dot_chunk(ca, cb, c, scale))
            .collect()
    } else {
        a.chunks(block)
            .zip(b.chunks(block))
            .map(|(ca, cb)| col_dot_chunk(ca, cb, c, scale))
            .collect()
    };
    merge_col_parts(parts, c)
}

/// Folds a `[batch, channels, *spatial]` tensor into `[M, channels]` (`M` = product of every axis
/// except the channel axis 1) by moving the channel axis last and flattening
///
/// This is what makes batch norm *spatial* for rank > 2 inputs: every `(batch, *spatial)` position
/// becomes a sample for its channel, so the per-channel statistics reduce over all of them (axis 0
/// of the folded view), matching Keras/PyTorch. A 2-D (or lower) input is already `[N, C]` and is
/// returned unchanged
fn fold_to_2d(t: &Tensor) -> Tensor {
    if t.ndim() <= 2 {
        return t.to_owned();
    }
    let channels = t.shape()[1];
    let r = t.ndim();
    // Move axis 1 (channels) to the end: [0, 2, 3, ..., r-1, 1]
    let mut perm: Vec<usize> = (0..r).filter(|&a| a != 1).collect();
    perm.push(1);
    let m = t.len() / channels;
    t.view()
        .permuted_axes(perm)
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order(IxDyn(&[m, channels]))
        .expect("fold to [M, channels] preserves element count")
}

/// Inverse of [`fold_to_2d`]: reshapes a `[M, channels]` tensor back to `orig_shape`
/// `[batch, channels, *spatial]`. A 2-D (or lower) `orig_shape` is returned unchanged
fn unfold_from_2d(t2: Tensor, orig_shape: &[usize]) -> Tensor {
    if orig_shape.len() <= 2 {
        return t2;
    }
    let r = orig_shape.len();
    let channels = orig_shape[1];
    // Channel-last shape: the non-channel axes (in order), then channels
    let mut cl_shape: Vec<usize> = (0..r).filter(|&a| a != 1).map(|a| orig_shape[a]).collect();
    cl_shape.push(channels);
    let cl = t2
        .into_shape_with_order(IxDyn(&cl_shape))
        .expect("unfold reshape preserves element count");
    // Inverse of the fold permutation [0, 2, .., r-1, 1]
    let mut inv: Vec<usize> = vec![0, r - 1];
    inv.extend(1..(r - 1));
    cl.permuted_axes(inv).as_standard_layout().to_owned()
}

/// Batch Normalization layer for neural networks
///
/// Normalizes each mini-batch to keep activations centered and scaled, improving
/// training stability and speed
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
    ///   **channel/feature** as dimension 1. The trainable `gamma`/`beta` (and the running
    ///   mean/variance) are per-channel, length `input_shape[1]`. For a 2-D `[batch, features]`
    ///   input this is standard per-feature BN; for a rank > 2 `[batch, channels, *spatial]` input
    ///   the statistics reduce over batch **and** all spatial positions (true spatial BN, matching
    ///   Keras/PyTorch), so there is one mean/variance/scale/shift per channel. A 1-D `input_shape`
    ///   (e.g. `vec![4]`) has no channel axis and yields scalar (length-1) parameters broadcast over
    ///   the whole input; pass `vec![batch, 4]` if you mean "4 features"
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

        // Parameters are per-channel
        let param_shape = if input_shape.len() > 1 {
            vec![input_shape[1]]
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

        // Fold to [M, channels] so the rest of the routine reduces per-channel over batch + spatial
        let orig_shape = input.shape().to_vec();
        let folded = fold_to_2d(input);
        let input = &folded;

        if self.training {
            let total_elements = input.len();
            let m_rows = input.shape()[0];
            // The folded input is [M, C] for everything except the 1-D scalar-parameter
            // branch, which keeps the serial ndarray path (and its 0-d statistics shapes)
            let use_col_fold = input.ndim() == 2;
            let col_stats_parallel = total_elements >= BN_COL_STATS_PARALLEL_MIN_ELEMS;
            let channels = if use_col_fold { input.shape()[1] } else { 1 };

            // Mean across the batch dimension (axis 0): a row-block deterministic fold, on
            // rayon above the column-stats gate
            let batch_mean = match input.as_slice() {
                Some(s) if use_col_fold => {
                    par_col_sum(s, channels, col_stats_parallel, 1.0) / m_rows as f32
                }
                _ => input.mean_axis(Axis(0)).unwrap(),
            };

            // Center the data
            let x_centered = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
                let mut x_centered = Tensor::zeros(input.raw_dim());
                x_centered
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .zip(input.as_slice().unwrap().par_iter())
                    .enumerate()
                    .for_each(|(i, (centered, &val))| {
                        let feature_size = batch_mean.len();
                        let feature_idx = i % feature_size;
                        let mean_val = batch_mean.as_slice().unwrap()[feature_idx];
                        *centered = val - mean_val;
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

            // Normalize
            let std_dev = (&batch_var + self.epsilon).mapv(|x| x.sqrt());
            let x_normalized = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
                // Parallel normalization
                let mut x_normalized = Tensor::zeros(x_centered.raw_dim());

                x_normalized
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .zip(x_centered.as_slice().unwrap().par_iter())
                    .enumerate()
                    .for_each(|(i, (norm, &centered))| {
                        let feature_size = std_dev.len();
                        let feature_idx = i % feature_size;
                        let std_val = std_dev.as_slice().unwrap()[feature_idx];
                        *norm = centered / std_val;
                    });

                x_normalized
            } else {
                // Sequential normalization
                &x_centered / &std_dev
            };

            // Scale and shift
            let output = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
                // Parallel scale and shift
                let mut output = Tensor::zeros(x_normalized.raw_dim());

                output
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .zip(x_normalized.as_slice().unwrap().par_iter())
                    .enumerate()
                    .for_each(|(i, (out, &norm))| {
                        let feature_size = self.gamma.len();
                        let feature_idx = i % feature_size;
                        let gamma_val = self.gamma.as_slice().unwrap()[feature_idx];
                        let beta_val = self.beta.as_slice().unwrap()[feature_idx];
                        *out = norm * gamma_val + beta_val;
                    });

                output
            } else {
                // Sequential scale and shift
                &x_normalized * &self.gamma + &self.beta
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

            Ok(unfold_from_2d(output, &orig_shape))
        } else {
            // Inference mode: use running statistics
            let std_dev = (&self.running_var + self.epsilon).mapv(|x| x.sqrt());
            let x_normalized = (input - &self.running_mean) / &std_dev;
            let output = &x_normalized * &self.gamma + &self.beta;

            Ok(unfold_from_2d(output, &orig_shape))
        }
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Fold to [M, channels], normalize per channel with the running stats, then unfold
        let orig_shape = input.shape().to_vec();
        let input = fold_to_2d(input);

        let std_dev = (&self.running_var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = (&input - &self.running_mean) / &std_dev;
        let output = &x_normalized * &self.gamma + &self.beta;

        Ok(unfold_from_2d(output, &orig_shape))
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        // Fold to [M, channels] to match the folded forward caches
        let orig_shape = grad_output.shape().to_vec();
        let folded = fold_to_2d(grad_output);
        let grad_output = &folded;

        let batch_size = grad_output.shape()[0] as f32;
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
        // 1-D (scalar-parameter) inputs keep the serial ndarray path and its 0-d statistics
        // shapes
        let use_col_fold = grad_output.ndim() == 2;
        let col_stats_parallel = total_elements >= BN_COL_STATS_PARALLEL_MIN_ELEMS;

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
        let grad_x_normalized = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());
            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    let feature_size = self.gamma.len();
                    let feature_idx = i % feature_size;
                    let gamma_val = self.gamma.as_slice().unwrap()[feature_idx];
                    *g_norm = g_out * gamma_val;
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
        let grad_input = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());
            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_x_normalized.as_slice().unwrap().par_iter())
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((g_inp, &g_norm), &x_cent))| {
                    let feature_size = inv_std.len();
                    let feature_idx = i % feature_size;
                    let inv_std_val = inv_std.as_slice().unwrap()[feature_idx];
                    let grad_var_val = grad_var.as_slice().unwrap()[feature_idx];
                    let grad_mean_val = grad_mean.as_slice().unwrap()[feature_idx];

                    *g_inp = g_norm * inv_std_val
                        + grad_var_val * x_cent * 2.0 / batch_size
                        + grad_mean_val / batch_size;
                });
            grad_inp
        } else {
            // Sequential computation
            &grad_x_normalized * &inv_std
                + &grad_var * (x_centered * 2.0 / batch_size)
                + &grad_mean / batch_size
        };

        Ok(unfold_from_2d(grad_input, &orig_shape))
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
        LayerWeight::BatchNormalization(BatchNormalizationLayerWeight {
            gamma: &self.gamma,
            beta: &self.beta,
            running_mean: &self.running_mean,
            running_var: &self.running_var,
        })
    }

    mode_dependent_layer_trait!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn test_matrix(m: usize, c: usize, salt: f32) -> Array2<f32> {
        Array2::from_shape_fn((m, c), |(i, j)| ((i * 31 + j * 17) as f32 * salt).sin())
    }

    /// The row-block fold must be bitwise identical to a serial fold over the same blocks,
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
    /// must agree with ndarray's serial sum_axis bit for bit regardless of grouping - this
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
}
