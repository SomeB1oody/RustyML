//! Batch normalization layer that normalizes each mini-batch per channel: over the batch axis for
//! 2-D inputs, and over the batch + spatial axes for rank > 2 (convolutional) inputs

use super::folds::{par_col_dot, par_col_sum, par_plane_dot, par_plane_sum};
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
use ndarray::{Axis, IxDyn};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
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
    /// (the plane folds over the native `[batch, channels, *spatial]` layout) run on rayon
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
    ///   **channel/feature** as dimension 1. The trainable `gamma`/`beta` (and the running
    ///   mean/variance) are per-channel, length `input_shape[1]`. For a 2-D `[batch, features]`
    ///   input this is standard per-feature BN; for a rank > 2 `[batch, channels, *spatial]` input
    ///   the statistics reduce over batch **and** all spatial positions (spatial BN, matching
    ///   Keras/PyTorch), so there is one mean/variance/scale/shift per channel. A 1-D `input_shape`
    ///   (e.g. `vec![4]`) has no channel axis and yields scalar (length-1) parameters broadcast over
    ///   the whole input; pass `vec![batch, 4]` to mean "4 features"
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

    /// Training/inference forward for rank >= 3 inputs on the native `[batch, channels, *spatial]`
    /// layout: the per-channel statistics are plane folds and every elementwise pass streams
    /// contiguous `[P]` planes, so no channel-last transpose copy is ever made. Backward caches
    /// (`x_centered`, `x_normalized`) keep the input's original shape
    fn forward_spatial(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let shape = input.shape().to_vec();
        let c = shape[1];
        let p: usize = shape[2..].iter().product();
        let total = input.len();
        if total == 0 {
            // Empty input: nothing to normalize, no caches to write
            return Ok(Tensor::zeros(IxDyn(&shape)));
        }

        // Plane streaming needs contiguous data; standardize a non-contiguous input once
        let std_input;
        let x: &[f32] = match input.as_slice() {
            Some(s) => s,
            None => {
                std_input = input.as_standard_layout().into_owned();
                std_input.as_slice().unwrap()
            }
        };

        // Samples per channel: batch x spatial, the row count of the folded [M, C] view
        let m = (total / c) as f32;
        let elem_parallel = total >= batch_norm_parallel_threshold();

        if !self.training {
            return Ok(self.normalize_with_running_stats(x, &shape, c, p, elem_parallel));
        }

        let stats_parallel = total >= bn_plane_stats_parallel_min_elems();
        let batch_mean = par_plane_sum(x, c, p, stats_parallel, 1.0) / m;

        // Center per plane
        let mut x_centered = Tensor::zeros(IxDyn(&shape));
        {
            let mean_s = batch_mean.as_slice().unwrap();
            let xc = x_centered.as_slice_mut().unwrap();
            let center = |(i, (dst, src)): (usize, (&mut [f32], &[f32]))| {
                let mean_val = mean_s[i % c];
                for (o, &v) in dst.iter_mut().zip(src) {
                    *o = v - mean_val;
                }
            };
            if elem_parallel {
                xc.par_chunks_mut(p)
                    .zip(x.par_chunks(p))
                    .enumerate()
                    .for_each(center);
            } else {
                xc.chunks_mut(p)
                    .zip(x.chunks(p))
                    .enumerate()
                    .for_each(center);
            }
        }

        // Per-channel variance of the centered data; the fused fold avoids the squared-diff temp
        let xc_s = x_centered.as_slice().unwrap();
        let batch_var = par_plane_dot(xc_s, xc_s, c, p, stats_parallel, 1.0) / m;
        let std_dev = (&batch_var + self.epsilon).mapv(|v| v.sqrt());

        // Normalize and scale-shift in one streaming pass that writes both the cached
        // x_normalized and the output
        let mut x_normalized = Tensor::zeros(IxDyn(&shape));
        let mut output = Tensor::zeros(IxDyn(&shape));
        {
            let std_s = std_dev.as_slice().unwrap();
            let gamma_s = self.gamma.as_slice().unwrap();
            let beta_s = self.beta.as_slice().unwrap();
            let xn = x_normalized.as_slice_mut().unwrap();
            let out = output.as_slice_mut().unwrap();
            type PlanesOut2In1<'a> = (usize, ((&'a mut [f32], &'a mut [f32]), &'a [f32]));
            let normalize = |(i, ((xn_pl, out_pl), xc_pl)): PlanesOut2In1| {
                let ch = i % c;
                let (std_val, gamma_val, beta_val) = (std_s[ch], gamma_s[ch], beta_s[ch]);
                for ((n, o), &centered) in xn_pl.iter_mut().zip(out_pl.iter_mut()).zip(xc_pl) {
                    *n = centered / std_val;
                    *o = *n * gamma_val + beta_val;
                }
            };
            if elem_parallel {
                xn.par_chunks_mut(p)
                    .zip(out.par_chunks_mut(p))
                    .zip(xc_s.par_chunks(p))
                    .enumerate()
                    .for_each(normalize);
            } else {
                xn.chunks_mut(p)
                    .zip(out.chunks_mut(p))
                    .zip(xc_s.chunks(p))
                    .enumerate()
                    .for_each(normalize);
            }
        }

        // Update running statistics
        self.running_mean =
            &self.running_mean * self.momentum + &batch_mean * (1.0 - self.momentum);
        self.running_var = &self.running_var * self.momentum + &batch_var * (1.0 - self.momentum);

        // Cache values for backward pass
        self.batch_mean = Some(batch_mean);
        self.batch_var = Some(batch_var);
        self.x_normalized = Some(x_normalized);
        self.x_centered = Some(x_centered);

        Ok(output)
    }

    /// One fused pass `((x - running_mean) / sqrt(running_var + eps)) * gamma + beta` streamed
    /// per `[P]` plane of a `[B, C, P]` slice. The per-element operations match the broadcast
    /// 2-D form exactly, so spatial inference outputs match the folded form; the `parallel`
    /// flag does not change the result
    fn normalize_with_running_stats(
        &self,
        x: &[f32],
        shape: &[usize],
        c: usize,
        p: usize,
        parallel: bool,
    ) -> Tensor {
        let std_dev = (&self.running_var + self.epsilon).mapv(|v| v.sqrt());
        let mut output = Tensor::zeros(IxDyn(shape));
        {
            let std_s = std_dev.as_slice().unwrap();
            let mean_s = self.running_mean.as_slice().unwrap();
            let gamma_s = self.gamma.as_slice().unwrap();
            let beta_s = self.beta.as_slice().unwrap();
            let out = output.as_slice_mut().unwrap();
            let kernel = |(i, (out_pl, x_pl)): (usize, (&mut [f32], &[f32]))| {
                let ch = i % c;
                let (mean_val, std_val) = (mean_s[ch], std_s[ch]);
                let (gamma_val, beta_val) = (gamma_s[ch], beta_s[ch]);
                for (o, &v) in out_pl.iter_mut().zip(x_pl) {
                    *o = ((v - mean_val) / std_val) * gamma_val + beta_val;
                }
            };
            if parallel {
                out.par_chunks_mut(p)
                    .zip(x.par_chunks(p))
                    .enumerate()
                    .for_each(kernel);
            } else {
                out.chunks_mut(p)
                    .zip(x.chunks(p))
                    .enumerate()
                    .for_each(kernel);
            }
        }
        output
    }

    /// Backward for rank >= 3 inputs on the native layout, mirroring [`Self::forward_spatial`]:
    /// the 5 per-channel reductions are plane folds and the 2 elementwise passes stream per
    /// plane, with no transpose copy of `grad_output` or `grad_input`
    fn backward_spatial(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let shape = grad_output.shape().to_vec();
        let c = shape[1];
        let p: usize = shape[2..].iter().product();
        let total = grad_output.len();
        if total == 0 {
            return Ok(Tensor::zeros(IxDyn(&shape)));
        }

        let std_grad;
        let g: &[f32] = match grad_output.as_slice() {
            Some(s) => s,
            None => {
                std_grad = grad_output.as_standard_layout().into_owned();
                std_grad.as_slice().unwrap()
            }
        };

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
        // The caches were built by forward_spatial, so they are standard layout
        let xn_s = x_normalized.as_slice().unwrap();
        let xc_s = x_centered.as_slice().unwrap();

        let m = (total / c) as f32;
        let stats_parallel = total >= bn_plane_stats_parallel_min_elems();
        let elem_parallel = total >= batch_norm_parallel_threshold();

        // Gradients for gamma and beta: fused plane folds (no product temporary)
        self.grad_gamma = Some(par_plane_dot(g, xn_s, c, p, stats_parallel, 1.0));
        self.grad_beta = Some(par_plane_sum(g, c, p, stats_parallel, 1.0));

        // Gradient with respect to the normalized input
        let mut grad_x_normalized = Tensor::zeros(IxDyn(&shape));
        {
            let gamma_s = self.gamma.as_slice().unwrap();
            let gxn = grad_x_normalized.as_slice_mut().unwrap();
            let scale_by_gamma = |(i, (dst, src)): (usize, (&mut [f32], &[f32]))| {
                let gamma_val = gamma_s[i % c];
                for (o, &v) in dst.iter_mut().zip(src) {
                    *o = v * gamma_val;
                }
            };
            if elem_parallel {
                gxn.par_chunks_mut(p)
                    .zip(g.par_chunks(p))
                    .enumerate()
                    .for_each(scale_by_gamma);
            } else {
                gxn.chunks_mut(p)
                    .zip(g.chunks(p))
                    .enumerate()
                    .for_each(scale_by_gamma);
            }
        }
        let gxn_s = grad_x_normalized.as_slice().unwrap();

        let std_dev = (batch_var + self.epsilon).mapv(|v| v.sqrt());
        let inv_std = std_dev.mapv(|v| 1.0 / v);

        // The -0.5 / -1.0 scales are applied per term inside the folds, matching the
        // elementwise forms of the 2-D path
        let grad_var_sum = par_plane_dot(gxn_s, xc_s, c, p, stats_parallel, -0.5);
        let grad_var = grad_var_sum * &inv_std * &inv_std * &inv_std;

        let grad_mean_1 = par_plane_sum(gxn_s, c, p, stats_parallel, -1.0) * &inv_std;
        let x_centered_sum = par_plane_sum(xc_s, c, p, stats_parallel, 1.0);
        let grad_mean_2 = &grad_var * (x_centered_sum * -2.0 / m);
        let grad_mean = grad_mean_1 + grad_mean_2;

        // Gradient with respect to the input, streamed per plane
        let mut grad_input = Tensor::zeros(IxDyn(&shape));
        {
            let inv_std_s = inv_std.as_slice().unwrap();
            let grad_var_s = grad_var.as_slice().unwrap();
            let grad_mean_s = grad_mean.as_slice().unwrap();
            let gi = grad_input.as_slice_mut().unwrap();
            type PlanesOut1In2<'a> = (usize, ((&'a mut [f32], &'a [f32]), &'a [f32]));
            let compose = |(i, ((gi_pl, gxn_pl), xc_pl)): PlanesOut1In2| {
                let ch = i % c;
                let (inv_std_val, grad_var_val) = (inv_std_s[ch], grad_var_s[ch]);
                let grad_mean_term = grad_mean_s[ch] / m;
                for ((o, &g_norm), &centered) in gi_pl.iter_mut().zip(gxn_pl).zip(xc_pl) {
                    *o =
                        g_norm * inv_std_val + grad_var_val * (centered * 2.0 / m) + grad_mean_term;
                }
            };
            if elem_parallel {
                gi.par_chunks_mut(p)
                    .zip(gxn_s.par_chunks(p))
                    .zip(xc_s.par_chunks(p))
                    .enumerate()
                    .for_each(compose);
            } else {
                gi.chunks_mut(p)
                    .zip(gxn_s.chunks(p))
                    .zip(xc_s.chunks(p))
                    .enumerate()
                    .for_each(compose);
            }
        }

        Ok(grad_input)
    }
}

impl Layer for BatchNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Rank >= 3 (spatial) inputs reduce per channel over batch + spatial on the native
        // layout, with no channel-last transpose copies
        if input.ndim() >= 3 {
            return self.forward_spatial(input);
        }

        // 1-D / 2-D path: the input is already [N, C] (or scalar-parameter 1-D); the owned
        // copy keeps the contiguity guarantee the parallel passes below rely on
        let folded = input.to_owned();
        let input = &folded;

        if self.training {
            let total_elements = input.len();
            let m_rows = input.shape()[0];
            // The folded input is [M, C] for everything except the 1-D scalar-parameter
            // branch, which keeps the serial ndarray path (and its 0-d statistics shapes)
            let use_col_fold = input.ndim() == 2;
            let col_stats_parallel = total_elements >= bn_col_stats_parallel_min_elems();
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
            let x_centered = if total_elements >= batch_norm_parallel_threshold() {
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
            let x_normalized = if total_elements >= batch_norm_parallel_threshold() {
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
            let output = if total_elements >= batch_norm_parallel_threshold() {
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

        // Rank >= 3: per-channel running-stat normalization streamed on the native layout
        if input.ndim() >= 3 {
            let shape = input.shape().to_vec();
            let c = shape[1];
            let p: usize = shape[2..].iter().product();
            if input.is_empty() {
                return Ok(Tensor::zeros(IxDyn(&shape)));
            }
            let std_input;
            let x: &[f32] = match input.as_slice() {
                Some(s) => s,
                None => {
                    std_input = input.as_standard_layout().into_owned();
                    std_input.as_slice().unwrap()
                }
            };
            let parallel = input.len() >= batch_norm_parallel_threshold();
            return Ok(self.normalize_with_running_stats(x, &shape, c, p, parallel));
        }

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

        // Rank >= 3 gradients flow through the native-layout path that produced the caches
        if grad_output.ndim() >= 3 {
            return self.backward_spatial(grad_output);
        }

        // 1-D / 2-D path; the owned copy keeps the contiguity guarantee the parallel passes
        // below rely on
        let folded = grad_output.to_owned();
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
        let grad_input = if total_elements >= batch_norm_parallel_threshold() {
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
    use super::super::folds::{plane_range_dot, plane_range_sum, rows_per_block};
    use super::*;
    use crate::math::reduction::DET_REDUCE_BLOCK;
    use ndarray::{Array1, Array2, Array3, Array4};

    fn test_matrix(m: usize, c: usize, salt: f32) -> Array2<f32> {
        Array2::from_shape_fn((m, c), |(i, j)| ((i * 31 + j * 17) as f32 * salt).sin())
    }

    /// Channel-last fold to `[M, C]`, the transpose-copy reference the plane path replaces
    fn fold_ref(t: &Tensor) -> Array2<f32> {
        let c = t.shape()[1];
        let r = t.ndim();
        let mut perm: Vec<usize> = (0..r).filter(|&a| a != 1).collect();
        perm.push(1);
        let m = t.len() / c;
        t.view()
            .permuted_axes(perm)
            .as_standard_layout()
            .to_owned()
            .into_shape_with_order((m, c))
            .unwrap()
    }

    /// Inverse of [`fold_ref`]
    fn unfold_ref(t2: Array2<f32>, orig_shape: &[usize]) -> Tensor {
        let r = orig_shape.len();
        let channels = orig_shape[1];
        let mut cl_shape: Vec<usize> = (0..r).filter(|&a| a != 1).map(|a| orig_shape[a]).collect();
        cl_shape.push(channels);
        let cl = t2.into_shape_with_order(IxDyn(&cl_shape)).unwrap();
        let mut inv: Vec<usize> = vec![0, r - 1];
        inv.extend(1..(r - 1));
        cl.permuted_axes(inv).as_standard_layout().to_owned()
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

    /// The plane folds must match a straight-line composition of the same
    /// per-channel block ranges, for both flag values, including planes that cross block
    /// boundaries, blocks that split a plane, and single-channel/single-batch shapes
    #[test]
    fn par_plane_folds_match_serial_blocked_reference() {
        for &(b, c, p) in &[
            (3usize, 5usize, 7usize),
            (2, 4, 16_384),
            (1, 3, 20_000),
            (4, 1, 1_000),
            (5, 8, 8_191),
            (2, 2, 3),
        ] {
            let x = Array3::from_shape_fn((b, c, p), |(i, j, k)| {
                ((i * 31 + j * 17 + k * 7) as f32 * 0.731).sin()
            });
            let y = Array3::from_shape_fn((b, c, p), |(i, j, k)| {
                ((i * 13 + j * 29 + k * 5) as f32 * 0.377).sin()
            });
            let xs = x.as_slice().unwrap();
            let ys = y.as_slice().unwrap();
            let len_per_chan = b * p;
            let n_blocks = len_per_chan.div_ceil(DET_REDUCE_BLOCK);

            for &scale in &[1.0f32, -0.5, -1.0] {
                let mut ref_sum = vec![0.0f32; c];
                let mut ref_dot = vec![0.0f32; c];
                for ch in 0..c {
                    for blk in 0..n_blocks {
                        let start = blk * DET_REDUCE_BLOCK;
                        let end = (start + DET_REDUCE_BLOCK).min(len_per_chan);
                        ref_sum[ch] += plane_range_sum(xs, ch, c, p, start..end, scale);
                        ref_dot[ch] += plane_range_dot(xs, ys, ch, c, p, start..end, scale);
                    }
                }

                for parallel in [false, true] {
                    let plane_sum = par_plane_sum(xs, c, p, parallel, scale);
                    let plane_dot = par_plane_dot(xs, ys, c, p, parallel, scale);
                    assert_eq!(
                        plane_sum.as_slice().unwrap(),
                        ref_sum.as_slice(),
                        "par_plane_sum mismatch at [{b}x{c}x{p}] scale {scale} \
                         (parallel={parallel})"
                    );
                    assert_eq!(
                        plane_dot.as_slice().unwrap(),
                        ref_dot.as_slice(),
                        "par_plane_dot mismatch at [{b}x{c}x{p}] scale {scale} \
                         (parallel={parallel})"
                    );
                }
            }
        }
    }

    /// On integer-valued data every per-channel sum is exact in f32, so the plane folds must
    /// agree with ndarray's axis reductions exactly regardless of grouping or kernel
    #[test]
    fn par_plane_folds_exact_on_integer_data() {
        let (b, c, p) = (8usize, 16usize, 4_096usize);
        let x = Array3::from_shape_fn((b, c, p), |(i, j, k)| ((i * 7 + j * 13 + k) % 9) as f32);
        let y = Array3::from_shape_fn((b, c, p), |(i, j, k)| ((i * 5 + j * 3 + k) % 7) as f32);

        let serial_sum = x.sum_axis(Axis(2)).sum_axis(Axis(0));
        let serial_dot = (&x * &y).sum_axis(Axis(2)).sum_axis(Axis(0));
        for parallel in [false, true] {
            let plane_sum = par_plane_sum(x.as_slice().unwrap(), c, p, parallel, 1.0);
            assert_eq!(
                plane_sum.as_slice().unwrap(),
                serial_sum.as_slice().unwrap(),
                "integer-data plane sums must be exact and grouping-independent \
                 (parallel={parallel})"
            );

            let plane_dot = par_plane_dot(
                x.as_slice().unwrap(),
                y.as_slice().unwrap(),
                c,
                p,
                parallel,
                1.0,
            );
            assert_eq!(
                plane_dot.as_slice().unwrap(),
                serial_dot.as_slice().unwrap(),
                "integer-data plane dots must be exact and grouping-independent \
                 (parallel={parallel})"
            );
        }
    }

    /// Spatial inference is pure per-element arithmetic, so the native-layout pass must
    /// reproduce the fold -> broadcast -> unfold reference exactly, both below and above
    /// the parallel threshold
    #[test]
    fn spatial_predict_matches_folded_reference_bitwise() {
        for &(b, c, h, w) in &[(3usize, 3usize, 5usize, 7usize), (16, 16, 32, 32)] {
            let mut bn = BatchNormalization::new(vec![b, c, h, w], 0.9, 1e-5).unwrap();
            let gamma = Array1::from_shape_fn(c, |j| 1.5 - 0.25 * j as f32);
            let beta = Array1::from_shape_fn(c, |j| -0.75 + 0.5 * j as f32);
            let running_mean = Array1::from_shape_fn(c, |j| 0.5 * j as f32 - 1.25);
            let running_var = Array1::from_shape_fn(c, |j| 0.25 + 0.5 * j as f32);
            bn.set_weights(
                gamma.clone().into_dyn(),
                beta.clone().into_dyn(),
                running_mean.clone().into_dyn(),
                running_var.clone().into_dyn(),
            )
            .unwrap();

            let x = Array4::from_shape_fn((b, c, h, w), |(i, j, k, l)| {
                ((i * 31 + j * 17 + k * 7 + l * 3) as f32 * 0.519).sin()
            })
            .into_dyn();

            let x_folded = fold_ref(&x);
            let std_dev = (&running_var + 1e-5f32).mapv(|v| v.sqrt());
            let x_normalized = (&x_folded - &running_mean) / &std_dev;
            let expected = unfold_ref(&x_normalized * &gamma + &beta, x.shape());

            let out = bn.predict(&x).unwrap();
            assert_eq!(
                out, expected,
                "spatial predict must be bitwise identical to the folded reference \
                 at [{b}x{c}x{h}x{w}]"
            );
        }
    }

    /// On integer-valued data with a power-of-two sample count the batch statistics are exact
    /// (grouping-independent), so the spatial training forward and its running stats must
    /// reproduce the fold -> broadcast -> unfold reference exactly
    #[test]
    fn spatial_forward_training_matches_folded_reference_on_exact_stats() {
        // B x P = 4 x 16 = 64 samples per channel: integer sums / 64 are exact dyadics
        let (b, c, h, w) = (4usize, 3usize, 4usize, 4usize);
        let mut bn = BatchNormalization::new(vec![b, c, h, w], 0.0, 1e-5).unwrap();
        let gamma = Array1::from(vec![1.5f32, -2.0, 0.5]);
        let beta = Array1::from(vec![0.25f32, 1.0, -0.75]);
        bn.set_weights(
            gamma.clone().into_dyn(),
            beta.clone().into_dyn(),
            Array1::zeros(c).into_dyn(),
            Array1::ones(c).into_dyn(),
        )
        .unwrap();

        let x = Array4::from_shape_fn((b, c, h, w), |(i, j, k, l)| {
            ((i * 7 + j * 5 + k * 3 + l) % 4) as f32
        })
        .into_dyn();

        let x_folded = fold_ref(&x);
        let mean = x_folded.mean_axis(Axis(0)).unwrap();
        let x_centered = &x_folded - &mean;
        let var = (&x_centered * &x_centered).mean_axis(Axis(0)).unwrap();
        let std_dev = (&var + 1e-5f32).mapv(|v| v.sqrt());
        let x_normalized = &x_centered / &std_dev;
        let expected = unfold_ref(&x_normalized * &gamma + &beta, x.shape());

        let out = bn.forward(&x).unwrap();
        assert_eq!(
            out, expected,
            "spatial training forward must be bitwise identical to the folded reference \
             on exact statistics"
        );

        // Momentum 0 makes the running stats equal the batch stats of this single step
        match bn.get_weights() {
            LayerWeight::BatchNormalization(weight) => {
                assert_eq!(
                    weight.running_mean.as_slice().unwrap(),
                    mean.as_slice().unwrap()
                );
                assert_eq!(
                    weight.running_var.as_slice().unwrap(),
                    var.as_slice().unwrap()
                );
            }
            _ => unreachable!(),
        }
    }

    /// The spatial path and an equivalent 2-D layer fed the folded input compute the same
    /// mathematics with different (but each deterministic) reduction groupings, so forward,
    /// grad_input, and the parameter gradients must agree to rounding
    #[test]
    fn spatial_path_matches_folded_2d_layer_closely() {
        let (b, c, h, w) = (2usize, 3usize, 4usize, 5usize);
        let m = b * h * w;
        let mut bn_spatial = BatchNormalization::new(vec![b, c, h, w], 0.9, 1e-5).unwrap();
        let mut bn_2d = BatchNormalization::new(vec![m, c], 0.9, 1e-5).unwrap();

        let x = Array4::from_shape_fn((b, c, h, w), |(i, j, k, l)| {
            ((i * 31 + j * 17 + k * 7 + l * 3) as f32 * 0.871).sin()
        })
        .into_dyn();
        let x_folded = fold_ref(&x).into_dyn();

        let out_spatial = bn_spatial.forward(&x).unwrap();
        let out_2d = bn_2d.forward(&x_folded).unwrap();
        let out_spatial_folded = fold_ref(&out_spatial).into_dyn();

        let grad = Array4::from_shape_fn((b, c, h, w), |(i, j, k, l)| {
            ((i * 13 + j * 29 + k * 5 + l * 11) as f32 * 0.433).sin()
        })
        .into_dyn();
        let grad_folded = fold_ref(&grad).into_dyn();

        let gi_spatial = bn_spatial.backward(&grad).unwrap();
        let gi_2d = bn_2d.backward(&grad_folded).unwrap();
        let gi_spatial_folded = fold_ref(&gi_spatial).into_dyn();

        let close = |a: f32, e: f32| (a - e).abs() <= 1e-5 + 1e-4 * e.abs();
        for (i, (&a, &e)) in out_spatial_folded.iter().zip(out_2d.iter()).enumerate() {
            assert!(close(a, e), "forward mismatch at {i}: {a} vs {e}");
        }
        for (i, (&a, &e)) in gi_spatial_folded.iter().zip(gi_2d.iter()).enumerate() {
            assert!(close(a, e), "grad_input mismatch at {i}: {a} vs {e}");
        }
        for (pg_s, pg_2d) in bn_spatial
            .parameters()
            .iter()
            .zip(bn_2d.parameters().iter())
        {
            for (i, (&a, &e)) in pg_s.grad.iter().zip(pg_2d.grad.iter()).enumerate() {
                assert!(close(a, e), "parameter grad mismatch at {i}: {a} vs {e}");
            }
        }
    }
}
