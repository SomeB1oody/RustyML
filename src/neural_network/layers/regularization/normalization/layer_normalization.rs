use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::{LayerNormalizationLayerWeight, LayerWeight};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape;
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_input_shape,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::Axis;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

/// Threshold for switching between sequential and parallel layer normalization computation.
/// Based on total elements in the tensor.
const LAYER_NORMALIZATION_PARALLEL_THRESHOLD: usize = 1024;

/// Axis selection for layer normalization.
///
/// # Variants
///
/// - `Default` - Normalize along the last dimension (feature dimension)
/// - `Custom(usize)` - Normalize along a single custom specified axis
/// - `Multiple(Vec<usize>)` - Normalize *jointly* over several axes (Keras-style axis list); the
///   statistics are computed over the combined elements of those axes, and `gamma`/`beta` are 1-D
///   with length equal to the product of those axes' sizes
#[derive(Debug, Clone)]
pub enum LayerNormalizationAxis {
    Default,
    Custom(usize),
    Multiple(Vec<usize>),
}

/// Permutes the `axes` to the trailing positions and merges them into a single axis, returning the
/// transformed contiguous tensor plus the permutation and pre-merge shape needed to invert it.
///
/// Multi-axis layer normalization reduces to single-axis normalization of this merged axis, so the
/// public methods bracket the existing (single-axis) core with this transform and its inverse,
/// [`unmerge_normalized_axes`].
fn merge_normalized_axes(
    input: &Tensor,
    axes: &[usize],
) -> Result<(Tensor, Vec<usize>, Vec<usize>), Error> {
    let ndim = input.ndim();
    if axes.is_empty() {
        return Err(Error::invalid_parameter(
            "normalized_axis",
            "LayerNormalization Multiple axis list must be non-empty",
        ));
    }
    for (i, &a) in axes.iter().enumerate() {
        if a >= ndim {
            return Err(Error::invalid_parameter(
                "normalized_axis",
                format!("Normalization axis {a} is out of bounds for input with {ndim} dimensions"),
            ));
        }
        if axes[..i].contains(&a) {
            return Err(Error::invalid_parameter(
                "normalized_axis",
                format!("Duplicate normalization axis {a}"),
            ));
        }
    }

    // perm = non-normalized axes (original order) followed by the normalized axes (given order).
    let mut perm: Vec<usize> = (0..ndim).filter(|ax| !axes.contains(ax)).collect();
    perm.extend_from_slice(axes);
    let permuted = input
        .view()
        .permuted_axes(perm.clone())
        .as_standard_layout()
        .to_owned();
    let permuted_shape = permuted.shape().to_vec();

    let outer = ndim - axes.len();
    let inner: usize = axes.iter().map(|&a| input.shape()[a]).product();
    let mut merged_shape: Vec<usize> = permuted_shape[..outer].to_vec();
    merged_shape.push(inner);
    let merged = permuted
        .into_shape_with_order(merged_shape)
        .context("layer-norm merge reshape")?;
    Ok((merged, perm, permuted_shape))
}

/// Inverts [`merge_normalized_axes`]: un-merges back to `permuted_shape`, then applies the inverse
/// of `perm`, returning a contiguous tensor in the original layout.
fn unmerge_normalized_axes(
    output_merged: Tensor,
    perm: &[usize],
    permuted_shape: &[usize],
) -> Tensor {
    let unmerged = output_merged
        .into_shape_with_order(permuted_shape.to_vec())
        .expect("unmerge reshape must succeed (element count is unchanged)");
    let ndim = perm.len();
    let mut inv = vec![0usize; ndim];
    for (new_pos, &old_ax) in perm.iter().enumerate() {
        inv[old_ax] = new_pos;
    }
    unmerged
        .view()
        .permuted_axes(inv)
        .as_standard_layout()
        .to_owned()
}

/// Layer Normalization layer for neural networks.
///
/// Normalizes across feature dimensions for each sample, which is effective
/// when batch sizes are small or variable.
///
/// # Fields
///
/// - `epsilon` - Small constant for numerical stability in normalization
/// - `normalized_axis` - Axis along which to normalize
/// - `input_shape` - Shape of the input tensor
/// - `gamma` - Scale parameter (trainable)
/// - `beta` - Shift parameter (trainable)
/// - `training` - Whether the layer is in training mode or inference mode
/// - `x_normalized` - Normalized input (used in backward pass)
/// - `x_centered` - Centered input (used in backward pass)
/// - `mean` - Mean computed during forward pass (used in backward pass)
/// - `std_dev` - Standard deviation computed during forward pass (used in backward pass)
/// - `grad_gamma` - Gradient for gamma parameter
/// - `grad_beta` - Gradient for beta parameter
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a LayerNormalization layer
/// let mut ln = LayerNormalization::new(vec![32, 128], LayerNormalizationAxis::Default, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, normalizes the input
/// let output = ln.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct LayerNormalization {
    epsilon: f32,
    normalized_axis: LayerNormalizationAxis,
    input_shape: Vec<usize>,
    gamma: Tensor,
    beta: Tensor,
    training: bool,
    // Cache for backward pass
    x_normalized: Option<Tensor>,
    x_centered: Option<Tensor>,
    mean: Option<Tensor>,
    std_dev: Option<Tensor>,
    // Gradients
    grad_gamma: Option<Tensor>,
    grad_beta: Option<Tensor>,
}

impl LayerNormalization {
    /// Creates a new LayerNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor
    /// - `normalized_axis` - Axis along which to normalize
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New LayerNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::invalid_parameter` - If `epsilon` is not positive or not finite
    pub fn new(
        input_shape: Vec<usize>,
        normalized_axis: LayerNormalizationAxis,
        epsilon: f32,
    ) -> Result<Self, Error> {
        validate_epsilon(epsilon)?;

        // gamma/beta are 1-D over the normalized dimension(s). For a single axis that is the size of
        // that axis; for Multiple it is the product of the listed axes' sizes (they are merged into
        // one axis at runtime).
        let param_shape = match &normalized_axis {
            LayerNormalizationAxis::Default => {
                if input_shape.is_empty() {
                    vec![1]
                } else {
                    vec![input_shape[input_shape.len() - 1]]
                }
            }
            LayerNormalizationAxis::Custom(axis) => {
                if input_shape.len() > *axis {
                    vec![input_shape[*axis]]
                } else {
                    vec![1]
                }
            }
            LayerNormalizationAxis::Multiple(axes) => {
                if axes.is_empty() {
                    return Err(Error::invalid_parameter(
                        "normalized_axis",
                        "LayerNormalization Multiple axis list must be non-empty",
                    ));
                }
                for (i, &a) in axes.iter().enumerate() {
                    if a >= input_shape.len() {
                        return Err(Error::invalid_parameter(
                            "normalized_axis",
                            format!(
                                "Normalization axis {a} is out of bounds for input shape with {} dimensions",
                                input_shape.len()
                            ),
                        ));
                    }
                    if axes[..i].contains(&a) {
                        return Err(Error::invalid_parameter(
                            "normalized_axis",
                            format!("Duplicate normalization axis {a}"),
                        ));
                    }
                }
                vec![axes.iter().map(|&a| input_shape[a]).product()]
            }
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(LayerNormalization {
            epsilon,
            normalized_axis,
            input_shape,
            gamma: Tensor::ones(param_shape_ndarray),
            beta: Tensor::zeros(param_shape_ndarray),
            training: true,
            x_normalized: None,
            x_centered: None,
            mean: None,
            std_dev: None,
            grad_gamma: None,
            grad_beta: None,
        })
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the LayerNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable)
    /// - `beta` - Shift parameter (trainable)
    pub fn set_weights(&mut self, gamma: Tensor, beta: Tensor) -> Result<(), Error> {
        validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}

impl Layer for LayerNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Resolve the working layout and axis. Default/Custom normalize an existing axis in place;
        // Multiple permutes the chosen axes to the trailing position and merges them into one axis,
        // running the same single-axis core and inverting the transform on the result.
        let merged = match &self.normalized_axis {
            LayerNormalizationAxis::Multiple(axes) => Some(merge_normalized_axes(input, axes)?),
            _ => None,
        };
        let (input, axis_idx): (&Tensor, usize) = match (&self.normalized_axis, &merged) {
            (LayerNormalizationAxis::Default, _) => {
                if input.ndim() == 0 {
                    return Err(Error::invalid_input("Cannot normalize a scalar tensor"));
                }
                (input, input.ndim() - 1)
            }
            (LayerNormalizationAxis::Custom(axis), _) => {
                if *axis >= input.ndim() {
                    return Err(Error::invalid_parameter(
                        "normalized_axis",
                        format!(
                            "Normalization axis {} is out of bounds for input with {} dimensions",
                            axis,
                            input.ndim()
                        ),
                    ));
                }
                (input, *axis)
            }
            (LayerNormalizationAxis::Multiple(_), Some((m, _, _))) => (m, m.ndim() - 1),
            (LayerNormalizationAxis::Multiple(_), None) => unreachable!(),
        };

        let total_elements = input.len();

        // Compute mean along the specified axis
        let mean = input.mean_axis(Axis(axis_idx)).unwrap();

        // Insert the axis back to make broadcasting work
        let mean = mean.insert_axis(Axis(axis_idx));

        // Center the data and compute variance
        let (x_centered, var) = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel centering and variance computation
            let mut x_centered = Tensor::zeros(input.raw_dim());
            let mut squared_diff = Tensor::zeros(input.raw_dim());

            // Calculate strides for index mapping
            let input_shape = input.shape();
            let axis_size = input_shape[axis_idx];
            let after_axis_size: usize = input_shape[axis_idx + 1..].iter().product();

            x_centered
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(squared_diff.as_slice_mut().unwrap().par_iter_mut())
                .zip(input.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((centered, sq_diff), &val))| {
                    // Map flat index to position in mean array
                    let mean_idx =
                        (i / after_axis_size / axis_size) * after_axis_size + (i % after_axis_size);
                    let mean_val = mean.as_slice().unwrap()[mean_idx];
                    let diff = val - mean_val;
                    *centered = diff;
                    *sq_diff = diff * diff;
                });

            let var = squared_diff
                .mean_axis(Axis(axis_idx))
                .unwrap()
                .insert_axis(Axis(axis_idx));
            (x_centered, var)
        } else {
            // Sequential computation
            let x_centered = input - &mean;
            let var = (&x_centered * &x_centered)
                .mean_axis(Axis(axis_idx))
                .unwrap();
            let var = var.insert_axis(Axis(axis_idx));
            (x_centered, var)
        };

        // Normalize
        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel normalization
            let mut x_normalized = Tensor::zeros(x_centered.raw_dim());

            let input_shape = input.shape();
            let axis_size = input_shape[axis_idx];
            let after_axis_size: usize = input_shape[axis_idx + 1..].iter().product();

            x_normalized
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (norm, &centered))| {
                    let std_idx =
                        (i / after_axis_size / axis_size) * after_axis_size + (i % after_axis_size);
                    let std_val = std_dev.as_slice().unwrap()[std_idx];
                    *norm = centered / std_val;
                });

            x_normalized
        } else {
            // Sequential normalization
            &x_centered / &std_dev
        };

        // Scale and shift
        // Reshape gamma and beta to match the input shape for broadcasting
        let mut gamma_shape = vec![1; input.ndim()];
        let mut beta_shape = vec![1; input.ndim()];

        // Set the dimensions from axis_idx onwards to match gamma/beta shape
        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
            beta_shape[axis_idx + i] = dim;
        }

        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();
        let beta_broadcast = self
            .beta
            .clone()
            .into_shape_with_order(beta_shape.as_slice())
            .unwrap();

        let output = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel scale and shift
            let mut output = Tensor::zeros(x_normalized.raw_dim());

            let input_shape = input.shape();
            let axis_size = input_shape[axis_idx];
            let after_axis_size: usize = input_shape[axis_idx + 1..].iter().product();

            output
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_normalized.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (out, &norm))| {
                    // Index of this element along the normalized axis (gamma/beta are 1-D).
                    let param_idx = (i / after_axis_size) % axis_size;
                    let gamma_val = gamma_broadcast.as_slice().unwrap()[param_idx];
                    let beta_val = beta_broadcast.as_slice().unwrap()[param_idx];
                    *out = norm * gamma_val + beta_val;
                });

            output
        } else {
            // Sequential scale and shift
            &x_normalized * &gamma_broadcast + &beta_broadcast
        };

        // Cache values for backward pass
        self.x_normalized = Some(x_normalized);
        self.x_centered = Some(x_centered);
        self.mean = Some(mean);
        self.std_dev = Some(std_dev);

        let output = match &merged {
            Some((_, perm, permuted_shape)) => {
                unmerge_normalized_axes(output, perm, permuted_shape)
            }
            None => output,
        };
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Resolve the working layout and axis. Default/Custom normalize an existing axis in place;
        // Multiple permutes the chosen axes to the trailing position and merges them into one axis,
        // running the same single-axis core and inverting the transform on the result.
        let merged = match &self.normalized_axis {
            LayerNormalizationAxis::Multiple(axes) => Some(merge_normalized_axes(input, axes)?),
            _ => None,
        };
        let (input, axis_idx): (&Tensor, usize) = match (&self.normalized_axis, &merged) {
            (LayerNormalizationAxis::Default, _) => {
                if input.ndim() == 0 {
                    return Err(Error::invalid_input("Cannot normalize a scalar tensor"));
                }
                (input, input.ndim() - 1)
            }
            (LayerNormalizationAxis::Custom(axis), _) => {
                if *axis >= input.ndim() {
                    return Err(Error::invalid_parameter(
                        "normalized_axis",
                        format!(
                            "Normalization axis {} is out of bounds for input with {} dimensions",
                            axis,
                            input.ndim()
                        ),
                    ));
                }
                (input, *axis)
            }
            (LayerNormalizationAxis::Multiple(_), Some((m, _, _))) => (m, m.ndim() - 1),
            (LayerNormalizationAxis::Multiple(_), None) => unreachable!(),
        };

        let total_elements = input.len();

        // Compute mean along the specified axis
        let mean = input.mean_axis(Axis(axis_idx)).unwrap();

        // Insert the axis back to make broadcasting work
        let mean = mean.insert_axis(Axis(axis_idx));

        // Center the data and compute variance
        let (x_centered, var) = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel centering and variance computation
            let mut x_centered = Tensor::zeros(input.raw_dim());
            let mut squared_diff = Tensor::zeros(input.raw_dim());

            // Calculate strides for index mapping
            let input_shape = input.shape();
            let axis_size = input_shape[axis_idx];
            let after_axis_size: usize = input_shape[axis_idx + 1..].iter().product();

            x_centered
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(squared_diff.as_slice_mut().unwrap().par_iter_mut())
                .zip(input.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((centered, sq_diff), &val))| {
                    // Map flat index to position in mean array
                    let mean_idx =
                        (i / after_axis_size / axis_size) * after_axis_size + (i % after_axis_size);
                    let mean_val = mean.as_slice().unwrap()[mean_idx];
                    let diff = val - mean_val;
                    *centered = diff;
                    *sq_diff = diff * diff;
                });

            let var = squared_diff
                .mean_axis(Axis(axis_idx))
                .unwrap()
                .insert_axis(Axis(axis_idx));
            (x_centered, var)
        } else {
            // Sequential computation
            let x_centered = input - &mean;
            let var = (&x_centered * &x_centered)
                .mean_axis(Axis(axis_idx))
                .unwrap();
            let var = var.insert_axis(Axis(axis_idx));
            (x_centered, var)
        };

        // Normalize
        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel normalization
            let mut x_normalized = Tensor::zeros(x_centered.raw_dim());

            let input_shape = input.shape();
            let axis_size = input_shape[axis_idx];
            let after_axis_size: usize = input_shape[axis_idx + 1..].iter().product();

            x_normalized
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (norm, &centered))| {
                    let std_idx =
                        (i / after_axis_size / axis_size) * after_axis_size + (i % after_axis_size);
                    let std_val = std_dev.as_slice().unwrap()[std_idx];
                    *norm = centered / std_val;
                });

            x_normalized
        } else {
            // Sequential normalization
            &x_centered / &std_dev
        };

        // Scale and shift
        // Reshape gamma and beta to match the input shape for broadcasting
        let mut gamma_shape = vec![1; input.ndim()];
        let mut beta_shape = vec![1; input.ndim()];

        // Set the dimensions from axis_idx onwards to match gamma/beta shape
        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
            beta_shape[axis_idx + i] = dim;
        }

        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();
        let beta_broadcast = self
            .beta
            .clone()
            .into_shape_with_order(beta_shape.as_slice())
            .unwrap();

        let output = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel scale and shift
            let mut output = Tensor::zeros(x_normalized.raw_dim());

            let input_shape = input.shape();
            let axis_size = input_shape[axis_idx];
            let after_axis_size: usize = input_shape[axis_idx + 1..].iter().product();

            output
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_normalized.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (out, &norm))| {
                    // Index of this element along the normalized axis (gamma/beta are 1-D).
                    let param_idx = (i / after_axis_size) % axis_size;
                    let gamma_val = gamma_broadcast.as_slice().unwrap()[param_idx];
                    let beta_val = beta_broadcast.as_slice().unwrap()[param_idx];
                    *out = norm * gamma_val + beta_val;
                });

            output
        } else {
            // Sequential scale and shift
            &x_normalized * &gamma_broadcast + &beta_broadcast
        };

        let output = match &merged {
            Some((_, perm, permuted_shape)) => {
                unmerge_normalized_axes(output, perm, permuted_shape)
            }
            None => output,
        };
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        // Same layout handling as `forward` (see there); the cached intermediates are already in the
        // merged layout, so we transform `grad_output` to match and invert on the input gradient.
        let merged = match &self.normalized_axis {
            LayerNormalizationAxis::Multiple(axes) => {
                Some(merge_normalized_axes(grad_output, axes)?)
            }
            _ => None,
        };
        let (grad_output, axis_idx): (&Tensor, usize) = match (&self.normalized_axis, &merged) {
            (LayerNormalizationAxis::Default, _) => (grad_output, grad_output.ndim() - 1),
            (LayerNormalizationAxis::Custom(axis), _) => (grad_output, *axis),
            (LayerNormalizationAxis::Multiple(_), Some((m, _, _))) => (m, m.ndim() - 1),
            (LayerNormalizationAxis::Multiple(_), None) => unreachable!(),
        };

        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;

        let x_centered = self
            .x_centered
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;

        let std_dev = self
            .std_dev
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;

        // Compute gradients for gamma and beta. Since gamma/beta are 1-D over the normalized axis,
        // reduce every other axis, leaving a gradient of shape `[axis_size]`.
        let mut grad_gamma = grad_output * x_normalized;
        let mut grad_beta = grad_output.clone();

        let ndim = grad_gamma.ndim();
        // Sum the axes after the normalized axis (from the end so lower indices are unaffected),
        // then the axes before it.
        for i in (axis_idx + 1..ndim).rev() {
            grad_gamma = grad_gamma.sum_axis(Axis(i));
            grad_beta = grad_beta.sum_axis(Axis(i));
        }
        for i in (0..axis_idx).rev() {
            grad_gamma = grad_gamma.sum_axis(Axis(i));
            grad_beta = grad_beta.sum_axis(Axis(i));
        }

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        let total_elements = grad_output.len();

        // Compute gradient with respect to normalized input
        // Reshape gamma for broadcasting
        let mut gamma_shape = vec![1; grad_output.ndim()];
        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
        }
        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();

        let grad_x_normalized = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());

            let output_shape = grad_output.shape();
            let axis_size = output_shape[axis_idx];
            let after_axis_size: usize = output_shape[axis_idx + 1..].iter().product();

            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    // Index of this element along the normalized axis (gamma is 1-D).
                    let param_idx = (i / after_axis_size) % axis_size;
                    let gamma_val = gamma_broadcast.as_slice().unwrap()[param_idx];
                    *g_norm = g_out * gamma_val;
                });
            grad_x_norm
        } else {
            // Sequential computation
            grad_output * &gamma_broadcast
        };

        // Compute inverse standard deviation
        let inv_std = std_dev.mapv(|x| 1.0 / x);

        // Get the size of the normalization dimension
        let norm_size = grad_output.shape()[axis_idx] as f32;

        // Compute gradient with respect to variance
        let grad_var = (&grad_x_normalized * x_centered * -0.5).sum_axis(Axis(axis_idx));
        let grad_var = grad_var.insert_axis(Axis(axis_idx));
        let grad_var = &grad_var * &inv_std * &inv_std * &inv_std;

        // Compute gradient with respect to mean
        let grad_mean_1 = (&grad_x_normalized * -1.0).sum_axis(Axis(axis_idx));
        let grad_mean_1 = grad_mean_1.insert_axis(Axis(axis_idx));
        let grad_mean_1 = &grad_mean_1 * &inv_std;

        let x_sum = x_centered
            .sum_axis(Axis(axis_idx))
            .insert_axis(Axis(axis_idx));
        let grad_mean_2 = &grad_var * (&x_sum * -2.0 / norm_size);
        let grad_mean = grad_mean_1 + grad_mean_2;

        // Compute gradient with respect to input
        let grad_input = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());

            let output_shape = grad_output.shape();
            let axis_size = output_shape[axis_idx];
            let after_axis_size: usize = output_shape[axis_idx + 1..].iter().product();

            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_x_normalized.as_slice().unwrap().par_iter())
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((g_inp, &g_norm), &x_cent))| {
                    let stat_idx =
                        (i / after_axis_size / axis_size) * after_axis_size + (i % after_axis_size);
                    let inv_std_val = inv_std.as_slice().unwrap()[stat_idx];
                    let grad_var_val = grad_var.as_slice().unwrap()[stat_idx];
                    let grad_mean_val = grad_mean.as_slice().unwrap()[stat_idx];

                    *g_inp = g_norm * inv_std_val
                        + grad_var_val * x_cent * 2.0 / norm_size
                        + grad_mean_val / norm_size;
                });
            grad_inp
        } else {
            // Sequential computation
            &grad_x_normalized * &inv_std
                + &grad_var * (x_centered * 2.0 / norm_size)
                + &grad_mean / norm_size
        };

        let grad_input = match &merged {
            Some((_, perm, permuted_shape)) => {
                unmerge_normalized_axes(grad_input, perm, permuted_shape)
            }
            None => grad_input,
        };
        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "LayerNormalization"
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
        LayerWeight::LayerNormalizationLayer(LayerNormalizationLayerWeight {
            gamma: &self.gamma,
            beta: &self.beta,
        })
    }

    mode_dependent_layer_trait!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    // Helper: build a Tensor from a flat Vec and shape.
    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        ArrayD::from_shape_vec(shape, data).expect("shape/data mismatch in test helper")
    }

    // ─── merge_normalized_axes: output shape ────────────────────────────────

    /// merge_normalized_axes on shape [3,4,5] with axes=[1,2] → merged shape [3,20].
    ///
    /// Derivation:
    ///   ndim=3, axes=[1,2].
    ///   Non-normalized axes: [0].  perm = [0, 1, 2].
    ///   permuted shape = [3, 4, 5] (perm is identity here).
    ///   outer = ndim - |axes| = 3 - 2 = 1.
    ///   inner = shape[1] * shape[2] = 4 * 5 = 20.
    ///   merged shape = [3, 20].
    #[test]
    fn test_merge_normalized_axes_shape() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (merged, _perm, _permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        assert_eq!(merged.shape(), &[3, 20]);
    }

    /// merge_normalized_axes: perm returned for [3,4,5] with axes=[1,2].
    ///
    /// Derivation:
    ///   Non-normalized axes (in order): 0.
    ///   Normalized axes (in given order): 1, 2.
    ///   perm = [0, 1, 2].
    #[test]
    fn test_merge_normalized_axes_perm() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (_merged, perm, _permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        assert_eq!(perm, vec![0usize, 1, 2]);
    }

    /// merge_normalized_axes: permuted_shape for [3,4,5] with axes=[1,2].
    ///
    /// Derivation:
    ///   perm = [0,1,2] (identity) → permuted shape = [3,4,5].
    #[test]
    fn test_merge_normalized_axes_permuted_shape() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (_merged, _perm, permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        assert_eq!(permuted_shape, vec![3usize, 4, 5]);
    }

    // ─── merge + unmerge round-trip ─────────────────────────────────────────

    /// unmerge_normalized_axes(merge_normalized_axes(x)) == x element-wise,
    /// for shape [3,4,5] with axes=[1,2].
    ///
    /// Derivation:
    ///   merge: reshapes [3,4,5] → [3,20] via identity perm [0,1,2], permuted_shape=[3,4,5].
    ///   unmerge: reshapes [3,20] → [3,4,5] then applies inverse of [0,1,2] = [0,1,2] (identity).
    ///   Net effect is a reshape→reshape identity, so every element is preserved.
    #[test]
    fn test_merge_unmerge_round_trip() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 1.1).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (merged, perm, permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        let recovered = unmerge_normalized_axes(merged, &perm, &permuted_shape);

        assert_eq!(recovered.shape(), x.shape());
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    /// Round-trip also holds when axes require a non-trivial permutation.
    /// Shape [2,3,4], axes=[0,2]:
    ///   Non-normalized axes: [1]. perm = [1, 0, 2].
    ///   permuted shape = [3, 2, 4].
    ///   merged shape = [3, 8].
    ///   unmerge inverts: reshape [3,8]→[3,2,4], then inverse of [1,0,2] is [1,0,2].
    ///   Result shape is [2,3,4] and elements match original.
    #[test]
    fn test_merge_unmerge_round_trip_nontrivial_perm() {
        let n = 2 * 3 * 4;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.7 + 0.3).collect();
        let x = make_tensor(data, &[2, 3, 4]);

        let (merged, perm, permuted_shape) =
            merge_normalized_axes(&x, &[0, 2]).expect("merge_normalized_axes failed");

        // Verify intermediate merged shape: outer=1 (axis 1), inner=2*4=8 → [3,8].
        assert_eq!(merged.shape(), &[3, 8]);

        let recovered = unmerge_normalized_axes(merged, &perm, &permuted_shape);

        assert_eq!(recovered.shape(), x.shape());
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "non-trivial round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    // ─── merge_normalized_axes: error cases ─────────────────────────────────

    /// merge_normalized_axes returns Err for an empty axes slice.
    #[test]
    fn test_merge_normalized_axes_empty_axes_error() {
        let x = make_tensor(vec![1.0f32, 2.0, 3.0], &[3]);
        let result = merge_normalized_axes(&x, &[]);
        assert!(result.is_err(), "expected Err for empty axes");
    }

    /// merge_normalized_axes returns Err when an axis is out of bounds.
    #[test]
    fn test_merge_normalized_axes_out_of_bounds_error() {
        let x = make_tensor(vec![1.0f32; 6], &[2, 3]);
        let result = merge_normalized_axes(&x, &[2]); // ndim=2, axis 2 is OOB
        assert!(result.is_err(), "expected Err for out-of-bounds axis");
    }

    /// merge_normalized_axes returns Err for duplicate axes.
    #[test]
    fn test_merge_normalized_axes_duplicate_axes_error() {
        let x = make_tensor(vec![1.0f32; 12], &[3, 4]);
        let result = merge_normalized_axes(&x, &[0, 0]);
        assert!(result.is_err(), "expected Err for duplicate axis");
    }
}
