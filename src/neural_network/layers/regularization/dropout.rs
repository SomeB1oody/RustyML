//! Regularization layers (Dropout and SpatialDropout1D/2D/3D) plus the shared
//! backward, output-shape, and masking helpers they share

use crate::error::Error;
use crate::neural_network::Tensor;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

/// Common backward pass shared by all dropout layers
///
/// Implements the backward logic used by every dropout variant (Dropout and
/// SpatialDropout1D/2D/3D)
///
/// # Parameters
///
/// - `grad_output` - Gradient from the next layer
/// - `mask` - The dropout mask applied during the forward pass
/// - `training` - Whether the layer is in training mode
/// - `rate` - The dropout rate
/// - `layer_name` - Concrete layer name, used in the "forward pass not run" error message so the
///   error identifies the actual layer (e.g. `SpatialDropout2D`) rather than always `Dropout`
///
/// # Returns
///
/// - `Result<Tensor, Error>` - Gradient to pass to the previous layer
///
/// # Errors
///
/// Returns an error when the forward pass has not been run and no mask is available
fn dropout_backward(
    grad_output: &Tensor,
    mask: &Option<Tensor>,
    training: bool,
    rate: f32,
    layer_name: &'static str,
) -> Result<Tensor, Error> {
    if !training || rate == 0.0 {
        // During inference or zero rate, pass the gradient through unchanged
        return Ok(grad_output.clone());
    }

    if rate == 1.0 {
        // Rate of 1.0 drops everything, so the gradient is zero
        return Ok(Tensor::zeros(grad_output.raw_dim()));
    }

    // Apply the same mask to the gradient
    if let Some(mask) = mask {
        let scale = 1.0 / (1.0 - rate);
        let grad_input = grad_output * mask * scale;
        Ok(grad_input)
    } else {
        Err(Error::forward_pass_not_run(layer_name))
    }
}

/// Common output-shape formatting shared by all dropout layers
///
/// Formats the input shape into a string representation, as the output shape
/// equals the input shape for every dropout variant
///
/// # Parameters
///
/// - `input_shape` - The input shape vector
///
/// # Returns
///
/// - `String` - Formatted output shape string
fn dropout_output_shape(input_shape: &[usize]) -> String {
    if !input_shape.is_empty() {
        format!(
            "({})",
            input_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::from("Unknown")
    }
}

/// Thresholds a random mask into a binary mask, in parallel or sequentially
///
/// Used by all spatial dropout layers to convert a random mask into a binary
/// mask based on the dropout rate. Larger masks use parallel computation
///
/// # Parameters
///
/// - `mask_2d` - The random mask to convert to binary (modified in place)
/// - `rate` - The dropout rate threshold
/// - `parallel_threshold` - Element count at or above which parallel computation is used
fn apply_spatial_dropout_threshold(mask_2d: &mut Tensor, rate: f32, parallel_threshold: usize) {
    let total_elements = mask_2d.len();

    // Use parallel computation for large masks, sequential otherwise
    if total_elements >= parallel_threshold {
        mask_2d.par_mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    } else {
        mask_2d.mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    }
}

/// Applies the per-channel inverted-dropout scale to a `[batch, channels, *spatial]` tensor
/// without materializing the full mask.
///
/// `channel_mask` holds one binary keep/drop value per `(batch, channel)` in row-major order
/// (flat index `b * channels + c`), matching the contiguous `spatial`-length segments of a
/// standard-layout input: segment `i` is `t[i * spatial .. (i + 1) * spatial]` and is scaled by
/// `channel_mask[i] / (1 - rate)`. Used for both the forward output (`t = input`) and the
/// backward input-gradient (`t = grad_output`), since both are the same elementwise scale.
///
/// Each output element depends only on its own input element and its channel's scalar - there
/// is no reduction - so the work splits into independent per-segment tasks and the `parallel`
/// flag (taken from the element-count gate) never changes the result bits. The per-channel
/// value broadcasts implicitly across the segment, so no input-sized mask is ever built or
/// stored.
///
/// Bit-identical to the explicit `t * broadcast(mask) * scale`: the mask is binary, so
/// `(x * 1) * scale == x * (1 * scale)` and `(x * 0) * scale == x * (0 * scale)` both hold
/// exactly.
fn spatial_dropout_scale(
    t: &Tensor,
    channel_mask: &[f32],
    rate: f32,
    parallel_threshold: usize,
) -> Tensor {
    let n_segments = channel_mask.len();
    let total = t.len();
    let segment = total / n_segments;
    let scale = 1.0 / (1.0 - rate);

    // Contiguous data lets the segments map onto fixed slices; standardize a non-contiguous view
    let t_std = t.as_standard_layout();
    let src = t_std.as_slice().unwrap();

    let mut out = Tensor::zeros(t.raw_dim());
    let dst = out.as_slice_mut().unwrap();

    let task = |((o, x), &m): ((&mut [f32], &[f32]), &f32)| {
        let factor = m * scale;
        for (o_elem, &x_elem) in o.iter_mut().zip(x) {
            *o_elem = x_elem * factor;
        }
    };

    if total >= parallel_threshold {
        dst.par_chunks_mut(segment)
            .zip(src.par_chunks(segment))
            .zip(channel_mask.par_iter())
            .for_each(task);
    } else {
        dst.chunks_mut(segment)
            .zip(src.chunks(segment))
            .zip(channel_mask.iter())
            .for_each(task);
    }
    out
}

/// Backward pass shared by the spatial-dropout layers: applies the stored per-channel mask to
/// the gradient with [`spatial_dropout_scale`], without rebuilding a full-size mask.
///
/// Mirrors the early-return structure of [`dropout_backward`] (inference / `rate == 0`
/// pass-through, `rate == 1` zeros, missing-mask error), but the stored `mask` is the small
/// `[batch, channels]` per-channel mask rather than a full-shape one.
fn spatial_dropout_backward(
    grad_output: &Tensor,
    mask: &Option<Tensor>,
    training: bool,
    rate: f32,
    layer_name: &'static str,
    parallel_threshold: usize,
) -> Result<Tensor, Error> {
    if !training || rate == 0.0 {
        return Ok(grad_output.clone());
    }
    if rate == 1.0 {
        return Ok(Tensor::zeros(grad_output.raw_dim()));
    }
    if let Some(mask) = mask {
        let channel_mask = mask
            .as_slice()
            .expect("per-channel dropout mask is contiguous");
        Ok(spatial_dropout_scale(
            grad_output,
            channel_mask,
            rate,
            parallel_threshold,
        ))
    } else {
        Err(Error::forward_pass_not_run(layer_name))
    }
}

/// Dropout layer for neural networks
// `Dropout` lives in a `dropout` submodule beside the sibling spatial-dropout modules; the
// repeated name is the intended file layout
#[allow(clippy::module_inception)]
pub mod dropout;
/// Spatial Dropout layer for 1D data
pub mod spatial_dropout_1d;
/// Spatial Dropout layer for 2D data
pub mod spatial_dropout_2d;
/// Spatial Dropout layer for 3D data
pub mod spatial_dropout_3d;

pub use dropout::Dropout;
pub use spatial_dropout_1d::SpatialDropout1D;
pub use spatial_dropout_2d::SpatialDropout2D;
pub use spatial_dropout_3d::SpatialDropout3D;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Each output element of the per-channel scale depends only on its own input element and
    /// its channel's scalar (no reduction), so forcing serial vs parallel must be bitwise
    /// identical - the gate is a pure performance knob. Covers segments shorter and longer
    /// than typical chunking and a non-divisor segment count.
    #[test]
    fn spatial_dropout_scale_parallel_flag_invariant() {
        for &(n_seg, seg) in &[(7usize, 5usize), (64, 256), (33, 4096), (512, 17)] {
            let total = n_seg * seg;
            let t = Tensor::from_shape_vec(
                IxDyn(&[n_seg, seg]),
                (0..total).map(|i| (i as f32 * 0.013).sin()).collect(),
            )
            .unwrap();
            // A mix of kept (1.0) and dropped (0.0) channels
            let channel_mask: Vec<f32> = (0..n_seg).map(|i| (i % 3 != 0) as u8 as f32).collect();
            let rate = 0.25f32;

            // total >= gate forces parallel; usize::MAX forces serial
            let serial = spatial_dropout_scale(&t, &channel_mask, rate, usize::MAX);
            let parallel = spatial_dropout_scale(&t, &channel_mask, rate, 0);
            assert_eq!(
                serial.as_slice().unwrap(),
                parallel.as_slice().unwrap(),
                "spatial_dropout_scale parallel flag changed the bits at [{n_seg}x{seg}]"
            );

            // And bit-identical to the explicit `t * broadcast(mask) * scale` it replaces
            let scale = 1.0 / (1.0 - rate);
            let mut expected = vec![0.0f32; total];
            for (i, e) in expected.iter_mut().enumerate() {
                let m = channel_mask[i / seg];
                let x = t.as_slice().unwrap()[i];
                *e = (x * m) * scale;
            }
            assert_eq!(
                serial.as_slice().unwrap(),
                expected.as_slice(),
                "spatial_dropout_scale differs from the explicit two-step form at [{n_seg}x{seg}]"
            );
        }
    }
}
