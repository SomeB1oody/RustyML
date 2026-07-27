//! Dropout-family regularization layers and the shared helpers that build them
//!
//! Re-exports the plain [`Dropout`](crate::neural_network::layers::regularization::dropout::Dropout)
//! layer alongside its spatial variants
//! [`SpatialDropout1D`](crate::neural_network::layers::regularization::dropout::SpatialDropout1D),
//! [`SpatialDropout2D`](crate::neural_network::layers::regularization::dropout::SpatialDropout2D),
//! and [`SpatialDropout3D`](crate::neural_network::layers::regularization::dropout::SpatialDropout3D),
//! grouping each in its own submodule (`dropout`, `spatial_dropout_1d`/`2d`/`3d`)
//!
//! Defines the infrastructure these layers share:
//!
//! - `dropout_backward` - the common inverted-dropout backward pass for the plain layer
//!   (pass-through at inference or `rate == 0`, zeros at `rate == 1`, otherwise scales by the
//!   stored mask)
//! - `dropout_output_shape` - formats the (unchanged) output shape, since dropout preserves the
//!   input shape
//! - `apply_spatial_dropout_threshold` - thresholds a random mask into a binary keep/drop mask,
//!   parallel or sequential by element count
//! - `spatial_dropout_scale` and `spatial_dropout_backward` - apply the per-channel inverted-dropout
//!   scale to a `[batch, *spatial, channels]` tensor from a small `[batch, channels]` mask without
//!   materializing a full-size mask

use crate::error::Error;
use crate::neural_network::Tensor;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

/// Common backward pass shared by all dropout variants (Dropout and SpatialDropout1D/2D/3D)
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
        // Inference or zero rate: pass the gradient through unchanged
        return Ok(grad_output.clone());
    }

    if rate == 1.0 {
        // Rate of 1.0 drops everything, so the gradient is zero
        return Ok(Tensor::zeros(grad_output.raw_dim()));
    }

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
/// The output shape equals the input shape for every dropout variant, so this formats
/// the input shape into a string representation
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
/// Used by all spatial dropout layers to convert a random mask into a binary mask based
/// on the dropout rate. Larger masks use parallel computation
///
/// # Parameters
///
/// - `mask_2d` - The random mask to convert to binary (modified in place)
/// - `rate` - The dropout rate threshold
/// - `parallel_threshold` - Element count at or above which parallel computation is used
fn apply_spatial_dropout_threshold(mask_2d: &mut Tensor, rate: f32, parallel_threshold: usize) {
    let total_elements = mask_2d.len();

    if total_elements >= parallel_threshold {
        mask_2d.par_mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    } else {
        mask_2d.mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    }
}

/// Applies the per-channel inverted-dropout scale to a `[batch, *spatial, channels]` tensor
/// without materializing the full mask
///
/// `channel_mask` holds one binary keep/drop value per `(batch, channel)` in row-major order
/// (flat index `b * channels + c`). Under the channels-last layout that is exactly one factor per
/// element of a position's channel vector, so every position of a batch item is scaled by the same
/// `channels`-long vector. Used for both the forward output (`t = input`) and the backward
/// input-gradient (`t = grad_output`), since both are the same elementwise scale
///
/// The scale is applied through a small repeated *tile* rather than one channel vector at a time.
/// Multiplying position by position would run a loop of `channels` elements per iteration, and at
/// the small channel counts a 1-D or early-conv spatial dropout sees that is far too short to
/// amortize anything. Repeating the factors up to about a kilobyte turns the pass into long flat
/// multiplies at every channel count, with one body and no special case
///
/// Each output element depends only on its own input element and its channel's scalar (no
/// reduction), so the `parallel` flag never changes the result. The per-channel value is broadcast
/// by the tile, so no input-sized mask is ever built or stored
///
/// Gives the same result as the explicit `t * broadcast(mask) * scale`: the mask is binary, so
/// `(x * 1) * scale == x * (1 * scale)` and `(x * 0) * scale == x * (0 * scale)` both hold
/// exactly
fn spatial_dropout_scale(
    t: &Tensor,
    channel_mask: &[f32],
    rate: f32,
    parallel_threshold: usize,
) -> Tensor {
    let channels = t.shape()[t.ndim() - 1].max(1);
    let batch = (channel_mask.len() / channels).max(1);
    let total = t.len();
    let item = total / batch;
    let positions = (item / channels).max(1);
    let scale = 1.0 / (1.0 - rate);

    // Contiguous data maps positions onto fixed slices; standardize a non-contiguous view
    let t_std = t.as_standard_layout();
    let src = t_std.as_slice().unwrap();

    let mut out = Tensor::zeros(t.raw_dim());
    let dst = out.as_slice_mut().unwrap();

    // Repetitions of the channel vector per tile. Snapping down to a divisor of `positions` keeps
    // every tile-sized block inside one batch item, so the whole tensor can be walked in a single
    // parallel pass instead of one rayon launch per item
    let mut reps = (1024 / channels).clamp(1, positions);
    while reps > 1 && !positions.is_multiple_of(reps) {
        reps -= 1;
    }
    let tile_len = reps * channels;

    // One tile per batch item, laid out `[batch, tile_len]`; a few hundred KB at most, and it turns
    // the pass below into a flat multiply that never looks up a channel index
    let mut tiles = Vec::with_capacity(batch * tile_len);
    for b in 0..batch {
        let mask = &channel_mask[b * channels..(b + 1) * channels];
        for _ in 0..reps {
            tiles.extend(mask.iter().map(|&m| m * scale));
        }
    }

    let apply = |(ci, (o, x)): (usize, (&mut [f32], &[f32]))| {
        let b = ci * tile_len / item;
        let tile = &tiles[b * tile_len..(b + 1) * tile_len];
        for ((o_elem, &x_elem), &f) in o.iter_mut().zip(x).zip(tile) {
            *o_elem = x_elem * f;
        }
    };
    if total >= parallel_threshold {
        dst.par_chunks_mut(tile_len)
            .zip(src.par_chunks(tile_len))
            .enumerate()
            .for_each(apply);
    } else {
        dst.chunks_mut(tile_len)
            .zip(src.chunks(tile_len))
            .enumerate()
            .for_each(apply);
    }
    out
}

/// Backward pass shared by the spatial-dropout layers
///
/// Applies the stored per-channel mask to the gradient with [`spatial_dropout_scale`], without
/// rebuilding a full-size mask
///
/// Mirrors the early-return structure of [`dropout_backward`] (inference / `rate == 0`
/// pass-through, `rate == 1` zeros, missing-mask error), but the stored `mask` is the small
/// `[batch, channels]` per-channel mask rather than a full-shape one
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
// `Dropout` lives in a `dropout` submodule beside the spatial-dropout modules; the repeated
// name is the intended file layout
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

    /// The mask lines up with the trailing channel axis
    ///
    /// Four channels with the mask `[1, 0, 1, 0]` over an all-ones input: every position must come
    /// back as `[2, 0, 2, 0]` at `rate = 0.5`. A channel axis read anywhere but last would drop
    /// whole positions instead of whole channels, and a mask off by one would shift the pattern -
    /// both show up immediately against a hand-written expectation
    #[test]
    fn spatial_dropout_scale_drops_whole_channels() {
        // [batch = 1, positions = 3, channels = 4]
        let t = Tensor::from_shape_vec(IxDyn(&[1, 3, 4]), vec![1.0f32; 12]).unwrap();
        let channel_mask = [1.0f32, 0.0, 1.0, 0.0];

        let out = spatial_dropout_scale(&t, &channel_mask, 0.5, usize::MAX);

        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
        );
    }

    /// Each batch item uses its own row of the mask
    #[test]
    fn spatial_dropout_scale_masks_are_per_batch_item() {
        // [batch = 2, positions = 1, channels = 2]
        let t = Tensor::from_shape_vec(IxDyn(&[2, 1, 2]), vec![1.0f32; 4]).unwrap();
        // Item 0 keeps channel 0, item 1 keeps channel 1
        let channel_mask = [1.0f32, 0.0, 0.0, 1.0];

        let out = spatial_dropout_scale(&t, &channel_mask, 0.5, usize::MAX);

        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 0.0, 0.0, 2.0]
        );
    }

    /// Each output element of the per-channel scale depends only on its own input element and
    /// its channel's scalar (no reduction), so forcing serial vs parallel must produce the same
    /// result: the gate is a pure performance knob. Covers channel counts above and below the
    /// tile width, and position counts that do not divide the tile
    #[test]
    fn spatial_dropout_scale_parallel_flag_invariant() {
        for &(batch, positions, channels) in &[
            (1usize, 5usize, 7usize),
            (2, 256, 64),
            (3, 4096, 1),
            (4, 17, 512),
            (1, 4093, 3),
        ] {
            let total = batch * positions * channels;
            let t = Tensor::from_shape_vec(
                IxDyn(&[batch, positions, channels]),
                (0..total).map(|i| (i as f32 * 0.013).sin()).collect(),
            )
            .unwrap();
            // A mix of kept (1.0) and dropped (0.0) channels, differing per batch item
            let channel_mask: Vec<f32> = (0..batch * channels)
                .map(|i| (i % 3 != 0) as u8 as f32)
                .collect();
            let rate = 0.25f32;

            // total >= gate forces parallel; usize::MAX forces serial
            let serial = spatial_dropout_scale(&t, &channel_mask, rate, usize::MAX);
            let parallel = spatial_dropout_scale(&t, &channel_mask, rate, 0);
            assert_eq!(
                serial.as_slice().unwrap(),
                parallel.as_slice().unwrap(),
                "parallel flag changed the bits at [{batch}, {positions}, {channels}]"
            );

            // And the same result as the explicit `t * broadcast(mask) * scale` it replaces
            let scale = 1.0 / (1.0 - rate);
            let item = positions * channels;
            let mut expected = vec![0.0f32; total];
            for (i, e) in expected.iter_mut().enumerate() {
                let m = channel_mask[(i / item) * channels + i % channels];
                let x = t.as_slice().unwrap()[i];
                *e = (x * m) * scale;
            }
            assert_eq!(
                serial.as_slice().unwrap(),
                expected.as_slice(),
                "differs from the explicit two-step form at [{batch}, {positions}, {channels}]"
            );
        }
    }
}
