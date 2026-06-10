//! Dimension-generic convolution engine shared by `Conv1D`, `Conv2D`, and `Conv3D`.
//!
//! A plain convolution is the same operation at every rank — only the number of spatial axes
//! changes — so one implementation, generic over the spatial rank `R = ndim - 2`, serves all
//! three layers. The layer wrappers keep their public API, weight storage, activation, and caches;
//! they delegate just the numeric forward/backward to [`conv_forward`] and [`conv_backward`].
//!
//! # Conventions (match the previous per-rank code exactly)
//!
//! - `Valid` output: `(in - k) / stride + 1`; `Same` output: `ceil(in / stride)`.
//! - `Same` padding splits the total evenly with the extra cell on the trailing edge
//!   (`pad_before = pad_total / 2`).
//! - Cross-correlation (no kernel flip); the bias is added last so results are bit-compatible with
//!   the previous implementations.
//!
//! # Layout
//!
//! Tensors are `[batch, channels, spatial...]`; weights are flat row-major `[F, Cin, k...]`. Work is
//! parallelized over `batch * filters` (forward), `filters` (weight/bias gradients), and
//! `batch * channels` (input gradient) — each task writes a disjoint output region.

use super::PaddingType;
use crate::neural_network::Tensor;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

/// Workload (output-element count) at or above which an engine pass runs in parallel.
const CONV_PARALLEL_THRESHOLD: usize = 10_000;

/// Analytic gradients returned by [`conv_backward`].
pub(super) struct ConvGradients {
    /// Weight gradient, flat row-major `[F, Cin, k...]` (reshape to the layer's weight array).
    pub weight_grad: Vec<f32>,
    /// Bias gradient, one value per filter `[F]`.
    pub bias_grad: Vec<f32>,
    /// Input gradient, shape `[batch, Cin, spatial...]`.
    pub input_grad: Tensor,
}

/// Row-major (C-order) strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Advances a multi-index `idx` (row-major, last axis fastest) within `dims`; `false` when it wraps.
#[inline]
fn increment_index(idx: &mut [usize], dims: &[usize]) -> bool {
    for k in (0..idx.len()).rev() {
        idx[k] += 1;
        if idx[k] < dims[k] {
            return true;
        }
        idx[k] = 0;
    }
    false
}

/// Runs `f` over `0..n`, in parallel when `parallel`, preserving index order.
fn map_indexed<R, F>(n: usize, parallel: bool, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if parallel {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Per-spatial-axis output size, leading padding, and padded size for the given padding mode.
fn conv_geometry(
    sp: &[usize],
    k_dims: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let r = sp.len();
    match padding {
        PaddingType::Valid => {
            let out_sp: Vec<usize> = (0..r)
                .map(|d| (sp[d] - k_dims[d]) / strides[d] + 1)
                .collect();
            (out_sp, vec![0; r], sp.to_vec())
        }
        PaddingType::Same => {
            let out_sp: Vec<usize> = (0..r).map(|d| sp[d].div_ceil(strides[d])).collect();
            let pad_before: Vec<usize> = (0..r)
                .map(|d| (((out_sp[d] - 1) * strides[d] + k_dims[d]).saturating_sub(sp[d])) / 2)
                .collect();
            let padded_sp: Vec<usize> = (0..r)
                .map(|d| ((out_sp[d] - 1) * strides[d] + k_dims[d]).max(sp[d]))
                .collect();
            (out_sp, pad_before, padded_sp)
        }
    }
}

/// Builds a zero-padded copy of the flat `[bc, sp...]` channel-plane data.
fn build_padded(
    in_flat: &[f32],
    bc: usize,
    sp: &[usize],
    padded_sp: &[usize],
    pad_before: &[usize],
) -> Vec<f32> {
    let r = sp.len();
    let in_plane: usize = sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(padded_sp);
    let mut out = vec![0.0f32; bc * padded_plane];

    for chan in 0..bc {
        let in_base = chan * in_plane;
        let pad_base = chan * padded_plane;
        let mut si = vec![0usize; r];
        let mut si_flat = 0usize;
        loop {
            let mut pidx = 0usize;
            for d in 0..r {
                pidx += (si[d] + pad_before[d]) * padded_strides[d];
            }
            out[pad_base + pidx] = in_flat[in_base + si_flat];
            si_flat += 1;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
    }
    out
}

/// Forward convolution. `weight_shape` is `[F, Cin, k...]`; `bias` is `[F]`; `strides` has length `R`.
pub(super) fn conv_forward(
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    strides: &[usize],
    padding: PaddingType,
) -> Tensor {
    let in_shape = input.shape();
    let (batch, cin) = (in_shape[0], in_shape[1]);
    let sp = &in_shape[2..];
    let r = sp.len();
    let filters = weight_shape[0];
    let k_dims = &weight_shape[2..];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding);
    let out_plane: usize = out_sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(&padded_sp);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let padded_storage = if padded_sp.as_slice() != sp {
        Some(build_padded(
            in_flat,
            batch * cin,
            sp,
            &padded_sp,
            &pad_before,
        ))
    } else {
        None
    };
    let padded: &[f32] = padded_storage.as_deref().unwrap_or(in_flat);

    // One [out_plane] tile per (batch, filter) pair.
    let process_bf = |bf: usize| -> Vec<f32> {
        let b = bf / filters;
        let f = bf % filters;
        let mut tile = vec![0.0f32; out_plane];
        let mut o = vec![0usize; r];
        let mut kk = vec![0usize; r];
        let mut o_flat = 0usize;
        loop {
            let mut sum = 0.0f32;
            for c in 0..cin {
                let w_base = (f * cin + c) * k_plane;
                let p_base = (b * cin + c) * padded_plane;
                kk.iter_mut().for_each(|x| *x = 0);
                let mut kk_flat = 0usize;
                loop {
                    let mut pidx = 0usize;
                    for d in 0..r {
                        pidx += (o[d] * strides[d] + kk[d]) * padded_strides[d];
                    }
                    sum += padded[p_base + pidx] * weights[w_base + kk_flat];
                    kk_flat += 1;
                    if !increment_index(&mut kk, k_dims) {
                        break;
                    }
                }
            }
            // Bias added last to match the previous per-rank accumulation order.
            tile[o_flat] = sum + bias[f];
            o_flat += 1;
            if !increment_index(&mut o, &out_sp) {
                break;
            }
        }
        tile
    };

    let parallel = batch * filters * out_plane >= CONV_PARALLEL_THRESHOLD;
    let tiles = map_indexed(batch * filters, parallel, process_bf);

    let mut out_flat = Vec::with_capacity(batch * filters * out_plane);
    for tile in tiles {
        out_flat.extend(tile);
    }
    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.push(filters);
    out_shape.extend_from_slice(&out_sp);
    ArrayD::from_shape_vec(IxDyn(&out_shape), out_flat).expect("conv output length matches shape")
}

/// Backward convolution. `input` is the original (unpadded) forward input; `grad_output` is the
/// gradient w.r.t. the convolution output (i.e. after the activation backward).
pub(super) fn conv_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> ConvGradients {
    let in_shape = input.shape();
    let (batch, cin) = (in_shape[0], in_shape[1]);
    let sp = &in_shape[2..];
    let r = sp.len();
    let filters = weight_shape[0];
    let k_dims = &weight_shape[2..];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding);
    let out_plane: usize = out_sp.iter().product();
    let in_plane: usize = sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(&padded_sp);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let padded_storage = if padded_sp.as_slice() != sp {
        Some(build_padded(
            in_flat,
            batch * cin,
            sp,
            &padded_sp,
            &pad_before,
        ))
    } else {
        None
    };
    let padded: &[f32] = padded_storage.as_deref().unwrap_or(in_flat);

    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // --- weight and bias gradients: one filter per task (reduces over batch and output). ---
    let per_f = |f: usize| -> (Vec<f32>, f32) {
        let mut wg = vec![0.0f32; cin * k_plane];
        let mut bias_sum = 0.0f32;
        let mut o = vec![0usize; r];
        let mut kk = vec![0usize; r];
        for b in 0..batch {
            let g_base = (b * filters + f) * out_plane;
            o.iter_mut().for_each(|x| *x = 0);
            let mut o_flat = 0usize;
            loop {
                let g = grad_flat[g_base + o_flat];
                bias_sum += g;
                for c in 0..cin {
                    let p_base = (b * cin + c) * padded_plane;
                    let w_base = c * k_plane;
                    kk.iter_mut().for_each(|x| *x = 0);
                    let mut kk_flat = 0usize;
                    loop {
                        let mut pidx = 0usize;
                        for d in 0..r {
                            pidx += (o[d] * strides[d] + kk[d]) * padded_strides[d];
                        }
                        wg[w_base + kk_flat] += g * padded[p_base + pidx];
                        kk_flat += 1;
                        if !increment_index(&mut kk, k_dims) {
                            break;
                        }
                    }
                }
                o_flat += 1;
                if !increment_index(&mut o, &out_sp) {
                    break;
                }
            }
        }
        (wg, bias_sum)
    };

    let parallel_w = batch * out_plane * cin * k_plane >= CONV_PARALLEL_THRESHOLD;
    let f_results = map_indexed(filters, parallel_w, per_f);
    let mut weight_grad = Vec::with_capacity(filters * cin * k_plane);
    let mut bias_grad = Vec::with_capacity(filters);
    for (wg, b) in f_results {
        weight_grad.extend(wg);
        bias_grad.push(b);
    }

    // --- input gradient: one channel-plane per (batch, channel), gathered per original position. ---
    let process_bc = |bc: usize| -> Vec<f32> {
        let b = bc / cin;
        let c = bc % cin;
        let mut plane = vec![0.0f32; in_plane];
        let mut si = vec![0usize; r];
        let mut kk = vec![0usize; r];
        let mut si_flat = 0usize;
        loop {
            let mut sum = 0.0f32;
            for f in 0..filters {
                let g_base = (b * filters + f) * out_plane;
                let w_base = (f * cin + c) * k_plane;
                kk.iter_mut().for_each(|x| *x = 0);
                let mut kk_flat = 0usize;
                loop {
                    // Map padded position (si + pad_before) back to the output position it came from.
                    let mut valid = true;
                    let mut o_flat = 0usize;
                    for d in 0..r {
                        let pd = si[d] + pad_before[d];
                        if pd < kk[d] {
                            valid = false;
                            break;
                        }
                        let diff = pd - kk[d];
                        if !diff.is_multiple_of(strides[d]) {
                            valid = false;
                            break;
                        }
                        let od = diff / strides[d];
                        if od >= out_sp[d] {
                            valid = false;
                            break;
                        }
                        o_flat = o_flat * out_sp[d] + od;
                    }
                    if valid {
                        sum += weights[w_base + kk_flat] * grad_flat[g_base + o_flat];
                    }
                    kk_flat += 1;
                    if !increment_index(&mut kk, k_dims) {
                        break;
                    }
                }
            }
            plane[si_flat] = sum;
            si_flat += 1;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
        plane
    };

    let parallel_ig = batch * cin * in_plane >= CONV_PARALLEL_THRESHOLD;
    let ig_planes = map_indexed(batch * cin, parallel_ig, process_bc);
    let mut in_grad_flat = Vec::with_capacity(batch * cin * in_plane);
    for plane in ig_planes {
        in_grad_flat.extend(plane);
    }
    let mut ig_shape = Vec::with_capacity(2 + r);
    ig_shape.push(batch);
    ig_shape.push(cin);
    ig_shape.extend_from_slice(sp);
    let input_grad =
        ArrayD::from_shape_vec(IxDyn(&ig_shape), in_grad_flat).expect("input grad matches shape");

    ConvGradients {
        weight_grad,
        bias_grad,
        input_grad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // row_major_strides
    // -----------------------------------------------------------------------

    /// 3-D shape [2, 3, 4]:
    ///   strides[2] = 1
    ///   strides[1] = 1 * 4 = 4
    ///   strides[0] = 4 * 3 = 12
    /// Expected: [12, 4, 1]
    #[test]
    fn test_row_major_strides_3d() {
        let got = row_major_strides(&[2, 3, 4]);
        assert_eq!(got, vec![12, 4, 1]);
    }

    /// 1-D shape [5]: only one element so strides[0] = 1. Expected: [1]
    #[test]
    fn test_row_major_strides_1d() {
        let got = row_major_strides(&[5]);
        assert_eq!(got, vec![1]);
    }

    /// Empty shape: no dimensions, no strides. Expected: []
    #[test]
    fn test_row_major_strides_empty() {
        let got = row_major_strides(&[]);
        assert_eq!(got, Vec::<usize>::new());
    }

    // -----------------------------------------------------------------------
    // increment_index
    // -----------------------------------------------------------------------

    /// dims = [2, 3] (last axis fastest).
    /// Walk: [0,0] -> [0,1] -> [0,2] -> [1,0] -> [1,1] -> [1,2] -> false (wraps to [0,0]).
    #[test]
    fn test_increment_index_2d() {
        let dims = [2usize, 3];
        let mut idx = vec![0usize, 0];

        // Step 1: [0,0] -> [0,1], returns true
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 1]);

        // Step 2: [0,1] -> [0,2], returns true
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 2]);

        // Step 3: [0,2] -> [1,0] (last-axis overflow carries), returns true
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 0]);

        // Step 4: [1,0] -> [1,1], returns true
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 1]);

        // Step 5: [1,1] -> [1,2], returns true
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 2]);

        // Step 6: [1,2] -> overflow on both axes, returns false; index wraps to [0,0]
        assert!(!increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 0]);
    }

    /// dims = [3] (single axis).
    /// Walk: 0 -> 1 -> 2 -> false.
    #[test]
    fn test_increment_index_1d() {
        let dims = [3usize];
        let mut idx = vec![0usize];

        // 0 -> 1
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1]);

        // 1 -> 2
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![2]);

        // 2 -> overflow, returns false
        assert!(!increment_index(&mut idx, &dims));
    }

    // -----------------------------------------------------------------------
    // conv_geometry – Valid padding
    // -----------------------------------------------------------------------

    /// sp=[5], k=[3], stride=[1], Valid:
    ///   out = (5 - 3) / 1 + 1 = 3
    ///   pad_before = [0]
    ///   padded_sp = [5]  (same as input, no padding needed)
    #[test]
    fn test_conv_geometry_valid_1d() {
        let (out_sp, pad_before, padded_sp) = conv_geometry(&[5], &[3], &[1], PaddingType::Valid);
        assert_eq!(out_sp, vec![3]);
        assert_eq!(pad_before, vec![0]);
        assert_eq!(padded_sp, vec![5]);
    }

    // -----------------------------------------------------------------------
    // conv_geometry – Same padding, 1-D
    // -----------------------------------------------------------------------

    /// sp=[7], k=[3], stride=[2], Same:
    ///   out = ceil(7 / 2) = 4
    ///   total_pad = (out-1)*stride + k - sp = (4-1)*2 + 3 - 7 = 6 + 3 - 7 = 2
    ///   pad_before = 2 / 2 = 1
    ///   padded_sp = max((4-1)*2 + 3, 7) = max(9, 7) = 9
    #[test]
    fn test_conv_geometry_same_1d() {
        let (out_sp, pad_before, padded_sp) = conv_geometry(&[7], &[3], &[2], PaddingType::Same);
        assert_eq!(out_sp, vec![4]);
        assert_eq!(pad_before, vec![1]);
        assert_eq!(padded_sp, vec![9]);
    }

    // -----------------------------------------------------------------------
    // conv_geometry – Same padding, 2-D
    // -----------------------------------------------------------------------

    /// sp=[4,4], k=[3,3], stride=[1,1], Same:
    ///   Per axis (identical for both):
    ///     out_d = ceil(4 / 1) = 4
    ///     total_pad = (4-1)*1 + 3 - 4 = 3 + 3 - 4 = 2
    ///     pad_before_d = 2 / 2 = 1
    ///     padded_d = max((4-1)*1 + 3, 4) = max(6, 4) = 6
    #[test]
    fn test_conv_geometry_same_2d() {
        let (out_sp, pad_before, padded_sp) =
            conv_geometry(&[4, 4], &[3, 3], &[1, 1], PaddingType::Same);
        assert_eq!(out_sp, vec![4, 4]);
        assert_eq!(pad_before, vec![1, 1]);
        assert_eq!(padded_sp, vec![6, 6]);
    }

    // -----------------------------------------------------------------------
    // build_padded
    // -----------------------------------------------------------------------

    /// bc=1, sp=[2,2], in_flat=[1,2,3,4] (row-major), padded_sp=[4,4], pad_before=[1,1].
    ///
    /// padded_strides for [4,4] = [4, 1].
    /// The 2×2 block is placed at (row+1, col+1) in the 4×4 padded grid:
    ///   in[0,0]=1 -> padded[1*4 + 1] = padded[5]  = 1.0
    ///   in[0,1]=2 -> padded[1*4 + 2] = padded[6]  = 2.0
    ///   in[1,0]=3 -> padded[2*4 + 1] = padded[9]  = 3.0
    ///   in[1,1]=4 -> padded[2*4 + 2] = padded[10] = 4.0
    /// All other 12 positions must be 0.0.
    #[test]
    fn test_build_padded_2x2_into_4x4() {
        let in_flat = [1.0f32, 2.0, 3.0, 4.0];
        let got = build_padded(&in_flat, 1, &[2, 2], &[4, 4], &[1, 1]);

        assert_eq!(got.len(), 16, "padded buffer should have 16 elements");

        // Positions that should hold data
        assert_eq!(got[5], 1.0, "padded[5] should be in[0,0]=1.0");
        assert_eq!(got[6], 2.0, "padded[6] should be in[0,1]=2.0");
        assert_eq!(got[9], 3.0, "padded[9] should be in[1,0]=3.0");
        assert_eq!(got[10], 4.0, "padded[10] should be in[1,1]=4.0");

        // All border positions must be zero
        let non_zero_positions = [5usize, 6, 9, 10];
        for (i, &val) in got.iter().enumerate() {
            if !non_zero_positions.contains(&i) {
                assert_eq!(val, 0.0, "padded[{i}] should be 0.0 (border), got {val}");
            }
        }
    }

    /// Two batch-channels (bc=2), same 2×2 content in each channel, placed at
    /// pad_before=[1,1] inside a 4×4 padded plane.  Each channel occupies a
    /// disjoint 16-element slice.
    #[test]
    fn test_build_padded_two_channels() {
        let in_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let got = build_padded(&in_flat, 2, &[2, 2], &[4, 4], &[1, 1]);

        assert_eq!(got.len(), 32);

        // Channel 0 (offset 0)
        assert_eq!(got[5], 1.0);
        assert_eq!(got[6], 2.0);
        assert_eq!(got[9], 3.0);
        assert_eq!(got[10], 4.0);

        // Channel 1 (offset 16)
        assert_eq!(got[16 + 5], 5.0);
        assert_eq!(got[16 + 6], 6.0);
        assert_eq!(got[16 + 9], 7.0);
        assert_eq!(got[16 + 10], 8.0);
    }

    /// 1-D case: bc=1, sp=[3], in_flat=[10,20,30], padded_sp=[5], pad_before=[1].
    ///
    /// padded_strides for [5] = [1].
    /// Placements: in[0]=10 -> padded[0+1]=padded[1]; in[1]=20 -> padded[2]; in[2]=30 -> padded[3].
    /// padded[0] and padded[4] must be 0.
    #[test]
    fn test_build_padded_1d() {
        let in_flat = [10.0f32, 20.0, 30.0];
        let got = build_padded(&in_flat, 1, &[3], &[5], &[1]);

        assert_eq!(got.len(), 5);
        assert_eq!(got[0], 0.0, "leading pad must be 0");
        assert_eq!(got[1], 10.0);
        assert_eq!(got[2], 20.0);
        assert_eq!(got[3], 30.0);
        assert_eq!(got[4], 0.0, "trailing pad must be 0");
    }

    /// Zero padding (pad_before all zeros): output must be identical to input.
    #[test]
    fn test_build_padded_no_padding() {
        let in_flat = [5.0f32, 6.0, 7.0, 8.0];
        let got = build_padded(&in_flat, 1, &[2, 2], &[2, 2], &[0, 0]);
        assert_eq!(got, vec![5.0f32, 6.0, 7.0, 8.0]);
    }
}
