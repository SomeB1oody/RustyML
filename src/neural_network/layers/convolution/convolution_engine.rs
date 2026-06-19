//! Dimension-generic convolution engine shared by `Conv1D`, `Conv2D`, and `Conv3D`
//!
//! A plain convolution is the same operation at every rank; only the number of spatial axes
//! changes. One implementation, generic over the spatial rank `R = ndim - 2`, serves all 3
//! layers. The layer wrappers keep their public API, weight storage, activation, and caches; they
//! delegate just the numeric forward/backward to [`conv_forward`] and [`conv_backward`]
//!
//! # Conventions
//!
//! - `Valid` output: `(in - k) / stride + 1`; `Same` output: `ceil(in / stride)`
//! - `Same` padding splits the total evenly with the extra cell on the trailing edge
//!   (`pad_before = pad_total / 2`)
//! - Cross-correlation (no kernel flip); the bias is added last
//!
//! # Layout
//!
//! Tensors are `[batch, channels, spatial...]`; weights are flat row-major `[F, Cin, k...]`. The
//! forward pass parallelizes over `(batch item, output-position block)` tasks. Splitting the
//! output positions lets a single large image use every core even at `batch == 1`, with each task
//! building its own im2col block and GEMM, writing a disjoint output region. The backward pass
//! parallelizes over batch items (their weight/bias partials are reduced in batch order, so
//! results do not depend on the thread count) and routes its two GEMMs through
//! [`gemm_par_switch`](crate::math::matmul::gemm_par_switch): the per-item GEMMs stay parallel
//! while the batch fan is too short to fill the pool, and switch to serial once the batch alone
//! fills it (so the batch tasks do not each fork rayon inside a GEMM)

use super::PaddingType;
use crate::error::Error;
use crate::math::matmul::{gemm_par_auto, gemm_par_switch};
use crate::neural_network::Tensor;
use ndarray::{Array2, Array3, ArrayD, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use rayon::prelude::*;

tunable_gate! {
    /// Minimum estimated GEMM FLOPs (`2 * batch * F * out_plane * Cin*k`) at or above which an engine
    /// pass runs in parallel
    ///
    /// Counting FLOPs rather than output elements keeps the gate meaningful across kernel sizes and
    /// channel counts (an output-element count would rate a `7x7x512` and a `3x3x3` convolution
    /// identically). The measured crossover bracket at batch == 1 is 2.1M-8.3M FLOPs
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) CONV_PARALLEL_MIN_FLOPS => conv_parallel_min_flops / set_conv_parallel_min_flops = 4_000_000
}

/// Minimum output-position columns per forward task
///
/// Each task's GEMM re-packs the weight matrix, so blocks need enough columns to amortize that
const CONV_MIN_CHUNK_COLS: usize = 64;

/// Analytic gradients returned by [`conv_backward`]
pub(super) struct ConvGradients {
    /// Weight gradient, flat row-major `[F, Cin, k...]` (reshape to the layer's weight array)
    pub weight_grad: Vec<f32>,
    /// Bias gradient, one value per filter `[F]`
    pub bias_grad: Vec<f32>,
    /// Input gradient, shape `[batch, Cin, spatial...]`
    pub input_grad: Tensor,
}

/// Row-major (C-order) strides for `shape`
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Advances a multi-index `idx` (row-major, last axis fastest) within `dims`; `false` when it wraps
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

/// Runs `f` over `0..n`, in parallel when `parallel`, preserving index order
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

/// Geometry a convolution pass needs: output spatial sizes, per-axis leading padding, and padded
/// spatial sizes, as `(out_sp, pad_before, padded_sp)`
type ConvGeometry = (Vec<usize>, Vec<usize>, Vec<usize>);

fn conv_geometry(
    sp: &[usize],
    k_dims: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> Result<ConvGeometry, Error> {
    let r = sp.len();
    match padding {
        PaddingType::Valid => {
            if let Some(d) = (0..r).find(|&d| sp[d] < k_dims[d]) {
                return Err(Error::invalid_input(format!(
                    "Valid-padding convolution requires every input spatial dimension to be at \
                     least the kernel size: axis {d} has input size {} < kernel size {}",
                    sp[d], k_dims[d]
                )));
            }
            let out_sp: Vec<usize> = (0..r)
                .map(|d| (sp[d] - k_dims[d]) / strides[d] + 1)
                .collect();
            Ok((out_sp, vec![0; r], sp.to_vec()))
        }
        PaddingType::Same => {
            let out_sp: Vec<usize> = (0..r).map(|d| sp[d].div_ceil(strides[d])).collect();
            let pad_before: Vec<usize> = (0..r)
                .map(|d| (((out_sp[d] - 1) * strides[d] + k_dims[d]).saturating_sub(sp[d])) / 2)
                .collect();
            let padded_sp: Vec<usize> = (0..r)
                .map(|d| ((out_sp[d] - 1) * strides[d] + k_dims[d]).max(sp[d]))
                .collect();
            Ok((out_sp, pad_before, padded_sp))
        }
    }
}

/// Builds a zero-padded copy of the flat `[bc, sp...]` channel-plane data
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

/// Inverse of [`build_padded`]: gathers the unpadded `[bc, sp...]` region out of a padded buffer
///
/// Used to crop the input-gradient back to the original spatial size after col2im
fn crop_padded(
    padded: &[f32],
    bc: usize,
    sp: &[usize],
    padded_sp: &[usize],
    pad_before: &[usize],
) -> Vec<f32> {
    let r = sp.len();
    let in_plane: usize = sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(padded_sp);
    let mut out = vec![0.0f32; bc * in_plane];

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
            out[in_base + si_flat] = padded[pad_base + pidx];
            si_flat += 1;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
    }
    out
}

/// Flat padded-plane offsets for im2col/col2im, laid out `[k_plane, out_plane]`
///
/// `offsets[kk * out_plane + o]` is the index, within a single padded channel-plane, that output
/// position `o` reads for kernel tap `kk`. Independent of batch and channel, so it is computed once
/// and reused for every im2col gather and every col2im scatter
fn im2col_offsets(
    out_sp: &[usize],
    k_dims: &[usize],
    strides: &[usize],
    padded_strides: &[usize],
) -> Vec<usize> {
    let r = out_sp.len();
    let out_plane: usize = out_sp.iter().product();
    let k_plane: usize = k_dims.iter().product();
    let mut offsets = vec![0usize; k_plane * out_plane];

    let mut o = vec![0usize; r];
    let mut o_flat = 0usize;
    loop {
        let mut kk = vec![0usize; r];
        let mut kk_flat = 0usize;
        loop {
            let mut pidx = 0usize;
            for d in 0..r {
                pidx += (o[d] * strides[d] + kk[d]) * padded_strides[d];
            }
            offsets[kk_flat * out_plane + o_flat] = pidx;
            kk_flat += 1;
            if !increment_index(&mut kk, k_dims) {
                break;
            }
        }
        o_flat += 1;
        if !increment_index(&mut o, out_sp) {
            break;
        }
    }
    offsets
}

/// Per-pass im2col inputs shared by every task: the padded data and the gather geometry
struct ColContext<'a> {
    /// Flat zero-padded input `[batch * Cin, padded_plane]`
    padded: &'a [f32],
    /// Input channels
    cin: usize,
    /// Elements per padded channel-plane
    padded_plane: usize,
    /// Kernel taps per channel
    k_plane: usize,
    /// Output positions per channel-plane
    out_plane: usize,
    /// `[k_plane, out_plane]` gather offsets from [`im2col_offsets`]
    offsets: &'a [usize],
}

/// im2col for one batch item, restricted to the output positions `[c0, c1)`
///
/// Gathers the padded data into a `[Cin*k_plane, c1-c0]` row-major matrix whose rows align with
/// the flat weight matrix `[F, Cin*k_plane]`. Pass the full `[0, out_plane)` range for the whole
/// item; a sub-range is one forward task's column block
fn build_col_range(ctx: &ColContext, b: usize, c0: usize, c1: usize) -> Vec<f32> {
    let cols = c1 - c0;
    let mut col = vec![0.0f32; ctx.cin * ctx.k_plane * cols];
    let b_base = b * ctx.cin * ctx.padded_plane;
    for c in 0..ctx.cin {
        let pc = b_base + c * ctx.padded_plane;
        for kk in 0..ctx.k_plane {
            let krow = (c * ctx.k_plane + kk) * cols;
            let off = kk * ctx.out_plane;
            for (i, o) in (c0..c1).enumerate() {
                col[krow + i] = ctx.padded[pc + ctx.offsets[off + o]];
            }
        }
    }
    col
}

/// Forward convolution; `weight_shape` is `[F, Cin, k...]`, `bias` is `[F]`, `strides` has length `R`
pub(super) fn conv_forward(
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    strides: &[usize],
    padding: PaddingType,
) -> Result<Tensor, Error> {
    conv_forward_impl(input, weights, weight_shape, bias, strides, padding, None)
}

/// `conv_forward` with an optional override of the parallel/serial gate decision, so a bench can
/// time both paths on either side of the gate
///
/// Reachable outside the crate only through `bench_internals`
pub fn conv_forward_impl(
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    strides: &[usize],
    padding: PaddingType,
    force_parallel: Option<bool>,
) -> Result<Tensor, Error> {
    let in_shape = input.shape();
    let (batch, cin) = (in_shape[0], in_shape[1]);
    let sp = &in_shape[2..];
    let r = sp.len();
    let filters = weight_shape[0];
    let k_dims = &weight_shape[2..];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding)?;
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

    // im2col + gemm
    let k_total = cin * k_plane;
    let offsets = im2col_offsets(&out_sp, k_dims, strides, &padded_strides);
    let w_mat = ArrayView2::from_shape((filters, k_total), weights)
        .expect("weights length matches [F, Cin*k]");

    // One task per (batch item, output-position block)
    let ctx = ColContext {
        padded,
        cin,
        padded_plane,
        k_plane,
        out_plane,
        offsets: &offsets,
    };
    let fill_block = |b: usize, c0: usize, mut blk: ArrayViewMut2<f32>| {
        let cols = blk.ncols();
        let col = build_col_range(&ctx, b, c0, c0 + cols);
        let col_mat = ArrayView2::from_shape((k_total, cols), &col)
            .expect("col block length matches [Cin*k, cols]");
        let mut prod = w_mat.dot(&col_mat); // [F, cols]
        // Bias added last
        for (f, mut row) in prod.outer_iter_mut().enumerate() {
            row += bias[f];
        }
        blk.assign(&prod);
    };

    let gemm_flops = 2usize
        .saturating_mul(batch)
        .saturating_mul(filters)
        .saturating_mul(out_plane)
        .saturating_mul(k_total);
    let parallel = force_parallel.unwrap_or(gemm_flops >= conv_parallel_min_flops());

    let mut out3 = Array3::<f32>::zeros((batch, filters, out_plane));
    if parallel {
        // Enough blocks to feed every thread once the batch alone cannot
        let chunks_per_item = rayon::current_num_threads().div_ceil(batch);
        let chunk_cols = out_plane.div_ceil(chunks_per_item).max(CONV_MIN_CHUNK_COLS);
        out3.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut out_b)| {
                out_b
                    .axis_chunks_iter_mut(Axis(1), chunk_cols)
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(ci, blk)| fill_block(b, ci * chunk_cols, blk));
            });
    } else {
        for (b, mut out_b) in out3.axis_iter_mut(Axis(0)).enumerate() {
            fill_block(b, 0, out_b.view_mut());
        }
    }

    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.push(filters);
    out_shape.extend_from_slice(&out_sp);
    Ok(out3
        .into_shape_with_order(IxDyn(&out_shape))
        .expect("conv output length matches shape"))
}

/// Backward convolution; `input` is the original (unpadded) forward input, `grad_output` is the
/// gradient w.r.t. the convolution output (i.e. after the activation backward)
pub(super) fn conv_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> Result<ConvGradients, Error> {
    let in_shape = input.shape();
    let (batch, cin) = (in_shape[0], in_shape[1]);
    let sp = &in_shape[2..];
    let r = sp.len();
    let filters = weight_shape[0];
    let k_dims = &weight_shape[2..];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding)?;
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

    // im2col + gemm
    let k_total = cin * k_plane;
    let offsets = im2col_offsets(&out_sp, k_dims, strides, &padded_strides);
    let w_mat = ArrayView2::from_shape((filters, k_total), weights)
        .expect("weights length matches [F, Cin*k]");

    let ctx = ColContext {
        padded,
        cin,
        padded_plane,
        k_plane,
        out_plane,
        offsets: &offsets,
    };
    // `serial_gemm` forces the two per-item products onto the serial `gemm` path (still the `gemm`
    // crate, just one thread), so a per-batch task does not fork rayon again inside its GEMM
    let process_b = |b: usize, serial_gemm: bool| -> (Array2<f32>, Vec<f32>, Vec<f32>) {
        let col = build_col_range(&ctx, b, 0, out_plane);
        let col_mat = ArrayView2::from_shape((k_total, out_plane), &col)
            .expect("col length matches [Cin*k, out_plane]");
        let g_slice = &grad_flat[b * filters * out_plane..(b + 1) * filters * out_plane];
        let g_mat = ArrayView2::from_shape((filters, out_plane), g_slice)
            .expect("grad slice matches [F, out_plane]");

        // Weight gradient [F, Cin*k] and input-gradient columns [Cin*k, out_plane]. When forcing
        // serial, take the explicit serial path; otherwise let the work-size gate decide (so a
        // below-gate small conv still runs its GEMMs serial instead of force-forking them).
        let (wg, dcol): (Array2<f32>, Array2<f32>) = if serial_gemm {
            (
                gemm_par_switch(&g_mat, &col_mat.t(), false),
                gemm_par_switch(&w_mat.t(), &g_mat, false),
            )
        } else {
            (
                gemm_par_auto(&g_mat, &col_mat.t()),
                gemm_par_auto(&w_mat.t(), &g_mat),
            )
        };
        let bias_p: Vec<f32> = g_mat.outer_iter().map(|row| row.sum()).collect(); // [F]

        let dcol = dcol.as_slice().expect("matmul result is standard layout");
        let mut pad_grad = vec![0.0f32; cin * padded_plane];
        for c in 0..cin {
            let pc = c * padded_plane;
            for kk in 0..k_plane {
                let krow = (c * k_plane + kk) * out_plane;
                let off = kk * out_plane;
                for o in 0..out_plane {
                    pad_grad[pc + offsets[off + o]] += dcol[krow + o];
                }
            }
        }
        let input_grad_b = if padded_sp.as_slice() != sp {
            crop_padded(&pad_grad, cin, sp, &padded_sp, &pad_before)
        } else {
            pad_grad
        };
        (wg, bias_p, input_grad_b)
    };

    // Two GEMMs per item (weight grad + input grad), ~4*F*out_plane*k_total FLOPs apiece
    let gemm_flops = 4usize
        .saturating_mul(batch)
        .saturating_mul(filters)
        .saturating_mul(out_plane)
        .saturating_mul(k_total);
    let parallel = gemm_flops >= conv_parallel_min_flops();
    // Parallelize over the batch above the gate. Only force the per-item GEMMs serial once the
    // batch axis alone already fills the pool (`batch >= threads`)
    let serial_gemm = parallel && batch >= rayon::current_num_threads();
    let per_b = map_indexed(batch, parallel, |b| process_b(b, serial_gemm));

    // Reduce the per-batch partials in batch order (weight/bias sum across the batch axis)
    let mut weight_grad_arr = Array2::<f32>::zeros((filters, k_total));
    let mut bias_grad = vec![0.0f32; filters];
    let mut in_grad_flat = Vec::with_capacity(batch * cin * in_plane);
    for (wg, bias_p, ig_b) in per_b {
        weight_grad_arr += &wg;
        for (acc, v) in bias_grad.iter_mut().zip(bias_p) {
            *acc += v;
        }
        in_grad_flat.extend(ig_b);
    }
    let weight_grad = weight_grad_arr.into_raw_vec_and_offset().0;

    let mut ig_shape = Vec::with_capacity(2 + r);
    ig_shape.push(batch);
    ig_shape.push(cin);
    ig_shape.extend_from_slice(sp);
    let input_grad =
        ArrayD::from_shape_vec(IxDyn(&ig_shape), in_grad_flat).expect("input grad matches shape");

    Ok(ConvGradients {
        weight_grad,
        bias_grad,
        input_grad,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // row_major_strides

    /// Row-major strides of a 3-D shape are the products of trailing dimensions
    #[test]
    fn test_row_major_strides_3d() {
        let got = row_major_strides(&[2, 3, 4]);
        assert_eq!(got, vec![12, 4, 1]);
    }

    /// A 1-D shape has a single unit stride
    #[test]
    fn test_row_major_strides_1d() {
        let got = row_major_strides(&[5]);
        assert_eq!(got, vec![1]);
    }

    /// An empty shape yields no strides
    #[test]
    fn test_row_major_strides_empty() {
        let got = row_major_strides(&[]);
        assert_eq!(got, Vec::<usize>::new());
    }

    // increment_index

    /// A 2-D index walks last axis first and wraps to the origin after the last cell
    #[test]
    fn test_increment_index_2d() {
        let dims = [2usize, 3];
        let mut idx = vec![0usize, 0];

        // [0,0] -> [0,1]
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 1]);

        // [0,1] -> [0,2]
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 2]);

        // [0,2] -> [1,0] (last-axis overflow carries)
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 0]);

        // [1,0] -> [1,1]
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 1]);

        // [1,1] -> [1,2]
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 2]);

        // [1,2] -> overflow on both axes, wraps to [0,0]
        assert!(!increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 0]);
    }

    /// A single-axis index advances then returns false on overflow
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

        // 2 -> overflow
        assert!(!increment_index(&mut idx, &dims));
    }

    // conv_geometry - Valid padding

    /// Valid 1-D geometry shrinks the output and adds no padding
    #[test]
    fn test_conv_geometry_valid_1d() {
        let (out_sp, pad_before, padded_sp) =
            conv_geometry(&[5], &[3], &[1], PaddingType::Valid).unwrap();
        assert_eq!(out_sp, vec![3]);
        assert_eq!(pad_before, vec![0]);
        assert_eq!(padded_sp, vec![5]);
    }

    /// Valid padding errors (no panic) when an input axis is smaller than the kernel
    #[test]
    fn test_conv_geometry_valid_input_smaller_than_kernel_errors() {
        let result = conv_geometry(&[2], &[3], &[1], PaddingType::Valid);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput, got {:?}",
            result
        );
    }

    // conv_geometry - Same padding, 1-D

    /// Same 1-D geometry rounds the output up and pads to preserve coverage
    #[test]
    fn test_conv_geometry_same_1d() {
        let (out_sp, pad_before, padded_sp) =
            conv_geometry(&[7], &[3], &[2], PaddingType::Same).unwrap();
        assert_eq!(out_sp, vec![4]);
        assert_eq!(pad_before, vec![1]);
        assert_eq!(padded_sp, vec![9]);
    }

    // conv_geometry - Same padding, 2-D

    /// Same 2-D geometry applies the per-axis padding rule independently on each axis
    #[test]
    fn test_conv_geometry_same_2d() {
        let (out_sp, pad_before, padded_sp) =
            conv_geometry(&[4, 4], &[3, 3], &[1, 1], PaddingType::Same).unwrap();
        assert_eq!(out_sp, vec![4, 4]);
        assert_eq!(pad_before, vec![1, 1]);
        assert_eq!(padded_sp, vec![6, 6]);
    }

    // build_padded

    /// A 2x2 block lands at the padded offset and the border stays zero
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

    /// Two batch-channels each pad into a disjoint 16-element slice
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

    /// 1-D padding shifts the data by pad_before and zeros the ends
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

    /// Zero padding returns the input unchanged
    #[test]
    fn test_build_padded_no_padding() {
        let in_flat = [5.0f32, 6.0, 7.0, 8.0];
        let got = build_padded(&in_flat, 1, &[2, 2], &[2, 2], &[0, 0]);
        assert_eq!(got, vec![5.0f32, 6.0, 7.0, 8.0]);
    }

    // crop_padded (inverse of build_padded)

    /// Cropping the padded buffer recovers exactly the data `build_padded` inserted
    #[test]
    fn test_crop_padded_roundtrip() {
        let in_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let padded = build_padded(&in_flat, 2, &[2, 2], &[4, 4], &[1, 1]);
        let got = crop_padded(&padded, 2, &[2, 2], &[4, 4], &[1, 1]);
        assert_eq!(got, in_flat.to_vec());
    }

    /// 1-D crop pulls the interior back out and drops the zero ends
    #[test]
    fn test_crop_padded_1d() {
        let padded = [0.0f32, 10.0, 20.0, 30.0, 0.0];
        let got = crop_padded(&padded, 1, &[3], &[5], &[1]);
        assert_eq!(got, vec![10.0f32, 20.0, 30.0]);
    }

    // im2col_offsets

    /// 1-D, kernel 3, stride 1 over a length-5 (already padded) plane: tap `kk` at output `o` reads
    /// index `o + kk`, laid out as `[k_plane, out_plane]`
    #[test]
    fn test_im2col_offsets_1d() {
        // out_sp = (5 - 3)/1 + 1 = 3; padded plane length 5, stride 1
        let offsets = im2col_offsets(&[3], &[3], &[1], &[1]);
        // rows = kk (0..3), cols = o (0..3); offset = o*1 + kk*1
        assert_eq!(
            offsets,
            vec![
                0, 1, 2, /*kk0*/ 1, 2, 3, /*kk1*/ 2, 3, 4 /*kk2*/
            ]
        );
    }
}
