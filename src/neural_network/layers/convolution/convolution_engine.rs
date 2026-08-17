//! Dimension-generic convolution engine shared by `Conv1D`, `Conv2D`, and `Conv3D`
//!
//! A plain convolution is the same operation at every rank. Only the number of spatial axes
//! changes. One implementation, driven by the spatial rank computed at runtime as `ndim - 2`,
//! serves all 3 layers. The layer wrappers keep their public API, weight storage, activation, and
//! caches. They delegate only the numeric forward and backward passes to [`conv_forward`] and
//! [`conv_backward`]. `SeparableConv2D` also calls these same 2 functions for its pointwise (1x1)
//! stage
//!
//! The geometry and im2col helpers below are `pub(super)`, because
//! [`conv_transpose_engine`](super::conv_transpose_engine) reuses every one of them. A transposed
//! convolution is the adjoint of a plain one, so it needs the same padding rule, the same offset
//! table, and the same block copy. Only the direction of the 2 GEMMs changes
//!
//! # Conventions
//!
//! - `Valid` output: `(in - k) / stride + 1`
//! - `Same` output: `ceil(in / stride)`
//! - `Same` padding splits the total padding evenly, and the extra cell (if any) goes on the
//!   trailing edge (`pad_before = pad_total / 2`)
//! - The engine computes a cross-correlation, so it does not flip the kernel
//! - The bias is added last, after the matrix product
//!
//! # Layout
//!
//! Tensors are channels-last, `[batch, spatial..., channels]`. Weights are flat row-major
//! `[k..., Cin, F]` (Keras' kernel shape). The channel axis is innermost. This turns im2col into
//! a copy instead of a gather. One kernel tap at one output position is `Cin` contiguous floats,
//! so the pass moves runs instead of scalars. The flat weight matrix `[k*Cin, F]` needs no
//! permutation to align with it
//!
//! The forward pass parallelizes over `(batch item, output-position block)` tasks. Splitting the
//! output positions lets a single large image use every core, even when `batch == 1`. Each task
//! builds its own im2col block and runs 1 serial GEMM directly into its disjoint output region.
//! Under this layout that region is a contiguous slab of rows, so the product needs no scatter.
//! The GEMM applies the per-filter bias as a fused `PerCol` epilogue. Each block then costs 1
//! pass, with no product temporary and no separate bias sweep. The fused result matches, bit for
//! bit, the plain product followed by the same scalar bias add.
//!
//! The backward pass parallelizes over batch items. It reduces their weight and bias partials in
//! batch order, so rerunning on the same machine gives the same result. Both GEMMs route through
//! the crate's [`dot_par`](crate::math::matmul::dot_par). The per-item GEMMs stay parallel while
//! the batch fan is too short to fill the thread pool. They switch to serial once the batch alone
//! fills the pool, so a batch task does not fork rayon again inside its own GEMM

use super::PaddingType;
use crate::error::Error;
use crate::math::matmul::dot_par;
use crate::neural_network::Tensor;
use gemmkit_ndarray::{Bias, Parallelism};
use ndarray::{Array2, Array3, ArrayD, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use rayon::prelude::*;

tunable_gate! {
    /// Minimum estimated GEMM FLOPs (`2 * batch * F * out_plane * Cin * k`) at or above which an
    /// engine pass runs in parallel
    ///
    /// Counting FLOPs rather than output elements keeps the gate meaningful across kernel sizes
    /// and channel counts. An output-element count would rate a `7x7x512` and a `3x3x3`
    /// convolution the same
    ///
    /// Overridable through [`crate::tuning`]
    pub(crate) CONV_PARALLEL_MIN_FLOPS => conv_parallel_min_flops / set_conv_parallel_min_flops = 4_000_000
}

/// Minimum output positions per forward task
///
/// Each task's GEMM re-packs the weight matrix, so blocks need enough positions to amortize that
const CONV_MIN_CHUNK_POSITIONS: usize = 64;

/// Analytic gradients returned by [`conv_backward`], and by the transposed-convolution backward
/// pass
pub(super) struct ConvGradients {
    /// Weight gradient, flat row-major in the layer's own kernel layout (reshape to the layer's
    /// weight array). That is `[k..., Cin, F]` for a plain convolution, and `[k..., F, Cin]` for
    /// a transposed one
    pub weight_grad: Vec<f32>,
    /// Bias gradient, one value per filter `[F]`
    pub bias_grad: Vec<f32>,
    /// Input gradient, shape `[batch, spatial..., Cin]`
    pub input_grad: Tensor,
}

/// Element strides of the spatial axes of one row-major `[spatial..., cin]` item
///
/// The channel axis is innermost with stride 1, so the innermost spatial step spans `cin` elements
/// and each outer axis multiplies up from there. Every offset the engine computes is in these
/// units, which is why a padded-buffer index lands directly on a position's first channel
pub(super) fn spatial_strides(sp: &[usize], cin: usize) -> Vec<usize> {
    let mut strides = vec![cin; sp.len()];
    for d in (0..sp.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * sp[d + 1];
    }
    strides
}

/// Advances a multi-index `idx` (row-major, last axis fastest) within `dims`. Returns `false`
/// when it wraps
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
pub(super) fn map_indexed<R, F>(n: usize, parallel: bool, f: F) -> Vec<R>
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
pub(super) type ConvGeometry = (Vec<usize>, Vec<usize>, Vec<usize>);

pub(super) fn conv_geometry(
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

/// Builds a zero-padded copy of the flat `[items, sp..., cin]` data
///
/// The copy pads only the spatial axes. The `cin` channels of a position stay contiguous and
/// travel together, so each step of the walk moves a run rather than a single float
pub(super) fn build_padded(
    in_flat: &[f32],
    items: usize,
    sp: &[usize],
    padded_sp: &[usize],
    pad_before: &[usize],
    cin: usize,
) -> Vec<f32> {
    let r = sp.len();
    let in_item: usize = sp.iter().product::<usize>() * cin;
    let padded_item: usize = padded_sp.iter().product::<usize>() * cin;
    let padded_strides = spatial_strides(padded_sp, cin);
    let mut out = vec![0.0f32; items * padded_item];

    for item in 0..items {
        let in_base = item * in_item;
        let pad_base = item * padded_item;
        let mut si = vec![0usize; r];
        let mut si_flat = 0usize;
        loop {
            let mut pidx = pad_base;
            for d in 0..r {
                pidx += (si[d] + pad_before[d]) * padded_strides[d];
            }
            out[pidx..pidx + cin]
                .copy_from_slice(&in_flat[in_base + si_flat..in_base + si_flat + cin]);
            si_flat += cin;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
    }
    out
}

/// Inverse of [`build_padded`]: gathers the unpadded `[items, sp..., cin]` region out of a padded
/// buffer
///
/// Crops the input gradient back to its original spatial size after col2im
pub(super) fn crop_padded(
    padded: &[f32],
    items: usize,
    sp: &[usize],
    padded_sp: &[usize],
    pad_before: &[usize],
    cin: usize,
) -> Vec<f32> {
    let r = sp.len();
    let in_item: usize = sp.iter().product::<usize>() * cin;
    let padded_item: usize = padded_sp.iter().product::<usize>() * cin;
    let padded_strides = spatial_strides(padded_sp, cin);
    let mut out = vec![0.0f32; items * in_item];

    for item in 0..items {
        let in_base = item * in_item;
        let pad_base = item * padded_item;
        let mut si = vec![0usize; r];
        let mut si_flat = 0usize;
        loop {
            let mut pidx = pad_base;
            for d in 0..r {
                pidx += (si[d] + pad_before[d]) * padded_strides[d];
            }
            out[in_base + si_flat..in_base + si_flat + cin]
                .copy_from_slice(&padded[pidx..pidx + cin]);
            si_flat += cin;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
    }
    out
}

/// Flat padded-item offsets for im2col/col2im, laid out `[k_plane, out_plane]`
///
/// `offsets[kk * out_plane + o]` is the element index, within one padded `[spatial..., cin]`
/// item, of the first channel that output position `o` reads for kernel tap `kk`. The remaining
/// `cin - 1` channels follow it contiguously. `padded_strides` carries the `cin` scaling (see
/// [`spatial_strides`]), so the table does not depend on batch. The engine computes it once and
/// reuses it for every im2col copy and every col2im accumulate
pub(super) fn im2col_offsets(
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

/// Per-pass im2col inputs shared by every task: the padded data and the copy geometry
pub(super) struct ColContext<'a> {
    /// Flat zero-padded source `[batch, padded_spatial..., cin]`
    pub(super) padded: &'a [f32],
    /// Channels of the padded source, which is also the length of one contiguous copy run. The
    /// plain convolution reads the input, so this is `Cin`. The transposed backward pass reads
    /// the output gradient, so there it is the filter count
    pub(super) cin: usize,
    /// Elements per padded batch item (`padded_plane * cin`)
    pub(super) padded_item: usize,
    /// Kernel taps per channel
    pub(super) k_plane: usize,
    /// Positions per batch item that the walk visits. The plain convolution visits its output
    /// positions. The transposed backward pass visits its input positions
    pub(super) out_plane: usize,
    /// `[k_plane, out_plane]` copy offsets from [`im2col_offsets`]
    pub(super) offsets: &'a [usize],
}

/// im2col for one batch item, restricted to the output positions `[c0, c1)`
///
/// Builds a `[c1-c0, k_plane*Cin]` row-major matrix, one row per output position, whose columns
/// align with the flat weight matrix `[k_plane*Cin, F]`. Within a row the kernel taps run
/// slowest and the channels fastest. This is exactly the order the padded buffer already holds
/// them in, so each tap is 1 `copy_from_slice` of `Cin` floats. Pass the full `[0, out_plane)`
/// range for the whole item. A sub-range builds one forward task's row block
pub(super) fn build_col_range(ctx: &ColContext, b: usize, c0: usize, c1: usize) -> Vec<f32> {
    let rows = c1 - c0;
    let k_total = ctx.k_plane * ctx.cin;
    let mut col = vec![0.0f32; rows * k_total];
    let b_base = b * ctx.padded_item;
    // The loop nests tap outer, position inner. At a fixed tap, consecutive output positions
    // read consecutive runs (adjacent runs at unit stride), so the source side stays a single
    // streaming read. The transposed nest would make the write sequential instead, at the cost
    // of `k_plane` live read streams at once. `col` stays cache-resident either way, so the read
    // side is the one worth keeping linear
    for kk in 0..ctx.k_plane {
        let off = kk * ctx.out_plane;
        let kbase = kk * ctx.cin;
        for (i, o) in (c0..c1).enumerate() {
            let src = b_base + ctx.offsets[off + o];
            let dst = i * k_total + kbase;
            col[dst..dst + ctx.cin].copy_from_slice(&ctx.padded[src..src + ctx.cin]);
        }
    }
    col
}

/// Runs the forward convolution. `weight_shape` is `[k..., Cin, F]`, `bias` is `[F]`, and
/// `strides` has one entry per spatial axis
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

/// `conv_forward` with an optional override of the parallel-or-serial gate decision, so a bench
/// can measure both paths on either side of the gate
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
    let r = in_shape.len() - 2;
    let batch = in_shape[0];
    let sp = &in_shape[1..1 + r];
    let cin = in_shape[1 + r];
    let filters = weight_shape[r + 1];
    let k_dims = &weight_shape[..r];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding)?;
    let out_plane: usize = out_sp.iter().product();
    let padded_item: usize = padded_sp.iter().product::<usize>() * cin;
    let padded_strides = spatial_strides(&padded_sp, cin);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let padded_storage = if padded_sp.as_slice() != sp {
        Some(build_padded(
            in_flat,
            batch,
            sp,
            &padded_sp,
            &pad_before,
            cin,
        ))
    } else {
        None
    };
    let padded: &[f32] = padded_storage.as_deref().unwrap_or(in_flat);

    // im2col + gemm
    let k_total = k_plane * cin;
    let offsets = im2col_offsets(&out_sp, k_dims, strides, &padded_strides);
    let w_mat = ArrayView2::from_shape((k_total, filters), weights)
        .expect("weights length matches [k*Cin, F]");

    // One task per (batch item, output-position block)
    let ctx = ColContext {
        padded,
        cin,
        padded_item,
        k_plane,
        out_plane,
        offsets: &offsets,
    };
    let fill_block = |b: usize, c0: usize, mut blk: ArrayViewMut2<f32>| {
        let rows = blk.nrows();
        let col = build_col_range(&ctx, b, c0, c0 + rows);
        let col_mat = ArrayView2::from_shape((rows, k_total), &col)
            .expect("col block length matches [positions, k*Cin]");
        // `C <- col_mat @ w_mat + bias`, in 1 fused pass. `beta == 0` overwrites the block
        // without reading it. `blk` is written in place, so the product needs no temporary and
        // no separate bias sweep. The per-filter bias is 1 value per column of the
        // `[positions, F]` product, so it is a `Bias::PerCol` epilogue applied after the
        // accumulation. That gives the same "bias added last" order as the elementwise pass it
        // replaces, bit for bit. `blk` is a contiguous row block of `out3`, so the product lands
        // in its final place with no scatter and no copy
        gemmkit_ndarray::gemm_fused(
            1.0,
            &col_mat,
            &w_mat,
            0.0,
            &mut blk,
            Some(Bias::PerCol(bias)),
            None,
            // Serial because `fill_block` already runs as 1 leaf task of the batch-by-position-
            // chunk par_iters above the gate, and as a plain loop body below it
            Parallelism::Serial,
        );
    };

    let gemm_flops = 2usize
        .saturating_mul(batch)
        .saturating_mul(filters)
        .saturating_mul(out_plane)
        .saturating_mul(k_total);
    let parallel = force_parallel.unwrap_or(gemm_flops >= conv_parallel_min_flops());

    let mut out3 = Array3::<f32>::zeros((batch, out_plane, filters));
    if parallel {
        // Enough blocks to feed every thread once the batch alone cannot
        let chunks_per_item = rayon::current_num_threads().div_ceil(batch);
        let chunk_rows = out_plane
            .div_ceil(chunks_per_item)
            .max(CONV_MIN_CHUNK_POSITIONS);
        out3.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut out_b)| {
                out_b
                    .axis_chunks_iter_mut(Axis(0), chunk_rows)
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(ci, blk)| fill_block(b, ci * chunk_rows, blk));
            });
    } else {
        for (b, mut out_b) in out3.axis_iter_mut(Axis(0)).enumerate() {
            fill_block(b, 0, out_b.view_mut());
        }
    }

    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.extend_from_slice(&out_sp);
    out_shape.push(filters);
    Ok(out3
        .into_shape_with_order(IxDyn(&out_shape))
        .expect("conv output length matches shape"))
}

/// Runs the backward convolution. `input` is the original, unpadded forward input.
/// `grad_output` is the gradient of the convolution output, taken after the activation backward
pub(super) fn conv_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> Result<ConvGradients, Error> {
    let in_shape = input.shape();
    let r = in_shape.len() - 2;
    let batch = in_shape[0];
    let sp = &in_shape[1..1 + r];
    let cin = in_shape[1 + r];
    let filters = weight_shape[r + 1];
    let k_dims = &weight_shape[..r];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding)?;
    let out_plane: usize = out_sp.iter().product();
    let in_item: usize = sp.iter().product::<usize>() * cin;
    let padded_item: usize = padded_sp.iter().product::<usize>() * cin;
    let padded_strides = spatial_strides(&padded_sp, cin);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let padded_storage = if padded_sp.as_slice() != sp {
        Some(build_padded(
            in_flat,
            batch,
            sp,
            &padded_sp,
            &pad_before,
            cin,
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
    let k_total = k_plane * cin;
    let offsets = im2col_offsets(&out_sp, k_dims, strides, &padded_strides);
    let w_mat = ArrayView2::from_shape((k_total, filters), weights)
        .expect("weights length matches [k*Cin, F]");

    let ctx = ColContext {
        padded,
        cin,
        padded_item,
        k_plane,
        out_plane,
        offsets: &offsets,
    };
    // `gemm_par` is `Serial` when the batch fan alone fills the pool, so a per-batch task does
    // not fork rayon again inside its GEMM. Otherwise it is `Rayon(0)`, and the backend decides
    // how to parallelize
    let process_b = |b: usize, gemm_par: Parallelism| -> (Array2<f32>, Vec<f32>, Vec<f32>) {
        let col = build_col_range(&ctx, b, 0, out_plane);
        let col_mat = ArrayView2::from_shape((out_plane, k_total), &col)
            .expect("col length matches [out_plane, k*Cin]");
        let g_slice = &grad_flat[b * out_plane * filters..(b + 1) * out_plane * filters];
        let g_mat = ArrayView2::from_shape((out_plane, filters), g_slice)
            .expect("grad slice matches [out_plane, F]");

        // Weight gradient `[k*Cin, F]` (already the layer's weight layout, so no permute
        // needed), and input-gradient rows `[out_plane, k*Cin]`
        let wg = dot_par(&col_mat.t(), &g_mat, gemm_par);
        let dcol = dot_par(&g_mat, &w_mat.t(), gemm_par);
        let bias_p: Vec<f32> = g_mat.sum_axis(Axis(0)).to_vec(); // `[F]`

        let dcol = dcol.as_slice().expect("matmul result is standard layout");
        let mut pad_grad = vec![0.0f32; padded_item];
        // Same tap-outer nest as `build_col_range`, for the same reason: a fixed tap walks the
        // padded buffer linearly, so the read-modify-write side stays 1 stream
        for kk in 0..k_plane {
            let off = kk * out_plane;
            let kbase = kk * cin;
            for o in 0..out_plane {
                let dst = offsets[off + o];
                let src = o * k_total + kbase;
                for c in 0..cin {
                    pad_grad[dst + c] += dcol[src + c];
                }
            }
        }
        let input_grad_b = if padded_sp.as_slice() != sp {
            crop_padded(&pad_grad, 1, sp, &padded_sp, &pad_before, cin)
        } else {
            pad_grad
        };
        (wg, bias_p, input_grad_b)
    };

    // Each item runs 2 GEMMs (weight grad and input grad), about 4 * F * out_plane * k_total
    // FLOPs apiece
    let gemm_flops = 4usize
        .saturating_mul(batch)
        .saturating_mul(filters)
        .saturating_mul(out_plane)
        .saturating_mul(k_total);
    let parallel = gemm_flops >= conv_parallel_min_flops();
    // The pass parallelizes over the batch above the gate. It forces the per-item GEMMs serial
    // only once the batch axis alone fills the pool (`batch >= threads`)
    let gemm_par = if parallel && batch >= rayon::current_num_threads() {
        Parallelism::Serial
    } else {
        Parallelism::Rayon(0)
    };
    let per_b = map_indexed(batch, parallel, |b| process_b(b, gemm_par));

    // Reduces the per-batch partials in batch order, summing weight and bias across the batch
    // axis
    let mut weight_grad_arr = Array2::<f32>::zeros((k_total, filters));
    let mut bias_grad = vec![0.0f32; filters];
    let mut in_grad_flat = Vec::with_capacity(batch * in_item);
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
    ig_shape.extend_from_slice(sp);
    ig_shape.push(cin);
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

    // spatial_strides

    /// Spatial strides of a 3-D shape are the trailing-dimension products, each scaled by the
    /// channel count that sits innermost
    #[test]
    fn test_spatial_strides_3d() {
        let got = spatial_strides(&[2, 3, 4], 8);
        assert_eq!(got, vec![96, 32, 8]);
    }

    /// A single channel degenerates to plain row-major strides
    #[test]
    fn test_spatial_strides_single_channel() {
        let got = spatial_strides(&[2, 3, 4], 1);
        assert_eq!(got, vec![12, 4, 1]);
    }

    /// A 1-D shape has 1 stride, equal to the channel count
    #[test]
    fn test_spatial_strides_1d() {
        let got = spatial_strides(&[5], 3);
        assert_eq!(got, vec![3]);
    }

    /// An empty shape yields no strides
    #[test]
    fn test_spatial_strides_empty() {
        let got = spatial_strides(&[], 4);
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

    // conv_geometry: Valid padding

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

    // conv_geometry: Same padding, 1-D

    /// Same 1-D geometry rounds the output up and pads to preserve coverage
    #[test]
    fn test_conv_geometry_same_1d() {
        let (out_sp, pad_before, padded_sp) =
            conv_geometry(&[7], &[3], &[2], PaddingType::Same).unwrap();
        assert_eq!(out_sp, vec![4]);
        assert_eq!(pad_before, vec![1]);
        assert_eq!(padded_sp, vec![9]);
    }

    // conv_geometry: Same padding, 2-D

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
        let got = build_padded(&in_flat, 1, &[2, 2], &[4, 4], &[1, 1], 1);

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

    /// 2 batch items each pad into a disjoint 16-element slice
    #[test]
    fn test_build_padded_two_items() {
        let in_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let got = build_padded(&in_flat, 2, &[2, 2], &[4, 4], &[1, 1], 1);

        assert_eq!(got.len(), 32);

        // Item 0 (offset 0)
        assert_eq!(got[5], 1.0);
        assert_eq!(got[6], 2.0);
        assert_eq!(got[9], 3.0);
        assert_eq!(got[10], 4.0);

        // Item 1 (offset 16)
        assert_eq!(got[16 + 5], 5.0);
        assert_eq!(got[16 + 6], 6.0);
        assert_eq!(got[16 + 9], 7.0);
        assert_eq!(got[16 + 10], 8.0);
    }

    /// The channels of a position stay adjacent through the pad. A 2x2x2 input lands as 4
    /// 2-float runs at the strides `spatial_strides` reports, and nothing interleaves them
    #[test]
    fn test_build_padded_keeps_channels_contiguous() {
        // Positions (0,0)=[1,2], (0,1)=[3,4], (1,0)=[5,6], (1,1)=[7,8]
        let in_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let got = build_padded(&in_flat, 1, &[2, 2], &[4, 4], &[1, 1], 2);

        assert_eq!(got.len(), 32, "4*4 positions x 2 channels");

        // Spatial strides are [8, 2], so padded position (1+si0, 1+si1) starts at 8+8*si0+2+2*si1
        assert_eq!(&got[10..12], &[1.0, 2.0], "padded (1,1)");
        assert_eq!(&got[12..14], &[3.0, 4.0], "padded (1,2)");
        assert_eq!(&got[18..20], &[5.0, 6.0], "padded (2,1)");
        assert_eq!(&got[20..22], &[7.0, 8.0], "padded (2,2)");

        for (i, &val) in got.iter().enumerate() {
            if !(10..14).contains(&i) && !(18..22).contains(&i) {
                assert_eq!(val, 0.0, "padded[{i}] should be 0.0 (border), got {val}");
            }
        }
    }

    /// 1-D padding shifts the data by pad_before and zeros the ends
    #[test]
    fn test_build_padded_1d() {
        let in_flat = [10.0f32, 20.0, 30.0];
        let got = build_padded(&in_flat, 1, &[3], &[5], &[1], 1);

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
        let got = build_padded(&in_flat, 1, &[2, 2], &[2, 2], &[0, 0], 1);
        assert_eq!(got, vec![5.0f32, 6.0, 7.0, 8.0]);
    }

    // crop_padded (inverse of build_padded)

    /// Cropping the padded buffer recovers exactly the data `build_padded` inserted
    #[test]
    fn test_crop_padded_roundtrip() {
        let in_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let padded = build_padded(&in_flat, 2, &[2, 2], &[4, 4], &[1, 1], 1);
        let got = crop_padded(&padded, 2, &[2, 2], &[4, 4], &[1, 1], 1);
        assert_eq!(got, in_flat.to_vec());
    }

    /// The round-trip holds with several channels too, which is the case where the walk moves
    /// runs rather than single floats
    #[test]
    fn test_crop_padded_roundtrip_multichannel() {
        let in_flat: Vec<f32> = (0..24).map(|v| v as f32).collect(); // 2 items x 2x2 x 3 channels
        let padded = build_padded(&in_flat, 2, &[2, 2], &[4, 4], &[1, 1], 3);
        let got = crop_padded(&padded, 2, &[2, 2], &[4, 4], &[1, 1], 3);
        assert_eq!(got, in_flat);
    }

    /// 1-D crop pulls the interior back out and drops the zero ends
    #[test]
    fn test_crop_padded_1d() {
        let padded = [0.0f32, 10.0, 20.0, 30.0, 0.0];
        let got = crop_padded(&padded, 1, &[3], &[5], &[1], 1);
        assert_eq!(got, vec![10.0f32, 20.0, 30.0]);
    }

    // im2col_offsets

    /// 1-D, kernel 3, stride 1, over an already-padded length-5 plane at 1 channel. Tap `kk` at
    /// output `o` reads index `o + kk`. The table is laid out as `[k_plane, out_plane]`
    #[test]
    fn test_im2col_offsets_1d() {
        // out_sp = (5 - 3)/1 + 1 = 3. The padded plane length is 5, at stride 1
        let offsets = im2col_offsets(&[3], &[3], &[1], &spatial_strides(&[5], 1));
        // rows = kk (0..3), cols = o (0..3). offset = o*1 + kk*1
        assert_eq!(
            offsets,
            vec![
                0, 1, 2, /*kk0*/ 1, 2, 3, /*kk1*/ 2, 3, 4 /*kk2*/
            ]
        );
    }

    /// With several channels every offset scales by the channel count, because it addresses a
    /// position's first channel and the rest follow contiguously
    #[test]
    fn test_im2col_offsets_1d_multichannel() {
        let offsets = im2col_offsets(&[3], &[3], &[1], &spatial_strides(&[5], 2));
        // Same table as the 1-channel case, doubled
        assert_eq!(
            offsets,
            vec![
                0, 2, 4, /*kk0*/ 2, 4, 6, /*kk1*/ 4, 6, 8 /*kk2*/
            ]
        );
    }

    // conv_forward: hand-derived values
    //
    // The helper tests above check the geometry math. These tests check the whole pass against
    // numbers worked out by hand from the cross-correlation definition. That distinction matters
    // here. A gradient check compares a layer against a finite difference of itself. It still
    // agrees even when the forward pass has a consistent axis-transposition error. Only an
    // independently derived expected value catches a moved axis.

    /// 2-D, 1 batch item, 3x3 input at 2 channels, a 2x2 all-ones kernel and 1 filter
    ///
    /// The channel-0 plane is 1..9 and the channel-1 plane is 10..90. Each output is the sum of
    /// a 2x2 window over both planes, plus the bias
    #[test]
    fn test_conv_forward_2d_two_channels_hand_derived() {
        // [1, 3, 3, 2] channels-last: position (h, w) holds [plane0, plane1]
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3, 2]),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, // row 0
                4.0, 40.0, 5.0, 50.0, 6.0, 60.0, // row 1
                7.0, 70.0, 8.0, 80.0, 9.0, 90.0, // row 2
            ],
        )
        .unwrap();
        // [kh, kw, Cin, F] all ones
        let w_shape = [2usize, 2, 2, 1];
        let weights = vec![1.0f32; w_shape.iter().product()];
        let bias = [0.5f32];

        let out = conv_forward(
            &input,
            &weights,
            &w_shape,
            &bias,
            &[1, 1],
            PaddingType::Valid,
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 2, 2, 1]);
        // (1+2+4+5) + (10+20+40+50) + 0.5 = 132.5, and so on across the 4 windows
        let got = out.iter().copied().collect::<Vec<f32>>();
        assert_eq!(got, vec![132.5, 176.5, 264.5, 308.5]);
    }

    /// The filter axis is the fastest-varying output axis. A second filter holding twice the
    /// first kernel's weights must land interleaved with it, not in a separate plane
    #[test]
    fn test_conv_forward_2d_filter_axis_is_innermost() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // [kh=2, kw=2, Cin=1, F=2]. Filter 0 is all ones, filter 1 is all twos
        let weights = vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let bias = [0.0f32, 0.0];

        let out = conv_forward(
            &input,
            &weights,
            &[2, 2, 1, 2],
            &bias,
            &[1, 1],
            PaddingType::Valid,
        )
        .unwrap();

        // 1 output position, 2 filters: sum = 10, so [10, 20] adjacent in memory
        assert_eq!(out.shape(), &[1, 1, 1, 2]);
        assert_eq!(out.iter().copied().collect::<Vec<f32>>(), vec![10.0, 20.0]);
    }

    /// 1-D `Same` padding at stride 1 keeps the length and reads zeros past the edges
    #[test]
    fn test_conv_forward_1d_same_padding_hand_derived() {
        // [1, 4, 1]
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // [k=3, Cin=1, F=1] all ones
        let weights = vec![1.0f32; 3];
        let bias = [0.0f32];

        let out =
            conv_forward(&input, &weights, &[3, 1, 1], &bias, &[1], PaddingType::Same).unwrap();

        // pad_before = 1, so the windows are [0,1,2], [1,2,3], [2,3,4], [3,4,0]
        assert_eq!(out.shape(), &[1, 4, 1]);
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![3.0, 6.0, 9.0, 7.0]
        );
    }
}
