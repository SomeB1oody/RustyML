//! Dimension-generic transposed-convolution engine shared by `Conv1DTranspose`,
//! `Conv2DTranspose`, and `Conv3DTranspose`
//!
//! A transposed convolution is the adjoint of a plain convolution. It maps a tensor with the
//! shape of a convolution output back to the shape of the convolution input, and it keeps the
//! same connectivity. So this engine is [`convolution_engine`](super::convolution_engine) with
//! its 2 halves exchanged. The forward pass here is the col2im half of `conv_backward`, and the
//! backward pass here is the im2col half of `conv_forward`. This engine reuses every geometry
//! helper, the offset table, and the block copy unchanged
//!
//! # Conventions
//!
//! - `Valid` output: `in * stride + max(kernel - stride, 0)`
//! - `Same` output: `in * stride`
//! - Weights are flat row-major `[k..., F, Cin]`. The filter axis comes before the input-channel
//!   axis, which is the reverse of the plain convolution kernel. This is the layout a transposed
//!   convolution needs, because the pass reads `Cin` and writes `F`
//! - Each input position `i` writes to output position `i * stride + tap - pad_before`
//! - This engine adds the bias last, once per output position, including positions that no input
//!   reached
//!
//! # Geometry
//!
//! The layer picks its output size with the 2 rules above. Everything else follows from 1 fact.
//! The plain convolution that maps the output size back to the input size is the convolution
//! this pass transposes. It uses the same kernel, stride, and padding mode. So
//! [`conv_geometry`] applied to the *output* size returns the input size, the leading padding to
//! crop, and the padded extent to scatter into. No second padding rule exists
//!
//! # Parallelism
//!
//! Both passes fan out over batch items only. The forward scatter accumulates into overlapping
//! output positions whenever `stride < kernel`, so splitting 1 item by output position would need
//! a merge. Batch items stay disjoint, so they need none. Below a full batch the per-item GEMMs
//! run parallel instead, which keeps a batch of 1 from running on 1 core.
//!
//! The pass reduces the weight and bias partials in batch order, so rerunning on the same machine
//! gives the same result. The gate is the convolution engine's
//! [`conv_parallel_min_flops`](super::convolution_engine::conv_parallel_min_flops), because both
//! engines run the same 2 GEMM shapes. The gate selects the batch fan-out, and through it the
//! per-item GEMM parallelism. The backend gives the same product at either setting, so moving the
//! gate changes no result

use super::PaddingType;
use super::convolution_engine::{
    ColContext, ConvGradients, build_col_range, build_padded, conv_geometry,
    conv_parallel_min_flops, crop_padded, im2col_offsets, map_indexed, spatial_strides,
};
use crate::error::Error;
use crate::math::matmul::dot_par;
use crate::neural_network::Tensor;
use gemmkit_ndarray::Parallelism;
use ndarray::{Array2, ArrayD, ArrayView2, Axis, IxDyn};

/// Output size of 1 spatial axis of a transposed convolution
///
/// # Parameters
///
/// - `input` - Input size of the axis
/// - `kernel` - Kernel size of the axis
/// - `stride` - Stride of the axis
/// - `padding` - Padding mode
///
/// # Returns
///
/// - `usize` - Output size of the axis
pub(super) fn transpose_output_length(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: PaddingType,
) -> usize {
    match padding {
        // A kernel no wider than the stride reaches no further than the stride already does.
        // The `max` therefore holds the output at `input * stride` instead of shrinking it
        PaddingType::Valid => input * stride + kernel.saturating_sub(stride),
        PaddingType::Same => input * stride,
    }
}

/// Shape facts both passes need
#[derive(Debug)]
struct TransposeGeometry {
    /// Output spatial sizes
    out_sp: Vec<usize>,
    /// Leading padding to crop off the scatter buffer, 1 entry per spatial axis
    pad_before: Vec<usize>,
    /// Spatial sizes of the scatter buffer, which is the output plus its padding
    padded_sp: Vec<usize>,
}

/// Derives the geometry of a transposed convolution from its input spatial sizes
///
/// # Errors
///
/// - `Error::InvalidInput` - If any input spatial size is 0
fn transpose_geometry(
    in_sp: &[usize],
    k_dims: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> Result<TransposeGeometry, Error> {
    if let Some(d) = in_sp.iter().position(|&n| n == 0) {
        return Err(Error::invalid_input(format!(
            "transposed convolution requires every input spatial dimension to be 1 or more: axis \
             {d} has size 0"
        )));
    }

    let out_sp: Vec<usize> = (0..in_sp.len())
        .map(|d| transpose_output_length(in_sp[d], k_dims[d], strides[d], padding))
        .collect();
    // The plain convolution from `out_sp` back to `in_sp` is the one this pass transposes, so its
    // geometry is this pass's geometry. `conv_geometry` cannot fail here. Its `Valid` branch
    // rejects only an output axis smaller than the kernel. Both output rules above make every
    // axis at least the kernel size, for an input of 1 or more
    let (check_sp, pad_before, padded_sp) = conv_geometry(&out_sp, k_dims, strides, padding)?;
    debug_assert_eq!(
        check_sp, in_sp,
        "the transposed output size must convolve back to the input size"
    );

    Ok(TransposeGeometry {
        out_sp,
        pad_before,
        padded_sp,
    })
}

/// Splits an input shape into `(rank, batch, spatial, channels)`
fn split_shape(in_shape: &[usize]) -> (usize, usize, &[usize], usize) {
    let r = in_shape.len() - 2;
    (r, in_shape[0], &in_shape[1..1 + r], in_shape[1 + r])
}

/// Rejects a weight tensor whose input-channel axis disagrees with the tensor being passed
fn check_channels(weight_shape: &[usize], r: usize, cin: usize) -> Result<(), Error> {
    if weight_shape[r + 1] != cin {
        return Err(Error::invalid_input(format!(
            "transposed convolution kernel expects {} input channels but the input tensor has {}",
            weight_shape[r + 1],
            cin
        )));
    }
    Ok(())
}

/// Chooses the parallelism of 1 pass from its estimated GEMM FLOPs
///
/// Returns `(parallel_over_batch, per_item_gemm_parallelism)`. The per-item GEMMs drop to serial
/// only when the fan-out runs and the batch by itself fills the pool. A batch task then never
/// forks rayon again inside its own GEMM. In every other case the GEMM keeps the backend's own
/// scheduling. This is what lets a batch of 1 use more than 1 core, on either side of the gate
fn parallel_plan(gemm_flops: usize, batch: usize) -> (bool, Parallelism) {
    let parallel = gemm_flops >= conv_parallel_min_flops();
    let gemm_par = if parallel && batch >= rayon::current_num_threads() {
        Parallelism::Serial
    } else {
        Parallelism::Rayon(0)
    };
    (parallel, gemm_par)
}

/// Runs the forward transposed convolution. `weight_shape` is `[k..., F, Cin]`, `bias` is `[F]`,
/// and `strides` has 1 entry per spatial axis
///
/// # Errors
///
/// - `Error::InvalidInput` - If any input spatial size is 0, or if the kernel's input-channel
///   count differs from the input tensor's
pub(super) fn conv_transpose_forward(
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    strides: &[usize],
    padding: PaddingType,
) -> Result<Tensor, Error> {
    let (r, batch, in_sp, cin) = split_shape(input.shape());
    check_channels(weight_shape, r, cin)?;
    let filters = weight_shape[r];
    let k_dims = &weight_shape[..r];
    let k_plane: usize = k_dims.iter().product();

    let geometry = transpose_geometry(in_sp, k_dims, strides, padding)?;
    let TransposeGeometry {
        out_sp,
        pad_before,
        padded_sp,
    } = &geometry;
    let in_plane: usize = in_sp.iter().product();
    let out_plane: usize = out_sp.iter().product();
    let in_item = in_plane * cin;
    let padded_item: usize = padded_sp.iter().product::<usize>() * filters;
    let padded_strides = spatial_strides(padded_sp, filters);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // 1 column per (tap, filter) pair, which is the width of the scatter matrix below
    let k_total = k_plane * filters;
    let offsets = im2col_offsets(in_sp, k_dims, strides, &padded_strides);
    let w_mat =
        ArrayView2::from_shape((k_total, cin), weights).expect("weights length matches [k*F, Cin]");

    let process_b = |b: usize, gemm_par: Parallelism| -> Vec<f32> {
        let x_mat =
            ArrayView2::from_shape((in_plane, cin), &in_flat[b * in_item..(b + 1) * in_item])
                .expect("input slice matches [in_plane, Cin]");
        // `[in_plane, k*F]`: row `i` holds everything input position `i` sends out, laid out tap
        // slowest and filter fastest. This is the order the scatter below walks, and the same
        // order `build_col_range` builds for the backward pass
        let dcol = dot_par(&x_mat, &w_mat.t(), gemm_par);
        let dcol = dcol.as_slice().expect("matmul result is standard layout");

        let mut pad_out = vec![0.0f32; padded_item];
        // Tap outer, position inner. At a fixed tap, consecutive input positions write runs that
        // advance by the stride. Adjacent runs touch at unit stride, so the read-modify-write
        // side stays 1 forward stream. Windows overlap whenever `stride < kernel`, which is why
        // this accumulates instead of copying
        for kk in 0..k_plane {
            let off = kk * in_plane;
            let kbase = kk * filters;
            for i in 0..in_plane {
                let dst = offsets[off + i];
                let src = i * k_total + kbase;
                for f in 0..filters {
                    pad_out[dst + f] += dcol[src + f];
                }
            }
        }

        let mut out_b = if padded_sp != out_sp {
            crop_padded(&pad_out, 1, out_sp, padded_sp, pad_before, filters)
        } else {
            pad_out
        };
        // The bias reaches every output position, including any that no input position wrote.
        // Adding it after the crop keeps these positions at exactly the bias
        for position in out_b.chunks_exact_mut(filters) {
            for (value, &b_f) in position.iter_mut().zip(bias) {
                *value += b_f;
            }
        }
        out_b
    };

    let gemm_flops = 2usize
        .saturating_mul(batch)
        .saturating_mul(in_plane)
        .saturating_mul(cin)
        .saturating_mul(k_total);
    let (parallel, gemm_par) = parallel_plan(gemm_flops, batch);
    let per_b = map_indexed(batch, parallel, |b| process_b(b, gemm_par));

    let mut out_flat = Vec::with_capacity(batch * out_plane * filters);
    for out_b in per_b {
        out_flat.extend(out_b);
    }

    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.extend_from_slice(out_sp);
    out_shape.push(filters);
    Ok(ArrayD::from_shape_vec(IxDyn(&out_shape), out_flat)
        .expect("transposed convolution output length matches shape"))
}

/// Runs the backward transposed convolution. `input` is the original forward input.
/// `grad_output` is the gradient of the transposed-convolution output, taken after the
/// activation's backward pass
///
/// # Errors
///
/// - `Error::InvalidInput` - If any input spatial size is 0, or if the kernel's input-channel
///   count differs from the input tensor's
/// - `Error::ShapeMismatch` - If `grad_output` does not have the forward output shape
pub(super) fn conv_transpose_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> Result<ConvGradients, Error> {
    let (r, batch, in_sp, cin) = split_shape(input.shape());
    check_channels(weight_shape, r, cin)?;
    let filters = weight_shape[r];
    let k_dims = &weight_shape[..r];
    let k_plane: usize = k_dims.iter().product();

    let geometry = transpose_geometry(in_sp, k_dims, strides, padding)?;
    let TransposeGeometry {
        out_sp,
        pad_before,
        padded_sp,
    } = &geometry;
    let in_plane: usize = in_sp.iter().product();
    let out_plane: usize = out_sp.iter().product();
    let in_item = in_plane * cin;
    let out_item = out_plane * filters;
    let padded_item: usize = padded_sp.iter().product::<usize>() * filters;
    let padded_strides = spatial_strides(padded_sp, filters);

    let mut expected = Vec::with_capacity(2 + r);
    expected.push(batch);
    expected.extend_from_slice(out_sp);
    expected.push(filters);
    if grad_output.shape() != expected.as_slice() {
        return Err(Error::shape_mismatch(expected, grad_output.shape()));
    }

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // The forward pass cropped the scatter buffer down to the output, so the backward pass pads
    // the output gradient back up. Every position the forward pass wrote then has a gradient to
    // read
    let padded_storage = if padded_sp != out_sp {
        Some(build_padded(
            grad_flat, batch, out_sp, padded_sp, pad_before, filters,
        ))
    } else {
        None
    };
    let padded: &[f32] = padded_storage.as_deref().unwrap_or(grad_flat);

    let k_total = k_plane * filters;
    let offsets = im2col_offsets(in_sp, k_dims, strides, &padded_strides);
    let w_mat =
        ArrayView2::from_shape((k_total, cin), weights).expect("weights length matches [k*F, Cin]");

    let ctx = ColContext {
        padded,
        cin: filters,
        padded_item,
        k_plane,
        out_plane: in_plane,
        offsets: &offsets,
    };
    let process_b = |b: usize, gemm_par: Parallelism| -> (Array2<f32>, Vec<f32>, Vec<f32>) {
        // `[in_plane, k*F]`, the exact matrix the forward pass scattered from. Gathering it back
        // is what makes the 2 GEMMs below plain products
        let col = build_col_range(&ctx, b, 0, in_plane);
        let col_mat = ArrayView2::from_shape((in_plane, k_total), &col)
            .expect("col length matches [in_plane, k*F]");
        let x_mat =
            ArrayView2::from_shape((in_plane, cin), &in_flat[b * in_item..(b + 1) * in_item])
                .expect("input slice matches [in_plane, Cin]");

        // Weight gradient `[k*F, Cin]`, already the layer's weight layout, and input gradient
        // `[in_plane, Cin]`
        let wg = dot_par(&col_mat.t(), &x_mat, gemm_par);
        let dx = dot_par(&col_mat, &w_mat, gemm_par);

        // The bias reached every output position once, so its gradient is the plain sum over
        // positions of the unpadded output gradient
        let g_mat = ArrayView2::from_shape(
            (out_plane, filters),
            &grad_flat[b * out_item..(b + 1) * out_item],
        )
        .expect("grad slice matches [out_plane, F]");
        let bias_p: Vec<f32> = g_mat.sum_axis(Axis(0)).to_vec();

        (wg, bias_p, dx.into_raw_vec_and_offset().0)
    };

    // Each item runs 2 GEMMs (weight gradient and input gradient) of about
    // `2 * in_plane * Cin * k_total` FLOPs apiece
    let gemm_flops = 4usize
        .saturating_mul(batch)
        .saturating_mul(in_plane)
        .saturating_mul(cin)
        .saturating_mul(k_total);
    let (parallel, gemm_par) = parallel_plan(gemm_flops, batch);
    let per_b = map_indexed(batch, parallel, |b| process_b(b, gemm_par));

    let mut weight_grad_arr = Array2::<f32>::zeros((k_total, cin));
    let mut bias_grad = vec![0.0f32; filters];
    let mut in_grad_flat = Vec::with_capacity(batch * in_item);
    for (wg, bias_p, dx_b) in per_b {
        weight_grad_arr += &wg;
        for (acc, v) in bias_grad.iter_mut().zip(bias_p) {
            *acc += v;
        }
        in_grad_flat.extend(dx_b);
    }

    let mut ig_shape = Vec::with_capacity(2 + r);
    ig_shape.push(batch);
    ig_shape.extend_from_slice(in_sp);
    ig_shape.push(cin);

    Ok(ConvGradients {
        weight_grad: weight_grad_arr.into_raw_vec_and_offset().0,
        bias_grad,
        input_grad: ArrayD::from_shape_vec(IxDyn(&ig_shape), in_grad_flat)
            .expect("input grad matches shape"),
    })
}

/// Tests the output-length formula, the geometry derivation, and the forward scatter against
/// hand-derived values.
#[cfg(test)]
mod tests {
    use super::*;

    // transpose_output_length

    /// `Valid` at stride 1 grows the axis by `kernel - 1`, which is what the matching convolution
    /// removed
    #[test]
    fn test_transpose_output_length_valid_stride_one() {
        assert_eq!(transpose_output_length(4, 3, 1, PaddingType::Valid), 6);
    }

    /// `Valid` at a stride below the kernel spaces the windows out and still overlaps them
    #[test]
    fn test_transpose_output_length_valid_strided() {
        assert_eq!(transpose_output_length(4, 3, 2, PaddingType::Valid), 9);
    }

    /// A kernel no wider than the stride cannot reach past `input * stride`, so the `max` clamps
    #[test]
    fn test_transpose_output_length_valid_kernel_below_stride() {
        assert_eq!(transpose_output_length(4, 2, 3, PaddingType::Valid), 12);
    }

    /// `Same` scales the axis by the stride, whatever the kernel size
    #[test]
    fn test_transpose_output_length_same() {
        assert_eq!(transpose_output_length(4, 3, 2, PaddingType::Same), 8);
        assert_eq!(transpose_output_length(4, 7, 2, PaddingType::Same), 8);
        assert_eq!(transpose_output_length(5, 3, 1, PaddingType::Same), 5);
    }

    // transpose_geometry

    /// `Valid` needs no crop: the scatter buffer is already the output
    #[test]
    fn test_transpose_geometry_valid_needs_no_crop() {
        let g = transpose_geometry(&[4], &[3], &[2], PaddingType::Valid).unwrap();
        assert_eq!(g.out_sp, vec![9]);
        assert_eq!(g.pad_before, vec![0]);
        assert_eq!(g.padded_sp, vec![9]);
    }

    /// `Same` at stride 1 with an odd kernel crops half the kernel off each end
    #[test]
    fn test_transpose_geometry_same_stride_one() {
        let g = transpose_geometry(&[4], &[3], &[1], PaddingType::Same).unwrap();
        assert_eq!(g.out_sp, vec![4]);
        assert_eq!(g.pad_before, vec![1]);
        assert_eq!(g.padded_sp, vec![6]);
    }

    /// `Same` at stride 2 with an even kernel crops 1 leading cell
    #[test]
    fn test_transpose_geometry_same_even_kernel() {
        let g = transpose_geometry(&[3], &[4], &[2], PaddingType::Same).unwrap();
        assert_eq!(g.out_sp, vec![6]);
        assert_eq!(g.pad_before, vec![1]);
        assert_eq!(g.padded_sp, vec![8]);
    }

    /// The geometry function rejects a 0-size spatial axis instead of letting the padding math
    /// underflow
    #[test]
    fn test_transpose_geometry_rejects_empty_axis() {
        let result = transpose_geometry(&[0], &[3], &[1], PaddingType::Same);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    // conv_transpose_forward: hand-derived values
    //
    // These tests check the forward pass against numbers worked out by hand from the scatter
    // definition. A gradient check compares a layer against a finite difference of itself, so it
    // agrees even when an axis is transposed. Only an independently derived value catches a moved
    // axis.

    /// 1-D, 1 batch item, 2 input positions at 1 channel, a length-2 all-ones kernel, 1 filter,
    /// stride 1, `Valid`. Input `[1, 2]` scatters to `[1, 1+2, 2]`, and the bias lands once per
    /// output position
    #[test]
    fn test_conv_transpose_forward_1d_hand_derived() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![1.0, 2.0]).unwrap();
        // `[k=2, F=1, Cin=1]`
        let weights = vec![1.0f32, 1.0];
        let bias = [0.5f32];

        let out = conv_transpose_forward(
            &input,
            &weights,
            &[2, 1, 1],
            &bias,
            &[1],
            PaddingType::Valid,
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 3, 1]);
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![1.5, 3.5, 2.5]
        );
    }

    /// A stride above the kernel leaves gaps that no input position reaches. These positions hold
    /// exactly the bias
    #[test]
    fn test_conv_transpose_forward_gap_positions_hold_only_the_bias() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![1.0, 2.0]).unwrap();
        // `[k=1, F=1, Cin=1]`, stride 3, so output length is 2 * 3 = 6
        let weights = vec![10.0f32];
        let bias = [0.25f32];

        let out = conv_transpose_forward(
            &input,
            &weights,
            &[1, 1, 1],
            &bias,
            &[3],
            PaddingType::Valid,
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 6, 1]);
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![10.25, 0.25, 0.25, 20.25, 0.25, 0.25]
        );
    }

    /// The filter axis is the fastest-varying output axis. A second filter holding twice the
    /// first one's weights must land interleaved with it, not in a separate plane
    #[test]
    fn test_conv_transpose_forward_filter_axis_is_innermost() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![3.0]).unwrap();
        // `[k=1, F=2, Cin=1]`. Filter 0 weighs 1, filter 1 weighs 2
        let weights = vec![1.0f32, 2.0];
        let bias = [0.0f32, 0.0];

        let out = conv_transpose_forward(
            &input,
            &weights,
            &[1, 2, 1],
            &bias,
            &[1],
            PaddingType::Valid,
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 1, 2]);
        assert_eq!(out.iter().copied().collect::<Vec<f32>>(), vec![3.0, 6.0]);
    }

    /// The transposed pass is the adjoint of the plain pass. For any `x` and `y`,
    /// `<transpose(x), y> == <x, conv(y)>` holds when both use the same kernel, stride, and
    /// padding. This is the property that makes the geometry above correct, and it fails for any
    /// off-by-one in the padding
    #[test]
    fn test_conv_transpose_forward_is_the_adjoint_of_conv_forward() {
        use super::super::convolution_engine::conv_forward;

        for (k, stride, padding) in [
            (3usize, 1usize, PaddingType::Valid),
            (3, 2, PaddingType::Valid),
            (2, 3, PaddingType::Valid),
            (3, 1, PaddingType::Same),
            (3, 2, PaddingType::Same),
            (4, 2, PaddingType::Same),
            (5, 3, PaddingType::Same),
        ] {
            let (n, cin, filters) = (4usize, 2usize, 3usize);
            let out_len = transpose_output_length(n, k, stride, padding);

            // 1 flat array serves both passes. The transposed kernel is `[k, F, Cin]`. The plain
            // convolution that runs the other way reads `filters` channels and writes `cin` of
            // them. Its `[k, Cin, F]` kernel is the same `[k, filters, cin]` block
            let weights: Vec<f32> = (0..k * filters * cin)
                .map(|v| (v as f32) * 0.25 - 1.0)
                .collect();
            let no_transpose_bias = vec![0.0f32; filters];
            let no_conv_bias = vec![0.0f32; cin];

            let x: Vec<f32> = (0..n * cin).map(|v| (v as f32) * 0.5 - 2.0).collect();
            let y: Vec<f32> = (0..out_len * filters)
                .map(|v| (v as f32) * 0.125 - 0.75)
                .collect();
            let x_t = ArrayD::from_shape_vec(IxDyn(&[1, n, cin]), x.clone()).unwrap();
            let y_t = ArrayD::from_shape_vec(IxDyn(&[1, out_len, filters]), y.clone()).unwrap();

            let tx = conv_transpose_forward(
                &x_t,
                &weights,
                &[k, filters, cin],
                &no_transpose_bias,
                &[stride],
                padding,
            )
            .unwrap();
            let cy = conv_forward(
                &y_t,
                &weights,
                &[k, filters, cin],
                &no_conv_bias,
                &[stride],
                padding,
            )
            .unwrap();

            let left: f32 = tx.iter().zip(&y).map(|(a, b)| a * b).sum();
            let right: f32 = x.iter().zip(cy.iter()).map(|(a, b)| a * b).sum();
            assert!(
                (left - right).abs() <= 1e-3 * left.abs().max(right.abs()).max(1.0),
                "adjoint identity failed at k={k}, stride={stride}, padding={padding:?}: \
                 {left} against {right}"
            );
        }
    }
}
