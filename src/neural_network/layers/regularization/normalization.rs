//! Normalization layers (batch, group, instance, layer) and shared layout/gradient helpers
//!
//! Provides channels-first permutation utilities and macros reused across the
//! normalization layer implementations

use crate::neural_network::Tensor;
use ndarray::{Array2, Axis};
use std::borrow::Cow;

/// Returns the axis permutation that moves `channel_axis` to position 1 (channels-first), keeping
/// the batch axis at 0 and the spatial axes in their original relative order
///
/// Instance/Group normalization is layout-equivariant: normalizing a tensor with the channel at
/// `channel_axis` is identical to permuting it channels-first, normalizing, and permuting back. The
/// channels-first numeric cores assume contiguous `[batch, channel, spatial...]` layout, so the
/// public methods bracket them with [`to_channels_first`] / [`from_channels_first`]
fn channels_first_perm(ndim: usize, channel_axis: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(ndim);
    perm.push(0);
    perm.push(channel_axis);
    perm.extend((1..ndim).filter(|&ax| ax != channel_axis));
    perm
}

/// Permutes `input` so the channel axis sits at position 1, returning a contiguous owned array
///
/// For `channel_axis == 1` the input is borrowed unchanged (no copy - the common fast path)
pub(super) fn to_channels_first(input: &Tensor, channel_axis: usize) -> Cow<'_, Tensor> {
    if channel_axis == 1 {
        Cow::Borrowed(input)
    } else {
        let perm = channels_first_perm(input.ndim(), channel_axis);
        Cow::Owned(
            input
                .view()
                .permuted_axes(perm)
                .as_standard_layout()
                .to_owned(),
        )
    }
}

/// Inverse of [`to_channels_first`]: moves the channel axis from position 1 back to `channel_axis`,
/// returning a contiguous owned array
///
/// A no-op for `channel_axis == 1`
pub(super) fn from_channels_first(output_cf: Tensor, channel_axis: usize) -> Tensor {
    if channel_axis == 1 {
        return output_cf;
    }
    let ndim = output_cf.ndim();
    let fwd = channels_first_perm(ndim, channel_axis);
    // Invert the forward permutation: channels-first position `new_pos` came from original axis
    // `old_ax`, so original axis `old_ax` reads from `new_pos`
    let mut inv = vec![0usize; ndim];
    for (new_pos, &old_ax) in fwd.iter().enumerate() {
        inv[old_ax] = new_pos;
    }
    output_cf
        .view()
        .permuted_axes(inv)
        .as_standard_layout()
        .to_owned()
}

/// Channels-first group-normalization forward core (reshape + axis reductions)
///
/// `input` is contiguous channels-first `[batch, channels, spatial...]`. Each sample's channels are
/// split into `num_groups` contiguous groups and normalized within the group. Because a group is a
/// contiguous `channels_per_group * spatial` block in this layout, the whole op reduces to a flat
/// `[num_instances, group_size]` reshape plus axis-1 reductions - no per-element index arithmetic
///
/// Returns `(output, x_normalized, inv_std)`:
/// - `output` already includes the per-channel affine (`gamma * x_norm + beta`)
/// - `x_normalized` is channels-first `[batch, channels, spatial...]`, cached for backward
/// - `inv_std` is `1 / sqrt(var + epsilon)`, one value per instance `[batch * num_groups]`
pub(super) fn group_norm_forward_core(
    input: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
    beta: &Tensor,
    epsilon: f32,
) -> (Tensor, Tensor, Tensor) {
    let shape = input.shape().to_vec();
    let (batch, channels) = (shape[0], shape[1]);
    let spatial: usize = shape[2..].iter().product();
    let group_size = (channels / num_groups) * spatial;
    let num_instances = batch * num_groups;

    let input_std = input.as_standard_layout();
    // Owned [num_instances, group_size]; all the group statistics are axis-1 reductions on this
    let flat: Array2<f32> = input_std
        .to_shape((num_instances, group_size))
        .expect("contiguous channels-first reshape to [num_instances, group_size]")
        .to_owned();

    let mean = flat.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)); // [N, 1]
    let centered = &flat - &mean; // [N, group_size]
    let var = centered.mapv(|v| v * v).mean_axis(Axis(1)).unwrap(); // [N]
    let inv_std = var.mapv(|v| 1.0 / (v + epsilon).sqrt()); // [N]
    let inv_std_col = inv_std.clone().insert_axis(Axis(1)); // [N, 1]
    let normalized = &centered * &inv_std_col; // [N, group_size]

    let x_normalized = normalized
        .to_shape(shape.as_slice())
        .expect("reshape normalized back to channels-first")
        .to_owned();

    // Per-channel affine, broadcast over batch and spatial via owned [1, C, 1...] arrays
    let mut affine_shape = vec![1usize; shape.len()];
    affine_shape[1] = channels;
    let gamma_b = gamma.to_shape(affine_shape.as_slice()).unwrap().to_owned();
    let beta_b = beta.to_shape(affine_shape.as_slice()).unwrap().to_owned();
    let output = &x_normalized * &gamma_b + &beta_b;

    (output, x_normalized, inv_std.into_dyn())
}

/// Channels-first group-normalization backward core (reshape + axis reductions)
///
/// Inverse of [`group_norm_forward_core`]; all arguments are channels-first / contiguous. Uses the
/// standard group-norm input-gradient identity
/// `dx = inv_std * (g - (sum(g) + x_norm * sum(g * x_norm)) / group_size)`, where `g = grad * gamma`
///
/// Returns `(grad_input, grad_gamma, grad_beta)` with `grad_input` channels-first and the parameter
/// gradients shaped `[channels]`
pub(super) fn group_norm_backward_core(
    grad_output: &Tensor,
    x_normalized: &Tensor,
    inv_std: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let shape = grad_output.shape().to_vec();
    let (batch, channels) = (shape[0], shape[1]);
    let spatial: usize = shape[2..].iter().product();
    let group_size = (channels / num_groups) * spatial;
    let num_instances = batch * num_groups;
    let gs = group_size as f32;

    let grad_std = grad_output.as_standard_layout();
    let xnorm_std = x_normalized.as_standard_layout();

    // Parameter gradients: sum over batch and spatial per channel (reshape to [B, C, spatial])
    let go_bcs = grad_std
        .to_shape((batch, channels, spatial))
        .unwrap()
        .to_owned();
    let xn_bcs = xnorm_std
        .to_shape((batch, channels, spatial))
        .unwrap()
        .to_owned();
    let grad_beta = go_bcs.sum_axis(Axis(2)).sum_axis(Axis(0)); // [C]
    let grad_gamma = (&go_bcs * &xn_bcs).sum_axis(Axis(2)).sum_axis(Axis(0)); // [C]

    // g = grad_output * gamma, broadcast over batch/spatial via an owned [1, C, 1...] array
    let mut affine_shape = vec![1usize; shape.len()];
    affine_shape[1] = channels;
    let gamma_b = gamma.to_shape(affine_shape.as_slice()).unwrap().to_owned();
    let grad_ixdyn = grad_std.to_owned();
    let grad_xnorm = &grad_ixdyn * &gamma_b;

    // Per-instance gradient reduction on the flat [num_instances, group_size] view
    let g: Array2<f32> = grad_xnorm
        .to_shape((num_instances, group_size))
        .unwrap()
        .to_owned();
    let xhat: Array2<f32> = xnorm_std
        .to_shape((num_instances, group_size))
        .unwrap()
        .to_owned();
    let sum_g = g.sum_axis(Axis(1)).insert_axis(Axis(1)); // [N, 1]
    let sum_gxhat = (&g * &xhat).sum_axis(Axis(1)).insert_axis(Axis(1)); // [N, 1]
    let inv_std_col = inv_std.to_shape((num_instances, 1)).unwrap().to_owned(); // [N, 1]

    let combo = (&sum_g + &(&xhat * &sum_gxhat)) / gs; // [N, group_size]
    let grad_flat = (&g - &combo) * &inv_std_col; // [N, group_size]

    let grad_input = grad_flat
        .to_shape(shape.as_slice())
        .expect("reshape grad_input back to channels-first")
        .to_owned();

    (grad_input, grad_gamma.into_dyn(), grad_beta.into_dyn())
}

/// Batch Normalization layer for neural networks
pub mod batch_normalization;
/// Group Normalization layer for neural networks
pub mod group_normalization;
/// Instance Normalization layer for neural networks
pub mod instance_normalization;
/// Layer Normalization layer for neural networks
pub mod layer_normalization;

pub use batch_normalization::BatchNormalization;
pub use group_normalization::GroupNormalization;
pub use instance_normalization::InstanceNormalization;
pub use layer_normalization::{LayerNormalization, LayerNormalizationAxis};

// Macros are defined after the `mod` declarations and path-exported via a `pub(in ...) use`
// re-export, so callers import them explicitly rather than relying on textual macro ordering
/// Common implementation for `output_shape` method in normalization layers
macro_rules! normalization_layer_output_shape {
    ($self:expr) => {
        if !$self.input_shape.is_empty() {
            format!(
                "({})",
                $self
                    .input_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::from("Unknown")
        }
    };
}

pub(in crate::neural_network::layers::regularization::normalization) use normalization_layer_output_shape;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    // Helper: build a Tensor from a flat Vec and shape
    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        ArrayD::from_shape_vec(shape, data).expect("shape/data mismatch in test helper")
    }

    // Helper: assert a tensor's elements (logical order) match `expected` within `tol`
    fn assert_close(actual: &Tensor, expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (got, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= tol,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    // channels_first_perm

    /// channels_first_perm(4, 3) puts channel axis 3 at position 1, yielding [0, 3, 1, 2]
    #[test]
    fn test_channels_first_perm_4_3() {
        let perm = channels_first_perm(4, 3);
        assert_eq!(perm, vec![0usize, 3, 1, 2]);
    }

    /// channels_first_perm(4, 1) is the identity [0, 1, 2, 3] since the channel is already at position 1
    #[test]
    fn test_channels_first_perm_4_1_identity() {
        let perm = channels_first_perm(4, 1);
        assert_eq!(perm, vec![0usize, 1, 2, 3]);
    }

    /// channels_first_perm(3, 2) puts channel axis 2 at position 1, yielding [0, 2, 1]
    #[test]
    fn test_channels_first_perm_3_2() {
        let perm = channels_first_perm(3, 2);
        assert_eq!(perm, vec![0usize, 2, 1]);
    }

    // to_channels_first + from_channels_first round-trip

    /// from_channels_first(to_channels_first(x), ca) recovers x for shape [2,4,3,3] with channel_axis=3
    #[test]
    fn test_to_from_channels_first_roundtrip() {
        // Non-constant tensor so any element swap is caught
        let n: usize = 2 * 4 * 3 * 3;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[2, 4, 3, 3]);

        let channel_axis = 3usize;
        let cf = to_channels_first(&x, channel_axis);
        let recovered = from_channels_first(cf.into_owned(), channel_axis);

        // Elementwise equality: every element must return to its original slot
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    /// When channel_axis==1, to_channels_first borrows without copying and the round-trip still holds
    #[test]
    fn test_to_from_channels_first_channel_axis_1_noop() {
        let data: Vec<f32> = (0..24usize).map(|i| i as f32 * 0.5).collect();
        let x = make_tensor(data, &[2, 4, 3]);
        let channel_axis = 1usize;

        let cf = to_channels_first(&x, channel_axis);
        // Must be a Borrow (no copy)
        assert!(matches!(cf, std::borrow::Cow::Borrowed(_)));
        let recovered = from_channels_first(cf.into_owned(), channel_axis);

        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(orig, got, "noop round-trip mismatch at flat index {i}");
        }
    }

    // group_norm_forward_core

    /// num_groups=1 normalizes all C*spatial values of a sample together, then applies
    /// the per-channel affine gamma*x_norm + beta. Input [1,2,3,4]: mean=2.5, var=1.25
    #[test]
    fn test_group_norm_forward_core_single_group() {
        // [batch=1, channels=2, spatial=2], channels-first flat [1,2,3,4]
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![2.0, 3.0], &[2]);
        let beta = make_tensor(vec![10.0, 20.0], &[2]);

        let (output, x_norm, inv_std) = group_norm_forward_core(&input, 1, &gamma, &beta, 0.0);

        // x_norm = (x - 2.5) * inv_std, inv_std = 1/sqrt(1.25)
        let inv = 1.0_f32 / 1.25_f32.sqrt();
        assert_close(
            &x_norm,
            &[-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv],
            1e-5,
        );
        assert_close(&inv_std, &[inv], 1e-6);
        // channel 0 (elems 0,1) affine 2x+10; channel 1 (elems 2,3) affine 3x+20
        assert_close(
            &output,
            &[
                2.0 * (-1.5 * inv) + 10.0,
                2.0 * (-0.5 * inv) + 10.0,
                3.0 * (0.5 * inv) + 20.0,
                3.0 * (1.5 * inv) + 20.0,
            ],
            1e-5,
        );
    }

    /// num_groups=2: each group is one channel of 2 spatial values, so every group
    /// [a,b] normalizes to [-1, 1] (var=0.25, inv_std=2) independently
    #[test]
    fn test_group_norm_forward_core_two_groups() {
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![1.0, 1.0], &[2]);
        let beta = make_tensor(vec![0.0, 0.0], &[2]);

        let (output, x_norm, inv_std) = group_norm_forward_core(&input, 2, &gamma, &beta, 0.0);

        assert_close(&x_norm, &[-1.0, 1.0, -1.0, 1.0], 1e-5);
        assert_close(&inv_std, &[2.0, 2.0], 1e-6);
        assert_close(&output, &[-1.0, 1.0, -1.0, 1.0], 1e-5);
    }

    // group_norm_backward_core

    /// Input gradient for one group, checked against the hand-evaluated identity
    /// dx = inv_std * (g - (sum(g) + x_norm*sum(g*x_norm))/group_size) with
    /// grad_output = [1,0,0,0] and gamma = 1; combo works out to [0.7, 0.4, 0.1, -0.2]
    #[test]
    fn test_group_norm_backward_core_single_group() {
        let inv = 1.0_f32 / 1.25_f32.sqrt();
        let x_norm = make_tensor(
            vec![-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv],
            &[1, 2, 2],
        );
        let inv_std = make_tensor(vec![inv], &[1]);
        let grad_output = make_tensor(vec![1.0, 0.0, 0.0, 0.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![1.0, 1.0], &[2]);

        let (grad_input, grad_gamma, grad_beta) =
            group_norm_backward_core(&grad_output, &x_norm, &inv_std, 1, &gamma);

        // grad = (g - combo) * inv_std
        assert_close(
            &grad_input,
            &[0.3 * inv, -0.4 * inv, -0.1 * inv, 0.2 * inv],
            1e-5,
        );
        // grad_gamma[c] = sum(grad_output * x_norm) over the channel
        assert_close(&grad_gamma, &[-1.5 * inv, 0.0], 1e-5);
        // grad_beta[c] = sum(grad_output) over the channel
        assert_close(&grad_beta, &[1.0, 0.0], 1e-6);
    }

    /// A group of two elements is fully determined (+/-1) after normalization, so its
    /// input gradient is exactly zero for any grad_output; parameter grads are the sums
    #[test]
    fn test_group_norm_backward_core_two_groups_zero_input_grad() {
        let x_norm = make_tensor(vec![-1.0, 1.0, -1.0, 1.0], &[1, 2, 2]);
        let inv_std = make_tensor(vec![2.0, 2.0], &[2]);
        let grad_output = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![1.0, 1.0], &[2]);

        let (grad_input, grad_gamma, grad_beta) =
            group_norm_backward_core(&grad_output, &x_norm, &inv_std, 2, &gamma);

        assert_close(&grad_input, &[0.0, 0.0, 0.0, 0.0], 1e-6);
        // grad_gamma[c] = sum(grad_output * x_norm): ch0 = 1*-1+2*1 = 1; ch1 = 3*-1+4*1 = 1
        assert_close(&grad_gamma, &[1.0, 1.0], 1e-6);
        // grad_beta[c] = sum(grad_output): ch0 = 1+2 = 3; ch1 = 3+4 = 7
        assert_close(&grad_beta, &[3.0, 7.0], 1e-6);
    }
}
