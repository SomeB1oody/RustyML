use crate::neural_network::Tensor;
use std::borrow::Cow;

/// Returns the axis permutation that moves `channel_axis` to position 1 (channels-first), keeping
/// the batch axis at 0 and the spatial axes in their original relative order.
///
/// Instance/Group normalization is layout-equivariant: normalizing a tensor with the channel at
/// `channel_axis` is identical to permuting it channels-first, normalizing, and permuting back. The
/// channels-first numeric cores assume contiguous `[batch, channel, spatial...]` layout, so the
/// public methods bracket them with [`to_channels_first`] / [`from_channels_first`].
fn channels_first_perm(ndim: usize, channel_axis: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(ndim);
    perm.push(0);
    perm.push(channel_axis);
    perm.extend((1..ndim).filter(|&ax| ax != channel_axis));
    perm
}

/// Permutes `input` so the channel axis sits at position 1, returning a contiguous owned array.
/// For `channel_axis == 1` the input is borrowed unchanged (no copy — the common fast path).
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
/// returning a contiguous owned array. A no-op for `channel_axis == 1`.
pub(super) fn from_channels_first(output_cf: Tensor, channel_axis: usize) -> Tensor {
    if channel_axis == 1 {
        return output_cf;
    }
    let ndim = output_cf.ndim();
    let fwd = channels_first_perm(ndim, channel_axis);
    // Invert the forward permutation: position `new_pos` in the channels-first array came from
    // original axis `old_ax`, so the original axis `old_ax` must read from `new_pos`.
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

// Macros are defined after the `mod` declarations and path-exported via a `pub(in ...) use` re-export,
// so callers import them explicitly rather than relying on textual macro ordering.
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

/// Common implementation for computing gamma and beta gradients in normalization layers
///
/// This macro computes the gradients for gamma and beta parameters by summing over
/// all spatial dimensions and batch dimensions for each channel. It handles both
/// parallel and sequential computation based on the total number of elements.
///
/// # Parameters
///
/// - `$grad_gamma:expr` - Mutable tensor to store gamma gradients
/// - `$grad_beta:expr` - Mutable tensor to store beta gradients
/// - `$grad_output:expr` - Gradient from the next layer
/// - `$x_normalized:expr` - Normalized input from forward pass
/// - `$batch_size:expr` - Number of samples in the batch
/// - `$num_channels:expr` - Number of channels
/// - `$spatial_size:expr` - Size of spatial dimensions (product of all spatial dims)
/// - `$total_elements:expr` - Total number of elements in the tensor
/// - `$parallel_threshold:expr` - Threshold for switching to parallel computation
macro_rules! compute_normalization_layer_parameter_gradients {
    (
        $grad_gamma:expr,
        $grad_beta:expr,
        $grad_output:expr,
        $x_normalized:expr,
        $batch_size:expr,
        $num_channels:expr,
        $spatial_size:expr,
        $total_elements:expr,
        $parallel_threshold:expr
    ) => {{
        if $total_elements >= $parallel_threshold {
            // Parallel computation of parameter gradients
            // Compute both gamma and beta gradients in a single pass
            let (grad_gamma_vec, grad_beta_vec): (Vec<f32>, Vec<f32>) = (0..$num_channels)
                .into_par_iter()
                .map(|channel_idx| {
                    let mut gamma_sum = 0.0f32;
                    let mut beta_sum = 0.0f32;

                    for batch_idx in 0..$batch_size {
                        let start = (batch_idx * $num_channels + channel_idx) * $spatial_size;
                        let end = start + $spatial_size;

                        for i in start..end {
                            let grad_out = $grad_output.as_slice().unwrap()[i];
                            gamma_sum += grad_out * $x_normalized.as_slice().unwrap()[i];
                            beta_sum += grad_out;
                        }
                    }

                    (gamma_sum, beta_sum)
                })
                .unzip();

            $grad_gamma
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&grad_gamma_vec);
            $grad_beta
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&grad_beta_vec);
        } else {
            // Sequential computation
            for channel_idx in 0..$num_channels {
                let mut gamma_sum = 0.0f32;
                let mut beta_sum = 0.0f32;

                for batch_idx in 0..$batch_size {
                    let start = (batch_idx * $num_channels + channel_idx) * $spatial_size;
                    let end = start + $spatial_size;

                    for i in start..end {
                        gamma_sum += $grad_output.as_slice().unwrap()[i]
                            * $x_normalized.as_slice().unwrap()[i];
                        beta_sum += $grad_output.as_slice().unwrap()[i];
                    }
                }

                $grad_gamma.as_slice_mut().unwrap()[channel_idx] = gamma_sum;
                $grad_beta.as_slice_mut().unwrap()[channel_idx] = beta_sum;
            }
        }
    }};
}
pub(in crate::neural_network::layers::regularization::normalization) use compute_normalization_layer_parameter_gradients;
pub(in crate::neural_network::layers::regularization::normalization) use normalization_layer_output_shape;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    // Helper: build a Tensor from a flat Vec and shape.
    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        ArrayD::from_shape_vec(shape, data).expect("shape/data mismatch in test helper")
    }

    // ─── channels_first_perm ────────────────────────────────────────────────

    /// (ndim=4, channel_axis=3) → [0,3,1,2]
    ///
    /// Derivation: batch stays at 0, channel_axis=3 moves to position 1,
    /// then the spatial axes from 1..4 excluding 3, i.e. [1,2], follow in order.
    /// Expected: [0, 3, 1, 2].
    #[test]
    fn test_channels_first_perm_4_3() {
        let perm = channels_first_perm(4, 3);
        assert_eq!(perm, vec![0usize, 3, 1, 2]);
    }

    /// (ndim=4, channel_axis=1) → identity [0,1,2,3]
    ///
    /// Derivation: batch stays at 0, channel_axis=1 is already position 1,
    /// spatial axes from 1..4 excluding 1 are [2,3].
    /// Expected: [0, 1, 2, 3].
    #[test]
    fn test_channels_first_perm_4_1_identity() {
        let perm = channels_first_perm(4, 1);
        assert_eq!(perm, vec![0usize, 1, 2, 3]);
    }

    /// (ndim=3, channel_axis=2) → [0,2,1]
    ///
    /// Derivation: batch stays at 0, channel_axis=2 moves to position 1,
    /// spatial axes from 1..3 excluding 2, i.e. [1], follow.
    /// Expected: [0, 2, 1].
    #[test]
    fn test_channels_first_perm_3_2() {
        let perm = channels_first_perm(3, 2);
        assert_eq!(perm, vec![0usize, 2, 1]);
    }

    // ─── to_channels_first + from_channels_first round-trip ─────────────────

    /// Round-trip identity: from_channels_first(to_channels_first(x), ca) == x,
    /// for shape [2,4,3,3] with channel_axis=3.
    ///
    /// Derivation:
    ///   forward perm for (ndim=4, ca=3) = [0,3,1,2]   (tested above)
    ///   inverse perm: inv[0]=0, inv[3]=1, inv[1]=2, inv[2]=3 → [0,2,3,1]
    ///   Composing [0,3,1,2] then [0,2,3,1] yields the identity permutation,
    ///   so every element returns to its original position.
    #[test]
    fn test_to_from_channels_first_roundtrip() {
        // Build a recognisably non-constant tensor so any element swap would
        // be caught.
        let n: usize = 2 * 4 * 3 * 3;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[2, 4, 3, 3]);

        let channel_axis = 3usize;
        let cf = to_channels_first(&x, channel_axis);
        let recovered = from_channels_first(cf.into_owned(), channel_axis);

        // Element-wise equality: every element must return to its original slot.
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    /// When channel_axis==1, to_channels_first is a no-op borrow and
    /// from_channels_first returns the same data — round-trip still holds.
    #[test]
    fn test_to_from_channels_first_channel_axis_1_noop() {
        let data: Vec<f32> = (0..24usize).map(|i| i as f32 * 0.5).collect();
        let x = make_tensor(data, &[2, 4, 3]);
        let channel_axis = 1usize;

        let cf = to_channels_first(&x, channel_axis);
        // Must be a Borrow (no copy).
        assert!(matches!(cf, std::borrow::Cow::Borrowed(_)));
        let recovered = from_channels_first(cf.into_owned(), channel_axis);

        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(orig, got, "noop round-trip mismatch at flat index {i}");
        }
    }
}
