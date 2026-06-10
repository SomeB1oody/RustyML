//! Integration tests for Dense and Flatten layers.
//!
//! All expected values are derived from the mathematical definition or hand calculations.
//! Gradient correctness is already covered by tests/neural_network/gradient_check.rs;
//! this file focuses on forward VALUES, error paths, param counts, and get_weights shape.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2, Array3, Array4};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::traits::Layer;

use super::common::assert_allclose;

// ─── helpers ────────────────────────────────────────────────────────────────

/// Build a 2D Tensor from row-major data.
fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 3D Tensor from row-major data.
fn t3(a: usize, b: usize, c: usize, data: Vec<f32>) -> Tensor {
    Array3::from_shape_vec((a, b, c), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 4D Tensor from row-major data.
fn t4(a: usize, b: usize, c: usize, d: usize, data: Vec<f32>) -> Tensor {
    Array4::from_shape_vec((a, b, c, d), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a Dense(2→2, Linear) with an injected weight matrix and zero bias.
///
/// W shape = (2, 2), b shape = (1, 2).
fn dense_2x2_with_weights(w_flat: Vec<f32>, b_flat: Vec<f32>) -> Dense {
    let mut d = Dense::new(2, 2, Linear::new(), None).unwrap();
    let w = Array2::from_shape_vec((2, 2), w_flat).unwrap();
    let b = Array2::from_shape_vec((1, 2), b_flat).unwrap();
    d.set_weights(w, b).unwrap();
    d
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — constructor validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_new_rejects_zero_input_dim() {
    let result = Dense::new(0, 4, Linear::new(), None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for input_dim=0, got {:?}",
        result
    );
}

#[test]
fn dense_new_rejects_zero_units() {
    let result = Dense::new(4, 0, Linear::new(), None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for units=0, got {:?}",
        result
    );
}

#[test]
fn dense_new_accepts_valid_dims() {
    Dense::new(3, 5, Linear::new(), None).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — param_count
// ═══════════════════════════════════════════════════════════════════════════

/// param_count = input_dim * units + units (weights + bias elements)
#[test]
fn dense_param_count_2x2() {
    use rustyml::neural_network::layers::TrainingParameters;
    let d = Dense::new(2, 2, Linear::new(), None).unwrap();
    // 2*2 weights + 2 bias = 6
    assert_eq!(d.param_count(), TrainingParameters::Trainable(6));
}

#[test]
fn dense_param_count_3x5() {
    use rustyml::neural_network::layers::TrainingParameters;
    let d = Dense::new(3, 5, Linear::new(), None).unwrap();
    // 3*5 weights + 5 bias = 20
    assert_eq!(d.param_count(), TrainingParameters::Trainable(20));
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — forward: identity weight → output == input
// ═══════════════════════════════════════════════════════════════════════════

/// Hand derivation:
///   W = I₂ = [[1,0],[0,1]], b = [[0,0]]
///   X = [[1,2],[3,4]]
///   Z = X·W + b = [[1,2],[3,4]]
///   Linear activation: output = [[1,2],[3,4]]
#[test]
fn dense_forward_identity_weight_output_equals_input() {
    let mut d = dense_2x2_with_weights(
        vec![1.0, 0.0, 0.0, 1.0], // identity
        vec![0.0, 0.0],
    );
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_allclose(&out, &expected, 1e-6_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — forward: known non-identity weight + bias
// ═══════════════════════════════════════════════════════════════════════════

/// Hand derivation:
///   W = [[2,0],[0,3]], b = [[1,2]]
///   X = [[1,2],[3,4]]
///   Row 0: z = [1·2+2·0+1, 1·0+2·3+2] = [3, 8]
///   Row 1: z = [3·2+4·0+1, 3·0+4·3+2] = [7, 14]
///   Linear activation: output = [[3,8],[7,14]]
#[test]
fn dense_forward_known_weights_and_bias() {
    let mut d = dense_2x2_with_weights(
        vec![2.0, 0.0, 0.0, 3.0], // W = diag(2,3)
        vec![1.0, 2.0],           // b = [1, 2]
    );
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(2, 2, vec![3.0, 8.0, 7.0, 14.0]);
    assert_allclose(&out, &expected, 1e-5_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — forward: ReLU activation zeroes negative pre-activations
// ═══════════════════════════════════════════════════════════════════════════

/// Hand derivation:
///   W = [[1,-1],[1,-1]], b = [[0,0]]
///   X = [[2,1],[1,3]]
///   Row 0: z = [2·1+1·1, 2·(-1)+1·(-1)] = [3, -3] → ReLU: [3, 0]
///   Row 1: z = [1·1+3·1, 1·(-1)+3·(-1)] = [4, -4] → ReLU: [4, 0]
#[test]
fn dense_forward_relu_zeroes_negative_preactivations() {
    let mut d = Dense::new(2, 2, ReLU::new(), None).unwrap();
    let w = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 1.0, -1.0]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    d.set_weights(w, b).unwrap();

    let x = t2(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(2, 2, vec![3.0, 0.0, 4.0, 0.0]);
    assert_allclose(&out, &expected, 1e-6_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — forward: single batch item (batch_size == 1)
// ═══════════════════════════════════════════════════════════════════════════

/// Hand derivation (3→2 layer):
///   W = [[1,2],[3,4],[5,6]], b = [[0,1]]
///   X = [[1,2,3]]
///   z = [1·1+2·3+3·5, 1·2+2·4+3·6] + [0,1]
///     = [1+6+15, 2+8+18] + [0,1]
///     = [22, 28] + [0,1]
///     = [22, 29]
///   Linear: output = [[22, 29]]
#[test]
fn dense_forward_3_to_2_linear_single_row() {
    let mut d = Dense::new(3, 2, Linear::new(), None).unwrap();
    let w = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
    d.set_weights(w, b).unwrap();

    let x = t2(1, 3, vec![1.0, 2.0, 3.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(1, 2, vec![22.0, 29.0]);
    assert_allclose(&out, &expected, 1e-5_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — predict == forward in eval mode (no activation side-effects)
// ═══════════════════════════════════════════════════════════════════════════

/// Dense has no mode-dependent behaviour (no dropout, no batchnorm), so
/// predict and forward must produce identical outputs.
#[test]
fn dense_predict_equals_forward() {
    let mut d = dense_2x2_with_weights(vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -0.5]);
    let x = t2(2, 2, vec![1.0, -1.0, 0.5, 2.0]);

    let fwd = d.forward(&x).unwrap();
    let pred = d.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — error paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_forward_rejects_non_2d_input_1d() {
    let mut d = Dense::new(3, 2, Linear::new(), None).unwrap();
    let x = Array::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
    let result = d.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 1D input, got {:?}",
        result
    );
}

#[test]
fn dense_forward_rejects_non_2d_input_3d() {
    let mut d = Dense::new(2, 2, Linear::new(), None).unwrap();
    let x = t3(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let result = d.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D input, got {:?}",
        result
    );
}

#[test]
fn dense_backward_before_forward_returns_err() {
    let mut d = Dense::new(2, 2, Linear::new(), None).unwrap();
    // no forward called yet
    let grad = t2(1, 2, vec![1.0, 1.0]);
    let result = d.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

#[test]
fn dense_set_weights_wrong_weight_shape_returns_err() {
    let mut d = Dense::new(2, 2, Linear::new(), None).unwrap();
    // correct shape is (2,2); supply (3,2) → should fail
    let w_bad = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let b_ok = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let result = d.set_weights(w_bad, b_ok);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

#[test]
fn dense_set_weights_wrong_bias_shape_returns_err() {
    let mut d = Dense::new(2, 2, Linear::new(), None).unwrap();
    let w_ok = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    // correct bias shape is (1,2); supply (1,3) → should fail
    let b_bad = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
    let result = d.set_weights(w_ok, b_bad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — get_weights returns Dense variant with correct shapes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_get_weights_returns_dense_variant_with_correct_shapes() {
    let mut d = Dense::new(3, 4, Linear::new(), None).unwrap();
    // Inject known weights so we can assert exact values too.
    let w = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let b = Array2::from_shape_vec((1, 4), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    d.set_weights(w.clone(), b.clone()).unwrap();

    match d.get_weights() {
        LayerWeight::Dense(lw) => {
            assert_eq!(lw.weight.shape(), &[3, 4]);
            assert_eq!(lw.bias.shape(), &[1, 4]);
            // Spot-check values
            assert_abs_diff_eq!(lw.weight[[0, 0]], 1.0_f32, epsilon = 1e-6);
            assert_abs_diff_eq!(lw.weight[[2, 3]], 12.0_f32, epsilon = 1e-6);
            assert_abs_diff_eq!(lw.bias[[0, 1]], 0.2_f32, epsilon = 1e-6);
        }
        _ => panic!("expected LayerWeight::Dense, got a different variant"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — backward restores correct grad shape after forward
// ═══════════════════════════════════════════════════════════════════════════

/// After a 2-row forward pass on a Dense(2→3) layer, backward(ones_like_output)
/// must return a gradient with the same shape as the input (2×2).
/// We do NOT assert specific gradient values — those are covered by gradient_check.rs.
#[test]
fn dense_backward_output_shape_matches_input() {
    let mut d = Dense::new(2, 3, Linear::new(), None).unwrap();
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = d.forward(&x).unwrap();
    let ones = Tensor::ones(out.raw_dim());
    let grad = d.backward(&ones).unwrap();
    assert_eq!(
        grad.shape(),
        x.shape(),
        "backward shape must match input shape"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — layer_type string
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_layer_type_is_dense() {
    let d = Dense::new(2, 2, Linear::new(), None).unwrap();
    assert_eq!(d.layer_type(), "Dense");
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — constructor validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn flatten_new_rejects_fewer_than_2_dims() {
    let result = Flatten::new(vec![4]);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 1-D input_shape, got {:?}",
        result
    );
}

#[test]
fn flatten_new_rejects_zero_dim() {
    let result = Flatten::new(vec![2, 0, 4]);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for zero dim, got {:?}",
        result
    );
}

#[test]
fn flatten_new_accepts_valid_3d_shape() {
    Flatten::new(vec![2, 3, 4]).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — forward VALUES: 3D input
// ═══════════════════════════════════════════════════════════════════════════

/// Input shape [2, 3, 4] → output shape [2, 12].
/// Values must be preserved in row-major (C) order:
///   batch 0 = first 12 elements, batch 1 = next 12.
#[test]
fn flatten_forward_3d_correct_shape_and_values() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = t3(2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    let out = fl.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 12]);

    // Row 0 should be 0..12, row 1 should be 12..24
    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — forward VALUES: 4D input
// ═══════════════════════════════════════════════════════════════════════════

/// Input shape [2, 2, 3, 4] → output shape [2, 24].
/// Same value preservation check as for 3D.
#[test]
fn flatten_forward_4d_correct_shape_and_values() {
    let data: Vec<f32> = (0..48).map(|v| v as f32).collect();
    let x = t4(2, 2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 2, 3, 4]).unwrap();
    let out = fl.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 24]);

    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — forward VALUES: 5D input
// ═══════════════════════════════════════════════════════════════════════════

/// Input shape [2, 2, 2, 3, 4] → output shape [2, 48].
#[test]
fn flatten_forward_5d_correct_shape_and_values() {
    use ndarray::Array5;
    let data: Vec<f32> = (0..96).map(|v| v as f32).collect();
    let x = Array5::from_shape_vec((2, 2, 2, 3, 4), data.clone())
        .unwrap()
        .into_dyn();

    let mut fl = Flatten::new(vec![2, 2, 2, 3, 4]).unwrap();
    let out = fl.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 48]);

    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — backward restores original shape and values
// ═══════════════════════════════════════════════════════════════════════════

/// After flatten.forward(x), backward(grad_flat) must return the original 3D shape
/// and the gradient values must be identical to grad_flat reshaped.
#[test]
fn flatten_backward_restores_3d_shape_and_values() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = t3(2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    let _out = fl.forward(&x).unwrap();

    // Gradient has the flattened shape [2, 12]
    let grad_flat_data: Vec<f32> = (0..24).map(|v| (v as f32) * 2.0).collect();
    let grad_flat = t2(2, 12, grad_flat_data.clone());

    let grad_input = fl.backward(&grad_flat).unwrap();

    // Shape must be restored to [2, 3, 4]
    assert_eq!(grad_input.shape(), &[2, 3, 4]);

    // Values must match the flattened gradient (same underlying data, different view)
    let gs = grad_input.as_slice().expect("grad not contiguous");
    for (i, &val) in gs.iter().enumerate() {
        assert_abs_diff_eq!(val, (i as f32) * 2.0, epsilon = 1e-6);
    }
}

/// Same for 4D.
#[test]
fn flatten_backward_restores_4d_shape_and_values() {
    let data: Vec<f32> = (0..48).map(|v| v as f32).collect();
    let x = t4(2, 2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 2, 3, 4]).unwrap();
    let _out = fl.forward(&x).unwrap();

    let grad_flat_data: Vec<f32> = (0..48).map(|v| -(v as f32)).collect();
    let grad_flat = t2(2, 24, grad_flat_data.clone());

    let grad_input = fl.backward(&grad_flat).unwrap();

    assert_eq!(grad_input.shape(), &[2, 2, 3, 4]);

    let gs = grad_input.as_slice().expect("grad not contiguous");
    for (i, &val) in gs.iter().enumerate() {
        assert_abs_diff_eq!(val, -(i as f32), epsilon = 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — error paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn flatten_forward_rejects_2d_input() {
    let mut fl = Flatten::new(vec![2, 3]).unwrap();
    let x = t2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = fl.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D input, got {:?}",
        result
    );
}

#[test]
fn flatten_forward_rejects_6d_input() {
    use ndarray::ArrayD;
    let mut fl = Flatten::new(vec![1, 2, 2, 2, 2]).unwrap();
    // Build a 6D tensor manually
    let x: Tensor = ArrayD::zeros(vec![1, 2, 2, 2, 2, 2]);
    let result = fl.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 6D input, got {:?}",
        result
    );
}

#[test]
fn flatten_backward_before_forward_returns_err() {
    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    // no forward called; the layer's input_cache is None
    let grad = t2(2, 12, vec![0.0_f32; 24]);
    let result = fl.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — predict == forward (no training-mode difference)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn flatten_predict_equals_forward() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = t3(2, 3, 4, data);

    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    let fwd = fl.forward(&x).unwrap();
    let pred = fl.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — get_weights returns Empty variant (no trainable parameters)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn flatten_get_weights_is_empty() {
    let fl = Flatten::new(vec![2, 3, 4]).unwrap();
    assert!(
        matches!(fl.get_weights(), LayerWeight::Empty),
        "Flatten must expose LayerWeight::Empty"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — param_count is NoTrainable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn flatten_param_count_is_no_trainable() {
    use rustyml::neural_network::layers::TrainingParameters;
    let fl = Flatten::new(vec![2, 3, 4]).unwrap();
    assert_eq!(fl.param_count(), TrainingParameters::NoTrainable);
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten — layer_type string
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn flatten_layer_type_is_flatten() {
    let fl = Flatten::new(vec![2, 3, 4]).unwrap();
    assert_eq!(fl.layer_type(), "Flatten");
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense — LayerWeight::Dense variant for Dense layer
// ═══════════════════════════════════════════════════════════════════════════

/// Dense.get_weights() must return a LayerWeight::Dense (not Empty or another variant).
#[test]
fn dense_get_weights_is_dense_variant() {
    let d = Dense::new(2, 3, Linear::new(), None).unwrap();
    assert!(
        matches!(d.get_weights(), LayerWeight::Dense(_)),
        "Dense must expose LayerWeight::Dense"
    );
}
