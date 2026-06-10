//! Integration tests for the AveragePooling and GlobalAveragePooling layers.
//!
//! Coverage:
//!  - AveragePooling{1D,2D,3D}: forward numerical values (hand-computed means), output shape,
//!    predict()==forward(), backward-before-forward error, wrong-rank input error,
//!    layer_type(), output_shape() string, constructor validation errors.
//!  - GlobalAveragePooling{1D,2D,3D}: forward numerical values, output shape,
//!    predict()==forward(), backward-before-forward error, wrong-rank input error, layer_type().
//!
//! Gradient correctness is NOT checked here — that is already covered by the finite-difference
//! harness in tests/neural_network/gradient_check.rs.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::pooling::average_pooling_1d::AveragePooling1D;
use rustyml::neural_network::layers::pooling::average_pooling_2d::AveragePooling2D;
use rustyml::neural_network::layers::pooling::average_pooling_3d::AveragePooling3D;
use rustyml::neural_network::layers::pooling::global_average_pooling_1d::GlobalAveragePooling1D;
use rustyml::neural_network::layers::pooling::global_average_pooling_2d::GlobalAveragePooling2D;
use rustyml::neural_network::layers::pooling::global_average_pooling_3d::GlobalAveragePooling3D;
use rustyml::neural_network::traits::Layer;

// ══════════════════════════════════════════════════════════════════════════════════════
// AveragePooling1D
// Input: [batch, channels, length]
// pooled_length = (length - pool_size) / stride + 1
// ══════════════════════════════════════════════════════════════════════════════════════

/// Forward mean values: pool_size=2, stride=2 over [1,2,3,4,5,6].
/// Windows: (1+2)/2=1.5, (3+4)/2=3.5, (5+6)/2=5.5.
#[test]
fn avg_pool_1d_forward_values_pool2_stride2() {
    let mut layer = AveragePooling1D::new(2, vec![1, 1, 6], Some(2)).unwrap();
    // Input shape [1, 1, 6], values 1..=6
    let x: Tensor = Array::from_shape_vec((1, 1, 6), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    // Output shape: (1 - 2)/2 + 1 = 3 => [1, 1, 3]
    assert_eq!(out.shape(), &[1, 1, 3]);
    assert_abs_diff_eq!(out[[0, 0, 0]], 1.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1]], 3.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 2]], 5.5_f32, epsilon = 1e-5);
}

/// pool_size=3, stride=1 over [0,1,2,3,4,5]: moving window of 3.
/// Windows: avg(0,1,2)=1.0, avg(1,2,3)=2.0, avg(2,3,4)=3.0, avg(3,4,5)=4.0.
/// pooled_length = (6-3)/1+1 = 4.
#[test]
fn avg_pool_1d_forward_values_pool3_stride1() {
    let mut layer = AveragePooling1D::new(3, vec![1, 2, 6], Some(1)).unwrap();
    // Two channels, both have the same [0,1,2,3,4,5] values.
    let vals: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let mut data = vec![0.0f32; 2 * 6];
    data[..6].copy_from_slice(&vals);
    data[6..].copy_from_slice(&vals);
    let x: Tensor = Array::from_shape_vec((1, 2, 6), data).unwrap().into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
    for c in 0..2 {
        assert_abs_diff_eq!(out[[0, c, 0]], 1.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, c, 1]], 2.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, c, 2]], 3.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, c, 3]], 4.0_f32, epsilon = 1e-5);
    }
}

/// Two-batch input: each batch has independent values; pooling acts per-batch.
/// Batch 0 channel 0: [2,4,6,8], pool=2, stride=2 → [3.0, 7.0].
/// Batch 1 channel 0: [1,3,5,7], pool=2, stride=2 → [2.0, 6.0].
#[test]
fn avg_pool_1d_forward_multi_batch() {
    let mut layer = AveragePooling1D::new(2, vec![2, 1, 4], Some(2)).unwrap();
    let x: Tensor =
        Array::from_shape_vec((2, 1, 4), vec![2.0f32, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0])
            .unwrap()
            .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1, 2]);
    // Batch 0
    assert_abs_diff_eq!(out[[0, 0, 0]], 3.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1]], 7.0_f32, epsilon = 1e-5);
    // Batch 1
    assert_abs_diff_eq!(out[[1, 0, 0]], 2.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 0, 1]], 6.0_f32, epsilon = 1e-5);
}

/// predict() must return the same result as forward() (no training state written).
#[test]
fn avg_pool_1d_predict_equals_forward() {
    let mut layer = AveragePooling1D::new(2, vec![1, 1, 6], Some(2)).unwrap();
    let x: Tensor = Array::from_shape_vec((1, 1, 6), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let fwd = layer.forward(&x).unwrap();
    // predict() is on &self so we build a fresh one (no forward cache) — they must agree.
    let layer_pred = AveragePooling1D::new(2, vec![1, 1, 6], Some(2)).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() must return Error::NeuralNetwork(NnError::ForwardPassNotRun).
#[test]
fn avg_pool_1d_backward_before_forward_errors() {
    let mut layer = AveragePooling1D::new(2, vec![1, 1, 4], Some(2)).unwrap();
    let grad: Tensor = Array::ones((1, 1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// forward() with wrong-rank input must return an error.
#[test]
fn avg_pool_1d_wrong_rank_input_errors() {
    let mut layer = AveragePooling1D::new(2, vec![1, 1, 4], Some(2)).unwrap();
    // 2D input instead of 3D
    let bad: Tensor = Array::ones((4, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    // predict also validates rank
    let layer_pred = AveragePooling1D::new(2, vec![1, 1, 4], Some(2)).unwrap();
    assert!(layer_pred.predict(&bad).is_err());
}

/// Constructor rejects pool_size=0.
#[test]
fn avg_pool_1d_constructor_rejects_zero_pool_size() {
    let result = AveragePooling1D::new(0, vec![1, 1, 6], Some(1));
    assert!(result.is_err(), "pool_size=0 must be rejected");
}

/// Constructor rejects stride=0.
#[test]
fn avg_pool_1d_constructor_rejects_zero_stride() {
    let result = AveragePooling1D::new(2, vec![1, 1, 6], Some(0));
    assert!(result.is_err(), "stride=0 must be rejected");
}

/// Constructor rejects pool_size larger than input length.
#[test]
fn avg_pool_1d_constructor_rejects_pool_size_larger_than_length() {
    // pool_size=8 > length=6
    let result = AveragePooling1D::new(8, vec![1, 1, 6], Some(1));
    assert!(result.is_err(), "pool_size > length must be rejected");
}

/// Constructor rejects non-3D input_shape.
#[test]
fn avg_pool_1d_constructor_rejects_wrong_input_shape_dims() {
    let result = AveragePooling1D::new(2, vec![1, 6], Some(2));
    assert!(result.is_err(), "2D input_shape must be rejected");
}

/// layer_type() and output_shape() return correct strings.
#[test]
fn avg_pool_1d_layer_type_and_output_shape() {
    let layer = AveragePooling1D::new(2, vec![1, 1, 6], Some(2)).unwrap();
    assert_eq!(layer.layer_type(), "AveragePooling1D");
    // output_shape: pool_size=2, stride=2, input=[1,1,6] -> output=[1,1,3]
    let shape_str = layer.output_shape();
    assert_eq!(shape_str, "(1, 1, 3)");
}

/// Stride defaults to pool_size when None is passed.
#[test]
fn avg_pool_1d_default_stride_equals_pool_size() {
    // With pool_size=3 and default stride=3, output_length = (6-3)/3+1 = 2
    let layer = AveragePooling1D::new(3, vec![1, 1, 6], None).unwrap();
    assert_eq!(layer.output_shape(), "(1, 1, 2)");
}

// ══════════════════════════════════════════════════════════════════════════════════════
// AveragePooling2D
// Input: [batch, channels, height, width]
// pooled_h = (h - pool_h) / stride_h + 1
// pooled_w = (w - pool_w) / stride_w + 1
// ══════════════════════════════════════════════════════════════════════════════════════

/// 4x4 input, pool=(2,2), stride=(2,2).
/// Values: row-major 0..15 → windows:
///   TL: (0+1+4+5)/4=2.5,   TR: (2+3+6+7)/4=4.5
///   BL: (8+9+12+13)/4=10.5, BR: (10+11+14+15)/4=12.5
#[test]
fn avg_pool_2d_forward_values_pool2x2_stride2x2() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    let vals: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 4, 4), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 2.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], 4.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 10.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 12.5_f32, epsilon = 1e-5);
}

/// Non-square pool and stride: pool=(2,3), stride=(1,1) on a [1,1,3,5] input.
/// Input values (row-major): 0..14.
/// Output shape: h=(3-2)/1+1=2, w=(5-3)/1+1=3  → [1,1,2,3].
/// Row 0 (h=0):
///   w=0: avg(0,1,2,5,6,7)=21/6=3.5
///   w=1: avg(1,2,3,6,7,8)=27/6=4.5
///   w=2: avg(2,3,4,7,8,9)=33/6=5.5
/// Row 1 (h=1):
///   w=0: avg(5,6,7,10,11,12)=51/6=8.5
///   w=1: avg(6,7,8,11,12,13)=57/6=9.5
///   w=2: avg(7,8,9,12,13,14)=63/6=10.5
#[test]
fn avg_pool_2d_forward_values_nonsquare_pool() {
    let mut layer = AveragePooling2D::new((2, 3), vec![1, 1, 3, 5], Some((1, 1))).unwrap();
    let vals: Vec<f32> = (0..15).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 3, 5), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 3]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 3.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], 4.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 0, 2]], 5.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 8.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 9.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 2]], 10.5_f32, epsilon = 1e-5);
}

/// Multi-channel 2D pooling: channels are pooled independently.
/// Two channels on a [1,2,4,4] input (ch0=all 2.0, ch1=all 4.0), pool=(2,2), stride=(2,2).
/// Mean of 4×2.0=2.0, mean of 4×4.0=4.0.
#[test]
fn avg_pool_2d_forward_multi_channel_independence() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 2, 4, 4], Some((2, 2))).unwrap();
    let mut vals = vec![2.0f32; 16];
    vals.extend(vec![4.0f32; 16]);
    let x: Tensor = Array::from_shape_vec((1, 2, 4, 4), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(out[[0, 0, i, j]], 2.0_f32, epsilon = 1e-5);
            assert_abs_diff_eq!(out[[0, 1, i, j]], 4.0_f32, epsilon = 1e-5);
        }
    }
}

/// predict() must produce the same values as forward() for AveragePooling2D.
#[test]
fn avg_pool_2d_predict_equals_forward() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    let vals: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 4, 4), vals)
        .unwrap()
        .into_dyn();
    let fwd = layer.forward(&x).unwrap();
    let layer_pred = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() must return ForwardPassNotRun.
#[test]
fn avg_pool_2d_backward_before_forward_errors() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    let grad: Tensor = Array::ones((1, 1, 2, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Wrong-rank input to forward/predict must error.
#[test]
fn avg_pool_2d_wrong_rank_input_errors() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    let bad: Tensor = Array::ones((4, 4, 4)).into_dyn(); // 3D, not 4D
    assert!(layer.forward(&bad).is_err());
    let layer_pred = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    assert!(layer_pred.predict(&bad).is_err());
}

/// Constructor rejects non-4D input_shape.
#[test]
fn avg_pool_2d_constructor_rejects_wrong_dims() {
    let result = AveragePooling2D::new((2, 2), vec![1, 4, 4], Some((2, 2)));
    assert!(result.is_err(), "3D input_shape must be rejected");
}

/// Constructor rejects zero stride.
#[test]
fn avg_pool_2d_constructor_rejects_zero_stride() {
    let result = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((0, 2)));
    assert!(result.is_err(), "zero height stride must be rejected");
    let result2 = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 0)));
    assert!(result2.is_err(), "zero width stride must be rejected");
}

/// Constructor rejects pool_size larger than spatial dimensions.
#[test]
fn avg_pool_2d_constructor_rejects_oversized_pool() {
    let result = AveragePooling2D::new((5, 2), vec![1, 1, 4, 4], Some((1, 1)));
    assert!(result.is_err(), "pool_h > input_h must be rejected");
}

/// Constructor rejects a zero batch or channel dimension (regression: these used to slip
/// through `new()` and only surface at forward).
#[test]
fn avg_pool_2d_constructor_rejects_zero_batch_or_channel() {
    assert!(
        AveragePooling2D::new((2, 2), vec![0, 1, 4, 4], Some((2, 2))).is_err(),
        "zero batch dimension must be rejected"
    );
    assert!(
        AveragePooling2D::new((2, 2), vec![1, 0, 4, 4], Some((2, 2))).is_err(),
        "zero channel dimension must be rejected"
    );
}

/// layer_type() and output_shape() return correct strings.
#[test]
fn avg_pool_2d_layer_type_and_output_shape() {
    let layer = AveragePooling2D::new((2, 2), vec![1, 1, 4, 4], Some((2, 2))).unwrap();
    assert_eq!(layer.layer_type(), "AveragePooling2D");
    // output: h=(4-2)/2+1=2, w=(4-2)/2+1=2 → (1,1,2,2)
    assert_eq!(layer.output_shape(), "(1, 1, 2, 2)");
}

/// Strides default to pool_size when None is passed.
#[test]
fn avg_pool_2d_default_stride_equals_pool_size() {
    // pool=(2,2), default stride=(2,2), input [1,1,6,6] → output [1,1,3,3]
    let layer = AveragePooling2D::new((2, 2), vec![1, 1, 6, 6], None).unwrap();
    assert_eq!(layer.output_shape(), "(1, 1, 3, 3)");
}

// ══════════════════════════════════════════════════════════════════════════════════════
// AveragePooling3D
// Input: [batch, channels, depth, height, width]
// pooled_d = (d - pd) / sd + 1, similarly for h, w.
// ══════════════════════════════════════════════════════════════════════════════════════

/// [1,1,2,2,2] input with values 0..7, single (2,2,2) window covering the whole volume.
/// Expected: (0+1+2+3+4+5+6+7)/8 = 28/8 = 3.5.
#[test]
fn avg_pool_3d_forward_values_single_window() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 2, 2, 2], Some((1, 1, 1))).unwrap();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], 3.5_f32, epsilon = 1e-5);
}

/// [1,1,4,2,2] input, pool=(2,2,2), stride_d=2.
/// d_out=(4-2)/2+1=2, h_out=(2-2)/1+1=1, w_out=(2-2)/1+1=1 → output [1,1,2,1,1].
/// Input values (row-major, depth-major): 0..15.
/// Window at d_start=0: covers d∈{0,1}, h∈{0,1}, w∈{0,1} → values 0..7, mean=3.5.
/// Window at d_start=2: covers d∈{2,3}, h∈{0,1}, w∈{0,1} → values 8..15, mean=11.5.
#[test]
fn avg_pool_3d_forward_values_two_depth_windows() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 2, 2], Some((2, 1, 1))).unwrap();
    let vals: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 4, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    // d_out = (4-2)/2+1 = 2, h_out = (2-2)/1+1 = 1, w_out = (2-2)/1+1 = 1
    assert_eq!(out.shape(), &[1, 1, 2, 1, 1]);
    // Window at depth_start=0: covers d∈{0,1}, h∈{0,1}, w∈{0,1} → values 0..7
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], 3.5_f32, epsilon = 1e-5);
    // Window at depth_start=2: covers d∈{2,3}, h∈{0,1}, w∈{0,1} → values 8..15
    assert_abs_diff_eq!(out[[0, 0, 1, 0, 0]], 11.5_f32, epsilon = 1e-5);
}

/// predict() must produce the same values as forward() for AveragePooling3D.
#[test]
fn avg_pool_3d_predict_equals_forward() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 2, 2, 2], Some((1, 1, 1))).unwrap();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let fwd = layer.forward(&x).unwrap();
    let layer_pred =
        AveragePooling3D::new((2, 2, 2), vec![1, 1, 2, 2, 2], Some((1, 1, 1))).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun for AveragePooling3D.
#[test]
fn avg_pool_3d_backward_before_forward_errors() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], Some((2, 2, 2))).unwrap();
    let grad: Tensor = Array::ones((1, 1, 2, 2, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Wrong-rank input (4D instead of 5D) must error.
#[test]
fn avg_pool_3d_wrong_rank_input_errors() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], Some((2, 2, 2))).unwrap();
    let bad: Tensor = Array::ones((1, 1, 4, 4)).into_dyn(); // 4D not 5D
    assert!(layer.forward(&bad).is_err());
    let layer_pred =
        AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], Some((2, 2, 2))).unwrap();
    assert!(layer_pred.predict(&bad).is_err());
}

/// Constructor rejects non-5D input_shape.
#[test]
fn avg_pool_3d_constructor_rejects_wrong_dims() {
    let result = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4], Some((2, 2, 2)));
    assert!(result.is_err(), "4D input_shape must be rejected");
}

/// Constructor rejects zero strides.
#[test]
fn avg_pool_3d_constructor_rejects_zero_stride() {
    let result = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], Some((0, 2, 2)));
    assert!(result.is_err(), "zero depth stride must be rejected");
}

/// Constructor rejects a zero batch or channel dimension (regression: these used to slip
/// through `new()` and only surface at forward).
#[test]
fn avg_pool_3d_constructor_rejects_zero_batch_or_channel() {
    assert!(
        AveragePooling3D::new((2, 2, 2), vec![0, 1, 4, 4, 4], Some((2, 2, 2))).is_err(),
        "zero batch dimension must be rejected"
    );
    assert!(
        AveragePooling3D::new((2, 2, 2), vec![1, 0, 4, 4, 4], Some((2, 2, 2))).is_err(),
        "zero channel dimension must be rejected"
    );
}

/// layer_type() and output_shape() return correct strings.
#[test]
fn avg_pool_3d_layer_type_and_output_shape() {
    let layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], Some((2, 2, 2))).unwrap();
    assert_eq!(layer.layer_type(), "AveragePooling3D");
    // d=(4-2)/2+1=2, h=2, w=2 → (1,1,2,2,2)
    assert_eq!(layer.output_shape(), "(1, 1, 2, 2, 2)");
}

/// Strides default to pool_size when None is passed for 3D.
#[test]
fn avg_pool_3d_default_stride_equals_pool_size() {
    // pool=(2,2,2), default stride=(2,2,2), input [1,1,4,4,4] → output [1,1,2,2,2]
    let layer = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();
    assert_eq!(layer.output_shape(), "(1, 1, 2, 2, 2)");
}

// ══════════════════════════════════════════════════════════════════════════════════════
// GlobalAveragePooling1D
// Input: [batch, channels, length] → output [batch, channels]
// Each (b, c) output = mean over the length dimension.
// ══════════════════════════════════════════════════════════════════════════════════════

/// Channel 0 = [1,2,3,4,5] → mean=3.0. Channel 1 = [10,20,30,40,50] → mean=30.0.
#[test]
fn global_avg_pool_1d_forward_distinct_channels() {
    let mut layer = GlobalAveragePooling1D::new();
    // Shape [1, 2, 5]: batch=1, channels=2, length=5.
    let vals = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, // channel 0
        10.0, 20.0, 30.0, 40.0, 50.0, // channel 1
    ];
    let x: Tensor = Array::from_shape_vec((1, 2, 5), vals).unwrap().into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], 30.0_f32, epsilon = 1e-5);
}

/// All-ones input: output must be all 1.0 regardless of shape.
#[test]
fn global_avg_pool_1d_forward_all_ones() {
    let mut layer = GlobalAveragePooling1D::new();
    let x: Tensor = Array::ones((3, 4, 7)).into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[3, 4]);
    for b in 0..3 {
        for c in 0..4 {
            assert_abs_diff_eq!(out[[b, c]], 1.0_f32, epsilon = 1e-5);
        }
    }
}

/// Multi-batch: batch 0 channel 0 = [0,2,4,6], mean=3.0. Batch 1 channel 0 = [1,3,5,7], mean=4.0.
#[test]
fn global_avg_pool_1d_forward_multi_batch() {
    let mut layer = GlobalAveragePooling1D::new();
    let vals = vec![
        0.0f32, 2.0, 4.0, 6.0, // batch 0, channel 0
        1.0, 3.0, 5.0, 7.0, // batch 1, channel 0
    ];
    let x: Tensor = Array::from_shape_vec((2, 1, 4), vals).unwrap().into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 0]], 4.0_f32, epsilon = 1e-5);
}

/// predict() must equal forward() for GlobalAveragePooling1D.
#[test]
fn global_avg_pool_1d_predict_equals_forward() {
    let mut layer_fwd = GlobalAveragePooling1D::new();
    let layer_pred = GlobalAveragePooling1D::new();
    let x: Tensor = Array::from_shape_vec(
        (1, 2, 5),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
    )
    .unwrap()
    .into_dyn();
    let fwd = layer_fwd.forward(&x).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() must return ForwardPassNotRun for GlobalAveragePooling1D.
#[test]
fn global_avg_pool_1d_backward_before_forward_errors() {
    let mut layer = GlobalAveragePooling1D::new();
    let grad: Tensor = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Wrong-rank input (2D instead of 3D) must error.
#[test]
fn global_avg_pool_1d_wrong_rank_input_errors() {
    let mut layer = GlobalAveragePooling1D::new();
    let bad: Tensor = Array::ones((3, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = GlobalAveragePooling1D::new();
    assert!(layer_pred.predict(&bad).is_err());
}

/// layer_type() returns correct string.
#[test]
fn global_avg_pool_1d_layer_type() {
    let layer = GlobalAveragePooling1D::new();
    assert_eq!(layer.layer_type(), "GlobalAveragePooling1D");
}

/// output_shape() returns "(batch, channels)" after forward has run, "Unknown" before.
#[test]
fn global_avg_pool_1d_output_shape() {
    let layer_new = GlobalAveragePooling1D::new();
    assert_eq!(layer_new.output_shape(), "Unknown");

    let mut layer = GlobalAveragePooling1D::new();
    let x: Tensor = Array::ones((2, 3, 5)).into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(2, 3)");
}

// ══════════════════════════════════════════════════════════════════════════════════════
// GlobalAveragePooling2D
// Input: [batch, channels, height, width] → output [batch, channels]
// Each (b, c) output = mean over h×w.
// ══════════════════════════════════════════════════════════════════════════════════════

/// ch0=[[1,2,3],[4,5,6]], ch1=[[10,20,30],[40,50,60]].
/// ch0 mean = (1+2+3+4+5+6)/6 = 21/6 = 3.5.
/// ch1 mean = (10+20+30+40+50+60)/6 = 210/6 = 35.0.
#[test]
fn global_avg_pool_2d_forward_distinct_channels() {
    let mut layer = GlobalAveragePooling2D::new();
    let vals = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // ch 0
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // ch 1
    ];
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 3), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], 35.0_f32, epsilon = 1e-5);
}

/// All-ones 4D input produces all-ones output.
#[test]
fn global_avg_pool_2d_forward_all_ones() {
    let mut layer = GlobalAveragePooling2D::new();
    let x: Tensor = Array::ones((2, 3, 5, 5)).into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    for b in 0..2 {
        for c in 0..3 {
            assert_abs_diff_eq!(out[[b, c]], 1.0_f32, epsilon = 1e-5);
        }
    }
}

/// Multi-batch: spatial mean is computed independently per batch.
/// batch 0: all 2.0 → mean=2.0; batch 1: all 6.0 → mean=6.0.
#[test]
fn global_avg_pool_2d_forward_multi_batch() {
    let mut layer = GlobalAveragePooling2D::new();
    let mut vals = vec![2.0f32; 9]; // batch 0, ch 0, 3×3
    vals.extend(vec![6.0f32; 9]); // batch 1, ch 0, 3×3
    let x: Tensor = Array::from_shape_vec((2, 1, 3, 3), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 2.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 0]], 6.0_f32, epsilon = 1e-5);
}

/// Single-pixel spatial input: mean equals the pixel value itself.
/// Input [2, 3, 1, 1] with known values; mean of a 1×1 window = the value.
#[test]
fn global_avg_pool_2d_forward_single_spatial_pixel() {
    let mut layer = GlobalAveragePooling2D::new();
    // 2 batches, 3 channels; each spatial extent is 1×1.
    // Values: batch b, channel c → (b*3 + c + 1) as f32.
    let vals: Vec<f32> = (0..6).map(|i| (i + 1) as f32).collect();
    let x: Tensor = Array::from_shape_vec((2, 3, 1, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    // Compare each output element to the matching input value.
    for b in 0..2usize {
        for c in 0..3usize {
            let expected = (b * 3 + c + 1) as f32;
            assert_abs_diff_eq!(out[[b, c]], expected, epsilon = 1e-5);
        }
    }
}

/// predict() must equal forward() for GlobalAveragePooling2D.
#[test]
fn global_avg_pool_2d_predict_equals_forward() {
    let mut layer_fwd = GlobalAveragePooling2D::new();
    let layer_pred = GlobalAveragePooling2D::new();
    let x: Tensor = Array::ones((2, 3, 4, 4)).into_dyn();
    let fwd = layer_fwd.forward(&x).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun for GlobalAveragePooling2D.
#[test]
fn global_avg_pool_2d_backward_before_forward_errors() {
    let mut layer = GlobalAveragePooling2D::new();
    let grad: Tensor = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Wrong-rank input (3D instead of 4D) must error.
#[test]
fn global_avg_pool_2d_wrong_rank_input_errors() {
    let mut layer = GlobalAveragePooling2D::new();
    let bad: Tensor = Array::ones((2, 3, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = GlobalAveragePooling2D::new();
    assert!(layer_pred.predict(&bad).is_err());
}

/// layer_type() returns correct string.
#[test]
fn global_avg_pool_2d_layer_type() {
    let layer = GlobalAveragePooling2D::new();
    assert_eq!(layer.layer_type(), "GlobalAveragePooling2D");
}

/// output_shape() returns "Unknown" before forward, "(batch, channels)" after.
#[test]
fn global_avg_pool_2d_output_shape() {
    let layer_new = GlobalAveragePooling2D::new();
    assert_eq!(layer_new.output_shape(), "Unknown");

    let mut layer = GlobalAveragePooling2D::new();
    let x: Tensor = Array::ones((3, 4, 5, 5)).into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(3, 4)");
}

// ══════════════════════════════════════════════════════════════════════════════════════
// GlobalAveragePooling3D
// Input: [batch, channels, depth, height, width] → output [batch, channels]
// Each (b, c) output = mean over d×h×w.
// ══════════════════════════════════════════════════════════════════════════════════════

/// [1,1,2,2,2] input with values 0..7: mean = (0+1+2+3+4+5+6+7)/8 = 3.5.
#[test]
fn global_avg_pool_3d_forward_single_batch_channel() {
    let mut layer = GlobalAveragePooling3D::new();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 3.5_f32, epsilon = 1e-5);
}

/// All-ones 5D input → all-ones output.
#[test]
fn global_avg_pool_3d_forward_all_ones() {
    let mut layer = GlobalAveragePooling3D::new();
    let x: Tensor = Array::ones((2, 4, 3, 3, 3)).into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    for b in 0..2 {
        for c in 0..4 {
            assert_abs_diff_eq!(out[[b, c]], 1.0_f32, epsilon = 1e-5);
        }
    }
}

/// Multi-channel with distinct values: ch0 all 5.0, ch1 all 9.0.
#[test]
fn global_avg_pool_3d_forward_multi_channel() {
    let mut layer = GlobalAveragePooling3D::new();
    // Shape [1, 2, 2, 2, 2]: 8 voxels per channel.
    let mut vals = vec![5.0f32; 8];
    vals.extend(vec![9.0f32; 8]);
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 5.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], 9.0_f32, epsilon = 1e-5);
}

/// Two-batch case: separate volume means per batch.
/// batch 0: all 3.0 → mean 3.0; batch 1: all 7.0 → mean 7.0.
#[test]
fn global_avg_pool_3d_forward_multi_batch() {
    let mut layer = GlobalAveragePooling3D::new();
    let mut vals = vec![3.0f32; 8]; // batch 0, ch 0, 2×2×2
    vals.extend(vec![7.0f32; 8]); // batch 1, ch 0, 2×2×2
    let x: Tensor = Array::from_shape_vec((2, 1, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 0]], 7.0_f32, epsilon = 1e-5);
}

/// predict() must equal forward() for GlobalAveragePooling3D.
#[test]
fn global_avg_pool_3d_predict_equals_forward() {
    let mut layer_fwd = GlobalAveragePooling3D::new();
    let layer_pred = GlobalAveragePooling3D::new();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 1, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let fwd = layer_fwd.forward(&x).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun for GlobalAveragePooling3D.
#[test]
fn global_avg_pool_3d_backward_before_forward_errors() {
    let mut layer = GlobalAveragePooling3D::new();
    let grad: Tensor = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Wrong-rank input (4D instead of 5D) must error.
#[test]
fn global_avg_pool_3d_wrong_rank_input_errors() {
    let mut layer = GlobalAveragePooling3D::new();
    let bad: Tensor = Array::ones((1, 2, 3, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = GlobalAveragePooling3D::new();
    assert!(layer_pred.predict(&bad).is_err());
}

/// layer_type() returns correct string.
#[test]
fn global_avg_pool_3d_layer_type() {
    let layer = GlobalAveragePooling3D::new();
    assert_eq!(layer.layer_type(), "GlobalAveragePooling3D");
}

/// output_shape() returns "Unknown" before forward, "(batch, channels)" after.
#[test]
fn global_avg_pool_3d_output_shape() {
    let layer_new = GlobalAveragePooling3D::new();
    assert_eq!(layer_new.output_shape(), "Unknown");

    let mut layer = GlobalAveragePooling3D::new();
    let x: Tensor = Array::ones((2, 5, 3, 3, 3)).into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(2, 5)");
}
