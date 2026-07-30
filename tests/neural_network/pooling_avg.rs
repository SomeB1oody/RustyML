//! Integration tests for the AveragePooling and GlobalAveragePooling layers, covering forward
//! values, output shape, predict()==forward(), backward-before-forward and wrong-rank errors,
//! layer_type()/output_shape(), and constructor validation.
//!
//! Gradient correctness is not checked here. The finite-difference harness in
//! tests/neural_network/gradient_check.rs covers that.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::pooling::average_pooling_1d::AveragePooling1D;
use rustyml::neural_network::layers::pooling::average_pooling_2d::AveragePooling2D;
use rustyml::neural_network::layers::pooling::average_pooling_3d::AveragePooling3D;
use rustyml::neural_network::layers::pooling::global_average_pooling_1d::GlobalAveragePooling1D;
use rustyml::neural_network::layers::pooling::global_average_pooling_2d::GlobalAveragePooling2D;
use rustyml::neural_network::layers::pooling::global_average_pooling_3d::GlobalAveragePooling3D;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

// AveragePooling1D, input [batch, length, channels]
// pooled_length = (length - pool_size) / stride + 1

/// Forward mean values for pool_size=2, stride=2 over a 1D channel
#[test]
fn avg_pool_1d_forward_values_pool2_stride2() {
    let mut layer = AveragePooling1D::new(2, vec![1, 6, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let x: Tensor = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 1]);
    // (1+2)/2, (3+4)/2, (5+6)/2
    assert_abs_diff_eq!(out[[0, 0, 0]], 1.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1, 0]], 3.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 2, 0]], 5.5_f32, epsilon = 1e-5);
}

/// Forward mean values for a moving window: pool_size=3, stride=1 over 2 identical channels
#[test]
fn avg_pool_1d_forward_values_pool3_stride1() {
    let mut layer = AveragePooling1D::new(3, vec![1, 6, 2])
        .unwrap()
        .with_stride(1)
        .unwrap();
    // Both channels carry the same values 0..5 along the length axis
    let mut data = vec![0.0f32; 6 * 2];
    for l in 0..6usize {
        for c in 0..2usize {
            data[l * 2 + c] = l as f32;
        }
    }
    let x: Tensor = Array::from_shape_vec((1, 6, 2), data).unwrap().into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 2]);
    for c in 0..2 {
        // (0+1+2)/3, (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
        assert_abs_diff_eq!(out[[0, 0, c]], 1.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, 1, c]], 2.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, 2, c]], 3.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(out[[0, 3, c]], 4.0_f32, epsilon = 1e-5);
    }
}

/// Pooling acts per-batch on independent values across a 2-batch input
#[test]
fn avg_pool_1d_forward_multi_batch() {
    let mut layer = AveragePooling1D::new(2, vec![2, 4, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let x: Tensor =
        Array::from_shape_vec((2, 4, 1), vec![2.0f32, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0])
            .unwrap()
            .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2, 1]);
    // Batch 0: (2+4)/2, (6+8)/2
    assert_abs_diff_eq!(out[[0, 0, 0]], 3.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1, 0]], 7.0_f32, epsilon = 1e-5);
    // Batch 1: (1+3)/2, (5+7)/2
    assert_abs_diff_eq!(out[[1, 0, 0]], 2.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 1, 0]], 6.0_f32, epsilon = 1e-5);
}

/// predict() returns the same result as forward()
#[test]
fn avg_pool_1d_predict_equals_forward() {
    let mut layer = AveragePooling1D::new(2, vec![1, 6, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let x: Tensor = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let fwd = layer.forward(&x).unwrap();
    // predict() is on &self, so a fresh layer (no forward cache) must still agree
    let layer_pred = AveragePooling1D::new(2, vec![1, 6, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun
#[test]
fn avg_pool_1d_backward_before_forward_errors() {
    let mut layer = AveragePooling1D::new(2, vec![1, 4, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let grad: Tensor = Array::ones((1, 2, 1)).into_dyn();
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

/// forward() and predict() reject wrong-rank input (2D instead of 3D)
#[test]
fn avg_pool_1d_wrong_rank_input_errors() {
    let mut layer = AveragePooling1D::new(2, vec![1, 4, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let bad: Tensor = Array::ones((4, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = AveragePooling1D::new(2, vec![1, 4, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    assert!(layer_pred.predict(&bad).is_err());
}

/// Constructor rejects pool_size=0
#[test]
fn avg_pool_1d_constructor_rejects_zero_pool_size() {
    let result = AveragePooling1D::new(0, vec![1, 6, 1]);
    assert!(result.is_err(), "pool_size=0 must be rejected");
}

/// Constructor rejects stride=0
#[test]
fn avg_pool_1d_constructor_rejects_zero_stride() {
    let result = AveragePooling1D::new(2, vec![1, 6, 1])
        .unwrap()
        .with_stride(0);
    assert!(result.is_err(), "stride=0 must be rejected");
}

/// Constructor rejects pool_size larger than input length
#[test]
fn avg_pool_1d_constructor_rejects_pool_size_larger_than_length() {
    let result = AveragePooling1D::new(8, vec![1, 6, 1]);
    assert!(result.is_err(), "pool_size > length must be rejected");
}

/// Constructor rejects non-3D input_shape
#[test]
fn avg_pool_1d_constructor_rejects_wrong_input_shape_dims() {
    let result = AveragePooling1D::new(2, vec![1, 6]);
    assert!(result.is_err(), "2D input_shape must be rejected");
}

/// layer_type() and output_shape() return correct strings
#[test]
fn avg_pool_1d_layer_type_and_output_shape() {
    let layer = AveragePooling1D::new(2, vec![1, 6, 1])
        .unwrap()
        .with_stride(2)
        .unwrap();
    assert_eq!(layer.layer_type(), "AveragePooling1D");
    let shape_str = layer.output_shape();
    assert_eq!(shape_str, "(1, 3, 1)");
}

/// Stride defaults to pool_size when the caller passes None
#[test]
fn avg_pool_1d_default_stride_equals_pool_size() {
    let layer = AveragePooling1D::new(3, vec![1, 6, 1]).unwrap();
    assert_eq!(layer.output_shape(), "(1, 2, 1)");
}

// AveragePooling2D, input [batch, height, width, channels]
// pooled_h = (h - pool_h) / stride_h + 1
// pooled_w = (w - pool_w) / stride_w + 1

/// Forward mean values for a 4x4 input, pool=(2,2), stride=(2,2)
#[test]
fn avg_pool_2d_forward_values_pool2x2_stride2x2() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    let vals: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 4, 4, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    // Rows are [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 2.5_f32, epsilon = 1e-5); // (0+1+4+5)/4
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 4.5_f32, epsilon = 1e-5); // (2+3+6+7)/4
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 10.5_f32, epsilon = 1e-5); // (8+9+12+13)/4
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 12.5_f32, epsilon = 1e-5); // (10+11+14+15)/4
}

/// Forward mean values for a non-square pool=(2,3) and stride=(1,1) on a [1,3,5,1] input
#[test]
fn avg_pool_2d_forward_values_nonsquare_pool() {
    let mut layer = AveragePooling2D::new((2, 3), vec![1, 3, 5, 1])
        .unwrap()
        .with_strides((1, 1))
        .unwrap();
    let vals: Vec<f32> = (0..15).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 3, 5, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3, 1]);
    // Rows are [0..4], [5..9], [10..14]. Each window is 2 rows x 3 cols.
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 3.5_f32, epsilon = 1e-5); // (0+1+2+5+6+7)/6
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 4.5_f32, epsilon = 1e-5); // (1+2+3+6+7+8)/6
    assert_abs_diff_eq!(out[[0, 0, 2, 0]], 5.5_f32, epsilon = 1e-5); // (2+3+4+7+8+9)/6
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 8.5_f32, epsilon = 1e-5); // (5+6+7+10+11+12)/6
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 9.5_f32, epsilon = 1e-5); // (6+7+8+11+12+13)/6
    assert_abs_diff_eq!(out[[0, 1, 2, 0]], 10.5_f32, epsilon = 1e-5); // (7+8+9+12+13+14)/6
}

/// The layer pools channels independently in 2D. Each channel varies over the spatial grid, so a
/// height/width/channel transposition changes the values, not just the shape.
#[test]
fn avg_pool_2d_forward_multi_channel_independence() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 4, 4, 2])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    // value(h, w, c) = 4*h + w + 100*c. Channels interleave on the last axis.
    // Channel 0 holds 0..15 row-major over the 4x4 grid. Channel 1 holds the same plus 100.
    let mut vals = Vec::with_capacity(4 * 4 * 2);
    for h in 0..4usize {
        for w in 0..4usize {
            vals.push((4 * h + w) as f32);
            vals.push((4 * h + w + 100) as f32);
        }
    }
    let x: Tensor = Array::from_shape_vec((1, 4, 4, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    // Channel 0 rows are [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 2.5_f32, epsilon = 1e-5); // (0+1+4+5)/4
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 4.5_f32, epsilon = 1e-5); // (2+3+6+7)/4
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 10.5_f32, epsilon = 1e-5); // (8+9+12+13)/4
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 12.5_f32, epsilon = 1e-5); // (10+11+14+15)/4
    // Channel 1 is channel 0 shifted by 100, so every mean shifts by exactly 100
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], 102.5_f32, epsilon = 1e-5); // (100+101+104+105)/4
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 104.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1, 0, 1]], 110.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1, 1, 1]], 112.5_f32, epsilon = 1e-5);
}

/// predict() produces the same values as forward() for AveragePooling2D
#[test]
fn avg_pool_2d_predict_equals_forward() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    let vals: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 4, 4, 1), vals)
        .unwrap()
        .into_dyn();
    let fwd = layer.forward(&x).unwrap();
    let layer_pred = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun
#[test]
fn avg_pool_2d_backward_before_forward_errors() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    let grad: Tensor = Array::ones((1, 2, 2, 1)).into_dyn();
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

/// Wrong-rank input to forward/predict errors (3D instead of 4D)
#[test]
fn avg_pool_2d_wrong_rank_input_errors() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    let bad: Tensor = Array::ones((4, 4, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    assert!(layer_pred.predict(&bad).is_err());
}

/// Constructor rejects non-4D input_shape
#[test]
fn avg_pool_2d_constructor_rejects_wrong_dims() {
    let result = AveragePooling2D::new((2, 2), vec![1, 4, 4]);
    assert!(result.is_err(), "3D input_shape must be rejected");
}

/// Constructor rejects zero stride
#[test]
fn avg_pool_2d_constructor_rejects_zero_stride() {
    let result = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((0, 2));
    assert!(result.is_err(), "zero height stride must be rejected");
    let result2 = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 0));
    assert!(result2.is_err(), "zero width stride must be rejected");
}

/// Constructor rejects pool_size larger than spatial dimensions
#[test]
fn avg_pool_2d_constructor_rejects_oversized_pool() {
    let result = AveragePooling2D::new((5, 2), vec![1, 4, 4, 1]);
    assert!(result.is_err(), "pool_h > input_h must be rejected");
}

/// Constructor rejects a zero batch or channel dimension. The check happens at construction,
/// not at forward.
#[test]
fn avg_pool_2d_constructor_rejects_zero_batch_or_channel() {
    assert!(
        AveragePooling2D::new((2, 2), vec![0, 4, 4, 1]).is_err(),
        "zero batch dimension must be rejected"
    );
    assert!(
        AveragePooling2D::new((2, 2), vec![1, 4, 4, 0]).is_err(),
        "zero channel dimension must be rejected"
    );
}

/// layer_type() and output_shape() return correct strings
#[test]
fn avg_pool_2d_layer_type_and_output_shape() {
    let layer = AveragePooling2D::new((2, 2), vec![1, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap();
    assert_eq!(layer.layer_type(), "AveragePooling2D");
    assert_eq!(layer.output_shape(), "(1, 2, 2, 1)");
}

/// Strides default to pool_size when the caller passes None
#[test]
fn avg_pool_2d_default_stride_equals_pool_size() {
    let layer = AveragePooling2D::new((2, 2), vec![1, 6, 6, 1]).unwrap();
    assert_eq!(layer.output_shape(), "(1, 3, 3, 1)");
}

// AveragePooling3D, input [batch, depth, height, width, channels]
// pooled_d = (d - pd) / sd + 1, similarly for h, w

/// Forward mean value for a single (2,2,2) window covering the whole [1,2,2,2,1] volume
#[test]
fn avg_pool_3d_forward_values_single_window() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 2, 2, 2, 1])
        .unwrap()
        .with_strides((1, 1, 1))
        .unwrap();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 2, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);
    // mean of 0..7 = 28/8
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], 3.5_f32, epsilon = 1e-5);
}

/// Forward mean values for 2 depth windows: pool=(2,2,2), stride_d=2 on a [1,4,2,2,1] input
#[test]
fn avg_pool_3d_forward_values_two_depth_windows() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 4, 2, 2, 1])
        .unwrap()
        .with_strides((2, 1, 1))
        .unwrap();
    let vals: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 4, 2, 2, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 1, 1, 1]);
    // Window at depth_start=0 covers values 0..7, mean 28/8
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], 3.5_f32, epsilon = 1e-5);
    // Window at depth_start=2 covers values 8..15, mean 92/8
    assert_abs_diff_eq!(out[[0, 1, 0, 0, 0]], 11.5_f32, epsilon = 1e-5);
}

/// predict() produces the same values as forward() for AveragePooling3D
#[test]
fn avg_pool_3d_predict_equals_forward() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 2, 2, 2, 1])
        .unwrap()
        .with_strides((1, 1, 1))
        .unwrap();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 2, 1), vals)
        .unwrap()
        .into_dyn();
    let fwd = layer.forward(&x).unwrap();
    let layer_pred = AveragePooling3D::new((2, 2, 2), vec![1, 2, 2, 2, 1])
        .unwrap()
        .with_strides((1, 1, 1))
        .unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun for AveragePooling3D
#[test]
fn avg_pool_3d_backward_before_forward_errors() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2, 2))
        .unwrap();
    let grad: Tensor = Array::ones((1, 2, 2, 2, 1)).into_dyn();
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

/// Wrong-rank input (4D instead of 5D) errors
#[test]
fn avg_pool_3d_wrong_rank_input_errors() {
    let mut layer = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2, 2))
        .unwrap();
    let bad: Tensor = Array::ones((1, 4, 4, 1)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2, 2))
        .unwrap();
    assert!(layer_pred.predict(&bad).is_err());
}

/// Constructor rejects non-5D input_shape
#[test]
fn avg_pool_3d_constructor_rejects_wrong_dims() {
    let result = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 1]);
    assert!(result.is_err(), "4D input_shape must be rejected");
}

/// Constructor rejects zero strides
#[test]
fn avg_pool_3d_constructor_rejects_zero_stride() {
    let result = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 1])
        .unwrap()
        .with_strides((0, 2, 2));
    assert!(result.is_err(), "zero depth stride must be rejected");
}

/// Constructor rejects a zero batch or channel dimension. The check happens at construction,
/// not at forward.
#[test]
fn avg_pool_3d_constructor_rejects_zero_batch_or_channel() {
    assert!(
        AveragePooling3D::new((2, 2, 2), vec![0, 4, 4, 4, 1]).is_err(),
        "zero batch dimension must be rejected"
    );
    assert!(
        AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 0]).is_err(),
        "zero channel dimension must be rejected"
    );
}

/// layer_type() and output_shape() return correct strings
#[test]
fn avg_pool_3d_layer_type_and_output_shape() {
    let layer = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 1])
        .unwrap()
        .with_strides((2, 2, 2))
        .unwrap();
    assert_eq!(layer.layer_type(), "AveragePooling3D");
    assert_eq!(layer.output_shape(), "(1, 2, 2, 2, 1)");
}

/// For 3D, strides default to pool_size when the caller passes None
#[test]
fn avg_pool_3d_default_stride_equals_pool_size() {
    let layer = AveragePooling3D::new((2, 2, 2), vec![1, 4, 4, 4, 1]).unwrap();
    assert_eq!(layer.output_shape(), "(1, 2, 2, 2, 1)");
}

// GlobalAveragePooling1D, input [batch, length, channels] -> output [batch, channels]
// Each (b, c) output = mean over the length dimension

/// Forward mean per channel over the length dimension
#[test]
fn global_avg_pool_1d_forward_distinct_channels() {
    let mut layer = GlobalAveragePooling1D::new();
    // Channel 0 carries 1..5, channel 1 carries 10..50. Channels interleave along the last axis.
    let vals = vec![
        1.0f32, 10.0, // length 0
        2.0, 20.0, // length 1
        3.0, 30.0, // length 2
        4.0, 40.0, // length 3
        5.0, 50.0, // length 4
    ];
    let x: Tensor = Array::from_shape_vec((1, 5, 2), vals).unwrap().into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-5); // 15/5
    assert_abs_diff_eq!(out[[0, 1]], 30.0_f32, epsilon = 1e-5); // 150/5
}

/// All-ones input yields all-1.0 output regardless of shape
#[test]
fn global_avg_pool_1d_forward_all_ones() {
    let mut layer = GlobalAveragePooling1D::new();
    let x: Tensor = Array::ones((3, 7, 4)).into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[3, 4]);
    for b in 0..3 {
        for c in 0..4 {
            assert_abs_diff_eq!(out[[b, c]], 1.0_f32, epsilon = 1e-5);
        }
    }
}

/// The layer computes the mean independently per batch
#[test]
fn global_avg_pool_1d_forward_multi_batch() {
    let mut layer = GlobalAveragePooling1D::new();
    let vals = vec![
        0.0f32, 2.0, 4.0, 6.0, // batch 0, channel 0
        1.0, 3.0, 5.0, 7.0, // batch 1, channel 0
    ];
    let x: Tensor = Array::from_shape_vec((2, 4, 1), vals).unwrap().into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-5); // 12/4
    assert_abs_diff_eq!(out[[1, 0]], 4.0_f32, epsilon = 1e-5); // 16/4
}

/// predict() equals forward() for GlobalAveragePooling1D
#[test]
fn global_avg_pool_1d_predict_equals_forward() {
    let mut layer_fwd = GlobalAveragePooling1D::new();
    let layer_pred = GlobalAveragePooling1D::new();
    let x: Tensor = Array::from_shape_vec(
        (1, 5, 2),
        vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
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

/// backward() before forward() returns ForwardPassNotRun for GlobalAveragePooling1D
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

/// Wrong-rank input (2D instead of 3D) errors
#[test]
fn global_avg_pool_1d_wrong_rank_input_errors() {
    let mut layer = GlobalAveragePooling1D::new();
    let bad: Tensor = Array::ones((3, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = GlobalAveragePooling1D::new();
    assert!(layer_pred.predict(&bad).is_err());
}

/// layer_type() returns correct string
#[test]
fn global_avg_pool_1d_layer_type() {
    let layer = GlobalAveragePooling1D::new();
    assert_eq!(layer.layer_type(), "GlobalAveragePooling1D");
}

/// output_shape() returns "Unknown" before forward and "(batch, channels)" after
#[test]
fn global_avg_pool_1d_output_shape() {
    let layer_new = GlobalAveragePooling1D::new();
    assert_eq!(layer_new.output_shape(), "Unknown");

    let mut layer = GlobalAveragePooling1D::new();
    // batch 2, length 5, channels 3
    let x: Tensor = Array::ones((2, 5, 3)).into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(2, 3)");
}

// GlobalAveragePooling2D, input [batch, height, width, channels] -> output [batch, channels]
// Each (b, c) output = mean over h x w

/// Forward mean per channel over the spatial dimensions
#[test]
fn global_avg_pool_2d_forward_distinct_channels() {
    let mut layer = GlobalAveragePooling2D::new();
    // 2x3 spatial extent. Channel 0 carries 1..6, channel 1 carries 10..60.
    let vals = vec![
        1.0f32, 10.0, // (h=0, w=0)
        2.0, 20.0, // (h=0, w=1)
        3.0, 30.0, // (h=0, w=2)
        4.0, 40.0, // (h=1, w=0)
        5.0, 50.0, // (h=1, w=1)
        6.0, 60.0, // (h=1, w=2)
    ];
    let x: Tensor = Array::from_shape_vec((1, 2, 3, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.5_f32, epsilon = 1e-5); // 21/6
    assert_abs_diff_eq!(out[[0, 1]], 35.0_f32, epsilon = 1e-5); // 210/6
}

/// All-ones 4D input produces all-ones output
#[test]
fn global_avg_pool_2d_forward_all_ones() {
    let mut layer = GlobalAveragePooling2D::new();
    let x: Tensor = Array::ones((2, 5, 5, 3)).into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    for b in 0..2 {
        for c in 0..3 {
            assert_abs_diff_eq!(out[[b, c]], 1.0_f32, epsilon = 1e-5);
        }
    }
}

/// The layer computes the spatial mean independently per batch
#[test]
fn global_avg_pool_2d_forward_multi_batch() {
    let mut layer = GlobalAveragePooling2D::new();
    let mut vals = vec![2.0f32; 9]; // batch 0, 3x3, ch 0
    vals.extend(vec![6.0f32; 9]); // batch 1, 3x3, ch 0
    let x: Tensor = Array::from_shape_vec((2, 3, 3, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 2.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 0]], 6.0_f32, epsilon = 1e-5);
}

/// With a single-pixel spatial extent the mean equals the pixel value itself
#[test]
fn global_avg_pool_2d_forward_single_spatial_pixel() {
    let mut layer = GlobalAveragePooling2D::new();
    // 2 batches, 1x1 spatial, 3 channels. Value at (b, c) is (b*3 + c + 1).
    let vals: Vec<f32> = (0..6).map(|i| (i + 1) as f32).collect();
    let x: Tensor = Array::from_shape_vec((2, 1, 1, 3), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    for b in 0..2usize {
        for c in 0..3usize {
            let expected = (b * 3 + c + 1) as f32;
            assert_abs_diff_eq!(out[[b, c]], expected, epsilon = 1e-5);
        }
    }
}

/// predict() equals forward() for GlobalAveragePooling2D
#[test]
fn global_avg_pool_2d_predict_equals_forward() {
    let mut layer_fwd = GlobalAveragePooling2D::new();
    let layer_pred = GlobalAveragePooling2D::new();
    let x: Tensor = Array::ones((2, 4, 4, 3)).into_dyn();
    let fwd = layer_fwd.forward(&x).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun for GlobalAveragePooling2D
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

/// Wrong-rank input (3D instead of 4D) errors
#[test]
fn global_avg_pool_2d_wrong_rank_input_errors() {
    let mut layer = GlobalAveragePooling2D::new();
    let bad: Tensor = Array::ones((2, 3, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = GlobalAveragePooling2D::new();
    assert!(layer_pred.predict(&bad).is_err());
}

/// layer_type() returns correct string
#[test]
fn global_avg_pool_2d_layer_type() {
    let layer = GlobalAveragePooling2D::new();
    assert_eq!(layer.layer_type(), "GlobalAveragePooling2D");
}

/// output_shape() returns "Unknown" before forward and "(batch, channels)" after
#[test]
fn global_avg_pool_2d_output_shape() {
    let layer_new = GlobalAveragePooling2D::new();
    assert_eq!(layer_new.output_shape(), "Unknown");

    let mut layer = GlobalAveragePooling2D::new();
    // batch 3, 5x5 spatial, channels 4
    let x: Tensor = Array::ones((3, 5, 5, 4)).into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(3, 4)");
}

// GlobalAveragePooling3D, input [batch, depth, height, width, channels] -> output [batch, channels]
// Each (b, c) output = mean over d x h x w

/// Forward mean over the whole volume for a single batch and channel
#[test]
fn global_avg_pool_3d_forward_single_batch_channel() {
    let mut layer = GlobalAveragePooling3D::new();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 2, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 3.5_f32, epsilon = 1e-5); // 28/8
}

/// All-ones 5D input yields all-ones output
#[test]
fn global_avg_pool_3d_forward_all_ones() {
    let mut layer = GlobalAveragePooling3D::new();
    let x: Tensor = Array::ones((2, 3, 3, 3, 4)).into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    for b in 0..2 {
        for c in 0..4 {
            assert_abs_diff_eq!(out[[b, c]], 1.0_f32, epsilon = 1e-5);
        }
    }
}

/// The layer pools channels independently in 3D
#[test]
fn global_avg_pool_3d_forward_multi_channel() {
    let mut layer = GlobalAveragePooling3D::new();
    // Channel 0 is uniformly 5.0, channel 1 uniformly 9.0, interleaved on the last axis
    let mut vals = Vec::with_capacity(2 * 2 * 2 * 2);
    for _ in 0..(2 * 2 * 2) {
        vals.push(5.0f32);
        vals.push(9.0f32);
    }
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 2, 2), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 5.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], 9.0_f32, epsilon = 1e-5);
}

/// The layer computes the volume mean independently per batch
#[test]
fn global_avg_pool_3d_forward_multi_batch() {
    let mut layer = GlobalAveragePooling3D::new();
    let mut vals = vec![3.0f32; 8]; // batch 0, 2x2x2, ch 0
    vals.extend(vec![7.0f32; 8]); // batch 1, 2x2x2, ch 0
    let x: Tensor = Array::from_shape_vec((2, 2, 2, 2, 1), vals)
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[1, 0]], 7.0_f32, epsilon = 1e-5);
}

/// predict() equals forward() for GlobalAveragePooling3D
#[test]
fn global_avg_pool_3d_predict_equals_forward() {
    let mut layer_fwd = GlobalAveragePooling3D::new();
    let layer_pred = GlobalAveragePooling3D::new();
    let vals: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x: Tensor = Array::from_shape_vec((1, 2, 2, 2, 1), vals)
        .unwrap()
        .into_dyn();
    let fwd = layer_fwd.forward(&x).unwrap();
    let pred = layer_pred.predict(&x).unwrap();
    assert_eq!(fwd.shape(), pred.shape());
    for (a, b) in fwd.iter().zip(pred.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

/// backward() before forward() returns ForwardPassNotRun for GlobalAveragePooling3D
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

/// Wrong-rank input (4D instead of 5D) errors
#[test]
fn global_avg_pool_3d_wrong_rank_input_errors() {
    let mut layer = GlobalAveragePooling3D::new();
    let bad: Tensor = Array::ones((1, 2, 3, 4)).into_dyn();
    assert!(layer.forward(&bad).is_err());
    let layer_pred = GlobalAveragePooling3D::new();
    assert!(layer_pred.predict(&bad).is_err());
}

/// layer_type() returns correct string
#[test]
fn global_avg_pool_3d_layer_type() {
    let layer = GlobalAveragePooling3D::new();
    assert_eq!(layer.layer_type(), "GlobalAveragePooling3D");
}

/// output_shape() returns "Unknown" before forward and "(batch, channels)" after
#[test]
fn global_avg_pool_3d_output_shape() {
    let layer_new = GlobalAveragePooling3D::new();
    assert_eq!(layer_new.output_shape(), "Unknown");

    let mut layer = GlobalAveragePooling3D::new();
    // batch 2, 3x3x3 volume, channels 5
    let x: Tensor = Array::ones((2, 3, 3, 3, 5)).into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(2, 5)");
}

/// Same padding excludes padded cells from the average divisor (Keras count_include_pad=False).
/// For a 3x3 input, pool 2x2, stride 2, the trailing windows divide by their in-bounds count.
#[test]
fn avg_pool_2d_same_padding_excludes_padding() {
    let mut layer = AveragePooling2D::new((2, 2), vec![1, 3, 3, 1])
        .unwrap()
        .with_strides((2, 2))
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec((1, 3, 3, 1), (1..=9).map(|v| v as f32).collect())
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 3.0, epsilon = 1e-6); // (1+2+4+5)/4
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 4.5, epsilon = 1e-6); // (3+6)/2, padding excluded
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 7.5, epsilon = 1e-6); // (7+8)/2, padding excluded
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 9.0, epsilon = 1e-6); // 9/1
}
