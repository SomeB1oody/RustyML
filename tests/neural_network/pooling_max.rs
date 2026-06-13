//! Integration tests for MaxPooling 1D/2D/3D and GlobalMaxPooling 1D/2D/3D:
//! forward values, predict-equals-forward, output_shape strings, and error paths;
//! gradient correctness lives in gradient_check.rs

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::pooling::global_max_pooling_1d::GlobalMaxPooling1D;
use rustyml::neural_network::layers::pooling::global_max_pooling_2d::GlobalMaxPooling2D;
use rustyml::neural_network::layers::pooling::global_max_pooling_3d::GlobalMaxPooling3D;
use rustyml::neural_network::layers::pooling::max_pooling_1d::MaxPooling1D;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::pooling::max_pooling_3d::MaxPooling3D;
use rustyml::neural_network::traits::Layer;

use super::common::assert_allclose;

// MaxPooling1D - forward values

/// MaxPooling1D forward output for pool=2, stride=2 on a single channel
#[test]
fn max_pooling_1d_forward_values_pool2_stride2() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None, PaddingType::Valid).unwrap();

    let x = Array::from_shape_vec((1, 1, 4), vec![3.0f32, 1.0, 4.0, 1.0])
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2]);

    let expected = Array::from_shape_vec((1, 1, 2), vec![3.0f32, 4.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling1D forward output for pool=3, stride=1 (overlapping windows)
#[test]
fn max_pooling_1d_forward_values_pool3_stride1() {
    let mut layer = MaxPooling1D::new(3, vec![1, 1, 6], Some(1), PaddingType::Valid).unwrap();

    let x = Array::from_shape_vec((1, 1, 6), vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 4]);

    let expected = Array::from_shape_vec((1, 1, 4), vec![2.0f32, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling1D pools each of two channels independently
#[test]
fn max_pooling_1d_forward_values_two_channels() {
    let mut layer = MaxPooling1D::new(2, vec![1, 2, 6], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = vec![
        // ch0
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // ch1
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0,
    ];
    let x = Array::from_shape_vec((1, 2, 6), data).unwrap().into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3]);

    // ch0: [1, 3, 5]; ch1: [10, 8, 6]
    let expected_data: Vec<f32> = vec![1.0, 3.0, 5.0, 10.0, 8.0, 6.0];
    let expected = Array::from_shape_vec((1, 2, 3), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling1D pools each batch element independently
#[test]
fn max_pooling_1d_forward_values_batch() {
    let mut layer = MaxPooling1D::new(2, vec![2, 1, 4], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = vec![
        // batch 0, ch 0
        5.0, 3.0, 7.0, 2.0, // batch 1, ch 0
        1.0, 9.0, 4.0, 6.0,
    ];
    let x = Array::from_shape_vec((2, 1, 4), data).unwrap().into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1, 2]);

    let expected_data: Vec<f32> = vec![5.0, 7.0, 9.0, 6.0];
    let expected = Array::from_shape_vec((2, 1, 2), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling1D - predict equals forward in eval mode

#[test]
fn max_pooling_1d_predict_equals_forward() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 6], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0];
    let x = Array::from_shape_vec((1, 1, 6), data).unwrap().into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// MaxPooling1D - output_shape string

/// output_shape reports the pooled spatial length as "(1, 2, 4)"
#[test]
fn max_pooling_1d_output_shape_string() {
    let layer = MaxPooling1D::new(2, vec![1, 2, 8], Some(2), PaddingType::Valid).unwrap();
    let s = layer.output_shape();
    assert_eq!(s, "(1, 2, 4)");
}

/// Global pooling layers return "Unknown" for output_shape before any forward pass
#[test]
fn max_pooling_1d_output_shape_unknown_before_forward() {
    let layer = GlobalMaxPooling1D::new();
    assert_eq!(layer.output_shape(), "Unknown");
}

// MaxPooling1D - constructor error paths

#[test]
fn max_pooling_1d_err_pool_size_zero() {
    let result = MaxPooling1D::new(0, vec![1, 1, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool_size=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_pool_size_larger_than_input() {
    let result = MaxPooling1D::new(5, vec![1, 1, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool_size > input length, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_stride_zero() {
    let result = MaxPooling1D::new(2, vec![1, 1, 4], Some(0), PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_wrong_ndim() {
    // input_shape must be 3D; 2D yields DimensionMismatch
    let result = MaxPooling1D::new(2, vec![1, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for 2D input_shape, got {:?}",
        result
    );
}

// MaxPooling1D - backward-before-forward error

#[test]
fn max_pooling_1d_backward_before_forward_err() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None, PaddingType::Valid).unwrap();
    let grad = Array::ones((1, 1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun before forward, got {:?}",
        result
    );
}

// MaxPooling1D - wrong ndim at forward time

#[test]
fn max_pooling_1d_forward_wrong_ndim_err() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None, PaddingType::Valid).unwrap();
    // Pass a 2D tensor instead of 3D
    let x = Array::ones((1, 4)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D input at forward, got {:?}",
        result
    );
}

// MaxPooling2D - forward values

/// MaxPooling2D forward output for a 4x4 input with pool=(2,2), stride=(2,2)
#[test]
fn max_pooling_2d_forward_values_4x4_pool2x2() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    let expected = Array::from_shape_vec((1, 1, 2, 2), vec![5.0f32, 7.0, 13.0, 15.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling2D forward output for a 4x4 input with pool=(2,2), stride=(1,1)
#[test]
fn max_pooling_2d_forward_values_stride1() {
    let mut layer =
        MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], Some((1, 1)), PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);

    let expected_data: Vec<f32> = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    let expected = Array::from_shape_vec((1, 1, 3, 3), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling2D pools each of two channels independently when pool equals the spatial dims
#[test]
fn max_pooling_2d_forward_values_two_channels() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 2, 2, 2], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 1, 1]);

    let expected = Array::from_shape_vec((1, 2, 1, 1), vec![3.0f32, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling2D - predict equals forward

#[test]
fn max_pooling_2d_predict_equals_forward() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// MaxPooling2D - output_shape string

/// output_shape reports the pooled 2D shape as "(1, 3, 3, 3)"
#[test]
fn max_pooling_2d_output_shape_string() {
    let layer = MaxPooling2D::new((2, 2), vec![1, 3, 6, 6], None, PaddingType::Valid).unwrap();
    assert_eq!(layer.output_shape(), "(1, 3, 3, 3)");
}

// MaxPooling2D - constructor error paths

#[test]
fn max_pooling_2d_err_pool_size_zero_height() {
    let result = MaxPooling2D::new((0, 2), vec![1, 1, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_pool_size_zero_width() {
    let result = MaxPooling2D::new((2, 0), vec![1, 1, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_pool_larger_than_height() {
    let result = MaxPooling2D::new((5, 2), vec![1, 1, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool > height, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_pool_larger_than_width() {
    let result = MaxPooling2D::new((2, 5), vec![1, 1, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool > width, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_stride_zero() {
    let result = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], Some((0, 1)), PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_zero_batch_or_channel() {
    // Regression: a zero batch/channel dimension once passed new() and only failed at forward;
    // validate_all_dims_positive now rejects it as InvalidInput
    let zero_batch = MaxPooling2D::new((2, 2), vec![0, 1, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(zero_batch, Err(Error::InvalidInput(_))),
        "expected InvalidInput for batch=0, got {:?}",
        zero_batch
    );
    let zero_channel = MaxPooling2D::new((2, 2), vec![1, 0, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(zero_channel, Err(Error::InvalidInput(_))),
        "expected InvalidInput for channel=0, got {:?}",
        zero_channel
    );
}

#[test]
fn max_pooling_2d_err_wrong_ndim() {
    // input_shape must be 4D; 3D yields DimensionMismatch
    let result = MaxPooling2D::new((2, 2), vec![1, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for 3D input_shape, got {:?}",
        result
    );
}

// MaxPooling2D - backward-before-forward error

#[test]
fn max_pooling_2d_backward_before_forward_err() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None, PaddingType::Valid).unwrap();
    let grad = Array::ones((1, 1, 2, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun before forward, got {:?}",
        result
    );
}

// MaxPooling2D - wrong ndim at forward time

#[test]
fn max_pooling_2d_forward_wrong_ndim_err() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None, PaddingType::Valid).unwrap();
    // Pass a 3D tensor instead of 4D
    let x = Array::ones((1, 4, 4)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D input at forward, got {:?}",
        result
    );
}

// MaxPooling3D - forward values

/// MaxPooling3D over a single window covering the whole 2x2x2 volume
#[test]
fn max_pooling_3d_forward_values_2x2x2_full_window() {
    let mut layer =
        MaxPooling3D::new((2, 2, 2), vec![1, 1, 2, 2, 2], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);

    let expected = Array::from_shape_vec((1, 1, 1, 1, 1), vec![7.0f32])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling3D forward output for a 4x4x4 volume with pool=(2,2,2), stride=(2,2,2)
#[test]
fn max_pooling_3d_forward_values_4x4x4_pool2x2x2() {
    let mut layer =
        MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..64).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4, 4), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);

    let expected_data: Vec<f32> = vec![21.0, 23.0, 29.0, 31.0, 53.0, 55.0, 61.0, 63.0];
    let expected = Array::from_shape_vec((1, 1, 2, 2, 2), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling3D - predict equals forward

#[test]
fn max_pooling_3d_predict_equals_forward() {
    let mut layer =
        MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..64).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4, 4), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// MaxPooling3D - output_shape string

/// output_shape reports the pooled 3D shape as "(1, 2, 2, 2, 2)"
#[test]
fn max_pooling_3d_output_shape_string() {
    let layer =
        MaxPooling3D::new((2, 2, 2), vec![1, 2, 4, 4, 4], None, PaddingType::Valid).unwrap();
    assert_eq!(layer.output_shape(), "(1, 2, 2, 2, 2)");
}

// MaxPooling3D - constructor error paths

#[test]
fn max_pooling_3d_err_pool_size_zero() {
    let result = MaxPooling3D::new((0, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_pool_larger_than_depth() {
    let result = MaxPooling3D::new((5, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool > depth, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_stride_zero() {
    let result = MaxPooling3D::new(
        (2, 2, 2),
        vec![1, 1, 4, 4, 4],
        Some((0, 2, 2)),
        PaddingType::Valid,
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_zero_batch_or_channel() {
    let zero_batch = MaxPooling3D::new((2, 2, 2), vec![0, 1, 4, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(zero_batch, Err(Error::InvalidInput(_))),
        "expected InvalidInput for batch=0, got {:?}",
        zero_batch
    );
    let zero_channel = MaxPooling3D::new((2, 2, 2), vec![1, 0, 4, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(zero_channel, Err(Error::InvalidInput(_))),
        "expected InvalidInput for channel=0, got {:?}",
        zero_channel
    );
}

#[test]
fn max_pooling_3d_err_wrong_ndim() {
    // input_shape must be 5D; 4D yields DimensionMismatch
    let result = MaxPooling3D::new((2, 2, 2), vec![1, 4, 4, 4], None, PaddingType::Valid);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for 4D input_shape, got {:?}",
        result
    );
}

// MaxPooling3D - backward-before-forward error

#[test]
fn max_pooling_3d_backward_before_forward_err() {
    let mut layer =
        MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid).unwrap();
    let grad = Array::ones((1, 1, 2, 2, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun before forward, got {:?}",
        result
    );
}

// MaxPooling3D - wrong ndim at forward time

#[test]
fn max_pooling_3d_forward_wrong_ndim_err() {
    let mut layer =
        MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid).unwrap();
    // Pass a 4D tensor instead of 5D
    let x = Array::ones((1, 1, 4, 4)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 4D input at forward, got {:?}",
        result
    );
}

// GlobalMaxPooling1D - forward values

/// GlobalMaxPooling1D reduces each channel to its maximum
#[test]
fn global_max_pooling_1d_forward_values() {
    let mut layer = GlobalMaxPooling1D::new();

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0, 5.0];
    let x = Array::from_shape_vec((1, 2, 5), data).unwrap().into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    let expected = Array::from_shape_vec((1, 2), vec![4.0f32, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GlobalMaxPooling1D reduces each channel per batch element to its maximum
#[test]
fn global_max_pooling_1d_forward_values_batch() {
    let mut layer = GlobalMaxPooling1D::new();

    let data: Vec<f32> = vec![
        // batch 0
        1.0, 5.0, 3.0, 2.0, // ch0
        0.0, 0.0, 0.0, 8.0, // ch1
        7.0, 6.0, 5.0, 4.0, // ch2
        // batch 1
        9.0, 1.0, 1.0, 1.0, // ch0
        2.0, 2.0, 6.0, 2.0, // ch1
        3.0, 3.0, 3.0, 10.0, // ch2
    ];
    let x = Array::from_shape_vec((2, 3, 4), data).unwrap().into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);

    let expected_data: Vec<f32> = vec![5.0, 8.0, 7.0, 9.0, 6.0, 10.0];
    let expected = Array::from_shape_vec((2, 3), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// GlobalMaxPooling1D - predict equals forward

#[test]
fn global_max_pooling_1d_predict_equals_forward() {
    let mut layer = GlobalMaxPooling1D::new();

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0, 5.0];
    let x = Array::from_shape_vec((1, 2, 5), data).unwrap().into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// GlobalMaxPooling1D - output_shape string

/// output_shape returns "(batch_size, channels)" once a forward pass caches the input shape
#[test]
fn global_max_pooling_1d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling1D::new();
    let x = Array::from_shape_vec((1, 2, 5), vec![0.0f32; 10])
        .unwrap()
        .into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(1, 2)");
}

// GlobalMaxPooling1D - backward-before-forward error

#[test]
fn global_max_pooling_1d_backward_before_forward_err() {
    let mut layer = GlobalMaxPooling1D::new();
    let grad = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun before forward, got {:?}",
        result
    );
}

// GlobalMaxPooling1D - wrong ndim at forward time

#[test]
fn global_max_pooling_1d_forward_wrong_ndim_err() {
    let mut layer = GlobalMaxPooling1D::new();
    // 2D instead of 3D
    let x = Array::ones((1, 5)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D input, got {:?}",
        result
    );
}

// GlobalMaxPooling2D - forward values

/// GlobalMaxPooling2D reduces each channel plane to its maximum
#[test]
fn global_max_pooling_2d_forward_values() {
    let mut layer = GlobalMaxPooling2D::new();

    // ch0: values 0..9 (first 9), ch1: values 10..19 (next 9)
    let data: Vec<f32> = (0..9)
        .map(|v| v as f32)
        .chain((10..19).map(|v| v as f32))
        .collect();
    let x = Array::from_shape_vec((1, 2, 3, 3), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    let expected = Array::from_shape_vec((1, 2), vec![8.0f32, 18.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GlobalMaxPooling2D reduces each channel plane per batch element to its maximum
#[test]
fn global_max_pooling_2d_forward_values_batch() {
    let mut layer = GlobalMaxPooling2D::new();

    let data: Vec<f32> = vec![
        3.0, 1.0, 2.0, 4.0, // batch0 ch0
        5.0, 9.0, 8.0, 7.0, // batch1 ch0
    ];
    let x = Array::from_shape_vec((2, 1, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);

    let expected = Array::from_shape_vec((2, 1), vec![4.0f32, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// GlobalMaxPooling2D - predict equals forward

#[test]
fn global_max_pooling_2d_predict_equals_forward() {
    let mut layer = GlobalMaxPooling2D::new();

    let data: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 2, 3, 3), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// GlobalMaxPooling2D - output_shape string

#[test]
fn global_max_pooling_2d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling2D::new();
    let x = Array::from_shape_vec((2, 3, 4, 4), vec![0.0f32; 2 * 3 * 4 * 4])
        .unwrap()
        .into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(2, 3)");
}

// GlobalMaxPooling2D - backward-before-forward error

#[test]
fn global_max_pooling_2d_backward_before_forward_err() {
    let mut layer = GlobalMaxPooling2D::new();
    let grad = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun before forward, got {:?}",
        result
    );
}

// GlobalMaxPooling2D - wrong ndim at forward time

#[test]
fn global_max_pooling_2d_forward_wrong_ndim_err() {
    let mut layer = GlobalMaxPooling2D::new();
    // 3D instead of 4D
    let x = Array::ones((1, 3, 3)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D input, got {:?}",
        result
    );
}

// GlobalMaxPooling3D - forward values

/// GlobalMaxPooling3D reduces each channel volume to its maximum
#[test]
fn global_max_pooling_3d_forward_values() {
    let mut layer = GlobalMaxPooling3D::new();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 2, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    let expected = Array::from_shape_vec((1, 2), vec![7.0f32, 15.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GlobalMaxPooling3D reduces each channel volume per batch element to its maximum
#[test]
fn global_max_pooling_3d_forward_values_batch() {
    let mut layer = GlobalMaxPooling3D::new();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((2, 1, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 1]);

    let expected = Array::from_shape_vec((2, 1), vec![7.0f32, 15.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// GlobalMaxPooling3D - predict equals forward

#[test]
fn global_max_pooling_3d_predict_equals_forward() {
    let mut layer = GlobalMaxPooling3D::new();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 2, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// GlobalMaxPooling3D - output_shape string

#[test]
fn global_max_pooling_3d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling3D::new();
    let x = Array::from_shape_vec((3, 4, 2, 2, 2), vec![0.0f32; 3 * 4 * 2 * 2 * 2])
        .unwrap()
        .into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(3, 4)");
}

// GlobalMaxPooling3D - backward-before-forward error

#[test]
fn global_max_pooling_3d_backward_before_forward_err() {
    let mut layer = GlobalMaxPooling3D::new();
    let grad = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun before forward, got {:?}",
        result
    );
}

// GlobalMaxPooling3D - wrong ndim at forward time

#[test]
fn global_max_pooling_3d_forward_wrong_ndim_err() {
    let mut layer = GlobalMaxPooling3D::new();
    // 4D instead of 5D
    let x = Array::ones((1, 2, 2, 2)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 4D input, got {:?}",
        result
    );
}

// MaxPooling1D - negative values and mixed signs

/// MaxPooling1D picks the algebraically largest value, not the largest absolute value
#[test]
fn max_pooling_1d_forward_negative_values() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None, PaddingType::Valid).unwrap();

    let x = Array::from_shape_vec((1, 1, 4), vec![-3.0f32, -1.0, -4.0, -2.0])
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2]);

    let expected = Array::from_shape_vec((1, 1, 2), vec![-1.0f32, -2.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling2D - non-square spatial dims

/// MaxPooling2D forward output when height and width pool to different sizes
#[test]
fn max_pooling_2d_forward_non_square_spatial() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 2, 4], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 2, 4), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 2]);

    let expected = Array::from_shape_vec((1, 1, 1, 2), vec![5.0f32, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling2D - parallel plane assembly (POOL_PARALLEL_THRESHOLD = 32)
// Distinct per-plane maxima at exactly 32 planes catch any mis-ordered parallel assembly

/// Parallel branch keeps planes in channel order: output[c] = per-channel max 4c+3
#[test]
fn max_pooling_2d_parallel_planes_forward_distinct_maxima() {
    let channels = 32; // bc_total = 1 * 32 = 32, exactly the parallel threshold
    let mut layer =
        MaxPooling2D::new((2, 2), vec![1, channels, 2, 2], None, PaddingType::Valid).unwrap();

    // Row-major fill 0..(1*32*2*2) = 0..128; channel c owns [4c, 4c+1, 4c+2, 4c+3]
    let data: Vec<f32> = (0..(channels * 4)).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, channels, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, channels, 1, 1]);

    // Expected: per-channel max = 4c+3
    let expected_data: Vec<f32> = (0..channels).map(|c| (4 * c + 3) as f32).collect();
    let expected = Array::from_shape_vec((1, channels, 1, 1), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// Parallel-branch backward routes each upstream gradient to its plane's arg-max in channel order
#[test]
fn max_pooling_2d_parallel_planes_backward_routes_per_plane() {
    let channels = 32; // bc_total = 32, parallel branch
    let mut layer =
        MaxPooling2D::new((2, 2), vec![1, channels, 2, 2], None, PaddingType::Valid).unwrap();

    let data: Vec<f32> = (0..(channels * 4)).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, channels, 2, 2), data)
        .unwrap()
        .into_dyn();
    // Forward to populate the arg-max cache (one per channel, all at local index 3)
    layer.forward(&x).unwrap();

    // Upstream gradient: distinct value per channel, grad[c] = c+1
    let grad_data: Vec<f32> = (0..channels).map(|c| (c + 1) as f32).collect();
    let grad_out = Array::from_shape_vec((1, channels, 1, 1), grad_data)
        .unwrap()
        .into_dyn();

    let grad_in = layer.backward(&grad_out).unwrap();
    assert_eq!(grad_in.shape(), &[1, channels, 2, 2]);

    // Expected: per channel c, the four positions are [0, 0, 0, c+1]
    let mut expected_data: Vec<f32> = Vec::with_capacity(channels * 4);
    for c in 0..channels {
        expected_data.extend_from_slice(&[0.0, 0.0, 0.0, (c + 1) as f32]);
    }
    let expected = Array::from_shape_vec((1, channels, 2, 2), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&grad_in, &expected, 1e-6);
}

/// Same padding rounds the output up to ceil(in/stride) and the trailing windows see only the
/// in-bounds cells (padding is virtual). For a 3x3 input, pool 2x2, stride 2: output is 2x2
#[test]
fn max_pool_2d_same_padding_3x3() {
    let mut layer =
        MaxPooling2D::new((2, 2), vec![1, 1, 3, 3], Some((2, 2)), PaddingType::Same).unwrap();
    let x = Array::from_shape_vec((1, 1, 3, 3), (1..=9).map(|v| v as f32).collect())
        .unwrap()
        .into_dyn();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 5.0, epsilon = 1e-6); // max(1,2,4,5)
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], 6.0, epsilon = 1e-6); // max(3,6) (col 3 is padding)
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 8.0, epsilon = 1e-6); // max(7,8) (row 3 is padding)
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 9.0, epsilon = 1e-6); // just 9
}
