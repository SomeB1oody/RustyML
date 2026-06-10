//! Integration tests for MaxPooling 1D / 2D / 3D and
//! GlobalMaxPooling 1D / 2D / 3D.
//!
//! Every expected value is derived from the mathematical definition of max-pooling
//! (the maximum element in each sliding window, or the global maximum per channel).
//! No test here duplicates gradient correctness — that lives in gradient_check.rs.

use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::pooling::global_max_pooling_1d::GlobalMaxPooling1D;
use rustyml::neural_network::layers::pooling::global_max_pooling_2d::GlobalMaxPooling2D;
use rustyml::neural_network::layers::pooling::global_max_pooling_3d::GlobalMaxPooling3D;
use rustyml::neural_network::layers::pooling::max_pooling_1d::MaxPooling1D;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::pooling::max_pooling_3d::MaxPooling3D;
use rustyml::neural_network::traits::Layer;

use super::common::assert_allclose;

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — forward values
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input shape [1, 1, 4], values [3, 1, 4, 1], pool=2, stride=2 (default).
///
///   window 0: max(3, 1) = 3
///   window 1: max(4, 1) = 4
///
/// Output shape: (4-2)/2+1 = 2  →  [1, 1, 2]
#[test]
fn max_pooling_1d_forward_values_pool2_stride2() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None).unwrap();

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

/// Hand-calculation:
/// Input [1, 1, 6], values 0..5, pool=3, stride=1.
///
///   window 0: max(0, 1, 2) = 2
///   window 1: max(1, 2, 3) = 3
///   window 2: max(2, 3, 4) = 4
///   window 3: max(3, 4, 5) = 5
///
/// Output shape: (6-3)/1+1 = 4  →  [1, 1, 4]
#[test]
fn max_pooling_1d_forward_values_pool3_stride1() {
    let mut layer = MaxPooling1D::new(3, vec![1, 1, 6], Some(1)).unwrap();

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

/// Hand-calculation:
/// Input [1, 2, 6], values per channel:
///   ch0 = [0, 1, 2, 3, 4, 5]
///   ch1 = [10, 9, 8, 7, 6, 5]
/// pool=2, stride=2 (default).
///
///   ch0: window [0,1]→max=1; [2,3]→max=3; [4,5]→max=5
///   ch1: window [0,1]→max=10; [2,3]→max=8; [4,5]→max=6
///
/// Output shape: (6-2)/2+1 = 3  →  [1, 2, 3]
#[test]
fn max_pooling_1d_forward_values_two_channels() {
    let mut layer = MaxPooling1D::new(2, vec![1, 2, 6], None).unwrap();

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

/// Hand-calculation with a batch dimension:
/// Input [2, 1, 4], batch0 = [5, 3, 7, 2], batch1 = [1, 9, 4, 6], pool=2, stride=2.
///
///   batch0: window [0,1]→5; window [2,3]→7
///   batch1: window [0,1]→9; window [2,3]→6
///
/// Output shape: [2, 1, 2]
#[test]
fn max_pooling_1d_forward_values_batch() {
    let mut layer = MaxPooling1D::new(2, vec![2, 1, 4], None).unwrap();

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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — predict equals forward in eval mode
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_1d_predict_equals_forward() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 6], None).unwrap();

    let data: Vec<f32> = vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0];
    let x = Array::from_shape_vec((1, 1, 6), data).unwrap().into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — output_shape string
// ══════════════════════════════════════════════════════════════════════════════

/// output_shape formula: (length - pool_size) / stride + 1
/// Input [1, 2, 8], pool=2, stride=2 → (8-2)/2+1 = 4 → output [1, 2, 4]
/// Expected string: "(1, 2, 4)"
#[test]
fn max_pooling_1d_output_shape_string() {
    let layer = MaxPooling1D::new(2, vec![1, 2, 8], Some(2)).unwrap();
    // The macro reads input_shape, so call output_shape and check the string.
    let s = layer.output_shape();
    assert_eq!(s, "(1, 2, 4)");
}

/// Before any forward pass the global pooling layers return "Unknown" for
/// output_shape (they need a forward pass to cache input_shape).
#[test]
fn max_pooling_1d_output_shape_unknown_before_forward() {
    // GlobalMaxPooling1D input_shape is initially empty.
    let layer = GlobalMaxPooling1D::new();
    assert_eq!(layer.output_shape(), "Unknown");
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — constructor error paths
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_1d_err_pool_size_zero() {
    let result = MaxPooling1D::new(0, vec![1, 1, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool_size=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_pool_size_larger_than_input() {
    let result = MaxPooling1D::new(5, vec![1, 1, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool_size > input length, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_stride_zero() {
    let result = MaxPooling1D::new(2, vec![1, 1, 4], Some(0));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_wrong_ndim() {
    // input_shape must be 3D: passing 2D should be DimensionMismatch
    let result = MaxPooling1D::new(2, vec![1, 4], None);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for 2D input_shape, got {:?}",
        result
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — backward-before-forward error
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_1d_backward_before_forward_err() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None).unwrap();
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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — wrong ndim at forward time
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_1d_forward_wrong_ndim_err() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None).unwrap();
    // Pass a 2D tensor instead of 3D
    let x = Array::ones((1, 4)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D input at forward, got {:?}",
        result
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — forward values
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 1, 4, 4], values 0..15 row-major, pool=(2,2), stride=(2,2).
///
///   Layout (row, col):
///     (0,0)=0  (0,1)=1  (0,2)=2  (0,3)=3
///     (1,0)=4  (1,1)=5  (1,2)=6  (1,3)=7
///     (2,0)=8  (2,1)=9  (2,2)=10 (2,3)=11
///     (3,0)=12 (3,1)=13 (3,2)=14 (3,3)=15
///
///   Window [0:2, 0:2]: max(0,1,4,5)   = 5
///   Window [0:2, 2:4]: max(2,3,6,7)   = 7
///   Window [2:4, 0:2]: max(8,9,12,13) = 13
///   Window [2:4, 2:4]: max(10,11,14,15) = 15
///
/// Output shape: (4-2)/2+1 = 2 each → [1, 1, 2, 2]
#[test]
fn max_pooling_2d_forward_values_4x4_pool2x2() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None).unwrap();

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

/// Hand-calculation:
/// Input [1, 1, 4, 4], values 0..15, pool=(2,2), stride=(1,1).
///
///   Output shape: (4-2)/1+1 = 3 each → [1, 1, 3, 3]
///
///   (0,0): max(0,1,4,5)   = 5
///   (0,1): max(1,2,5,6)   = 6
///   (0,2): max(2,3,6,7)   = 7
///   (1,0): max(4,5,8,9)   = 9
///   (1,1): max(5,6,9,10)  = 10
///   (1,2): max(6,7,10,11) = 11
///   (2,0): max(8,9,12,13) = 13
///   (2,1): max(9,10,13,14)= 14
///   (2,2): max(10,11,14,15)= 15
#[test]
fn max_pooling_2d_forward_values_stride1() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], Some((1, 1))).unwrap();

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

/// Hand-calculation (two channels):
/// Input [1, 2, 2, 2], pool=(2,2), stride=(2,2) (default; pool equals spatial dims).
///
///   ch0: values [0,1,2,3] → max = 3
///   ch1: values [4,5,6,7] → max = 7
///
/// Output shape: (2-2)/2+1 = 1 → [1, 2, 1, 1]
#[test]
fn max_pooling_2d_forward_values_two_channels() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 2, 2, 2], None).unwrap();

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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — predict equals forward
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_2d_predict_equals_forward() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None).unwrap();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — output_shape string
// ══════════════════════════════════════════════════════════════════════════════

/// Input [1, 3, 6, 6], pool=(2,2), stride=(2,2)
/// output: (6-2)/2+1 = 3 → "(1, 3, 3, 3)"
#[test]
fn max_pooling_2d_output_shape_string() {
    let layer = MaxPooling2D::new((2, 2), vec![1, 3, 6, 6], None).unwrap();
    assert_eq!(layer.output_shape(), "(1, 3, 3, 3)");
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — constructor error paths
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_2d_err_pool_size_zero_height() {
    let result = MaxPooling2D::new((0, 2), vec![1, 1, 4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_pool_size_zero_width() {
    let result = MaxPooling2D::new((2, 0), vec![1, 1, 4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_pool_larger_than_height() {
    let result = MaxPooling2D::new((5, 2), vec![1, 1, 4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool > height, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_pool_larger_than_width() {
    let result = MaxPooling2D::new((2, 5), vec![1, 1, 4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool > width, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_stride_zero() {
    let result = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], Some((0, 1)));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_zero_batch_or_channel() {
    // Regression: a zero batch/channel dimension used to pass `new()` and only fail at forward.
    // `validate_all_dims_positive` now rejects it as InvalidInput.
    let zero_batch = MaxPooling2D::new((2, 2), vec![0, 1, 4, 4], None);
    assert!(
        matches!(zero_batch, Err(Error::InvalidInput(_))),
        "expected InvalidInput for batch=0, got {:?}",
        zero_batch
    );
    let zero_channel = MaxPooling2D::new((2, 2), vec![1, 0, 4, 4], None);
    assert!(
        matches!(zero_channel, Err(Error::InvalidInput(_))),
        "expected InvalidInput for channel=0, got {:?}",
        zero_channel
    );
}

#[test]
fn max_pooling_2d_err_wrong_ndim() {
    // input_shape must be 4D; passing 3D → DimensionMismatch
    let result = MaxPooling2D::new((2, 2), vec![1, 4, 4], None);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for 3D input_shape, got {:?}",
        result
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — backward-before-forward error
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_2d_backward_before_forward_err() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None).unwrap();
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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — wrong ndim at forward time
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_2d_forward_wrong_ndim_err() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None).unwrap();
    // Pass a 3D tensor instead of 4D
    let x = Array::ones((1, 4, 4)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D input at forward, got {:?}",
        result
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling3D — forward values
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 1, 2, 2, 2], values 0..7, pool=(2,2,2), stride=(2,2,2).
///
///   Single window covers all 8 elements; max = 7.
///
/// Output shape: (2-2)/2+1 = 1 each → [1, 1, 1, 1, 1]
#[test]
fn max_pooling_3d_forward_values_2x2x2_full_window() {
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 2, 2, 2], None).unwrap();

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

/// Hand-calculation:
/// Input [1, 1, 4, 4, 4], values 0..63, pool=(2,2,2), stride=(2,2,2).
///
/// Output shape: (4-2)/2+1 = 2 each → [1, 1, 2, 2, 2].
///
/// The 8 output positions correspond to 2×2×2 windows over the 4×4×4 volume.
/// The input is stored in row-major order:
///   index = d*16 + h*4 + w
///
/// Window at (od, oh, ow): covers d in [2*od, 2*od+2), h in [2*oh, 2*oh+2), w in [2*ow, 2*ow+2).
/// Max element in each window = max corner at (2*od+1, 2*oh+1, 2*ow+1):
///   (1,1,1) → index 1*16+1*4+1 = 21
///   (1,1,3) → index 1*16+1*4+3 = 23
///   (1,3,1) → index 1*16+3*4+1 = 29
///   (1,3,3) → index 1*16+3*4+3 = 31
///   (3,1,1) → index 3*16+1*4+1 = 53
///   (3,1,3) → index 3*16+1*4+3 = 55
///   (3,3,1) → index 3*16+3*4+1 = 61
///   (3,3,3) → index 3*16+3*4+3 = 63  (out of range for 0..63, last = 63)
///
/// Actually last index is 63 (value 63) since we have 4^3=64 elements (0..63).
/// Window (od=1,oh=1,ow=1): d in [2,4), h in [2,4), w in [2,4)
///   indices = {32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47} — highest = 47? No.
///   d=3,h=3,w=3: index=3*16+3*4+3=63 — but that is window (od=1,oh=1,ow=1).
///   d=3,h=3,w=2: index=3*16+3*4+2=62.
///   So max of window(1,1,1) = 63.
///
/// Let me be systematic. Window (od,oh,ow) covers:
///   d in {2*od, 2*od+1}, h in {2*oh, 2*oh+1}, w in {2*ow, 2*ow+1}.
///   max index = (2*od+1)*16 + (2*oh+1)*4 + (2*ow+1)
///
///   (0,0,0): max_idx = 1*16+1*4+1 = 21
///   (0,0,1): max_idx = 1*16+1*4+3 = 23
///   (0,1,0): max_idx = 1*16+3*4+1 = 29
///   (0,1,1): max_idx = 1*16+3*4+3 = 31
///   (1,0,0): max_idx = 3*16+1*4+1 = 53
///   (1,0,1): max_idx = 3*16+1*4+3 = 55
///   (1,1,0): max_idx = 3*16+3*4+1 = 61
///   (1,1,1): max_idx = 3*16+3*4+3 = 63
#[test]
fn max_pooling_3d_forward_values_4x4x4_pool2x2x2() {
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();

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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling3D — predict equals forward
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_3d_predict_equals_forward() {
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();

    let data: Vec<f32> = (0..64).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 1, 4, 4, 4), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling3D — output_shape string
// ══════════════════════════════════════════════════════════════════════════════

/// Input [1, 2, 4, 4, 4], pool=(2,2,2), stride=(2,2,2)
/// output: (4-2)/2+1 = 2 each → "(1, 2, 2, 2, 2)"
#[test]
fn max_pooling_3d_output_shape_string() {
    let layer = MaxPooling3D::new((2, 2, 2), vec![1, 2, 4, 4, 4], None).unwrap();
    assert_eq!(layer.output_shape(), "(1, 2, 2, 2, 2)");
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling3D — constructor error paths
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_3d_err_pool_size_zero() {
    let result = MaxPooling3D::new((0, 2, 2), vec![1, 1, 4, 4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_pool_larger_than_depth() {
    let result = MaxPooling3D::new((5, 2, 2), vec![1, 1, 4, 4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for pool > depth, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_stride_zero() {
    let result = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], Some((0, 2, 2)));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_zero_batch_or_channel() {
    let zero_batch = MaxPooling3D::new((2, 2, 2), vec![0, 1, 4, 4, 4], None);
    assert!(
        matches!(zero_batch, Err(Error::InvalidInput(_))),
        "expected InvalidInput for batch=0, got {:?}",
        zero_batch
    );
    let zero_channel = MaxPooling3D::new((2, 2, 2), vec![1, 0, 4, 4, 4], None);
    assert!(
        matches!(zero_channel, Err(Error::InvalidInput(_))),
        "expected InvalidInput for channel=0, got {:?}",
        zero_channel
    );
}

#[test]
fn max_pooling_3d_err_wrong_ndim() {
    // input_shape must be 5D; passing 4D → DimensionMismatch
    let result = MaxPooling3D::new((2, 2, 2), vec![1, 4, 4, 4], None);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for 4D input_shape, got {:?}",
        result
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling3D — backward-before-forward error
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_3d_backward_before_forward_err() {
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();
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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling3D — wrong ndim at forward time
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn max_pooling_3d_forward_wrong_ndim_err() {
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();
    // Pass a 4D tensor instead of 5D
    let x = Array::ones((1, 1, 4, 4)).into_dyn();
    let result = layer.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 4D input at forward, got {:?}",
        result
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling1D — forward values
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 2, 5]:
///   ch0 = [0, 1, 2, 3, 4]   → max = 4
///   ch1 = [9, 8, 7, 6, 5]   → max = 9
///
/// Output shape: [1, 2]
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

/// Hand-calculation (batch):
/// Input [2, 3, 4]:
///   batch0 ch0 = [1, 5, 3, 2]  → max = 5
///   batch0 ch1 = [0, 0, 0, 8]  → max = 8
///   batch0 ch2 = [7, 6, 5, 4]  → max = 7
///   batch1 ch0 = [9, 1, 1, 1]  → max = 9
///   batch1 ch1 = [2, 2, 6, 2]  → max = 6
///   batch1 ch2 = [3, 3, 3, 10] → max = 10
///
/// Output shape: [2, 3]
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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling1D — predict equals forward
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn global_max_pooling_1d_predict_equals_forward() {
    let mut layer = GlobalMaxPooling1D::new();

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0, 5.0];
    let x = Array::from_shape_vec((1, 2, 5), data).unwrap().into_dyn();

    let fwd = layer.forward(&x).unwrap();
    let pred = layer.predict(&x).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling1D — output_shape string
// ══════════════════════════════════════════════════════════════════════════════

/// After a forward pass with input [1, 2, 5], input_shape is cached as [1,2,5].
/// output_shape() returns "(batch_size, channels)" = "(1, 2)".
#[test]
fn global_max_pooling_1d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling1D::new();
    let x = Array::from_shape_vec((1, 2, 5), vec![0.0f32; 10])
        .unwrap()
        .into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(1, 2)");
}

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling1D — backward-before-forward error
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling1D — wrong ndim at forward time
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling2D — forward values
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 2, 3, 3]:
///   ch0 = [[0,1,2],[3,4,5],[6,7,8]]        → max = 8
///   ch1 = [[10,11,12],[13,14,15],[16,17,18]] → max = 18
///
/// Output shape: [1, 2]
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

/// Hand-calculation (batch):
/// Input [2, 1, 2, 2]:
///   batch0 ch0 = [[3, 1],[2, 4]]    → max = 4
///   batch1 ch0 = [[5, 9],[8, 7]]    → max = 9
///
/// Output shape: [2, 1]
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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling2D — predict equals forward
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling2D — output_shape string
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn global_max_pooling_2d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling2D::new();
    let x = Array::from_shape_vec((2, 3, 4, 4), vec![0.0f32; 2 * 3 * 4 * 4])
        .unwrap()
        .into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(2, 3)");
}

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling2D — backward-before-forward error
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling2D — wrong ndim at forward time
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling3D — forward values
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 2, 2, 2, 2]:
///   ch0 = values 0..7   → max = 7
///   ch1 = values 8..15  → max = 15
///
/// Output shape: [1, 2]
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

/// Hand-calculation (batch):
/// Input [2, 1, 2, 2, 2]:
///   batch0 ch0 = 0..7  → max = 7
///   batch1 ch0 = 8..15 → max = 15
///
/// Output shape: [2, 1]
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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling3D — predict equals forward
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling3D — output_shape string
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn global_max_pooling_3d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling3D::new();
    let x = Array::from_shape_vec((3, 4, 2, 2, 2), vec![0.0f32; 3 * 4 * 2 * 2 * 2])
        .unwrap()
        .into_dyn();
    layer.forward(&x).unwrap();
    assert_eq!(layer.output_shape(), "(3, 4)");
}

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling3D — backward-before-forward error
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// GlobalMaxPooling3D — wrong ndim at forward time
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling1D — negative values and mixed signs
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 1, 4] = [-3, -1, -4, -2], pool=2, stride=2.
///
///   window 0: max(-3, -1) = -1
///   window 1: max(-4, -2) = -2
///
/// This confirms max-pooling picks the algebraically largest, not the
/// largest absolute value.
#[test]
fn max_pooling_1d_forward_negative_values() {
    let mut layer = MaxPooling1D::new(2, vec![1, 1, 4], None).unwrap();

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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — non-square pool and non-square spatial dims
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 1, 2, 4], pool=(2,2), stride=(2,2).
///
///   Values (row-major):
///     (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3
///     (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
///
///   Window [0:2, 0:2]: max(0,1,4,5) = 5
///   Window [0:2, 2:4]: max(2,3,6,7) = 7
///
/// Output shape: (2-2)/2+1 = 1 height, (4-2)/2+1 = 2 width → [1, 1, 1, 2]
#[test]
fn max_pooling_2d_forward_non_square_spatial() {
    let mut layer = MaxPooling2D::new((2, 2), vec![1, 1, 2, 4], None).unwrap();

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

// ══════════════════════════════════════════════════════════════════════════════
// MaxPooling2D — parallel plane assembly (POOL_PARALLEL_THRESHOLD = 32)
//
// The pooling engine processes each batch*channels "plane" independently and
// runs them in PARALLEL once `batch * channels >= 32`. These two tests build a
// layer with exactly 32 channel planes (bc_total == 32, the threshold), each
// with a DIFFERENT per-plane maximum, so that any mis-ordered parallel
// plane-assembly (planes concatenated out of `bc` order) is caught by the
// position of each value in the flat output.
// ══════════════════════════════════════════════════════════════════════════════

/// Hand-calculation:
/// Input [1, 32, 2, 2], values 0..127 row-major (so channel `c` owns the four
/// contiguous values [4c, 4c+1, 4c+2, 4c+3]). pool=(2,2), stride=(2,2): the single
/// 2x2 window per channel covers the whole plane, so output[c] = max of that plane
/// = 4c+3.
///
///   bc_total = 1 * 32 = 32 == POOL_PARALLEL_THRESHOLD  → parallel branch.
///   Output shape: (2-2)/2+1 = 1 each → [1, 32, 1, 1].
///   Expected flat output (in channel order): [3, 7, 11, ..., 127] = 4c+3.
///
/// Because every channel's maximum is distinct and strictly increasing, a parallel
/// assembly that concatenated planes out of order would produce a permuted output
/// and fail this check.
#[test]
fn max_pooling_2d_parallel_planes_forward_distinct_maxima() {
    let channels = 32; // bc_total = 1 * 32 = 32, exactly the parallel threshold
    let mut layer = MaxPooling2D::new((2, 2), vec![1, channels, 2, 2], None).unwrap();

    // Row-major fill 0..(1*32*2*2) = 0..128; channel c owns [4c, 4c+1, 4c+2, 4c+3].
    let data: Vec<f32> = (0..(channels * 4)).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, channels, 2, 2), data)
        .unwrap()
        .into_dyn();

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, channels, 1, 1]);

    // Expected: per-channel max = 4c+3.
    let expected_data: Vec<f32> = (0..channels).map(|c| (4 * c + 3) as f32).collect();
    let expected = Array::from_shape_vec((1, channels, 1, 1), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// Hand-calculation (backward through the parallel branch):
/// After the forward above, the arg-max of channel `c`'s 2x2 plane is its local
/// flat index 3 (the value 4c+3 sits at the last of the four contiguous slots).
///
/// With grad_output of shape [1, 32, 1, 1] and grad_output[c] = c+1, max-pool
/// backward routes each upstream gradient to its plane's arg-max and leaves every
/// other input position at 0. So channel c's 2x2 grad-input plane is
///   [0, 0, 0, c+1]   (only local index 3 receives the gradient),
/// and the full flat grad-input (shape [1, 32, 2, 2]) is
///   [0,0,0,1, 0,0,0,2, 0,0,0,3, ..., 0,0,0,32].
///
/// Distinct per-channel gradients again catch any mis-ordered parallel assembly
/// in `windowed_pool_backward`.
#[test]
fn max_pooling_2d_parallel_planes_backward_routes_per_plane() {
    let channels = 32; // bc_total = 32, parallel branch
    let mut layer = MaxPooling2D::new((2, 2), vec![1, channels, 2, 2], None).unwrap();

    let data: Vec<f32> = (0..(channels * 4)).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, channels, 2, 2), data)
        .unwrap()
        .into_dyn();
    // Forward to populate the arg-max cache (one per channel, all at local index 3).
    layer.forward(&x).unwrap();

    // Upstream gradient: distinct value per channel, grad[c] = c+1.
    let grad_data: Vec<f32> = (0..channels).map(|c| (c + 1) as f32).collect();
    let grad_out = Array::from_shape_vec((1, channels, 1, 1), grad_data)
        .unwrap()
        .into_dyn();

    let grad_in = layer.backward(&grad_out).unwrap();
    assert_eq!(grad_in.shape(), &[1, channels, 2, 2]);

    // Expected: per channel c, the four positions are [0, 0, 0, c+1].
    let mut expected_data: Vec<f32> = Vec::with_capacity(channels * 4);
    for c in 0..channels {
        expected_data.extend_from_slice(&[0.0, 0.0, 0.0, (c + 1) as f32]);
    }
    let expected = Array::from_shape_vec((1, channels, 2, 2), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&grad_in, &expected, 1e-6);
}
