//! Integration tests for MaxPooling 1D/2D/3D and GlobalMaxPooling 1D/2D/3D: forward values,
//! predict-equals-forward, output_shape strings, and error paths. Gradient correctness lives
//! in gradient_check.rs.
//!
//! Every tensor here is channels-last. The 1D layers use `[batch, length, channels]`, the 2D
//! layers use `[batch, height, width, channels]`, and the 3D layers use
//! `[batch, depth, height, width, channels]`.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::pooling::global_max_pooling_1d::GlobalMaxPooling1D;
use rustyml::neural_network::layers::pooling::global_max_pooling_2d::GlobalMaxPooling2D;
use rustyml::neural_network::layers::pooling::global_max_pooling_3d::GlobalMaxPooling3D;
use rustyml::neural_network::layers::pooling::max_pooling_1d::MaxPooling1D;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::pooling::max_pooling_3d::MaxPooling3D;
use rustyml::neural_network::traits::{Layer, UnaryLayer};
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// MaxPooling1D - forward values

/// MaxPooling1D forward output for pool=2, stride=2 on a single channel
#[test]
fn max_pooling_1d_forward_values_pool2_stride2() {
    let mut layer = MaxPooling1D::new(2);

    let x = Array::from_shape_vec((1, 4, 1), vec![3.0f32, 1.0, 4.0, 1.0])
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2, 1]);

    // Windows [3, 1] -> 3 and [4, 1] -> 4
    let expected = Array::from_shape_vec((1, 2, 1), vec![3.0f32, 4.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling1D forward output for pool=3, stride=1 (overlapping windows)
#[test]
fn max_pooling_1d_forward_values_pool3_stride1() {
    let mut layer = MaxPooling1D::new(3).with_stride(1).unwrap();

    let x = Array::from_shape_vec((1, 6, 1), vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 4, 1]);

    // (6 - 3) / 1 + 1 = 4 windows: [0,1,2] -> 2, [1,2,3] -> 3, [2,3,4] -> 4, [3,4,5] -> 5
    let expected = Array::from_shape_vec((1, 4, 1), vec![2.0f32, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling1D pools each of 2 channels independently
#[test]
fn max_pooling_1d_forward_values_two_channels() {
    let mut layer = MaxPooling1D::new(2);

    // Channels-last: each row is 1 position, holding (ch0, ch1)
    // ch0 along the length is [0, 1, 2, 3, 4, 5]. ch1 is [10, 9, 8, 7, 6, 5].
    let data: Vec<f32> = vec![
        0.0, 10.0, // pos 0
        1.0, 9.0, // pos 1
        2.0, 8.0, // pos 2
        3.0, 7.0, // pos 3
        4.0, 6.0, // pos 4
        5.0, 5.0, // pos 5
    ];
    let x = Array::from_shape_vec((1, 6, 2), data).unwrap().into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 3, 2]);

    // ch0: max(0,1)=1, max(2,3)=3, max(4,5)=5. ch1: max(10,9)=10, max(8,7)=8, max(6,5)=6.
    let expected_data: Vec<f32> = vec![1.0, 10.0, 3.0, 8.0, 5.0, 6.0];
    let expected = Array::from_shape_vec((1, 3, 2), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling1D pools each batch element independently
#[test]
fn max_pooling_1d_forward_values_batch() {
    let mut layer = MaxPooling1D::new(2);

    let data: Vec<f32> = vec![
        // batch 0, single channel
        5.0, 3.0, 7.0, 2.0, // batch 1, single channel
        1.0, 9.0, 4.0, 6.0,
    ];
    let x = Array::from_shape_vec((2, 4, 1), data).unwrap().into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[2, 2, 1]);

    // batch 0: max(5,3)=5, max(7,2)=7. batch 1: max(1,9)=9, max(4,6)=6.
    let expected_data: Vec<f32> = vec![5.0, 7.0, 9.0, 6.0];
    let expected = Array::from_shape_vec((2, 2, 1), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling1D - predict equals forward in eval mode

#[test]
fn max_pooling_1d_predict_equals_forward() {
    let mut layer = MaxPooling1D::new(2);
    layer.build(&Shape::known(&[1, 6, 1])).unwrap();

    let data: Vec<f32> = vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0];
    let x = Array::from_shape_vec((1, 6, 1), data).unwrap().into_dyn();

    let fwd = layer.forward(&x, &mut Ctx::training()).unwrap();
    let pred = layer.forward(&x, &mut Ctx::inference()).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// MaxPooling1D - output_shape string

/// output_shape reports the pooled spatial length as "(None, 4, 2)"
#[test]
fn max_pooling_1d_output_shape_string() {
    // [batch, length, channels] = [1, 8, 2]. The length pools to (8 - 2) / 2 + 1 = 4.
    let mut layer = MaxPooling1D::new(2).with_stride(2).unwrap();
    layer.build(&Shape::known(&[1, 8, 2])).unwrap();
    let s = layer.output_shape();
    assert_eq!(s, "(None, 4, 2)");
}

/// Global pooling layers return "Unknown" for output_shape before any forward pass
#[test]
fn max_pooling_1d_output_shape_unknown_before_forward() {
    let layer = GlobalMaxPooling1D::new();
    assert_eq!(layer.output_shape(), "Unknown");
}

// MaxPooling1D - build error paths

#[test]
fn max_pooling_1d_err_pool_size_invalid() {
    // pool_size=0 and pool_size > input length both hit the InvalidParameter guard of the build
    for pool_size in [0_usize, 5] {
        let result = MaxPooling1D::new(pool_size).build(&Shape::known(&[2, 4, 3]));
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for pool_size={pool_size}, got {result:?}"
        );
    }
}

#[test]
fn max_pooling_1d_err_stride_zero() {
    let result = MaxPooling1D::new(2).with_stride(0);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_1d_err_wrong_ndim() {
    // The build shape must be rank 3. A rank-2 shape names the layer and the rank it wanted
    let result = MaxPooling1D::new(2).build(&Shape::known(&[2, 8]));
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for a rank-2 build shape, got {result:?}"
    );
}

// MaxPooling1D - backward-before-forward error

#[test]
fn max_pooling_1d_backward_before_forward_err() {
    let layer = MaxPooling1D::new(2);
    let grad = Array::ones((1, 2, 1)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
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
    let mut layer = MaxPooling1D::new(2);
    // Pass a 2D tensor instead of 3D
    let x = Array::ones((1, 4)).into_dyn();
    let result = layer.forward_mut(&x, &mut Ctx::training());
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
    let mut layer = MaxPooling2D::new((2, 2));

    // Single channel, so the row-major fill is just value(h, w) = 4h + w
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 4, 4, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);

    // max(0,1,4,5)=5, max(2,3,6,7)=7, max(8,9,12,13)=13, max(10,11,14,15)=15
    let expected = Array::from_shape_vec((1, 2, 2, 1), vec![5.0f32, 7.0, 13.0, 15.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling2D forward output for a 4x4 input with pool=(2,2), stride=(1,1)
#[test]
fn max_pooling_2d_forward_values_stride1() {
    let mut layer = MaxPooling2D::new((2, 2)).with_strides((1, 1)).unwrap();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 4, 4, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 3, 3, 1]);

    // value(h, w) = 4h + w, so the window at (i, j) maxes at its bottom-right cell 4(i+1)+(j+1)
    let expected_data: Vec<f32> = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    let expected = Array::from_shape_vec((1, 3, 3, 1), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling2D pools each of 2 channels independently when pool equals the spatial dims
#[test]
fn max_pooling_2d_forward_values_two_channels() {
    let mut layer = MaxPooling2D::new((2, 2));

    // Channels-last: each (h, w) cell holds (ch0, ch1). ch0 covers 0..3 over the 2x2 plane and
    // ch1 covers 4..7, interleaved in row-major (h, w, c) order
    let data: Vec<f32> = vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
    let x = Array::from_shape_vec((1, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 2]);

    // ch0 plane max = 3, ch1 plane max = 7
    let expected = Array::from_shape_vec((1, 1, 1, 2), vec![3.0f32, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling2D - predict equals forward

#[test]
fn max_pooling_2d_predict_equals_forward() {
    let mut layer = MaxPooling2D::new((2, 2));
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 4, 4, 1), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x, &mut Ctx::training()).unwrap();
    let pred = layer.forward(&x, &mut Ctx::inference()).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// MaxPooling2D - output_shape string

/// output_shape reports the pooled 2D shape as "(None, 3, 3, 3)"
#[test]
fn max_pooling_2d_output_shape_string() {
    // [batch, height, width, channels] = [1, 6, 6, 3]. 6 -> (6 - 2) / 2 + 1 = 3 on both axes.
    let mut layer = MaxPooling2D::new((2, 2));
    layer.build(&Shape::known(&[1, 6, 6, 3])).unwrap();
    assert_eq!(layer.output_shape(), "(None, 3, 3, 3)");
}

// MaxPooling2D - build error paths

#[test]
fn max_pooling_2d_err_pool_size_invalid() {
    // The build rejects a zero or too-large pool extent in the height or the width position
    for pool in [(0, 2), (2, 0), (5, 2), (2, 5)] {
        let result = MaxPooling2D::new(pool).build(&Shape::known(&[2, 4, 4, 3]));
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for pool={pool:?}, got {result:?}"
        );
    }
}

#[test]
fn max_pooling_2d_err_stride_zero() {
    let result = MaxPooling2D::new((2, 2)).with_strides((0, 1));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_2d_err_zero_spatial_extent() {
    // A window of 2 does not fit an extent of 0, so the build refuses both cases
    let zero_height = MaxPooling2D::new((2, 2)).build(&Shape::known(&[2, 0, 4, 3]));
    assert!(
        matches!(zero_height, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for height=0, got {zero_height:?}"
    );
    let zero_width = MaxPooling2D::new((2, 2)).build(&Shape::known(&[2, 4, 0, 3]));
    assert!(
        matches!(zero_width, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for width=0, got {zero_width:?}"
    );
}

#[test]
fn max_pooling_2d_err_wrong_ndim() {
    // The build shape must be rank 4. A rank-3 shape names the layer and the rank it wanted
    let result = MaxPooling2D::new((2, 2)).build(&Shape::known(&[2, 4, 3]));
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for a rank-3 build shape, got {result:?}"
    );
}

// MaxPooling2D - backward-before-forward error

#[test]
fn max_pooling_2d_backward_before_forward_err() {
    let layer = MaxPooling2D::new((2, 2));
    let grad = Array::ones((1, 2, 2, 1)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
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
    let mut layer = MaxPooling2D::new((2, 2));
    // Pass a 3D tensor instead of 4D
    let x = Array::ones((1, 4, 4)).into_dyn();
    let result = layer.forward_mut(&x, &mut Ctx::training());
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
    let mut layer = MaxPooling3D::new((2, 2, 2));

    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 2, 2, 2, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);

    // The single window covers the whole volume, so the output is max(0..7) = 7
    let expected = Array::from_shape_vec((1, 1, 1, 1, 1), vec![7.0f32])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// MaxPooling3D forward output for a 4x4x4 volume with pool=(2,2,2), stride=(2,2,2)
#[test]
fn max_pooling_3d_forward_values_4x4x4_pool2x2x2() {
    let mut layer = MaxPooling3D::new((2, 2, 2));

    // Single channel, so value(d, h, w) = 16d + 4h + w
    let data: Vec<f32> = (0..64).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 4, 4, 4, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2, 1]);

    // Window (i, j, k) maxes at (2i+1, 2j+1, 2k+1): 16(2i+1) + 4(2j+1) + (2k+1)
    let expected_data: Vec<f32> = vec![21.0, 23.0, 29.0, 31.0, 53.0, 55.0, 61.0, 63.0];
    let expected = Array::from_shape_vec((1, 2, 2, 2, 1), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling3D - predict equals forward

#[test]
fn max_pooling_3d_predict_equals_forward() {
    let mut layer = MaxPooling3D::new((2, 2, 2));
    layer.build(&Shape::known(&[1, 4, 4, 4, 1])).unwrap();

    let data: Vec<f32> = (0..64).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 4, 4, 4, 1), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward(&x, &mut Ctx::training()).unwrap();
    let pred = layer.forward(&x, &mut Ctx::inference()).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// MaxPooling3D - output_shape string

/// output_shape reports the pooled 3D shape as "(None, 2, 2, 2, 2)"
#[test]
fn max_pooling_3d_output_shape_string() {
    // [batch, depth, height, width, channels] = [1, 4, 4, 4, 2]. Each spatial axis 4 -> 2.
    let mut layer = MaxPooling3D::new((2, 2, 2));
    layer.build(&Shape::known(&[1, 4, 4, 4, 2])).unwrap();
    assert_eq!(layer.output_shape(), "(None, 2, 2, 2, 2)");
}

// MaxPooling3D - build error paths

#[test]
fn max_pooling_3d_err_pool_size_invalid() {
    // A pool depth of 0, and a pool depth above the input depth, both hit the build guard
    for pool in [(0, 2, 2), (5, 2, 2)] {
        let result = MaxPooling3D::new(pool).build(&Shape::known(&[1, 4, 4, 4, 2]));
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for pool={pool:?}, got {result:?}"
        );
    }
}

#[test]
fn max_pooling_3d_err_stride_zero() {
    let result = MaxPooling3D::new((2, 2, 2)).with_strides((0, 2, 2));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

#[test]
fn max_pooling_3d_err_zero_spatial_extent() {
    // A window of 2 does not fit an extent of 0, so the build refuses both cases
    let zero_depth = MaxPooling3D::new((2, 2, 2)).build(&Shape::known(&[1, 0, 4, 4, 2]));
    assert!(
        matches!(zero_depth, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for depth=0, got {zero_depth:?}"
    );
    let zero_width = MaxPooling3D::new((2, 2, 2)).build(&Shape::known(&[1, 4, 4, 0, 2]));
    assert!(
        matches!(zero_width, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for width=0, got {zero_width:?}"
    );
}

#[test]
fn max_pooling_3d_err_wrong_ndim() {
    // The build shape must be rank 5. A rank-4 shape names the layer and the rank it wanted
    let result = MaxPooling3D::new((2, 2, 2)).build(&Shape::known(&[1, 4, 4, 2]));
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for a rank-4 build shape, got {result:?}"
    );
}

// MaxPooling3D - backward-before-forward error

#[test]
fn max_pooling_3d_backward_before_forward_err() {
    let layer = MaxPooling3D::new((2, 2, 2));
    let grad = Array::ones((1, 2, 2, 2, 1)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
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
    let mut layer = MaxPooling3D::new((2, 2, 2));
    // Pass a 4D tensor instead of 5D
    let x = Array::ones((1, 4, 4, 1)).into_dyn();
    let result = layer.forward_mut(&x, &mut Ctx::training());
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

    // [batch, length, channels] = [1, 5, 2]. ch0 = [0,1,2,3,4], ch1 = [9,8,7,6,5].
    let data: Vec<f32> = vec![0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0];
    let x = Array::from_shape_vec((1, 5, 2), data).unwrap().into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    // ch0 max = 4, ch1 max = 9
    let expected = Array::from_shape_vec((1, 2), vec![4.0f32, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GlobalMaxPooling1D reduces each channel per batch element to its maximum
#[test]
fn global_max_pooling_1d_forward_values_batch() {
    let mut layer = GlobalMaxPooling1D::new();

    // [batch, length, channels] = [2, 4, 3]. Each row is 1 position holding (ch0, ch1, ch2).
    // batch 0: ch0 = [1,5,3,2], ch1 = [0,0,0,8], ch2 = [7,6,5,4]
    // batch 1: ch0 = [9,1,1,1], ch1 = [2,2,6,2], ch2 = [3,3,3,10]
    let data: Vec<f32> = vec![
        // batch 0
        1.0, 0.0, 7.0, // pos 0
        5.0, 0.0, 6.0, // pos 1
        3.0, 0.0, 5.0, // pos 2
        2.0, 8.0, 4.0, // pos 3
        // batch 1
        9.0, 2.0, 3.0, // pos 0
        1.0, 2.0, 3.0, // pos 1
        1.0, 6.0, 3.0, // pos 2
        1.0, 2.0, 10.0, // pos 3
    ];
    let x = Array::from_shape_vec((2, 4, 3), data).unwrap().into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[2, 3]);

    // batch 0: [5, 8, 7]. batch 1: [9, 6, 10].
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

    let data: Vec<f32> = vec![0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0];
    let x = Array::from_shape_vec((1, 5, 2), data).unwrap().into_dyn();

    let fwd = layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    let pred = layer.forward(&x, &mut Ctx::inference()).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// GlobalMaxPooling1D - output_shape string

/// output_shape returns "(batch_size, channels)" once a forward pass caches the input shape
#[test]
fn global_max_pooling_1d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling1D::new();
    let x = Array::from_shape_vec((1, 5, 2), vec![0.0f32; 10])
        .unwrap()
        .into_dyn();
    layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(layer.output_shape(), "(None, 2)");
}

// GlobalMaxPooling1D - backward-before-forward error

#[test]
fn global_max_pooling_1d_backward_before_forward_err() {
    let layer = GlobalMaxPooling1D::new();
    let grad = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
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
    let result = layer.forward_mut(&x, &mut Ctx::training());
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

    // [batch, height, width, channels] = [1, 3, 3, 2]. ch0 holds 0..8 over the 3x3 plane and
    // ch1 holds 10..18, interleaved in row-major (h, w, c) order.
    let data: Vec<f32> = (0..9).flat_map(|v| [v as f32, (v + 10) as f32]).collect();
    let x = Array::from_shape_vec((1, 3, 3, 2), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    // ch0 max = 8, ch1 max = 18
    let expected = Array::from_shape_vec((1, 2), vec![8.0f32, 18.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GlobalMaxPooling2D reduces each channel plane per batch element to its maximum
#[test]
fn global_max_pooling_2d_forward_values_batch() {
    let mut layer = GlobalMaxPooling2D::new();

    // [batch, height, width, channels] = [2, 2, 2, 1]
    let data: Vec<f32> = vec![
        3.0, 1.0, 2.0, 4.0, // batch 0
        5.0, 9.0, 8.0, 7.0, // batch 1
    ];
    let x = Array::from_shape_vec((2, 2, 2, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[2, 1]);

    // batch 0 max = 4, batch 1 max = 9
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
    let x = Array::from_shape_vec((1, 3, 3, 2), data)
        .unwrap()
        .into_dyn();

    let fwd = layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    let pred = layer.forward(&x, &mut Ctx::inference()).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// GlobalMaxPooling2D - output_shape string

#[test]
fn global_max_pooling_2d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling2D::new();
    let x = Array::from_shape_vec((2, 4, 4, 3), vec![0.0f32; 2 * 4 * 4 * 3])
        .unwrap()
        .into_dyn();
    layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(layer.output_shape(), "(None, 3)");
}

// GlobalMaxPooling2D - backward-before-forward error

#[test]
fn global_max_pooling_2d_backward_before_forward_err() {
    let layer = GlobalMaxPooling2D::new();
    let grad = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
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
    let result = layer.forward_mut(&x, &mut Ctx::training());
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

    // [batch, depth, height, width, channels] = [1, 2, 2, 2, 2]. ch0 holds 0..7 over the volume
    // and ch1 holds 8..15, interleaved in row-major (d, h, w, c) order.
    let data: Vec<f32> = (0..8).flat_map(|v| [v as f32, (v + 8) as f32]).collect();
    let x = Array::from_shape_vec((1, 2, 2, 2, 2), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    // ch0 max = 7, ch1 max = 15
    let expected = Array::from_shape_vec((1, 2), vec![7.0f32, 15.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GlobalMaxPooling3D reduces each channel volume per batch element to its maximum
#[test]
fn global_max_pooling_3d_forward_values_batch() {
    let mut layer = GlobalMaxPooling3D::new();

    // [batch, depth, height, width, channels] = [2, 2, 2, 2, 1]: batch 0 holds 0..7, batch 1 8..15
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((2, 2, 2, 2, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
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

    let fwd = layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    let pred = layer.forward(&x, &mut Ctx::inference()).unwrap();

    assert_allclose(&pred, &fwd, 1e-6);
}

// GlobalMaxPooling3D - output_shape string

#[test]
fn global_max_pooling_3d_output_shape_after_forward() {
    let mut layer = GlobalMaxPooling3D::new();
    let x = Array::from_shape_vec((3, 2, 2, 2, 4), vec![0.0f32; 3 * 2 * 2 * 2 * 4])
        .unwrap()
        .into_dyn();
    layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(layer.output_shape(), "(None, 4)");
}

// GlobalMaxPooling3D - backward-before-forward error

#[test]
fn global_max_pooling_3d_backward_before_forward_err() {
    let layer = GlobalMaxPooling3D::new();
    let grad = Array::ones((1, 2)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
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
    let result = layer.forward_mut(&x, &mut Ctx::training());
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
    let mut layer = MaxPooling1D::new(2);

    let x = Array::from_shape_vec((1, 4, 1), vec![-3.0f32, -1.0, -4.0, -2.0])
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2, 1]);

    // max(-3, -1) = -1 and max(-4, -2) = -2
    let expected = Array::from_shape_vec((1, 2, 1), vec![-1.0f32, -2.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling2D - non-square spatial dims

/// MaxPooling2D forward output when height and width pool to different sizes
#[test]
fn max_pooling_2d_forward_non_square_spatial() {
    // [batch, height, width, channels] = [1, 2, 4, 1]: height 2 -> 1, width 4 -> 2
    let mut layer = MaxPooling2D::new((2, 2));

    // Single channel, so value(h, w) = 4h + w
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, 2, 4, 1), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 1]);

    // max(0,1,4,5) = 5 and max(2,3,6,7) = 7
    let expected = Array::from_shape_vec((1, 1, 2, 1), vec![5.0f32, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// MaxPooling2D - parallel assembly

// The pooling engine runs in parallel once batch * out_positions * channels * window taps
// clears the gate in `crate::tuning::pool`. Forward splits by output-position block, and
// backward splits by channel slab, needing more than 1 slab to leave the serial path.

/// Parallel forward keeps every output position and channel in place across position blocks
#[test]
fn max_pooling_2d_parallel_forward_distinct_maxima() {
    // 24 * 24 = 576 output positions, clearing the parallel gate in `crate::tuning::pool`
    let (side, channels) = (48_usize, 8_usize);
    let mut layer = MaxPooling2D::new((2, 2));

    // Row-major fill: value(h, w, c) = (h * 48 + w) * 8 + c
    let data: Vec<f32> = (0..(side * side * channels)).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, side, side, channels), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    let out_side = side / 2;
    assert_eq!(out.shape(), &[1, out_side, out_side, channels]);

    // Within window (i, j), the value grows with h then w. The max is the bottom-right tap
    // (h, w) = (2i + 1, 2j + 1). That tap gives value ((2i + 1) * 48 + 2j + 1) * 8 + c.
    let mut expected_data: Vec<f32> = Vec::with_capacity(out_side * out_side * channels);
    for i in 0..out_side {
        for j in 0..out_side {
            for c in 0..channels {
                expected_data.push((((2 * i + 1) * side + 2 * j + 1) * channels + c) as f32);
            }
        }
    }
    let expected = Array::from_shape_vec((1, out_side, out_side, channels), expected_data)
        .unwrap()
        .into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// Parallel backward routes each upstream gradient to its window's arg-max, channel slab by slab
#[test]
fn max_pooling_2d_parallel_backward_routes_per_window() {
    // 64 channels, clearing the parallel gate in `crate::tuning::pool`
    let (side, channels) = (16_usize, 64_usize);
    let mut layer = MaxPooling2D::new((2, 2));

    let data: Vec<f32> = (0..(side * side * channels)).map(|v| v as f32).collect();
    let x = Array::from_shape_vec((1, side, side, channels), data)
        .unwrap()
        .into_dyn();
    // Forward to populate the arg-max cache: window (i, j) picks (2i + 1, 2j + 1) in every channel
    let mut ctx = Ctx::training();
    layer.forward_mut(&x, &mut ctx).unwrap();

    // Upstream gradient: a distinct value per (position, channel), grad(i, j, c) = flat index + 1
    let out_side = side / 2;
    let grad_data: Vec<f32> = (0..(out_side * out_side * channels))
        .map(|v| (v + 1) as f32)
        .collect();
    let grad_out = Array::from_shape_vec((1, out_side, out_side, channels), grad_data)
        .unwrap()
        .into_dyn();

    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();
    assert_eq!(grad_in.shape(), &[1, side, side, channels]);

    // Max pooling routes each upstream value to the arg-max cell alone. The other 3 taps
    // of every window receive zero.
    let mut expected = Array::zeros((1, side, side, channels));
    for i in 0..out_side {
        for j in 0..out_side {
            for c in 0..channels {
                let g = (((i * out_side + j) * channels + c) + 1) as f32;
                expected[[0, 2 * i + 1, 2 * j + 1, c]] = g;
            }
        }
    }
    assert_allclose(&grad_in, &expected.into_dyn(), 1e-6);
}

/// Same padding rounds the output up to ceil(in/stride) and the trailing windows see only the
/// in-bounds cells (padding is virtual). For a 3x3 input, pool 2x2, stride 2: output is 2x2
#[test]
fn max_pool_2d_same_padding_3x3() {
    let mut layer = MaxPooling2D::new((2, 2))
        .with_strides((2, 2))
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec((1, 3, 3, 1), (1..=9).map(|v| v as f32).collect())
        .unwrap()
        .into_dyn();
    let out = layer.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    // pad_total = (2 - 1) * 2 + 2 - 3 = 1, all of it trailing, so the windows start at 0 and 2
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 5.0, epsilon = 1e-6); // max(1,2,4,5)
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 6.0, epsilon = 1e-6); // max(3,6) (col 3 is padding)
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 8.0, epsilon = 1e-6); // max(7,8) (row 3 is padding)
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 9.0, epsilon = 1e-6); // just 9
}

// Non-finite input: the arg-max of a window that no element wins

// The max fold starts each window at `f32::NEG_INFINITY`, and `-inf > -inf` is false, so no
// element of an all-negative-infinity window ever wins the fold. The recorded arg-max is
// therefore the seed alone: the first element of the window, on the channel of the output
// element. Keras 3 on the JAX backend follows the same rule for the windowed layers.

/// MaxPooling1D routes the gradient of an all-negative-infinity window to the first position of
/// that window, on the channel of the output element
#[test]
fn max_pooling_1d_all_negative_infinity_window_keeps_gradient_inside_window() {
    let mut layer = MaxPooling1D::new(2);

    // Positions 0 and 1 hold finite values, and positions 2 and 3 hold negative infinity on
    // both channels. Window 1 therefore covers negative infinity alone
    let x = Array::from_shape_vec(
        (1, 4, 2),
        vec![
            1.0f32,
            5.0,
            2.0,
            6.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
    )
    .unwrap()
    .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2]);
    // Window 0 keeps its maxima, and window 1 keeps the start value of the fold
    assert_abs_diff_eq!(out[[0, 0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out[[0, 0, 1]], 6.0, epsilon = 1e-6);
    assert_eq!(out[[0, 1, 0]], f32::NEG_INFINITY);
    assert_eq!(out[[0, 1, 1]], f32::NEG_INFINITY);

    let grad_out = Array::from_shape_vec((1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();

    // Window 1 starts at position 2, so its gradient lands there and not at position 0
    let expected =
        Array::from_shape_vec((1, 4, 2), vec![0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
            .unwrap()
            .into_dyn();
    assert_allclose(&grad_in, &expected, 1e-6);
}

/// MaxPooling2D routes the gradient of an all-negative-infinity window to the first cell of that
/// window in row-major window order
#[test]
fn max_pooling_2d_all_negative_infinity_window_keeps_gradient_inside_window() {
    let mut layer = MaxPooling2D::new((2, 2));

    // The top-right window holds negative infinity in all 4 cells
    let x = Array::from_shape_vec(
        (1, 4, 4, 1),
        vec![
            1.0f32,
            9.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            9.0,
            3.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            4.0,
            -1.0,
            7.0,
            7.0,
            0.5,
            4.0,
            7.0,
            5.0,
        ],
    )
    .unwrap()
    .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 9.0, epsilon = 1e-6);
    assert_eq!(out[[0, 0, 1, 0]], f32::NEG_INFINITY);
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 7.0, epsilon = 1e-6);

    let grad_out = Array::from_shape_vec((1, 2, 2, 1), vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();

    // Window (0, 0) ties at (0, 1) and (1, 0), and the first tie wins. Window (0, 1) covers
    // negative infinity alone, so its gradient lands on its own first cell (0, 2). Window
    // (1, 1) ties at (2, 2), (2, 3), and (3, 2), and the first tie wins
    let mut expected = Array::zeros((1, 4, 4, 1));
    expected[[0, 0, 1, 0]] = 1.0;
    expected[[0, 0, 2, 0]] = 2.0;
    expected[[0, 2, 0, 0]] = 3.0;
    expected[[0, 2, 2, 0]] = 4.0;
    assert_allclose(&grad_in, &expected.into_dyn(), 1e-6);
}

/// MaxPooling3D routes the gradient of an all-negative-infinity window to the first voxel of
/// that window in row-major window order
#[test]
fn max_pooling_3d_all_negative_infinity_window_keeps_gradient_inside_window() {
    let mut layer = MaxPooling3D::new((2, 2, 2));

    // The width axis holds 4 positions, so the output holds 2 windows. Widths 2 and 3 hold
    // negative infinity at every depth and height, so window 1 covers negative infinity alone
    let n = f32::NEG_INFINITY;
    let x = Array::from_shape_vec(
        (1, 2, 2, 4, 1),
        vec![
            1.0f32, 3.0, n, n, // depth 0, height 0
            8.0, 2.0, n, n, // depth 0, height 1
            5.0, 0.0, n, n, // depth 1, height 0
            -4.0, 6.0, n, n, // depth 1, height 1
        ],
    )
    .unwrap()
    .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 2, 1]);
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], 8.0, epsilon = 1e-6);
    assert_eq!(out[[0, 0, 0, 1, 0]], f32::NEG_INFINITY);

    let grad_out = Array::from_shape_vec((1, 1, 1, 2, 1), vec![1.0f32, 2.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();

    // Window 0 has its maximum 8.0 at depth 0, height 1, width 0. Window 1 covers widths 2 and
    // 3, and its first voxel in window order is depth 0, height 0, width 2
    let mut expected = Array::zeros((1, 2, 2, 4, 1));
    expected[[0, 0, 1, 0, 0]] = 1.0;
    expected[[0, 0, 0, 2, 0]] = 2.0;
    assert_allclose(&grad_in, &expected.into_dyn(), 1e-6);
}

/// GlobalMaxPooling1D keeps the gradient of an all-negative-infinity channel on that channel
#[test]
fn global_max_pooling_1d_all_negative_infinity_keeps_gradient_on_its_channel() {
    let mut layer = GlobalMaxPooling1D::new();

    // Channel 0 holds finite values, and channel 1 holds negative infinity at every position
    let n = f32::NEG_INFINITY;
    let x = Array::from_shape_vec((1, 4, 2), vec![1.0f32, n, 2.0, n, 3.0, n, 0.0, n])
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0, epsilon = 1e-6);
    assert_eq!(out[[0, 1]], f32::NEG_INFINITY);

    let grad_out = Array::from_shape_vec((1, 2), vec![1.0f32, 2.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();

    // Channel 0 has its maximum at position 2. Channel 1 has no winner, so its gradient goes to
    // position 0 of channel 1, and never to channel 0
    let mut expected = Array::zeros((1, 4, 2));
    expected[[0, 2, 0]] = 1.0;
    expected[[0, 0, 1]] = 2.0;
    assert_allclose(&grad_in, &expected.into_dyn(), 1e-6);
}

/// GlobalMaxPooling1D keeps the channel of a gradient when the reduction spans more than 1
/// fold block
#[test]
fn global_max_pooling_1d_all_negative_infinity_across_fold_blocks() {
    // The fold block of a 2-channel input holds 8192 positions, so 8200 positions need 2 blocks
    let positions = 8200_usize;
    let mut layer = GlobalMaxPooling1D::new();

    let n = f32::NEG_INFINITY;
    let mut data = Vec::with_capacity(positions * 2);
    for p in 0..positions {
        data.push(p as f32);
        data.push(n);
    }
    let x = Array::from_shape_vec((1, positions, 2), data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_abs_diff_eq!(out[[0, 0]], (positions - 1) as f32, epsilon = 1e-6);
    assert_eq!(out[[0, 1]], f32::NEG_INFINITY);

    let grad_out = Array::from_shape_vec((1, 2), vec![1.0f32, 2.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();
    assert_eq!(grad_in.shape(), &[1, positions, 2]);

    // Channel 0 peaks in the last block, and channel 1 keeps position 0 of its own channel
    assert_abs_diff_eq!(grad_in[[0, positions - 1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_in[[0, 0, 1]], 2.0, epsilon = 1e-6);
    let total: f32 = grad_in.iter().sum();
    assert_abs_diff_eq!(total, 3.0, epsilon = 1e-6);
}

/// GlobalMaxPooling2D keeps the gradient of an all-negative-infinity channel on that channel
#[test]
fn global_max_pooling_2d_all_negative_infinity_keeps_gradient_on_its_channel() {
    let mut layer = GlobalMaxPooling2D::new();

    let n = f32::NEG_INFINITY;
    let x = Array::from_shape_vec((1, 2, 2, 2), vec![1.0f32, n, 2.0, n, 3.0, n, 0.0, n])
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0, epsilon = 1e-6);
    assert_eq!(out[[0, 1]], f32::NEG_INFINITY);

    let grad_out = Array::from_shape_vec((1, 2), vec![1.0f32, 2.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();

    let mut expected = Array::zeros((1, 2, 2, 2));
    expected[[0, 1, 0, 0]] = 1.0;
    expected[[0, 0, 0, 1]] = 2.0;
    assert_allclose(&grad_in, &expected.into_dyn(), 1e-6);
}

/// GlobalMaxPooling3D keeps the gradient of an all-negative-infinity channel on that channel
#[test]
fn global_max_pooling_3d_all_negative_infinity_keeps_gradient_on_its_channel() {
    let mut layer = GlobalMaxPooling3D::new();

    let n = f32::NEG_INFINITY;
    let x = Array::from_shape_vec((1, 2, 1, 2, 2), vec![1.0f32, n, 2.0, n, 3.0, n, 0.0, n])
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    assert_abs_diff_eq!(out[[0, 0]], 3.0, epsilon = 1e-6);
    assert_eq!(out[[0, 1]], f32::NEG_INFINITY);

    let grad_out = Array::from_shape_vec((1, 2), vec![1.0f32, 2.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad_out, &mut ctx).unwrap();

    let mut expected = Array::zeros((1, 2, 1, 2, 2));
    expected[[0, 1, 0, 0, 0]] = 1.0;
    expected[[0, 0, 0, 0, 1]] = 2.0;
    assert_allclose(&grad_in, &expected.into_dyn(), 1e-6);
}
