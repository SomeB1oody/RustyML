//! Integration tests for Sequential model save/load round-trips.
//!
//! Each test builds a model, either with known injected weights or trained for a few steps. It
//! saves the model and reloads it into a fresh model of the same architecture. Then it checks
//! that predict output matches element-wise (within ~1e-6) across the round-trip.
//!
//! Error paths cover layer-count, layer-type, and weight-shape mismatches (all
//! ModelStructureMismatch). They also cover a nonexistent file (IoError::Std), corrupt binary
//! data (IoError::Serialization), and a wrong magic tag or format version
//! (IoError::UnsupportedModelFormat).
//!
//! The last 2 tests pin the atomicity of a refusal. A load validates the whole file before it
//! writes any array, so a refusal leaves every array of the model bit for bit as it was.

use crate::common::assert_allclose;
use ndarray::Array;
use rustyml::error::{Error, IoError};
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::p_relu::PReLU;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::checkpoint::{
    LayerCheckpoint, MODEL_FORMAT_VERSION, MODEL_MAGIC, ModelCheckpoint, WeightRecord,
};
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_1d_transpose::Conv1DTranspose;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_2d_transpose::Conv2DTranspose;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::conv_3d_transpose::Conv3DTranspose;
use rustyml::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::embedding::Embedding;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::layers::regularization::Dropout;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::layers::regularization::normalization::layer_normalization::LayerNormalization;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::UnaryLayer;
use rustyml::neural_network::traits::WeightKind;
use std::borrow::Cow;
use std::env;

// Helpers

/// Temporary file that deletes itself when dropped
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(name: &str) -> Self {
        let path = env::temp_dir().join(format!("rustyml_serialize_test_{}.bin", name));
        TempFile(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Save `model` to `path`, load into a fresh model built by `make_fresh`, and return it
fn round_trip(
    model: &Sequential,
    make_fresh: impl Fn() -> Sequential,
    path: &std::path::Path,
) -> Sequential {
    model.save_to_path(path).expect("save_to_path failed");
    let mut fresh = make_fresh();
    fresh.load_from_path(path).expect("load_from_path failed");
    fresh
}

// Dense with identity weights (W = I, b = 0): output equals input, and survives a round-trip
#[test]
fn dense_identity_weights_value_check_and_round_trip() {
    let tmp = TempFile::new("dense_identity");

    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    let w = Array::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let b = Array::zeros((1, 2));
    layer.set_weights(w, b).unwrap();

    let model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    let x: Tensor = Array::from_shape_vec((1, 2), vec![3.0f32, -5.0])
        .unwrap()
        .into_dyn();

    // Output == input when W = I, b = 0
    let expected: Tensor = Array::from_shape_vec((1, 2), vec![3.0f32, -5.0])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    assert_allclose(&before, &expected, 1e-6_f32);

    let fresh = round_trip(
        &model,
        || {
            SequentialBuilder::new()
                .add(Dense::new(2, Linear::new()).unwrap())
                .build(&Shape::known(&[2, 2]))
                .unwrap()
        },
        tmp.path(),
    );

    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &expected, 1e-6_f32);
}

// Dense with scaled-identity weights (W = 2*I, b = [1, 1]): value check and round-trip
#[test]
fn dense_scaled_identity_value_check_and_round_trip() {
    let tmp = TempFile::new("dense_scaled");

    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    let w = Array::from_shape_vec((2, 2), vec![2.0f32, 0.0, 0.0, 2.0]).unwrap();
    let b = Array::from_shape_vec((1, 2), vec![1.0f32, 1.0]).unwrap();
    layer.set_weights(w, b).unwrap();

    let model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    let x: Tensor = Array::from_shape_vec((1, 2), vec![0.0f32, 1.0])
        .unwrap()
        .into_dyn();

    // [0*2+1*0+1, 0*0+1*2+1] = [1.0, 3.0]
    let expected: Tensor = Array::from_shape_vec((1, 2), vec![1.0f32, 3.0])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    assert_allclose(&before, &expected, 1e-6_f32);

    let fresh = round_trip(
        &model,
        || {
            SequentialBuilder::new()
                .add(Dense::new(2, Linear::new()).unwrap())
                .build(&Shape::known(&[2, 2]))
                .unwrap()
        },
        tmp.path(),
    );
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &expected, 1e-6_f32);
}

// Dense with zero weights (W = 0): output is the bias broadcast over the batch, and survives
// a round-trip
#[test]
fn dense_zero_weights_bias_only_value_check_and_round_trip() {
    let tmp = TempFile::new("dense_zero_w");

    let mut layer = Dense::new(3, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    let w = Array::zeros((2, 3));
    let b = Array::from_shape_vec((1, 3), vec![0.5f32, -0.5, 1.0]).unwrap();
    layer.set_weights(w, b).unwrap();

    let model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(&[2, 2]))
        .unwrap();

    let x: Tensor = Array::from_shape_vec((2, 2), vec![100.0f32, -200.0, 0.0, 99.0])
        .unwrap()
        .into_dyn();

    // W=0 => any input maps to the bias
    let expected: Tensor = Array::from_shape_vec((2, 3), vec![0.5f32, -0.5, 1.0, 0.5, -0.5, 1.0])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    assert_allclose(&before, &expected, 1e-6_f32);

    let fresh = round_trip(
        &model,
        || {
            SequentialBuilder::new()
                .add(Dense::new(3, Linear::new()).unwrap())
                .build(&Shape::known(&[2, 2]))
                .unwrap()
        },
        tmp.path(),
    );
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &expected, 1e-6_f32);
}

// 2-layer Dense model trained a few steps round-trips
#[test]
fn dense_two_layer_trained_round_trip() {
    let tmp = TempFile::new("dense2");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Dense::new(3, Linear::new()).unwrap())
            .add(Dense::new(2, Linear::new()).unwrap())
            .build(&Shape::known(&[2, 4]))
            .unwrap()
    };

    let mut model = make_arch();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let x: Tensor =
        Array::from_shape_vec((2, 4), vec![0.1f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4])
            .unwrap()
            .into_dyn();
    let y: Tensor = Array::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();
    model.fit(&x, &y, 5).unwrap();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Conv1D round-trip
#[test]
fn conv1d_round_trip() {
    let tmp = TempFile::new("conv1d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Conv1D::new(2, 2, 1, Linear::new()).unwrap())
            .build(&Shape::known(&[1, 5, 1]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec((1, 5, 1), vec![0.1f32, 0.3, -0.2, 0.5, -0.4])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Conv2D round-trip
#[test]
fn conv2d_round_trip() {
    let tmp = TempFile::new("conv2d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Conv2D::new(2, (2, 2), (1, 1), Linear::new()).unwrap())
            .build(&Shape::known(&[1, 4, 4, 1]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 4, 4, 1),
        (0..16).map(|v| 0.1 * v as f32 - 0.7).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Conv3D round-trip
#[test]
fn conv3d_round_trip() {
    let tmp = TempFile::new("conv3d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Conv3D::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap())
            .build(&Shape::known(&[1, 3, 3, 3, 1]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 3, 3, 3, 1),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// DepthwiseConv2D round-trip
#[test]
fn depthwise_conv2d_round_trip() {
    let tmp = TempFile::new("depthwise_conv2d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap())
            .build(&Shape::known(&[1, 4, 4, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// DepthwiseConv1D round-trip
///
/// `depth_multiplier = 2` widens both the kernel and the bias, so the saved shapes differ from
/// the constructor default. A container that dropped the multiplier would fail the shape check
/// on load.
#[test]
fn depthwise_conv1d_round_trip() {
    let tmp = TempFile::new("depthwise_conv1d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(
                DepthwiseConv1D::new(3, 1, Linear::new())
                    .unwrap()
                    .with_depth_multiplier(2)
                    .unwrap(),
            )
            .build(&Shape::known(&[1, 8, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 8, 2),
        (0..16).map(|v| 0.05 * v as f32 - 0.4).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// SeparableConv1D round-trip
///
/// This carries 3 parameter tensors, so it checks that the container writes and reads all 3 in
/// the same order.
#[test]
fn separable_conv1d_round_trip() {
    let tmp = TempFile::new("separable_conv1d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(
                SeparableConv1D::new(3, 3, 1, 2, Linear::new())
                    .unwrap()
                    .with_padding(PaddingType::Same),
            )
            .build(&Shape::known(&[1, 8, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 8, 2),
        (0..16).map(|v| 0.05 * v as f32 - 0.4).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Conv1DTranspose round-trip
#[test]
fn conv1d_transpose_round_trip() {
    let tmp = TempFile::new("conv1d_transpose");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Conv1DTranspose::new(2, 3, 2, Linear::new()).unwrap())
            .build(&Shape::known(&[1, 4, 1]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 4, 1),
        (0..4).map(|v| 0.1 * v as f32 - 0.3).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Conv2DTranspose round-trip
#[test]
fn conv2d_transpose_round_trip() {
    let tmp = TempFile::new("conv2d_transpose");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Conv2DTranspose::new(2, (3, 3), (2, 2), Linear::new()).unwrap())
            .build(&Shape::known(&[1, 3, 3, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 3, 3, 2),
        (0..18).map(|v| 0.05 * v as f32 - 0.4).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Conv3DTranspose round-trip
#[test]
fn conv3d_transpose_round_trip() {
    let tmp = TempFile::new("conv3d_transpose");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Conv3DTranspose::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap())
            .build(&Shape::known(&[1, 2, 2, 2, 1]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 2, 2, 2, 1),
        (0..8).map(|v| 0.05 * v as f32 - 0.2).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// SeparableConv2D round-trip
#[test]
fn separable_conv2d_round_trip() {
    let tmp = TempFile::new("separable_conv2d");

    let make_arch = || {
        SequentialBuilder::new()
            .add(SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap())
            .build(&Shape::known(&[1, 4, 4, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// SimpleRNN round-trip
#[test]
fn simple_rnn_round_trip() {
    let tmp = TempFile::new("simple_rnn");

    let make_arch = || {
        SequentialBuilder::new()
            .add(SimpleRNN::new(3, Tanh::new()).unwrap())
            .build(&Shape::known(&[1, 3, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec((1, 3, 2), vec![0.3f32, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// LSTM round-trip
#[test]
fn lstm_round_trip() {
    let tmp = TempFile::new("lstm");

    let make_arch = || {
        SequentialBuilder::new()
            .add(LSTM::new(3, Tanh::new()).unwrap())
            .build(&Shape::known(&[1, 3, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec((1, 3, 2), vec![0.3f32, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// GRU round-trip
#[test]
fn gru_round_trip() {
    let tmp = TempFile::new("gru");

    let make_arch = || {
        SequentialBuilder::new()
            .add(GRU::new(3, Tanh::new()).unwrap())
            .build(&Shape::known(&[1, 3, 2]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec((1, 3, 2), vec![0.3f32, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Embedding round-trip: the lookup table is the only weight, and a reloaded table must gather
// the identical vectors
#[test]
fn embedding_trained_round_trip_preserves_the_lookup_table() {
    let tmp = TempFile::new("embedding");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Embedding::new(6, 4).unwrap().with_random_state(21))
            .add(Flatten::new())
            .add(Dense::new(1, Linear::new()).unwrap())
            .build(&Shape::known(&[2, 3]))
            .unwrap()
    };

    let x: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32, 5.0, 0.0, 3.0, 3.0, 2.0])
        .unwrap()
        .into_dyn();
    let y: Tensor = Array::from_shape_vec((2, 1), vec![0.5f32, -0.5])
        .unwrap()
        .into_dyn();

    // Train first, so the saved table differs from the one a fresh model initializes
    let mut model = make_arch();
    model.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    model.fit(&x, &y, 5).unwrap();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// PReLU round-trip: the trained slopes must survive, including the shared-axes shape
#[test]
fn p_relu_trained_round_trip_preserves_the_slopes() {
    let tmp = TempFile::new("p_relu");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Dense::new(3, Linear::new()).unwrap())
            .add(PReLU::new(0.25).unwrap())
            .add(Dense::new(1, Linear::new()).unwrap())
            .build(&Shape::known(&[4, 3]))
            .unwrap()
    };

    let x: Tensor = Array::from_shape_vec(
        (4, 3),
        vec![
            -1.0f32, 0.5, -2.0, 1.0, -0.5, 2.0, -3.0, 1.5, -1.0, 0.25, -2.5, 0.75,
        ],
    )
    .unwrap()
    .into_dyn();
    let y: Tensor = Array::from_shape_vec((4, 1), vec![0.5f32, -0.5, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    // Train first, so the saved slopes differ from the 0.25 a fresh model starts from
    let mut model = make_arch();
    model.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    model.fit(&x, &y, 10).unwrap();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// PReLU with shared axes carries a rank-3 slope array, which the dynamic-rank container
// must reproduce exactly
#[test]
fn p_relu_shared_axes_round_trip_keeps_the_slope_rank() {
    let tmp = TempFile::new("p_relu_shared");

    let make_arch = || {
        SequentialBuilder::new()
            .add(
                PReLU::new(0.3)
                    .unwrap()
                    .with_shared_axes(vec![1, 2])
                    .unwrap(),
            )
            .build(&Shape::known(&[2, 3, 3, 2]))
            .unwrap()
    };

    // The saved model carries injected slopes, so a fresh model cannot match it by accident
    let mut injected = PReLU::new(0.3)
        .unwrap()
        .with_shared_axes(vec![1, 2])
        .unwrap();
    injected.build(&Shape::known(&[2, 3, 3, 2])).unwrap();
    injected
        .set_weights(
            Array::from_shape_vec((1, 1, 2), vec![0.4f32, -0.6])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
    let model = SequentialBuilder::new()
        .add(injected)
        .build(&Shape::known(&[2, 3, 3, 2]))
        .unwrap();
    assert_eq!(
        model
            .weight("0.alpha")
            .expect("layer 0 is the PReLU")
            .shape(),
        &[1, 1, 2]
    );

    let x: Tensor = Array::from_shape_vec(
        (2, 3, 3, 2),
        (0..36).map(|v| 0.1 * v as f32 - 1.85).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
    assert_eq!(
        fresh
            .weight("0.alpha")
            .expect("layer 0 is the PReLU")
            .shape(),
        &[1, 1, 2]
    );
}

// BatchNormalization round-trip: the trained moving_mean/moving_variance must survive
// serialization, so eval-mode predict returns the identical tensor afterward
#[test]
fn batch_normalization_trained_round_trip_preserves_running_stats() {
    let tmp = TempFile::new("batchnorm");

    let x_train: Tensor = Array::from_shape_vec(
        (4, 3),
        vec![
            0.5f32, -1.0, 2.0, 1.5, 0.2, -0.7, -1.2, 0.8, 1.1, 0.3, -0.4, 0.9,
        ],
    )
    .unwrap()
    .into_dyn();

    let make_arch = || {
        SequentialBuilder::new()
            .add(BatchNormalization::new(0.9, 1e-5).unwrap())
            .build(&Shape::known(&[4, 3]))
            .unwrap()
    };

    // Train a copy to move running stats away from their defaults
    let mut trainable_model = make_arch();
    trainable_model.compile(
        SGD::new(0.001, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    trainable_model.fit(&x_train, &x_train, 8).unwrap();

    // Eval-mode prediction uses moving_mean / moving_variance
    let before = trainable_model.predict(&x_train).unwrap();

    // Save and restore into a fresh (untrained) model
    trainable_model.save_to_path(tmp.path()).unwrap();
    let mut fresh = make_arch();
    fresh.load_from_path(tmp.path()).unwrap();

    let after = fresh.predict(&x_train).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

/// predict() is deterministic: 2 calls on the same fresh eval-mode model return the same tensor
#[test]
fn batch_normalization_predict_is_deterministic_after_round_trip() {
    let tmp = TempFile::new("batchnorm_det");

    let x: Tensor = Array::from_shape_vec(
        (4, 3),
        vec![
            0.1f32, -0.3, 0.5, 0.7, -1.2, 0.4, -0.5, 0.8, 0.2, 1.1, -0.6, 0.3,
        ],
    )
    .unwrap()
    .into_dyn();

    let make_arch = || {
        SequentialBuilder::new()
            .add(BatchNormalization::new(0.9, 1e-5).unwrap())
            .build(&Shape::known(&[4, 3]))
            .unwrap()
    };

    let mut trainable = make_arch();
    trainable.compile(
        SGD::new(0.001, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    trainable.fit(&x, &x, 5).unwrap();
    trainable.save_to_path(tmp.path()).unwrap();

    let mut fresh = make_arch();
    fresh.load_from_path(tmp.path()).unwrap();

    // 2 calls to predict must agree element-wise
    let p1 = fresh.predict(&x).unwrap();
    let p2 = fresh.predict(&x).unwrap();
    assert_allclose(&p2, &p1, 1e-7_f32);
}

// LayerNormalization round-trip
#[test]
fn layer_normalization_round_trip() {
    let tmp = TempFile::new("layer_norm");

    let make_arch = || {
        SequentialBuilder::new()
            .add(LayerNormalization::new(1e-5).unwrap())
            .build(&Shape::known(&[2, 4]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (2, 4),
        (0..8).map(|v| 0.5 * v as f32 - 1.5).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// GroupNormalization round-trip
#[test]
fn group_normalization_round_trip() {
    let tmp = TempFile::new("group_norm");

    let make_arch = || {
        SequentialBuilder::new()
            .add(GroupNormalization::new(2, 1e-5).unwrap())
            .build(&Shape::known(&[1, 4, 4]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 4, 4),
        (0..16).map(|v| 0.1 * v as f32 - 0.8).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// InstanceNormalization round-trip
#[test]
fn instance_normalization_round_trip() {
    let tmp = TempFile::new("instance_norm");

    let make_arch = || {
        SequentialBuilder::new()
            .add(InstanceNormalization::new(1e-5).unwrap())
            .build(&Shape::known(&[1, 3, 4]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 3, 4),
        (0..12).map(|v| 0.15 * v as f32 - 0.8).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Dense -> Dropout -> Dense round-trips a parameterless layer (Dropout is transparent in eval
// mode). An empty input_shape skips shape validation, so batch size is not fixed.
#[test]
fn mixed_model_with_dropout_round_trip() {
    let tmp = TempFile::new("mixed_dropout");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Dense::new(4, Linear::new()).unwrap())
            // empty input_shape => Dropout skips its shape validator at runtime
            .add(Dropout::new(0.3).unwrap())
            .add(Dense::new(2, Linear::new()).unwrap())
            .build(&Shape::known(&[1, 3]))
            .unwrap()
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec((1, 3), vec![0.5f32, -1.0, 1.5])
        .unwrap()
        .into_dyn();

    // predict() uses eval mode => Dropout is transparent
    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Mixed model Dense -> Dropout -> Dense trained a few steps round-trips
#[test]
fn mixed_model_trained_round_trip() {
    let tmp = TempFile::new("mixed_trained");

    let make_arch = || {
        SequentialBuilder::new()
            .add(Dense::new(4, Linear::new()).unwrap())
            .add(Dropout::new(0.3).unwrap())
            .add(Dense::new(2, Linear::new()).unwrap())
            .build(&Shape::known(&[2, 3]))
            .unwrap()
    };

    let mut model = make_arch();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    // Consistent batch size. Dropout shape validation is off (empty vec).
    let x: Tensor = Array::from_shape_vec((2, 3), vec![0.5f32, -1.0, 1.5, -0.5, 1.0, -1.5])
        .unwrap()
        .into_dyn();
    let y: Tensor = Array::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();
    model.fit(&x, &y, 5).unwrap();

    let before = model.predict(&x).unwrap();
    let fresh = round_trip(&model, make_arch, tmp.path());
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

// Error paths

/// Nonexistent file gives Error::Io(IoError::Std)
#[test]
fn load_from_nonexistent_file_gives_io_error() {
    let mut model = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[2, 2]))
        .unwrap();

    let result = model.load_from_path("/tmp/this_file_definitely_does_not_exist_rustyml_99999.bin");
    match result {
        Err(Error::Io(IoError::Std(_))) => {}
        other => panic!("expected IoError::Std, got {:?}", other),
    }
}

/// A valid header followed by corrupt data gives Error::Io(IoError::Serialization). The loader
/// checks the header first, so this pins the failure mode for a truncated or damaged body.
#[test]
fn load_from_invalid_data_gives_serialization_error() {
    let tmp = TempFile::new("invalid_data");
    let mut bytes = postcard::to_allocvec(&(MODEL_MAGIC, MODEL_FORMAT_VERSION)).unwrap();
    bytes.extend_from_slice(b"\xff\xff\xff not valid postcard data");
    std::fs::write(tmp.path(), &bytes).unwrap();

    let mut model = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[2, 2]))
        .unwrap();

    let result = model.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::Serialization(_))) => {}
        other => panic!("expected IoError::Serialization, got {:?}", other),
    }
}

/// Layer-count mismatch gives Error::Io(IoError::ModelStructureMismatch)
#[test]
fn load_layer_count_mismatch_gives_structure_error() {
    let tmp = TempFile::new("count_mismatch");

    // Save a 1-layer model
    let model_1 = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();
    model_1.save_to_path(tmp.path()).unwrap();

    // Try to load into a 2-layer model
    let mut model_2 = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    let result = model_2.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(_))) => {}
        other => panic!("expected ModelStructureMismatch, got {:?}", other),
    }
}

/// Layer-type mismatch (Dense saved, Conv1D target) gives
/// Error::Io(IoError::ModelStructureMismatch)
#[test]
fn load_layer_type_mismatch_gives_structure_error() {
    let tmp = TempFile::new("type_mismatch");

    let dense_model = SequentialBuilder::new()
        .add(Dense::new(3, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 3]))
        .unwrap();
    dense_model.save_to_path(tmp.path()).unwrap();

    let mut conv_model = SequentialBuilder::new()
        .add(Conv1D::new(2, 2, 1, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 5, 1]))
        .unwrap();

    let result = conv_model.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(_))) => {}
        other => panic!("expected ModelStructureMismatch, got {:?}", other),
    }
}

/// 2 normalization layers that name and shape their arrays identically are still told apart
///
/// InstanceNormalization and GroupNormalization both hold `gamma` and `beta` of length
/// `channels`. A name and a shape therefore say nothing about which of the 2 wrote the file,
/// and the layer type of the position is what separates them. The refusal names both types
#[test]
fn load_refuses_a_layer_type_that_no_longer_matches() {
    let tmp = TempFile::new("norm_type_mismatch");

    let saved = SequentialBuilder::new()
        .add(InstanceNormalization::new(1e-5).unwrap())
        .build(&Shape::known(&[2, 4, 3]))
        .unwrap();
    saved.save_to_path(tmp.path()).unwrap();

    // The same 2 arrays, under the same 2 names, at the same extent
    let mut target = SequentialBuilder::new()
        .add(GroupNormalization::new(3, 1e-5).unwrap())
        .build(&Shape::known(&[2, 4, 3]))
        .unwrap();
    assert_eq!(target.weight_paths(), vec!["0.gamma", "0.beta"]);

    match target.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::ModelStructureMismatch(message))) => {
            assert!(
                message.contains("layer 0")
                    && message.contains("GroupNormalization")
                    && message.contains("InstanceNormalization"),
                "the refusal must name the position and both types, got {message:?}"
            );
        }
        other => panic!("expected ModelStructureMismatch, got {other:?}"),
    }
}

/// Weight-shape mismatch (Dense 2->2 saved, Dense 3->3 target) gives
/// Error::Io(IoError::ModelStructureMismatch)
#[test]
fn load_weight_shape_mismatch_gives_structure_error() {
    let tmp = TempFile::new("shape_mismatch");

    let model_small = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();
    model_small.save_to_path(tmp.path()).unwrap();

    let mut model_big = SequentialBuilder::new()
        .add(Dense::new(3, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    let result = model_big.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(message))) => {
            assert!(
                message.contains("`0.kernel`"),
                "the refusal must name the parameter path, got {message:?}"
            );
        }
        other => panic!("expected ModelStructureMismatch, got {:?}", other),
    }
}

/// A file whose magic tag does not match gives Error::Io(IoError::UnsupportedModelFormat). This
/// catches a pre-header file, which began with its layer count where the tag is now expected.
#[test]
fn load_wrong_magic_gives_unsupported_format_error() {
    let tmp = TempFile::new("wrong_magic");

    let headerless = ModelCheckpoint {
        magic: 1,
        format_version: MODEL_FORMAT_VERSION,
        layers: Vec::new(),
    };
    std::fs::write(tmp.path(), postcard::to_allocvec(&headerless).unwrap()).unwrap();

    let mut model = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    match model.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::UnsupportedModelFormat(_))) => {}
        other => panic!("expected UnsupportedModelFormat, got {:?}", other),
    }
}

/// A file with the right magic but the wrong format version gives
/// Error::Io(IoError::UnsupportedModelFormat), even when every layer and weight shape matches.
#[test]
fn load_wrong_format_version_gives_unsupported_format_error() {
    let tmp = TempFile::new("wrong_version");

    let saved = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();
    saved.save_to_path(tmp.path()).unwrap();

    // Replace only the header, keeping the body byte for byte
    let bytes = std::fs::read(tmp.path()).unwrap();
    let (_, body) = postcard::take_from_bytes::<(u32, u32)>(&bytes).unwrap();
    let mut bumped = postcard::to_allocvec(&(MODEL_MAGIC, MODEL_FORMAT_VERSION + 1)).unwrap();
    bumped.extend_from_slice(body);
    std::fs::write(tmp.path(), &bumped).unwrap();

    let mut model = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    match model.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::UnsupportedModelFormat(_))) => {}
        other => panic!("expected UnsupportedModelFormat, got {:?}", other),
    }
}

// The named checkpoint

/// A file of the format version before this one is refused, and the refusal names both numbers
///
/// Version 2 replaced the closed weight enum with the named checkpoint, so no byte of a
/// version 1 file means what this build reads. The number in the header is what says so
#[test]
fn load_older_format_version_names_the_version_it_found_and_the_one_it_wants() {
    let tmp = TempFile::new("older_version");

    let saved = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[2, 2]))
        .unwrap();
    saved.save_to_path(tmp.path()).unwrap();

    // Keep the body, and put the version of the format before this one in the header
    let bytes = std::fs::read(tmp.path()).unwrap();
    let (_, body) = postcard::take_from_bytes::<(u32, u32)>(&bytes).unwrap();
    let mut older = postcard::to_allocvec(&(MODEL_MAGIC, MODEL_FORMAT_VERSION - 1)).unwrap();
    older.extend_from_slice(body);
    std::fs::write(tmp.path(), &older).unwrap();

    let mut model = SequentialBuilder::new()
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&Shape::known(&[2, 2]))
        .unwrap();

    match model.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::UnsupportedModelFormat(message))) => {
            assert!(
                message.contains(&format!("version {}", MODEL_FORMAT_VERSION - 1))
                    && message.contains(&format!("version {MODEL_FORMAT_VERSION}")),
                "the refusal must name the version it found and the version it wants, got \
                 {message:?}"
            );
        }
        other => panic!("expected UnsupportedModelFormat, got {other:?}"),
    }
}

/// A model whose layers repeat a type round trips, and no position takes another position's
/// values
///
/// The 3 Dense layers here share every shape, so nothing but the position half of a path tells
/// their arrays apart. Each one carries its own injected values, and an exchange of any 2 moves
/// both
#[test]
fn repeated_layer_type_round_trips_by_position() {
    let tmp = TempFile::new("repeated_type");

    // Every layer holds the same shapes, and its own values
    let injected = |base: f32| {
        let mut layer = Dense::new(2, Linear::new()).unwrap();
        layer.build(&Shape::known(&[1, 2])).unwrap();
        layer
            .set_weights(
                Array::from_shape_vec((2, 2), vec![base, base + 1.0, base + 2.0, base + 3.0])
                    .unwrap(),
                Array::from_shape_vec((1, 2), vec![base + 4.0, base + 5.0]).unwrap(),
            )
            .unwrap();
        layer
    };
    let make_arch = || {
        SequentialBuilder::new()
            .add(Dense::new(2, Linear::new()).unwrap())
            .add(Dense::new(2, Linear::new()).unwrap())
            .add(Dense::new(2, Linear::new()).unwrap())
            .build(&Shape::known(&[1, 2]))
            .unwrap()
    };

    let model = SequentialBuilder::new()
        .add(injected(1.0))
        .add(injected(10.0))
        .add(injected(100.0))
        .build(&Shape::known(&[1, 2]))
        .unwrap();
    assert_eq!(
        model.weight_paths(),
        vec![
            "0.kernel", "0.bias", "1.kernel", "1.bias", "2.kernel", "2.bias"
        ]
    );

    let x: Tensor = Array::from_shape_vec((1, 2), vec![0.5f32, -0.25])
        .unwrap()
        .into_dyn();
    let before = model.predict(&x).unwrap();
    model.save_to_path(tmp.path()).unwrap();

    let mut fresh = make_arch();
    fresh.load_from_path(tmp.path()).unwrap();
    assert_allclose(&fresh.predict(&x).unwrap(), &before, 0.0_f32);
    for path in model.weight_paths() {
        assert_eq!(
            fresh.weight(&path).unwrap(),
            model.weight(&path).unwrap(),
            "the path {path} did not come back"
        );
    }
}

/// The lenient load applies what matches, and reports the missing and the unused paths
///
/// The file holds 4 layers and the model holds 3. Layer 0 agrees, layer 1 disagrees on every
/// extent, layer 2 holds another layer type, and layer 3 of the file reaches no layer at all
#[test]
fn load_partial_reports_applied_missing_and_unused() {
    let tmp = TempFile::new("partial");

    let saved = SequentialBuilder::new()
        .add(Dense::new(3, Linear::new()).unwrap())
        .add(Dense::new(4, Linear::new()).unwrap())
        .add(InstanceNormalization::new(1e-5).unwrap())
        .add(Dense::new(1, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();
    saved.save_to_path(tmp.path()).unwrap();

    let mut model = SequentialBuilder::new()
        .add(Dense::new(3, Linear::new()).unwrap())
        .add(Dense::new(5, Linear::new()).unwrap())
        .add(GroupNormalization::new(5, 1e-5).unwrap())
        .build(&Shape::known(&[1, 2]))
        .unwrap();

    let report = model.load_partial_from_path(tmp.path()).unwrap();
    assert_eq!(report.applied, vec!["0.kernel", "0.bias"]);
    assert_eq!(
        report.missing,
        vec!["1.kernel", "1.bias", "2.gamma", "2.beta"]
    );
    // Layer 2 of the file names the same 2 arrays at the same extent, and its layer type
    // differs, so neither side reaches the other
    assert_eq!(
        report.unused,
        vec![
            "1.kernel", "1.bias", "2.gamma", "2.beta", "3.kernel", "3.bias"
        ]
    );

    // What the report calls applied really reached the model
    assert_eq!(
        model.weight("0.kernel").unwrap(),
        saved.weight("0.kernel").unwrap()
    );
    // What it calls missing did not, so layer 1 keeps the shape it was built with
    assert_eq!(model.weight("1.kernel").unwrap().shape(), &[3, 5]);

    // The strict load still refuses the same file
    assert!(matches!(
        model.load_from_path(tmp.path()),
        Err(Error::Io(IoError::ModelStructureMismatch(_)))
    ));
}

// The atomic refusal

/// Every array of 1 layer of a model, as a path and the raw f32 bits of its elements
///
/// A load either writes an array or leaves it alone, and nothing between the 2. The comparison
/// is therefore a bit comparison, and no epsilon takes part in it
fn layer_bits(model: &Sequential, scope: usize) -> Vec<(String, Vec<u32>)> {
    let prefix = format!("{scope}.");
    model
        .weight_paths()
        .into_iter()
        .filter(|path| path.starts_with(&prefix))
        .map(|path| {
            let bits = model
                .weight(&path)
                .unwrap_or_else(|| panic!("the model holds no array at `{path}`"))
                .iter()
                .map(|value| value.to_bits())
                .collect();
            (path, bits)
        })
        .collect()
}

/// A refused load leaves every layer before the disagreement exactly as it was
///
/// The file holds 2 Dense layers. Position 0 agrees with the model on the layer type, on the
/// names, on the kinds, and on the shapes. It carries other values there. Position 1 has 4
/// units where the model has 5, so both of its arrays disagree on shape.
///
/// The validate pass reads the model and writes nothing, and the write pass runs only after
/// it. The refusal at position 1 must therefore leave position 0 untouched. A load that wrote
/// position 0 on its way to the refusal would leave the model holding 1 layer of the file next
/// to 1 layer of its own, and no error would say so
#[test]
fn a_refused_load_writes_no_array_of_the_layers_before_the_mismatch() {
    let tmp = TempFile::new("atomic_refusal_across_layers");

    let mut saved_head = Dense::new(2, Linear::new()).unwrap();
    saved_head.build(&Shape::known(&[1, 3])).unwrap();
    saved_head
        .set_weights(
            Array::from_shape_vec((3, 2), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Array::from_shape_vec((1, 2), vec![7.0f32, 8.0]).unwrap(),
        )
        .unwrap();
    let saved = SequentialBuilder::new()
        .add(saved_head)
        .add(Dense::new(4, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 3]))
        .unwrap();
    saved.save_to_path(tmp.path()).unwrap();

    let mut head = Dense::new(2, Linear::new()).unwrap();
    head.build(&Shape::known(&[1, 3])).unwrap();
    head.set_weights(
        Array::from_shape_vec((3, 2), vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0]).unwrap(),
        Array::from_shape_vec((1, 2), vec![-7.0f32, -8.0]).unwrap(),
    )
    .unwrap();
    let mut model = SequentialBuilder::new()
        .add(head)
        .add(Dense::new(5, Linear::new()).unwrap())
        .build(&Shape::known(&[1, 3]))
        .unwrap();

    // The 2 sides hold other values at position 0, so a write there really shows
    let before = layer_bits(&model, 0);
    assert_eq!(
        before
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>(),
        vec!["0.kernel".to_string(), "0.bias".to_string()]
    );
    assert_ne!(
        before,
        layer_bits(&saved, 0),
        "the file must carry other values at position 0, or the test proves nothing"
    );

    match model.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::ModelStructureMismatch(message))) => {
            assert!(
                message.contains("`1.kernel`"),
                "the refusal must name the array that disagrees, got {message:?}"
            );
        }
        other => panic!("expected ModelStructureMismatch, got {other:?}"),
    }

    assert_eq!(
        layer_bits(&model, 0),
        before,
        "the refusal wrote layer 0, so a load is not atomic"
    );
}

/// A refused load writes no earlier array of the layer that disagrees either
///
/// The file holds 1 Dense layer. Its `kernel` record agrees on the name, on the kind, and on
/// the shape, and it carries other values. Its `bias` record carries a shape that the layer
/// does not hold. The validate pass reaches the last array of the last layer before any write
/// starts, so the kernel keeps its value
#[test]
fn a_refused_load_writes_no_earlier_array_of_the_layer_that_disagrees() {
    let tmp = TempFile::new("atomic_refusal_within_a_layer");

    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 3])).unwrap();
    layer
        .set_weights(
            Array::from_shape_vec((3, 2), vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0]).unwrap(),
            Array::from_shape_vec((1, 2), vec![-7.0f32, -8.0]).unwrap(),
        )
        .unwrap();
    let mut model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(&[1, 3]))
        .unwrap();

    let file = ModelCheckpoint {
        magic: MODEL_MAGIC,
        format_version: MODEL_FORMAT_VERSION,
        layers: vec![LayerCheckpoint {
            layer_type: Cow::Borrowed("Dense"),
            build: None,
            weights: vec![
                // Agrees on every check, and holds other values
                WeightRecord {
                    name: Cow::Borrowed("kernel"),
                    kind: WeightKind::Trainable,
                    shape: vec![3, 2],
                    data: Cow::Owned(vec![9.0f32; 6]),
                },
                // The layer holds a bias of shape [1, 2], and this record is 1 element longer
                WeightRecord {
                    name: Cow::Borrowed("bias"),
                    kind: WeightKind::Trainable,
                    shape: vec![1, 3],
                    data: Cow::Owned(vec![9.0f32; 3]),
                },
            ],
        }],
    };
    std::fs::write(tmp.path(), postcard::to_allocvec(&file).unwrap()).unwrap();

    let before = layer_bits(&model, 0);
    match model.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::ModelStructureMismatch(message))) => {
            assert!(
                message.contains("`0.bias`"),
                "the refusal must name the array that disagrees, got {message:?}"
            );
        }
        other => panic!("expected ModelStructureMismatch, got {other:?}"),
    }

    assert_eq!(
        layer_bits(&model, 0),
        before,
        "the refusal wrote the kernel before it read the bias, so a load is not atomic"
    );
}

/// A saved model reloads through the build path: construct, build, load
///
/// The layers of the target hold their arrays because the build allocated them, and the file
/// carries the shape each layer was built for. The load compares that shape, so the round trip
/// pins the build step as well as the values
#[test]
fn a_saved_model_reloads_through_the_build_path() {
    let tmp = TempFile::new("build_path_round_trip");
    let shape = Shape::known(&[2, 3]);

    let mut source = SequentialBuilder::new()
        .add(Dense::new(4, Linear::new()).unwrap().with_random_state(19))
        .add(Dense::new(2, Linear::new()).unwrap().with_random_state(23))
        .build(&shape)
        .unwrap();
    source.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let x: Tensor = Array::from_shape_vec((2, 3), vec![0.5f32, -1.0, 1.5, -0.5, 1.0, -1.5])
        .unwrap()
        .into_dyn();
    let y: Tensor = Array::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();
    source.fit(&x, &y, 3).unwrap();
    let expected = source.predict(&x).unwrap();
    source.save_to_path(tmp.path()).unwrap();

    // The target is a fresh stack of the same architecture. It holds every array only because
    // the build allocated them
    let mut target = SequentialBuilder::new()
        .add(Dense::new(4, Linear::new()).unwrap())
        .add(Dense::new(2, Linear::new()).unwrap())
        .build(&shape)
        .unwrap();
    target.load_from_path(tmp.path()).unwrap();

    assert_allclose(&target.predict(&x).unwrap(), &expected, 0.0_f32);
}

/// A file written by a model built for another input shape does not load
///
/// The 2 models hold arrays of the same names and the same extents, because a convolution
/// kernel reads the channel count alone and never a spatial extent. The build shape in the
/// file is what tells them apart
#[test]
fn a_checkpoint_of_another_build_shape_is_refused() {
    let tmp = TempFile::new("build_shape_mismatch");

    let source = SequentialBuilder::new()
        .add(Conv2D::new(3, (2, 2), (1, 1), Linear::new()).unwrap())
        .build(&Shape::known(&[1, 4, 4, 2]))
        .unwrap();
    source.save_to_path(tmp.path()).unwrap();

    // The same kernel extents, over a larger image
    let mut target = SequentialBuilder::new()
        .add(Conv2D::new(3, (2, 2), (1, 1), Linear::new()).unwrap())
        .build(&Shape::known(&[1, 6, 6, 2]))
        .unwrap();
    assert_eq!(target.weight("0.kernel").unwrap().shape(), &[2, 2, 2, 3]);

    match target.load_from_path(tmp.path()) {
        Err(Error::Io(IoError::ModelStructureMismatch(message))) => {
            assert!(message.contains("layer 0"), "{message}");
            assert!(message.contains("built for input shape"), "{message}");
        }
        other => panic!("expected ModelStructureMismatch, got {other:?}"),
    }

    // The lenient path skips the layer instead of failing, and reports every path of both sides
    let report = target.load_partial_from_path(tmp.path()).unwrap();
    assert!(report.applied.is_empty(), "{report:?}");
    assert_eq!(report.missing, vec!["0.kernel", "0.bias"]);
    assert_eq!(report.unused, vec!["0.kernel", "0.bias"]);
}

/// The batch extent is not part of a build shape, so a checkpoint moves between batch sizes
#[test]
fn a_checkpoint_moves_between_2_batch_sizes() {
    let tmp = TempFile::new("build_shape_free_batch");

    let source = SequentialBuilder::new()
        .add(Conv2D::new(3, (2, 2), (1, 1), Linear::new()).unwrap())
        .build(&Shape::known(&[1, 4, 4, 2]))
        .unwrap();
    source.save_to_path(tmp.path()).unwrap();

    let mut target = SequentialBuilder::new()
        .add(Conv2D::new(3, (2, 2), (1, 1), Linear::new()).unwrap())
        .build(&Shape::known(&[8, 4, 4, 2]))
        .unwrap();
    target.load_from_path(tmp.path()).unwrap();

    assert_allclose(
        &target.weight("0.kernel").unwrap(),
        &source.weight("0.kernel").unwrap(),
        0.0_f32,
    );
}
