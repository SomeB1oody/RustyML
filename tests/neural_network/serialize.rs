//! Integration tests for Sequential model save/load round-trips
//!
//! Each test builds a model (with known injected weights or trained a few steps),
//! saves and reloads it into a fresh model of the same architecture, and asserts
//! that predict output matches element-wise (within ~1e-6) across the round-trip
//! Error paths cover layer-count, layer-type, and weight-shape mismatches (all
//! ModelStructureMismatch), a nonexistent file (IoError::Std), and corrupt binary
//! data (IoError::Serialization)

use crate::common::assert_allclose;
use ndarray::Array;
use rustyml::error::{Error, IoError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
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
use std::env;

// Helpers

/// Temporary file that deletes itself when dropped
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(name: &str) -> Self {
        let path = env::temp_dir().join(format!("rustyml_serialize_test_{}.json", name));
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

    let mut layer = Dense::new(2, 2, Linear::new()).unwrap();
    let w = Array::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let b = Array::zeros((1, 2));
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(layer);

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
            let mut m = Sequential::new();
            m.add(Dense::new(2, 2, Linear::new()).unwrap());
            m
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

    let mut layer = Dense::new(2, 2, Linear::new()).unwrap();
    let w = Array::from_shape_vec((2, 2), vec![2.0f32, 0.0, 0.0, 2.0]).unwrap();
    let b = Array::from_shape_vec((1, 2), vec![1.0f32, 1.0]).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(layer);

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
            let mut m = Sequential::new();
            m.add(Dense::new(2, 2, Linear::new()).unwrap());
            m
        },
        tmp.path(),
    );
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &expected, 1e-6_f32);
}

// Dense with zero weights (W = 0): output is the bias broadcast over the batch, and survives a round-trip
#[test]
fn dense_zero_weights_bias_only_value_check_and_round_trip() {
    let tmp = TempFile::new("dense_zero_w");

    let mut layer = Dense::new(2, 3, Linear::new()).unwrap();
    let w = Array::zeros((2, 3));
    let b = Array::from_shape_vec((1, 3), vec![0.5f32, -0.5, 1.0]).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(layer);

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
            let mut m = Sequential::new();
            m.add(Dense::new(2, 3, Linear::new()).unwrap());
            m
        },
        tmp.path(),
    );
    let after = fresh.predict(&x).unwrap();
    assert_allclose(&after, &expected, 1e-6_f32);
}

// Two-layer Dense model trained a few steps round-trips
#[test]
fn dense_two_layer_trained_round_trip() {
    let tmp = TempFile::new("dense2");

    let make_arch = || {
        let mut m = Sequential::new();
        m.add(Dense::new(4, 3, Linear::new()).unwrap())
            .add(Dense::new(3, 2, Linear::new()).unwrap());
        m
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
        let mut m = Sequential::new();
        m.add(Conv1D::new(2, 2, vec![1, 1, 5], 1, Linear::new()).unwrap());
        m
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec((1, 1, 5), vec![0.1f32, 0.3, -0.2, 0.5, -0.4])
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
        let mut m = Sequential::new();
        m.add(Conv2D::new(2, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap());
        m
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 1, 4, 4),
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
        let mut m = Sequential::new();
        m.add(Conv3D::new(2, (2, 2, 2), vec![1, 1, 3, 3, 3], (1, 1, 1), Linear::new()).unwrap());
        m
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 1, 3, 3, 3),
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
        let mut m = Sequential::new();
        m.add(DepthwiseConv2D::new(2, (2, 2), vec![1, 2, 4, 4], (1, 1), Linear::new()).unwrap());
        m
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect::<Vec<_>>(),
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
        let mut m = Sequential::new();
        m.add(SeparableConv2D::new(2, (2, 2), vec![1, 2, 4, 4], (1, 1), 1, Linear::new()).unwrap());
        m
    };

    let model = make_arch();

    let x: Tensor = Array::from_shape_vec(
        (1, 2, 4, 4),
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
        let mut m = Sequential::new();
        m.add(SimpleRNN::new(2, 3, Tanh::new()).unwrap());
        m
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
        let mut m = Sequential::new();
        m.add(LSTM::new(2, 3, Tanh::new()).unwrap());
        m
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
        let mut m = Sequential::new();
        m.add(GRU::new(2, 3, Tanh::new()).unwrap());
        m
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

// BatchNormalization round-trip: trained running_mean/running_var must survive serialization,
// so eval-mode predict returns the identical tensor afterward
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
        let mut m = Sequential::new();
        m.add(BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap());
        m
    };

    // Train a copy to move running stats away from their defaults
    let mut trainable_model = make_arch();
    trainable_model.compile(
        SGD::new(0.001, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    trainable_model.fit(&x_train, &x_train, 8).unwrap();

    // Eval-mode prediction uses running_mean / running_var
    let before = trainable_model.predict(&x_train).unwrap();

    // Save and restore into a fresh (untrained) model
    trainable_model.save_to_path(tmp.path()).unwrap();
    let mut fresh = make_arch();
    fresh.load_from_path(tmp.path()).unwrap();

    let after = fresh.predict(&x_train).unwrap();
    assert_allclose(&after, &before, 1e-6_f32);
}

/// predict() is deterministic: two calls on the same fresh eval-mode model return the same tensor
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
        let mut m = Sequential::new();
        m.add(BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap());
        m
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

    // Two calls to predict must agree element-wise
    let p1 = fresh.predict(&x).unwrap();
    let p2 = fresh.predict(&x).unwrap();
    assert_allclose(&p2, &p1, 1e-7_f32);
}

// LayerNormalization round-trip
#[test]
fn layer_normalization_round_trip() {
    let tmp = TempFile::new("layer_norm");

    let make_arch = || {
        let mut m = Sequential::new();
        m.add(LayerNormalization::new(vec![2, 4], 1e-5).unwrap());
        m
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
        let mut m = Sequential::new();
        m.add(GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap());
        m
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
        let mut m = Sequential::new();
        m.add(InstanceNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap());
        m
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

// Mixed model Dense -> Dropout -> Dense: round-trips a parameterless layer (Dropout is
// transparent in eval mode); empty input_shape skips shape validation, so batch size is not baked in
#[test]
fn mixed_model_with_dropout_round_trip() {
    let tmp = TempFile::new("mixed_dropout");

    let make_arch = || {
        let mut m = Sequential::new();
        m.add(Dense::new(3, 4, Linear::new()).unwrap())
            // empty input_shape => Dropout's shape validator is skipped at runtime
            .add(Dropout::new(0.3, vec![]).unwrap())
            .add(Dense::new(4, 2, Linear::new()).unwrap());
        m
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
        let mut m = Sequential::new();
        m.add(Dense::new(3, 4, Linear::new()).unwrap())
            .add(Dropout::new(0.3, vec![]).unwrap())
            .add(Dense::new(4, 2, Linear::new()).unwrap());
        m
    };

    let mut model = make_arch();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    // Consistent batch size; Dropout shape validation is off (empty vec)
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
    let mut model = Sequential::new();
    model.add(Dense::new(2, 2, Linear::new()).unwrap());

    let result =
        model.load_from_path("/tmp/this_file_definitely_does_not_exist_rustyml_99999.json");
    match result {
        Err(Error::Io(IoError::Std(_))) => {}
        other => panic!("expected IoError::Std, got {:?}", other),
    }
}

/// Corrupt binary data gives Error::Io(IoError::Serialization)
#[test]
fn load_from_invalid_data_gives_serialization_error() {
    let tmp = TempFile::new("invalid_data");
    std::fs::write(tmp.path(), b"\xff\xff\xff not valid postcard data").unwrap();

    let mut model = Sequential::new();
    model.add(Dense::new(2, 2, Linear::new()).unwrap());

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
    let mut model_1 = Sequential::new();
    model_1.add(Dense::new(2, 2, Linear::new()).unwrap());
    model_1.save_to_path(tmp.path()).unwrap();

    // Try to load into a 2-layer model
    let mut model_2 = Sequential::new();
    model_2
        .add(Dense::new(2, 2, Linear::new()).unwrap())
        .add(Dense::new(2, 2, Linear::new()).unwrap());

    let result = model_2.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(_))) => {}
        other => panic!("expected ModelStructureMismatch, got {:?}", other),
    }
}

/// Layer-type mismatch (Dense saved, Conv1D target) gives Error::Io(IoError::ModelStructureMismatch)
#[test]
fn load_layer_type_mismatch_gives_structure_error() {
    let tmp = TempFile::new("type_mismatch");

    let mut dense_model = Sequential::new();
    dense_model.add(Dense::new(3, 3, Linear::new()).unwrap());
    dense_model.save_to_path(tmp.path()).unwrap();

    let mut conv_model = Sequential::new();
    conv_model.add(Conv1D::new(2, 2, vec![1, 1, 5], 1, Linear::new()).unwrap());

    let result = conv_model.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(_))) => {}
        other => panic!("expected ModelStructureMismatch, got {:?}", other),
    }
}

/// Weight-shape mismatch (Dense 2->2 saved, Dense 3->3 target) gives Error::Io(IoError::ModelStructureMismatch)
#[test]
fn load_weight_shape_mismatch_gives_structure_error() {
    let tmp = TempFile::new("shape_mismatch");

    let mut model_small = Sequential::new();
    model_small.add(Dense::new(2, 2, Linear::new()).unwrap());
    model_small.save_to_path(tmp.path()).unwrap();

    let mut model_big = Sequential::new();
    model_big.add(Dense::new(3, 3, Linear::new()).unwrap());

    let result = model_big.load_from_path(tmp.path());
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(_))) => {}
        other => panic!("expected ModelStructureMismatch, got {:?}", other),
    }
}
