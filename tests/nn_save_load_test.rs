#![cfg(feature = "neural_network")]

//! Integration tests for `Sequential::save_to_path` / `load_from_path`.
//!
//! These exercise the full save/load round-trip (architecture + weights) and the structural
//! validation performed on load. The `Dense<Linear>` round-trip in particular guards against a
//! regression where Linear-activated parametric layers could not be reloaded at all.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array4};
use rustyml::error::{Error, IoError};
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

/// Builds a unique temp file path so concurrent tests do not collide.
fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustyml_nn_saveload_{}.json", name))
}

fn assert_tensors_close(a: &rustyml::neural_network::Tensor, b: &rustyml::neural_network::Tensor) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch after round-trip");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_abs_diff_eq!(x, y, epsilon = 1e-6);
    }
}

#[test]
fn dense_linear_round_trip_preserves_predictions() {
    // Regression test: a model whose output layer uses `Linear` previously failed to load with
    // "Expected Dense layer but got Dense" because the downcast cascade omitted Linear.
    let x = Array::ones((4, 6)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(6, 8, ReLU::new()).unwrap())
        .add(Dense::new(8, 3, Linear::new()).unwrap());
    let expected = model.predict(&x).unwrap();

    let path = temp_path("dense_linear");
    model.save_to_path(&path).unwrap();

    let mut loaded = Sequential::new();
    loaded
        .add(Dense::new(6, 8, ReLU::new()).unwrap())
        .add(Dense::new(8, 3, Linear::new()).unwrap());
    loaded.load_from_path(&path).unwrap();

    let got = loaded.predict(&x).unwrap();
    assert_tensors_close(&expected, &got);

    std::fs::remove_file(&path).ok();
}

#[test]
fn conv2d_round_trip_preserves_trained_weights() {
    let x = Array4::ones((2, 1, 5, 5)).into_dyn();
    let y = Array4::ones((2, 3, 3, 3)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            Conv2D::new(
                3,
                (3, 3),
                vec![2, 1, 5, 5],
                (1, 1),
                PaddingType::Valid,
                ReLU::new(),
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
    model.fit(&x, &y, 3).unwrap();
    let expected = model.predict(&x).unwrap();

    let path = temp_path("conv2d");
    model.save_to_path(&path).unwrap();

    let mut loaded = Sequential::new();
    loaded.add(
        Conv2D::new(
            3,
            (3, 3),
            vec![2, 1, 5, 5],
            (1, 1),
            PaddingType::Valid,
            ReLU::new(),
        )
        .unwrap(),
    );
    loaded.load_from_path(&path).unwrap();

    let got = loaded.predict(&x).unwrap();
    assert_tensors_close(&expected, &got);

    std::fs::remove_file(&path).ok();
}

#[test]
fn batch_normalization_round_trip_preserves_running_stats() {
    let x = Array::ones((5, 4)).into_dyn();
    let y = Array::ones((5, 6)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 6, ReLU::new()).unwrap())
        .add(BatchNormalization::new(vec![5, 6], 0.9, 1e-5).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
    model.fit(&x, &y, 3).unwrap();
    let expected = model.predict(&x).unwrap();

    let path = temp_path("batch_norm");
    model.save_to_path(&path).unwrap();

    let mut loaded = Sequential::new();
    loaded
        .add(Dense::new(4, 6, ReLU::new()).unwrap())
        .add(BatchNormalization::new(vec![5, 6], 0.9, 1e-5).unwrap());
    loaded.load_from_path(&path).unwrap();

    let got = loaded.predict(&x).unwrap();
    assert_tensors_close(&expected, &got);

    std::fs::remove_file(&path).ok();
}

#[test]
fn load_layer_count_mismatch_is_reported() {
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3, ReLU::new()).unwrap());
    let path = temp_path("count_mismatch");
    model.save_to_path(&path).unwrap();

    let mut loaded = Sequential::new();
    loaded
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 2, Linear::new()).unwrap());
    let err = loaded.load_from_path(&path).unwrap_err();
    assert!(
        matches!(err, Error::Io(IoError::ModelStructureMismatch(_))),
        "expected ModelStructureMismatch, got {:?}",
        err
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn load_layer_type_mismatch_is_reported() {
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3, ReLU::new()).unwrap());
    let path = temp_path("type_mismatch");
    model.save_to_path(&path).unwrap();

    // A ReLU layer at position 0 where a Dense was saved must be rejected.
    let mut loaded = Sequential::new();
    loaded.add(ReLU::new());
    let err = loaded.load_from_path(&path).unwrap_err();
    assert!(
        matches!(err, Error::Io(IoError::ModelStructureMismatch(_))),
        "expected ModelStructureMismatch, got {:?}",
        err
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn load_weight_shape_mismatch_is_reported() {
    // Same layer type, different output dimension: must be caught by shape validation rather than
    // silently corrupting the layer and panicking later inside `dot`.
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3, Linear::new()).unwrap());
    let path = temp_path("shape_mismatch");
    model.save_to_path(&path).unwrap();

    let mut loaded = Sequential::new();
    loaded.add(Dense::new(4, 5, Linear::new()).unwrap());
    let err = loaded.load_from_path(&path).unwrap_err();
    assert!(
        matches!(err, Error::Io(IoError::ModelStructureMismatch(_))),
        "expected ModelStructureMismatch, got {:?}",
        err
    );

    std::fs::remove_file(&path).ok();
}
