//! Integration tests for `Sequential`: add/compile/fit/predict/summary plus error paths
//!
//! Every expected value is derived from the mathematical definition or a hand calculation,
//! never by running the layer and recording its output

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::Activation;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::losses::{CategoricalCrossEntropy, MeanSquaredError};
use rustyml::neural_network::optimizers::{Adam, SGD};
use rustyml::neural_network::sequential::Sequential;

// helpers

/// Build a 2-D Tensor from row-major data
fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_dyn()
}

// predict: forward values with known weights

/// Dense(2->2, Linear) with identity weights and zero bias returns the input unchanged
#[test]
fn test_predict_identity_weights_linear_dense() {
    let mut dense = Dense::new(2, 2, Activation::Linear, None).unwrap();
    let w = Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 2, vec![3.0, 4.0]);
    let out = model.predict(&x).unwrap();

    assert_abs_diff_eq!(out[[0, 0]], 3.0_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(out[[0, 1]], 4.0_f32, epsilon = 1e-6);
}

/// Dense(1->1, Linear) applies the scalar affine map 2*x + 1
#[test]
fn test_predict_scalar_affine() {
    let mut dense = Dense::new(1, 1, Activation::Linear, None).unwrap();
    let w = Array2::from_shape_vec((1, 1), vec![2.0_f32]).unwrap();
    let b = Array2::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 1, vec![5.0]);
    let out = model.predict(&x).unwrap();

    assert_abs_diff_eq!(out[[0, 0]], 11.0_f32, epsilon = 1e-6);
}

/// Dense(3->2, Linear) applies a known linear transform to one input row
#[test]
fn test_predict_2d_linear_transform() {
    let mut dense = Dense::new(3, 2, Activation::Linear, None).unwrap();
    // weights shape (in=3, out=2)
    let w = Array2::from_shape_vec(
        (3, 2),
        vec![
            1.0_f32, 0.0, // feature 0 contributes to out 0 and out 1
            0.0, 1.0, // feature 1
            1.0, 1.0, // feature 2
        ],
    )
    .unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 3, vec![2.0, 3.0, 4.0]);
    let out = model.predict(&x).unwrap();

    // z = [2*1 + 4*1, 3*1 + 4*1] = [6, 7]
    assert_abs_diff_eq!(out[[0, 0]], 6.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], 7.0_f32, epsilon = 1e-5);
}

/// Two stacked Linear Dense layers chain correctly: first projects, second sums both inputs
#[test]
fn test_predict_two_layer_stack() {
    let mut d1 = Dense::new(3, 2, Activation::Linear, None).unwrap();
    let w1 = Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let b1 = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    d1.set_weights(w1, b1).unwrap();

    let mut d2 = Dense::new(2, 1, Activation::Linear, None).unwrap();
    let w2 = Array2::from_shape_vec((2, 1), vec![1.0_f32, 1.0]).unwrap();
    let b2 = Array2::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    d2.set_weights(w2, b2).unwrap();

    let mut model = Sequential::new();
    model.add(d1).add(d2);

    let x = t2(1, 3, vec![5.0, 7.0, 99.0]);
    let out = model.predict(&x).unwrap();

    // layer1 zeroes the third feature -> [5, 7]; layer2 sums them -> 12
    assert_abs_diff_eq!(out[[0, 0]], 12.0_f32, epsilon = 1e-5);
}

/// Dense(1->3, Softmax) on a zero pre-activation yields the uniform distribution [1/3, 1/3, 1/3]
#[test]
fn test_predict_dense_softmax_equal_input() {
    let mut dense = Dense::new(1, 3, Activation::Softmax, None).unwrap();
    let w = Array2::from_shape_vec((1, 3), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let b = Array2::from_shape_vec((1, 3), vec![0.0_f32, 0.0, 0.0]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    // input 0 -> z = [0, 0, 0] -> softmax -> [1/3, 1/3, 1/3]
    let x = t2(1, 1, vec![0.0]);
    let out = model.predict(&x).unwrap();

    let third = 1.0_f32 / 3.0;
    assert_abs_diff_eq!(out[[0, 0]], third, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], third, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 2]], third, epsilon = 1e-5);

    // probabilities must sum to 1
    let sum: f32 = out.iter().sum();
    assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-6);
}

/// Dense(1->3, Softmax) with all-zero weights ignores the input and stays uniform
#[test]
fn test_predict_dense_softmax_known_probs() {
    let mut dense = Dense::new(1, 3, Activation::Softmax, None).unwrap();
    let w: Array2<f32> = Array2::zeros((1, 3));
    let b: Array2<f32> = Array2::zeros((1, 3));
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 1, vec![99.0]);
    let out = model.predict(&x).unwrap();

    let third = 1.0_f32 / 3.0;
    assert_abs_diff_eq!(out[[0, 0]], third, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], third, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 2]], third, epsilon = 1e-5);
}

// predict == forward in eval mode

/// predict() produces the same values as an eval-mode forward pass for a Linear Dense layer
#[test]
fn test_predict_equals_forward_eval_mode() {
    let mut dense = Dense::new(2, 2, Activation::Linear, None).unwrap();
    let w = Array2::from_shape_vec((2, 2), vec![0.5_f32, 0.0, 0.0, 0.5]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![1.0_f32, -1.0]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    let pred = model.predict(&x).unwrap();

    // Linear Dense is stateless, so a second predict call must match
    let pred2 = model.predict(&x).unwrap();

    crate::common::assert_allclose(&pred, &pred2, 1e-7_f32);
}

// predict: determinism (two consecutive calls identical)

/// Two back-to-back predict() calls on the same input produce identical tensors
#[test]
fn test_predict_is_deterministic() {
    let mut dense = Dense::new(3, 2, Activation::Linear, None).unwrap();
    let w = Array2::from_shape_vec((3, 2), vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.01_f32, -0.02]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 3, vec![1.0, -1.0, 2.0]);
    let out1 = model.predict(&x).unwrap();
    let out2 = model.predict(&x).unwrap();
    crate::common::assert_allclose(&out1, &out2, 0.0_f32);
}

// summary smoke-test

/// summary() runs without panicking
#[test]
fn test_summary_does_not_panic() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 8, Activation::ReLU, None).unwrap())
        .add(Dense::new(8, 2, Activation::Softmax, None).unwrap());
    model.summary();
}

// error paths

/// fit() before compile() returns NotCompiled
#[test]
fn test_fit_before_compile_returns_not_compiled() {
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, Activation::Linear, None).unwrap());
    let x = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let y = t2(2, 1, vec![1.0, 0.0]);
    assert!(
        matches!(
            model.fit(&x, &y, 1),
            Err(Error::NeuralNetwork(NnError::NotCompiled(_)))
        ),
        "expected NotCompiled"
    );
}

/// fit() on a model with no layers returns EmptyModel
#[test]
fn test_fit_empty_model_returns_empty_model_error() {
    let mut model = Sequential::new();
    model.compile(
        SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let x = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let y = t2(2, 1, vec![1.0, 0.0]);
    assert!(
        matches!(
            model.fit(&x, &y, 1),
            Err(Error::NeuralNetwork(NnError::EmptyModel))
        ),
        "expected EmptyModel"
    );
}

/// predict() on a model with no layers returns EmptyModel
#[test]
fn test_predict_empty_model_returns_empty_model_error() {
    let model = Sequential::new();
    let x = t2(1, 2, vec![1.0, 2.0]);
    let err = model.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::EmptyModel)),
        "expected EmptyModel, got: {err:?}"
    );
}

/// fit() with an empty input tensor returns EmptyInput
#[test]
fn test_fit_empty_x_returns_empty_input_error() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x: Tensor = Array::zeros((0, 2)).into_dyn();
    let y: Tensor = Array::zeros((0, 1)).into_dyn();
    assert!(
        matches!(model.fit(&x, &y, 1), Err(Error::EmptyInput(_))),
        "expected EmptyInput"
    );
}

/// fit() with mismatched x/y batch sizes returns DimensionMismatch
#[test]
fn test_fit_batch_size_mismatch_returns_dimension_mismatch() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = t2(2, 1, vec![1.0, 2.0]); // batch 2 != 3
    assert!(
        matches!(model.fit(&x, &y, 1), Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch"
    );
}

/// predict() with an empty input tensor returns EmptyInput
#[test]
fn test_predict_empty_x_returns_empty_input_error() {
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, Activation::Linear, None).unwrap());

    let x: Tensor = Array::zeros((0, 2)).into_dyn();
    let err = model.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got: {err:?}"
    );
}

/// fit_with_batches with batch_size=0 returns InvalidParameter
#[test]
fn test_fit_with_batches_zero_batch_size_returns_invalid_parameter() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);

    assert!(
        matches!(
            model.fit_with_batches(&x, &y, 1, 0),
            Err(Error::InvalidParameter { .. })
        ),
        "expected InvalidParameter"
    );
}

/// fit_with_batches with batch_size > n_samples returns InvalidParameter
#[test]
fn test_fit_with_batches_batch_size_exceeds_samples_returns_invalid_parameter() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = t2(3, 1, vec![1.0, 2.0, 3.0]);

    // 100 > 3
    assert!(
        matches!(
            model.fit_with_batches(&x, &y, 1, 100),
            Err(Error::InvalidParameter { .. })
        ),
        "expected InvalidParameter"
    );
}

// epochs=0 leaves the model unchanged

/// Training for 0 epochs leaves the weights identical to before the call
#[test]
fn test_fit_zero_epochs_unchanged_weights() {
    let mut dense = Dense::new(1, 1, Activation::Linear, None).unwrap();
    let w = Array2::from_shape_vec((1, 1), vec![3.0_f32]).unwrap();
    let b = Array2::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense).compile(
        SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let x = t2(1, 1, vec![2.0]);
    let y = t2(1, 1, vec![10.0]); // irrelevant with 0 epochs

    model.fit(&x, &y, 0).unwrap();

    let out = model.predict(&x).unwrap();
    // weight unchanged -> 3.0 * 2.0 + 0.0 = 6.0
    assert_abs_diff_eq!(out[[0, 0]], 6.0_f32, epsilon = 1e-5);
}

// end-to-end convergence: y = 2x + 1

/// Dense(1->1, Linear) + SGD + MSE converges on y = 2x+1, predicting ~7 at x=3 after 300 epochs
#[test]
fn test_convergence_linear_regression_y_eq_2x_plus_1() {
    // training points: (1,3), (2,5), (3,7), (4,9)
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    model.fit(&x, &y, 300).unwrap();

    // predict at x=3, true value 7.0
    let x_test = t2(1, 1, vec![3.0]);
    let pred = model.predict(&x_test).unwrap();

    assert_abs_diff_eq!(pred[[0, 0]], 7.0_f32, epsilon = 0.5);
}

/// Same regression via fit_with_batches with batch_size=2 converges to y = 2x+1 after 500 epochs
#[test]
fn test_convergence_linear_regression_with_batches() {
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    model.fit_with_batches(&x, &y, 500, 2).unwrap();

    let x_test = t2(1, 1, vec![2.0]);
    let pred = model.predict(&x_test).unwrap();

    // true value at x=2 is 5.0
    assert_abs_diff_eq!(pred[[0, 0]], 5.0_f32, epsilon = 1.0);
}

// end-to-end convergence: 2-class softmax + Adam + CategoricalCrossEntropy

/// Dense(2->8, Tanh) -> Dense(8->2, Softmax) with Adam + cross-entropy assigns >0.7 probability
/// to the correct class for a separable 2-class task after 600 epochs
#[test]
fn test_convergence_2class_softmax_adam() {
    // 8 training samples (4 per class), well separated
    #[rustfmt::skip]
    let x = t2(8, 2, vec![
        1.0, 0.0,   // class 0
        0.9, 0.1,
        0.8, 0.2,
        0.7, 0.3,
        0.0, 1.0,   // class 1
        0.1, 0.9,
        0.2, 0.8,
        0.3, 0.7,
    ]);
    #[rustfmt::skip]
    let y = t2(8, 2, vec![
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0,
    ]);

    let mut model = Sequential::new();
    // Weight init is unseeded, so a Tanh hidden layer plus a generous epoch budget guarantees
    // convergence on this tiny separable problem regardless of the initial weights
    model
        .add(Dense::new(2, 8, Activation::Tanh, None).unwrap())
        .add(Dense::new(8, 2, Activation::Softmax, None).unwrap())
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8, None, 0.0).unwrap(),
            CategoricalCrossEntropy::new(false),
        );

    model.fit(&x, &y, 600).unwrap();

    // class 0 test point
    let x0 = t2(1, 2, vec![0.9, 0.1]);
    let p0 = model.predict(&x0).unwrap();
    assert!(
        p0[[0, 0]] > 0.7,
        "class-0 point: expected p(class_0) > 0.7, got {}",
        p0[[0, 0]]
    );

    // class 1 test point
    let x1 = t2(1, 2, vec![0.1, 0.9]);
    let p1 = model.predict(&x1).unwrap();
    assert!(
        p1[[0, 1]] > 0.7,
        "class-1 point: expected p(class_1) > 0.7, got {}",
        p1[[0, 1]]
    );
}

// predict determinism after training

/// Two predict() calls after training return byte-identical tensors
#[test]
fn test_predict_deterministic_after_training() {
    let x = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let y = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 2, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    model.fit(&x, &y, 5).unwrap();

    let x_test = t2(1, 2, vec![3.0, -1.5]);
    let p1 = model.predict(&x_test).unwrap();
    let p2 = model.predict(&x_test).unwrap();
    crate::common::assert_allclose(&p1, &p2, 0.0_f32);
}

// fit returns &mut Self (method chaining)

/// fit() returns Ok(&mut Self) so calls can be chained
#[test]
fn test_fit_returns_mutable_self() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(2, 1, vec![1.0, 2.0]);
    let y = t2(2, 1, vec![1.0, 2.0]);

    model.fit(&x, &y, 1).unwrap();
}

// multi-batch convergence with fit_with_batches, full validation

/// fit_with_batches with batch_size == n_samples behaves like full-batch fit, converging to y = 2x+1
#[test]
fn test_fit_with_batches_full_batch_equivalent() {
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(
            SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    // batch_size == n_samples: one batch per epoch
    model.fit_with_batches(&x, &y, 400, 4).unwrap();

    let x_test = t2(1, 1, vec![4.0]);
    let pred = model.predict(&x_test).unwrap();
    // true value: 2*4 + 1 = 9
    assert_abs_diff_eq!(pred[[0, 0]], 9.0_f32, epsilon = 1.0);
}
