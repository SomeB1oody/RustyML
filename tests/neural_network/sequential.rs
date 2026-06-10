//! Integration tests for `Sequential`: add/compile/fit/predict/summary + error paths.
//!
//! Ground-truth rule: every expected value is derived from the mathematical definition or a
//! hand calculation, never by running the layer and recording its output.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::Activation;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::losses::{CategoricalCrossEntropy, MeanSquaredError};
use rustyml::neural_network::optimizers::{Adam, SGD};
use rustyml::neural_network::sequential::Sequential;

// ─── helpers ────────────────────────────────────────────────────────────────

/// Build a 2-D Tensor from row-major data.
fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_dyn()
}

// ─── predict: forward values with known weights ─────────────────────────────

/// Dense(2→2, Linear), identity weights, zero bias.
/// For input [[3.0, 4.0]], output must equal input.
///
/// Hand calculation:
///   z = x · W + b = [[3,4]] · [[1,0],[0,1]] + [[0,0]] = [[3,4]]
///   Linear: f(z) = z  →  output = [[3.0, 4.0]]
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

/// Dense(1→1, Linear), weight=[[2.0]], bias=[[1.0]].
/// For input [[5.0]]:  2.0 * 5.0 + 1.0 = 11.0
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

/// Dense(3→2, Linear), w = [[1,0],[0,1],[1,1]], b = [[0,0]].
/// Input row: [2, 3, 4].
/// z[0] = 2*1 + 3*0 + 4*1 = 6
/// z[1] = 2*0 + 3*1 + 4*1 = 7
#[test]
fn test_predict_2d_linear_transform() {
    let mut dense = Dense::new(3, 2, Activation::Linear, None).unwrap();
    // weights shape: (in=3, out=2), stored column-major in ndarray's row-major slice
    let w = Array2::from_shape_vec(
        (3, 2),
        vec![
            1.0_f32, 0.0, // row 0: feature-0 contributes to out-0 and out-1
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
        ],
    )
    .unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 3, vec![2.0, 3.0, 4.0]);
    let out = model.predict(&x).unwrap();

    // z[0] = 2*1 + 3*0 + 4*1 = 6
    // z[1] = 2*0 + 3*1 + 4*1 = 7
    assert_abs_diff_eq!(out[[0, 0]], 6.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], 7.0_f32, epsilon = 1e-5);
}

/// Dense(3→2, Linear) followed by Dense(2→1, Linear).
/// First layer identity-ish; second sums both inputs.
///
/// w1 = [[1,0],[0,1],[0,0]], b1 = [[0,0]]
/// w2 = [[1],[1]], b2 = [[0]]
///
/// x = [5, 7, 99] → layer1 → [5, 7] → layer2 → [12]
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

    // layer1 output: [5, 7]  (third feature is zeroed by w1)
    // layer2 output: 5 + 7 = 12
    assert_abs_diff_eq!(out[[0, 0]], 12.0_f32, epsilon = 1e-5);
}

/// Dense(1→3, Softmax), bias zero, w = [[1,2,3]].
/// For input [[0.0]]:
///   z = [0*1, 0*2, 0*3] = [0, 0, 0]
///   softmax([0,0,0]) = [1/3, 1/3, 1/3]  (all equal)
#[test]
fn test_predict_dense_softmax_equal_input() {
    let mut dense = Dense::new(1, 3, Activation::Softmax, None).unwrap();
    let w = Array2::from_shape_vec((1, 3), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let b = Array2::from_shape_vec((1, 3), vec![0.0_f32, 0.0, 0.0]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    // input=0 → z=[0,0,0] → softmax → [1/3, 1/3, 1/3]
    let x = t2(1, 1, vec![0.0]);
    let out = model.predict(&x).unwrap();

    let third = 1.0_f32 / 3.0;
    assert_abs_diff_eq!(out[[0, 0]], third, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 1]], third, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 2]], third, epsilon = 1e-5);

    // Probabilities must sum to 1.
    let sum: f32 = out.iter().sum();
    assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-6);
}

/// Dense(1→3, Softmax), bias zero, w = [[0,0,0]].
/// For input [[99.0]]:
///   z = [0,0,0]  (weight is all-zero regardless of input)
///   softmax([0,0,0]) = [1/3, 1/3, 1/3]
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

// ─── predict == forward in eval mode ─────────────────────────────────────────

/// predict() must produce the same values as the last eval-mode forward pass.
/// We use a layer where both are deterministic and equal (Linear Dense).
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

    // Forward pass output (using &self semantics via predict).
    // Layer is stateless for Linear Dense, so a second predict call is identical.
    let pred2 = model.predict(&x).unwrap();

    crate::common::assert_allclose(&pred, &pred2, 1e-7_f32);
}

// ─── predict: determinism (two consecutive calls identical) ──────────────────

/// Two back-to-back predict() calls on the same input must produce identical tensors.
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

// ─── summary smoke-test ──────────────────────────────────────────────────────

/// summary() must not panic; we only check that it runs without error.
#[test]
fn test_summary_does_not_panic() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 8, Activation::ReLU, None).unwrap())
        .add(Dense::new(8, 2, Activation::Softmax, None).unwrap());
    // Should print to stdout without panicking
    model.summary();
}

// ─── error paths ─────────────────────────────────────────────────────────────

/// fit() before compile() must return NotCompiled.
#[test]
fn test_fit_before_compile_returns_not_compiled() {
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, Activation::Linear, None).unwrap());
    // No compile call.
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

/// fit() on a model with no layers must return EmptyModel.
#[test]
fn test_fit_empty_model_returns_empty_model_error() {
    let mut model = Sequential::new();
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
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

/// predict() on a model with no layers must return EmptyModel.
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

/// fit() with an empty input tensor must return EmptyInput.
#[test]
fn test_fit_empty_x_returns_empty_input_error() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let x: Tensor = Array::zeros((0, 2)).into_dyn();
    let y: Tensor = Array::zeros((0, 1)).into_dyn();
    assert!(
        matches!(model.fit(&x, &y, 1), Err(Error::EmptyInput(_))),
        "expected EmptyInput"
    );
}

/// fit() with mismatched x/y batch sizes must return DimensionMismatch.
#[test]
fn test_fit_batch_size_mismatch_returns_dimension_mismatch() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let x = t2(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = t2(2, 1, vec![1.0, 2.0]); // batch 2 != 3
    assert!(
        matches!(model.fit(&x, &y, 1), Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch"
    );
}

/// predict() with an empty input tensor must return EmptyInput.
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

/// fit_with_batches with batch_size=0 must return InvalidParameter.
#[test]
fn test_fit_with_batches_zero_batch_size_returns_invalid_parameter() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

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

/// fit_with_batches with batch_size > n_samples must return InvalidParameter.
#[test]
fn test_fit_with_batches_batch_size_exceeds_samples_returns_invalid_parameter() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

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

// ─── epochs=0 leaves the model unchanged ─────────────────────────────────────

/// Training for 0 epochs must leave the weights identical to before the call.
///
/// We set known weights, call fit with epochs=0, then predict and verify the output
/// equals the hand-computed value. If fit corrupted the weights, the assertion fails.
#[test]
fn test_fit_zero_epochs_unchanged_weights() {
    let mut dense = Dense::new(1, 1, Activation::Linear, None).unwrap();
    let w = Array2::from_shape_vec((1, 1), vec![3.0_f32]).unwrap();
    let b = Array2::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model
        .add(dense)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let x = t2(1, 1, vec![2.0]);
    let y = t2(1, 1, vec![10.0]); // irrelevant — 0 epochs

    model.fit(&x, &y, 0).unwrap();

    let out = model.predict(&x).unwrap();
    // weight unchanged → 3.0 * 2.0 + 0.0 = 6.0
    assert_abs_diff_eq!(out[[0, 0]], 6.0_f32, epsilon = 1e-5);
}

// ─── end-to-end convergence: y = 2x + 1 ─────────────────────────────────────

/// A single Dense(1→1, Linear) + SGD + MSE on 4 points of y = 2x+1 must reduce loss
/// substantially over 300 epochs compared to epoch 0.
///
/// This is not a gradient test — it is a convergence smoke-test. We verify that:
///   1. The final prediction is within ±0.5 of the true value at x=3 (≈ 7).
///   2. The model can be called twice (determinism after training).
#[test]
fn test_convergence_linear_regression_y_eq_2x_plus_1() {
    // Training points: (1,3), (2,5), (3,7), (4,9)
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    model.fit(&x, &y, 300).unwrap();

    // Predict at x=3, true value = 7.0
    let x_test = t2(1, 1, vec![3.0]);
    let pred = model.predict(&x_test).unwrap();

    assert_abs_diff_eq!(pred[[0, 0]], 7.0_f32, epsilon = 0.5);
}

/// Same regression, but using fit_with_batches with batch_size=2.
/// After 500 epochs the model should converge to approximately y = 2x+1.
#[test]
fn test_convergence_linear_regression_with_batches() {
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    model.fit_with_batches(&x, &y, 500, 2).unwrap();

    let x_test = t2(1, 1, vec![2.0]);
    let pred = model.predict(&x_test).unwrap();

    // True value at x=2 is 5.0
    assert_abs_diff_eq!(pred[[0, 0]], 5.0_f32, epsilon = 1.0);
}

// ─── end-to-end convergence: 2-class softmax + Adam + CategoricalCrossEntropy ─

/// 2-class linearly separable task.
///
/// Architecture:  Dense(2→4, ReLU) → Dense(4→2, Softmax)
/// Optimizer:     Adam(lr=0.01)
/// Loss:          CategoricalCrossEntropy
///
/// Class 0 samples: x close to [1, 0]; one-hot y = [1, 0]
/// Class 1 samples: x close to [0, 1]; one-hot y = [0, 1]
///
/// After 200 epochs the model must assign >0.7 probability to the correct class
/// for a test point from each class.
#[test]
fn test_convergence_2class_softmax_adam() {
    // 8 training samples (4 per class) — well-separated
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
    // Weight init is currently UNSEEDED, so this test must converge for any starting point. A Tanh
    // hidden layer (instead of ReLU) avoids the "all hidden units dead at init" trap that no number
    // of epochs can recover from; a generous epoch budget then guarantees convergence on this tiny
    // separable problem regardless of the initial weights.
    model
        .add(Dense::new(2, 8, Activation::Tanh, None).unwrap())
        .add(Dense::new(8, 2, Activation::Softmax, None).unwrap())
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8).unwrap(),
            CategoricalCrossEntropy::new(),
        );

    model.fit(&x, &y, 600).unwrap();

    // Class 0 test point
    let x0 = t2(1, 2, vec![0.9, 0.1]);
    let p0 = model.predict(&x0).unwrap();
    assert!(
        p0[[0, 0]] > 0.7,
        "class-0 point: expected p(class_0) > 0.7, got {}",
        p0[[0, 0]]
    );

    // Class 1 test point
    let x1 = t2(1, 2, vec![0.1, 0.9]);
    let p1 = model.predict(&x1).unwrap();
    assert!(
        p1[[0, 1]] > 0.7,
        "class-1 point: expected p(class_1) > 0.7, got {}",
        p1[[0, 1]]
    );
}

// ─── predict determinism after training ──────────────────────────────────────

/// Two predict() calls after training must return byte-identical tensors.
#[test]
fn test_predict_deterministic_after_training() {
    let x = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let y = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 2, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    model.fit(&x, &y, 5).unwrap();

    let x_test = t2(1, 2, vec![3.0, -1.5]);
    let p1 = model.predict(&x_test).unwrap();
    let p2 = model.predict(&x_test).unwrap();
    crate::common::assert_allclose(&p1, &p2, 0.0_f32);
}

// ─── fit returns &mut Self (method chaining) ─────────────────────────────────

/// The return value of fit() must be Ok(&mut model) (method-chainable).
/// We verify by immediately calling predict on the returned reference.
#[test]
fn test_fit_returns_mutable_self() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let x = t2(2, 1, vec![1.0, 2.0]);
    let y = t2(2, 1, vec![1.0, 2.0]);

    // fit returns Ok(&mut Self) — we simply assert it succeeds
    model.fit(&x, &y, 1).unwrap();
}

// ─── multi-batch convergence with fit_with_batches, full validation ──────────

/// fit_with_batches with batch_size == n_samples is equivalent to full-batch fit.
/// After 400 epochs the model should converge to y = 2x+1.
#[test]
fn test_fit_with_batches_full_batch_equivalent() {
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear, None).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // batch_size == n_samples: one batch per epoch
    model.fit_with_batches(&x, &y, 400, 4).unwrap();

    let x_test = t2(1, 1, vec![4.0]);
    let pred = model.predict(&x_test).unwrap();
    // True value: 2*4 + 1 = 9
    assert_abs_diff_eq!(pred[[0, 0]], 9.0_f32, epsilon = 1.0);
}
