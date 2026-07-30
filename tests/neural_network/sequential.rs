//! Integration tests for `Sequential`: add/compile/fit/predict/summary plus error paths.
//!
//! Every test derives its expected value from a mathematical definition or a hand calculation.
//! No test derives it by running the layer and recording the output.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::Activation;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::regularization::dropout::dropout::Dropout;
use rustyml::neural_network::losses::{CategoricalCrossEntropy, MeanSquaredError};
use rustyml::neural_network::optimizers::{Adam, SGD};
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Loss;
use rustyml::{error::Error, neural_network::NnError};

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
    let mut dense = Dense::new(2, 2, Activation::Linear).unwrap();
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
    let mut dense = Dense::new(1, 1, Activation::Linear).unwrap();
    let w = Array2::from_shape_vec((1, 1), vec![2.0_f32]).unwrap();
    let b = Array2::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense);

    let x = t2(1, 1, vec![5.0]);
    let out = model.predict(&x).unwrap();

    assert_abs_diff_eq!(out[[0, 0]], 11.0_f32, epsilon = 1e-6);
}

/// Dense(3->2, Linear) applies a known linear transform to 1 input row
#[test]
fn test_predict_2d_linear_transform() {
    let mut dense = Dense::new(3, 2, Activation::Linear).unwrap();
    // weights shape (in=3, out=2)
    let w = Array2::from_shape_vec(
        (3, 2),
        vec![
            1.0_f32, 0.0, // feature 0
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

/// 2 stacked Linear Dense layers chain correctly: first projects, second sums both inputs
#[test]
fn test_predict_two_layer_stack() {
    let mut d1 = Dense::new(3, 2, Activation::Linear).unwrap();
    let w1 = Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let b1 = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    d1.set_weights(w1, b1).unwrap();

    let mut d2 = Dense::new(2, 1, Activation::Linear).unwrap();
    let w2 = Array2::from_shape_vec((2, 1), vec![1.0_f32, 1.0]).unwrap();
    let b2 = Array2::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    d2.set_weights(w2, b2).unwrap();

    let mut model = Sequential::new();
    model.add(d1).add(d2);

    let x = t2(1, 3, vec![5.0, 7.0, 99.0]);
    let out = model.predict(&x).unwrap();

    // Layer 1 zeroes the third feature, giving [5, 7]. Layer 2 sums them, giving 12.
    assert_abs_diff_eq!(out[[0, 0]], 12.0_f32, epsilon = 1e-5);
}

/// Dense(1->3, Softmax) on a zero pre-activation yields the uniform distribution [1/3, 1/3, 1/3]
#[test]
fn test_predict_dense_softmax_equal_input() {
    let mut dense = Dense::new(1, 3, Activation::Softmax).unwrap();
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
    let mut dense = Dense::new(1, 3, Activation::Softmax).unwrap();
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

// predict: determinism (2 consecutive calls identical)

/// 2 back-to-back predict() calls on the same input produce identical tensors
#[test]
fn test_predict_is_deterministic() {
    let mut dense = Dense::new(3, 2, Activation::Linear).unwrap();
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
        .add(Dense::new(4, 8, Activation::ReLU).unwrap())
        .add(Dense::new(8, 2, Activation::Softmax).unwrap());
    model.summary();
}

// error paths

/// fit() before compile() returns NotCompiled
#[test]
fn test_fit_before_compile_returns_not_compiled() {
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, Activation::Linear).unwrap());
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
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
        .add(Dense::new(2, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
        .add(Dense::new(2, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
    model.add(Dense::new(2, 1, Activation::Linear).unwrap());

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
        .add(Dense::new(2, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
        .add(Dense::new(2, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
    let mut dense = Dense::new(1, 1, Activation::Linear).unwrap();
    let w = Array2::from_shape_vec((1, 1), vec![3.0_f32]).unwrap();
    let b = Array2::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    dense.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(dense).compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
        .add(
            Dense::new(1, 1, Activation::Linear)
                .unwrap()
                .with_random_state(0),
        )
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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

    let mut model = Sequential::new_with_seed(0);
    model
        .add(
            Dense::new(1, 1, Activation::Linear)
                .unwrap()
                .with_random_state(0),
        )
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
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
    // Seeds the weight init to avoid a pathological draw. Otherwise the Tanh hidden layer could
    // saturate and push the test below the 0.7 probability threshold within the epoch budget.
    model
        .add(
            Dense::new(2, 8, Activation::Tanh)
                .unwrap()
                .with_random_state(0),
        )
        .add(
            Dense::new(8, 2, Activation::Softmax)
                .unwrap()
                .with_random_state(0),
        )
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap(),
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

/// 2 predict() calls after training return byte-identical tensors
#[test]
fn test_predict_deterministic_after_training() {
    let x = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let y = t2(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 2, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    model.fit(&x, &y, 5).unwrap();

    let x_test = t2(1, 2, vec![3.0, -1.5]);
    let p1 = model.predict(&x_test).unwrap();
    let p2 = model.predict(&x_test).unwrap();
    crate::common::assert_allclose(&p1, &p2, 0.0_f32);
}

// fit returns a History of per-epoch losses

/// fit() records exactly 1 loss per epoch, in epoch order, and training drives them down
#[test]
fn test_fit_history_has_one_loss_per_epoch() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let history = model.fit(&x, &y, 25).unwrap();

    assert_eq!(history.loss().len(), 25, "one entry per epoch");
    assert!(
        history.loss().iter().all(|l| l.is_finite()),
        "every recorded loss is finite: {:?}",
        history.loss()
    );
    assert!(
        history.loss()[24] < history.loss()[0],
        "25 epochs of gradient descent must reduce the loss: {} -> {}",
        history.loss()[0],
        history.loss()[24]
    );
}

/// fit() over 0 epochs still validates, and returns a history with no entries
#[test]
fn test_fit_zero_epochs_yields_empty_history() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(2, 1, vec![1.0, 2.0]);
    let y = t2(2, 1, vec![1.0, 2.0]);

    assert!(model.fit(&x, &y, 0).unwrap().loss().is_empty());
}

/// Each entry records the loss before that epoch's own update, so entry 0 is the untrained
/// model's loss (Keras' convention). The last entry overstates the trained model's loss.
#[test]
fn test_fit_history_records_the_pre_update_loss() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let before = model.evaluate(&x, &y).unwrap();
    let history = model.fit(&x, &y, 10).unwrap();
    let after = model.evaluate(&x, &y).unwrap();

    assert_abs_diff_eq!(history.loss()[0], before, epsilon = 1e-5);
    assert!(
        after < history.loss()[9],
        "the trained model's loss ({}) is below the last recorded epoch loss ({}), which was \
         measured one update earlier",
        after,
        history.loss()[9]
    );
}

/// `fit_with_batches` weights each batch by its sample count. The epoch loss is therefore the
/// dataset-wide mean per-sample loss, matching Keras and identical across shuffles.
#[test]
fn test_fit_with_batches_weights_batches_by_sample_count() {
    // 5 samples in batches of 2 leaves 1 sample alone in the trailing batch. Call the
    // per-sample losses l_0 through l_4, with total S. For whichever sample s lands alone,
    // a plain mean over the 3 batches is ((S - l_s) / 2 + l_s) / 3. This reduces to
    // (S + l_s) / 6, for any pairing of the other 4 samples. Weighting by sample count instead
    // gives S/5. The 2 rules agree only when l_s equals S/5, which the assertion below rules
    // out for every sample.
    let x = t2(5, 1, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    let y = t2(5, 1, vec![0.0, 1.0, 4.0, 9.0, 100.0]);

    for seed in 0..5_u64 {
        let mut model = Sequential::new_with_seed(seed);
        model
            .add(
                Dense::new(1, 1, Activation::Linear)
                    .unwrap()
                    .with_random_state(7),
            )
            // A step this small is below the f32 resolution of the weights it would move, so
            // the model is frozen. The per-sample losses stay fixed across the epoch
            .compile(
                SGD::new(1e-30, 0.0, false, 0.0).unwrap(),
                MeanSquaredError::new(),
            );

        let predictions = model.predict(&x).unwrap();
        let per_sample: Vec<f32> = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .collect();
        let total: f32 = per_sample.iter().sum();
        let weighted = total / 5.0;

        for (i, &l) in per_sample.iter().enumerate() {
            assert!(
                (l - weighted).abs() > 1.0,
                "sample {i}'s loss ({l}) must differ from the mean ({weighted}), or the two \
                 aggregation rules would coincide and the test would prove nothing"
            );
        }

        let history = model.fit_with_batches(&x, &y, 1, 2).unwrap();

        assert_abs_diff_eq!(model.evaluate(&x, &y).unwrap(), weighted, epsilon = 1e-3);
        assert_abs_diff_eq!(history.loss()[0], weighted, epsilon = 1e-3);
        for &l in &per_sample {
            let unweighted = (total + l) / 6.0;
            assert!(
                (history.loss()[0] - unweighted).abs() > 1e-2,
                "seed {seed}: the epoch loss ({}) must not be a plain mean over batches ({})",
                history.loss()[0],
                unweighted
            );
        }
    }
}

// multi-batch convergence with fit_with_batches, full validation

/// fit_with_batches with batch_size == n_samples behaves like full-batch fit, converging to
/// y = 2x+1
#[test]
fn test_fit_with_batches_full_batch_equivalent() {
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    // batch_size == n_samples: 1 batch per epoch
    model.fit_with_batches(&x, &y, 400, 4).unwrap();

    let x_test = t2(1, 1, vec![4.0]);
    let pred = model.predict(&x_test).unwrap();
    // true value: 2*4 + 1 = 9
    assert_abs_diff_eq!(pred[[0, 0]], 9.0_f32, epsilon = 1.0);
}

// train_batch: the public single-step primitive

/// `fit` is exactly a loop over `train_batch`: the same data through either path leaves the same
/// weights, and the history holds precisely the per-step losses
#[test]
fn test_train_batch_reproduces_fit() {
    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let build = || {
        let mut model = Sequential::new();
        model
            .add(
                Dense::new(1, 1, Activation::Linear)
                    .unwrap()
                    .with_random_state(11),
            )
            .compile(
                SGD::new(0.01, 0.0, false, 0.0).unwrap(),
                MeanSquaredError::new(),
            );
        model
    };

    let mut looped = build();
    let history = looped.fit(&x, &y, 3).unwrap();

    let mut stepped = build();
    let manual: Vec<f32> = (0..3)
        .map(|_| stepped.train_batch(&x, &y).unwrap())
        .collect();

    assert_eq!(history.loss(), manual.as_slice());
    crate::common::assert_allclose(
        &looped.predict(&x).unwrap(),
        &stepped.predict(&x).unwrap(),
        0.0_f32,
    );
}

/// `train_batch` returns the loss from the forward pass that precedes its own update. That loss
/// equals the loss of the model as it stood on entry
#[test]
fn test_train_batch_returns_the_pre_update_loss() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = t2(4, 1, vec![3.0, 5.0, 7.0, 9.0]);

    let before = model.evaluate(&x, &y).unwrap();
    let reported = model.train_batch(&x, &y).unwrap();
    let after = model.evaluate(&x, &y).unwrap();

    assert_abs_diff_eq!(reported, before, epsilon = 1e-5);
    assert!(
        after < reported,
        "the update the call performed must lower the loss it reported: {reported} -> {after}"
    );
}

/// Being public, `train_batch` has to validate for itself rather than lean on `fit` having done
/// it: an uncompiled model is an error, not a panic
#[test]
fn test_train_batch_on_uncompiled_model_is_an_error() {
    let mut model = Sequential::new();
    model.add(Dense::new(1, 1, Activation::Linear).unwrap());

    let x = t2(2, 1, vec![1.0, 2.0]);
    let y = t2(2, 1, vec![1.0, 2.0]);

    assert!(
        matches!(
            model.train_batch(&x, &y),
            Err(Error::NeuralNetwork(NnError::NotCompiled(_)))
        ),
        "expected NotCompiled"
    );
}

// evaluate: scoring without training

/// `evaluate` is `predict` scored with the compiled loss, and leaves the model untouched
#[test]
fn test_evaluate_matches_predict_then_compute_loss() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, Activation::ReLU).unwrap())
        .add(Dense::new(3, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = t2(3, 2, vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.5]);
    let y = t2(3, 1, vec![1.0, -2.0, 0.5]);

    let manual = MeanSquaredError::new()
        .compute_loss(&y, &model.predict(&x).unwrap())
        .unwrap();

    assert_abs_diff_eq!(model.evaluate(&x, &y).unwrap(), manual, epsilon = 1e-6);
    // Scoring is not training: a second call sees exactly the same model
    assert_abs_diff_eq!(model.evaluate(&x, &y).unwrap(), manual, epsilon = 0.0_f32);
}

/// `evaluate` runs in inference mode, so dropout is the identity and repeated calls agree.
/// The training path samples a fresh mask each call on the same frozen model, so it does not.
#[test]
fn test_evaluate_runs_in_inference_mode() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 32, Activation::Linear).unwrap())
        .add(Dropout::new(0.5, vec![16, 32]).unwrap())
        .add(Dense::new(32, 1, Activation::Linear).unwrap())
        // Frozen: with a step this far below the weights' f32 resolution, every difference
        // observed below comes from the dropout mask alone
        .compile(
            SGD::new(1e-30, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let x = Array::linspace(-1.0, 1.0, 64)
        .into_shape_with_order((16, 4))
        .unwrap()
        .into_dyn();
    let y = Array::linspace(0.0, 1.0, 16)
        .into_shape_with_order((16, 1))
        .unwrap()
        .into_dyn();

    let eval_first = model.evaluate(&x, &y).unwrap();
    let eval_second = model.evaluate(&x, &y).unwrap();
    assert_abs_diff_eq!(eval_first, eval_second, epsilon = 0.0_f32);

    let train_first = model.train_batch(&x, &y).unwrap();
    let train_second = model.train_batch(&x, &y).unwrap();
    assert!(
        (train_first - train_second).abs() > 1e-6,
        "training mode must resample the dropout mask, but two passes agreed: \
         {train_first} and {train_second}"
    );
    assert!(
        (train_first - eval_first).abs() > 1e-6,
        "a dropout model's training loss ({train_first}) must not match its inference loss \
         ({eval_first})"
    );
}

/// `evaluate` needs a loss but no optimizer, so an uncompiled model is told what is actually
/// missing for the call it made
#[test]
fn test_evaluate_on_uncompiled_model_reports_the_missing_loss() {
    let mut model = Sequential::new();
    model.add(Dense::new(1, 1, Activation::Linear).unwrap());

    let x = t2(2, 1, vec![1.0, 2.0]);
    let y = t2(2, 1, vec![1.0, 2.0]);

    assert!(
        matches!(
            model.evaluate(&x, &y),
            Err(Error::NeuralNetwork(NnError::NotCompiled("loss function")))
        ),
        "expected NotCompiled(\"loss function\")"
    );
}

/// A rank-0 tensor holds 1 element, so it is not "empty", but it has no batch axis to compare.
/// The training and scoring paths reject it rather than panic on the missing axis.
#[test]
fn test_rank_zero_inputs_are_rejected() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let scalar: Tensor = Array::from_elem(IxDyn(&[]), 1.0_f32);
    let y = t2(1, 1, vec![1.0]);

    assert!(matches!(
        model.fit(&scalar, &y, 1),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        model.evaluate(&y, &scalar),
        Err(Error::InvalidInput(_))
    ));
}

// learning rate: readable, not just writable

/// The model reports the optimizer's current rate, so a schedule can derive the next step from
/// it. An uncompiled model has no rate to report.
#[test]
fn test_learning_rate_reads_back_through_the_model() {
    let mut model = Sequential::new();
    assert_eq!(model.learning_rate(), None);

    model
        .add(Dense::new(1, 1, Activation::Linear).unwrap())
        .compile(
            Adam::new(0.003, 0.9, 0.999, 1e-8, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    assert_eq!(model.learning_rate(), Some(0.003));

    // A halving schedule written the way it should be: read, scale, write
    let halved = model.learning_rate().unwrap() * 0.5;
    model.set_learning_rate(halved);
    assert_eq!(model.learning_rate(), Some(0.0015));
}

/// Cross-checked against Keras 3.15 (jax backend): `Dense(1)`, `w = 0.5`, `b = 0`, `SGD(0.1)`,
/// `mse`, on `y = 2x` for `x = 0..4`, batched by 2. Every literal below matches Keras' output.
#[test]
fn test_batch_losses_and_epoch_mean_match_keras() {
    let mut dense = Dense::new(1, 1, Activation::Linear).unwrap();
    dense
        .set_weights(
            Array2::from_shape_vec((1, 1), vec![0.5_f32]).unwrap(),
            Array2::from_shape_vec((1, 1), vec![0.0_f32]).unwrap(),
        )
        .unwrap();

    let mut model = Sequential::new();
    model.add(dense).compile(
        SGD::new(0.1, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let x = t2(5, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let y = t2(5, 1, vec![0.0, 2.0, 4.0, 6.0, 8.0]);

    // keras: model.evaluate(x, y) == 13.5
    assert_abs_diff_eq!(model.evaluate(&x, &y).unwrap(), 13.5_f32, epsilon = 1e-5);

    // keras: 3 successive train_on_batch calls, in this order
    let batches = [
        (
            t2(2, 1, vec![0.0, 1.0]),
            t2(2, 1, vec![0.0, 2.0]),
            1.125_f32,
        ),
        (
            t2(2, 1, vec![2.0, 3.0]),
            t2(2, 1, vec![4.0, 6.0]),
            10.856_251_f32,
        ),
        (t2(1, 1, vec![4.0]), t2(1, 1, vec![8.0]), 4.473_228_f32),
    ];

    let mut weighted = 0.0_f64;
    let mut samples = 0_usize;
    for (batch_x, batch_y, expected) in &batches {
        let loss = model.train_batch(batch_x, batch_y).unwrap();
        assert_abs_diff_eq!(loss, *expected, epsilon = 1e-4);
        weighted += loss as f64 * batch_x.shape()[0] as f64;
        samples += batch_x.shape()[0];
    }

    // keras: history.history['loss'][0] == 5.687146186828613 for the same epoch. Weighting each
    // batch by its sample count reproduces it. A plain mean over the 3 batches gives
    // 5.4848262, which is not what Keras reports.
    assert_abs_diff_eq!(
        (weighted / samples as f64) as f32,
        5.687_146_f32,
        epsilon = 1e-4
    );

    // keras: model.evaluate(x, y) after the epoch == 9.241999626159668. This value is above the
    // epoch figure because this learning rate overshoots. That is why the recorded loss cannot
    // be read as the trained model's loss.
    assert_abs_diff_eq!(model.evaluate(&x, &y).unwrap(), 9.242_0_f32, epsilon = 1e-4);
}
