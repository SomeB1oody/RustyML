//! Integration tests for the `Embedding` layer.
//!
//! Covers the constructor, forward values against a Keras reference, the index rules, the
//! scatter-add backward pass, memory layout, the error paths, and the weight surface.
//! `gradient_check.rs` covers gradient values against finite differences. This file does not
//! duplicate them.
//!
//! The pinned reference numbers come from Keras 3.15 on the jax backend. Each one is written in
//! the shortest decimal form that reads back as the same `f32`.

use ndarray::{Array1, Array2, Array3, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::embedding::Embedding;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// helpers

/// Build a 2D Tensor from row-major data
fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 3D Tensor from row-major data
fn t3(a: usize, b: usize, c: usize, data: Vec<f32>) -> Tensor {
    Array3::from_shape_vec((a, b, c), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build an `Embedding` whose table holds the given row-major values
fn embedding_with_table(input_dim: usize, output_dim: usize, table: Vec<f32>) -> Embedding {
    let mut layer = Embedding::new(input_dim, output_dim).unwrap();
    layer
        .set_weights(Array2::from_shape_vec((input_dim, output_dim), table).unwrap())
        .unwrap();
    layer
}

/// The 4-by-3 table that the Keras reference cases share
fn reference_table() -> Vec<f32> {
    vec![
        -0.902, -0.914, 0.246, 0.414, -0.467, 0.544, -0.565, 0.87, -0.547, 0.905, -0.421, 0.491,
    ]
}

// Constructor validation

/// A table needs at least 1 row and at least 1 column
#[test]
fn embedding_new_rejects_a_zero_dimension() {
    for (input_dim, output_dim, which) in [(0usize, 4usize, "input_dim"), (4, 0, "output_dim")] {
        let result = Embedding::new(input_dim, output_dim);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "{which} of 0 must be rejected"
        );
    }
}

/// A well-formed pair builds a layer whose parameter count is the table size
#[test]
fn embedding_param_count_is_the_table_size() {
    for (input_dim, output_dim) in [(1usize, 1usize), (10, 5), (128, 64)] {
        let layer = Embedding::new(input_dim, output_dim).unwrap();
        assert_eq!(
            layer.param_count(),
            TrainingParameters::Trainable(input_dim * output_dim)
        );
    }
}

// Forward values

/// A rank-2 index batch gathers the matching rows, matching Keras 3.15 case `rank2_batch`
#[test]
fn embedding_forward_matches_the_keras_rank_2_reference() {
    let table = vec![
        -0.015, -0.383, 0.963, -0.741, -0.828, -0.583, -0.099, -0.909,
    ];
    let mut layer = embedding_with_table(4, 2, table);
    let x = t2(2, 3, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

    let out = layer.forward(&x).unwrap();
    let expected = t3(
        2,
        3,
        2,
        vec![
            -0.015, -0.383, 0.963, -0.741, -0.828, -0.583, -0.099, -0.909, -0.015, -0.383, 0.963,
            -0.741,
        ],
    );
    assert_allclose(&out, &expected, 1e-6_f32);
}

/// A rank-1 index vector gives a rank-2 output, and a rank-3 batch gives a rank-4 output
#[test]
fn embedding_forward_adds_exactly_1_axis_at_any_rank() {
    let mut layer = embedding_with_table(4, 3, reference_table());

    let rank_1 = Array1::from_vec(vec![2.0f32, 0.0]).into_dyn();
    assert_eq!(layer.forward(&rank_1).unwrap().shape(), &[2, 3]);

    let rank_2 = t2(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(layer.forward(&rank_2).unwrap().shape(), &[2, 2, 3]);

    let rank_3 = t3(2, 1, 2, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(layer.forward(&rank_3).unwrap().shape(), &[2, 1, 2, 3]);
}

/// An index truncates toward zero, exactly as the Keras cast to a whole number does
#[test]
fn embedding_forward_truncates_an_index_toward_zero() {
    let table = vec![
        -0.352, 0.378, -0.549, 0.775, 0.492, 0.323, 0.203, -0.655, 0.872, 0.114, 0.373, -0.47,
        0.77, 0.008, 0.255,
    ];
    let mut layer = embedding_with_table(5, 3, table.clone());
    // Keras case `float_index_truncate_toward_zero`: these 5 values cast to 0, 0, 1, 2, and 4
    let x = Array1::from_vec(vec![-0.5f32, 0.5, 1.7, 2.9, 4.999]).into_dyn();

    let out = layer.forward(&x).unwrap();
    let mut expected = Vec::new();
    for row in [0usize, 0, 1, 2, 4] {
        expected.extend_from_slice(&table[row * 3..row * 3 + 3]);
    }
    assert_allclose(&out, &t2(5, 3, expected), 1e-6_f32);
}

/// The eval path gives the same values as the training path
#[test]
fn embedding_predict_equals_forward() {
    let mut layer = embedding_with_table(4, 3, reference_table());
    let x = t2(2, 4, vec![1.0, 1.0, 0.0, 3.0, 3.0, 1.0, 2.0, 1.0]);

    let training = layer.forward(&x).unwrap();
    let inference = layer.predict(&x).unwrap();
    assert_allclose(&inference, &training, 0.0_f32);
}

// Layout

/// An index tensor whose memory is not in C order still gives the right values, in C order
#[test]
fn embedding_accepts_an_input_that_is_not_in_c_order() {
    let mut layer = embedding_with_table(4, 3, reference_table());
    let base = t2(2, 3, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

    // A transposed view holds the same values in a different memory order
    let transposed = base.clone().permuted_axes(IxDyn(&[1, 0]));
    assert!(!transposed.is_standard_layout());

    let out = layer.forward(&transposed).unwrap();
    assert!(out.is_standard_layout(), "the output must be in C order");

    // Row j of the transposed input holds column j of the base input
    let expected_indices = [0usize, 3, 1, 0, 2, 1];
    let table = reference_table();
    let mut expected = Vec::new();
    for row in expected_indices {
        expected.extend_from_slice(&table[row * 3..row * 3 + 3]);
    }
    assert_allclose(&out, &t3(3, 2, 3, expected), 0.0_f32);
}

// Backward

/// The gradient of a row is the sum over every position that selected it
///
/// The values come from Keras 3.15 case `grad_repeated_indices_linear`, where row 1 is selected
/// 4 times, row 3 twice, and rows 0 and 2 once each
#[test]
fn embedding_backward_matches_the_keras_scatter_add_reference() {
    let mut layer = embedding_with_table(4, 3, reference_table());
    let x = t2(2, 4, vec![1.0, 1.0, 0.0, 3.0, 3.0, 1.0, 2.0, 1.0]);
    layer.forward(&x).unwrap();

    let upstream = t3(
        2,
        4,
        3,
        vec![
            0.247, -0.787, 0.243, -0.156, -0.42, 0.425, -0.331, 0.724, -0.838, 0.54, -0.842, 0.128,
            0.215, -0.744, -0.342, -0.491, -0.391, -0.666, -0.353, -0.202, 0.067, 0.659, 0.993,
            0.326,
        ],
    );
    layer.backward(&upstream).unwrap();

    let expected = Array2::from_shape_vec(
        (4, 3),
        vec![
            -0.331, 0.724, -0.838, 0.25899997, -0.6050001, 0.32799998, -0.353, -0.202, 0.067,
            0.755, -1.586, -0.214,
        ],
    )
    .unwrap();

    let params = layer.parameters();
    assert_eq!(params.len(), 1, "the layer exposes exactly 1 tensor");
    let grad = Array2::from_shape_vec((4, 3), params[0].grad.to_vec()).unwrap();
    assert_allclose(&grad, &expected, 1e-6_f32);
}

/// A row that no index selected keeps a gradient of exactly 0
#[test]
fn embedding_backward_leaves_an_unused_row_at_zero() {
    let table = vec![
        -0.801, 0.112, -0.008, 0.734, -0.247, -0.6, 0.372, 0.384, -0.024, -0.424,
    ];
    let mut layer = embedding_with_table(5, 2, table);
    // Keras case `grad_rank1_untouched_rows_stay_zero`: rows 1, 3, and 4 are never selected
    let x = Array1::from_vec(vec![2.0f32, 2.0, 2.0, 0.0]).into_dyn();
    layer.forward(&x).unwrap();

    let upstream = t2(
        4,
        2,
        vec![-0.76, 0.854, -0.992, 0.408, 0.544, 0.466, -0.867, 0.282],
    );
    layer.backward(&upstream).unwrap();

    let expected = Array2::from_shape_vec(
        (5, 2),
        vec![
            -0.867, 0.282, 0.0, 0.0, -1.208, 1.7279999, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let params = layer.parameters();
    let grad = Array2::from_shape_vec((5, 2), params[0].grad.to_vec()).unwrap();
    assert_allclose(&grad, &expected, 1e-6_f32);
    for row in [1usize, 3, 4] {
        assert_eq!(grad[[row, 0]], 0.0, "row {row} must stay exactly 0");
        assert_eq!(grad[[row, 1]], 0.0, "row {row} must stay exactly 0");
    }
}

/// A second backward pass replaces the first gradient rather than adding to it
///
/// The layer keeps its gradient buffer allocated between steps. A buffer that is not cleared
/// would double the gradient on the second step, and every later step would drift further
#[test]
fn embedding_backward_clears_the_gradient_of_the_previous_step() {
    let mut layer = embedding_with_table(3, 2, vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
    let x = Array1::from_vec(vec![1.0f32, 1.0]).into_dyn();
    let upstream = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    layer.forward(&x).unwrap();
    layer.backward(&upstream).unwrap();
    let first = layer.parameters()[0].grad.to_vec();

    layer.forward(&x).unwrap();
    layer.backward(&upstream).unwrap();
    let second = layer.parameters()[0].grad.to_vec();

    assert_eq!(first, vec![0.0, 0.0, 4.0, 6.0, 0.0, 0.0]);
    assert_eq!(
        second, first,
        "the buffer must be refilled, not accumulated"
    );
}

/// The gradient handed to the layer before this one is 0, and it has the input's shape
#[test]
fn embedding_backward_returns_a_zero_input_gradient() {
    let mut layer = embedding_with_table(4, 3, reference_table());
    let x = t2(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    let out = layer.forward(&x).unwrap();

    let grad_input = layer.backward(&Tensor::ones(out.raw_dim())).unwrap();
    assert_eq!(grad_input.shape(), x.shape());
    assert!(
        grad_input.iter().all(|&v| v == 0.0),
        "an index carries no derivative"
    );
    assert!(grad_input.is_standard_layout());
}

// Serial and parallel agreement

/// A gather above the parallel gate gives the same bits as the same gather below it
///
/// The whole batch clears the gate. 1 sample of it does not, so the per-sample lookups run on
/// the serial path. A gather copies rather than accumulates, so the 2 paths must agree exactly
#[test]
fn embedding_parallel_gather_matches_the_serial_gather() {
    let (samples, steps, output_dim) = (4usize, 1024usize, 1024usize);
    let work = samples * steps * output_dim;
    let gate = rustyml::tuning::elementwise::get_cheap_map_f32();
    assert!(work >= gate, "the whole batch must clear the gate");
    assert!(work / samples < gate, "1 sample must stay under the gate");

    let input_dim = 512;
    let indices: Vec<f32> = (0..samples * steps)
        .map(|k| ((k * 7919 + 13) % input_dim) as f32)
        .collect();
    let x = t2(samples, steps, indices.clone());

    let mut layer = Embedding::new(input_dim, output_dim)
        .unwrap()
        .with_random_state(31);
    let parallel = layer.forward(&x).unwrap();

    for sample in 0..samples {
        let row = t2(
            1,
            steps,
            indices[sample * steps..(sample + 1) * steps].to_vec(),
        );
        let serial = layer.forward(&row).unwrap();
        let expected = parallel
            .slice(ndarray::s![sample..sample + 1, .., ..])
            .to_owned()
            .into_dyn();
        assert_allclose(&serial, &expected, 0.0_f32);
    }
}

// Error paths

/// An index at or above the table height names no row
#[test]
fn embedding_rejects_an_index_outside_the_table() {
    let mut layer = Embedding::new(3, 2).unwrap();
    for bad in [3.0f32, 3.5, 100.0, -1.0, -2.5] {
        let x = Array1::from_vec(vec![0.0f32, bad]).into_dyn();
        assert!(
            matches!(layer.forward(&x), Err(Error::InvalidInput(_))),
            "the index {bad} must be rejected"
        );
    }
}

/// A non-finite index names no row either
#[test]
fn embedding_rejects_a_non_finite_index() {
    let mut layer = Embedding::new(3, 2).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = Array1::from_vec(vec![0.0f32, bad]).into_dyn();
        assert!(
            matches!(layer.forward(&x), Err(Error::InvalidInput(_))),
            "the index {bad} must be rejected"
        );
    }
}

/// An input with no element has nothing to look up
#[test]
fn embedding_rejects_an_empty_input() {
    let mut layer = Embedding::new(3, 2).unwrap();
    let x = t2(0, 4, Vec::new());
    assert!(matches!(layer.forward(&x), Err(Error::EmptyInput(_))));
}

/// A scalar input would give an output with no batch axis
#[test]
fn embedding_rejects_a_scalar_input() {
    let mut layer = Embedding::new(3, 2).unwrap();
    let x = Tensor::zeros(IxDyn(&[]));
    assert!(matches!(layer.forward(&x), Err(Error::InvalidInput(_))));
}

/// The backward pass needs the indices the forward pass read
#[test]
fn embedding_backward_before_forward_is_an_error() {
    let mut layer = Embedding::new(3, 2).unwrap();
    let grad = t2(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    assert!(matches!(
        layer.backward(&grad),
        Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
    ));
}

/// A gradient that is not the shape the layer produced is an error
#[test]
fn embedding_backward_checks_the_gradient_shape() {
    let mut layer = Embedding::new(4, 3).unwrap();
    let x = t2(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    layer.forward(&x).unwrap();

    let wrong = t3(2, 2, 4, vec![0.0; 16]);
    assert!(matches!(
        layer.backward(&wrong),
        Err(Error::ShapeMismatch { .. })
    ));
}

/// A table of the wrong shape never reaches the layer
#[test]
fn embedding_set_weights_checks_the_table_shape() {
    let mut layer = Embedding::new(4, 3).unwrap();
    let wrong = Array2::zeros((3, 4));
    assert!(matches!(
        layer.set_weights(wrong),
        Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
    ));
}

// Layer trait surface

/// The layer names itself for `summary()`, and reports its shape once a forward pass has run
#[test]
fn embedding_reports_its_type_and_output_shape() {
    let mut layer = Embedding::new(4, 3).unwrap();
    assert_eq!(layer.layer_type(), "Embedding");
    assert_eq!(layer.output_shape(), "Unknown");

    layer.forward(&t2(2, 5, vec![0.0; 10])).unwrap();
    assert_eq!(layer.output_shape(), "(None, 5, 3)");

    layer
        .forward(&Array1::from_vec(vec![0.0f32, 1.0]).into_dyn())
        .unwrap();
    assert_eq!(layer.output_shape(), "(None, 3)");
}

/// `get_weights` borrows the live table
#[test]
fn embedding_get_weights_returns_the_embedding_variant() {
    let layer = embedding_with_table(4, 3, reference_table());
    match layer.get_weights() {
        LayerWeight::Embedding(w) => {
            assert_eq!(w.embeddings.shape(), &[4, 3]);
            assert_eq!(w.embeddings[[0, 0]], -0.902);
            assert_eq!(w.embeddings[[3, 2]], 0.491);
        }
        other => panic!("expected the Embedding variant, got {other:?}"),
    }
}

/// The layer exposes no parameter until a backward pass has produced a gradient
#[test]
fn embedding_exposes_no_parameter_before_the_backward_pass() {
    let mut layer = Embedding::new(4, 3).unwrap();
    assert!(layer.parameters().is_empty());

    let x = t2(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    let out = layer.forward(&x).unwrap();
    assert!(
        layer.parameters().is_empty(),
        "a forward pass alone produces no gradient"
    );

    layer.backward(&Tensor::ones(out.raw_dim())).unwrap();
    assert_eq!(layer.parameters().len(), 1);
}

/// A seed makes the starting table reproducible, and the table stays inside the uniform range
#[test]
fn embedding_with_random_state_is_reproducible_and_bounded() {
    let first = Embedding::new(64, 8).unwrap().with_random_state(7);
    let second = Embedding::new(64, 8).unwrap().with_random_state(7);
    let third = Embedding::new(64, 8).unwrap().with_random_state(8);

    let (LayerWeight::Embedding(a), LayerWeight::Embedding(b), LayerWeight::Embedding(c)) = (
        first.get_weights(),
        second.get_weights(),
        third.get_weights(),
    ) else {
        panic!("every Embedding must report the Embedding variant");
    };

    assert_allclose(&*a.embeddings, &*b.embeddings, 0.0_f32);
    assert!(
        a.embeddings != c.embeddings,
        "a different seed must give a different table"
    );
    assert!(
        a.embeddings.iter().all(|v| v.abs() <= 0.05),
        "the table starts inside the uniform range of the Keras default"
    );
}

// Model integration

/// An embedding front end trains inside a model, and the loss falls
#[test]
fn embedding_trains_inside_a_sequential_model() {
    // 4 samples of 3 word indices each, drawn from a vocabulary of 6
    let x = t2(
        4,
        3,
        vec![1.0, 2.0, 3.0, 0.0, 5.0, 1.0, 4.0, 4.0, 2.0, 3.0, 1.0, 0.0],
    );
    let y = t2(4, 1, vec![1.0, -1.0, 0.5, -0.5]);

    let mut model = Sequential::new();
    model
        .add(Embedding::new(6, 4).unwrap().with_random_state(11))
        .add(Flatten::new(vec![4, 3, 4]).unwrap())
        .add(Dense::new(12, 1, Linear::new()).unwrap())
        .compile(
            SGD::new(0.05, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let history = model.fit(&x, &y, 40).unwrap();
    let losses = history.loss();
    assert!(
        losses[losses.len() - 1] < losses[0],
        "the loss must fall: {:?} then {:?}",
        losses[0],
        losses[losses.len() - 1]
    );
}

/// Training changes only the rows the batch selected
#[test]
fn embedding_training_leaves_an_unselected_row_untouched() {
    let x = t2(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let y = t2(2, 1, vec![1.0, -1.0]);

    let mut model = Sequential::new();
    model
        .add(Embedding::new(4, 2).unwrap().with_random_state(3))
        .add(Flatten::new(vec![2, 2, 2]).unwrap())
        .add(Dense::new(4, 1, Linear::new()).unwrap())
        .compile(
            SGD::new(0.1, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let before = match model.get_weights().remove(0) {
        LayerWeight::Embedding(w) => w.embeddings.into_owned(),
        _ => panic!("layer 0 must be the Embedding layer"),
    };
    model.fit(&x, &y, 5).unwrap();
    let after = match model.get_weights().remove(0) {
        LayerWeight::Embedding(w) => w.embeddings.into_owned(),
        _ => panic!("layer 0 must be the Embedding layer"),
    };

    // Rows 2 and 3 never appear in the batch, so plain SGD never moves them
    for row in [2usize, 3] {
        assert_eq!(after[[row, 0]], before[[row, 0]]);
        assert_eq!(after[[row, 1]], before[[row, 1]]);
    }
    assert!(
        (0..2).any(|row| after[[row, 0]] != before[[row, 0]]),
        "a selected row must move"
    );
}
