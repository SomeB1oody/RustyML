//! Integration tests for the Permute and RepeatVector layers.
//!
//! Covers axis order, value placement, memory layout, error paths, and round trips.
//! `gradient_check.rs` covers gradient values. This file does not duplicate them.

use ndarray::{Array2, Array3, Array4, Array5, IxDyn};
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::softmax::Softmax;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::permute::Permute;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::repeat_vector::RepeatVector;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::{Layer, LayerBase, UnaryLayer};
use rustyml::prelude::Activation;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// helpers

/// Build a 2D Tensor from row-major data
fn t2(a: usize, b: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((a, b), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 3D Tensor from row-major data
fn t3(a: usize, b: usize, c: usize, data: Vec<f32>) -> Tensor {
    Array3::from_shape_vec((a, b, c), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 4D Tensor from row-major data
fn t4(a: usize, b: usize, c: usize, d: usize, data: Vec<f32>) -> Tensor {
    Array4::from_shape_vec((a, b, c, d), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build the value list 1, 2, 3, ... with `count` entries, in C order
fn ramp(count: usize) -> Vec<f32> {
    (1..=count).map(|v| v as f32).collect()
}

/// Build a tensor of `1, 2, 3, ...` with the given shape, in C order
fn ramp_of(shape: &[usize]) -> Tensor {
    let count: usize = shape.iter().product();
    Tensor::from_shape_vec(IxDyn(shape), ramp(count)).expect("shape/data mismatch")
}

// Permute: the constructor

/// The constructor accepts any permutation of 1..=n at every rank the layer serves
#[test]
fn permute_new_accepts_every_permutation() {
    for dims in [
        vec![1],
        vec![2, 1],
        vec![1, 2],
        vec![3, 1, 2],
        vec![2, 1, 3],
    ] {
        assert!(
            Permute::new(dims.clone()).is_ok(),
            "dims {dims:?} must be accepted"
        );
    }
}

/// An empty dims names no axis, so the constructor rejects it
#[test]
fn permute_new_rejects_empty_dims() {
    let result = Permute::new(vec![]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for an empty dims, got {result:?}"
    );
}

/// An axis named twice is not a permutation
#[test]
fn permute_new_rejects_a_repeated_axis() {
    let result = Permute::new(vec![2, 2]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for a repeated axis, got {result:?}"
    );
}

/// dims counts from 1, so a 0 entry is out of range
#[test]
fn permute_new_rejects_zero_axis() {
    let result = Permute::new(vec![0, 1]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for a 0 entry, got {result:?}"
    );
}

/// An entry above dims.len() names an axis the input does not have
#[test]
fn permute_new_rejects_axis_above_range() {
    let result = Permute::new(vec![1, 3]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for an out-of-range entry, got {result:?}"
    );
}

// Permute: forward

/// Swapping the 2 axes after the batch axis transposes each sample
#[test]
fn permute_forward_swaps_two_axes_by_value() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    let x = t3(1, 2, 3, ramp(6));
    let out = p.forward_mut(&x, &mut Ctx::training()).unwrap();

    // Row-major [[1, 2, 3], [4, 5, 6]] transposes to [[1, 4], [2, 5], [3, 6]]
    let want = t3(1, 3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_allclose(&out, &want, 1e-6_f32);
}

/// A rotation moves the last axis to the front of the non-batch axes
#[test]
fn permute_forward_rotates_three_axes_by_value() {
    let mut p = Permute::new(vec![3, 1, 2]).unwrap();
    let x = t4(1, 2, 2, 2, ramp(8));
    let out = p.forward_mut(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    // out[0, a, b, c] reads x[0, b, c, a]
    let want = t4(1, 2, 2, 2, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
    assert_allclose(&out, &want, 1e-6_f32);
}

/// The identity permutation copies the input
#[test]
fn permute_forward_identity_copies_input() {
    let mut p = Permute::new(vec![1, 2, 3]).unwrap();
    let x = ramp_of(&[2, 3, 4, 2]);
    let out = p.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_allclose(&out, &x, 1e-6_f32);
}

/// A rank-5 input permutes on all 4 axes after the batch axis
#[test]
fn permute_forward_serves_rank_5() {
    let mut p = Permute::new(vec![4, 3, 2, 1]).unwrap();
    let x: Tensor = Array5::ones((2, 3, 4, 5, 6)).into_dyn();
    let out = p.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(out.shape(), &[2, 6, 5, 4, 3]);
}

/// The output is in C order, so a consumer can read it as 1 contiguous slice
#[test]
fn permute_output_is_in_c_order() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    let x = ramp_of(&[2, 3, 4]);
    let mut ctx = Ctx::training();

    let out = p.forward_mut(&x, &mut ctx).unwrap();
    assert!(
        out.is_standard_layout(),
        "the forward output must be in C order"
    );
    assert!(
        out.as_slice().is_some(),
        "the forward output must expose a contiguous slice"
    );

    let grad = ramp_of(&[2, 4, 3]);
    let grad_input = p.backward(&grad, &mut ctx).unwrap();
    assert!(
        grad_input.is_standard_layout(),
        "the backward output must be in C order"
    );
}

/// A Softmax after a Permute works, because the Permute output is in C order
#[test]
fn permute_feeds_softmax() {
    let x = ramp_of(&[2, 3, 4]);

    let mut model = SequentialBuilder::new()
        .add(Permute::new(vec![2, 1]).unwrap())
        .add(Softmax::new())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let out = model.predict(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4, 3]);
    // Softmax normalizes the last axis, so each group of 3 sums to 1
    for group in out.as_slice().unwrap().chunks(3) {
        let total: f32 = group.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "a softmax group sums to {total}"
        );
    }
}

/// A forward pass with an inference context matches one with a training context. A layer now
/// builds itself from the first tensor it receives no matter which context that pass runs in,
/// so `output_shape` is known right after the first pass
#[test]
fn permute_predict_matches_forward() {
    let x = ramp_of(&[2, 3, 4]);
    let mut p = Permute::new(vec![2, 1]).unwrap();

    let from_predict = p.forward_mut(&x, &mut Ctx::inference()).unwrap();
    assert_eq!(p.output_shape(), "(None, 4, 3)");

    let from_forward = p.forward(&x, &mut Ctx::training()).unwrap();
    assert_allclose(&from_predict, &from_forward, 1e-6_f32);
}

// Permute: backward

/// The backward pass applies the inverse order
#[test]
fn permute_backward_applies_the_inverse_order() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    let x = t3(1, 2, 3, ramp(6));
    let mut ctx = Ctx::training();
    p.forward_mut(&x, &mut ctx).unwrap();

    let grad = t3(1, 3, 2, ramp(6));
    let grad_input = p.backward(&grad, &mut ctx).unwrap();

    // [[1, 2], [3, 4], [5, 6]] transposes back to [[1, 3, 5], [2, 4, 6]]
    let want = t3(1, 2, 3, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    assert_allclose(&grad_input, &want, 1e-6_f32);
}

/// A rotation and its inverse recover the input
#[test]
fn permute_rotation_round_trips_through_its_inverse() {
    let x = ramp_of(&[2, 3, 4, 5]);

    let mut forward = Permute::new(vec![3, 1, 2]).unwrap();
    let rotated = forward.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(rotated.shape(), &[2, 5, 3, 4]);

    let mut back = Permute::new(vec![2, 3, 1]).unwrap();
    let restored = back.forward_mut(&rotated, &mut Ctx::training()).unwrap();
    assert_allclose(&restored, &x, 1e-6_f32);
}

// Permute: error paths

/// The input rank must be dims.len() + 1
#[test]
fn permute_forward_rejects_wrong_rank() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    for shape in [vec![2usize, 3], vec![2, 3, 4, 5]] {
        let result = p.forward_mut(&ramp_of(&shape), &mut Ctx::training());
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput for shape {shape:?}, got {result:?}"
        );
    }
}

/// forward rejects a tensor with a 0 extent
#[test]
fn permute_forward_rejects_empty_input() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    let empty: Tensor = Tensor::zeros(IxDyn(&[0, 3, 4]));
    assert!(
        matches!(
            p.forward_mut(&empty, &mut Ctx::training()),
            Err(Error::EmptyInput(_))
        ),
        "Permute must reject an empty input"
    );
}

/// backward before any forward pass reports ForwardPassNotRun
#[test]
fn permute_backward_before_forward_returns_err() {
    let p = Permute::new(vec![2, 1]).unwrap();
    let result = p.backward(&ramp_of(&[2, 4, 3]), &mut Ctx::training());
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {result:?}"
    );
}

/// backward rejects a gradient whose shape is not the forward output shape
#[test]
fn permute_backward_wrong_grad_shape_returns_err() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    let mut ctx = Ctx::training();
    p.forward_mut(&ramp_of(&[2, 3, 4]), &mut ctx).unwrap();

    let result = p.backward(&ramp_of(&[2, 3, 4]), &mut ctx);
    assert!(
        matches!(
            &result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == &[2, 4, 3] && found == &[2, 3, 4]
        ),
        "expected ShapeMismatch of [2, 4, 3] against [2, 3, 4], got {result:?}"
    );
}

// Permute: metadata and the free batch axis

/// layer_type names the layer, and the layer holds no parameters
#[test]
fn permute_metadata() {
    let p = Permute::new(vec![2, 1]).unwrap();
    assert_eq!(p.layer_type(), "Permute");
    assert_eq!(p.param_count(), ParamCounts::none());
    assert!(p.weights().is_empty(), "Permute must expose no weight");
}

/// output_shape is unknown before the first forward pass and resolved after it. The resolved
/// shape is the shape the layer built for, so the batch axis prints its real extent
#[test]
fn permute_output_shape_before_and_after_forward() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    assert_eq!(p.output_shape(), "Unknown");

    p.forward_mut(&ramp_of(&[2, 3, 4]), &mut Ctx::training())
        .unwrap();
    assert_eq!(p.output_shape(), "(None, 4, 3)");
}

/// 1 layer instance serves every batch size
#[test]
fn permute_one_instance_serves_every_batch_size() {
    let mut p = Permute::new(vec![2, 1]).unwrap();
    for batch in [1_usize, 4, 7] {
        let out = p
            .forward_mut(&ramp_of(&[batch, 3, 5]), &mut Ctx::training())
            .unwrap();
        assert_eq!(
            out.shape(),
            &[batch, 5, 3],
            "wrong output shape at batch size {batch}"
        );
    }
}

// RepeatVector: the constructor

/// A count of 0 leaves the output with no step, so the constructor rejects it
#[test]
fn repeat_vector_new_rejects_zero() {
    let result = RepeatVector::new(0);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for n = 0, got {result:?}"
    );
}

// RepeatVector: forward

/// Each feature vector appears at every step, unchanged
#[test]
fn repeat_vector_forward_repeats_each_row() {
    let mut r = RepeatVector::new(2).unwrap();
    let x = t2(2, 3, ramp(6));
    let out = r.forward_mut(&x, &mut Ctx::training()).unwrap();

    let want = t3(
        2,
        2,
        3,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
    );
    assert_allclose(&out, &want, 1e-6_f32);
}

/// A count of 1 adds the step axis and keeps every value
#[test]
fn repeat_vector_forward_with_one_step_adds_an_axis() {
    let mut r = RepeatVector::new(1).unwrap();
    let x = t2(2, 3, ramp(6));
    let out = r.forward_mut(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[2, 1, 3]);
    let want = t3(2, 1, 3, ramp(6));
    assert_allclose(&out, &want, 1e-6_f32);
}

/// A forward pass with an inference context matches one with a training context. A layer now
/// builds itself from the first tensor it receives no matter which context that pass runs in,
/// so `output_shape` is known right after the first pass
#[test]
fn repeat_vector_predict_matches_forward() {
    let x = t2(3, 4, ramp(12));
    let mut r = RepeatVector::new(5).unwrap();

    let from_predict = r.forward_mut(&x, &mut Ctx::inference()).unwrap();
    assert_eq!(r.output_shape(), "(None, 5, 4)");

    let from_forward = r.forward(&x, &mut Ctx::training()).unwrap();
    assert_allclose(&from_predict, &from_forward, 1e-6_f32);
}

// RepeatVector: backward

/// Every step reads the same input, so the input gradient is the sum over the step axis
#[test]
fn repeat_vector_backward_sums_over_the_step_axis() {
    let mut r = RepeatVector::new(2).unwrap();
    let x = t2(2, 3, ramp(6));
    let mut ctx = Ctx::training();
    r.forward_mut(&x, &mut ctx).unwrap();

    let grad = t3(2, 2, 3, ramp(12));
    let grad_input = r.backward(&grad, &mut ctx).unwrap();

    // Sample 0 sums [1, 2, 3] with [4, 5, 6]. Sample 1 sums [7, 8, 9] with [10, 11, 12]
    let want = t2(2, 3, vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);
    assert_allclose(&grad_input, &want, 1e-6_f32);
}

// RepeatVector: error paths

/// The input must be rank 2
#[test]
fn repeat_vector_forward_rejects_wrong_rank() {
    let mut r = RepeatVector::new(3).unwrap();
    for shape in [vec![5usize], vec![2, 3, 4]] {
        let result = r.forward_mut(&ramp_of(&shape), &mut Ctx::training());
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput for shape {shape:?}, got {result:?}"
        );
    }
}

/// forward rejects a tensor with a 0 extent
#[test]
fn repeat_vector_forward_rejects_empty_input() {
    let mut r = RepeatVector::new(3).unwrap();
    let empty: Tensor = Tensor::zeros(IxDyn(&[0, 4]));
    assert!(
        matches!(
            r.forward_mut(&empty, &mut Ctx::training()),
            Err(Error::EmptyInput(_))
        ),
        "RepeatVector must reject an empty input"
    );
}

/// backward before any forward pass reports ForwardPassNotRun
#[test]
fn repeat_vector_backward_before_forward_returns_err() {
    let r = RepeatVector::new(3).unwrap();
    let result = r.backward(&ramp_of(&[2, 3, 4]), &mut Ctx::training());
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {result:?}"
    );
}

/// backward rejects a gradient whose shape is not the forward output shape
#[test]
fn repeat_vector_backward_wrong_grad_shape_returns_err() {
    let mut r = RepeatVector::new(3).unwrap();
    let mut ctx = Ctx::training();
    r.forward_mut(&t2(2, 4, ramp(8)), &mut ctx).unwrap();

    let result = r.backward(&ramp_of(&[2, 2, 4]), &mut ctx);
    assert!(
        matches!(
            &result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == &[2, 3, 4] && found == &[2, 2, 4]
        ),
        "expected ShapeMismatch of [2, 3, 4] against [2, 2, 4], got {result:?}"
    );
}

// RepeatVector: metadata and the free batch axis

/// layer_type names the layer, and the layer holds no parameters
#[test]
fn repeat_vector_metadata() {
    let r = RepeatVector::new(3).unwrap();
    assert_eq!(r.layer_type(), "RepeatVector");
    assert_eq!(r.param_count(), ParamCounts::none());
    assert!(r.weights().is_empty(), "RepeatVector must expose no weight");
}

/// output_shape is unknown before the first forward pass and resolved after it. The resolved
/// shape is the shape the layer built for, so the batch axis prints its real extent
#[test]
fn repeat_vector_output_shape_before_and_after_forward() {
    let mut r = RepeatVector::new(3).unwrap();
    assert_eq!(r.output_shape(), "Unknown");

    r.forward_mut(&t2(2, 5, ramp(10)), &mut Ctx::training())
        .unwrap();
    assert_eq!(r.output_shape(), "(None, 3, 5)");
}

/// 1 layer instance serves every batch size
#[test]
fn repeat_vector_one_instance_serves_every_batch_size() {
    let mut r = RepeatVector::new(4).unwrap();
    for batch in [1_usize, 4, 7] {
        let out = r
            .forward_mut(&t2(batch, 3, ramp(batch * 3)), &mut Ctx::training())
            .unwrap();
        assert_eq!(
            out.shape(),
            &[batch, 4, 3],
            "wrong output shape at batch size {batch}"
        );
    }
}

// Inside a Sequential model

/// RepeatVector bridges 1 recurrent layer to the next, which is the encoder-decoder pattern
#[test]
fn repeat_vector_bridges_two_recurrent_layers() {
    // 4 samples, 6 steps, 3 features
    let x = ramp_of(&[4, 6, 3]).map(|v| v * 0.01);
    let y = t2(4, 2, vec![0.0, 1.0, 0.5, -0.5, 0.25, 0.75, -1.0, 0.0]);

    let mut model = SequentialBuilder::new()
        // The encoder returns its last state, a rank-2 [4, 5] tensor
        .add(LSTM::new(5, Activation::Tanh).unwrap().with_random_state(7))
        // The repeat turns that state into a 6-step sequence the decoder can read
        .add(RepeatVector::new(6).unwrap())
        .add(LSTM::new(4, Activation::Tanh).unwrap().with_random_state(9))
        .add(Dense::new(2, Linear::new()).unwrap().with_random_state(11))
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let history = model.fit(&x, &y, 5).unwrap();
    assert_eq!(history.loss().len(), 5);
    for (epoch, &loss) in history.loss().iter().enumerate() {
        assert!(loss.is_finite(), "loss at epoch {epoch} is not finite");
    }

    let out = model.predict(&x).unwrap();
    assert_eq!(out.shape(), &[4, 2]);
}

/// A Permute feeds a Dense head after it moves the axis the head reads
#[test]
fn permute_inside_sequential_model_trains() {
    // 4 samples, 3 steps, 6 features. The permute makes the step axis last
    let x = ramp_of(&[4, 3, 6]).map(|v| v * 0.01);
    let y = t2(4, 1, vec![0.0, 1.0, 0.5, -0.5]);

    let mut model = SequentialBuilder::new()
        .add(Permute::new(vec![2, 1]).unwrap())
        .add(rustyml::neural_network::layers::flatten::Flatten::new())
        .add(Dense::new(1, Linear::new()).unwrap().with_random_state(3))
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let history = model.fit(&x, &y, 5).unwrap();
    assert_eq!(history.loss().len(), 5);
    for (epoch, &loss) in history.loss().iter().enumerate() {
        assert!(loss.is_finite(), "loss at epoch {epoch} is not finite");
    }

    let out = model.predict(&x).unwrap();
    assert_eq!(out.shape(), &[4, 1]);
}
