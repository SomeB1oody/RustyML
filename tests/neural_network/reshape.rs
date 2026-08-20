//! Integration tests for the Reshape layer.
//!
//! Covers target-shape resolution, C-order value preservation, error paths, and round trips.
//! `gradient_check.rs` covers gradient values. This file does not duplicate them.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2, Array3, Array4, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::layers::reshape::Reshape;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// helpers

/// Build a 1D Tensor from a value list
fn t1(data: Vec<f32>) -> Tensor {
    Array::from_vec(data).into_dyn()
}

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

/// Build a 4D Tensor from row-major data
fn t4(a: usize, b: usize, c: usize, d: usize, data: Vec<f32>) -> Tensor {
    Array4::from_shape_vec((a, b, c, d), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build the value list 0, 1, 2, ... with `count` entries, in C order
fn ramp(count: usize) -> Vec<f32> {
    (0..count).map(|v| v as f32).collect()
}

// Reshape: forward shapes

/// A rank-2 input splits into a rank-3 output when the target names 2 axes
#[test]
fn reshape_forward_splits_rank_2_into_rank_3() {
    let mut r = Reshape::new(vec![2, 2]).unwrap();
    let x = t2(5, 4, ramp(20));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 2, 2]);
}

/// A rank-3 input merges into a rank-2 output when the target names 1 axis
#[test]
fn reshape_forward_merges_rank_3_into_rank_2() {
    let mut r = Reshape::new(vec![4]).unwrap();
    let x = t3(5, 2, 2, ramp(20));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 4]);
}

/// A leading -1 takes the extent that makes the element count match
#[test]
fn reshape_forward_infers_leading_axis() {
    let mut r = Reshape::new(vec![-1, 2]).unwrap();
    let x = t2(5, 4, ramp(20));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 2, 2]);
}

/// A trailing -1 takes the extent that makes the element count match
#[test]
fn reshape_forward_infers_trailing_axis() {
    let mut r = Reshape::new(vec![2, -1]).unwrap();
    let x = t2(5, 4, ramp(20));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 2, 2]);
}

/// A target of a single -1 folds every axis after the batch axis into 1 axis
#[test]
fn reshape_forward_single_inferred_axis_folds_all_axes() {
    let mut r = Reshape::new(vec![-1]).unwrap();
    let x = t3(5, 2, 3, ramp(30));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 6]);
}

/// Unit extents in the target add axes of size 1 to the output
#[test]
fn reshape_forward_adds_unit_axes() {
    let mut r = Reshape::new(vec![1, 1, 4]).unwrap();
    let x = t2(5, 4, ramp(20));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 1, 1, 4]);
}

/// An empty target gives the rank-1 output shape [batch]
#[test]
fn reshape_forward_empty_target_gives_rank_1_output() {
    let mut r = Reshape::new(vec![]).unwrap();
    let x = t2(5, 1, ramp(5));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5]);
}

/// A rank-1 input gains a trailing axis of size 1 when the target names 1 unit axis
#[test]
fn reshape_forward_rank_1_input_gains_trailing_axis() {
    let mut r = Reshape::new(vec![1]).unwrap();
    let x = t1(ramp(5));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[5, 1]);
}

// Reshape: values, not just shapes

/// A split from rank 2 to rank 3 keeps the flat element order, so the layer moves no data
#[test]
fn reshape_split_keeps_c_order_values() {
    let mut r = Reshape::new(vec![2, 3]).unwrap();
    let x = t2(2, 6, ramp(12));
    let out = r.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 2, 3]);
    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

/// A merge from rank 3 to rank 2 keeps the flat element order, so the layer moves no data
#[test]
fn reshape_merge_keeps_c_order_values() {
    let mut r = Reshape::new(vec![12]).unwrap();
    let x = t3(2, 3, 4, ramp(24));
    let out = r.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 12]);
    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// Reshape: predict

/// The layer has no mode-dependent behavior, so predict and forward give identical outputs
#[test]
fn reshape_predict_equals_forward() {
    let mut r = Reshape::new(vec![-1, 2]).unwrap();
    let x = t2(3, 4, ramp(12));

    let fwd = r.forward(&x).unwrap();
    let pred = r.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

/// predict needs no cache, so it works on a layer that never ran a forward pass
#[test]
fn reshape_predict_works_without_forward() {
    let r = Reshape::new(vec![2, 2]).unwrap();
    let x = t2(5, 4, ramp(20));
    let out = r.predict(&x).unwrap();

    assert_eq!(out.shape(), &[5, 2, 2]);
    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// Reshape: backward round trip

/// backward restores the input shape and keeps the flat element order of the gradient
#[test]
fn reshape_backward_restores_input_shape_and_values() {
    let mut r = Reshape::new(vec![2, 3]).unwrap();
    let x = t2(2, 6, ramp(12));
    let out = r.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2, 3]);

    let grad_output = t3(2, 2, 3, ramp(12).into_iter().map(|v| v * 2.0).collect());
    let grad_input = r.backward(&grad_output).unwrap();

    assert_eq!(grad_input.shape(), x.shape());
    let gs = grad_input.as_slice().expect("grad not contiguous");
    for (i, &val) in gs.iter().enumerate() {
        assert_abs_diff_eq!(val, (i as f32) * 2.0, epsilon = 1e-6);
    }
}

// Reshape: error paths

/// The constructor rejects a target that holds 2 entries of -1
#[test]
fn reshape_new_rejects_2_inferred_axes() {
    let result = Reshape::new(vec![-1, -1]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for 2 inferred axes, got {:?}",
        result
    );
}

/// The constructor rejects a 0 extent, which no non-empty input can match
#[test]
fn reshape_new_rejects_zero_extent() {
    let result = Reshape::new(vec![0, 4]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for a 0 extent, got {:?}",
        result
    );
}

/// The constructor rejects an extent below -1
#[test]
fn reshape_new_rejects_extent_below_minus_1() {
    let result = Reshape::new(vec![-2, 2]);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for an extent below -1, got {:?}",
        result
    );
}

/// forward reports ShapeMismatch with both shapes when the element count cannot match
#[test]
fn reshape_forward_element_count_mismatch_returns_shape_mismatch() {
    let mut r = Reshape::new(vec![2, 3]).unwrap();
    let x = t2(5, 4, ramp(20));
    let result = r.forward(&x);
    assert!(
        matches!(
            &result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == &[5, 2, 3] && found == &[5, 4]
        ),
        "expected ShapeMismatch of [5, 2, 3] against [5, 4], got {:?}",
        result
    );
}

/// forward rejects a rank-0 tensor, which carries no batch axis
#[test]
fn reshape_forward_rejects_rank_0_input() {
    let mut r = Reshape::new(vec![1]).unwrap();
    let x: Tensor = Tensor::zeros(IxDyn(&[]));
    let result = r.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for a rank-0 input, got {:?}",
        result
    );
}

/// forward rejects an empty tensor
#[test]
fn reshape_forward_rejects_empty_input() {
    let mut r = Reshape::new(vec![2, 2]).unwrap();
    let x: Tensor = Tensor::zeros(IxDyn(&[0, 4]));
    let result = r.forward(&x);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput for an empty input, got {:?}",
        result
    );
}

/// backward before any forward pass reports ForwardPassNotRun
#[test]
fn reshape_backward_before_forward_returns_err() {
    let mut r = Reshape::new(vec![2, 2]).unwrap();
    // No forward call happened yet, so the layer holds no input shape.
    let grad = t3(5, 2, 2, ramp(20));
    let result = r.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// backward rejects a gradient whose shape is not the forward output shape
#[test]
fn reshape_backward_wrong_grad_shape_returns_err() {
    let mut r = Reshape::new(vec![2, 2]).unwrap();
    let x = t2(5, 4, ramp(20));
    r.forward(&x).unwrap();

    // The forward output shape is [5, 2, 2]. This gradient keeps the input shape instead.
    let bad_grad = t2(5, 4, ramp(20));
    let result = r.backward(&bad_grad);
    assert!(
        matches!(
            &result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == &[5, 2, 2] && found == &[5, 4]
        ),
        "expected ShapeMismatch of [5, 2, 2] against [5, 4], got {:?}",
        result
    );
}

// Reshape: the batch axis is free

/// 1 layer instance serves every batch size, since the batch axis remains untouched
#[test]
fn reshape_one_instance_serves_every_batch_size() {
    let mut r = Reshape::new(vec![2, 3]).unwrap();
    for batch in [1_usize, 4, 7] {
        let x = t2(batch, 6, ramp(batch * 6));
        let out = r.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[batch, 2, 3],
            "wrong output shape at batch size {}",
            batch
        );
    }
}

// Reshape: equivalence with Flatten

/// Reshape with a single inferred axis gives the same output as Flatten on a rank-4 input
#[test]
fn reshape_single_inferred_axis_equals_flatten() {
    let x = t4(2, 2, 3, 4, ramp(48));

    let mut r = Reshape::new(vec![-1]).unwrap();
    let mut fl = Flatten::new(vec![2, 2, 3, 4]).unwrap();

    let reshaped = r.forward(&x).unwrap();
    let flattened = fl.forward(&x).unwrap();

    assert_eq!(reshaped.shape(), &[2, 24]);
    assert_allclose(&reshaped, &flattened, 1e-6_f32);
}

// Reshape: parameter count and layer type

/// The layer has no parameters, so param_count is NoTrainable
#[test]
fn reshape_param_count_is_no_trainable() {
    let r = Reshape::new(vec![2, 2]).unwrap();
    assert_eq!(r.param_count(), TrainingParameters::NoTrainable);
}

/// The layer has no parameters, so get_weights gives the Empty variant
#[test]
fn reshape_get_weights_is_empty() {
    let r = Reshape::new(vec![2, 2]).unwrap();
    assert!(
        matches!(r.get_weights(), LayerWeight::Empty),
        "Reshape must expose LayerWeight::Empty"
    );
}

/// layer_type names the layer
#[test]
fn reshape_layer_type_is_reshape() {
    let r = Reshape::new(vec![2, 2]).unwrap();
    assert_eq!(r.layer_type(), "Reshape");
}

// Reshape: output_shape

/// Before any forward pass, a target with no -1 already reports the full output shape
#[test]
fn reshape_output_shape_before_forward_without_inferred_axis() {
    let r = Reshape::new(vec![2, 2]).unwrap();
    assert_eq!(r.output_shape(), "(None, 2, 2)");

    // An empty target names no axis after the batch axis
    let empty = Reshape::new(vec![]).unwrap();
    assert_eq!(empty.output_shape(), "(None,)");
}

/// Before any forward pass, a target with a -1 reports Unknown
#[test]
fn reshape_output_shape_before_forward_with_inferred_axis() {
    let r = Reshape::new(vec![-1, 2]).unwrap();
    assert_eq!(r.output_shape(), "Unknown");
}

/// After a forward pass, output_shape reports the resolved shape
#[test]
fn reshape_output_shape_after_forward() {
    let mut r = Reshape::new(vec![-1, 2]).unwrap();
    let x = t2(5, 4, ramp(20));
    r.forward(&x).unwrap();
    assert_eq!(r.output_shape(), "(None, 2, 2)");
}

// Reshape: inside a Sequential model

/// Reshape feeds a Dense layer, and fit keeps the loss finite across every epoch
#[test]
fn reshape_inside_sequential_model_trains() {
    let x = t3(4, 2, 3, ramp(24).into_iter().map(|v| v * 0.1).collect());
    let y = t2(4, 1, vec![0.0, 1.0, 0.5, -0.5]);

    let mut model = Sequential::new();
    model
        .add(Reshape::new(vec![-1]).unwrap())
        .add(Dense::new(6, 1, Linear::new()).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let history = model.fit(&x, &y, 5).unwrap();
    assert_eq!(history.loss().len(), 5);
    for (epoch, &loss) in history.loss().iter().enumerate() {
        assert!(loss.is_finite(), "loss at epoch {} is not finite", epoch);
    }

    let out = model.predict(&x).unwrap();
    assert_eq!(out.shape(), &[4, 1]);
}
