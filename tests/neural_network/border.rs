//! Integration tests for the border layers: `ZeroPadding1D/2D/3D` and `Cropping1D/2D/3D`.
//!
//! Covers the constructor forms, value placement, error paths, and round trips.
//! `gradient_check.rs` covers gradient values. This file does not duplicate them.

use ndarray::{Array3, Array4, Array5, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::border::{
    Cropping1D, Cropping2D, Cropping3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D,
};
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// helpers

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

/// Build a 5D Tensor from row-major data
fn t5(a: usize, b: usize, c: usize, d: usize, e: usize, data: Vec<f32>) -> Tensor {
    Array5::from_shape_vec((a, b, c, d, e), data)
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

// Constructor forms: how a plain value expands into per-axis amounts

/// An integer gives the same amount at both ends of the step axis
#[test]
fn zero_padding_1d_integer_pads_both_ends_equally() {
    let mut pad = ZeroPadding1D::new(2);
    let out = pad.forward(&ramp_of(&[1, 3, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 7, 2]);
}

/// A pair names the amount before the first step and after the last
#[test]
fn zero_padding_1d_pair_names_each_end() {
    let mut pad = ZeroPadding1D::new((1, 3));
    let out = pad.forward(&ramp_of(&[1, 3, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 7, 2]);

    // The first step is padding and the last 3 are padding, so only step 1 holds the first row
    assert_eq!(out[[0, 0, 0]], 0.0);
    assert_eq!(out[[0, 1, 0]], 1.0);
    assert_eq!(out[[0, 3, 0]], 5.0);
    assert_eq!(out[[0, 4, 0]], 0.0);
}

/// An integer gives the same amount at all 4 edges
#[test]
fn zero_padding_2d_integer_pads_all_4_edges() {
    let mut pad = ZeroPadding2D::new(1);
    let out = pad.forward(&ramp_of(&[1, 3, 4, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 5, 6, 2]);
}

/// A pair gives 1 amount per axis, not the 2 ends of 1 axis
#[test]
fn zero_padding_2d_pair_gives_one_amount_per_axis() {
    let mut pad = ZeroPadding2D::new((1, 2));
    let out = pad.forward(&ramp_of(&[1, 3, 4, 2])).unwrap();

    // Height grows by 1 at each end and width by 2 at each end
    assert_eq!(out.shape(), &[1, 5, 8, 2]);
}

/// A pair of pairs names all 4 edges on its own
#[test]
fn zero_padding_2d_pair_of_pairs_names_all_4_edges() {
    let mut pad = ZeroPadding2D::new(((1, 0), (0, 2)));
    let out = pad.forward(&ramp_of(&[1, 3, 4, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 4, 6, 2]);
}

/// An integer gives the same amount at all 6 faces
#[test]
fn zero_padding_3d_integer_pads_all_6_faces() {
    let mut pad = ZeroPadding3D::new(1);
    let out = pad.forward(&ramp_of(&[1, 2, 3, 4, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 4, 5, 6, 2]);
}

/// A triple gives 1 amount per axis, not the ends of 1 axis
#[test]
fn zero_padding_3d_triple_gives_one_amount_per_axis() {
    let mut pad = ZeroPadding3D::new((1, 2, 0));
    let out = pad.forward(&ramp_of(&[1, 2, 3, 4, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 4, 7, 4, 2]);
}

/// A triple of pairs names all 6 faces on its own
#[test]
fn zero_padding_3d_triple_of_pairs_names_all_6_faces() {
    let mut pad = ZeroPadding3D::new(((1, 0), (0, 2), (1, 1)));
    let out = pad.forward(&ramp_of(&[1, 2, 3, 4, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 3, 5, 6, 2]);
}

/// A cropping pair gives 1 amount per axis, the same as the padding pair
#[test]
fn cropping_2d_pair_gives_one_amount_per_axis() {
    let mut crop = Cropping2D::new((1, 2));
    let out = crop.forward(&ramp_of(&[1, 5, 8, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 3, 4, 2]);
}

/// A cropping triple gives 1 amount per axis, the same as the padding triple
#[test]
fn cropping_3d_triple_gives_one_amount_per_axis() {
    let mut crop = Cropping3D::new((1, 2, 0));
    let out = crop.forward(&ramp_of(&[1, 4, 7, 4, 2])).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3, 4, 2]);
}

// Forward values

/// The padded steps hold zeros and the original steps keep their values and their order
#[test]
fn zero_padding_1d_forward_places_values_and_zeros() {
    let mut pad = ZeroPadding1D::new((1, 2));
    let x = t3(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = pad.forward(&x).unwrap();

    let want = t3(
        1,
        5,
        2,
        vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert_allclose(&out, &want, 1e-6_f32);
}

/// A 1-row top border and a 1-column right border leave the input in the other corner
#[test]
fn zero_padding_2d_forward_places_values_and_zeros() {
    let mut pad = ZeroPadding2D::new(((1, 0), (0, 1)));
    let x = t4(1, 2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let out = pad.forward(&x).unwrap();

    let want = t4(
        1,
        3,
        3,
        1,
        vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
    );
    assert_allclose(&out, &want, 1e-6_f32);
}

/// A 3D pad leaves every original value in place and writes 0 in every new position
#[test]
fn zero_padding_3d_forward_places_values_and_zeros() {
    let mut pad = ZeroPadding3D::new(((1, 0), (0, 1), (0, 0)));
    let x = t5(1, 1, 1, 2, 1, vec![1.0, 2.0]);
    let out = pad.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2, 1]);
    let want = t5(1, 2, 2, 2, 1, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
    assert_allclose(&out, &want, 1e-6_f32);
}

/// The cropped output keeps the interior steps in their original order
#[test]
fn cropping_1d_forward_keeps_interior() {
    let mut crop = Cropping1D::new((1, 2));
    let x = t3(1, 6, 1, ramp(6));
    let out = crop.forward(&x).unwrap();

    let want = t3(1, 3, 1, vec![2.0, 3.0, 4.0]);
    assert_allclose(&out, &want, 1e-6_f32);
}

/// The cropped output keeps the interior rows and columns
#[test]
fn cropping_2d_forward_keeps_interior() {
    let mut crop = Cropping2D::new(((1, 0), (0, 1)));
    let x = t4(1, 3, 3, 1, ramp(9));
    let out = crop.forward(&x).unwrap();

    let want = t4(1, 2, 2, 1, vec![4.0, 5.0, 7.0, 8.0]);
    assert_allclose(&out, &want, 1e-6_f32);
}

/// The cropped output keeps the interior of all 3 spatial axes
#[test]
fn cropping_3d_forward_keeps_interior() {
    let mut crop = Cropping3D::new(1);
    let x = t5(1, 3, 3, 3, 1, ramp(27));
    let out = crop.forward(&x).unwrap();

    // Only the center position of a 3x3x3 volume survives a 1-plane crop at every face
    let want = t5(1, 1, 1, 1, 1, vec![14.0]);
    assert_allclose(&out, &want, 1e-6_f32);
}

/// A border of 0 at every end copies the input
#[test]
fn zero_padding_2d_with_zero_amounts_copies_input() {
    let mut pad = ZeroPadding2D::new(0);
    let x = ramp_of(&[2, 3, 4, 2]);
    let out = pad.forward(&x).unwrap();
    assert_allclose(&out, &x, 1e-6_f32);
}

/// A border of 0 at every end copies the input
#[test]
fn cropping_2d_with_zero_amounts_copies_input() {
    let mut crop = Cropping2D::new(0);
    let x = ramp_of(&[2, 3, 4, 2]);
    let out = crop.forward(&x).unwrap();
    assert_allclose(&out, &x, 1e-6_f32);
}

/// predict gives the same output as forward, and writes no cache
#[test]
fn predict_matches_forward_and_leaves_output_shape_unknown() {
    let x = ramp_of(&[2, 4, 4, 3]);

    let mut pad = ZeroPadding2D::new(((0, 2), (1, 0)));
    let from_predict = pad.predict(&x).unwrap();
    assert_eq!(pad.output_shape(), "Unknown");
    let from_forward = pad.forward(&x).unwrap();
    assert_allclose(&from_predict, &from_forward, 1e-6_f32);

    let mut crop = Cropping2D::new(((0, 2), (1, 0)));
    let from_predict = crop.predict(&x).unwrap();
    assert_eq!(crop.output_shape(), "Unknown");
    let from_forward = crop.forward(&x).unwrap();
    assert_allclose(&from_predict, &from_forward, 1e-6_f32);
}

// Backward

/// The backward pass of a zero-padding layer returns the interior of the gradient
#[test]
fn zero_padding_2d_backward_returns_interior_of_gradient() {
    let mut pad = ZeroPadding2D::new(1);
    let x = t4(1, 2, 2, 1, ramp(4));
    pad.forward(&x).unwrap();

    // The output is 4x4, and the interior 2x2 holds 6, 7, 10 and 11 of a 1-based ramp
    let grad = t4(1, 4, 4, 1, ramp(16));
    let grad_input = pad.backward(&grad).unwrap();

    let want = t4(1, 2, 2, 1, vec![6.0, 7.0, 10.0, 11.0]);
    assert_allclose(&grad_input, &want, 1e-6_f32);
}

/// The backward pass of a cropping layer writes 0 in every position it removed
#[test]
fn cropping_2d_backward_scatters_gradient_and_zeros_border() {
    let mut crop = Cropping2D::new(((1, 0), (0, 1)));
    let x = t4(1, 3, 3, 1, ramp(9));
    crop.forward(&x).unwrap();

    let grad = t4(1, 2, 2, 1, ramp(4));
    let grad_input = crop.backward(&grad).unwrap();

    let want = t4(
        1,
        3,
        3,
        1,
        vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
    );
    assert_allclose(&grad_input, &want, 1e-6_f32);
}

/// A crop with the same border undoes a pad, so the pair is an identity on values
#[test]
fn pad_then_crop_restores_the_input() {
    let x = ramp_of(&[2, 3, 4, 2]);

    let mut pad = ZeroPadding2D::new(((2, 1), (0, 3)));
    let mut crop = Cropping2D::new(((2, 1), (0, 3)));

    let padded = pad.forward(&x).unwrap();
    assert_eq!(padded.shape(), &[2, 6, 7, 2]);

    let restored = crop.forward(&padded).unwrap();
    assert_allclose(&restored, &x, 1e-6_f32);
}

// Error paths

/// A zero-padding layer rejects an input whose rank is not the rank it serves
#[test]
fn zero_padding_rejects_wrong_rank() {
    let mut pad_1d = ZeroPadding1D::new(1);
    assert!(
        matches!(
            pad_1d.forward(&ramp_of(&[2, 3, 4, 2])),
            Err(Error::InvalidInput(_))
        ),
        "ZeroPadding1D must reject a rank-4 input"
    );

    let mut pad_2d = ZeroPadding2D::new(1);
    assert!(
        matches!(
            pad_2d.forward(&ramp_of(&[2, 3, 4])),
            Err(Error::InvalidInput(_))
        ),
        "ZeroPadding2D must reject a rank-3 input"
    );

    let mut pad_3d = ZeroPadding3D::new(1);
    assert!(
        matches!(
            pad_3d.forward(&ramp_of(&[2, 3, 4, 2])),
            Err(Error::InvalidInput(_))
        ),
        "ZeroPadding3D must reject a rank-4 input"
    );
}

/// A cropping layer rejects an input whose rank is not the rank it serves
#[test]
fn cropping_rejects_wrong_rank() {
    let mut crop_1d = Cropping1D::new(1);
    assert!(
        matches!(
            crop_1d.forward(&ramp_of(&[2, 5, 5, 2])),
            Err(Error::InvalidInput(_))
        ),
        "Cropping1D must reject a rank-4 input"
    );

    let mut crop_2d = Cropping2D::new(1);
    assert!(
        matches!(
            crop_2d.forward(&ramp_of(&[2, 5, 2])),
            Err(Error::InvalidInput(_))
        ),
        "Cropping2D must reject a rank-3 input"
    );

    let mut crop_3d = Cropping3D::new(1);
    assert!(
        matches!(
            crop_3d.forward(&ramp_of(&[2, 5, 5, 2])),
            Err(Error::InvalidInput(_))
        ),
        "Cropping3D must reject a rank-4 input"
    );
}

/// Both families reject a tensor with a 0 extent
#[test]
fn border_layers_reject_empty_input() {
    let empty: Tensor = Tensor::zeros(IxDyn(&[0, 4, 4, 3]));

    let mut pad = ZeroPadding2D::new(1);
    assert!(
        matches!(pad.forward(&empty), Err(Error::EmptyInput(_))),
        "ZeroPadding2D must reject an empty input"
    );

    let mut crop = Cropping2D::new(1);
    assert!(
        matches!(crop.forward(&empty), Err(Error::EmptyInput(_))),
        "Cropping2D must reject an empty input"
    );
}

/// A crop must leave at least 1 position on each axis, so an exact match of the extent fails
#[test]
fn cropping_rejects_a_border_that_removes_the_whole_axis() {
    let x = t3(1, 5, 1, ramp(5));

    // 2 + 3 removes all 5 steps
    let mut too_much = Cropping1D::new((2, 3));
    let result = too_much.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput when the crop leaves no step, got {result:?}"
    );

    // 2 + 2 leaves exactly 1 step, which is allowed
    let mut just_fits = Cropping1D::new((2, 2));
    let out = just_fits.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1]);
    assert_eq!(out[[0, 0, 0]], 3.0);
}

/// backward before any forward pass reports ForwardPassNotRun
#[test]
fn border_layers_backward_before_forward_returns_err() {
    let mut pad = ZeroPadding2D::new(1);
    let result = pad.backward(&ramp_of(&[1, 4, 4, 1]));
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun from ZeroPadding2D, got {result:?}"
    );

    let mut crop = Cropping2D::new(1);
    let result = crop.backward(&ramp_of(&[1, 2, 2, 1]));
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun from Cropping2D, got {result:?}"
    );
}

/// backward rejects a gradient whose shape is not the forward output shape
#[test]
fn border_layers_backward_wrong_grad_shape_returns_err() {
    let mut pad = ZeroPadding2D::new(1);
    pad.forward(&ramp_of(&[1, 2, 2, 1])).unwrap();
    let result = pad.backward(&ramp_of(&[1, 2, 2, 1]));
    assert!(
        matches!(
            &result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == &[1, 4, 4, 1] && found == &[1, 2, 2, 1]
        ),
        "expected ShapeMismatch of [1, 4, 4, 1] against [1, 2, 2, 1], got {result:?}"
    );

    let mut crop = Cropping2D::new(1);
    crop.forward(&ramp_of(&[1, 4, 4, 1])).unwrap();
    let result = crop.backward(&ramp_of(&[1, 4, 4, 1]));
    assert!(
        matches!(
            &result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == &[1, 2, 2, 1] && found == &[1, 4, 4, 1]
        ),
        "expected ShapeMismatch of [1, 2, 2, 1] against [1, 4, 4, 1], got {result:?}"
    );
}

// Layer metadata

/// layer_type names each layer
#[test]
fn border_layer_types_are_named() {
    assert_eq!(ZeroPadding1D::new(1).layer_type(), "ZeroPadding1D");
    assert_eq!(ZeroPadding2D::new(1).layer_type(), "ZeroPadding2D");
    assert_eq!(ZeroPadding3D::new(1).layer_type(), "ZeroPadding3D");
    assert_eq!(Cropping1D::new(1).layer_type(), "Cropping1D");
    assert_eq!(Cropping2D::new(1).layer_type(), "Cropping2D");
    assert_eq!(Cropping3D::new(1).layer_type(), "Cropping3D");
}

/// No border layer has parameters, so param_count is NoTrainable
#[test]
fn border_layers_have_no_trainable_parameters() {
    assert_eq!(
        ZeroPadding2D::new(1).param_count(),
        TrainingParameters::NoTrainable
    );
    assert_eq!(
        Cropping3D::new(1).param_count(),
        TrainingParameters::NoTrainable
    );
}

/// No border layer has weights, so get_weights gives the Empty variant
#[test]
fn border_layers_expose_empty_weights() {
    assert!(
        matches!(ZeroPadding1D::new(1).get_weights(), LayerWeight::Empty),
        "ZeroPadding1D must expose LayerWeight::Empty"
    );
    assert!(
        matches!(Cropping1D::new(1).get_weights(), LayerWeight::Empty),
        "Cropping1D must expose LayerWeight::Empty"
    );
}

/// Before any forward pass the output shape is unknown, because the input shape is unknown
#[test]
fn border_output_shape_is_unknown_before_forward() {
    assert_eq!(ZeroPadding2D::new(1).output_shape(), "Unknown");
    assert_eq!(Cropping2D::new(1).output_shape(), "Unknown");
}

/// After a forward pass, output_shape reports the shape with the batch axis as "None"
#[test]
fn border_output_shape_after_forward() {
    let mut pad = ZeroPadding2D::new(((1, 2), (0, 1)));
    pad.forward(&ramp_of(&[2, 3, 4, 5])).unwrap();
    assert_eq!(pad.output_shape(), "(None, 6, 5, 5)");

    let mut crop = Cropping1D::new((1, 2));
    crop.forward(&ramp_of(&[2, 8, 3])).unwrap();
    assert_eq!(crop.output_shape(), "(None, 5, 3)");
}

// The batch axis is free

/// 1 layer instance serves every batch size, since the batch axis passes through untouched
#[test]
fn one_instance_serves_every_batch_size() {
    let mut pad = ZeroPadding2D::new(1);
    let mut crop = Cropping2D::new(1);
    for batch in [1_usize, 4, 7] {
        let x = ramp_of(&[batch, 3, 3, 2]);

        let padded = pad.forward(&x).unwrap();
        assert_eq!(
            padded.shape(),
            &[batch, 5, 5, 2],
            "wrong padded shape at batch size {batch}"
        );

        let cropped = crop.forward(&x).unwrap();
        assert_eq!(
            cropped.shape(),
            &[batch, 1, 1, 2],
            "wrong cropped shape at batch size {batch}"
        );
    }
}

// Inside a Sequential model

/// A pad, a valid convolution, and a crop stack into a model whose loss stays finite
#[test]
fn border_layers_inside_sequential_model_train() {
    let x = ramp_of(&[4, 4, 4, 1]).map(|v| v * 0.05);
    let y = t4(4, 1, 1, 1, vec![0.0, 1.0, 0.5, -0.5])
        .to_shape(IxDyn(&[4, 1]))
        .unwrap()
        .to_owned();

    let mut model = Sequential::new();
    model
        // 4x4 grows to 6x6, the 3x3 valid convolution brings it back to 4x4, and the crop
        // trims it to 2x2
        .add(ZeroPadding2D::new(1))
        .add(Conv2D::new(2, (3, 3), vec![4, 6, 6, 1], (1, 1), Linear::new()).unwrap())
        .add(Cropping2D::new(1))
        .add(Flatten::new(vec![4, 2, 2, 2]).unwrap())
        .add(Dense::new(8, 1, Linear::new()).unwrap())
        .compile(
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
