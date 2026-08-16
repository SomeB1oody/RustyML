//! Integration tests for the UnitNormalization layer.
//!
//! Covers the forward values, the axis forms, the row and strided paths, the near-zero cap,
//! memory layout, and the error paths. `gradient_check.rs` covers the gradient against finite
//! differences. This file pins the gradient against reference values instead.

use ndarray::{Array2, Array3, Array4, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::layers::regularization::normalization::unit_normalization::{
    UnitNormalization, UnitNormalizationAxis,
};
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
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

/// A layer on the default axis, which is the last one
fn last_axis() -> UnitNormalization {
    UnitNormalization::new(UnitNormalizationAxis::Default).expect("Default axis is always valid")
}

/// The upstream gradient `gradient_check.rs` uses, so the pinned reference tables below were
/// produced from the same weights
fn upstream(like: &Tensor) -> Tensor {
    let flat: Vec<f32> = (0..like.len())
        .map(|k| 1.0 + 0.1 * ((k % 7) as f32 - 3.0))
        .collect();
    Tensor::from_shape_vec(like.raw_dim(), flat).expect("same length as the tensor it copies")
}

/// A ramp of distinct values, in the given shape
fn ramp(shape: &[usize], offset: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|k| k as f32 - offset).collect();
    Tensor::from_shape_vec(IxDyn(shape), data).expect("product of the shape")
}

// forward values, pinned against reference values

/// Hand-derived values on the 2 best known right triangles
///
/// 3-4-5 gives 0.6 and 0.8, and 5-12-13 gives 5/13 and 12/13. Neither number can come out of a
/// wrong axis or a wrong reduction by accident
#[test]
fn unit_normalization_forward_hand_derived() {
    let x = t2(2, 3, vec![3.0, 4.0, 0.0, 0.0, 5.0, 12.0]);
    let out = last_axis().predict(&x).unwrap();

    let want = t2(2, 3, vec![0.6, 0.8, 0.0, 0.0, 5.0 / 13.0, 12.0 / 13.0]);
    assert_allclose(&out, &want, 1e-6);
}

/// Every group leaves the layer with a length of 1
#[test]
fn unit_normalization_gives_every_group_a_length_of_1() {
    let x = ramp(&[4, 5], 7.0);
    let out = last_axis().predict(&x).unwrap();

    for row in 0..4 {
        let length: f32 = (0..5).map(|c| out[[row, c]] * out[[row, c]]).sum();
        assert!(
            (length - 1.0).abs() < 1e-6,
            "row {row} has a length of {length}"
        );
    }
}

/// The rank-2 forward and backward values are pinned against reference values
///
/// The formula `x * minimum(rsqrt(sum(x^2)), 1e12)`, differentiated, produced the pinned
/// values below.
#[test]
fn unit_normalization_matches_reference_on_the_last_axis() {
    let x = t2(2, 3, vec![1.0, 2.0, 3.0, -4.0, 0.0, 3.0]);
    let mut layer = last_axis();

    let out = layer.forward(&x).unwrap();
    let want = t2(2, 3, vec![0.2672612, 0.5345225, 0.8017837, -0.8, 0.0, 0.6]);
    assert_allclose(&out, &want, 1e-6);

    let grad = layer.backward(&upstream(&out)).unwrap();
    let want_grad = t2(
        2,
        3,
        vec![0.09163242, 0.02290812, -0.04581621, 0.1872, 0.22, 0.2496],
    );
    assert_allclose(&grad, &want_grad, 1e-6);
}

/// A middle axis leaves the groups as strided lanes, which is the layer's other path
#[test]
fn unit_normalization_matches_reference_on_a_middle_axis() {
    let x = ramp(&[2, 3, 4], 11.0);
    let mut layer = UnitNormalization::new(UnitNormalizationAxis::Custom(1)).unwrap();

    let out = layer.forward(&x).unwrap();
    let want = t3(
        2,
        3,
        4,
        vec![
            -0.8221786,
            -0.8451542,
            -0.8700628,
            -0.8944272,
            -0.5232046,
            -0.5070925,
            -0.4833682,
            -0.4472136,
            -0.2242305,
            -0.1690308,
            -0.09667365,
            0.0,
            0.09667365,
            0.1690308,
            0.2242305,
            0.2672612,
            0.4833682,
            0.5070925,
            0.5232046,
            0.5345225,
            0.8700628,
            0.8451542,
            0.8221786,
            0.8017837,
        ],
    );
    assert_allclose(&out, &want, 1e-6);

    let grad = layer.backward(&upstream(&out)).unwrap();
    let want_grad = t3(
        2,
        3,
        4,
        vec![
            -0.02943808,
            -0.03501353,
            -0.039844,
            -0.008944273,
            0.0301897,
            0.03984299,
            0.05520336,
            0.01788854,
            0.03749703,
            0.05553871,
            0.08257917,
            0.1229837,
            0.1002876,
            0.09103518,
            0.02902052,
            0.02529437,
            0.008402474,
            0.02801082,
            0.02785136,
            0.02386262,
            -0.0158111,
            -0.03501354,
            -0.02563827,
            -0.02433986,
        ],
    );
    assert_allclose(&grad, &want_grad, 1e-6);
}

/// 2 axes that are not next to each other still form 1 group each
#[test]
fn unit_normalization_matches_reference_on_2_separated_axes() {
    let x = ramp(&[2, 3, 4], 11.0);
    let mut layer = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![0, 2])).unwrap();

    let out = layer.forward(&x).unwrap();
    let want = t3(
        2,
        3,
        4,
        vec![
            -0.5527708,
            -0.5025189,
            -0.452267,
            -0.4020151,
            -0.4041452,
            -0.3464102,
            -0.2886751,
            -0.2309401,
            -0.1398757,
            -0.09325048,
            -0.04662524,
            0.0,
            0.05025189,
            0.1005038,
            0.1507557,
            0.2010076,
            0.2886751,
            0.3464102,
            0.4041452,
            0.4618802,
            0.4196272,
            0.4662524,
            0.5128776,
            0.5595029,
        ],
    );
    assert_allclose(&out, &want, 1e-6);

    let grad = layer.backward(&upstream(&out)).unwrap();
    let want_grad = t3(
        2,
        3,
        4,
        vec![
            0.003489718,
            0.01139551,
            0.0193013,
            0.02720709,
            0.06835827,
            0.07343896,
            0.07851963,
            0.0431858,
            0.04736517,
            0.0486727,
            0.04998023,
            0.05128777,
            0.06318287,
            0.07108866,
            0.04381812,
            0.05172391,
            0.04849742,
            0.0535781,
            0.05865879,
            0.06373947,
            0.0304179,
            -0.0009122305,
            0.0003953055,
            0.001702838,
        ],
    );
    assert_allclose(&grad, &want_grad, 1e-6);
}

// axis forms

/// The 3 ways of naming the last axis of a rank-3 input are the same layer
#[test]
fn unit_normalization_axis_spellings_agree_on_the_last_axis() {
    let x = ramp(&[2, 3, 4], 5.0);

    let by_default = last_axis().predict(&x).unwrap();
    let by_custom = UnitNormalization::new(UnitNormalizationAxis::Custom(2))
        .unwrap()
        .predict(&x)
        .unwrap();
    let by_list = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![2]))
        .unwrap()
        .predict(&x)
        .unwrap();

    assert_eq!(by_default, by_custom);
    assert_eq!(by_default, by_list);
}

/// An axis list names a set, so its order cannot change the result
#[test]
fn unit_normalization_axis_list_order_does_not_matter() {
    let x = ramp(&[2, 3, 4], 5.0);

    let ascending = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![0, 2]))
        .unwrap()
        .predict(&x)
        .unwrap();
    let descending = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![2, 0]))
        .unwrap()
        .predict(&x)
        .unwrap();

    assert_eq!(ascending, descending);
}

/// Naming every axis normalizes the whole tensor as 1 group
#[test]
fn unit_normalization_over_every_axis_makes_1_group() {
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![0, 1]))
        .unwrap()
        .predict(&x)
        .unwrap();

    // sqrt(1 + 4 + 9 + 16) = sqrt(30)
    let norm = 30.0f32.sqrt();
    let want = t2(2, 2, vec![1.0 / norm, 2.0 / norm, 3.0 / norm, 4.0 / norm]);
    assert_allclose(&out, &want, 1e-6);

    let length: f32 = out.iter().map(|v| v * v).sum();
    assert!((length - 1.0).abs() < 1e-6);
}

/// The batch axis is a legal choice, and it makes each column a unit vector
#[test]
fn unit_normalization_on_the_batch_axis_normalizes_columns() {
    let x = t2(2, 2, vec![3.0, 5.0, 4.0, 12.0]);
    let out = UnitNormalization::new(UnitNormalizationAxis::Custom(0))
        .unwrap()
        .predict(&x)
        .unwrap();

    // Column 0 is the 3-4-5 triangle, and column 1 is the 5-12-13 triangle
    let want = t2(2, 2, vec![0.6, 5.0 / 13.0, 0.8, 12.0 / 13.0]);
    assert_allclose(&out, &want, 1e-6);
}

/// The input sets the rank the layer serves, not the constructor
#[test]
fn unit_normalization_serves_every_rank_from_2_up() {
    for shape in [
        vec![2, 3],
        vec![2, 3, 4],
        vec![2, 3, 4, 5],
        vec![2, 2, 2, 2, 2],
    ] {
        let x = ramp(&shape, 3.0);
        let out = last_axis().predict(&x).unwrap();
        assert_eq!(out.shape(), shape.as_slice());

        let group_len = *shape.last().unwrap();
        for group in out.as_slice().unwrap().chunks_exact(group_len) {
            let length: f32 = group.iter().map(|v| v * v).sum();
            assert!(
                (length - 1.0).abs() < 1e-5,
                "a group of rank-{} input has a length of {length}",
                shape.len()
            );
        }
    }
}

// properties

/// Scaling the input leaves the output alone, because only the direction survives
#[test]
fn unit_normalization_ignores_the_length_of_its_input() {
    let x = ramp(&[3, 6], 8.0);
    let scaled = x.mapv(|v| v * 137.0);

    let plain = last_axis().predict(&x).unwrap();
    let stretched = last_axis().predict(&scaled).unwrap();
    assert_allclose(&plain, &stretched, 1e-5);
}

/// A second pass changes nothing, since the groups already have a length of 1
#[test]
fn unit_normalization_is_idempotent() {
    let x = ramp(&[3, 6], 8.0);
    let once = last_axis().predict(&x).unwrap();
    let twice = last_axis().predict(&once).unwrap();
    assert_allclose(&once, &twice, 1e-6);
}

/// The gradient of an unclamped group is orthogonal to that group's output
///
/// Moving along the output direction only changes the length, which the layer discards, so that
/// component of the gradient must be 0
#[test]
fn unit_normalization_gradient_is_orthogonal_to_its_output() {
    let x = ramp(&[4, 5], 9.0);
    let mut layer = last_axis();
    let out = layer.forward(&x).unwrap();
    let grad = layer.backward(&upstream(&out)).unwrap();

    for row in 0..4 {
        let projection: f32 = (0..5).map(|c| grad[[row, c]] * out[[row, c]]).sum();
        assert!(
            projection.abs() < 1e-5,
            "row {row} keeps a component along its own output: {projection}"
        );
    }
}

// the near-zero cap

/// An all-zero group comes back all zero instead of dividing by zero
#[test]
fn unit_normalization_leaves_an_all_zero_group_at_zero() {
    let x = t2(2, 3, vec![0.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    let mut layer = last_axis();

    let out = layer.forward(&x).unwrap();
    let want = t2(2, 3, vec![0.0, 0.0, 0.0, 0.6, 0.8, 0.0]);
    assert_allclose(&out, &want, 1e-6);

    // The cap scaled the first group, so its gradient is the cap itself. Without the cap, the
    // derivative of the reciprocal square root would overflow to NaN here
    let grad = layer.backward(&upstream(&out)).unwrap();
    for c in 0..3 {
        let want = 1e12 * (1.0 + 0.1 * (c as f32 - 3.0));
        assert!(
            (grad[[0, c]] / want - 1.0).abs() < 1e-6,
            "capped gradient at column {c} is {}",
            grad[[0, c]]
        );
    }
    // The second group is far from the cap, so the pinned reference values still apply
    assert!((grad[[1, 0]] - 0.02239999).abs() < 1e-6);
    assert!((grad[[1, 1]] + 0.0168).abs() < 1e-6);
    assert!((grad[[1, 2]] - 0.24).abs() < 1e-6);
}

/// The cap scales a group below the cap boundary, and the group's own norm scales one above it
///
/// The boundary sits where the norm is `1e-12`, which is a sum of squares of `1e-24`
#[test]
fn unit_normalization_cap_takes_over_below_a_norm_of_1e_minus_12() {
    // Group 0 has a norm of 1e-13, below the boundary. Group 1 has 1e-11, above it
    let x = t2(2, 1, vec![1e-13, 1e-11]);
    let out = last_axis().predict(&x).unwrap();

    // Below the boundary the cap of 1e12 applies, so the length stays below 1
    assert!((out[[0, 0]] - 0.1).abs() < 1e-6, "got {}", out[[0, 0]]);
    // Above it the norm applies, so the group reaches a length of exactly 1
    assert!((out[[1, 0]] - 1.0).abs() < 1e-6, "got {}", out[[1, 0]]);
}

/// A capped group whose elements are not all zero drops the projection term of the gradient
///
/// The cap replaces the norm with a constant, so the derivative of that group is the constant
/// alone. The all-zero test above cannot see this. Its output is 0, which zeroes the projection
/// term whether or not the layer drops it. Here the output is `0.1`, so a layer that kept the
/// term would return `0.99e12` in the first column instead of `1e12`
#[test]
fn unit_normalization_capped_group_gradient_drops_the_projection_term() {
    // Both spellings hold 1 group with a norm of 1e-13, below the cap boundary. The rank-2
    // row [1e-13, 0] takes the trailing-axis row path, and the column takes the strided path
    let cases = [
        (UnitNormalizationAxis::Default, t2(1, 2, vec![1e-13, 0.0])),
        (UnitNormalizationAxis::Custom(0), t2(2, 1, vec![1e-13, 0.0])),
    ];

    for (axis, x) in cases {
        let mut layer = UnitNormalization::new(axis.clone()).unwrap();
        let out = layer.forward(&x).unwrap();
        // The cap scaled the group, so its length stays at 0.1 rather than reaching 1
        assert!(
            (out.iter().map(|v| v * v).sum::<f32>() - 0.01).abs() < 1e-6,
            "{axis:?} did not take the cap"
        );

        let grad = layer.backward(&Tensor::ones(x.raw_dim())).unwrap();
        for (k, g) in grad.iter().enumerate() {
            assert!(
                (g / 1e12 - 1.0).abs() < 1e-6,
                "{axis:?} element {k} is {g:e}, and the cap alone gives 1e12"
            );
        }
    }
}

/// A group holding a NaN carries it across the whole group
#[test]
fn unit_normalization_spreads_a_nan_across_its_group() {
    let x = t2(2, 3, vec![f32::NAN, 1.0, 2.0, 3.0, 4.0, 0.0]);
    let out = last_axis().predict(&x).unwrap();

    for c in 0..3 {
        assert!(
            out[[0, c]].is_nan(),
            "column {c} of the NaN group is finite"
        );
    }
    // A neighboring group is untouched
    assert!((out[[1, 0]] - 0.6).abs() < 1e-6);
    assert!((out[[1, 1]] - 0.8).abs() < 1e-6);
}

/// A sum of squares that overflows `f32` gives a scale of 0
#[test]
fn unit_normalization_returns_zero_when_the_sum_of_squares_overflows() {
    let x = t2(1, 3, vec![1e20, 2e20, 3e20]);
    let out = last_axis().predict(&x).unwrap();

    for c in 0..3 {
        assert_eq!(out[[0, c]], 0.0, "column {c} is {}", out[[0, c]]);
    }
}

// layout

/// Both passes emit C order, so a consumer can read either as 1 contiguous slice
#[test]
fn unit_normalization_output_is_in_c_order() {
    for axis in [
        UnitNormalizationAxis::Default,
        UnitNormalizationAxis::Custom(1),
        UnitNormalizationAxis::Multiple(vec![0, 2]),
    ] {
        let x = ramp(&[2, 3, 4], 7.0);
        let mut layer = UnitNormalization::new(axis.clone()).unwrap();
        let out = layer.forward(&x).unwrap();
        assert!(
            out.is_standard_layout(),
            "{axis:?} forward is not in C order"
        );

        let grad = layer.backward(&upstream(&out)).unwrap();
        assert!(
            grad.is_standard_layout(),
            "{axis:?} backward is not in C order"
        );
    }
}

/// An input whose strides are not in C order gives the same values as one that is
#[test]
fn unit_normalization_accepts_input_that_is_not_in_c_order() {
    let base = t2(3, 5, (0..15).map(|k| k as f32 - 6.0).collect());
    let transposed = base.t().to_owned();
    assert!(
        !transposed.is_standard_layout(),
        "the test needs a transposed input that kept its strides"
    );

    let settled = Tensor::from_shape_vec(
        IxDyn(&[5, 3]),
        transposed.iter().copied().collect::<Vec<f32>>(),
    )
    .unwrap();

    let from_strided = last_axis().predict(&transposed).unwrap();
    let from_c_order = last_axis().predict(&settled).unwrap();
    assert_eq!(from_strided, from_c_order);
}

/// The rayon split is a performance knob, so it cannot move a bit
///
/// The full tensor clears the element-count gate in `crate::tuning::elementwise`, and each half
/// of it stays below the gate. The 2 halves therefore run the serial path. Both passes are
/// gated, so the forward and the backward run of each path is compared
#[test]
fn unit_normalization_parallel_path_matches_the_serial_path() {
    let (rows, features) = (3000usize, 1400usize);
    assert!(
        rows * features >= rustyml::tuning::elementwise::get_cheap_map_f32(),
        "the whole tensor must clear the gate"
    );
    assert!(
        rows / 2 * features < rustyml::tuning::elementwise::get_cheap_map_f32(),
        "each half must stay below the gate"
    );

    let data: Vec<f32> = (0..rows * features)
        .map(|k| ((k % 97) as f32 - 48.0) * 0.125)
        .collect();
    // This test slices the upstream gradient from 1 global pattern. A half therefore sees the
    // same values, at the same positions, as the matching part of the whole
    let upstream_data: Vec<f32> = (0..rows * features)
        .map(|k| 1.0 + 0.1 * ((k % 13) as f32 - 6.0))
        .collect();

    // Run the whole tensor and then each half through forward and backward, and compare the
    // concatenated halves against the whole
    let run = |chunk: &[f32], grad_chunk: &[f32], count: usize| -> (Vec<f32>, Vec<f32>) {
        let x = t2(count, features, chunk.to_vec());
        let mut layer = last_axis();
        let out = layer.forward(&x).unwrap();
        let grad = layer
            .backward(&t2(count, features, grad_chunk.to_vec()))
            .unwrap();
        (
            out.as_slice().unwrap().to_vec(),
            grad.as_slice().unwrap().to_vec(),
        )
    };

    let (parallel_out, parallel_grad) = run(&data, &upstream_data, rows);
    let split = rows / 2 * features;
    let (top_out, top_grad) = run(&data[..split], &upstream_data[..split], rows / 2);
    let (bottom_out, bottom_grad) = run(&data[split..], &upstream_data[split..], rows / 2);

    let serial_out: Vec<f32> = top_out.into_iter().chain(bottom_out).collect();
    let serial_grad: Vec<f32> = top_grad.into_iter().chain(bottom_grad).collect();
    assert_eq!(parallel_out, serial_out, "the gate moved a forward bit");
    assert_eq!(parallel_grad, serial_grad, "the gate moved a backward bit");
}

// contract

/// `predict` writes no cache, so it cannot serve a later backward pass
#[test]
fn unit_normalization_predict_matches_forward_and_caches_nothing() {
    let x = ramp(&[2, 4], 3.0);
    let mut layer = last_axis();

    let inferred = layer.predict(&x).unwrap();
    assert!(
        matches!(
            layer.backward(&inferred),
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "predict must not leave a cache behind"
    );

    let trained = layer.forward(&x).unwrap();
    assert_eq!(inferred, trained);
}

/// The layer holds nothing an optimizer can update
#[test]
fn unit_normalization_holds_no_parameter() {
    let mut layer = last_axis();
    assert_eq!(layer.layer_type(), "UnitNormalization");
    assert!(matches!(
        layer.param_count(),
        TrainingParameters::NoTrainable
    ));
    assert!(matches!(layer.get_weights(), LayerWeight::Empty));
    assert!(layer.parameters().is_empty());
}

/// The summary reads "Unknown" until a tensor has been through the layer
#[test]
fn unit_normalization_output_shape_needs_a_forward_pass() {
    let mut layer = last_axis();
    assert_eq!(layer.output_shape(), "Unknown");

    layer.forward(&ramp(&[2, 3, 4], 5.0)).unwrap();
    assert_eq!(layer.output_shape(), "(None, 3, 4)");
}

/// The layer runs inside a model, in inference and in training
#[test]
fn unit_normalization_runs_inside_a_sequential_model() {
    let x =
        Array4::from_shape_fn((2, 3, 3, 4), |(b, h, w, c)| (b + h + w + c) as f32 - 4.0).into_dyn();
    let y = last_axis().predict(&x).unwrap();

    let mut model = Sequential::new();
    model.add(last_axis()).compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let predicted = model.predict(&x).unwrap();
    assert_allclose(&predicted, &y, 1e-6);

    let history = model.fit(&x, &y, 2).unwrap();
    assert_eq!(history.loss().len(), 2);
}

// error paths

/// An empty axis list reduces over nothing, which no norm can do
#[test]
fn unit_normalization_rejects_an_empty_axis_list() {
    let result = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![]));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "the constructor must reject an empty axis list"
    );
}

/// An axis named twice would count its elements twice
#[test]
fn unit_normalization_rejects_a_repeated_axis() {
    let result = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![1, 0, 1]));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "the constructor must reject a repeated axis"
    );
}

/// A rank the layer cannot serve is an error at the forward pass
#[test]
fn unit_normalization_rejects_rank_below_2() {
    let rank_1 = Tensor::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(
        matches!(last_axis().predict(&rank_1), Err(Error::InvalidInput(_))),
        "a rank-1 input has no axis to normalize besides the batch axis"
    );
}

/// An axis past the end of the input is an error at the forward pass, not at construction
#[test]
fn unit_normalization_rejects_an_axis_out_of_bounds() {
    let layer = UnitNormalization::new(UnitNormalizationAxis::Custom(3)).unwrap();
    assert!(
        matches!(
            layer.predict(&ramp(&[2, 3], 0.0)),
            Err(Error::InvalidInput(_))
        ),
        "axis 3 does not exist in a rank-2 input"
    );

    let listed = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![0, 4])).unwrap();
    assert!(
        matches!(
            listed.predict(&ramp(&[2, 3, 4], 0.0)),
            Err(Error::InvalidInput(_))
        ),
        "axis 4 does not exist in a rank-3 input"
    );
}

/// An axis with no elements is an error, since the layer cannot normalize any group
#[test]
fn unit_normalization_rejects_empty_input() {
    let empty = Tensor::zeros(IxDyn(&[0, 3]));
    assert!(
        matches!(last_axis().forward(&empty), Err(Error::EmptyInput(_))),
        "forward must reject an input with no element"
    );
}

/// A backward pass before any forward pass has no cache to read
#[test]
fn unit_normalization_backward_needs_a_forward_pass() {
    let mut layer = last_axis();
    assert!(
        matches!(
            layer.backward(&ramp(&[2, 3], 0.0)),
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "backward must reject a missing cache"
    );
}

/// The backward pass rejects a gradient that does not have the shape of the forward output
#[test]
fn unit_normalization_backward_rejects_a_wrong_shape() {
    let mut layer = last_axis();
    layer.forward(&ramp(&[2, 3], 0.0)).unwrap();

    let result = layer.backward(&ramp(&[2, 4], 0.0));
    assert!(
        matches!(
            result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == vec![2, 3] && found == vec![2, 4]
        ),
        "backward must report the shape it expected"
    );
}
