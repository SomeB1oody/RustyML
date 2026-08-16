//! Integration tests for the upsampling layers: `UpSampling1D/2D/3D`.
//!
//! Covers the constructor forms, value placement, every interpolation mode, memory layout, the
//! error paths, and the serial-versus-parallel agreement. `gradient_check.rs` covers gradient
//! values against finite differences. This file does not duplicate them.

use ndarray::{Array3, Array4, Array5, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::upsampling::{
    Interpolation, UpSampling1D, UpSampling2D, UpSampling3D,
};
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

/// Build a tensor of `1, 2, 3, ...` with the given shape, in C order
fn ramp_of(shape: &[usize]) -> Tensor {
    let count: usize = shape.iter().product();
    let data: Vec<f32> = (1..=count).map(|v| v as f32).collect();
    Tensor::from_shape_vec(IxDyn(shape), data).expect("shape/data mismatch")
}

/// Every interpolation mode, so a property test can walk all of them
const MODES: [Interpolation; 5] = [
    Interpolation::Nearest,
    Interpolation::Bilinear,
    Interpolation::Bicubic,
    Interpolation::Lanczos3,
    Interpolation::Lanczos5,
];

// Shapes and value placement

/// The layer repeats each step `size` times, in place, and the features stay together
#[test]
fn up_sampling_1d_repeats_each_step_in_place() {
    let x = t3(1, 3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut layer = UpSampling1D::new(3).unwrap();
    let out = layer.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 9, 2]);
    // Pinned against reference values
    let want = t3(
        1,
        9,
        2,
        vec![
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 4.0,
            5.0,
        ],
    );
    assert_eq!(out, want);
}

/// A size of 1 leaves the tensor alone, and it still returns an owned tensor
#[test]
fn up_sampling_1d_size_one_returns_the_input_values() {
    let x = ramp_of(&[2, 3, 2]);
    let mut layer = UpSampling1D::new(1).unwrap();
    let out = layer.forward(&x).unwrap();

    assert_eq!(out, x);
    assert_eq!(layer.output_shape(), "(None, 3, 2)");
}

/// A pixel becomes a block of `size` rows by `size` columns, and the channels stay together
#[test]
fn up_sampling_2d_nearest_expands_each_pixel_into_a_block() {
    let x = t4(1, 2, 2, 2, vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
    let mut layer = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
    let out = layer.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 4, 4, 2]);
    for row in 0..4 {
        for column in 0..4 {
            let source = [[1.0, 2.0], [3.0, 4.0]][row / 2][column / 2];
            assert_eq!(out[[0, row, column, 0]], source);
            assert_eq!(out[[0, row, column, 1]], -source);
        }
    }
}

/// The 2 factors of the pair belong to the 2 spatial axes, not to the 2 ends of 1 axis
#[test]
fn up_sampling_2d_pair_names_1_factor_per_axis() {
    let mut layer = UpSampling2D::new((2, 3), Interpolation::Nearest).unwrap();
    let out = layer.forward(&ramp_of(&[2, 4, 5, 3])).unwrap();
    assert_eq!(out.shape(), &[2, 8, 15, 3]);
}

/// Every spatial axis grows by its own factor, and the batch and channel axes stay put
#[test]
fn up_sampling_3d_grows_each_spatial_axis_by_its_own_factor() {
    let x = t5(1, 2, 1, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let mut layer = UpSampling3D::new((2, 3, 1)).unwrap();
    let out = layer.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 4, 3, 1, 2]);
    for first in 0..4 {
        for second in 0..3 {
            let source = if first < 2 { 1.0 } else { 3.0 };
            assert_eq!(out[[0, first, second, 0, 0]], source);
            assert_eq!(out[[0, first, second, 0, 1]], source + 1.0);
        }
    }
}

/// An integer gives the same factor to all 3 spatial axes
#[test]
fn up_sampling_3d_integer_grows_every_axis_equally() {
    let mut layer = UpSampling3D::new(2).unwrap();
    let out = layer.forward(&ramp_of(&[2, 2, 3, 1, 2])).unwrap();
    assert_eq!(out.shape(), &[2, 4, 6, 2, 2]);
}

// Interpolation modes

/// The interpolated modes match reference values for the same input
#[test]
fn up_sampling_2d_interpolated_modes_match_reference() {
    let x = t4(1, 2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);

    // Pinned against reference values, with a factor of 2 and each interpolation mode
    let expected: [(Interpolation, [f32; 16]); 4] = [
        (
            Interpolation::Bilinear,
            [
                1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75,
                4.0,
            ],
        ),
        (
            Interpolation::Bicubic,
            [
                0.7352942, 1.030672, 1.616387, 1.911765, 1.326051, 1.621429, 2.207143, 2.502521,
                2.497479, 2.792857, 3.378572, 3.67395, 3.088236, 3.383614, 3.969328, 4.264707,
            ],
        ),
        (
            Interpolation::Lanczos3,
            [
                0.4735667, 0.8819152, 1.416174, 1.824522, 1.290264, 1.698612, 2.232871, 2.641219,
                2.358781, 2.76713, 3.301388, 3.709737, 3.175478, 3.583826, 4.118085, 4.526433,
            ],
        ),
        (
            Interpolation::Lanczos5,
            [
                0.3378642, 0.8024077, 1.314744, 1.779288, 1.266951, 1.731495, 2.243831, 2.708375,
                2.291624, 2.756168, 3.268504, 3.733048, 3.220711, 3.685255, 4.197591, 4.662135,
            ],
        ),
    ];

    for (mode, want) in expected {
        let mut layer = UpSampling2D::new(2, mode).unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 4, 1], "{mode:?}");
        assert_allclose(&out, &t4(1, 4, 4, 1, want.to_vec()), 2e-5);
    }
}

/// A constant image stays constant under every mode, because the weights of an output position
/// always add up to 1. The edge positions are the ones that would drift, because part of their
/// kernel falls outside the input
#[test]
fn up_sampling_2d_keeps_a_constant_image_constant() {
    for mode in MODES {
        for factor in [2usize, 3, 5] {
            let x = Tensor::from_elem(IxDyn(&[1, 4, 3, 2]), 7.5);
            let mut layer = UpSampling2D::new(factor, mode).unwrap();
            let out = layer.forward(&x).unwrap();
            for (index, &value) in out.iter().enumerate() {
                assert!(
                    (value - 7.5).abs() < 1e-5,
                    "{mode:?} factor {factor} drifted to {value} at {index}"
                );
            }
        }
    }
}

/// Only the wide kernels have negative lobes, so only they can leave the input range. Bilinear
/// stays inside it, and the repeat mode reproduces the input values exactly
#[test]
fn up_sampling_2d_only_the_wide_kernels_overshoot() {
    // A single bright pixel against a dark field is what makes a negative lobe visible
    let mut data = vec![0.0f32; 25];
    data[12] = 1.0;
    let x = t4(1, 5, 5, 1, data);

    for mode in MODES {
        let mut layer = UpSampling2D::new(3, mode).unwrap();
        let out = layer.forward(&x).unwrap();
        let lowest = out.iter().fold(f32::MAX, |m, &v| m.min(v));
        match mode {
            Interpolation::Nearest | Interpolation::Bilinear => {
                assert!(lowest >= -1e-6, "{mode:?} undershot to {lowest}")
            }
            _ => assert!(lowest < -1e-3, "{mode:?} never undershot, lowest {lowest}"),
        }
    }
}

// Backward pass

/// The repeat mode sends every output gradient back to the position the forward pass copied it from
#[test]
fn up_sampling_1d_backward_sums_each_repeated_run() {
    let x = ramp_of(&[1, 2, 2]);
    let mut layer = UpSampling1D::new(3).unwrap();
    layer.forward(&x).unwrap();

    // 1 distinct value per output position, so a gradient landing on the wrong step shows up
    let grad = layer.backward(&ramp_of(&[1, 6, 2])).unwrap();

    assert_eq!(grad.shape(), &[1, 2, 2]);
    // Steps 0, 1, 2 feed input step 0, and their first features are 1, 3, 5
    assert_eq!(grad[[0, 0, 0]], 1.0 + 3.0 + 5.0);
    assert_eq!(grad[[0, 0, 1]], 2.0 + 4.0 + 6.0);
    assert_eq!(grad[[0, 1, 0]], 7.0 + 9.0 + 11.0);
    assert_eq!(grad[[0, 1, 1]], 8.0 + 10.0 + 12.0);
}

/// Under an all-ones upstream the gradient of every input position is the total weight it sent
/// out. Each of the 4 pixels feeds 4 output pixels whose weights add up to 1, so every entry is
/// exactly 4
#[test]
fn up_sampling_2d_bilinear_backward_matches_reference() {
    let x = t4(1, 2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let mut layer = UpSampling2D::new(2, Interpolation::Bilinear).unwrap();
    let out = layer.forward(&x).unwrap();

    let grad = layer.backward(&Tensor::ones(out.raw_dim())).unwrap();

    // Pinned against reference values through `jax.vjp`
    assert_allclose(&grad, &t4(1, 2, 2, 1, vec![4.0; 4]), 1e-5);
}

/// The backward pass is the transpose of the forward pass, so the 2 inner products agree for
/// every mode. That is the test the finite-difference check cannot make exact
#[test]
fn up_sampling_2d_backward_is_the_transpose_of_forward() {
    // The last 2 shapes are shorter than the widest kernel, so every weight there survives a
    // clipping and a rescale. That is the case the transposed band is easiest to get wrong
    let cases = [
        (vec![2, 3, 4, 2], (2usize, 3usize)),
        (vec![1, 1, 1, 1], (5, 5)),
        (vec![1, 2, 1, 3], (1, 5)),
    ];
    for mode in MODES {
        for (shape, factors) in &cases {
            let x = ramp_of(shape);
            let mut layer = UpSampling2D::new(*factors, mode).unwrap();
            let out = layer.forward(&x).unwrap();

            // A varied upstream, so no symmetry can hide a misplaced weight
            let upstream_values: Vec<f32> = (0..out.len())
                .map(|k| ((k % 17) as f32 - 8.0) * 0.125)
                .collect();
            let upstream = Tensor::from_shape_vec(out.raw_dim(), upstream_values).unwrap();
            let grad = layer.backward(&upstream).unwrap();

            let forward_product: f32 = out.iter().zip(&upstream).map(|(&a, &b)| a * b).sum();
            let backward_product: f32 = x.iter().zip(&grad).map(|(&a, &b)| a * b).sum();
            assert!(
                (forward_product - backward_product).abs() <= 1e-3 * forward_product.abs().max(1.0),
                "{mode:?} {shape:?} forward {forward_product} backward {backward_product}"
            );
        }
    }
}

/// A factor large enough to take the output past what an index can hold is an error, not a panic
#[test]
fn up_sampling_rejects_a_factor_that_overflows_the_output() {
    let mut layer = UpSampling2D::new((usize::MAX, 1), Interpolation::Nearest).unwrap();
    match layer.forward(&ramp_of(&[2, 3, 4, 2])).unwrap_err() {
        Error::InvalidInput(message) => {
            assert!(
                message.contains("does not fit"),
                "unexpected message {message}"
            )
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

// Layout

/// An input whose memory is not in C order still gives the right values, in C order
#[test]
fn up_sampling_2d_accepts_an_input_that_is_not_in_c_order() {
    let base = ramp_of(&[2, 3, 4, 2]);
    // A permuted view shares the buffer under reordered strides, so it is not in C order
    let permuted = base.view().permuted_axes(IxDyn(&[0, 2, 1, 3])).to_owned();
    assert!(!permuted.is_standard_layout());

    let mut layer = UpSampling2D::new(2, Interpolation::Bilinear).unwrap();
    let from_view = layer.forward(&permuted).unwrap();

    let mut contiguous = Tensor::zeros(permuted.raw_dim());
    contiguous.assign(&permuted);
    let mut twin = UpSampling2D::new(2, Interpolation::Bilinear).unwrap();
    let from_contiguous = twin.forward(&contiguous).unwrap();

    assert_eq!(from_view, from_contiguous);
    assert!(from_view.is_standard_layout());
}

/// Every layer of the family emits a gradient in C order too
#[test]
fn up_sampling_layers_emit_gradients_in_c_order() {
    let mut first = UpSampling1D::new(2).unwrap();
    let out = first.forward(&ramp_of(&[2, 3, 2])).unwrap();
    assert!(first.backward(&out).unwrap().is_standard_layout());

    let mut second = UpSampling2D::new((2, 3), Interpolation::Lanczos3).unwrap();
    let out = second.forward(&ramp_of(&[2, 3, 4, 2])).unwrap();
    assert!(second.backward(&out).unwrap().is_standard_layout());

    let mut third = UpSampling3D::new(2).unwrap();
    let out = third.forward(&ramp_of(&[1, 2, 2, 2, 3])).unwrap();
    assert!(third.backward(&out).unwrap().is_standard_layout());
}

// Serial and parallel agreement

/// The repeat mode above the parallel gate gives the same bits as the same work below it
///
/// Samples never mix, so 1 batch of 2 must equal 2 batches of 1. The whole clears the gate on
/// the forward pass and on the backward pass, and each single sample stays under it
#[test]
fn up_sampling_repeat_parallel_path_matches_the_serial_path() {
    let gate = rustyml::tuning::upsampling::get_parallel_min_ops();
    let (samples, steps, channels, factor) = (2usize, 64usize, 2048usize, 8usize);
    // Forward reads 1 position per output element, and backward reads `factor` per input
    // element, so both passes come to the same element-op count here
    let work = samples * steps * factor * channels;
    assert!(work >= gate, "the whole batch must clear the gate");
    assert!(work / samples < gate, "1 sample must stay below the gate");

    let run = |count: usize, offset: usize| -> (Vec<f32>, Vec<f32>) {
        let x_values: Vec<f32> = (0..count * steps * channels)
            .map(|k| ((k + offset) % 23) as f32 * 0.5 - 5.0)
            .collect();
        let x = Tensor::from_shape_vec(IxDyn(&[count, steps, channels]), x_values).unwrap();
        let mut layer = UpSampling1D::new(factor).unwrap();
        let out = layer.forward(&x).unwrap();
        let upstream_values: Vec<f32> = (0..out.len())
            .map(|k| ((k + offset * factor) % 13) as f32 - 6.0)
            .collect();
        let upstream = Tensor::from_shape_vec(out.raw_dim(), upstream_values).unwrap();
        let grad = layer.backward(&upstream).unwrap();
        (
            out.as_slice().unwrap().to_vec(),
            grad.as_slice().unwrap().to_vec(),
        )
    };

    let (parallel_out, parallel_grad) = run(samples, 0);
    let (first_out, first_grad) = run(1, 0);
    let (second_out, second_grad) = run(1, steps * channels);

    let serial_out: Vec<f32> = first_out.into_iter().chain(second_out).collect();
    let serial_grad: Vec<f32> = first_grad.into_iter().chain(second_grad).collect();
    assert_eq!(parallel_out, serial_out, "forward differs across the gate");
    assert_eq!(
        parallel_grad, serial_grad,
        "backward differs across the gate"
    );
}

/// The weighted path above the parallel gate gives the same bits as the same work below it
#[test]
fn up_sampling_weighted_parallel_path_matches_the_serial_path() {
    let gate = rustyml::tuning::upsampling::get_parallel_min_ops();
    let (samples, side, channels) = (8usize, 91usize, 3usize);
    // The first pass doubles 1 axis, and `Lanczos5` reads 11 positions per output position
    let first_pass_work = samples * (side * 2) * side * channels * 11;
    assert!(
        first_pass_work >= gate,
        "the whole batch must clear the gate"
    );
    assert!(
        2 * first_pass_work / samples < gate,
        "1 sample must stay below the gate on both passes"
    );

    let run = |count: usize, offset: usize| -> (Vec<f32>, Vec<f32>) {
        let x_values: Vec<f32> = (0..count * side * side * channels)
            .map(|k| ((k + offset) % 29) as f32 * 0.25 - 3.5)
            .collect();
        let x = Tensor::from_shape_vec(IxDyn(&[count, side, side, channels]), x_values).unwrap();
        let mut layer = UpSampling2D::new(2, Interpolation::Lanczos5).unwrap();
        let out = layer.forward(&x).unwrap();
        let upstream_values: Vec<f32> = (0..out.len())
            .map(|k| ((k + offset * 4) % 19) as f32 * 0.5 - 4.5)
            .collect();
        let upstream = Tensor::from_shape_vec(out.raw_dim(), upstream_values).unwrap();
        let grad = layer.backward(&upstream).unwrap();
        (
            out.as_slice().unwrap().to_vec(),
            grad.as_slice().unwrap().to_vec(),
        )
    };

    let (parallel_out, parallel_grad) = run(samples, 0);
    let sample_size = side * side * channels;
    let mut serial_out = Vec::with_capacity(parallel_out.len());
    let mut serial_grad = Vec::with_capacity(parallel_grad.len());
    for sample in 0..samples {
        let (out, grad) = run(1, sample * sample_size);
        serial_out.extend(out);
        serial_grad.extend(grad);
    }

    assert_eq!(parallel_out, serial_out, "forward differs across the gate");
    assert_eq!(
        parallel_grad, serial_grad,
        "backward differs across the gate"
    );
}

// Error paths

/// A factor of 0 would empty the tensor, so the constructor rejects it
#[test]
fn up_sampling_rejects_a_factor_of_zero() {
    for error in [
        UpSampling1D::new(0).unwrap_err(),
        UpSampling2D::new(0, Interpolation::Nearest).unwrap_err(),
        UpSampling2D::new((2, 0), Interpolation::Bilinear).unwrap_err(),
        UpSampling3D::new((1, 0, 2)).unwrap_err(),
    ] {
        match error {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "size"),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }
}

/// Each layer serves 1 rank, and any other rank is an error rather than a panic
#[test]
fn up_sampling_rejects_the_wrong_rank() {
    let cases: [(Box<dyn Layer>, Tensor); 3] = [
        (
            Box::new(UpSampling1D::new(2).unwrap()),
            ramp_of(&[2, 3, 4, 5]),
        ),
        (
            Box::new(UpSampling2D::new(2, Interpolation::Nearest).unwrap()),
            ramp_of(&[2, 3, 4]),
        ),
        (Box::new(UpSampling3D::new(2).unwrap()), ramp_of(&[2, 3, 4])),
    ];
    for (mut layer, x) in cases {
        match layer.forward(&x).unwrap_err() {
            Error::InvalidInput(message) => {
                assert!(message.contains("D input"), "unexpected message {message}")
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}

/// An axis with no extent has nothing to enlarge
#[test]
fn up_sampling_rejects_an_empty_input() {
    let mut layer = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
    let empty = Tensor::zeros(IxDyn(&[2, 0, 4, 3]));
    assert!(matches!(
        layer.forward(&empty).unwrap_err(),
        Error::EmptyInput(_)
    ));
}

/// The backward pass needs the shape the forward pass saw
#[test]
fn up_sampling_backward_before_forward_is_an_error() {
    let mut layer = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
    assert!(matches!(
        layer.backward(&ramp_of(&[1, 2, 2, 1])).unwrap_err(),
        Error::NeuralNetwork(NnError::ForwardPassNotRun(_))
    ));
}

/// A gradient that is not the shape the layer produced is an error
#[test]
fn up_sampling_backward_checks_the_gradient_shape() {
    let mut layer = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
    layer.forward(&ramp_of(&[1, 2, 2, 1])).unwrap();

    match layer.backward(&ramp_of(&[1, 4, 3, 1])).unwrap_err() {
        Error::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![1, 4, 4, 1]);
            assert_eq!(found, vec![1, 4, 3, 1]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

// Layer trait surface

/// No layer of the family holds a parameter, and each one names itself for `summary()`
#[test]
fn up_sampling_layers_hold_no_parameter() {
    let mut layers: [Box<dyn Layer>; 3] = [
        Box::new(UpSampling1D::new(2).unwrap()),
        Box::new(UpSampling2D::new(2, Interpolation::Bicubic).unwrap()),
        Box::new(UpSampling3D::new(2).unwrap()),
    ];
    let names = ["UpSampling1D", "UpSampling2D", "UpSampling3D"];

    for (layer, name) in layers.iter_mut().zip(names) {
        assert!(matches!(
            layer.param_count(),
            TrainingParameters::NoTrainable
        ));
        assert!(matches!(layer.get_weights(), LayerWeight::Empty));
        assert!(layer.parameters().is_empty());
        // No forward pass has run, so the layer cannot know its output shape yet
        assert_eq!(layer.output_shape(), "Unknown");
        assert_eq!(layer.layer_type(), name);
    }
}

/// After a forward pass the summary prints the enlarged shape, with the batch axis as "None"
#[test]
fn up_sampling_output_shape_reports_the_enlarged_shape() {
    let mut layer = UpSampling2D::new((2, 3), Interpolation::Nearest).unwrap();
    layer.forward(&ramp_of(&[4, 5, 6, 2])).unwrap();
    assert_eq!(layer.output_shape(), "(None, 10, 18, 2)");

    let mut volume = UpSampling3D::new((1, 2, 3)).unwrap();
    volume.forward(&ramp_of(&[2, 2, 3, 4, 1])).unwrap();
    assert_eq!(volume.output_shape(), "(None, 2, 6, 12, 1)");
}

// Model integration

/// A pooling stage and an upsampling stage of the same factor restore the original shape
#[test]
fn up_sampling_2d_undoes_the_shape_change_of_pooling() {
    let x = ramp_of(&[2, 8, 8, 3]);

    let mut model = Sequential::new();
    model
        .add(MaxPooling2D::new((2, 2), vec![2, 8, 8, 3]).unwrap())
        .add(UpSampling2D::new(2, Interpolation::Nearest).unwrap())
        .compile(
            SGD::new(0.01, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let out = model.predict(&x).unwrap();
    assert_eq!(out.shape(), x.shape());
}

/// A model carrying an upsampling layer survives a save and load round trip
///
/// The layer stores no weight, but it still holds its position in the structure check. The
/// rebuilt model must therefore carry it at the same index with the same type
#[test]
fn up_sampling_2d_survives_a_save_and_load_round_trip() {
    let x = Tensor::from_shape_vec(
        IxDyn(&[2, 3, 3, 1]),
        (0..18).map(|k| (k % 5) as f32 * 0.2 - 0.4).collect(),
    )
    .unwrap();

    let build = || {
        let mut model = Sequential::new();
        model
            .add(UpSampling2D::new(2, Interpolation::Lanczos3).unwrap())
            .add(Conv2D::new(2, (3, 3), vec![2, 6, 6, 1], (1, 1), Linear::new()).unwrap())
            .add(Flatten::new(vec![2, 4, 4, 2]).unwrap())
            .add(Dense::new(32, 2, Linear::new()).unwrap());
        model
    };

    let mut model = build();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let before = model.predict(&x).unwrap();

    let path = std::env::temp_dir().join("rustyml_upsampling_round_trip.bin");
    model.save_to_path(&path).unwrap();

    let mut restored = build();
    restored.load_from_path(&path).unwrap();
    let after = restored.predict(&x).unwrap();

    assert_allclose(&before, &after, 0.0);
    std::fs::remove_file(&path).unwrap();
}

/// A decoder that upsamples and then convolves trains without an error, and the loss falls
#[test]
fn up_sampling_2d_trains_inside_a_decoder() {
    let x = Tensor::from_shape_vec(
        IxDyn(&[2, 4, 4, 1]),
        (0..32).map(|k| (k % 7) as f32 * 0.1).collect(),
    )
    .unwrap();
    // A 3x3 kernel with no padding takes the 8x8 the upsampling produced down to 6x6
    let y = Tensor::from_shape_vec(
        IxDyn(&[2, 6, 6, 2]),
        (0..144).map(|k| (k % 5) as f32 * 0.05).collect(),
    )
    .unwrap();

    let mut model = Sequential::new();
    model
        .add(UpSampling2D::new(2, Interpolation::Bilinear).unwrap())
        .add(Conv2D::new(2, (3, 3), vec![2, 8, 8, 1], (1, 1), Linear::new()).unwrap())
        .compile(
            SGD::new(0.05, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let history = model.fit(&x, &y, 6).unwrap();
    let losses = history.loss();
    assert_eq!(losses.len(), 6);
    assert!(
        losses[5] < losses[0],
        "loss did not fall: {} then {}",
        losses[0],
        losses[5]
    );
    assert_eq!(model.predict(&x).unwrap().shape(), &[2, 6, 6, 2]);
}
