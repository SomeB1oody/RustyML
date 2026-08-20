//! Integration tests for the `PReLU` layer.
//!
//! Covers the constructor, the `shared_axes` rule, forward values, and the 2 gradients. It also
//! covers the branch at exactly 0, memory layout, the parallel gate, the error paths, and the
//! weight surface. `gradient_check.rs` covers the gradients against finite differences, and
//! `serialize.rs` covers the save and load round trip. This file does not duplicate them.
//!
//! Every pinned number here is exact in `f32`, so the test writes it as a plain decimal rather
//! than as a rounded reference value. The branch rule at exactly 0 is the one place a reasonable
//! implementation could differ. A comparison with Keras 3.15 on the jax backend confirms this.
//! The derivative there is 0, and not 1 and not `alpha`.

use ndarray::{Array, Array1, Array2, Array4, ArrayD, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::leaky_relu::LeakyReLU;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::p_relu::PReLU;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// helpers

/// Build a Tensor of any rank from row-major data
fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape/data mismatch")
}

/// Build a `PReLU` whose slopes hold the given row-major values
fn p_relu_with(input_shape: Vec<usize>, shared_axes: Vec<usize>, slopes: Vec<f32>) -> PReLU {
    let mut layer = PReLU::new(input_shape, 0.0).unwrap();
    if !shared_axes.is_empty() {
        layer = layer.with_shared_axes(shared_axes).unwrap();
    }
    let shape = slopes_of(&layer).shape().to_vec();
    layer.set_weights(tensor(&shape, slopes)).unwrap();
    layer
}

/// Read the slope array back out of a layer
fn slopes_of(layer: &PReLU) -> ArrayD<f32> {
    match layer.get_weights() {
        LayerWeight::PReLU(w) => w.alpha.into_owned(),
        other => panic!("PReLU must report LayerWeight::PReLU, got {other:?}"),
    }
}

/// Read the slope gradient the last backward pass wrote
fn slope_gradient(layer: &mut PReLU) -> Vec<f32> {
    let params = layer.parameters();
    assert_eq!(params.len(), 1, "PReLU exposes exactly 1 parameter tensor");
    assert!(
        !params[0].decays,
        "decoupled weight decay must skip the slopes"
    );
    params[0].grad.to_vec()
}

// Constructor validation

/// The batch axis plus at least 1 slope axis is the smallest usable input
#[test]
fn p_relu_rejects_an_input_shape_of_rank_below_2() {
    for shape in [vec![], vec![4]] {
        let err = PReLU::new(shape.clone(), 0.0).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "rank {} must be rejected, got {err:?}",
            shape.len()
        );
    }
}

/// A 0 anywhere leaves an axis with no position, so no slope array exists
#[test]
fn p_relu_rejects_a_zero_dimension_on_any_axis() {
    for shape in [vec![0, 3], vec![2, 0], vec![2, 3, 0, 4], vec![2, 3, 4, 0]] {
        let err = PReLU::new(shape.clone(), 0.0).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "{shape:?} must be rejected, got {err:?}"
        );
    }
}

/// A non-finite slope makes every negative element non-finite on the first forward pass
#[test]
fn p_relu_rejects_a_non_finite_alpha() {
    for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = PReLU::new(vec![2, 3], alpha).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "alpha {alpha} must be rejected, got {err:?}"
        );
    }
}

/// A negative or zero slope is a legal starting point, unlike `LeakyReLU`'s constant
#[test]
fn p_relu_accepts_a_zero_or_negative_alpha() {
    for alpha in [0.0f32, -0.5, -3.0] {
        let layer = PReLU::new(vec![2, 3], alpha).unwrap();
        assert!(slopes_of(&layer).iter().all(|&v| v == alpha));
    }
}

/// Without shared axes the layer holds 1 slope per position of the input, batch axis removed
#[test]
fn p_relu_holds_1_slope_per_position_without_shared_axes() {
    for (shape, count) in [
        (vec![8, 5], 5usize),
        (vec![2, 4, 6], 24),
        (vec![2, 3, 4, 5], 60),
        (vec![2, 2, 3, 4, 3], 72),
    ] {
        let layer = PReLU::new(shape.clone(), 0.25).unwrap();
        assert_eq!(
            layer.param_count(),
            TrainingParameters::Trainable(count),
            "shape {shape:?}"
        );
        assert_eq!(slopes_of(&layer).shape(), &shape[1..]);
    }
}

// shared_axes validation

/// Axis 0 is the batch axis. Every slope already covers the whole batch
#[test]
fn p_relu_rejects_the_batch_axis_in_shared_axes() {
    let err = PReLU::new(vec![2, 3, 4], 0.0)
        .unwrap()
        .with_shared_axes(vec![0])
        .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }), "{err:?}");
}

/// An axis at or above the rank names no axis of the input
#[test]
fn p_relu_rejects_a_shared_axis_at_or_above_the_rank() {
    for axis in [3usize, 4, 99] {
        let err = PReLU::new(vec![2, 3, 4], 0.0)
            .unwrap()
            .with_shared_axes(vec![axis])
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "axis {axis} must be rejected, got {err:?}"
        );
    }
}

/// A repeated axis is a caller mistake, not a second reduction
#[test]
fn p_relu_rejects_a_repeated_shared_axis() {
    let err = PReLU::new(vec![2, 3, 4], 0.0)
        .unwrap()
        .with_shared_axes(vec![1, 2, 1])
        .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }), "{err:?}");
}

/// Each shared axis drops to extent 1, which is what cuts the slope count
#[test]
fn p_relu_shared_axes_set_the_slope_shape() {
    for (shape, shared, want) in [
        (vec![2, 4, 6], vec![1usize], vec![1usize, 6]),
        (vec![2, 4, 6], vec![2], vec![4, 1]),
        (vec![2, 4, 6], vec![1, 2], vec![1, 1]),
        (vec![2, 3, 4, 5], vec![1, 2], vec![1, 1, 5]),
        (vec![2, 3, 4, 5], vec![3], vec![3, 4, 1]),
        (vec![2, 3, 4, 5], vec![1, 2, 3], vec![1, 1, 1]),
        (vec![2, 2, 3, 4, 3], vec![4], vec![2, 3, 4, 1]),
    ] {
        let layer = PReLU::new(shape.clone(), 0.25)
            .unwrap()
            .with_shared_axes(shared.clone())
            .unwrap();
        assert_eq!(
            slopes_of(&layer).shape(),
            want.as_slice(),
            "shape {shape:?} shared {shared:?}"
        );
        let count: usize = want.iter().product();
        assert_eq!(layer.param_count(), TrainingParameters::Trainable(count));
    }
}

/// The axes name a set, so the caller does not have to sort them
#[test]
fn p_relu_shared_axes_ignores_the_order_given() {
    let ordered = PReLU::new(vec![2, 3, 4, 5], 0.25)
        .unwrap()
        .with_shared_axes(vec![1, 3])
        .unwrap();
    let reversed = PReLU::new(vec![2, 3, 4, 5], 0.25)
        .unwrap()
        .with_shared_axes(vec![3, 1])
        .unwrap();
    assert_eq!(slopes_of(&ordered).shape(), slopes_of(&reversed).shape());
}

/// The starting slope survives the resize, so the 2 builder steps commute
#[test]
fn p_relu_shared_axes_keeps_the_starting_slope() {
    let layer = PReLU::new(vec![2, 3, 4], 0.25)
        .unwrap()
        .with_shared_axes(vec![1])
        .unwrap();
    let slopes = slopes_of(&layer);
    assert_eq!(slopes.shape(), &[1, 4]);
    assert!(slopes.iter().all(|&v| v == 0.25), "{slopes:?}");
}

// Forward values

/// 1 slope per feature, applied only below 0
#[test]
fn p_relu_forward_uses_1_slope_per_feature() {
    let mut layer = p_relu_with(vec![2, 3], vec![], vec![0.25, -0.1, 0.5]);
    let x = tensor(&[2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 0.0]);

    let got = layer.forward(&x).unwrap();
    let want = tensor(&[2, 3], vec![-0.25, 2.0, -1.5, 4.0, 0.5, 0.0]);
    assert_allclose(&got, &want, 0.0_f32);
}

/// With the 2 spatial axes shared, 1 slope covers a whole channel plane
#[test]
fn p_relu_forward_broadcasts_a_channel_slope_over_space() {
    let mut layer = p_relu_with(vec![1, 2, 2, 2], vec![1, 2], vec![0.5, -0.25]);
    let x = tensor(
        &[1, 2, 2, 2],
        vec![-1.0, 2.0, 3.0, -4.0, -5.0, 6.0, 7.0, -8.0],
    );

    let got = layer.forward(&x).unwrap();
    let want = tensor(
        &[1, 2, 2, 2],
        vec![-0.5, 2.0, 3.0, 1.0, -2.5, 6.0, 7.0, 2.0],
    );
    assert_allclose(&got, &want, 0.0_f32);
}

/// 0 sits on the positive branch, so no slope reaches it
#[test]
fn p_relu_forward_leaves_0_alone_whatever_the_slope() {
    for alpha in [0.0f32, 0.25, -2.0, 100.0] {
        let mut layer = PReLU::new(vec![1, 2], alpha).unwrap();
        let got = layer.forward(&tensor(&[1, 2], vec![0.0, -0.0])).unwrap();
        assert_eq!(got[[0, 0]], 0.0, "alpha {alpha}");
        assert_eq!(got[[0, 1]], 0.0, "alpha {alpha}");
    }
}

/// A slope of 0 gives the ReLU transform and the ReLU gradient, including at exactly 0
#[test]
fn p_relu_with_a_zero_slope_matches_relu() {
    let x = tensor(&[2, 4], vec![-1.5, 0.0, 2.0, -0.25, 3.0, -4.0, 0.0, 0.75]);
    let g = tensor(&[2, 4], vec![1.0, 2.0, -3.0, 4.0, 0.5, -0.5, 6.0, -7.0]);

    let mut learned = PReLU::new(vec![2, 4], 0.0).unwrap();
    let mut fixed = ReLU::new();

    assert_allclose(
        &learned.forward(&x).unwrap(),
        &fixed.forward(&x).unwrap(),
        0.0_f32,
    );
    assert_allclose(
        &learned.backward(&g).unwrap(),
        &fixed.backward(&g).unwrap(),
        0.0_f32,
    );
}

/// A uniform slope gives the LeakyReLU transform. The 2 layers part company only at exactly 0,
/// where LeakyReLU passes the gradient and this layer stops it
#[test]
fn p_relu_with_a_uniform_slope_matches_leaky_relu_away_from_0() {
    let x = tensor(&[2, 4], vec![-1.5, 0.5, 2.0, -0.25, 3.0, -4.0, 1.0, 0.75]);
    let g = tensor(&[2, 4], vec![1.0, 2.0, -3.0, 4.0, 0.5, -0.5, 6.0, -7.0]);

    let mut learned = PReLU::new(vec![2, 4], 0.2).unwrap();
    let mut fixed = LeakyReLU::new(0.2).unwrap();
    assert_allclose(
        &learned.forward(&x).unwrap(),
        &fixed.forward(&x).unwrap(),
        0.0_f32,
    );
    assert_allclose(
        &learned.backward(&g).unwrap(),
        &fixed.backward(&g).unwrap(),
        0.0_f32,
    );

    let zero = tensor(&[1, 1], vec![0.0]);
    let ones = tensor(&[1, 1], vec![1.0]);
    let mut learned = PReLU::new(vec![1, 1], 0.2).unwrap();
    let mut fixed = LeakyReLU::new(0.2).unwrap();
    learned.forward(&zero).unwrap();
    fixed.forward(&zero).unwrap();
    assert_eq!(learned.backward(&ones).unwrap()[[0, 0]], 0.0);
    assert_eq!(fixed.backward(&ones).unwrap()[[0, 0]], 1.0);
}

// Backward values

/// The input gradient passes above 0, scales below 0, and stops at exactly 0
#[test]
fn p_relu_input_gradient_follows_the_sign_of_the_input() {
    let mut layer = p_relu_with(vec![2, 3], vec![], vec![0.25, -0.1, 0.5]);
    let x = tensor(&[2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 0.0]);
    layer.forward(&x).unwrap();

    let got = layer.backward(&Tensor::ones(x.raw_dim())).unwrap();
    let want = tensor(&[2, 3], vec![0.25, 1.0, 0.5, 1.0, -0.1, 0.0]);
    assert_allclose(&got, &want, 0.0_f32);
}

/// The slope gradient collects `g * x` from every negative element the slope covers
#[test]
fn p_relu_slope_gradient_sums_over_the_batch() {
    let mut layer = p_relu_with(vec![2, 3], vec![], vec![0.25, -0.1, 0.5]);
    let x = tensor(&[2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 0.0]);
    layer.forward(&x).unwrap();
    layer.backward(&Tensor::ones(x.raw_dim())).unwrap();

    assert_eq!(slope_gradient(&mut layer), vec![-1.0, -5.0, -3.0]);
}

/// A shared axis folds into the same slope as the batch axis does
#[test]
fn p_relu_slope_gradient_sums_over_every_shared_axis() {
    let mut layer = p_relu_with(vec![1, 2, 2, 2], vec![1, 2], vec![0.5, -0.25]);
    let x = tensor(
        &[1, 2, 2, 2],
        vec![-1.0, 2.0, 3.0, -4.0, -5.0, 6.0, 7.0, -8.0],
    );
    layer.forward(&x).unwrap();
    layer.backward(&Tensor::ones(x.raw_dim())).unwrap();

    // Channel 0 holds -1 and -5. Channel 1 holds -4 and -8
    assert_eq!(slope_gradient(&mut layer), vec![-6.0, -12.0]);
}

/// The upstream gradient weights each contribution, so it is not a plain sum of the input
#[test]
fn p_relu_slope_gradient_weights_by_the_upstream_gradient() {
    let mut layer = p_relu_with(vec![2, 2], vec![], vec![0.3, 0.3]);
    let x = tensor(&[2, 2], vec![-1.0, -2.0, -4.0, 5.0]);
    let g = tensor(&[2, 2], vec![2.0, 0.5, -1.0, 10.0]);
    layer.forward(&x).unwrap();
    layer.backward(&g).unwrap();

    // Column 0: 2 * -1 plus -1 * -4. Column 1: 0.5 * -2, and 5 is positive so it adds nothing
    assert_eq!(slope_gradient(&mut layer), vec![2.0, -1.0]);
}

/// A gradient exists only after a backward pass, so the optimizer skips a fresh layer
#[test]
fn p_relu_exposes_no_parameter_before_the_first_backward() {
    let mut layer = PReLU::new(vec![2, 3], 0.25).unwrap();
    assert!(layer.parameters().is_empty());
    layer.forward(&Tensor::zeros(IxDyn(&[2, 3]))).unwrap();
    assert!(layer.parameters().is_empty());
    layer.backward(&Tensor::ones(IxDyn(&[2, 3]))).unwrap();
    assert_eq!(layer.parameters().len(), 1);
}

/// Resizing the slope array drops a gradient that no longer matches it
#[test]
fn p_relu_shared_axes_drops_a_stale_gradient() {
    let mut layer = PReLU::new(vec![2, 3, 4], 0.25).unwrap();
    layer.forward(&Tensor::zeros(IxDyn(&[2, 3, 4]))).unwrap();
    layer.backward(&Tensor::ones(IxDyn(&[2, 3, 4]))).unwrap();
    assert_eq!(layer.parameters().len(), 1);

    let mut layer = layer.with_shared_axes(vec![1]).unwrap();
    assert!(
        layer.parameters().is_empty(),
        "the old gradient does not fit the resized slope array"
    );
}

// Memory layout

/// A strided input must still give a C-order output and a C-order input gradient
#[test]
fn p_relu_emits_c_order_tensors_from_a_strided_input() {
    let strided = Array2::from_shape_vec((3, 4), (0..12).map(|v| v as f32 - 6.0).collect())
        .unwrap()
        .reversed_axes()
        .into_dyn();
    assert!(!strided.is_standard_layout(), "the fixture must be strided");

    let mut layer = PReLU::new(vec![4, 3], 0.25).unwrap();
    let out = layer.forward(&strided).unwrap();
    assert!(out.is_standard_layout(), "forward output must be C order");

    let grad = layer.backward(&strided).unwrap();
    assert!(grad.is_standard_layout(), "input gradient must be C order");

    // The values must not depend on the layout either
    let packed = strided.as_standard_layout().into_owned();
    let mut twin = PReLU::new(vec![4, 3], 0.25).unwrap();
    assert_allclose(&twin.forward(&packed).unwrap(), &out, 0.0_f32);
    assert_allclose(&twin.backward(&packed).unwrap(), &grad, 0.0_f32);
    assert_eq!(slope_gradient(&mut twin), slope_gradient(&mut layer));
}

/// `set_weights` normalizes a strided slope array, because `parameters()` hands out a slice
#[test]
fn p_relu_set_weights_repacks_a_strided_array() {
    let strided = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        .unwrap()
        .reversed_axes()
        .into_dyn();
    assert!(!strided.is_standard_layout(), "the fixture must be strided");

    let mut layer = PReLU::new(vec![5, 2, 3], 0.0).unwrap();
    layer.set_weights(strided.clone()).unwrap();
    assert_allclose(&slopes_of(&layer), &strided, 0.0_f32);

    layer.forward(&Tensor::ones(IxDyn(&[5, 2, 3]))).unwrap();
    layer.backward(&Tensor::ones(IxDyn(&[5, 2, 3]))).unwrap();
    assert_eq!(layer.parameters()[0].value.len(), 6);
}

// The parallel gate

/// Above the elementwise gate the layer must give the same bits as below it
///
/// The whole batch clears the gate. 1 sample of it does not, so each single-sample pass runs on
/// the serial path. Both passes are elementwise, so the 2 paths must agree exactly
#[test]
fn p_relu_parallel_path_matches_the_serial_path() {
    let (samples, rows, cols) = (4usize, 1024usize, 1024usize);
    let work = samples * rows * cols;
    let gate = rustyml::tuning::elementwise::get_cheap_map_f32();
    assert!(work >= gate, "the whole batch must clear the gate");
    assert!(work / samples < gate, "1 sample must stay under the gate");

    let data: Vec<f32> = (0..work).map(|k| (k % 17) as f32 - 8.0).collect();
    let x = tensor(&[samples, rows, cols], data.clone());
    let g = tensor(
        &[samples, rows, cols],
        data.iter().map(|v| v * 0.5).collect(),
    );

    let mut layer = PReLU::new(vec![samples, rows, cols], 0.25)
        .unwrap()
        .with_shared_axes(vec![1])
        .unwrap();
    let parallel_out = layer.forward(&x).unwrap();
    let parallel_grad = layer.backward(&g).unwrap();

    let sample_elements = rows * cols;
    for sample in 0..samples {
        let span = sample * sample_elements..(sample + 1) * sample_elements;
        let xs = tensor(&[1, rows, cols], data[span.clone()].to_vec());
        let gs = tensor(
            &[1, rows, cols],
            data[span.clone()].iter().map(|v| v * 0.5).collect(),
        );

        let serial_out = layer.forward(&xs).unwrap();
        let serial_grad = layer.backward(&gs).unwrap();
        assert_eq!(
            serial_out.as_slice().unwrap(),
            &parallel_out.as_slice().unwrap()[span.clone()],
            "sample {sample} forward"
        );
        assert_eq!(
            serial_grad.as_slice().unwrap(),
            &parallel_grad.as_slice().unwrap()[span],
            "sample {sample} input gradient"
        );
    }
}

// Input validation

/// The rank is fixed by the slope array, so a tensor of another rank cannot broadcast
#[test]
fn p_relu_rejects_an_input_of_the_wrong_rank() {
    let mut layer = PReLU::new(vec![2, 3, 4], 0.25).unwrap();
    for shape in [vec![2usize, 3], vec![2, 3, 4, 1], vec![24]] {
        let err = layer.forward(&Tensor::ones(IxDyn(&shape))).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "{shape:?} must be rejected, got {err:?}"
        );
    }
}

/// An axis that is not shared holds 1 slope per position, so its extent is fixed
#[test]
fn p_relu_rejects_a_mismatched_extent_on_an_axis_that_is_not_shared() {
    let mut layer = PReLU::new(vec![2, 3, 4], 0.25).unwrap();
    for shape in [vec![2usize, 5, 4], vec![2, 3, 7]] {
        let err = layer.forward(&Tensor::ones(IxDyn(&shape))).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "{shape:?} must be rejected, got {err:?}"
        );
    }
}

/// A shared axis carries 1 slope for every extent, so the layer serves several input sizes
#[test]
fn p_relu_accepts_any_extent_on_a_shared_axis() {
    let mut layer = p_relu_with(vec![1, 4, 4, 2], vec![1, 2], vec![0.5, -0.25]);
    for (height, width) in [(4usize, 4usize), (1, 9), (7, 2), (13, 13)] {
        let x = Tensor::from_elem(IxDyn(&[2, height, width, 2]), -2.0);
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, height, width, 2]);
        assert_eq!(out[[0, 0, 0, 0]], -1.0);
        assert_eq!(out[[0, 0, 0, 1]], 0.5);
    }
}

/// The layer never checks the batch axis, so a partial final mini-batch still passes
#[test]
fn p_relu_accepts_any_batch_size() {
    let mut layer = PReLU::new(vec![8, 3], 0.25).unwrap();
    for batch in [1usize, 3, 8, 40] {
        let out = layer
            .forward(&Tensor::from_elem(IxDyn(&[batch, 3]), -4.0))
            .unwrap();
        assert_eq!(out.shape(), &[batch, 3]);
        assert!(out.iter().all(|&v| v == -1.0));
    }
}

/// An input with no element has nothing to activate
#[test]
fn p_relu_rejects_an_empty_input() {
    let mut layer = PReLU::new(vec![2, 3], 0.25).unwrap();
    let err = layer.forward(&Tensor::zeros(IxDyn(&[0, 3]))).unwrap_err();
    assert!(matches!(err, Error::EmptyInput(_)), "{err:?}");
}

// Error paths on backward

/// Backward needs the cached input, and the message names the layer
#[test]
fn p_relu_rejects_backward_before_forward() {
    let mut layer = PReLU::new(vec![2, 3], 0.25).unwrap();
    let err = layer.backward(&Tensor::ones(IxDyn(&[2, 3]))).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("PReLU"), "{text}");
}

/// The layer keeps the shape, so the upstream gradient must match the cached input
#[test]
fn p_relu_rejects_a_gradient_of_the_wrong_shape() {
    let mut layer = PReLU::new(vec![2, 3], 0.25).unwrap();
    layer.forward(&Tensor::ones(IxDyn(&[2, 3]))).unwrap();
    let err = layer.backward(&Tensor::ones(IxDyn(&[2, 4]))).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }), "{err:?}");
}

// The weight surface

/// `set_weights` guards the slope shape, which a shared axis changes
#[test]
fn p_relu_set_weights_rejects_a_mismatched_shape() {
    let mut layer = PReLU::new(vec![2, 3, 4], 0.25)
        .unwrap()
        .with_shared_axes(vec![1])
        .unwrap();
    // The layer holds [1, 4]. The unshared shape [3, 4] is what a caller reaches for first
    for shape in [vec![3usize, 4], vec![4], vec![1, 5]] {
        let err = layer.set_weights(Tensor::zeros(IxDyn(&shape))).unwrap_err();
        assert!(
            matches!(
                err,
                Error::NeuralNetwork(NnError::WeightShape { ref name, .. }) if name == "alpha"
            ),
            "{shape:?} must be rejected, got {err:?}"
        );
    }
    layer.set_weights(Tensor::zeros(IxDyn(&[1, 4]))).unwrap();
}

/// The reported type name and output shape are what `summary()` prints
#[test]
fn p_relu_reports_its_type_and_output_shape() {
    let mut layer = PReLU::new(vec![2, 3, 4], 0.25).unwrap();
    assert_eq!(layer.layer_type(), "PReLU");
    // Before the first forward pass the configured shape is all the layer knows
    assert_eq!(layer.output_shape(), "(2, 3, 4)");

    layer.forward(&Tensor::ones(IxDyn(&[7, 3, 4]))).unwrap();
    assert_eq!(layer.output_shape(), "(7, 3, 4)");
}

/// `predict` runs the same transform as `forward` and writes no cache
#[test]
fn p_relu_predict_equals_forward_and_caches_nothing() {
    let mut layer = p_relu_with(vec![2, 3], vec![], vec![0.25, -0.1, 0.5]);
    let x = tensor(&[2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 0.0]);

    let predicted = layer.predict(&x).unwrap();
    let err = layer.backward(&Tensor::ones(x.raw_dim())).unwrap_err();
    assert!(err.to_string().contains("PReLU"), "predict must not cache");

    let forwarded = layer.forward(&x).unwrap();
    assert_allclose(&predicted, &forwarded, 0.0_f32);
}

// Training

/// The slopes move under an optimizer, and each one moves on its own
#[test]
fn p_relu_trains_its_slopes() {
    let x = Array2::from_shape_vec(
        (4, 3),
        vec![
            -1.0, 0.5, -2.0, 1.0, -0.5, 2.0, -3.0, 1.5, -1.0, 0.25, -2.5, 0.75,
        ],
    )
    .unwrap()
    .into_dyn();
    let y = Array2::from_shape_vec((4, 1), vec![0.5f32, -0.5, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(3, 3, Linear::new()).unwrap())
        .add(PReLU::new(vec![4, 3], 0.25).unwrap())
        .add(Dense::new(3, 1, Linear::new()).unwrap())
        .compile(
            SGD::new(0.05, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    model.fit(&x, &y, 20).unwrap();

    let weights = model.get_weights();
    let LayerWeight::PReLU(w) = &weights[1] else {
        panic!("layer 1 must be the PReLU layer");
    };
    let trained: Vec<f32> = w.alpha.iter().cloned().collect();
    assert_eq!(trained.len(), 3);
    assert!(
        trained.iter().any(|&v| (v - 0.25).abs() > 1e-4),
        "the slopes must move away from the starting 0.25, got {trained:?}"
    );
    assert!(
        trained.iter().all(|v| v.is_finite()),
        "the slopes must stay finite, got {trained:?}"
    );
}

/// A model with a per-channel slope trains end to end on a 4-D input
#[test]
fn p_relu_trains_with_shared_spatial_axes() {
    let x: Tensor = Array::from_shape_vec(
        (2, 3, 3, 2),
        (0..36).map(|v| 0.1 * v as f32 - 1.8).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let mut layer = PReLU::new(vec![2, 3, 3, 2], 0.25)
        .unwrap()
        .with_shared_axes(vec![1, 2])
        .unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(2));

    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3, 3, 2]);
    let grad = layer.backward(&Tensor::ones(x.raw_dim())).unwrap();
    assert_eq!(grad.shape(), &[2, 3, 3, 2]);
    assert_eq!(slope_gradient(&mut layer).len(), 2);
}

/// A 1-D slope array assigned through the container reaches the live layer
#[test]
fn p_relu_get_weights_borrows_the_live_slopes() {
    let mut layer = PReLU::new(vec![2, 3], 0.25).unwrap();
    layer
        .set_weights(Array1::from_vec(vec![0.1f32, 0.2, 0.3]).into_dyn())
        .unwrap();

    let LayerWeight::PReLU(w) = layer.get_weights() else {
        panic!("PReLU must report LayerWeight::PReLU");
    };
    assert_eq!(w.alpha.shape(), &[3]);
    assert_eq!(
        w.alpha.iter().cloned().collect::<Vec<_>>(),
        vec![0.1, 0.2, 0.3]
    );
}

/// A 4-D per-channel slope array keeps its rank through the container
#[test]
fn p_relu_get_weights_keeps_a_shared_axis_at_extent_1() {
    let layer = PReLU::new(vec![2, 4, 4, 3], 0.25)
        .unwrap()
        .with_shared_axes(vec![1, 2])
        .unwrap();
    let LayerWeight::PReLU(w) = layer.get_weights() else {
        panic!("PReLU must report LayerWeight::PReLU");
    };
    assert_eq!(w.alpha.shape(), &[1, 1, 3]);

    // The same slopes as a plain 3-element array do not fit
    let mut layer = layer;
    assert!(layer.set_weights(Array1::zeros(3).into_dyn()).is_err());
    assert!(
        layer
            .set_weights(Array4::zeros((1, 1, 3, 1)).into_dyn())
            .is_err()
    );
}
