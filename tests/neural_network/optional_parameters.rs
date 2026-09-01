//! Integration tests for the optional parameters of a layer: `use_bias` on Dense and on the
//! convolution family, and `center` and `scale` on the normalization layers.
//!
//! A name-addressed checkpoint can express an array that a layer does not hold, which the
//! closed weight enum could not. These tests pin the 3 defects that an optional parameter
//! makes possible:
//!
//! 1. **A layer that yields nothing.** The parameter walk used to gate every push behind 1
//!    all-or-nothing match on every gradient. A bias that is never present then held back the
//!    kernel of the same layer, and the layer trained nothing with no error anywhere.
//! 2. **A dropped array that renumbers the arrays after it.** On the normalization layers the
//!    optional array is the first one, so `scale = false` used to move `beta` into the slot of
//!    `gamma`. A positional key then gave `beta` the optimizer state of `gamma`.
//! 3. **A parameter count read from the configuration.** A count derived from `input_dim` and
//!    `units` counts a bias that the layer does not hold.
//!
//! The tests also pin both directions of the checkpoint refusal, and the Keras 3 reference
//! values for a bias-free Dense and a scale-free LayerNormalization.

use crate::common::assert_allclose;
use ndarray::{Array, Array1, Array2, Array3, Array4, Array5, IxDyn};
use rustyml::error::{Error, IoError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_1d_transpose::Conv1DTranspose;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_2d_transpose::Conv2DTranspose;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::conv_3d_transpose::Conv3DTranspose;
use rustyml::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::layers::regularization::normalization::layer_normalization::LayerNormalization;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::{Layer, Optimizer, WeightKind};

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

/// Temporary file that deletes itself when dropped
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(name: &str) -> Self {
        TempFile(std::env::temp_dir().join(format!("rustyml_optional_param_{name}.bin")))
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The deterministic f32 sequence that the Keras oracle also used
fn val(i: usize) -> f32 {
    let n = (i * 37 + 11) % 101;
    (n as f32) / 100.0 - 0.5
}

/// `n` values of [`val`], starting at `offset`
fn seq(n: usize, offset: usize) -> Vec<f32> {
    (0..n).map(|i| val(i + offset)).collect()
}

/// A tensor of the given shape, filled from [`seq`]
fn tensor(shape: &[usize], offset: usize) -> Tensor {
    let n: usize = shape.iter().product();
    Array::from_shape_vec(IxDyn(shape), seq(n, offset)).unwrap()
}

/// Every array name that a layer exposes, in order
fn weight_names(layer: &dyn Layer) -> Vec<&'static str> {
    layer.weights().iter().map(|entry| entry.name).collect()
}

/// Every parameter name that a layer yields, in order
fn parameter_names(layer: &mut dyn Layer) -> Vec<&'static str> {
    layer.parameters().iter().map(|pg| pg.name).collect()
}

/// Runs 1 forward pass and 1 backward pass, and gives the parameter names back
fn train_once(layer: &mut dyn Layer, input: &Tensor, upstream: &Tensor) -> Vec<&'static str> {
    layer.forward(input).unwrap();
    layer.backward(upstream).unwrap();
    parameter_names(layer)
}

/// Every element of 1 named array of a layer
fn array_of(layer: &dyn Layer, name: &str) -> Vec<f32> {
    layer
        .weight(name)
        .unwrap_or_else(|| panic!("layer holds no array named `{name}`"))
        .iter()
        .copied()
        .collect()
}

// ---------------------------------------------------------------------------------------
// Trap 1: a layer without a bias must still train its kernel
// ---------------------------------------------------------------------------------------

/// The former all-or-nothing gate returned the empty parameter list whenever any 1 gradient of
/// a layer was absent. Without a bias the bias gradient is absent forever, so the kernel never
/// reached the optimizer and the layer trained nothing. This trains a bias-free Dense and
/// checks that the kernel really moves
#[test]
fn a_bias_free_dense_trains_its_kernel() {
    let mut layer = Dense::new(3, 2, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    layer
        .set_weights(Array2::from_shape_vec((3, 2), seq(6, 5)).unwrap(), None)
        .unwrap();

    let mut model = Sequential::new();
    model.add(layer);
    model.compile(
        SGD::new(0.1, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let before: Vec<f32> = model.weight("0.kernel").unwrap().iter().copied().collect();
    let x = tensor(&[4, 3], 0);
    let y = tensor(&[4, 2], 60);
    model.fit(&x, &y, 3).unwrap();
    let after: Vec<f32> = model.weight("0.kernel").unwrap().iter().copied().collect();

    assert_eq!(before.len(), 6);
    let moved = before
        .iter()
        .zip(after.iter())
        .filter(|(b, a)| (*b - *a).abs() > 1e-6)
        .count();
    assert_eq!(
        moved, 6,
        "every kernel element must move: before {before:?}, after {after:?}"
    );
}

/// The same defect on the convolution side, where the bias is also the last array
#[test]
fn a_bias_free_conv2d_trains_its_kernel() {
    let mut layer = Conv2D::new(2, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 2, 2), seq(16, 5)).unwrap(),
            None,
        )
        .unwrap();

    let mut model = Sequential::new();
    model.add(layer);
    model.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let before: Vec<f32> = model.weight("0.kernel").unwrap().iter().copied().collect();
    let x = tensor(&[1, 4, 4, 2], 0);
    let y = tensor(&[1, 3, 3, 2], 80);
    model.fit(&x, &y, 3).unwrap();
    let after: Vec<f32> = model.weight("0.kernel").unwrap().iter().copied().collect();

    let moved = before
        .iter()
        .zip(after.iter())
        .filter(|(b, a)| (*b - *a).abs() > 1e-6)
        .count();
    assert_eq!(moved, before.len(), "every kernel element must move");
}

/// 1 bias-free layer, the input it takes, and the arrays it must expose
struct BiasFreeCase {
    /// Name of the layer, for the failure message
    name: &'static str,
    /// The layer, built with `use_bias` set to false
    layer: Box<dyn Layer>,
    /// An input of the shape the layer was built for
    input: Tensor,
    /// The arrays the layer must expose, in order
    arrays: Vec<&'static str>,
}

/// Every layer that takes `use_bias` must still yield its kernel with the bias left out. This
/// is the same defect as the 2 tests above, over the whole family in 1 pass
#[test]
fn every_bias_free_layer_yields_its_kernels() {
    let x3 = tensor(&[1, 6, 2], 0);
    let x4 = tensor(&[1, 4, 4, 2], 0);
    let x5 = tensor(&[1, 3, 3, 3, 2], 0);

    let cases: Vec<BiasFreeCase> = vec![
        BiasFreeCase {
            name: "Dense",
            layer: Box::new(
                Dense::new(2, 3, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: tensor(&[2, 2], 0),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "Conv1D",
            layer: Box::new(
                Conv1D::new(3, 2, vec![1, 6, 2], 1, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x3.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "Conv2D",
            layer: Box::new(
                Conv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x4.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "Conv3D",
            layer: Box::new(
                Conv3D::new(
                    3,
                    (2, 2, 2),
                    vec![1, 3, 3, 3, 2],
                    (1, 1, 1),
                    Activation::Linear,
                )
                .unwrap()
                .with_use_bias(false),
            ),
            input: x5.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "Conv1DTranspose",
            layer: Box::new(
                Conv1DTranspose::new(3, 2, vec![1, 6, 2], 1, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x3.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "Conv2DTranspose",
            layer: Box::new(
                Conv2DTranspose::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x4.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "Conv3DTranspose",
            layer: Box::new(
                Conv3DTranspose::new(
                    3,
                    (2, 2, 2),
                    vec![1, 3, 3, 3, 2],
                    (1, 1, 1),
                    Activation::Linear,
                )
                .unwrap()
                .with_use_bias(false),
            ),
            input: x5.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "DepthwiseConv1D",
            layer: Box::new(
                DepthwiseConv1D::new(2, vec![1, 6, 2], 1, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x3.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "DepthwiseConv2D",
            layer: Box::new(
                DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x4.clone(),
            arrays: vec!["kernel"],
        },
        BiasFreeCase {
            name: "SeparableConv1D",
            layer: Box::new(
                SeparableConv1D::new(3, 2, vec![1, 6, 2], 1, 1, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x3.clone(),
            arrays: vec!["depthwise_kernel", "pointwise_kernel"],
        },
        BiasFreeCase {
            name: "SeparableConv2D",
            layer: Box::new(
                SeparableConv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
            input: x4.clone(),
            arrays: vec!["depthwise_kernel", "pointwise_kernel"],
        },
    ];

    assert_eq!(
        cases.len(),
        11,
        "every layer that takes use_bias is covered"
    );

    for case in cases {
        let BiasFreeCase {
            name,
            mut layer,
            input,
            arrays,
        } = case;
        assert_eq!(
            weight_names(&*layer),
            arrays,
            "{name} exposes the wrong arrays without a bias"
        );
        let output = layer.forward(&input).unwrap();
        let upstream = Tensor::ones(output.raw_dim());
        layer.backward(&upstream).unwrap();
        assert_eq!(
            parameter_names(&mut *layer),
            arrays,
            "{name} yields the wrong parameters without a bias"
        );
        for pg in layer.parameters() {
            assert!(
                pg.grad.iter().any(|g| g.abs() > 0.0),
                "{name} yielded an all-zero gradient for `{}`",
                pg.name
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// Trap 2: on a normalization layer the optional array is the first one
// ---------------------------------------------------------------------------------------

/// `scale = false` drops `gamma`, which is the array at index 0. A positional key would then
/// give `beta` the optimizer state of `gamma`.
///
/// The check needs no formula. The input and the upstream gradient are built so that the
/// gradient of `beta` is exactly 0 and the gradient of `gamma` is not. 1 SGD step with
/// momentum on a full layer therefore fills the momentum buffer of `gamma` and leaves the
/// buffer of `beta` at 0. A second step on a scale-free layer must then leave `beta` exactly
/// where it was. With a positional key it would move by the momentum of `gamma`
#[test]
fn dropping_gamma_does_not_give_beta_the_optimizer_state_of_gamma() {
    // Row 1 is row 0 reversed, so the 2 normalized rows differ elementwise
    let x: Tensor =
        Array::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
            .unwrap();
    // The 2 rows cancel, so every column sum is exactly 0
    let upstream: Tensor = Array::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    )
    .unwrap();

    let mut optimizer = SGD::new(0.1, 0.9, false, 0.0).unwrap();

    // Step 1 fills the momentum buffer of `gamma`, and leaves the buffer of `beta` at 0
    let mut full = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();
    assert_eq!(train_once(&mut full, &x, &upstream), vec!["gamma", "beta"]);
    for pg in full.parameters() {
        let all_zero = pg.grad.iter().all(|g| *g == 0.0);
        match pg.name {
            "gamma" => assert!(!all_zero, "the gamma gradient must not be all zero"),
            "beta" => assert!(all_zero, "the beta gradient must be exactly zero"),
            other => panic!("unexpected parameter `{other}`"),
        }
    }
    optimizer.step();
    optimizer.update(0, &mut full, 1.0);
    assert!(
        array_of(&full, "gamma").iter().any(|v| *v != 1.0),
        "gamma must have moved, or the buffer of gamma holds nothing"
    );
    assert!(
        array_of(&full, "beta").iter().all(|v| *v == 0.0),
        "beta must not have moved on a zero gradient"
    );

    // Step 2 runs a scale-free layer of the same shape through the same optimizer, at the same
    // scope. `beta` is the only parameter, and it sits at index 0
    let mut scale_free = LayerNormalization::new(vec![2, 4], 1e-5)
        .unwrap()
        .with_scale(false);
    assert_eq!(weight_names(&scale_free), vec!["beta"]);
    assert_eq!(train_once(&mut scale_free, &x, &upstream), vec!["beta"]);
    optimizer.step();
    optimizer.update(0, &mut scale_free, 1.0);

    assert_eq!(
        array_of(&scale_free, "beta"),
        vec![0.0_f32; 4],
        "beta took the momentum of gamma, so the optimizer key is positional"
    );
}

/// Dropping the first array of a layer moves no path of the arrays after it. The running
/// statistics of BatchNormalization are the case that matters, because they follow `gamma` and
/// `beta` and a positional key would renumber them
#[test]
fn dropping_an_array_moves_no_other_checkpoint_path() {
    let mut both = Sequential::new();
    both.add(BatchNormalization::new(vec![2, 4], 0.9, 1e-5).unwrap());
    assert_eq!(
        both.weight_paths(),
        vec!["0.gamma", "0.beta", "0.moving_mean", "0.moving_variance"]
    );

    let mut no_scale = Sequential::new();
    no_scale.add(
        BatchNormalization::new(vec![2, 4], 0.9, 1e-5)
            .unwrap()
            .with_scale(false),
    );
    assert_eq!(
        no_scale.weight_paths(),
        vec!["0.beta", "0.moving_mean", "0.moving_variance"]
    );

    let mut neither = Sequential::new();
    neither.add(
        BatchNormalization::new(vec![2, 4], 0.9, 1e-5)
            .unwrap()
            .with_center(false)
            .with_scale(false),
    );
    assert_eq!(
        neither.weight_paths(),
        vec!["0.moving_mean", "0.moving_variance"]
    );
}

/// Builds 1 normalization layer with the given `center` and `scale` flags
type MakeNormalization = fn(bool, bool) -> Box<dyn Layer>;

/// Each of the 4 normalization layers drops the array that its flag names, and keeps the other
#[test]
fn every_normalization_layer_drops_the_array_its_flag_names() {
    let make: [(&str, MakeNormalization); 4] = [
        ("BatchNormalization", |center, scale| {
            Box::new(
                BatchNormalization::new(vec![2, 4], 0.9, 1e-5)
                    .unwrap()
                    .with_center(center)
                    .with_scale(scale),
            )
        }),
        ("LayerNormalization", |center, scale| {
            Box::new(
                LayerNormalization::new(vec![2, 4], 1e-5)
                    .unwrap()
                    .with_center(center)
                    .with_scale(scale),
            )
        }),
        ("GroupNormalization", |center, scale| {
            Box::new(
                GroupNormalization::new(vec![2, 3, 4], 2, 1e-5)
                    .unwrap()
                    .with_center(center)
                    .with_scale(scale),
            )
        }),
        ("InstanceNormalization", |center, scale| {
            Box::new(
                InstanceNormalization::new(vec![2, 3, 4], 1e-5)
                    .unwrap()
                    .with_center(center)
                    .with_scale(scale),
            )
        }),
    ];

    for (name, build) in make {
        let trainable = |center: bool, scale: bool| -> Vec<&'static str> {
            build(center, scale)
                .weights()
                .iter()
                .filter(|entry| entry.kind == WeightKind::Trainable)
                .map(|entry| entry.name)
                .collect()
        };
        assert_eq!(trainable(true, true), vec!["gamma", "beta"], "{name}");
        assert_eq!(trainable(true, false), vec!["beta"], "{name}");
        assert_eq!(trainable(false, true), vec!["gamma"], "{name}");
        assert!(trainable(false, false).is_empty(), "{name}");
    }
}

// ---------------------------------------------------------------------------------------
// Trap 3: the parameter count must read the arrays, not the configuration
// ---------------------------------------------------------------------------------------

/// A count derived from `input_dim` and `units` counts a bias that the layer does not hold.
/// This compares every count against the arrays that the same layer exposes, so a formula that
/// reads the configuration fails here as soon as an array disappears
#[test]
fn param_count_matches_the_arrays_a_layer_holds() {
    let layers: Vec<(&str, Box<dyn Layer>)> = vec![
        (
            "Dense with a bias",
            Box::new(Dense::new(3, 4, Activation::Linear).unwrap()),
        ),
        (
            "Dense without a bias",
            Box::new(
                Dense::new(3, 4, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
        ),
        (
            "Conv2D without a bias",
            Box::new(
                Conv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
        ),
        (
            "SeparableConv2D without a bias",
            Box::new(
                SeparableConv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
        ),
        (
            "DepthwiseConv2D without a bias",
            Box::new(
                DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
                    .unwrap()
                    .with_use_bias(false),
            ),
        ),
        (
            "BatchNormalization without gamma",
            Box::new(
                BatchNormalization::new(vec![2, 4], 0.9, 1e-5)
                    .unwrap()
                    .with_scale(false),
            ),
        ),
        (
            "LayerNormalization without beta",
            Box::new(
                LayerNormalization::new(vec![2, 4], 1e-5)
                    .unwrap()
                    .with_center(false),
            ),
        ),
        (
            "GroupNormalization without either",
            Box::new(
                GroupNormalization::new(vec![2, 3, 4], 2, 1e-5)
                    .unwrap()
                    .with_center(false)
                    .with_scale(false),
            ),
        ),
        (
            "InstanceNormalization without gamma",
            Box::new(
                InstanceNormalization::new(vec![2, 3, 4], 1e-5)
                    .unwrap()
                    .with_scale(false),
            ),
        ),
        (
            "SimpleRNN",
            Box::new(
                rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN::new(
                    2,
                    3,
                    Activation::Tanh,
                )
                .unwrap(),
            ),
        ),
        (
            "LSTM",
            Box::new(
                rustyml::neural_network::layers::recurrent::lstm::LSTM::new(2, 3, Activation::Tanh)
                    .unwrap(),
            ),
        ),
        (
            "GRU",
            Box::new(
                rustyml::neural_network::layers::recurrent::gru::GRU::new(2, 3, Activation::Tanh)
                    .unwrap(),
            ),
        ),
        (
            "Embedding",
            Box::new(rustyml::neural_network::layers::embedding::Embedding::new(5, 3).unwrap()),
        ),
    ];

    for (name, layer) in layers {
        let mut trainable = 0;
        let mut non_trainable = 0;
        for entry in layer.weights() {
            match entry.kind {
                WeightKind::Trainable => trainable += entry.value.len(),
                WeightKind::NonTrainable => non_trainable += entry.value.len(),
            }
        }
        let counts = layer.param_count();
        assert_eq!(counts.trainable, trainable, "{name}: trainable count");
        assert_eq!(
            counts.non_trainable, non_trainable,
            "{name}: non-trainable count"
        );
    }
}

/// The counts of the optional configurations, against Keras 3.15.1 read through its own
/// `count_params`. A bias-free Dense of 3 inputs and 4 units counts 12, not 16
#[test]
fn param_count_agrees_with_keras_for_the_optional_configurations() {
    let dense = Dense::new(3, 4, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    assert_eq!(dense.param_count().trainable, 12);
    assert_eq!(dense.param_count().non_trainable, 0);

    let conv = Conv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    assert_eq!(conv.param_count().trainable, 24);

    let separable =
        SeparableConv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Activation::Linear)
            .unwrap()
            .with_use_bias(false);
    assert_eq!(separable.param_count().trainable, 14);

    let depthwise = DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    assert_eq!(depthwise.param_count().trainable, 8);

    // Keras reports 12 total for a scale-free BatchNormalization of 4 channels: 4 trainable
    // and 8 non-trainable
    let batch = BatchNormalization::new(vec![2, 4], 0.9, 1e-5)
        .unwrap()
        .with_scale(false);
    assert_eq!(batch.param_count().trainable, 4);
    assert_eq!(batch.param_count().non_trainable, 8);
    assert_eq!(batch.param_count().total(), 12);
}

// ---------------------------------------------------------------------------------------
// The Keras 3 reference values for the optional-parameter path
// ---------------------------------------------------------------------------------------

/// A bias-free Dense against Keras 3.15.1 on the jax backend, forward and backward. The kernel,
/// the input, and the upstream gradient come from the same deterministic sequence on both sides
#[test]
fn a_bias_free_dense_matches_keras() {
    let mut layer = Dense::new(3, 4, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    layer
        .set_weights(Array2::from_shape_vec((3, 4), seq(12, 50)).unwrap(), None)
        .unwrap();

    let x = tensor(&[2, 3], 0);
    let upstream = tensor(&[2, 4], 200);
    let output = layer.forward(&x).unwrap();
    let grad_input = layer.backward(&upstream).unwrap();

    let wanted_output: Tensor = Array::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            -0.029700005,
            -0.03169999,
            -0.013500013,
            -0.035699993,
            -0.0107,
            -0.0027000038,
            -0.07549999,
            0.013300005,
        ],
    )
    .unwrap();
    assert_allclose(&output, &wanted_output, 1e-6_f32);

    let wanted_grad_input: Tensor = Array::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![
            0.21540001,
            -0.16870001,
            0.235,
            -0.12520002,
            0.445,
            -0.16649999,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &wanted_grad_input, 1e-6_f32);

    let wanted_grad_kernel = [
        -0.054700017,
        -0.013399984,
        0.12889999,
        -0.122700006,
        0.030399997,
        -0.028199999,
        0.014200001,
        0.03639999,
        0.1155,
        -0.043,
        -0.10050001,
        0.19549999,
    ];
    let params = layer.parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].name, "kernel");
    for (got, wanted) in params[0].grad.iter().zip(wanted_grad_kernel.iter()) {
        assert!(
            (got - wanted).abs() < 1e-6,
            "kernel gradient {got} differs from the Keras value {wanted}"
        );
    }
}

/// A scale-free LayerNormalization against the same oracle. Keras drops `gamma` and keeps
/// `beta`, and the forward output is then the normalized value plus `beta`
#[test]
fn a_scale_free_layer_normalization_matches_keras() {
    let mut layer = LayerNormalization::new(vec![2, 4], 1e-5)
        .unwrap()
        .with_scale(false);

    let x = tensor(&[2, 4], 20);
    let upstream = tensor(&[2, 4], 700);
    let output = layer.forward(&x).unwrap();
    layer.backward(&upstream).unwrap();

    let wanted_output: Tensor = Array::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            -0.21830113,
            1.3971269,
            -1.3971269,
            0.21830113,
            1.3100104,
            -0.53264153,
            0.53264153,
            -1.3100104,
        ],
    )
    .unwrap();
    assert_allclose(&output, &wanted_output, 1e-5_f32);

    let wanted_grad_beta = [-0.44, 0.3, 0.030000001, -0.24000001];
    let params = layer.parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].name, "beta");
    for (got, wanted) in params[0].grad.iter().zip(wanted_grad_beta.iter()) {
        assert!(
            (got - wanted).abs() < 1e-6,
            "beta gradient {got} differs from the Keras value {wanted}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// A checkpoint of a layer with an optional array, in both directions
// ---------------------------------------------------------------------------------------

/// The message of a structural refusal
fn refusal(result: Result<(), Error>) -> String {
    match result {
        Err(Error::Io(IoError::ModelStructureMismatch(message))) => message,
        other => panic!("expected a structural mismatch, got {other:?}"),
    }
}

/// A checkpoint written by a Dense that holds a bias must not load into a Dense that holds
/// none, and the refusal must name both rosters
#[test]
fn a_checkpoint_with_a_bias_fails_to_load_into_a_bias_free_layer() {
    let file = TempFile::new("dense_bias_into_bias_free");

    let mut with_bias = Sequential::new();
    with_bias.add(Dense::new(3, 4, Activation::Linear).unwrap());
    with_bias.save_to_path(file.path()).unwrap();

    let mut bias_free = Sequential::new();
    bias_free.add(
        Dense::new(3, 4, Activation::Linear)
            .unwrap()
            .with_use_bias(false),
    );
    let message = refusal(bias_free.load_from_path(file.path()));
    assert!(message.contains("layer 0"), "{message}");
    assert!(message.contains("Dense"), "{message}");
    assert!(message.contains("bias"), "{message}");

    // The refusal writes nothing, so the model still holds the kernel it started with
    assert_eq!(bias_free.weight_paths(), vec!["0.kernel"]);
}

/// The other direction: a checkpoint written without a bias must not load into a layer that
/// holds one, because the bias would keep whatever it already held
#[test]
fn a_bias_free_checkpoint_fails_to_load_into_a_layer_with_a_bias() {
    let file = TempFile::new("bias_free_into_dense_bias");

    let mut bias_free = Sequential::new();
    bias_free.add(
        Dense::new(3, 4, Activation::Linear)
            .unwrap()
            .with_use_bias(false),
    );
    bias_free.save_to_path(file.path()).unwrap();

    let mut with_bias = Sequential::new();
    with_bias.add(Dense::new(3, 4, Activation::Linear).unwrap());
    let message = refusal(with_bias.load_from_path(file.path()));
    assert!(message.contains("layer 0"), "{message}");
    assert!(message.contains("bias"), "{message}");
}

/// 2 layers can hold the same number of arrays and still disagree. A scale-free
/// LayerNormalization holds `beta` alone and a center-free one holds `gamma` alone, so the
/// refusal comes from the name and it must name the checkpoint path
#[test]
fn a_checkpoint_of_the_other_optional_array_fails_by_path() {
    let file = TempFile::new("layer_norm_beta_into_gamma");

    let mut scale_free = Sequential::new();
    scale_free.add(
        LayerNormalization::new(vec![2, 4], 1e-5)
            .unwrap()
            .with_scale(false),
    );
    scale_free.save_to_path(file.path()).unwrap();

    let mut center_free = Sequential::new();
    center_free.add(
        LayerNormalization::new(vec![2, 4], 1e-5)
            .unwrap()
            .with_center(false),
    );
    let message = refusal(center_free.load_from_path(file.path()));
    assert!(message.contains("0.gamma"), "{message}");
    assert!(message.contains("0.beta"), "{message}");
}

/// A checkpoint of a bias-free layer round-trips into another bias-free layer, so the refusals
/// above come from the roster and not from the format
#[test]
fn a_bias_free_checkpoint_round_trips() {
    let file = TempFile::new("bias_free_round_trip");

    let mut source = Sequential::new();
    let mut layer = Dense::new(3, 4, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    layer
        .set_weights(Array2::from_shape_vec((3, 4), seq(12, 50)).unwrap(), None)
        .unwrap();
    source.add(layer);
    source.save_to_path(file.path()).unwrap();

    let mut target = Sequential::new();
    target.add(
        Dense::new(3, 4, Activation::Linear)
            .unwrap()
            .with_use_bias(false),
    );
    target.load_from_path(file.path()).unwrap();

    assert_eq!(target.weight_paths(), vec!["0.kernel"]);
    let loaded: Vec<f32> = target.weight("0.kernel").unwrap().iter().copied().collect();
    assert_eq!(loaded, seq(12, 50));
}

/// The lenient load reports the array that the file holds and the model does not
#[test]
fn a_partial_load_reports_the_array_the_model_dropped() {
    let file = TempFile::new("partial_gamma");

    let mut full = Sequential::new();
    full.add(LayerNormalization::new(vec![2, 4], 1e-5).unwrap());
    full.save_to_path(file.path()).unwrap();

    let mut center_free = Sequential::new();
    center_free.add(
        LayerNormalization::new(vec![2, 4], 1e-5)
            .unwrap()
            .with_center(false),
    );
    let report = center_free.load_partial_from_path(file.path()).unwrap();
    assert_eq!(report.applied, vec!["0.gamma"]);
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    assert_eq!(report.unused, vec!["0.beta"]);
}

// ---------------------------------------------------------------------------------------
// The setters refuse a value that reaches nothing
// ---------------------------------------------------------------------------------------

/// A bias given to a layer that holds none reaches nothing, so the setter refuses it by name
#[test]
fn set_weights_refuses_a_bias_that_the_layer_does_not_hold() {
    let mut layer = Dense::new(3, 4, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    let result = layer.set_weights(
        Array2::from_shape_vec((3, 4), seq(12, 50)).unwrap(),
        Array2::zeros((1, 4)),
    );
    match result {
        Err(Error::InvalidParameter { name, reason }) => {
            assert_eq!(name, "bias");
            assert!(reason.contains("use_bias"), "{reason}");
        }
        other => panic!("expected an invalid-parameter error, got {other:?}"),
    }
}

/// The other half of the same rule: a layer that holds a bias must be given one
#[test]
fn set_weights_requires_the_bias_that_the_layer_holds() {
    let mut layer = Dense::new(3, 4, Activation::Linear).unwrap();
    let result = layer.set_weights(Array2::from_shape_vec((3, 4), seq(12, 50)).unwrap(), None);
    match result {
        Err(Error::InvalidParameter { name, reason }) => {
            assert_eq!(name, "bias");
            assert!(reason.contains("use_bias"), "{reason}");
        }
        other => panic!("expected an invalid-parameter error, got {other:?}"),
    }
}

/// The same rule on a normalization layer, for both flags
#[test]
fn set_weights_refuses_a_normalization_array_that_the_layer_does_not_hold() {
    let mut scale_free = LayerNormalization::new(vec![2, 4], 1e-5)
        .unwrap()
        .with_scale(false);
    match scale_free.set_weights(Tensor::ones(IxDyn(&[4])), Tensor::zeros(IxDyn(&[4]))) {
        Err(Error::InvalidParameter { name, reason }) => {
            assert_eq!(name, "gamma");
            assert!(reason.contains("scale"), "{reason}");
        }
        other => panic!("expected an invalid-parameter error, got {other:?}"),
    }
    // The array the layer does hold still takes a value
    scale_free
        .set_weights(
            None,
            Tensor::from_shape_vec(IxDyn(&[4]), seq(4, 9)).unwrap(),
        )
        .unwrap();
    assert_eq!(array_of(&scale_free, "beta"), seq(4, 9));

    let mut center_free = LayerNormalization::new(vec![2, 4], 1e-5)
        .unwrap()
        .with_center(false);
    match center_free.set_weights(Tensor::ones(IxDyn(&[4])), Tensor::zeros(IxDyn(&[4]))) {
        Err(Error::InvalidParameter { name, reason }) => {
            assert_eq!(name, "beta");
            assert!(reason.contains("center"), "{reason}");
        }
        other => panic!("expected an invalid-parameter error, got {other:?}"),
    }
}

/// The conv setters take the same rule, at every rank
#[test]
fn the_conv_setters_refuse_a_bias_that_the_layer_does_not_hold() {
    let mut c1 = Conv1D::new(2, 2, vec![1, 6, 2], 1, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    assert!(
        c1.set_weights(Array3::zeros((2, 2, 2)), Array1::zeros(2))
            .is_err()
    );
    assert!(c1.set_weights(Array3::zeros((2, 2, 2)), None).is_ok());

    let mut c3 = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 3, 3, 3, 2],
        (1, 1, 1),
        Activation::Linear,
    )
    .unwrap()
    .with_use_bias(false);
    assert!(
        c3.set_weights(Array5::zeros((2, 2, 2, 2, 2)), Array1::zeros(2))
            .is_err()
    );
    assert!(c3.set_weights(Array5::zeros((2, 2, 2, 2, 2)), None).is_ok());

    let mut sc = SeparableConv1D::new(2, 2, vec![1, 6, 2], 1, 1, Activation::Linear)
        .unwrap()
        .with_use_bias(false);
    assert!(
        sc.set_weights(
            Array3::zeros((2, 2, 1)),
            Array3::zeros((1, 2, 2)),
            Array1::zeros(2)
        )
        .is_err()
    );
    assert!(
        sc.set_weights(Array3::zeros((2, 2, 1)), Array3::zeros((1, 2, 2)), None)
            .is_ok()
    );
}

// ---------------------------------------------------------------------------------------
// The defaults move nothing
// ---------------------------------------------------------------------------------------

/// Every default keeps the roster that the layers had before the flags existed
#[test]
fn the_defaults_keep_every_array() {
    let dense = Dense::new(3, 4, Activation::Linear).unwrap();
    assert_eq!(weight_names(&dense), vec!["kernel", "bias"]);

    let conv = Conv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear).unwrap();
    assert_eq!(weight_names(&conv), vec!["kernel", "bias"]);

    let separable =
        SeparableConv2D::new(3, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Activation::Linear).unwrap();
    assert_eq!(
        weight_names(&separable),
        vec!["depthwise_kernel", "pointwise_kernel", "bias"]
    );

    let batch = BatchNormalization::new(vec![2, 4], 0.9, 1e-5).unwrap();
    assert_eq!(
        weight_names(&batch),
        vec!["gamma", "beta", "moving_mean", "moving_variance"]
    );
}

// ---------------------------------------------------------------------------------------
// The roster of the parameter walk, over every layer that takes an optional array
// ---------------------------------------------------------------------------------------

/// The 4 settings of `center` and `scale`, in a fixed order
const CENTER_AND_SCALE: [(bool, bool); 4] =
    [(true, true), (true, false), (false, true), (false, false)];

/// The trainable arrays that a normalization layer owns under 1 setting of the 2 flags
///
/// The order is the order the layers declare, which puts the scale before the shift
fn normalization_roster(center: bool, scale: bool) -> Vec<&'static str> {
    let mut names = Vec::new();
    if scale {
        names.push("gamma");
    }
    if center {
        names.push("beta");
    }
    names
}

/// Every normalization layer yields exactly the parameters that its 2 flags leave it
///
/// The gate that drops an array lives in 2 places: `weights` is the checkpoint roster, and
/// `parameters` is the optimizer roster that a backward pass fills. The 2 must name the same
/// trainable arrays. A layer that keeps the gradient of a dropped array hands the optimizer an
/// array that the layer does not own, and the optimizer then writes a buffer that no
/// checkpoint path reaches. A layer that drops the gradient of an array it keeps stops
/// training that array, and nothing reports it
///
/// This walks the 4 normalization layers times the 4 settings of `center` and `scale`, and it
/// reads both rosters after a real forward pass and a real backward pass
#[test]
fn every_normalization_layer_yields_exactly_the_parameters_it_owns() {
    let cases: [(&str, MakeNormalization, &[usize]); 4] = [
        (
            "BatchNormalization",
            |center, scale| {
                Box::new(
                    BatchNormalization::new(vec![4, 4], 0.9, 1e-5)
                        .unwrap()
                        .with_center(center)
                        .with_scale(scale),
                )
            },
            &[4, 4],
        ),
        (
            "LayerNormalization",
            |center, scale| {
                Box::new(
                    LayerNormalization::new(vec![4, 4], 1e-5)
                        .unwrap()
                        .with_center(center)
                        .with_scale(scale),
                )
            },
            &[4, 4],
        ),
        (
            "GroupNormalization",
            |center, scale| {
                Box::new(
                    GroupNormalization::new(vec![2, 3, 4], 2, 1e-5)
                        .unwrap()
                        .with_center(center)
                        .with_scale(scale),
                )
            },
            &[2, 3, 4],
        ),
        (
            "InstanceNormalization",
            |center, scale| {
                Box::new(
                    InstanceNormalization::new(vec![2, 3, 4], 1e-5)
                        .unwrap()
                        .with_center(center)
                        .with_scale(scale),
                )
            },
            &[2, 3, 4],
        ),
    ];

    for (name, build, shape) in cases {
        let input = tensor(shape, 0);
        let upstream = tensor(shape, 47);
        for (center, scale) in CENTER_AND_SCALE {
            let owned = normalization_roster(center, scale);
            let mut layer = build(center, scale);

            let exposed: Vec<&'static str> = layer
                .weights()
                .iter()
                .filter(|entry| entry.kind == WeightKind::Trainable)
                .map(|entry| entry.name)
                .collect();
            assert_eq!(
                exposed, owned,
                "{name} with center {center} and scale {scale} exposes the wrong trainable arrays"
            );

            assert_eq!(
                train_once(&mut *layer, &input, &upstream),
                owned,
                "{name} with center {center} and scale {scale} yields the wrong parameters"
            );

            for entry in layer.parameters() {
                assert_eq!(
                    entry.value.len(),
                    entry.grad.len(),
                    "{name} yielded `{}` against a gradient of another length",
                    entry.name
                );
            }
        }
    }
}

/// Dense and Conv2D yield exactly the parameters that `use_bias` leaves them
///
/// The same roster assertion as above, on the family where the optional array is the last one
/// rather than the first. Both settings of the flag are read, so a gate that is lost in either
/// direction fails here
#[test]
fn the_use_bias_layers_yield_exactly_the_parameters_they_own() {
    for use_bias in [true, false] {
        let owned: Vec<&'static str> = if use_bias {
            vec!["kernel", "bias"]
        } else {
            vec!["kernel"]
        };

        let mut dense = Dense::new(3, 2, Activation::Linear)
            .unwrap()
            .with_use_bias(use_bias);
        assert_eq!(
            weight_names(&dense),
            owned,
            "Dense with use_bias {use_bias} exposes the wrong arrays"
        );
        assert_eq!(
            train_once(&mut dense, &tensor(&[4, 3], 0), &tensor(&[4, 2], 47)),
            owned,
            "Dense with use_bias {use_bias} yields the wrong parameters"
        );

        let mut conv = Conv2D::new(2, (2, 2), vec![1, 4, 4, 2], (1, 1), Activation::Linear)
            .unwrap()
            .with_use_bias(use_bias);
        assert_eq!(
            weight_names(&conv),
            owned,
            "Conv2D with use_bias {use_bias} exposes the wrong arrays"
        );
        assert_eq!(
            train_once(
                &mut conv,
                &tensor(&[1, 4, 4, 2], 0),
                &tensor(&[1, 3, 3, 2], 47)
            ),
            owned,
            "Conv2D with use_bias {use_bias} yields the wrong parameters"
        );
    }
}

/// Builds 1 model of exactly 1 layer that drops 1 of its 2 normalization arrays
type MakeModel = fn() -> Sequential;

/// A normalization layer that drops 1 array must still train the array it keeps
///
/// The mirror of the bias-free Dense above, on the side where the optional array comes first.
/// A silent stop is the worst outcome of a lost gate: the layer yields nothing, the optimizer
/// writes nothing, the loss stands still, and no error says so. This trains each layer for
/// several steps and reads the array back
#[test]
fn a_normalization_layer_that_drops_1_array_trains_the_array_it_keeps() {
    let cases: [(&str, MakeModel, &str); 2] = [
        (
            "LayerNormalization without a scale",
            || {
                let mut model = Sequential::new();
                model.add(
                    LayerNormalization::new(vec![4, 4], 1e-5)
                        .unwrap()
                        .with_scale(false),
                );
                model
            },
            "0.beta",
        ),
        (
            "BatchNormalization without a center",
            || {
                let mut model = Sequential::new();
                model.add(
                    BatchNormalization::new(vec![4, 4], 0.9, 1e-5)
                        .unwrap()
                        .with_center(false),
                );
                model
            },
            "0.gamma",
        ),
    ];

    let x = tensor(&[4, 4], 0);
    let y = tensor(&[4, 4], 61);

    for (name, build, path) in cases {
        let mut model = build();
        model.compile(
            SGD::new(0.1, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

        let before: Vec<f32> = model.weight(path).unwrap().iter().copied().collect();
        model.fit(&x, &y, 4).unwrap();
        let after: Vec<f32> = model.weight(path).unwrap().iter().copied().collect();

        assert_eq!(before.len(), 4, "{name} holds the wrong extent at `{path}`");
        let moved = before
            .iter()
            .zip(after.iter())
            .filter(|(b, a)| (*b - *a).abs() > 1e-6)
            .count();
        assert_eq!(
            moved,
            before.len(),
            "{name} left `{path}` where it was: before {before:?}, after {after:?}"
        );
    }
}
