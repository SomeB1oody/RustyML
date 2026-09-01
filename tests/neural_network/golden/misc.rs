//! Golden fixtures for the misc family, the reference family of the net.
//!
//! Covers Dense, the shape layers (Flatten, Identity, Reshape, Permute, RepeatVector),
//! Rescaling, and every standalone activation layer. The data file is
//! `golden/data/misc.golden`.
//!
//! This file is the worked example that every other family follows. The pattern is:
//!
//! 1. Write 1 function per layer type. It returns the cases for that layer type
//! 2. Register those functions in [`fixtures`], which lives in this file alone
//! 3. Add 1 test that hands the family name and the list to
//!    [`run_family`](super::run_family)
//!
//! A family never edits another family file, and never edits the harness.
//!
//! Every layer here builds its weights from [`golden_weights`](super::golden_weights), so no
//! initializer and no random number generator can move a recorded value. See the module doc
//! comment of the parent module for the whole contract.

use super::{GoldenCase, LayerFixture, golden_weights, golden_weights_from};
use ndarray::Ix2;
use rustyml::neural_network::layers::activation::{
    Activation, ELU, Exponential, HardSigmoid, LeakyReLU, Linear, PReLU, ReLU, SELU, Sigmoid,
    Softmax, Softplus, Softsign, Tanh,
};
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::identity::Identity;
use rustyml::neural_network::layers::permute::Permute;
use rustyml::neural_network::layers::repeat_vector::RepeatVector;
use rustyml::neural_network::layers::rescaling::Rescaling;
use rustyml::neural_network::layers::reshape::Reshape;
use rustyml::neural_network::traits::Layer;

/// Every layer type of the misc family, in the order the data file records them.
fn fixtures() -> Vec<LayerFixture> {
    vec![
        LayerFixture::new("Dense", dense_cases),
        LayerFixture::new("Flatten", flatten_cases),
        LayerFixture::new("Identity", identity_cases),
        LayerFixture::new("Reshape", reshape_cases),
        LayerFixture::new("Permute", permute_cases),
        LayerFixture::new("RepeatVector", repeat_vector_cases),
        LayerFixture::new("ReLU", relu_cases),
        LayerFixture::new("LeakyReLU", leaky_relu_cases),
        LayerFixture::new("ELU", elu_cases),
        LayerFixture::new("SELU", selu_cases),
        LayerFixture::new("Softplus", softplus_cases),
        LayerFixture::new("Softsign", softsign_cases),
        LayerFixture::new("HardSigmoid", hard_sigmoid_cases),
        LayerFixture::new("Exponential", exponential_cases),
        LayerFixture::new("Linear", linear_cases),
        LayerFixture::new("Sigmoid", sigmoid_cases),
        LayerFixture::new("Tanh", tanh_cases),
        LayerFixture::new("Softmax", softmax_cases),
        LayerFixture::new("PReLU", p_relu_cases),
        LayerFixture::new("Rescaling", rescaling_cases),
    ]
}

/// Replays the misc family against `golden/data/misc.golden`.
#[test]
fn golden_misc_family() {
    super::run_family("misc", &fixtures());
}

/// The input shape that every parameter-free activation case uses.
const ACTIVATION_SHAPE: [usize; 2] = [2, 5];

/// 1 case for a parameter-free activation layer over a rank-2 input.
fn activation_case(label: &'static str, build: fn() -> Box<dyn Layer>) -> GoldenCase {
    GoldenCase::new(label, &ACTIVATION_SHAPE, build)
}

// ---------------------------------------------------------------------------------------
// Dense and the shape layers
// ---------------------------------------------------------------------------------------

/// Dense cases: 3 input features into 4 units, over 3 activation paths and 3 input ranks.
///
/// The 3 activations cover the 3 forward paths of the layer. `Linear` runs the fused product
/// with no epilogue activation, `ReLU` runs the fused epilogue activation, and `Tanh` runs a
/// separate activation pass over the fused product.
///
/// The 2 rank cases cover the fold that a rank above 2 needs. The layer folds every leading
/// axis into 1 row axis, runs the same matrix product, and then restores the input rank. Each
/// leading position therefore shares 1 kernel.
///
/// The 2 rank cases hold 24 input values each, and the input formula reads the flat index
/// alone. The 2 inputs therefore hold the same 8 rows of 3 features in 2 different shapes, and
/// the 2 recorded forward tensors hold the same 32 numbers in 2 different shapes. A fold that
/// reads the wrong axes breaks that agreement. The backward pass restores the rank of the input
/// gradient from the cached input shape, so a lost rank moves a recorded `grad_input` shape.
fn dense_cases() -> Vec<GoldenCase> {
    /// Builds a 3 by 4 Dense layer with fixed weights and the given activation.
    fn build(activation: Activation) -> Box<dyn Layer> {
        let mut layer = Dense::new(3, 4, activation).expect("3 features into 4 units");
        let weight = golden_weights(&[3, 4])
            .into_dimensionality::<Ix2>()
            .expect("the weight is rank 2");
        // The bias continues the weight formula, so it never repeats the weight values
        let bias = golden_weights_from(&[1, 4], 12)
            .into_dimensionality::<Ix2>()
            .expect("the bias is rank 2");
        layer
            .set_weights(weight, bias)
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    let names = ["kernel", "bias"];
    vec![
        GoldenCase::new("units_4_linear", &[2, 3], || build(Activation::Linear))
            .with_parameter_grads(&names),
        GoldenCase::new("units_4_relu", &[2, 3], || build(Activation::ReLU))
            .with_parameter_grads(&names),
        GoldenCase::new("units_4_tanh", &[2, 3], || build(Activation::Tanh))
            .with_parameter_grads(&names),
        // Rank 3: 2 batch items of 4 positions, and every position shares the 1 kernel
        GoldenCase::new("rank_3_linear", &[2, 4, 3], || build(Activation::Linear))
            .with_parameter_grads(&names),
        // Rank 4: the same 8 rows again, in 2 leading axes of 2 instead of 1 axis of 4
        GoldenCase::new("rank_4_linear", &[2, 2, 2, 3], || build(Activation::Linear))
            .with_parameter_grads(&names),
    ]
}

/// Flatten cases: 1 rank-3 input and 1 rank-4 input.
fn flatten_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("rank_3", &[2, 3, 4], || {
            Box::new(Flatten::new(vec![2, 3, 4]).expect("a rank 3 shape"))
        }),
        GoldenCase::new("rank_4", &[2, 2, 3, 2], || {
            Box::new(Flatten::new(vec![2, 2, 3, 2]).expect("a rank 4 shape"))
        }),
    ]
}

/// Identity cases: 1 rank-2 input and 1 rank-4 input.
fn identity_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("rank_2", &[2, 5], || Box::new(Identity::new())),
        GoldenCase::new("rank_4", &[2, 2, 3, 2], || Box::new(Identity::new())),
    ]
}

/// Reshape cases: 1 explicit target shape and 1 target shape with an inferred axis.
fn reshape_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("to_6", &[2, 2, 3], || {
            Box::new(Reshape::new(vec![6]).expect("a positive target shape"))
        }),
        GoldenCase::new("inferred_axis", &[2, 3, 4], || {
            Box::new(Reshape::new(vec![-1, 2]).expect("1 inferred axis"))
        }),
    ]
}

/// Permute cases: 1 swap of 2 axes and 1 rotation of 3 axes.
fn permute_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("swap_2_axes", &[2, 3, 4], || {
            Box::new(Permute::new(vec![2, 1]).expect("a permutation of 2 axes"))
        }),
        GoldenCase::new("rotate_3_axes", &[2, 2, 3, 4], || {
            Box::new(Permute::new(vec![3, 1, 2]).expect("a permutation of 3 axes"))
        }),
    ]
}

/// RepeatVector cases: a rank-2 input repeated into 3 steps.
fn repeat_vector_cases() -> Vec<GoldenCase> {
    vec![GoldenCase::new("steps_3", &[2, 4], || {
        Box::new(RepeatVector::new(3).expect("3 steps"))
    })]
}

// ---------------------------------------------------------------------------------------
// The activation layers
// ---------------------------------------------------------------------------------------

/// ReLU cases.
fn relu_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(ReLU::new()))]
}

/// LeakyReLU cases: the default slope, and a small slope.
fn leaky_relu_cases() -> Vec<GoldenCase> {
    vec![
        activation_case("slope_0p3", || {
            Box::new(LeakyReLU::new(0.3).expect("a positive slope"))
        }),
        activation_case("slope_0p01", || {
            Box::new(LeakyReLU::new(0.01).expect("a positive slope"))
        }),
    ]
}

/// ELU cases: the default scale, and a half scale.
fn elu_cases() -> Vec<GoldenCase> {
    vec![
        activation_case("alpha_1p0", || {
            Box::new(ELU::new(1.0).expect("a positive scale"))
        }),
        activation_case("alpha_0p5", || {
            Box::new(ELU::new(0.5).expect("a positive scale"))
        }),
    ]
}

/// SELU cases.
fn selu_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(SELU::new()))]
}

/// Softplus cases.
fn softplus_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(Softplus::new()))]
}

/// Softsign cases.
fn softsign_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(Softsign::new()))]
}

/// HardSigmoid cases.
fn hard_sigmoid_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(HardSigmoid::new()))]
}

/// Exponential cases.
fn exponential_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(Exponential::new()))]
}

/// Linear cases.
fn linear_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(Linear::new()))]
}

/// Sigmoid cases.
fn sigmoid_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(Sigmoid::new()))]
}

/// Tanh cases.
fn tanh_cases() -> Vec<GoldenCase> {
    vec![activation_case("default", || Box::new(Tanh::new()))]
}

/// Softmax cases: 1 rank-2 input and 1 rank-3 input, which normalizes over the last axis.
fn softmax_cases() -> Vec<GoldenCase> {
    vec![
        activation_case("rank_2", || Box::new(Softmax::new())),
        GoldenCase::new("rank_3", &[2, 2, 5], || Box::new(Softmax::new())),
    ]
}

/// PReLU cases: 1 slope per feature, and 1 slope per channel through a shared axis.
fn p_relu_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("per_feature", &ACTIVATION_SHAPE, || {
            let mut layer = PReLU::new(ACTIVATION_SHAPE.to_vec(), 0.25).expect("a finite slope");
            layer
                .set_weights(golden_weights(&[5]))
                .expect("1 slope per feature");
            Box::new(layer)
        })
        .with_parameter_grads(&["alpha"]),
        GoldenCase::new("shared_axis_1", &[2, 3, 4], || {
            let mut layer = PReLU::new(vec![2, 3, 4], 0.25)
                .expect("a finite slope")
                .with_shared_axes(vec![1])
                .expect("axis 1 is after the batch axis");
            layer
                .set_weights(golden_weights(&[1, 4]))
                .expect("1 slope per channel");
            Box::new(layer)
        })
        .with_parameter_grads(&["alpha"]),
    ]
}

// ---------------------------------------------------------------------------------------
// The rescaling layer
// ---------------------------------------------------------------------------------------

/// Rescaling cases: a positive scale, a negative scale, a non-zero offset, and a rank-4 input.
///
/// The layer applies `y = x * scale + offset` to every element, and it holds no parameter. The
/// 4 cases divide the map into the parts that a later stage can break separately.
///
/// `scale_0p5` pins the plain multiplication with the default offset of 0. `scale_neg_1p5`
/// pins a negative scale, which the layer must apply like any other value. A negative scale
/// also flips the sign of every recorded input gradient, so a rule that took the magnitude of
/// the scale would show up there and nowhere else.
///
/// `scale_2p0_offset_neg_1p0` pins the offset. The offset is a constant, so it reaches the
/// forward output and it must reach no input gradient. The recorded `grad_input` of this case
/// therefore holds the upstream gradient times 2, with no trace of the offset.
///
/// `rank_4_scale_0p25` pins that the map stays elementwise at a higher rank, and that the
/// output keeps the input shape.
///
/// The layer reads no training mode and keeps no cache, so `predict` returns exactly the
/// `forward` output. Every case keeps the assertion of the harness that pins this.
fn rescaling_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("scale_0p5", &ACTIVATION_SHAPE, || {
            Box::new(Rescaling::new(0.5))
        }),
        GoldenCase::new("scale_neg_1p5", &ACTIVATION_SHAPE, || {
            Box::new(Rescaling::new(-1.5))
        }),
        GoldenCase::new("scale_2p0_offset_neg_1p0", &ACTIVATION_SHAPE, || {
            Box::new(Rescaling::new(2.0).with_offset(-1.0))
        }),
        GoldenCase::new("rank_4_scale_0p25", &[2, 2, 3, 2], || {
            Box::new(Rescaling::new(0.25))
        }),
    ]
}
