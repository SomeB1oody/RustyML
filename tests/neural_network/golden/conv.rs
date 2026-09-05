//! Golden fixtures for the convolution family.
//!
//! Covers Conv1D, Conv2D, Conv3D, the transposed convolutions, and the depthwise and separable
//! convolutions. The data file is `golden/data/conv.golden`.
//!
//! See the module doc comment of the parent module for the contract, and see `misc.rs` for a
//! worked family.
//!
//! # How the configurations divide the branches
//!
//! Each layer type gets at least 2 cases, and 2 of them take opposite branches:
//!
//! - A `valid_` case uses [`PaddingType::Valid`], stride 1 where the layer allows it, and the
//!   `Linear` activation. The record then holds the plain convolution numbers, with no
//!   activation between the convolution and the recorded value
//! - A `same_` case uses [`PaddingType::Same`], a stride above 1, and the `ReLU` activation.
//!   The record then holds the padded geometry, the strided window positions, and the fused
//!   activation together with the activation backward pass
//!
//! A non-square kernel appears in the Conv2D, DepthwiseConv2D, and SeparableConv2D cases. A
//! `depth_multiplier` above 1 appears in 1 case of each depthwise layer and 1 case of each
//! separable layer. Each transposed layer gets 1 case where the stride divides the kernel size
//! and 1 case where it does not, because an uneven division leaves window overlap that the even
//! division does not.
//!
//! # The dilated cases and the tap map
//!
//! Each forward convolution, each depthwise convolution, and each separable convolution also
//! gets at least 1 case with a `dilation_rate` above 1. A dilation of `d` spaces the taps of
//! the kernel `d` cells apart, so `k` taps span `(k - 1) * d + 1` input cells. The window still
//! advances by the stride.
//!
//! The map from an output position and a tap number to an input position is therefore
//! `position * stride + tap * dilation`. A defect that reads the stride for the tap spacing, or
//! the dilation for the window advance, gives the same map when the stride and the dilation
//! agree. Every dilated case here therefore keeps the stride away from the dilation.
//!
//! Conv1D, Conv2D, and Conv3D reject a stride above 1 together with a dilation above 1, so
//! their dilated cases hold the stride at 1. The depthwise and separable layers take the
//! combination, and `DepthwiseConv1D/same_dilation_3_stride_2` uses it: the stride is 2, the
//! dilation is 3, and no defect that exchanges the 2 gives the recorded numbers.
//!
//! A dilation that differs between 2 axes appears in `Conv2D/valid_dilation_2x1`,
//! `Conv3D/valid_dilation_2x1x1`, `DepthwiseConv2D/valid_dilation_1x2`, and
//! `SeparableConv2D/valid_dilation_2x1`. A pass that reads the dilation of the wrong axis then
//! moves a recorded output shape.
//!
//! # The causal cases
//!
//! Conv1D is the 1 layer that takes `ConvPadding::Causal`. Causal padding puts every 1 of the
//! `(k - 1) * dilation` pad cells on the leading edge, so an output position never reads a
//! later input position. The output length is the length that `Same` gives.
//!
//! 2 cases record it. `causal_dilation_1` holds the pad count at 2, and `causal_dilation_3`
//! raises it to 6, which is the whole input length of 6. Output position 0 of the second case
//! therefore reads 1 real input cell and 2 pad cells. A rule that split the pad cells over the
//! 2 edges, as `Same` does, moves every recorded value of both cases.
//!
//! # Weight layout
//!
//! The forward convolutions carry the kernel as `[spatial..., channels, filters]`. The
//! transposed convolutions carry the filter axis before the input-channel axis, as
//! `[spatial..., filters, channels]`. A depthwise kernel is `[spatial..., channels,
//! depth_multiplier]`, and its bias holds `channels * depth_multiplier` entries. A separable
//! layer carries a depthwise kernel, a pointwise kernel, and 1 bias over the filters.
//!
//! Every weight tensor comes from [`golden_weights_from`](super::golden_weights_from), and each
//! tensor of a layer starts the formula where the previous tensor ended. No 2 parameter tensors
//! of 1 layer therefore hold the same values, so a swap of 2 parameters cannot pass unseen.

use super::{GoldenCase, LayerFixture, golden_weights, golden_weights_from};
use ndarray::{Ix1, Ix3, Ix4, Ix5};
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::{Conv1D, ConvPadding};
use rustyml::neural_network::layers::convolution::conv_1d_transpose::Conv1DTranspose;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_2d_transpose::Conv2DTranspose;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::conv_3d_transpose::Conv3DTranspose;
use rustyml::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::traits::{Layer, UnaryLayer};

/// Every layer type of the conv family, in the order the data file records them.
fn fixtures() -> Vec<LayerFixture> {
    vec![
        LayerFixture::new("Conv1D", conv_1d_cases),
        LayerFixture::new("Conv2D", conv_2d_cases),
        LayerFixture::new("Conv3D", conv_3d_cases),
        LayerFixture::new("Conv1DTranspose", conv_1d_transpose_cases),
        LayerFixture::new("Conv2DTranspose", conv_2d_transpose_cases),
        LayerFixture::new("Conv3DTranspose", conv_3d_transpose_cases),
        LayerFixture::new("DepthwiseConv1D", depthwise_conv_1d_cases),
        LayerFixture::new("DepthwiseConv2D", depthwise_conv_2d_cases),
        LayerFixture::new("SeparableConv1D", separable_conv_1d_cases),
        LayerFixture::new("SeparableConv2D", separable_conv_2d_cases),
    ]
}

/// Replays the conv family against `golden/data/conv.golden`.
#[test]
fn golden_conv_family() {
    super::run_family("conv", &fixtures());
}

/// The parameter names of a layer that carries 1 kernel and 1 bias.
const KERNEL_AND_BIAS: [&str; 2] = ["kernel", "bias"];

/// The parameter names of a separable layer, in `LayerBase::parameters_mut` order.
const SEPARABLE_PARAMETERS: [&str; 3] = ["depthwise_kernel", "pointwise_kernel", "bias"];

/// The rank-3 input shape that every 1D case uses, as \[batch, length, channels\].
const SHAPE_1D: [usize; 3] = [2, 6, 2];

/// The rank-4 input shape that every 2D forward case uses, as \[batch, height, width, channels\].
const SHAPE_2D: [usize; 4] = [2, 5, 4, 2];

/// The rank-5 input shape that every 3D forward case uses.
const SHAPE_3D: [usize; 5] = [2, 3, 3, 3, 2];

/// The rank-5 input shape of the dilated Conv3D case.
///
/// The depth axis holds 4 cells instead of 3. A kernel of 2 taps 2 cells apart spans 3 cells,
/// so the depth axis keeps 2 windows. A depth of 3 would keep 1 window, and the record would
/// then say nothing about how the pass advances the window on that axis.
const SHAPE_3D_DILATED: [usize; 5] = [2, 4, 3, 3, 2];

/// The rank-3 input shape that every 1D transposed case uses.
const SHAPE_1D_TRANSPOSE: [usize; 3] = [2, 3, 2];

/// The rank-4 input shape that every 2D transposed case uses.
const SHAPE_2D_TRANSPOSE: [usize; 4] = [2, 3, 3, 2];

/// The rank-5 input shape that every 3D transposed case uses.
const SHAPE_3D_TRANSPOSE: [usize; 5] = [2, 2, 2, 2, 2];

/// Builds a rank-1 bias tensor from the weight formula, starting at a chosen flat index.
fn bias(length: usize, first_index: usize) -> ndarray::Array1<f32> {
    golden_weights_from(&[length], first_index)
        .into_dimensionality::<Ix1>()
        .expect("the bias is rank 1")
}

// ---------------------------------------------------------------------------------------
// The forward convolutions
// ---------------------------------------------------------------------------------------

/// Conv1D cases: a Valid unit-stride pass, a Same pass at stride 2, 1 dilated pass, and 2
/// causal passes.
///
/// `valid_stride_1` reads 4 windows over a length of 6 and applies no activation.
/// `same_stride_2` pads the length, keeps 3 strided windows, and runs the ReLU epilogue.
///
/// `valid_dilation_2` spaces the 3 taps 2 cells apart, so they span 5 of the 6 input cells and
/// 2 windows are left. `causal_dilation_1` and `causal_dilation_3` each keep the output length
/// at 6 and put every pad cell on the leading edge. The 3 new cases apply no activation, so the
/// record holds the convolution numbers alone.
fn conv_1d_cases() -> Vec<GoldenCase> {
    /// Builds a 3-filter Conv1D over 2 channels with a kernel of 3.
    fn build(
        stride: usize,
        padding: ConvPadding,
        dilation_rate: usize,
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer = Conv1D::new(3, 3, stride, activation)
            .expect("3 filters and a kernel of 3 fit the input")
            .with_padding(padding)
            .with_dilation_rate(dilation_rate)
            .expect("a positive dilation that no stride above 1 accompanies");
        layer
            .build(&Shape::known(&SHAPE_1D))
            .expect("the layer accepts the shape of the case");
        let weight = golden_weights(&[3, 2, 3])
            .into_dimensionality::<Ix3>()
            .expect("the kernel is rank 3");
        layer
            .set_weights(weight, bias(3, 18))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_stride_1", &SHAPE_1D, || {
            build(1, ConvPadding::Valid, 1, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_2", &SHAPE_1D, || {
            build(2, ConvPadding::Same, 1, Activation::ReLU)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // 3 taps 2 cells apart span 5 cells, so a length of 6 keeps 2 windows
        GoldenCase::new("valid_dilation_2", &SHAPE_1D, || {
            build(1, ConvPadding::Valid, 2, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // 2 pad cells, all on the leading edge, and an output length of 6
        GoldenCase::new("causal_dilation_1", &SHAPE_1D, || {
            build(1, ConvPadding::Causal, 1, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // 6 pad cells, which is the whole input length, and an output length of 6
        GoldenCase::new("causal_dilation_3", &SHAPE_1D, || {
            build(1, ConvPadding::Causal, 3, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

/// Conv2D cases: a Valid pass with a non-square kernel, a Same pass at stride 2, and 2 dilated
/// passes.
///
/// `valid_kernel_3x2` uses a kernel of 3 rows by 2 columns, so a swap of the 2 kernel axes
/// changes the output shape. `same_stride_2` pads both spatial axes and strides both by 2.
///
/// `valid_dilation_2x1` spaces the 2 kernel rows 2 cells apart and leaves the 2 kernel columns
/// solid. The taps then span 3 rows and 2 columns of the input, which gives 3 windows on each
/// axis. `same_dilation_2` dilates both axes of a 3 by 3 kernel, so the taps span 5 rows and 5
/// columns, and the Same rule pads the input up to that span while it holds the output at the
/// input size.
fn conv_2d_cases() -> Vec<GoldenCase> {
    /// Builds a 3-filter Conv2D over 2 channels.
    fn build(
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: PaddingType,
        dilation_rate: (usize, usize),
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer = Conv2D::new(3, kernel_size, strides, activation)
            .expect("the kernel fits the input")
            .with_padding(padding)
            .with_dilation_rate(dilation_rate)
            .expect("a positive dilation that no stride above 1 accompanies");
        layer
            .build(&Shape::known(&SHAPE_2D))
            .expect("the layer accepts the shape of the case");
        let kernel_shape = [kernel_size.0, kernel_size.1, 2, 3];
        let count: usize = kernel_shape.iter().product();
        let weight = golden_weights(&kernel_shape)
            .into_dimensionality::<Ix4>()
            .expect("the kernel is rank 4");
        layer
            .set_weights(weight, bias(3, count))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_kernel_3x2", &SHAPE_2D, || {
            build(
                (3, 2),
                (1, 1),
                PaddingType::Valid,
                (1, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_2", &SHAPE_2D, || {
            build((3, 3), (2, 2), PaddingType::Same, (1, 1), Activation::ReLU)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // The 2 axes take a different dilation, so a pass that reads the wrong 1 of the 2
        // moves the output shape
        GoldenCase::new("valid_dilation_2x1", &SHAPE_2D, || {
            build(
                (2, 2),
                (1, 1),
                PaddingType::Valid,
                (2, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // The Same rule pads for the dilated span of 5 on both axes, and keeps the output at
        // the input size of 5 rows by 4 columns
        GoldenCase::new("same_dilation_2", &SHAPE_2D, || {
            build(
                (3, 3),
                (1, 1),
                PaddingType::Same,
                (2, 2),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

/// Conv3D cases: a Valid unit-stride pass, a Same pass at stride 2 on all 3 axes, and 2 dilated
/// passes.
///
/// The first 2 cases use a kernel of 2 on every spatial axis over a cube of 3, so the Same case
/// pads an odd extent and the Valid case does not pad at all.
///
/// `valid_dilation_2x1x1` dilates the depth axis alone, over [`SHAPE_3D_DILATED`]. The 3 axes
/// therefore hold a different tap spacing from each other, and a pass that reads the dilation
/// of the wrong axis moves the output shape. `same_dilation_2` dilates all 3 axes over the cube
/// of 3, where the taps span 3 cells and the Same rule holds the output at the input size.
fn conv_3d_cases() -> Vec<GoldenCase> {
    /// Builds a 2-filter Conv3D over 2 channels with a kernel of 2 on every spatial axis.
    fn build(
        input_shape: &[usize],
        strides: (usize, usize, usize),
        padding: PaddingType,
        dilation_rate: (usize, usize, usize),
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer = Conv3D::new(2, (2, 2, 2), strides, activation)
            .expect("a kernel of 2 fits every spatial extent of the input")
            .with_padding(padding)
            .with_dilation_rate(dilation_rate)
            .expect("a positive dilation that no stride above 1 accompanies");
        layer
            .build(&Shape::known(input_shape))
            .expect("the layer accepts the shape of the case");
        let weight = golden_weights(&[2, 2, 2, 2, 2])
            .into_dimensionality::<Ix5>()
            .expect("the kernel is rank 5");
        layer
            .set_weights(weight, bias(2, 32))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_stride_1", &SHAPE_3D, || {
            build(
                &SHAPE_3D,
                (1, 1, 1),
                PaddingType::Valid,
                (1, 1, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_2", &SHAPE_3D, || {
            build(
                &SHAPE_3D,
                (2, 2, 2),
                PaddingType::Same,
                (1, 1, 1),
                Activation::ReLU,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // The depth axis takes a dilation of 2, and the other 2 axes keep a solid kernel
        GoldenCase::new("valid_dilation_2x1x1", &SHAPE_3D_DILATED, || {
            build(
                &SHAPE_3D_DILATED,
                (1, 1, 1),
                PaddingType::Valid,
                (2, 1, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // All 3 axes take a dilation of 2, and the Same rule keeps the output at the cube of 3
        GoldenCase::new("same_dilation_2", &SHAPE_3D, || {
            build(
                &SHAPE_3D,
                (1, 1, 1),
                PaddingType::Same,
                (2, 2, 2),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

// ---------------------------------------------------------------------------------------
// The transposed convolutions
// ---------------------------------------------------------------------------------------

/// Conv1DTranspose cases: 1 stride that divides the kernel, and 1 stride that does not.
///
/// `valid_stride_2_kernel_4` divides 4 by 2, so every output position takes the same number of
/// contributions. `same_stride_3_kernel_4` leaves a remainder of 1, so the scatter windows
/// overlap unevenly and the output picks up an uneven contribution count.
fn conv_1d_transpose_cases() -> Vec<GoldenCase> {
    /// Builds a 2-filter Conv1DTranspose over 2 channels with a kernel of 4.
    ///
    /// The kernel is `[kernel_size, filters, channels]`, so the filter axis comes before the
    /// input-channel axis.
    fn build(stride: usize, padding: PaddingType, activation: Activation) -> Box<dyn Layer> {
        let mut layer = Conv1DTranspose::new(2, 4, stride, activation)
            .expect("2 filters and a kernel of 4")
            .with_padding(padding);
        layer
            .build(&Shape::known(&SHAPE_1D_TRANSPOSE))
            .expect("the layer accepts the shape of the case");
        let weight = golden_weights(&[4, 2, 2])
            .into_dimensionality::<Ix3>()
            .expect("the kernel is rank 3");
        layer
            .set_weights(weight, bias(2, 16))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_stride_2_kernel_4", &SHAPE_1D_TRANSPOSE, || {
            build(2, PaddingType::Valid, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_3_kernel_4", &SHAPE_1D_TRANSPOSE, || {
            build(3, PaddingType::Same, Activation::ReLU)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

/// Conv2DTranspose cases: 1 stride that divides the kernel, and 1 stride that does not.
///
/// `valid_stride_2_kernel_2` divides 2 by 2, so the scatter windows tile the output and never
/// overlap. `same_stride_2_kernel_3` leaves a remainder of 1, so neighboring windows overlap by
/// 1 position on both spatial axes.
fn conv_2d_transpose_cases() -> Vec<GoldenCase> {
    /// Builds a 2-filter Conv2DTranspose over 2 channels at stride 2 on both axes.
    fn build(
        kernel_size: (usize, usize),
        padding: PaddingType,
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer = Conv2DTranspose::new(2, kernel_size, (2, 2), activation)
            .expect("2 filters over 2 channels")
            .with_padding(padding);
        layer
            .build(&Shape::known(&SHAPE_2D_TRANSPOSE))
            .expect("the layer accepts the shape of the case");
        // The kernel is [kernel_height, kernel_width, filters, channels]
        let kernel_shape = [kernel_size.0, kernel_size.1, 2, 2];
        let count: usize = kernel_shape.iter().product();
        let weight = golden_weights(&kernel_shape)
            .into_dimensionality::<Ix4>()
            .expect("the kernel is rank 4");
        layer
            .set_weights(weight, bias(2, count))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_stride_2_kernel_2", &SHAPE_2D_TRANSPOSE, || {
            build((2, 2), PaddingType::Valid, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_2_kernel_3", &SHAPE_2D_TRANSPOSE, || {
            build((3, 3), PaddingType::Same, Activation::ReLU)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

/// Conv3DTranspose cases: 1 stride that divides the kernel, and 1 stride that does not.
///
/// Both cases use 1 filter, which keeps the output at 128 elements. `valid_stride_2_kernel_2`
/// tiles the output, and `same_stride_2_kernel_3` overlaps the windows on all 3 spatial axes.
fn conv_3d_transpose_cases() -> Vec<GoldenCase> {
    /// Builds a 1-filter Conv3DTranspose over 2 channels at stride 2 on every spatial axis.
    fn build(
        kernel_size: (usize, usize, usize),
        padding: PaddingType,
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer = Conv3DTranspose::new(1, kernel_size, (2, 2, 2), activation)
            .expect("1 filter over 2 channels")
            .with_padding(padding);
        layer
            .build(&Shape::known(&SHAPE_3D_TRANSPOSE))
            .expect("the layer accepts the shape of the case");
        // The kernel is [kernel_depth, kernel_height, kernel_width, filters, channels]
        let kernel_shape = [kernel_size.0, kernel_size.1, kernel_size.2, 1, 2];
        let count: usize = kernel_shape.iter().product();
        let weight = golden_weights(&kernel_shape)
            .into_dimensionality::<Ix5>()
            .expect("the kernel is rank 5");
        layer
            .set_weights(weight, bias(1, count))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_stride_2_kernel_2", &SHAPE_3D_TRANSPOSE, || {
            build((2, 2, 2), PaddingType::Valid, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_2_kernel_3", &SHAPE_3D_TRANSPOSE, || {
            build((3, 3, 3), PaddingType::Same, Activation::ReLU)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

// ---------------------------------------------------------------------------------------
// The depthwise convolutions
// ---------------------------------------------------------------------------------------

/// DepthwiseConv1D cases: 1 kernel per channel, 2 kernels per channel at stride 2, and 2
/// dilated passes.
///
/// `same_multiplier_2_stride_2` doubles the output channel count. Input channel `c` and
/// multiplier `m` land at output channel `c * 2 + m`, so a wrong interleave shows up at once.
///
/// `valid_dilation_2` holds the stride at 1 and spaces the 3 taps 2 cells apart, so they span 5
/// of the 6 input cells. `same_dilation_3_stride_2` is the 1 case of the family that raises the
/// stride and the dilation together. The stride is 2 and the dilation is 3, so the tap map
/// `position * 2 + tap * 3` gives a different input cell for every exchange of the 2 numbers.
/// The taps span 7 cells over an input of 6, which the Same rule pads for.
fn depthwise_conv_1d_cases() -> Vec<GoldenCase> {
    /// Builds a DepthwiseConv1D over 2 channels with a kernel of 3.
    fn build(
        stride: usize,
        padding: PaddingType,
        depth_multiplier: usize,
        dilation_rate: usize,
        activation: Activation,
    ) -> Box<dyn Layer> {
        // `with_depth_multiplier` re-initializes the weights, so it runs before `set_weights`
        let mut layer = DepthwiseConv1D::new(3, stride, activation)
            .expect("a kernel of 3 fits a length of 6")
            .with_padding(padding)
            .with_dilation_rate(dilation_rate)
            .expect("a positive dilation")
            .with_depth_multiplier(depth_multiplier)
            .expect("a positive depth multiplier");
        layer
            .build(&Shape::known(&SHAPE_1D))
            .expect("the layer accepts the shape of the case");
        let count = 3 * 2 * depth_multiplier;
        let weight = golden_weights(&[3, 2, depth_multiplier])
            .into_dimensionality::<Ix3>()
            .expect("the kernel is rank 3");
        layer
            .set_weights(weight, bias(2 * depth_multiplier, count))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_multiplier_1", &SHAPE_1D, || {
            build(1, PaddingType::Valid, 1, 1, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_multiplier_2_stride_2", &SHAPE_1D, || {
            build(2, PaddingType::Same, 2, 1, Activation::ReLU)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // 3 taps 2 cells apart span 5 cells, so a length of 6 keeps 2 windows
        GoldenCase::new("valid_dilation_2", &SHAPE_1D, || {
            build(1, PaddingType::Valid, 1, 2, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // The stride is 2 and the dilation is 3, so neither number can stand for the other
        GoldenCase::new("same_dilation_3_stride_2", &SHAPE_1D, || {
            build(2, PaddingType::Same, 1, 3, Activation::Linear)
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

/// DepthwiseConv2D cases: 2 kernels per channel with a non-square kernel, and 1 kernel per
/// channel at stride 2.
///
/// `valid_multiplier_2_kernel_2x3` combines a depth multiplier of 2 with a kernel of 2 rows by
/// 3 columns, so it exercises both the channel expansion and the unequal spatial extents.
///
/// `valid_dilation_1x2` dilates the column axis alone. The taps then span 2 rows and 3 columns,
/// which gives 4 windows down the rows and 2 across the columns. The 2 axes therefore hold a
/// different tap spacing, and a pass that reads the dilation of the wrong axis moves the output
/// shape.
fn depthwise_conv_2d_cases() -> Vec<GoldenCase> {
    /// Builds a DepthwiseConv2D over 2 channels.
    fn build(
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: PaddingType,
        depth_multiplier: usize,
        dilation_rate: (usize, usize),
        activation: Activation,
    ) -> Box<dyn Layer> {
        // `with_depth_multiplier` re-initializes the weights, so it runs before `set_weights`
        let mut layer = DepthwiseConv2D::new(kernel_size, strides, activation)
            .expect("the kernel fits the input")
            .with_padding(padding)
            .with_dilation_rate(dilation_rate)
            .expect("a positive dilation")
            .with_depth_multiplier(depth_multiplier)
            .expect("a positive depth multiplier");
        layer
            .build(&Shape::known(&SHAPE_2D))
            .expect("the layer accepts the shape of the case");
        let kernel_shape = [kernel_size.0, kernel_size.1, 2, depth_multiplier];
        let count: usize = kernel_shape.iter().product();
        let weight = golden_weights(&kernel_shape)
            .into_dimensionality::<Ix4>()
            .expect("the kernel is rank 4");
        layer
            .set_weights(weight, bias(2 * depth_multiplier, count))
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_multiplier_2_kernel_2x3", &SHAPE_2D, || {
            build(
                (2, 3),
                (1, 1),
                PaddingType::Valid,
                2,
                (1, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        GoldenCase::new("same_stride_2", &SHAPE_2D, || {
            build(
                (3, 3),
                (2, 2),
                PaddingType::Same,
                1,
                (1, 1),
                Activation::ReLU,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
        // The column axis takes a dilation of 2, and the row axis keeps a solid kernel
        GoldenCase::new("valid_dilation_1x2", &SHAPE_2D, || {
            build(
                (2, 2),
                (1, 1),
                PaddingType::Valid,
                1,
                (1, 2),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&KERNEL_AND_BIAS),
    ]
}

// ---------------------------------------------------------------------------------------
// The separable convolutions
// ---------------------------------------------------------------------------------------

/// SeparableConv1D cases: 1 depthwise kernel per channel, 2 per channel at stride 2, and 1
/// dilated pass.
///
/// A separable layer runs a depthwise stage and then a pointwise stage. The record therefore
/// holds 3 parameter gradients, and the pointwise kernel is `[1, channels * depth_multiplier,
/// filters]`.
///
/// `valid_dilation_2` dilates the depthwise stage alone. The pointwise stage reads 1 cell, so
/// no tap spacing reaches it. The 3 taps of the depthwise kernel span 5 of the 6 input cells,
/// which leaves 2 windows.
fn separable_conv_1d_cases() -> Vec<GoldenCase> {
    /// Builds a 3-filter SeparableConv1D over 2 channels with a kernel of 3.
    fn build(
        stride: usize,
        padding: PaddingType,
        depth_multiplier: usize,
        dilation_rate: usize,
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer = SeparableConv1D::new(3, 3, stride, depth_multiplier, activation)
            .expect("a kernel of 3 fits a length of 6")
            .with_padding(padding)
            .with_dilation_rate(dilation_rate)
            .expect("a positive dilation");
        layer
            .build(&Shape::known(&SHAPE_1D))
            .expect("the layer accepts the shape of the case");
        let depthwise_count = 3 * 2 * depth_multiplier;
        let pointwise_count = 2 * depth_multiplier * 3;
        let depthwise = golden_weights(&[3, 2, depth_multiplier])
            .into_dimensionality::<Ix3>()
            .expect("the depthwise kernel is rank 3");
        let pointwise = golden_weights_from(&[1, 2 * depth_multiplier, 3], depthwise_count)
            .into_dimensionality::<Ix3>()
            .expect("the pointwise kernel is rank 3");
        layer
            .set_weights(
                depthwise,
                pointwise,
                bias(3, depthwise_count + pointwise_count),
            )
            .expect("every shape matches the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_multiplier_1", &SHAPE_1D, || {
            build(1, PaddingType::Valid, 1, 1, Activation::Linear)
        })
        .with_parameter_grads(&SEPARABLE_PARAMETERS),
        GoldenCase::new("same_multiplier_2_stride_2", &SHAPE_1D, || {
            build(2, PaddingType::Same, 2, 1, Activation::ReLU)
        })
        .with_parameter_grads(&SEPARABLE_PARAMETERS),
        // 3 depthwise taps 2 cells apart span 5 cells, so a length of 6 keeps 2 windows
        GoldenCase::new("valid_dilation_2", &SHAPE_1D, || {
            build(1, PaddingType::Valid, 1, 2, Activation::Linear)
        })
        .with_parameter_grads(&SEPARABLE_PARAMETERS),
    ]
}

/// SeparableConv2D cases: a Valid pass with a non-square kernel, and a Same pass at stride 2
/// with 2 depthwise kernels per channel.
///
/// The pointwise kernel is `[1, 1, channels * depth_multiplier, filters]`, so the 2 leading
/// axes stay at 1 while the depth multiplier widens the third axis.
///
/// `valid_dilation_2x1` dilates the row axis of the depthwise stage alone. The taps then span 3
/// rows and 2 columns, which gives 3 windows on each axis. The pointwise stage reads 1 cell, so
/// no tap spacing reaches it.
fn separable_conv_2d_cases() -> Vec<GoldenCase> {
    /// Builds a SeparableConv2D over 2 channels.
    fn build(
        filters: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: PaddingType,
        depth_multiplier: usize,
        dilation_rate: (usize, usize),
        activation: Activation,
    ) -> Box<dyn Layer> {
        let mut layer =
            SeparableConv2D::new(filters, kernel_size, strides, depth_multiplier, activation)
                .expect("the kernel fits the input")
                .with_padding(padding)
                .with_dilation_rate(dilation_rate)
                .expect("a positive dilation");
        layer
            .build(&Shape::known(&SHAPE_2D))
            .expect("the layer accepts the shape of the case");
        let depthwise_shape = [kernel_size.0, kernel_size.1, 2, depth_multiplier];
        let pointwise_shape = [1, 1, 2 * depth_multiplier, filters];
        let depthwise_count: usize = depthwise_shape.iter().product();
        let pointwise_count: usize = pointwise_shape.iter().product();
        let depthwise = golden_weights(&depthwise_shape)
            .into_dimensionality::<Ix4>()
            .expect("the depthwise kernel is rank 4");
        let pointwise = golden_weights_from(&pointwise_shape, depthwise_count)
            .into_dimensionality::<Ix4>()
            .expect("the pointwise kernel is rank 4");
        layer
            .set_weights(
                depthwise,
                pointwise,
                bias(filters, depthwise_count + pointwise_count),
            )
            .expect("every shape matches the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("valid_kernel_3x2", &SHAPE_2D, || {
            build(
                3,
                (3, 2),
                (1, 1),
                PaddingType::Valid,
                1,
                (1, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&SEPARABLE_PARAMETERS),
        GoldenCase::new("same_multiplier_2_stride_2", &SHAPE_2D, || {
            build(
                2,
                (3, 3),
                (2, 2),
                PaddingType::Same,
                2,
                (1, 1),
                Activation::ReLU,
            )
        })
        .with_parameter_grads(&SEPARABLE_PARAMETERS),
        // The row axis takes a dilation of 2, and the column axis keeps a solid kernel
        GoldenCase::new("valid_dilation_2x1", &SHAPE_2D, || {
            build(
                3,
                (2, 2),
                (1, 1),
                PaddingType::Valid,
                1,
                (2, 1),
                Activation::Linear,
            )
        })
        .with_parameter_grads(&SEPARABLE_PARAMETERS),
    ]
}
