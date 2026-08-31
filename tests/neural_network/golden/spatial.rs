//! Golden fixtures for the spatial family.
//!
//! Covers the pooling layers (windowed and global, max and average, at rank 1, 2, and 3), the
//! zero-padding and cropping border layers, and the upsampling layers. The data file is
//! `golden/data/spatial.golden`.
//!
//! None of these layers holds a parameter, so every case records only `input`, `forward`,
//! `predict`, and `grad_input`. See the module doc comment of the parent module for the
//! contract, and see `misc.rs` for a worked family.
//!
//! Coverage notes:
//!
//! - `MaxPooling2D/same_remainder_5x5` and `MaxPooling3D/depth_remainder` pick a spatial extent
//!   that a pooling window does not divide evenly, so a remainder position goes unused
//! - `ZeroPadding2D/asymmetric_named`, `ZeroPadding3D/asymmetric_named`,
//!   `Cropping2D/asymmetric_named`, and `Cropping3D/asymmetric_named` name a different amount at
//!   each edge of an axis
//! - `UpSampling2D` records 1 case per interpolation mode it supports
//! - every global pooling layer builds with no input shape argument, so its cached shape starts
//!   empty and the forward pass is what first learns it. Every case here exercises that path
//! - every case with a `handbuilt_` label reads a tensor that this file writes by hand, in place
//!   of the generated input. See [`HandBuiltInput`] for why the generated input cannot reach the
//!   tie rule, the NaN rule, or the negative-infinity start value of the max fold
//! - average pooling has no rule of that class. Its window fold is a sum and a divide, so a NaN
//!   spreads by arithmetic and a repeated value changes nothing. Its 1 comparable branch is the
//!   guard that gives an empty window the scale 0.0, and no input reaches it: a constructor
//!   rejects a window wider than its axis, and `Same` padding puts the leading pad at
//!   `pad_total / 2`, which stays below the window width, so every window keeps at least 1
//!   position inside the input. That branch is therefore untestable through the layer API

use super::{GoldenCase, LayerFixture};
use ndarray::IxDyn;
use rustyml::error::Error;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::border::{Cropping1D, Cropping2D, Cropping3D};
use rustyml::neural_network::layers::border::{ZeroPadding1D, ZeroPadding2D, ZeroPadding3D};
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::layers::pooling::{
    AveragePooling1D, AveragePooling2D, AveragePooling3D,
};
use rustyml::neural_network::layers::pooling::{
    GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D,
};
use rustyml::neural_network::layers::pooling::{
    GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,
};
use rustyml::neural_network::layers::pooling::{MaxPooling1D, MaxPooling2D, MaxPooling3D};
use rustyml::neural_network::layers::upsampling::{
    Interpolation, UpSampling1D, UpSampling2D, UpSampling3D,
};
use rustyml::neural_network::traits::{Layer, ParamGrad};

/// Every layer type of the spatial family, in the order the data file records them.
fn fixtures() -> Vec<LayerFixture> {
    vec![
        LayerFixture::new("MaxPooling1D", max_pooling_1d_cases),
        LayerFixture::new("MaxPooling2D", max_pooling_2d_cases),
        LayerFixture::new("MaxPooling3D", max_pooling_3d_cases),
        LayerFixture::new("AveragePooling1D", average_pooling_1d_cases),
        LayerFixture::new("AveragePooling2D", average_pooling_2d_cases),
        LayerFixture::new("AveragePooling3D", average_pooling_3d_cases),
        LayerFixture::new("GlobalMaxPooling1D", global_max_pooling_1d_cases),
        LayerFixture::new("GlobalMaxPooling2D", global_max_pooling_2d_cases),
        LayerFixture::new("GlobalMaxPooling3D", global_max_pooling_3d_cases),
        LayerFixture::new("GlobalAveragePooling1D", global_average_pooling_1d_cases),
        LayerFixture::new("GlobalAveragePooling2D", global_average_pooling_2d_cases),
        LayerFixture::new("GlobalAveragePooling3D", global_average_pooling_3d_cases),
        LayerFixture::new("ZeroPadding1D", zero_padding_1d_cases),
        LayerFixture::new("ZeroPadding2D", zero_padding_2d_cases),
        LayerFixture::new("ZeroPadding3D", zero_padding_3d_cases),
        LayerFixture::new("Cropping1D", cropping_1d_cases),
        LayerFixture::new("Cropping2D", cropping_2d_cases),
        LayerFixture::new("Cropping3D", cropping_3d_cases),
        LayerFixture::new("UpSampling1D", up_sampling_1d_cases),
        LayerFixture::new("UpSampling2D", up_sampling_2d_cases),
        LayerFixture::new("UpSampling3D", up_sampling_3d_cases),
    ]
}

/// Replays the spatial family against `golden/data/spatial.golden`.
#[test]
fn golden_spatial_family() {
    super::run_family("spatial", &fixtures());
}

// ---------------------------------------------------------------------------------------
// Hand-built inputs for the max-fold rules that the generated input cannot reach
// ---------------------------------------------------------------------------------------

/// A layer wrapper that gives the layer under test a tensor that this file writes by hand.
///
/// # Why a fixture needs this
///
/// The harness builds every input from 1 formula, `((index * 37) % 101 - 50) / 25`. The formula
/// takes 101 distinct values and repeats with a period of 101 in the flat index. A pooling
/// window at a fixed channel never spans 2 flat indices that are 101 apart, because that needs
/// more than 101 channels or a window of more than 101 positions, and the recorded tensors hold
/// at most 256 elements. No generated pooling window can therefore hold the same value 2 times.
/// The formula also produces no NaN and no infinity.
///
/// 3 documented rules of `fold_max` in the pooling engine are out of reach of such an input:
///
/// 1. on a tie, the first maximum in window order wins, and that choice decides the input
///    position that the gradient reaches
/// 2. a NaN wins the window, the first NaN keeps the position, a later NaN does not replace it,
///    and no finite value displaces it
/// 3. a window whose every element loses to the negative-infinity start value keeps the seed of
///    the arg-max, which is the first position that the window covers, on the channel of the
///    output element. A global fold reduces the whole item, so its seed is position 0 of the
///    channel of the output element
///
/// # How it works
///
/// The wrapper ignores the tensor that the harness supplies and hands the stored tensor to the
/// layer under test. Every other method goes straight to that layer, so the case records the
/// layer type, the metadata, and the gradients of the pooling layer alone.
///
/// A case that uses the wrapper declares the input shape `[1]`. No pooling layer accepts a
/// rank-1 input, so the recorded `input` tensor is visibly a placeholder. A reader of the data
/// file cannot mistake it for the data that produced the recorded output. The hand-built values
/// are in the builder function of the case, next to the expected result.
struct HandBuiltInput {
    /// The pooling layer under test
    inner: Box<dyn Layer>,
    /// The tensor that the pooling layer reads, in place of the harness input
    input: Tensor,
}

impl HandBuiltInput {
    /// Wraps a layer together with the tensor it reads.
    ///
    /// # Parameters
    ///
    /// - `shape` - Shape of the hand-built tensor, batch axis first
    /// - `values` - Every element of the hand-built tensor, in flat C order
    /// - `inner` - The pooling layer under test
    ///
    /// # Returns
    ///
    /// - `Box<dyn Layer>` - The wrapped layer, ready for a `GoldenCase` builder
    ///
    /// # Panics
    ///
    /// - If the value count is not the product of the shape
    fn boxed(shape: &[usize], values: &[f32], inner: Box<dyn Layer>) -> Box<dyn Layer> {
        let input = Tensor::from_shape_vec(IxDyn(shape), values.to_vec())
            .expect("the value count is the shape product");
        Box::new(Self { inner, input })
    }
}

impl Layer for HandBuiltInput {
    fn forward(&mut self, _harness_input: &Tensor) -> Result<Tensor, Error> {
        self.inner.forward(&self.input)
    }

    fn predict(&self, _harness_input: &Tensor) -> Result<Tensor, Error> {
        self.inner.predict(&self.input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        self.inner.backward(grad_output)
    }

    fn layer_type(&self) -> &str {
        self.inner.layer_type()
    }

    fn output_shape(&self) -> String {
        self.inner.output_shape()
    }

    fn param_count(&self) -> TrainingParameters {
        self.inner.param_count()
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        self.inner.parameters()
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        self.inner.get_weights()
    }

    fn set_training_if_mode_dependent(&mut self, is_training: bool) {
        self.inner.set_training_if_mode_dependent(is_training);
    }
}

/// The input shape that a hand-built case declares, so the placeholder stays 1 value wide.
const PLACEHOLDER_SHAPE: [usize; 1] = [1];

// The upstream gradient of a hand-built case comes from the harness formula
// `((index * 29) % 71 - 35) / 20`, over the flat index of the forward output. The first 4 values
// are -1.75, -0.3, 1.15, and -0.95. Every expected gradient below names them by value.

// ---------------------------------------------------------------------------------------
// Windowed pooling: MaxPooling and AveragePooling at rank 1, 2, and 3
// ---------------------------------------------------------------------------------------

/// MaxPooling1D cases: a window that divides the length evenly, a custom stride that leaves a
/// trailing position unused, and 3 hand-built cases for the rules of the max fold.
fn max_pooling_1d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("pool2_stride2", &[2, 8, 3], || {
            Box::new(MaxPooling1D::new(2, vec![2, 8, 3]).expect("a 3D input shape"))
        }),
        GoldenCase::new("pool3_stride2_remainder", &[2, 7, 4], || {
            Box::new(
                MaxPooling1D::new(3, vec![2, 7, 4])
                    .expect("a 3D input shape")
                    .with_stride(2)
                    .expect("a positive stride"),
            )
        }),
        // Pins rule 1 of `HandBuiltInput`: a tie goes to the first maximum in window order.
        //
        // The input is [1, 6, 2], and the window is 3 wide with a stride of 3. The flat index of
        // position `p` on channel `c` is `p * 2 + c`. The 2 channels put the tie in a different
        // place, so a rule that always takes position 0 of the window also fails here:
        //
        //   channel 0: -1.0,  3.0,  3.0 | 0.5, -2.0, 0.25
        //   channel 1:  4.0,  4.0,  1.0 | 6.0,  6.0, 6.00
        //
        // Window 0, channel 0 ties at position 1 and position 2. First wins, so the winner is
        // position 1, at flat index 2. Last-wins would take position 2, at flat index 4.
        // Window 0, channel 1 ties at position 0 and position 1. First wins, so the winner is
        // position 0, at flat index 1. Last-wins would take position 1, at flat index 3.
        // Window 1, channel 0 has the single maximum 0.5 at position 3, at flat index 6.
        // Window 1, channel 1 ties at all 3 positions. First wins, so the winner is position 3,
        // at flat index 7. Last-wins would take position 5, at flat index 11.
        //
        // The forward output is [3.0, 4.0, 0.5, 6.0]. The upstream gradient is
        // [-1.75, -0.3, 1.15, -0.95], so grad_input must hold -1.75 at flat index 2, -0.3 at
        // flat index 1, 1.15 at flat index 6, -0.95 at flat index 7, and 0.0 elsewhere.
        GoldenCase::new("handbuilt_tie_pool3", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 6, 2],
                &[
                    -1.0, 4.0, // position 0
                    3.0, 4.0, // position 1
                    3.0, 1.0, // position 2
                    0.5, 6.0, // position 3
                    -2.0, 6.0, // position 4
                    0.25, 6.0, // position 5
                ],
                Box::new(MaxPooling1D::new(3, vec![1, 6, 2]).expect("a 3D input shape")),
            )
        }),
        // Pins rule 2 of `HandBuiltInput`: the NaN rule of the max fold.
        //
        // The input is [1, 6, 1], and the window is 3 wide with a stride of 3:
        //
        //   1.0, NaN, 9.0 | NaN, 2.0, NaN
        //
        // Window 0 takes 1.0 first, then the NaN replaces it, and then 9.0 cannot displace the
        // NaN because `9.0 > NaN` is false. The winner is flat index 1.
        // Window 1 takes the NaN first, 2.0 cannot displace it, and the second NaN does not
        // replace the first one. The winner is flat index 3, not flat index 5.
        //
        // The forward output is [NaN, NaN], and the harness compares raw bits, so each NaN
        // compares equal to itself. The upstream gradient is [-1.75, -0.3], so grad_input must
        // hold -1.75 at flat index 1, -0.3 at flat index 3, and 0.0 elsewhere. The gradient
        // placement is what shows which NaN won.
        GoldenCase::new("handbuilt_nan_wins_window", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 6, 1],
                &[1.0, f32::NAN, 9.0, f32::NAN, 2.0, f32::NAN],
                Box::new(MaxPooling1D::new(3, vec![1, 6, 1]).expect("a 3D input shape")),
            )
        }),
        // Pins rule 3 of `HandBuiltInput`: a window that no element wins keeps its own seed.
        //
        // The input is [1, 4, 1], and the window is 2 wide with a stride of 2:
        //
        //   1.0, 2.0 | -inf, -inf
        //
        // The accumulator starts at negative infinity. Window 1 holds negative infinity 2 times,
        // and `-inf > -inf` is false, so no element of the window ever wins the fold. The
        // recorded arg-max of window 1 therefore stays at the seed, which is the first position
        // that window 1 covers, at flat index 2. The output of window 1 is negative infinity.
        //
        // The forward output is [2.0, -inf]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 1, which is the winner of window 0, -0.3 at
        // flat index 2, which is the first position of window 1, and 0.0 at flat index 0 and
        // flat index 3.
        //
        // Keras 3 on the JAX backend does the same. It accepts the non-finite input, gives
        // negative infinity as the pooled value, and sends the gradient of the window to the
        // first position that the window covers.
        GoldenCase::new(
            "handbuilt_all_negative_infinity",
            &PLACEHOLDER_SHAPE,
            || {
                HandBuiltInput::boxed(
                    &[1, 4, 1],
                    &[1.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
                    Box::new(MaxPooling1D::new(2, vec![1, 4, 1]).expect("a 3D input shape")),
                )
            },
        ),
    ]
}

/// MaxPooling2D cases: a square window that divides both axes evenly, Same padding over an odd
/// extent that the window does not divide, a hand-built tie in every window, and a hand-built
/// window of negative infinity alone.
fn max_pooling_2d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("pool2x2_stride2x2", &[2, 6, 6, 3], || {
            Box::new(MaxPooling2D::new((2, 2), vec![2, 6, 6, 3]).expect("a 4D input shape"))
        }),
        GoldenCase::new("same_remainder_5x5", &[2, 5, 5, 2], || {
            Box::new(
                MaxPooling2D::new((2, 2), vec![2, 5, 5, 2])
                    .expect("a 4D input shape")
                    .with_padding(PaddingType::Same),
            )
        }),
        // Pins rule 1 of `HandBuiltInput` at rank 2, and with it the window scan order.
        //
        // The input is [1, 4, 4, 1], and the window is 2 by 2 with a stride of 2 on both axes.
        // The engine walks a window with the last axis fastest, so the scan order is
        // (0, 0), (0, 1), (1, 0), (1, 1). The flat index of row `r` and column `c` is
        // `r * 4 + c`. The grid is:
        //
        //   row 0:  1.0   9.0 |  2.0   2.0
        //   row 1:  9.0   3.0 |  0.0   2.0
        //   ------------------+-----------
        //   row 2:  4.0  -1.0 |  7.0   7.0
        //   row 3:  0.5   4.0 |  7.0   5.0
        //
        // Window (0, 0) ties at (0, 1) and (1, 0). First wins, so the winner is (0, 1), at flat
        // index 1. A last-wins rule takes (1, 0), at flat index 4. A column-first scan would
        // also take (1, 0), so this window pins the scan order as well.
        // Window (0, 1) ties at (0, 2), (0, 3), and (1, 3). First wins, so the winner is (0, 2),
        // at flat index 2. Last-wins takes (1, 3), at flat index 7.
        // Window (1, 0) ties at (2, 0) and (3, 1). First wins, so the winner is (2, 0), at flat
        // index 8. Last-wins takes (3, 1), at flat index 13.
        // Window (1, 1) ties at (2, 2), (2, 3), and (3, 2). First wins, so the winner is (2, 2),
        // at flat index 10. Last-wins takes (3, 2), at flat index 14.
        //
        // The forward output is [9.0, 2.0, 4.0, 7.0]. The upstream gradient is
        // [-1.75, -0.3, 1.15, -0.95], so grad_input must hold -1.75 at flat index 1, -0.3 at
        // flat index 2, 1.15 at flat index 8, -0.95 at flat index 10, and 0.0 elsewhere.
        GoldenCase::new("handbuilt_tie_row_major", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 4, 4, 1],
                &[
                    1.0, 9.0, 2.0, 2.0, // row 0
                    9.0, 3.0, 0.0, 2.0, // row 1
                    4.0, -1.0, 7.0, 7.0, // row 2
                    0.5, 4.0, 7.0, 5.0, // row 3
                ],
                Box::new(MaxPooling2D::new((2, 2), vec![1, 4, 4, 1]).expect("a 4D input shape")),
            )
        }),
        // Pins rule 3 of `HandBuiltInput` at rank 2.
        //
        // The input is [1, 2, 4, 1], and the window is 2 by 2 with a stride of 2 on both axes.
        // The flat index of row `r` and column `c` is `r * 4 + c`. The grid is:
        //
        //   row 0:  1.0   2.0 | -inf  -inf
        //   row 1:  3.0   0.5 | -inf  -inf
        //
        // Window (0, 0) has its maximum 3.0 at (1, 0), at flat index 4.
        // Window (0, 1) holds negative infinity in all 4 cells, so no element wins the fold. Its
        // arg-max stays at the seed, which is the first cell of the window in window order, at
        // (0, 2) and flat index 2.
        //
        // The forward output is [3.0, -inf]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 4, -0.3 at flat index 2, and 0.0 elsewhere.
        GoldenCase::new(
            "handbuilt_all_negative_infinity",
            &PLACEHOLDER_SHAPE,
            || {
                HandBuiltInput::boxed(
                    &[1, 2, 4, 1],
                    &[
                        1.0,
                        2.0,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY, // row 0
                        3.0,
                        0.5,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY, // row 1
                    ],
                    Box::new(
                        MaxPooling2D::new((2, 2), vec![1, 2, 4, 1]).expect("a 4D input shape"),
                    ),
                )
            },
        ),
    ]
}

/// MaxPooling3D cases: a cube window that divides every axis evenly, a window that leaves a
/// remainder on the depth axis alone, a hand-built tie that spans the depth axis, and a
/// hand-built window of negative infinity alone.
fn max_pooling_3d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("pool2x2x2_valid", &[2, 4, 4, 4, 1], || {
            Box::new(MaxPooling3D::new((2, 2, 2), vec![2, 4, 4, 4, 1]).expect("a 5D input shape"))
        }),
        GoldenCase::new("depth_remainder", &[2, 5, 3, 3, 1], || {
            Box::new(MaxPooling3D::new((2, 3, 3), vec![2, 5, 3, 3, 1]).expect("a 5D input shape"))
        }),
        // Pins rule 1 of `HandBuiltInput` at rank 3, with a tie that spans all 3 spatial axes.
        //
        // The input is [1, 2, 2, 4, 1], and the window is 2 by 2 by 2 with a stride of 2 on each
        // axis. The width axis holds 4 positions, so the output holds 2 windows. The flat index
        // of depth `d`, height `h`, and width `w` is `d * 8 + h * 4 + w`. The engine walks a
        // window with the last axis fastest, so the scan order of a window is
        // (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1).
        //
        //   depth 0, height 0:  1.0   3.0 |  6.0   6.0
        //   depth 0, height 1:  8.0   2.0 | -1.0   0.0
        //   depth 1, height 0:  8.0   0.0 |  2.0   6.0
        //   depth 1, height 1: -4.0   8.0 |  5.0   1.0
        //
        // Window 0 covers the widths 0 and 1, and it ties at (0, 1, 0), (1, 0, 0), and
        // (1, 1, 1). First wins, so the winner is (0, 1, 0), at flat index 4. Last-wins takes
        // (1, 1, 1), at flat index 13.
        // Window 1 covers the widths 2 and 3, and it ties at (0, 0, 2), (0, 0, 3), and
        // (1, 0, 3). First wins, so the winner is (0, 0, 2), at flat index 2. Last-wins takes
        // (1, 0, 3), at flat index 11.
        //
        // The forward output is [8.0, 6.0]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 4, -0.3 at flat index 2, and 0.0 elsewhere.
        GoldenCase::new("handbuilt_tie_across_depth", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 2, 2, 4, 1],
                &[
                    1.0, 3.0, 6.0, 6.0, // depth 0, height 0
                    8.0, 2.0, -1.0, 0.0, // depth 0, height 1
                    8.0, 0.0, 2.0, 6.0, // depth 1, height 0
                    -4.0, 8.0, 5.0, 1.0, // depth 1, height 1
                ],
                Box::new(
                    MaxPooling3D::new((2, 2, 2), vec![1, 2, 2, 4, 1]).expect("a 5D input shape"),
                ),
            )
        }),
        // Pins rule 3 of `HandBuiltInput` at rank 3.
        //
        // The input is [1, 2, 2, 4, 1], and the window is 2 by 2 by 2 with a stride of 2 on each
        // axis. The flat index of depth `d`, height `h`, and width `w` is `d * 8 + h * 4 + w`.
        // The widths 2 and 3 hold negative infinity at every depth and height:
        //
        //   depth 0, height 0:  1.0   3.0 | -inf  -inf
        //   depth 0, height 1:  8.0   2.0 | -inf  -inf
        //   depth 1, height 0:  5.0   0.0 | -inf  -inf
        //   depth 1, height 1: -4.0   6.0 | -inf  -inf
        //
        // Window 0 covers the widths 0 and 1, and its maximum 8.0 sits at (0, 1, 0), at flat
        // index 4.
        // Window 1 covers the widths 2 and 3, and it holds negative infinity in all 8 voxels, so
        // no element wins the fold. Its arg-max stays at the seed, which is the first voxel of
        // the window in window order, at (0, 0, 2) and flat index 2.
        //
        // The forward output is [8.0, -inf]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 4, -0.3 at flat index 2, and 0.0 elsewhere.
        GoldenCase::new(
            "handbuilt_all_negative_infinity",
            &PLACEHOLDER_SHAPE,
            || {
                let negative = f32::NEG_INFINITY;
                HandBuiltInput::boxed(
                    &[1, 2, 2, 4, 1],
                    &[
                        1.0, 3.0, negative, negative, // depth 0, height 0
                        8.0, 2.0, negative, negative, // depth 0, height 1
                        5.0, 0.0, negative, negative, // depth 1, height 0
                        -4.0, 6.0, negative, negative, // depth 1, height 1
                    ],
                    Box::new(
                        MaxPooling3D::new((2, 2, 2), vec![1, 2, 2, 4, 1])
                            .expect("a 5D input shape"),
                    ),
                )
            },
        ),
    ]
}

/// AveragePooling1D cases: a window that divides the length evenly, Same padding with a custom
/// stride over a length the window does not divide, and a hand-built NaN.
///
/// Average pooling holds no tie rule and no NaN rule. Its window fold is a sum and a divide, so
/// a NaN spreads through the sum by arithmetic alone. Its 1 comparable branch is the guard for
/// an empty window, and no input can reach that guard. See the module doc comment.
fn average_pooling_1d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("pool3_stride3", &[2, 6, 2], || {
            Box::new(AveragePooling1D::new(3, vec![2, 6, 2]).expect("a 3D input shape"))
        }),
        GoldenCase::new("pool3_stride2_same", &[2, 7, 2], || {
            Box::new(
                AveragePooling1D::new(3, vec![2, 7, 2])
                    .expect("a 3D input shape")
                    .with_stride(2)
                    .expect("a positive stride")
                    .with_padding(PaddingType::Same),
            )
        }),
        // Records how average pooling answers a NaN, as the contrast to the max fold.
        //
        // The input is [1, 4, 1], and the window is 2 wide with a stride of 2:
        //
        //   1.0, NaN | 4.0, 6.0
        //
        // Window 0 sums 1.0 and the NaN, which gives a NaN, and the divide by 2 keeps it. The
        // NaN therefore reaches only its own window. Window 1 gives (4.0 + 6.0) / 2, which is
        // 5.0, so the forward output is [NaN, 5.0].
        //
        // The backward pass of average pooling spreads each output gradient evenly over its
        // window, and it never reads an input value. The NaN therefore leaves no mark on
        // grad_input. The upstream gradient is [-1.75, -0.3], so grad_input must hold
        // -0.875 at flat index 0 and flat index 1, and -0.15 at flat index 2 and flat index 3.
        GoldenCase::new("handbuilt_nan_in_window", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 4, 1],
                &[1.0, f32::NAN, 4.0, 6.0],
                Box::new(AveragePooling1D::new(2, vec![1, 4, 1]).expect("a 3D input shape")),
            )
        }),
    ]
}

/// AveragePooling2D cases: a non-square window that divides both axes evenly, and a stride
/// smaller than the window so neighboring windows overlap.
fn average_pooling_2d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("pool3x2_nonsquare", &[2, 6, 4, 2], || {
            Box::new(AveragePooling2D::new((3, 2), vec![2, 6, 4, 2]).expect("a 4D input shape"))
        }),
        GoldenCase::new("overlap_stride1x1", &[2, 5, 5, 2], || {
            Box::new(
                AveragePooling2D::new((2, 2), vec![2, 5, 5, 2])
                    .expect("a 4D input shape")
                    .with_strides((1, 1))
                    .expect("positive strides"),
            )
        }),
    ]
}

/// AveragePooling3D cases: Same padding over a depth extent the window does not divide, and
/// an overlapping stride on every axis.
fn average_pooling_3d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("same_depth_remainder", &[2, 5, 4, 4, 1], || {
            Box::new(
                AveragePooling3D::new((2, 2, 2), vec![2, 5, 4, 4, 1])
                    .expect("a 5D input shape")
                    .with_padding(PaddingType::Same),
            )
        }),
        GoldenCase::new("overlap_stride", &[2, 4, 4, 4, 1], || {
            Box::new(
                AveragePooling3D::new((2, 2, 2), vec![2, 4, 4, 4, 1])
                    .expect("a 5D input shape")
                    .with_strides((1, 1, 1))
                    .expect("positive strides"),
            )
        }),
    ]
}

// ---------------------------------------------------------------------------------------
// Global pooling: rank 1, 2, and 3, max and average
// ---------------------------------------------------------------------------------------
//
// A global pooling layer takes no input shape at construction. Its cached shape starts empty
// and the forward pass is what first learns it, so every case below exercises that path.

/// GlobalMaxPooling1D cases: a rank-3 input, a hand-built tie, a hand-built NaN, and a
/// hand-built channel of negative infinity alone.
///
/// The global fold reduces every position of a channel with the same `fold_max` that a window
/// uses, so the tie rule, the NaN rule, and the seed rule hold here too.
fn global_max_pooling_1d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new(
            "default",
            &[2, 7, 3],
            || Box::new(GlobalMaxPooling1D::new()),
        ),
        // Pins rule 1 of `HandBuiltInput` for the global fold, on 2 channels at once.
        //
        // The input is [1, 4, 2]. The flat index of position `p` on channel `c` is `p * 2 + c`.
        //
        //   channel 0: 2.0, 6.0, 6.0, -1.0
        //   channel 1: 7.0, 7.0, 3.0,  7.0
        //
        // Channel 0 ties at position 1 and position 2. First wins, so the winner is position 1,
        // at flat index 2. Last-wins takes position 2, at flat index 4.
        // Channel 1 ties at position 0, position 1, and position 3. First wins, so the winner is
        // position 0, at flat index 1. Last-wins takes position 3, at flat index 7.
        //
        // The forward output is [6.0, 7.0]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 2, -0.3 at flat index 1, and 0.0 elsewhere.
        GoldenCase::new("handbuilt_tie_first_position", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 4, 2],
                &[
                    2.0, 7.0, // position 0
                    6.0, 7.0, // position 1
                    6.0, 3.0, // position 2
                    -1.0, 7.0, // position 3
                ],
                Box::new(GlobalMaxPooling1D::new()),
            )
        }),
        // Pins rule 2 of `HandBuiltInput` for the global fold.
        //
        // The input is [1, 4, 1]:
        //
        //   3.0, NaN, 5.0, NaN
        //
        // The fold takes 3.0 first, the first NaN replaces it, 5.0 cannot displace the NaN, and
        // the second NaN does not replace the first one. The winner is flat index 1, not flat
        // index 3.
        //
        // The forward output is [NaN]. The upstream gradient is [-1.75], so grad_input must hold
        // -1.75 at flat index 1 and 0.0 elsewhere.
        GoldenCase::new("handbuilt_nan_wins_channel", &PLACEHOLDER_SHAPE, || {
            HandBuiltInput::boxed(
                &[1, 4, 1],
                &[3.0, f32::NAN, 5.0, f32::NAN],
                Box::new(GlobalMaxPooling1D::new()),
            )
        }),
        // Pins rule 3 of `HandBuiltInput` for the global fold, where the seed must keep the
        // channel of the output element.
        //
        // The input is [1, 4, 2]. The flat index of position `p` on channel `c` is `p * 2 + c`.
        //
        //   channel 0:  1.0,  2.0,  3.0,  0.0
        //   channel 1: -inf, -inf, -inf, -inf
        //
        // Channel 0 has its maximum 3.0 at position 2, at flat index 4.
        // Channel 1 holds negative infinity at every position, so no element wins the fold. Its
        // arg-max stays at the seed, which is position 0 of channel 1, at flat index 1. A seed
        // of flat index 0 would send the gradient of channel 1 to channel 0.
        //
        // The forward output is [3.0, -inf]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 4, -0.3 at flat index 1, and 0.0 elsewhere.
        GoldenCase::new(
            "handbuilt_all_negative_infinity",
            &PLACEHOLDER_SHAPE,
            || {
                let negative = f32::NEG_INFINITY;
                HandBuiltInput::boxed(
                    &[1, 4, 2],
                    &[
                        1.0, negative, // position 0
                        2.0, negative, // position 1
                        3.0, negative, // position 2
                        0.0, negative, // position 3
                    ],
                    Box::new(GlobalMaxPooling1D::new()),
                )
            },
        ),
    ]
}

/// GlobalMaxPooling2D cases: a rank-4 input, and a hand-built channel of negative infinity
/// alone.
fn global_max_pooling_2d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("default", &[2, 5, 4, 3], || {
            Box::new(GlobalMaxPooling2D::new())
        }),
        // Pins rule 3 of `HandBuiltInput` for the global fold at rank 2.
        //
        // The input is [1, 2, 2, 2], which holds 4 positions on 2 channels. The flat index of
        // position `p` on channel `c` is `p * 2 + c`, and the positions run row by row.
        //
        //   channel 0:  1.0,  2.0,  3.0,  0.0
        //   channel 1: -inf, -inf, -inf, -inf
        //
        // Channel 0 has its maximum 3.0 at position 2, at flat index 4. Channel 1 keeps the
        // seed, which is position 0 of channel 1, at flat index 1.
        //
        // The forward output is [3.0, -inf]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 4, -0.3 at flat index 1, and 0.0 elsewhere.
        GoldenCase::new(
            "handbuilt_all_negative_infinity",
            &PLACEHOLDER_SHAPE,
            || {
                let negative = f32::NEG_INFINITY;
                HandBuiltInput::boxed(
                    &[1, 2, 2, 2],
                    &[
                        1.0, negative, // row 0, column 0
                        2.0, negative, // row 0, column 1
                        3.0, negative, // row 1, column 0
                        0.0, negative, // row 1, column 1
                    ],
                    Box::new(GlobalMaxPooling2D::new()),
                )
            },
        ),
    ]
}

/// GlobalMaxPooling3D cases: a rank-5 input, and a hand-built channel of negative infinity
/// alone.
fn global_max_pooling_3d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("default", &[2, 3, 3, 2, 2], || {
            Box::new(GlobalMaxPooling3D::new())
        }),
        // Pins rule 3 of `HandBuiltInput` for the global fold at rank 3.
        //
        // The input is [1, 2, 1, 2, 2], which holds 4 positions on 2 channels. The flat index of
        // position `p` on channel `c` is `p * 2 + c`, and the positions run depth by depth.
        //
        //   channel 0:  1.0,  2.0,  3.0,  0.0
        //   channel 1: -inf, -inf, -inf, -inf
        //
        // Channel 0 has its maximum 3.0 at position 2, at flat index 4. Channel 1 keeps the
        // seed, which is position 0 of channel 1, at flat index 1.
        //
        // The forward output is [3.0, -inf]. The upstream gradient is [-1.75, -0.3], so
        // grad_input must hold -1.75 at flat index 4, -0.3 at flat index 1, and 0.0 elsewhere.
        GoldenCase::new(
            "handbuilt_all_negative_infinity",
            &PLACEHOLDER_SHAPE,
            || {
                let negative = f32::NEG_INFINITY;
                HandBuiltInput::boxed(
                    &[1, 2, 1, 2, 2],
                    &[
                        1.0, negative, // depth 0, width 0
                        2.0, negative, // depth 0, width 1
                        3.0, negative, // depth 1, width 0
                        0.0, negative, // depth 1, width 1
                    ],
                    Box::new(GlobalMaxPooling3D::new()),
                )
            },
        ),
    ]
}

/// GlobalAveragePooling1D case over a rank-3 input.
fn global_average_pooling_1d_cases() -> Vec<GoldenCase> {
    vec![GoldenCase::new("default", &[2, 7, 3], || {
        Box::new(GlobalAveragePooling1D::new())
    })]
}

/// GlobalAveragePooling2D case over a rank-4 input.
fn global_average_pooling_2d_cases() -> Vec<GoldenCase> {
    vec![GoldenCase::new("default", &[2, 5, 4, 3], || {
        Box::new(GlobalAveragePooling2D::new())
    })]
}

/// GlobalAveragePooling3D case over a rank-5 input.
fn global_average_pooling_3d_cases() -> Vec<GoldenCase> {
    vec![GoldenCase::new("default", &[2, 3, 3, 2, 2], || {
        Box::new(GlobalAveragePooling3D::new())
    })]
}

// ---------------------------------------------------------------------------------------
// Border layers: ZeroPadding and Cropping at rank 1, 2, and 3
// ---------------------------------------------------------------------------------------

/// ZeroPadding1D cases: an equal amount at both ends, and an unequal `(before, after)` pair.
fn zero_padding_1d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("symmetric_2", &[2, 4, 3], || {
            Box::new(ZeroPadding1D::new(2))
        }),
        GoldenCase::new("asymmetric_1_3", &[2, 4, 3], || {
            Box::new(ZeroPadding1D::new((1, 3)))
        }),
    ]
}

/// ZeroPadding2D cases: an equal amount at all 4 edges, and a distinct amount at each of the
/// 4 edges.
fn zero_padding_2d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("symmetric_1", &[2, 4, 4, 2], || {
            Box::new(ZeroPadding2D::new(1))
        }),
        GoldenCase::new("asymmetric_named", &[2, 4, 4, 2], || {
            Box::new(ZeroPadding2D::new(((1, 0), (0, 2))))
        }),
    ]
}

/// ZeroPadding3D cases: an equal amount at all 6 faces, and a distinct amount at each of the
/// 6 faces.
fn zero_padding_3d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("symmetric_1", &[2, 3, 3, 3, 1], || {
            Box::new(ZeroPadding3D::new(1))
        }),
        GoldenCase::new("asymmetric_named", &[2, 2, 2, 2, 1], || {
            Box::new(ZeroPadding3D::new(((1, 0), (0, 2), (1, 1))))
        }),
    ]
}

/// Cropping1D cases: an equal amount off both ends, and an unequal `(before, after)` pair.
fn cropping_1d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("symmetric_1", &[2, 6, 3], || Box::new(Cropping1D::new(1))),
        GoldenCase::new("asymmetric_1_2", &[2, 6, 3], || {
            Box::new(Cropping1D::new((1, 2)))
        }),
    ]
}

/// Cropping2D cases: an equal amount off all 4 edges, and a distinct amount off each of the
/// 4 edges.
fn cropping_2d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("symmetric_1", &[2, 6, 6, 2], || {
            Box::new(Cropping2D::new(1))
        }),
        GoldenCase::new("asymmetric_named", &[2, 8, 8, 2], || {
            Box::new(Cropping2D::new(((1, 1), (2, 0))))
        }),
    ]
}

/// Cropping3D cases: an equal amount off all 6 faces, and a distinct amount off each of the
/// 6 faces.
fn cropping_3d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("symmetric_1", &[2, 4, 4, 4, 1], || {
            Box::new(Cropping3D::new(1))
        }),
        GoldenCase::new("asymmetric_named", &[2, 4, 4, 4, 1], || {
            Box::new(Cropping3D::new(((1, 0), (0, 2), (1, 1))))
        }),
    ]
}

// ---------------------------------------------------------------------------------------
// Upsampling: rank 1, 2, and 3
// ---------------------------------------------------------------------------------------

/// UpSampling1D cases: a size that repeats every step 3 times, and a size of 1 that leaves the
/// length alone.
fn up_sampling_1d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("size_3", &[2, 3, 2], || {
            Box::new(UpSampling1D::new(3).expect("a positive size"))
        }),
        GoldenCase::new("size_1_identity", &[2, 3, 2], || {
            Box::new(UpSampling1D::new(1).expect("a positive size"))
        }),
    ]
}

/// UpSampling2D cases: 1 case per interpolation mode the layer supports, over the same input
/// shape and factor, so the recorded values isolate the kernel each mode uses.
fn up_sampling_2d_cases() -> Vec<GoldenCase> {
    /// Builds a 2x factor UpSampling2D layer with 1 interpolation mode.
    fn build(interpolation: Interpolation) -> Box<dyn Layer> {
        Box::new(UpSampling2D::new(2, interpolation).expect("a positive factor"))
    }

    vec![
        GoldenCase::new("nearest", &[2, 4, 4, 2], || build(Interpolation::Nearest)),
        GoldenCase::new("bilinear", &[2, 4, 4, 2], || build(Interpolation::Bilinear)),
        GoldenCase::new("bicubic", &[2, 4, 4, 2], || build(Interpolation::Bicubic)),
        GoldenCase::new("lanczos3", &[2, 4, 4, 2], || build(Interpolation::Lanczos3)),
        GoldenCase::new("lanczos5", &[2, 4, 4, 2], || build(Interpolation::Lanczos5)),
    ]
}

/// UpSampling3D cases: a distinct factor per axis, and a uniform factor on every axis.
fn up_sampling_3d_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("factor_2_1_3", &[2, 2, 2, 1, 2], || {
            Box::new(UpSampling3D::new((2, 1, 3)).expect("positive factors"))
        }),
        GoldenCase::new("factor_2_uniform", &[2, 2, 2, 2, 1], || {
            Box::new(UpSampling3D::new(2).expect("a positive factor"))
        }),
    ]
}
