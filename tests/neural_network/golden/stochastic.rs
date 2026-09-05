//! Golden fixtures for the stochastic and normalization family.
//!
//! This file records Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D,
//! GaussianNoise, GaussianDropout, and BatchNormalization. The data file is
//! `golden/data/stochastic.golden`. The remaining normalization layers of the family
//! (LayerNormalization, GroupNormalization, InstanceNormalization, and UnitNormalization) are
//! not recorded yet.
//!
//! Every mode-dependent layer here records at least 2 cases. The training case pins the seed
//! through the layer builder, and the inference case records the pass-through behavior. Some
//! layers add a case for a rate of 0 or a rate of 1. Those 2 rates take early return paths that
//! skip the sampler altogether.
//!
//! BatchNormalization mutates its running mean and its running variance during a training
//! forward pass. That state is internal, and the harness records only what a layer returns.
//! The 2 readout cases therefore route each statistic through the output of a second layer.
//! See [`batch_norm_statistic_readout`] for how that works and why.
//!
//! See the module doc comment of the parent module for the contract, and see `misc.rs` for a
//! worked family.

use super::{GoldenCase, LayerFixture, golden_input, golden_weights, golden_weights_from};
use ndarray::IxDyn;
use rustyml::neural_network::layers::regularization::dropout::{
    Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D,
};
use rustyml::neural_network::layers::regularization::noise_injection::{
    GaussianDropout, GaussianNoise,
};
use rustyml::neural_network::layers::regularization::normalization::BatchNormalization;
use rustyml::neural_network::traits::{Layer, LayerBase, UnaryLayer};
use rustyml::neural_network::{Ctx, Shape, Tensor};

/// Every layer type of the stochastic family, in the order the data file records them.
fn fixtures() -> Vec<LayerFixture> {
    vec![
        LayerFixture::new("Dropout", dropout_cases),
        LayerFixture::new("SpatialDropout1D", spatial_dropout_1d_cases),
        LayerFixture::new("SpatialDropout2D", spatial_dropout_2d_cases),
        LayerFixture::new("SpatialDropout3D", spatial_dropout_3d_cases),
        LayerFixture::new("GaussianNoise", gaussian_noise_cases),
        LayerFixture::new("GaussianDropout", gaussian_dropout_cases),
        LayerFixture::new("BatchNormalization", batch_normalization_cases),
    ]
}

/// Replays the stochastic family against `golden/data/stochastic.golden`.
#[test]
fn golden_stochastic_family() {
    super::run_family("stochastic", &fixtures());
}

/// The seed that every sampling layer of this family takes through its own builder.
///
/// An explicit seed never reads and never advances the thread-local global seed. A test that
/// runs beside this one therefore cannot move a recorded value. A layer built without the seed
/// would draw from entropy, and nothing here would be reproducible.
///
/// This value is not arbitrary. It was picked so that every recorded mask keeps some units and
/// drops some, in both batch items. That holds for the plain layer and for all 3 spatial
/// layers. A seed that dropped nothing would record a forward output that the pass-through path
/// also produces, and the record would then prove much less.
const GOLDEN_SEED: u64 = 27;

/// The input shape of every elementwise case: 2 batch rows of 8 features.
const ELEMENTWISE_SHAPE: [usize; 2] = [2, 8];

/// The input shape of the SpatialDropout1D cases: 2 batch items, 4 positions, 5 channels.
const SPATIAL_1D_SHAPE: [usize; 3] = [2, 4, 5];

/// The input shape of the SpatialDropout2D cases: 2 batch items, 3 by 3 positions, 4 channels.
const SPATIAL_2D_SHAPE: [usize; 4] = [2, 3, 3, 4];

/// The input shape of the SpatialDropout3D cases: 2 items, 2 by 2 by 3 positions, 4 channels.
const SPATIAL_3D_SHAPE: [usize; 5] = [2, 2, 2, 3, 4];

// ---------------------------------------------------------------------------------------
// The dropout layers
// ---------------------------------------------------------------------------------------

/// Dropout cases: the sampled path in both modes, the 2 early return paths, and a shared mask
/// axis in both modes.
///
/// `rate_0p5_training` samples a per-element mask and scales the kept elements by
/// `1 / (1 - rate)`. `rate_0p5_inference` takes the pass-through path, which draws nothing.
/// `rate_0p0_training` takes the zero-rate early return, a legitimate configuration that skips
/// the sampler and leaves no mask behind. `rate_1p0_training` takes the drop-everything path,
/// where the forward output and the input gradient are both all zeros.
///
/// The 2 `noise_shape` cases pin the coarse mask. The noise shape is `[Some(1), None]`, which
/// resolves against the input `[2, 8]` to a mask of `[1, 8]`. The layer then takes 8 draws
/// instead of 16, and the batch axis of extent 1 broadcasts up. Both batch rows therefore keep
/// and drop the same 8 columns, and the recorded forward output shows 1 pattern of zeros 2
/// times. A mask that stayed at the input shape, or a broadcast that read the wrong axis, moves
/// that record.
///
/// `noise_shape_shared_batch_training` records the sampled path, and
/// `noise_shape_shared_batch_inference` records the inference path. The inference path returns
/// the input unchanged and draws nothing, so the noise shape must reach no recorded value
/// there. Both cases pin the seed through the builder of the layer, the same way every other
/// sampling case of this family does.
fn dropout_cases() -> Vec<GoldenCase> {
    /// Builds a Dropout layer with a pinned mask seed.
    fn build(rate: f32) -> Box<dyn Layer> {
        let mut layer = Dropout::new(rate)
            .expect("a rate between 0 and 1")
            .with_random_state(GOLDEN_SEED);
        layer
            .build(&Shape::known(&ELEMENTWISE_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    /// Builds a Dropout layer whose mask shares 1 draw over the whole batch axis.
    fn build_shared_batch() -> Box<dyn Layer> {
        let mut layer = Dropout::new(0.5)
            .expect("a rate between 0 and 1")
            .with_random_state(GOLDEN_SEED)
            .with_noise_shape(vec![Some(1), None])
            .expect("the noise shape has the rank of the input");
        layer
            .build(&Shape::known(&ELEMENTWISE_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    vec![
        // An inference pass takes the inference path, so it differs from a masked training
        // forward
        GoldenCase::new("rate_0p5_training", &ELEMENTWISE_SHAPE, || build(0.5))
            .with_inference_that_differs(),
        GoldenCase::new("rate_0p5_inference", &ELEMENTWISE_SHAPE, || build(0.5))
            .in_inference_mode(),
        // A rate of 0 makes the training forward the identity, so an inference pass agrees
        // with it
        GoldenCase::new("rate_0p0_training", &ELEMENTWISE_SHAPE, || build(0.0)),
        GoldenCase::new("rate_1p0_training", &ELEMENTWISE_SHAPE, || build(1.0))
            .with_inference_that_differs(),
        GoldenCase::new(
            "noise_shape_shared_batch_training",
            &ELEMENTWISE_SHAPE,
            build_shared_batch,
        )
        .with_inference_that_differs(),
        GoldenCase::new(
            "noise_shape_shared_batch_inference",
            &ELEMENTWISE_SHAPE,
            build_shared_batch,
        )
        .in_inference_mode(),
    ]
}

/// SpatialDropout1D cases: the per-channel sampled path, and the inference pass-through.
///
/// The mask holds 1 value per pair of a batch item and a channel. A dropped channel therefore
/// loses every position of that batch item. A mask read against any axis but the last would
/// show up as a different pattern of zeros in the recorded output.
fn spatial_dropout_1d_cases() -> Vec<GoldenCase> {
    /// Builds a SpatialDropout1D layer with a pinned mask seed.
    fn build() -> Box<dyn Layer> {
        let mut layer = SpatialDropout1D::new(0.5)
            .expect("a rate between 0 and 1")
            .with_random_state(GOLDEN_SEED);
        layer
            .build(&Shape::known(&SPATIAL_1D_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("rate_0p5_training", &SPATIAL_1D_SHAPE, build)
            .with_inference_that_differs(),
        GoldenCase::new("rate_0p5_inference", &SPATIAL_1D_SHAPE, build).in_inference_mode(),
    ]
}

/// SpatialDropout2D cases: the per-channel sampled path, and the inference pass-through.
fn spatial_dropout_2d_cases() -> Vec<GoldenCase> {
    /// Builds a SpatialDropout2D layer with a pinned mask seed.
    fn build() -> Box<dyn Layer> {
        let mut layer = SpatialDropout2D::new(0.5)
            .expect("a rate between 0 and 1")
            .with_random_state(GOLDEN_SEED);
        layer
            .build(&Shape::known(&SPATIAL_2D_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("rate_0p5_training", &SPATIAL_2D_SHAPE, build)
            .with_inference_that_differs(),
        GoldenCase::new("rate_0p5_inference", &SPATIAL_2D_SHAPE, build).in_inference_mode(),
    ]
}

/// SpatialDropout3D cases: the per-channel sampled path, and the inference pass-through.
fn spatial_dropout_3d_cases() -> Vec<GoldenCase> {
    /// Builds a SpatialDropout3D layer with a pinned mask seed.
    fn build() -> Box<dyn Layer> {
        let mut layer = SpatialDropout3D::new(0.5)
            .expect("a rate between 0 and 1")
            .with_random_state(GOLDEN_SEED);
        layer
            .build(&Shape::known(&SPATIAL_3D_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("rate_0p5_training", &SPATIAL_3D_SHAPE, build)
            .with_inference_that_differs(),
        GoldenCase::new("rate_0p5_inference", &SPATIAL_3D_SHAPE, build).in_inference_mode(),
    ]
}

// ---------------------------------------------------------------------------------------
// The noise injection layers
// ---------------------------------------------------------------------------------------

/// GaussianNoise cases: the additive noise path, the inference path, and the zero path.
///
/// The backward pass of this layer is a pass-through in every mode, because the noise does not
/// depend on the input. The 3 recorded input gradients therefore have to agree with each other.
/// A change that scaled any 1 of them would break that agreement.
fn gaussian_noise_cases() -> Vec<GoldenCase> {
    /// Builds a GaussianNoise layer with a pinned noise seed.
    fn build(stddev: f32) -> Box<dyn Layer> {
        let mut layer = GaussianNoise::new(stddev)
            .expect("a finite non-negative standard deviation")
            .with_random_state(GOLDEN_SEED);
        layer
            .build(&Shape::known(&ELEMENTWISE_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("stddev_0p5_training", &ELEMENTWISE_SHAPE, || build(0.5))
            .with_inference_that_differs(),
        GoldenCase::new("stddev_0p5_inference", &ELEMENTWISE_SHAPE, || build(0.5))
            .in_inference_mode(),
        // A standard deviation of 0 makes the training forward the identity
        GoldenCase::new("stddev_0p0_training", &ELEMENTWISE_SHAPE, || build(0.0)),
    ]
}

/// GaussianDropout cases: the multiplicative noise path, the inference path, and the zero path.
///
/// The forward pass multiplies the input by a draw from a normal distribution of mean 1 and
/// standard deviation `sqrt(rate / (1 - rate))`. The backward pass reuses that exact draw, so
/// the recorded input gradient pins the cache as well as the sampler.
fn gaussian_dropout_cases() -> Vec<GoldenCase> {
    /// Builds a GaussianDropout layer with a pinned noise seed.
    fn build(rate: f32) -> Box<dyn Layer> {
        let mut layer = GaussianDropout::new(rate)
            .expect("a rate in the range 0 through 1, with 1 excluded")
            .with_random_state(GOLDEN_SEED);
        layer
            .build(&Shape::known(&ELEMENTWISE_SHAPE))
            .expect("the layer accepts the shape of the case");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("rate_0p3_training", &ELEMENTWISE_SHAPE, || build(0.3))
            .with_inference_that_differs(),
        GoldenCase::new("rate_0p3_inference", &ELEMENTWISE_SHAPE, || build(0.3))
            .in_inference_mode(),
        // A rate of 0 makes the training forward the identity and writes no noise cache
        GoldenCase::new("rate_0p0_training", &ELEMENTWISE_SHAPE, || build(0.0)),
    ]
}

// ---------------------------------------------------------------------------------------
// Batch normalization
// ---------------------------------------------------------------------------------------

/// The momentum of every BatchNormalization case.
///
/// A value away from 0 and away from 1 keeps both terms of the running-statistic update
/// visible. A momentum of 1 would freeze the statistics, and a momentum of 0 would drop the
/// old value.
const BATCH_NORM_MOMENTUM: f32 = 0.9;

/// The epsilon of every BatchNormalization case.
const BATCH_NORM_EPSILON: f32 = 1e-5;

/// The rank-2 BatchNormalization shape: 2 batch rows of 4 channels.
const BATCH_NORM_SHAPE: [usize; 2] = [2, 4];

/// The spatial BatchNormalization shape: 2 batch items, 3 by 3 positions, 3 channels.
const BATCH_NORM_SPATIAL_SHAPE: [usize; 4] = [2, 3, 3, 3];

/// The rank-1 BatchNormalization shape, which gives the layer scalar parameters.
const BATCH_NORM_RANK_1_SHAPE: [usize; 1] = [4];

/// The shape that a running-statistic readout case reads, 1 row of 4 channels.
const BATCH_NORM_READOUT_SHAPE: [usize; 2] = [1, 4];

/// Builds a BatchNormalization layer with every parameter and statistic set from a formula.
///
/// The scale, the shift, the running mean, and the running variance each start the weight
/// formula at a different index. No 2 of the 4 tensors then hold the same values, so a swap of
/// any pair cannot pass unseen.
///
/// The running variance takes the weight formula shifted up by 1. The formula alone reaches
/// -0.975, and the inference path takes the square root of the variance. A negative variance
/// would turn that square root into a NaN.
///
/// # Parameters
///
/// - `input_shape` - Shape that the layer declares, batch axis first
///
/// # Returns
///
/// - `BatchNormalization` - A layer with all 4 tensors set, in training mode
fn batch_norm_layer(input_shape: &[usize]) -> BatchNormalization {
    // A rank-1 shape has no channel axis, and the layer then keeps scalar parameters
    let channels = if input_shape.len() > 1 {
        input_shape[input_shape.len() - 1]
    } else {
        1
    };
    let mut layer = BatchNormalization::new(BATCH_NORM_MOMENTUM, BATCH_NORM_EPSILON)
        .expect("a non-empty shape, a momentum in range, and a positive epsilon");
    layer
        .build(&Shape::known(input_shape))
        .expect("the layer accepts the shape of the case");
    layer
        .set_weights(
            golden_weights(&[channels]),
            golden_weights_from(&[channels], 5),
            golden_weights_from(&[channels], 11),
            golden_weights_from(&[channels], 17).mapv(|value| value + 1.0),
        )
        .expect("all 4 tensors have the per-channel shape");
    layer
}

/// Runs 1 training forward pass and returns the running statistics that it leaves behind.
///
/// The layer, the input, and the momentum are the ones of the `channels_4_training` case. The
/// returned pair is therefore exactly the state that that case leaves in the layer.
///
/// # Returns
///
/// - `(Tensor, Tensor)` - The running mean and the running variance, each 1 value per channel
fn batch_norm_running_statistics() -> (Tensor, Tensor) {
    // Only a training context updates the running statistics, so the pass below takes one
    let mut layer = batch_norm_layer(&BATCH_NORM_SHAPE);
    let mut ctx = Ctx::training();
    layer
        .forward(&golden_input(&BATCH_NORM_SHAPE), &mut ctx)
        .expect("the fixture input has the declared shape");
    // A forward pass proposes the new running statistics in the context, and never writes them
    // into the layer. The layer takes them here, which is the step a model runs after every
    // forward pass. The position is 0, because nothing set an owner
    layer.apply_state(&mut ctx.state_slot(0));
    (
        layer
            .weight("moving_mean")
            .expect("BatchNormalization names its running mean")
            .to_owned(),
        layer
            .weight("moving_variance")
            .expect("BatchNormalization names its running variance")
            .to_owned(),
    )
}

/// Builds a layer that returns 1 running statistic as its inference output.
///
/// A running statistic is internal state, and the harness records only what a layer returns. To
/// put the statistic in the data file, this builder makes a second BatchNormalization layer,
/// puts the statistic in `beta`, and sets `gamma` to 0. The inference output is then
/// `x_normalized * 0 + beta`, which is `beta` for every finite `x_normalized`. The recorded
/// forward tensor therefore holds the statistic itself, 1 value per channel.
///
/// The running mean of 0 and the running variance of 1 in this second layer only keep
/// `x_normalized` finite. Neither one reaches the output.
///
/// # Parameters
///
/// - `statistic` - The per-channel statistic to record, 1 value per channel
///
/// # Returns
///
/// - `Box<dyn Layer>` - A BatchNormalization layer whose inference output is `statistic`
fn batch_norm_statistic_readout(statistic: Tensor) -> Box<dyn Layer> {
    let channels = statistic.len();
    let mut layer = BatchNormalization::new(BATCH_NORM_MOMENTUM, BATCH_NORM_EPSILON)
        .expect("a momentum in range and a positive epsilon");
    layer
        .build(&Shape::known(&[1, channels]))
        .expect("the layer accepts the shape of the case");
    layer
        .set_weights(
            Tensor::zeros(IxDyn(&[channels])),
            statistic,
            Tensor::zeros(IxDyn(&[channels])),
            Tensor::ones(IxDyn(&[channels])),
        )
        .expect("all 4 tensors have the per-channel shape");
    Box::new(layer)
}

/// BatchNormalization cases: 3 training shapes, 1 inference case, and 2 statistic readouts.
///
/// `channels_4_training` runs the rank-2 path, where the per-channel folds see 2 rows.
/// `spatial_channels_3_training` runs the same folds over a rank-4 input, where the statistics
/// reduce over the batch axis and every spatial position. `rank_1_training` runs the other
/// arm, which a rank-1 input takes: the layer then holds scalar parameters and the folds give
/// way to the serial ndarray reduction. `channels_4_inference` runs the running-statistic path.
///
/// The 2 readout cases record the running mean and the running variance that the
/// `channels_4_training` forward pass leaves behind. A later stage that drops the momentum
/// update, or that updates the statistics in the wrong mode, changes those 2 tensors.
///
/// Only a training case declares parameter gradients. The inference backward pass returns at
/// once and writes no gradient, and `parameters` then yields nothing at all.
fn batch_normalization_cases() -> Vec<GoldenCase> {
    /// The parameter gradients, in the order that `parameters` returns them.
    const NAMES: [&str; 2] = ["gamma", "beta"];

    vec![
        GoldenCase::new("channels_4_training", &BATCH_NORM_SHAPE, || {
            Box::new(batch_norm_layer(&BATCH_NORM_SHAPE))
        })
        .with_parameter_grads(&NAMES)
        // A training forward reads the batch statistics, and an inference pass reads the
        // running ones
        .with_inference_that_differs(),
        GoldenCase::new(
            "spatial_channels_3_training",
            &BATCH_NORM_SPATIAL_SHAPE,
            || Box::new(batch_norm_layer(&BATCH_NORM_SPATIAL_SHAPE)),
        )
        .with_parameter_grads(&NAMES)
        .with_inference_that_differs(),
        GoldenCase::new("rank_1_training", &BATCH_NORM_RANK_1_SHAPE, || {
            Box::new(batch_norm_layer(&BATCH_NORM_RANK_1_SHAPE))
        })
        .with_parameter_grads(&NAMES)
        .with_inference_that_differs(),
        GoldenCase::new("channels_4_inference", &BATCH_NORM_SHAPE, || {
            Box::new(batch_norm_layer(&BATCH_NORM_SHAPE))
        })
        .in_inference_mode(),
        GoldenCase::new(
            "running_mean_after_1_step",
            &BATCH_NORM_READOUT_SHAPE,
            || batch_norm_statistic_readout(batch_norm_running_statistics().0),
        )
        .in_inference_mode(),
        GoldenCase::new(
            "running_var_after_1_step",
            &BATCH_NORM_READOUT_SHAPE,
            || batch_norm_statistic_readout(batch_norm_running_statistics().1),
        )
        .in_inference_mode(),
    ]
}
