//! Weight initializers: the value that decides how a layer draws a new array
//!
//! [`Initializer`] is a closed enum, in the same shape as the
//! [`Activation`](crate::neural_network::layers::Activation) enum. It is `Copy`, it compares by
//! value, and it serializes, so a layer holds it by value and a checkpoint can record it. A
//! boxed trait object would give none of those 3 properties, and it would put a
//! `Send + Sync` bound on every layer that holds one.
//!
//! 14 functions of the module draw a weight array, and all 14 draw through this type. 15 of
//! their draws are Glorot uniform, 1 is a fixed uniform range, and the 3 recurrent layers draw
//! their recurrent kernels as square orthogonal matrices. The dropout and noise masks of a
//! forward pass are not weights, and they do not come from here.
//!
//! # The fan comes from the layer, never from the shape of the array
//!
//! [`Initializer::GlorotUniform`] scales its range by a fan pair. The pair reaches the draw as a
//! [`Fans`] value that the layer builds from its own configuration. No function of this module
//! reads the shape of the array to derive a fan.
//!
//! The transposed convolution is the reason. A plain convolution holds its kernel as
//! `(spatial axes, channels, filters)`, and a transposed convolution holds the same kernel as
//! `(spatial axes, filters, channels)`. The last 2 axes carry opposite roles in the 2 layouts. A
//! rule that reads the last 2 axes therefore gives a transposed layer a `fan_in` of
//! `filters * receptive_field` and a `fan_out` of `channels * receptive_field`. That is the swap
//! of the correct pair.
//!
//! Glorot hides the swap. Its range reads the sum of the 2 fans, and the sum is symmetric, so no
//! drawn value moves. The defect stays invisible until an initializer that reads 1 fan alone
//! arrives. A layer that names its 2 counts cannot make the mistake at all.
//!
//! # The draw order is part of the contract
//!
//! 4 layers thread 1 generator through more than 1 draw:
//!
//! - [`SeparableConv1D`](crate::neural_network::layers::SeparableConv1D) and
//!   [`SeparableConv2D`](crate::neural_network::layers::SeparableConv2D) draw the depthwise
//!   kernel first and the pointwise kernel second.
//! - [`SimpleRNN`](crate::neural_network::layers::SimpleRNN) draws the input kernel first and
//!   the orthogonal recurrent kernel second.
//! - [`FusedGates`](crate::neural_network::layers::recurrent::gate::FusedGates), which serves
//!   [`LSTM`](crate::neural_network::layers::LSTM) and
//!   [`GRU`](crate::neural_network::layers::GRU), draws the fused input kernel first and then 1
//!   orthogonal block per gate, in gate order.
//!
//! 1 generator gives 1 stream, and each draw takes the next values of that stream. A second
//! generator, or a different order, therefore changes every value from the second draw onward.
//! The order is part of the contract of those 4 layers. Keep the draws of 1 layer in 1 function,
//! against 1 generator.

use crate::{Deserialize, Serialize};
use ndarray::{Array, Array2, Dimension, ShapeBuilder};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Half-width of the uniform range that an orthogonal draw starts from
///
/// Gram-Schmidt normalizes every column, so the starting range only sets an arbitrary
/// orientation. Any finite half-width gives an orthonormal result
const ORTHOGONAL_START_LIMIT: f32 = 1.0;

/// Smallest column norm that an orthogonal draw keeps
///
/// A column that shrinks below this value after the projections carries no reliable direction.
/// The draw replaces it with a standard basis vector
const ORTHOGONAL_EPSILON: f32 = 1e-8;

/// The 2 fans of a weight array, as the layer that owns the array reports them
///
/// `fan_in` counts the input elements that reach 1 output element. `fan_out` counts the output
/// elements that 1 input element reaches. A convolution multiplies both counts by the receptive
/// field, because 1 kernel holds 1 weight per tap.
///
/// The layer builds the pair from its own configuration. Nothing derives the pair from the shape
/// of the array. See the module documentation for the transposed-convolution layout that makes a
/// shape-derived rule wrong.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Fans;
///
/// // A convolution of 2 input channels, 4 filters, and 3 taps
/// assert_eq!(Fans::conv(2, 4, 3), Fans::new(6, 12));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fans {
    /// Input elements that reach 1 output element
    pub fan_in: usize,
    /// Output elements that 1 input element reaches
    pub fan_out: usize,
}

impl Fans {
    /// The pair that an initializer which reads no fan receives
    ///
    /// [`Initializer::Uniform`] and [`Initializer::Orthogonal`] read no fan, so a caller of
    /// either passes this value. [`Initializer::GlorotUniform`] divides by the sum of the 2
    /// fans, so it must never receive this value
    pub const NONE: Self = Self {
        fan_in: 0,
        fan_out: 0,
    };

    /// A fan pair from 2 counts that the caller already holds
    ///
    /// # Parameters
    ///
    /// - `fan_in` - Input elements that reach 1 output element
    /// - `fan_out` - Output elements that 1 input element reaches
    ///
    /// # Returns
    ///
    /// - `Fans` - The 2 given counts
    #[inline]
    pub const fn new(fan_in: usize, fan_out: usize) -> Self {
        Self { fan_in, fan_out }
    }

    /// The fan pair of a convolution kernel, plain or transposed
    ///
    /// Every convolution kernel of the crate holds 1 weight for each input channel, output
    /// channel, and tap. `fan_in` is therefore `channels * receptive_field`, and `fan_out` is
    /// `filters * receptive_field`. The rule holds for the plain layout and for the transposed
    /// layout alike, because the 2 counts arrive by name and not by axis position.
    ///
    /// A depthwise kernel passes its depth multiplier as `filters`. The `fan_in` then counts
    /// every input channel, although a depthwise unit reads only 1 of them. This follows the
    /// convolution rule rather than a depthwise rule, which is what Keras does, and it makes the
    /// range narrower by about the square root of the channel count
    ///
    /// # Parameters
    ///
    /// - `channels` - Input channels of the convolution
    /// - `filters` - Output channels of the convolution, or the depth multiplier of a depthwise
    ///   kernel
    /// - `receptive_field` - Taps of 1 kernel, which is the product of the kernel extents
    ///
    /// # Returns
    ///
    /// - `Fans` - `channels * receptive_field` and `filters * receptive_field`
    #[inline]
    pub const fn conv(channels: usize, filters: usize, receptive_field: usize) -> Self {
        Self {
            fan_in: channels * receptive_field,
            fan_out: filters * receptive_field,
        }
    }
}

/// How a layer draws the starting values of a weight array
///
/// The set is closed, and it holds the 3 rules that this crate draws with. A layer names 1 of
/// them and supplies the shape, the [`Fans`], and the generator.
///
/// # Notes
///
/// Every variant starts from 1 uniform range that is symmetric about 0. The variants differ in
/// the half-width of that range, and [`Initializer::Orthogonal`] adds a second step.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::{Fans, Initializer};
///
/// // A dense kernel of 4 inputs and 4 outputs draws over -w to w, with w = sqrt(6 / 8)
/// let width = Initializer::GlorotUniform.limit(Fans::new(4, 4));
/// assert_eq!(width, (6.0_f32 / 8.0).sqrt());
///
/// // A fixed range reads no fan
/// assert_eq!(Initializer::Uniform { limit: 0.05 }.limit(Fans::NONE), 0.05);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Initializer {
    /// Glorot (Xavier) uniform, over `-sqrt(6 / (fan_in + fan_out))` to the same value positive
    ///
    /// This is the rule of every kernel of the crate except the embedding table and the
    /// recurrent kernels. The 2 fans come from the layer. See the module documentation
    GlorotUniform,
    /// A uniform range of a fixed half-width, which reads no fan
    ///
    /// The embedding table draws this way. An embedding reads exactly 1 row per index, so its
    /// fan-in is 1 and a fan-scaled range has nothing to scale
    Uniform {
        /// Half-width of the range. The draw covers `-limit` to `limit`
        limit: f32,
    },
    /// A square matrix whose columns are mutually orthonormal
    ///
    /// The recurrent kernels draw this way. Orthonormal columns keep the hidden-state
    /// transition norm-preserving, which reduces vanishing and exploding gradients.
    /// [`Initializer::draw_orthogonal`] is the entry point of this variant
    Orthogonal,
}

impl Initializer {
    /// The half-width of the uniform range that this initializer draws its first values over
    ///
    /// # Parameters
    ///
    /// - `fans` - The fan pair of the array. [`Initializer::GlorotUniform`] reads it, and the
    ///   other 2 variants ignore it and accept [`Fans::NONE`]
    ///
    /// # Returns
    ///
    /// - `f32` - Half-width of the range, so the draw covers the negative of this value to this
    ///   value
    #[inline]
    pub fn limit(self, fans: Fans) -> f32 {
        match self {
            Initializer::GlorotUniform => (6.0 / (fans.fan_in + fans.fan_out) as f32).sqrt(),
            Initializer::Uniform { limit } => limit,
            Initializer::Orthogonal => ORTHOGONAL_START_LIMIT,
        }
    }

    /// Draws an array of `shape`, 1 element at a time, over the range of this initializer
    ///
    /// The elements come from `rng` in the row-major order of `shape`. The draw advances `rng` by
    /// 1 value per element, so a caller that draws twice from 1 generator gets 2 different
    /// results and gets them in a fixed order. See the module documentation
    ///
    /// # Parameters
    ///
    /// - `shape` - Shape of the array, at the rank the layer holds it
    /// - `fans` - The fan pair of the array, or [`Fans::NONE`] for an initializer that reads no
    ///   fan
    /// - `rng` - Generator of the owning layer
    ///
    /// # Returns
    ///
    /// - `Array<f32, D>` - A new array of `shape`, in row-major memory order
    ///
    /// # Panics
    ///
    /// - If the half-width from [`Initializer::limit`] is not a positive finite number.
    ///   [`Initializer::GlorotUniform`] gives an infinite half-width for [`Fans::NONE`]
    pub fn draw<Sh, D>(self, shape: Sh, fans: Fans, rng: &mut StdRng) -> Array<f32, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        D: Dimension,
    {
        let limit = self.limit(fans);
        Array::random_using(shape, Uniform::new(-limit, limit).unwrap(), rng)
    }

    /// Draws a square matrix whose columns are mutually orthonormal
    ///
    /// The matrix starts as a `(size, size)` draw of this initializer. Gram-Schmidt then
    /// subtracts from each column its projection on every earlier column, and normalizes it. A
    /// column that collapses becomes a standard basis vector. Independent normalization alone
    /// would give unit-norm columns that are not mutually orthogonal.
    ///
    /// [`Initializer::Orthogonal`] is the variant that names this draw. Another variant only
    /// changes the starting range, which sets an arbitrary orientation and nothing else
    ///
    /// # Parameters
    ///
    /// - `size` - Extent of both axes of the matrix
    /// - `fans` - The fan pair of the starting draw, or [`Fans::NONE`] for
    ///   [`Initializer::Orthogonal`]
    /// - `rng` - Generator of the owning layer
    ///
    /// # Returns
    ///
    /// - `Array2<f32>` - A `(size, size)` matrix with mutually orthonormal columns
    ///
    /// # Panics
    ///
    /// - If the starting draw panics. See [`Initializer::draw`]
    pub fn draw_orthogonal(self, size: usize, fans: Fans, rng: &mut StdRng) -> Array2<f32> {
        let mut matrix: Array2<f32> = self.draw((size, size), fans, rng);

        for i in 0..size {
            for j in 0..i {
                let mut projection = 0.0;
                for k in 0..size {
                    projection += matrix[[k, i]] * matrix[[k, j]];
                }
                for k in 0..size {
                    matrix[[k, i]] -= projection * matrix[[k, j]];
                }
            }

            let mut norm = 0.0f32;
            for k in 0..size {
                norm += matrix[[k, i]] * matrix[[k, i]];
            }
            norm = norm.sqrt();

            if norm > ORTHOGONAL_EPSILON {
                for k in 0..size {
                    matrix[[k, i]] /= norm;
                }
            } else {
                for k in 0..size {
                    matrix[[k, i]] = if k == i { 1.0 } else { 0.0 };
                }
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::*;
    use crate::neural_network::traits::{LayerBase, UnaryLayer};
    use crate::random::make_rng;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array3, Array4, Array5, ArrayD, IxDyn};
    use ndarray_rand::rand::SeedableRng;

    /// Seed of every layer that a layout case builds
    const SEED: u64 = 12345;

    /// Builds 1 layer for the given input shape, and gives it back
    ///
    /// A constructor draws nothing, so every case here builds before it reads an array
    fn built<L: UnaryLayer>(mut layer: L, input_shape: &[usize]) -> L {
        layer
            .build(&crate::neural_network::Shape::known(input_shape))
            .expect("the layer accepts the shape of the case");
        layer
    }

    /// Reads 1 named array of a layer and returns it as a dynamic-rank copy
    fn weight_of(layer: &dyn LayerBase, name: &str) -> ArrayD<f32> {
        let weights = layer.weights();
        let found = weights
            .iter()
            .find(|weight| weight.name == name)
            .unwrap_or_else(|| panic!("the layer holds no array named {name}"));
        let mut owned = ArrayD::<f32>::zeros(IxDyn(found.value.shape()));
        owned.assign(&found.value);
        owned
    }

    /// Asserts that a layer holds `name` as the bit-exact Glorot draw of `shape` at `fans`
    ///
    /// The comparison redraws the array from a fresh generator of the same seed. It therefore
    /// pins the shape, the fan sum, the distribution, and the position of the draw in the
    /// stream of the layer. It cannot see a swap of `fan_in` and `fan_out`, because the Glorot
    /// range reads only their sum. The `fans_*` tests below pin the 2 counts apart
    fn assert_glorot<Sh, D>(layer: &dyn LayerBase, name: &str, shape: Sh, fans: Fans)
    where
        Sh: ShapeBuilder<Dim = D>,
        D: Dimension,
    {
        let mut rng = make_rng(Some(SEED));
        let expected: Array<f32, D> = Initializer::GlorotUniform.draw(shape, fans, &mut rng);
        assert_eq!(weight_of(layer, name), expected.into_dyn());
    }

    // The fan of each distinct kernel layout
    //
    // The layouts come from the 14 weight-drawing functions of the crate. Each test states the
    // shape the layer stores and the 2 counts the layer reports. Every case gives the 2 counts
    // different values, so a swap of the pair changes the asserted numbers

    /// Layout 1, the dense kernel `(input_dim, units)`, has the 2 dimensions as its 2 fans
    #[test]
    fn fans_of_the_dense_layout() {
        assert_eq!(
            Fans::new(3, 5),
            Fans {
                fan_in: 3,
                fan_out: 5
            }
        );
    }

    /// Layouts 2, 3, and 4, the plain convolution kernels, count the taps into both fans
    #[test]
    fn fans_of_the_plain_convolution_layouts() {
        // 1D `(k, channels, filters)` with k = 3
        assert_eq!(Fans::conv(2, 4, 3), Fans::new(6, 12));
        // 2D `(kh, kw, channels, filters)` with kh = 2 and kw = 3
        assert_eq!(Fans::conv(2, 4, 2 * 3), Fans::new(12, 24));
        // 3D `(kd, kh, kw, channels, filters)` with kd = 2, kh = 3, and kw = 4
        assert_eq!(Fans::conv(2, 4, 2 * 3 * 4), Fans::new(48, 96));
    }

    /// Layout 5, the transposed kernels, keeps the channel count in `fan_in`
    ///
    /// The transposed kernel stores the filter axis before the channel axis, so its last 2 axes
    /// are the reverse of the plain layout. The fans do not follow the axes. They follow the 2
    /// counts, exactly as the plain layout does
    #[test]
    fn fans_of_the_transposed_convolution_layout() {
        let channels = 2;
        let filters = 4;
        let receptive_field = 3;

        let transposed = Fans::conv(channels, filters, receptive_field);
        assert_eq!(transposed, Fans::new(6, 12));

        // A rule that read the last 2 axes of the stored kernel would give this pair instead
        let from_the_axes = Fans::conv(filters, channels, receptive_field);
        assert_eq!(from_the_axes, Fans::new(12, 6));
        assert_ne!(transposed, from_the_axes);

        // Glorot reads the sum alone, so the wrong pair draws the same values. Only this test
        // separates the 2 counts
        assert_eq!(
            Initializer::GlorotUniform.limit(transposed),
            Initializer::GlorotUniform.limit(from_the_axes)
        );
    }

    /// Layouts 6 and 7, the depthwise kernels, pass the depth multiplier as the filter count
    #[test]
    fn fans_of_the_depthwise_layouts() {
        // 1D `(k, channels, depth_multiplier)` with k = 3
        assert_eq!(Fans::conv(4, 2, 3), Fans::new(12, 6));
        // 2D `(kh, kw, channels, depth_multiplier)` with kh = 2 and kw = 3
        assert_eq!(Fans::conv(4, 2, 2 * 3), Fans::new(24, 12));
    }

    /// Layout 8, the separable pointwise kernel, holds a receptive field of 1
    ///
    /// The stored width is already `channels * depth_multiplier`, so `fan_in` is that product
    /// and `fan_out` is the filter count with no tap factor
    #[test]
    fn fans_of_the_separable_pointwise_layout() {
        assert_eq!(Fans::new(3 * 2, 5), Fans::new(6, 5));
        assert_eq!(Fans::conv(3 * 2, 5, 1), Fans::new(6, 5));
    }

    /// Layouts 9 and 11, the embedding table and the recurrent kernel, read no fan
    #[test]
    fn the_fan_free_layouts_ignore_the_pair() {
        assert_eq!(Initializer::Uniform { limit: 0.05 }.limit(Fans::NONE), 0.05);
        assert_eq!(
            Initializer::Uniform { limit: 0.05 }.limit(Fans::new(7, 9)),
            0.05
        );
        assert_eq!(
            Initializer::Orthogonal.limit(Fans::NONE),
            ORTHOGONAL_START_LIMIT
        );
    }

    // Each layer draws its kernel at the shape and the fan pair of its layout

    /// Layout 1: `Dense` stores `(input_dim, units)` and reports `(input_dim, units)`
    #[test]
    fn dense_draws_the_dense_layout() {
        let layer = built(
            Dense::new(5, Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[2, 3],
        );
        assert_glorot(&layer, "kernel", (3, 5), Fans::new(3, 5));
    }

    /// Layout 2: `Conv1D` stores `(k, channels, filters)`
    #[test]
    fn conv_1d_draws_the_plain_layout() {
        let layer = built(
            Conv1D::new(4, 3, 1, Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 10, 2],
        );
        assert_glorot(&layer, "kernel", (3, 2, 4), Fans::conv(2, 4, 3));
    }

    /// Layout 3: `Conv2D` stores `(kh, kw, channels, filters)`
    #[test]
    fn conv_2d_draws_the_plain_layout() {
        let layer = built(
            Conv2D::new(4, (2, 3), (1, 1), Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 8, 8, 2],
        );
        assert_glorot(&layer, "kernel", (2, 3, 2, 4), Fans::conv(2, 4, 2 * 3));
    }

    /// Layout 4: `Conv3D` stores `(kd, kh, kw, channels, filters)`
    #[test]
    fn conv_3d_draws_the_plain_layout() {
        let layer = built(
            Conv3D::new(4, (2, 3, 4), (1, 1, 1), Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 8, 8, 8, 2],
        );
        assert_glorot(
            &layer,
            "kernel",
            (2, 3, 4, 2, 4),
            Fans::conv(2, 4, 2 * 3 * 4),
        );
    }

    /// Layout 5: `Conv1DTranspose` stores `(k, filters, channels)` and keeps the plain fan pair
    #[test]
    fn conv_1d_transpose_draws_the_transposed_layout() {
        let layer = built(
            Conv1DTranspose::new(4, 3, 1, Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 10, 2],
        );
        let kernel = weight_of(&layer, "kernel");
        assert_eq!(kernel.shape(), &[3, 4, 2]);
        assert_glorot(&layer, "kernel", (3, 4, 2), Fans::conv(2, 4, 3));
    }

    /// Layout 5: `Conv2DTranspose` stores `(kh, kw, filters, channels)`
    #[test]
    fn conv_2d_transpose_draws_the_transposed_layout() {
        let layer = built(
            Conv2DTranspose::new(4, (2, 3), (1, 1), Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 8, 8, 2],
        );
        let kernel = weight_of(&layer, "kernel");
        assert_eq!(kernel.shape(), &[2, 3, 4, 2]);
        assert_glorot(&layer, "kernel", (2, 3, 4, 2), Fans::conv(2, 4, 2 * 3));
    }

    /// Layout 5: `Conv3DTranspose` stores `(kd, kh, kw, filters, channels)`
    #[test]
    fn conv_3d_transpose_draws_the_transposed_layout() {
        let layer = built(
            Conv3DTranspose::new(4, (2, 3, 4), (1, 1, 1), Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 8, 8, 8, 2],
        );
        let kernel = weight_of(&layer, "kernel");
        assert_eq!(kernel.shape(), &[2, 3, 4, 4, 2]);
        assert_glorot(
            &layer,
            "kernel",
            (2, 3, 4, 4, 2),
            Fans::conv(2, 4, 2 * 3 * 4),
        );
    }

    /// Layout 6: `DepthwiseConv1D` stores `(k, channels, depth_multiplier)`
    #[test]
    fn depthwise_conv_1d_draws_the_depthwise_layout() {
        let layer = built(
            DepthwiseConv1D::new(3, 1, Activation::Linear)
                .unwrap()
                .with_depth_multiplier(2)
                .unwrap()
                .with_random_state(SEED),
            &[1, 10, 4],
        );
        assert_glorot(&layer, "kernel", (3, 4, 2), Fans::conv(4, 2, 3));
    }

    /// Layout 6: `DepthwiseConv2D` stores `(kh, kw, channels, depth_multiplier)`
    #[test]
    fn depthwise_conv_2d_draws_the_depthwise_layout() {
        let layer = built(
            DepthwiseConv2D::new((2, 3), (1, 1), Activation::Linear)
                .unwrap()
                .with_depth_multiplier(2)
                .unwrap()
                .with_random_state(SEED),
            &[1, 8, 8, 4],
        );
        assert_glorot(&layer, "kernel", (2, 3, 4, 2), Fans::conv(4, 2, 2 * 3));
    }

    /// Layouts 7 and 8: `SeparableConv1D` draws the depthwise kernel and then the pointwise one
    #[test]
    fn separable_conv_1d_draws_both_layouts_from_1_stream() {
        let layer = built(
            SeparableConv1D::new(5, 3, 1, 2, Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 10, 4],
        );

        let mut rng = make_rng(Some(SEED));
        let depthwise: Array3<f32> =
            Initializer::GlorotUniform.draw((3, 4, 2), Fans::conv(4, 2, 3), &mut rng);
        let pointwise: Array3<f32> =
            Initializer::GlorotUniform.draw((1, 4 * 2, 5), Fans::new(4 * 2, 5), &mut rng);

        assert_eq!(weight_of(&layer, "depthwise_kernel"), depthwise.into_dyn());
        assert_eq!(weight_of(&layer, "pointwise_kernel"), pointwise.into_dyn());
    }

    /// Layouts 7 and 8: `SeparableConv2D` draws the depthwise kernel and then the pointwise one
    #[test]
    fn separable_conv_2d_draws_both_layouts_from_1_stream() {
        let layer = built(
            SeparableConv2D::new(5, (2, 3), (1, 1), 2, Activation::Linear)
                .unwrap()
                .with_random_state(SEED),
            &[1, 8, 8, 4],
        );

        let mut rng = make_rng(Some(SEED));
        let depthwise: Array4<f32> =
            Initializer::GlorotUniform.draw((2, 3, 4, 2), Fans::conv(4, 2, 2 * 3), &mut rng);
        let pointwise: Array4<f32> =
            Initializer::GlorotUniform.draw((1, 1, 4 * 2, 5), Fans::new(4 * 2, 5), &mut rng);

        assert_eq!(weight_of(&layer, "depthwise_kernel"), depthwise.into_dyn());
        assert_eq!(weight_of(&layer, "pointwise_kernel"), pointwise.into_dyn());
    }

    /// Layout 9: `Embedding` draws a fixed range and reads no fan
    #[test]
    fn embedding_draws_the_fixed_range() {
        let layer = built(
            Embedding::new(7, 3).unwrap().with_random_state(SEED),
            &[2, 4],
        );
        let mut rng = make_rng(Some(SEED));
        let expected: Array2<f32> =
            Initializer::Uniform { limit: 0.05 }.draw((7, 3), Fans::NONE, &mut rng);
        assert_eq!(weight_of(&layer, "embeddings"), expected.into_dyn());
    }

    /// Layouts 10 and 11: `SimpleRNN` draws the input kernel and then the recurrent kernel
    #[test]
    fn simple_rnn_draws_both_layouts_from_1_stream() {
        let layer = built(
            SimpleRNN::new(5, Activation::Tanh)
                .unwrap()
                .with_random_state(SEED),
            &[2, 6, 3],
        );

        let mut rng = make_rng(Some(SEED));
        let kernel: Array2<f32> =
            Initializer::GlorotUniform.draw((3, 5), Fans::new(3, 5), &mut rng);
        let recurrent = Initializer::Orthogonal.draw_orthogonal(5, Fans::NONE, &mut rng);

        assert_eq!(weight_of(&layer, "kernel"), kernel.into_dyn());
        assert_eq!(
            weight_of(&layer, "recurrent_kernel"),
            recurrent.clone().into_dyn()
        );
    }

    /// Layouts 10 and 11: `LSTM` draws the fused kernel and then 1 orthogonal block per gate
    ///
    /// The Glorot range of the fused kernel reads the per-gate fan, `input_dim + units`, and not
    /// the fused width
    #[test]
    fn lstm_draws_the_fused_layouts_from_1_stream() {
        let input_dim = 3;
        let units = 4;
        let gates = 4;
        let layer = built(
            LSTM::new(units, Activation::Tanh)
                .unwrap()
                .with_random_state(SEED),
            &[2, 6, input_dim],
        );

        let mut rng = make_rng(Some(SEED));
        let kernel: Array2<f32> = Initializer::GlorotUniform.draw(
            (input_dim, gates * units),
            Fans::new(input_dim, units),
            &mut rng,
        );
        assert_eq!(weight_of(&layer, "kernel"), kernel.into_dyn());

        let recurrent = weight_of(&layer, "recurrent_kernel");
        for gate in 0..gates {
            let block = Initializer::Orthogonal.draw_orthogonal(units, Fans::NONE, &mut rng);
            for row in 0..units {
                for column in 0..units {
                    assert_eq!(
                        recurrent[[row, gate * units + column]],
                        block[[row, column]]
                    );
                }
            }
        }
    }

    /// Layouts 10 and 11: `GRU` draws the fused kernel and then 1 orthogonal block per gate
    #[test]
    fn gru_draws_the_fused_layouts_from_1_stream() {
        let input_dim = 3;
        let units = 4;
        let gates = 3;
        let layer = built(
            GRU::new(units, Activation::Tanh)
                .unwrap()
                .with_random_state(SEED),
            &[2, 6, input_dim],
        );

        let mut rng = make_rng(Some(SEED));
        let kernel: Array2<f32> = Initializer::GlorotUniform.draw(
            (input_dim, gates * units),
            Fans::new(input_dim, units),
            &mut rng,
        );
        assert_eq!(weight_of(&layer, "kernel"), kernel.into_dyn());

        let recurrent = weight_of(&layer, "recurrent_kernel");
        for gate in 0..gates {
            let block = Initializer::Orthogonal.draw_orthogonal(units, Fans::NONE, &mut rng);
            for row in 0..units {
                for column in 0..units {
                    assert_eq!(
                        recurrent[[row, gate * units + column]],
                        block[[row, column]]
                    );
                }
            }
        }
    }

    // The draw itself

    /// The Glorot range is the square root of 6 over the sum of the 2 fans
    #[test]
    fn the_glorot_range_reads_the_sum_of_the_fans() {
        assert_abs_diff_eq!(
            Initializer::GlorotUniform.limit(Fans::new(2, 6)),
            (6.0_f32 / 8.0).sqrt(),
            epsilon = 0.0
        );
    }

    /// Every drawn value stays inside the range of the initializer
    #[test]
    fn a_draw_stays_inside_the_range() {
        let fans = Fans::new(4, 6);
        let limit = Initializer::GlorotUniform.limit(fans);
        let mut rng = StdRng::seed_from_u64(0);
        let drawn: Array5<f32> = Initializer::GlorotUniform.draw((2, 3, 4, 2, 3), fans, &mut rng);
        assert_eq!(drawn.shape(), &[2, 3, 4, 2, 3]);
        assert!(drawn.iter().all(|value| *value >= -limit && *value < limit));
    }

    /// 2 draws from 1 generator give different arrays, and the pair repeats from the same seed
    #[test]
    fn the_draw_order_of_1_generator_is_fixed() {
        let fans = Fans::new(3, 3);
        let mut rng = StdRng::seed_from_u64(7);
        let first: Array2<f32> = Initializer::GlorotUniform.draw((2, 2), fans, &mut rng);
        let second: Array2<f32> = Initializer::GlorotUniform.draw((2, 2), fans, &mut rng);
        assert_ne!(first, second);

        let mut again = StdRng::seed_from_u64(7);
        let first_again: Array2<f32> = Initializer::GlorotUniform.draw((2, 2), fans, &mut again);
        let second_again: Array2<f32> = Initializer::GlorotUniform.draw((2, 2), fans, &mut again);
        assert_eq!(first, first_again);
        assert_eq!(second, second_again);
    }

    /// For size 3, the transpose of an orthogonal draw times the draw is the identity
    #[test]
    fn an_orthogonal_draw_has_orthonormal_columns() {
        let matrix =
            Initializer::Orthogonal.draw_orthogonal(3, Fans::NONE, &mut StdRng::seed_from_u64(0));
        let gram = matrix.t().dot(&matrix);
        for row in 0..3 {
            for column in 0..3 {
                let expected = if row == column { 1.0_f32 } else { 0.0_f32 };
                assert_abs_diff_eq!(gram[[row, column]], expected, epsilon = 1e-5);
            }
        }
    }

    /// For size 1, the single entry of an orthogonal draw has absolute value 1
    #[test]
    fn an_orthogonal_draw_of_size_1_is_a_sign() {
        let matrix =
            Initializer::Orthogonal.draw_orthogonal(1, Fans::NONE, &mut StdRng::seed_from_u64(0));
        assert_eq!(matrix.shape(), &[1, 1]);
        assert_abs_diff_eq!(matrix[[0, 0]].abs(), 1.0_f32, epsilon = 1e-6);
    }
}
