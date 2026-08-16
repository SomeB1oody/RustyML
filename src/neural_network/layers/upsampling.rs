//! Upsampling layers that enlarge the spatial axes by a whole-number factor
//!
//! [`UpSampling1D`], [`UpSampling2D`], and [`UpSampling3D`] multiply the extent of each spatial
//! axis by its factor. The batch axis and the channel axis pass through unchanged. Under the
//! crate's channels-last layout a rank-4 input is `[batch, height, width, channels]`, so
//! [`UpSampling2D`] acts on axes 1 and 2
//!
//! No layer in the family holds a parameter. Each one caches the shape of the most recent
//! forward input, because the backward pass needs that shape to restore it
//!
//! The family is the decoder counterpart of the pooling family. A pooling layer divides a
//! spatial extent, and an upsampling layer multiplies it back. An autoencoder or a segmentation
//! decoder pairs each pooling stage with 1 upsampling stage of the same factor
//!
//! [`Factor2D`] and [`Factor3D`] carry the per-axis factors. Both constructors take
//! `impl Into<..>`, so a call site passes a plain integer or a plain tuple. [`UpSampling1D`]
//! takes a plain integer, because a rank-3 input has only 1 spatial axis
//!
//! Only [`UpSampling2D`] takes an [`Interpolation`]

/// Resize kernels shared by every upsampling layer
pub(crate) mod resize_engine;
/// 1D upsampling layer
pub mod up_sampling_1d;
/// 2D upsampling layer
pub mod up_sampling_2d;
/// 3D upsampling layer
pub mod up_sampling_3d;

pub use up_sampling_1d::UpSampling1D;
pub use up_sampling_2d::UpSampling2D;
pub use up_sampling_3d::UpSampling3D;

/// How [`UpSampling2D`] fills the new positions
///
/// [`Interpolation::Nearest`] repeats each input position, which is the default and the only
/// mode of the 1D and 3D layers. Every other mode resamples with a separable kernel, so a new
/// position takes a weighted sum of its neighbors
///
/// The weights of 1 output position always add up to 1. Near an edge, the kernel reaches past
/// the input, so the layer rescales the remaining weights to keep that sum
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::{Interpolation, UpSampling2D};
///
/// // Repeat each pixel into a 2x2 block
/// let blocky = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
///
/// // Blend each new pixel from its neighbors instead
/// let smooth = UpSampling2D::new(2, Interpolation::Bilinear).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Repeat each input position. Every new position copies the position it came from
    #[default]
    Nearest,
    /// Triangle kernel over the 2 nearest positions per axis
    Bilinear,
    /// Keys cubic kernel over the 4 nearest positions per axis, with `a = -0.5`
    Bicubic,
    /// Lanczos kernel of radius 3, over the 6 nearest positions per axis
    Lanczos3,
    /// Lanczos kernel of radius 5, over the 10 nearest positions per axis
    Lanczos5,
}

/// The upsampling factors of a rank-4 layer, as 1 whole number per spatial axis
///
/// [`UpSampling2D::new`] takes `impl Into<Factor2D>`. There are 2 forms:
///
/// - an integer `n` multiplies both spatial axes by `n`
/// - a pair `(height, width)` names the factor of each axis
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::{Interpolation, UpSampling2D};
///
/// // Twice the height and twice the width
/// let square = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
///
/// // Twice the height and 3 times the width
/// let uneven = UpSampling2D::new((2, 3), Interpolation::Nearest).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factor2D([usize; 2]);

impl From<usize> for Factor2D {
    fn from(factor: usize) -> Self {
        Factor2D([factor; 2])
    }
}

impl From<(usize, usize)> for Factor2D {
    fn from(per_axis: (usize, usize)) -> Self {
        Factor2D([per_axis.0, per_axis.1])
    }
}

/// The upsampling factors of a rank-5 layer, as 1 whole number per spatial axis
///
/// [`UpSampling3D::new`] takes `impl Into<Factor3D>`. There are 2 forms:
///
/// - an integer `n` multiplies all 3 spatial axes by `n`
/// - a triple `(dim1, dim2, dim3)` names the factor of each axis
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::UpSampling3D;
///
/// // Twice the extent on all 3 spatial axes
/// let cube = UpSampling3D::new(2).unwrap();
///
/// // A different factor on each axis
/// let uneven = UpSampling3D::new((1, 2, 4)).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factor3D([usize; 3]);

impl From<usize> for Factor3D {
    fn from(factor: usize) -> Self {
        Factor3D([factor; 3])
    }
}

impl From<(usize, usize, usize)> for Factor3D {
    fn from(per_axis: (usize, usize, usize)) -> Self {
        Factor3D([per_axis.0, per_axis.1, per_axis.2])
    }
}
