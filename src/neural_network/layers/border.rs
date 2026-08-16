//! Border layers that add or remove positions at the ends of the spatial axes
//!
//! The family has 2 halves. [`ZeroPadding1D`], [`ZeroPadding2D`], and [`ZeroPadding3D`] add
//! zero positions. [`Cropping1D`], [`Cropping2D`], and [`Cropping3D`] remove positions. Each
//! half is the backward pass of the other half, so both halves share 1 pair of kernels
//!
//! Every layer in the family leaves the batch axis and the channel axis unchanged. Only the
//! axes between them change extent. Under the crate's channels-last layout a rank-4 input is
//! `[batch, height, width, channels]`, so [`ZeroPadding2D`] and [`Cropping2D`] act on axes 1
//! and 2
//!
//! No layer in the family holds a parameter. Each one caches the shape of the most recent
//! forward input, because the backward pass needs that shape to restore it
//!
//! [`Border1D`], [`Border2D`], and [`Border3D`] carry the per-axis amounts. Every constructor
//! takes `impl Into<..>`, so a call site passes a plain integer or a plain tuple

/// 1D cropping layer
pub mod cropping_1d;
/// 2D cropping layer
pub mod cropping_2d;
/// 3D cropping layer
pub mod cropping_3d;
/// Pad and crop kernels shared by every border layer
mod pad_crop_engine;
/// 1D zero-padding layer
pub mod zero_padding_1d;
/// 2D zero-padding layer
pub mod zero_padding_2d;
/// 3D zero-padding layer
pub mod zero_padding_3d;

pub use cropping_1d::Cropping1D;
pub use cropping_2d::Cropping2D;
pub use cropping_3d::Cropping3D;
pub use zero_padding_1d::ZeroPadding1D;
pub use zero_padding_2d::ZeroPadding2D;
pub use zero_padding_3d::ZeroPadding3D;

/// The border amount of a rank-3 border layer, as 1 `(before, after)` pair
///
/// [`ZeroPadding1D::new`] and [`Cropping1D::new`] take `impl Into<Border1D>`. Pass an integer
/// for an equal amount at both ends of the axis. Pass a `(before, after)` pair for an unequal
/// one
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::ZeroPadding1D;
///
/// // 2 zero steps before the first step and 2 after the last
/// let even = ZeroPadding1D::new(2);
///
/// // 1 zero step before the first step and 3 after the last
/// let uneven = ZeroPadding1D::new((1, 3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Border1D([(usize, usize); 1]);

impl From<usize> for Border1D {
    fn from(amount: usize) -> Self {
        Border1D([(amount, amount)])
    }
}

impl From<(usize, usize)> for Border1D {
    fn from(ends: (usize, usize)) -> Self {
        Border1D([ends])
    }
}

/// The border amounts of a rank-4 border layer, as 1 `(before, after)` pair per spatial axis
///
/// [`ZeroPadding2D::new`] and [`Cropping2D::new`] take `impl Into<Border2D>`. There are 3
/// forms:
///
/// - an integer `n` puts `n` at all 4 ends
/// - a pair `(height, width)` puts an equal amount at both ends of each axis
/// - a pair of pairs `((top, bottom), (left, right))` names all 4 ends
///
/// The 2-integer form gives 1 amount per axis. It does not give the 2 ends of 1 axis
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::ZeroPadding2D;
///
/// // 1 zero row and 1 zero column at every edge
/// let ring = ZeroPadding2D::new(1);
///
/// // 2 zero rows at the top and bottom, 3 zero columns at the left and right
/// let per_axis = ZeroPadding2D::new((2, 3));
///
/// // 1 zero row at the top only, and 2 zero columns at the right only
/// let named = ZeroPadding2D::new(((1, 0), (0, 2)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Border2D([(usize, usize); 2]);

impl From<usize> for Border2D {
    fn from(amount: usize) -> Self {
        Border2D([(amount, amount); 2])
    }
}

impl From<(usize, usize)> for Border2D {
    fn from(per_axis: (usize, usize)) -> Self {
        Border2D([(per_axis.0, per_axis.0), (per_axis.1, per_axis.1)])
    }
}

impl From<((usize, usize), (usize, usize))> for Border2D {
    fn from(ends: ((usize, usize), (usize, usize))) -> Self {
        Border2D([ends.0, ends.1])
    }
}

/// The border amounts of a rank-5 border layer, as 1 `(before, after)` pair per spatial axis
///
/// [`ZeroPadding3D::new`] and [`Cropping3D::new`] take `impl Into<Border3D>`. There are 3
/// forms:
///
/// - an integer `n` puts `n` at all 6 ends
/// - a triple `(dim1, dim2, dim3)` puts an equal amount at both ends of each axis
/// - a triple of pairs names all 6 ends
///
/// The 3-integer form gives 1 amount per axis. It does not give the ends of 1 axis
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::ZeroPadding3D;
///
/// // 1 zero plane at every one of the 6 faces
/// let shell = ZeroPadding3D::new(1);
///
/// // 1 plane on the first axis, 2 on the second, and none on the third
/// let per_axis = ZeroPadding3D::new((1, 2, 0));
///
/// // Every end named on its own
/// let named = ZeroPadding3D::new(((1, 0), (0, 2), (1, 1)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Border3D([(usize, usize); 3]);

impl From<usize> for Border3D {
    fn from(amount: usize) -> Self {
        Border3D([(amount, amount); 3])
    }
}

impl From<(usize, usize, usize)> for Border3D {
    fn from(per_axis: (usize, usize, usize)) -> Self {
        Border3D([
            (per_axis.0, per_axis.0),
            (per_axis.1, per_axis.1),
            (per_axis.2, per_axis.2),
        ])
    }
}

impl From<((usize, usize), (usize, usize), (usize, usize))> for Border3D {
    fn from(ends: ((usize, usize), (usize, usize), (usize, usize))) -> Self {
        Border3D([ends.0, ends.1, ends.2])
    }
}
