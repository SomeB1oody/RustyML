//! Convolutional layers and shared padding types
//!
//! Re-exports the 1D, 2D, and 3D convolution layers, plus the depthwise and separable
//! convolution layers. Defines [`PaddingType`], which controls spatial padding

/// Padding method used by convolutional and pooling layers
///
/// Determines how the layer pads its input before the operation runs. Defaults to
/// [`PaddingType::Valid`] (no padding)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingType {
    /// Applies no padding. The convolution runs only where the filter fully overlaps the
    /// input, so the output has smaller spatial dimensions than the input
    #[default]
    Valid,

    /// Adds zeros around the input borders so the output keeps the same spatial dimensions
    /// as the input, when the stride is 1
    Same,
}

/// 1D convolutional layer
pub mod conv_1d;
/// 2D convolutional layer
pub mod conv_2d;
/// 3D convolutional layer
pub mod conv_3d;
/// Dimension-generic convolution engine shared by Conv1D, Conv2D, Conv3D, and the pointwise
/// stage of SeparableConv2D
pub(crate) mod convolution_engine;
/// 2D depthwise convolutional layer
pub mod depthwise_conv_2d;
/// 2D separable convolutional layer
pub mod separable_conv_2d;
/// Parameter and input-shape validation shared by the convolution layers
mod validation;

pub use conv_1d::Conv1D;
pub use conv_2d::Conv2D;
pub use conv_3d::Conv3D;
pub use depthwise_conv_2d::DepthwiseConv2D;
pub use separable_conv_2d::SeparableConv2D;
