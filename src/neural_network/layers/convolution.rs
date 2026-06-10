//! Convolutional layers and shared padding types
//!
//! Re-exports the 1D/2D/3D, depthwise, and separable convolution layers, and
//! defines the [`PaddingType`] used to control spatial padding

/// Padding method used by convolutional layers
///
/// Determines how the input is padded before convolution is applied
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingType {
    /// No padding is applied; the convolution is computed only where the filter
    /// fully overlaps the input, producing an output with reduced dimensions
    Valid,

    /// Zeros are added around the input borders so the output keeps the same
    /// spatial dimensions as the input (when stride is 1)
    Same,
}

/// 1D convolutional layer
pub mod conv_1d;
/// 2D convolutional layer
pub mod conv_2d;
/// 3D convolutional layer
pub mod conv_3d;
/// Dimension-generic convolution engine shared by Conv1D, Conv2D, and Conv3D
mod convolution_engine;
/// 2D depthwise convolutional layer
pub mod depthwise_conv_2d;
/// 2D separable convolutional layer
pub mod separable_conv_2d;
/// Input validation helpers for convolutional layers
mod validation;

pub use conv_1d::Conv1D;
pub use conv_2d::Conv2D;
pub use conv_3d::Conv3D;
pub use depthwise_conv_2d::DepthwiseConv2D;
pub use separable_conv_2d::SeparableConv2D;
