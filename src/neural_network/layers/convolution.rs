/// Defines the padding method used in convolutional layers.
///
/// The padding type determines how the input is padded before applying convolution:
/// - `Valid`: No padding is applied, which reduces the output dimensions.
/// - `Same`: Padding is added to preserve the input spatial dimensions in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingType {
    /// No padding is applied. The convolution is only computed where the filter
    /// fully overlaps with the input, resulting in an output with reduced dimensions.
    Valid,

    /// Padding is added around the input to ensure that the output has the same
    /// spatial dimensions as the input (when stride is 1). This is done by adding
    /// zeros around the borders of the input.
    Same,
}

/// 1D Convolutional Layer
pub mod conv_1d;
/// 2D Convolutional Layer
pub mod conv_2d;
/// 3D Convolutional Layer
pub mod conv_3d;
/// Dimension-generic convolution engine shared by Conv1D/Conv2D/Conv3D
mod convolution_engine;
/// 2D Depthwise Convolutional Layer
pub mod depthwise_conv_2d;
/// 2D Separable Convolutional Layer
pub mod separable_conv_2d;
/// Input validation functions for convolutional layers
mod validation;

pub use conv_1d::Conv1D;
pub use conv_2d::Conv2D;
pub use conv_3d::Conv3D;
pub use depthwise_conv_2d::DepthwiseConv2D;
pub use separable_conv_2d::SeparableConv2D;
