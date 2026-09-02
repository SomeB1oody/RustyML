//! Rescaling layer that applies a fixed affine map to every element of its input

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;
use ndarray::Zip;

/// Applies `y = x * scale + offset` to every element
///
/// The map is elementwise and it keeps the input shape at every rank. The layer holds no
/// parameter, so `scale` and `offset` stay at the values that the constructor got. Training
/// never moves them
///
/// The layer has no training mode and no inference mode. `predict` therefore returns exactly
/// what `forward` returns. The backward pass multiplies the incoming gradient by `scale`,
/// because the derivative of the map is `scale` at every element. The `offset` is a constant,
/// so it has no part in the gradient
///
/// Its use is input normalization in front of a model. An 8-bit image scales into `[0, 1]`
/// with `Rescaling::new(1.0 / 255.0)`, and into `[-1, 1]` with
/// `Rescaling::new(1.0 / 127.5).with_offset(-1.0)`. A layer keeps that step inside the model,
/// so inference applies the same map as training
///
/// A `scale` of 0 and a negative `scale` are both legal, and the layer applies them like any
/// other value
///
/// # Notes
///
/// The layer stores no cache, not even the shape of the last input. Nothing in the forward
/// pass or the backward pass needs one. `known_input_shape` therefore reports `None` and
/// `output_shape` reads "Unknown", and `backward` runs correctly before any forward pass.
/// `compute_output_shape` still answers for any shape a caller passes, because the layer
/// changes values and not extents
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array2;
///
/// // A rank-2 input: 2 samples of 3 features each
/// let x = Array2::from_shape_vec((2, 3), vec![0.0, 51.0, 102.0, 153.0, 204.0, 255.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(Rescaling::new(1.0 / 255.0))
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let scaled = model.predict(&x).unwrap();
///
/// // Every value lands in [0, 1]
/// assert!(scaled.iter().all(|v| (0.0..=1.0).contains(v)));
/// assert!((scaled[[0, 1]] - 0.2).abs() < 1e-6);
/// ```
#[derive(Debug)]
pub struct Rescaling {
    /// Factor that multiplies every element
    scale: f32,
    /// Constant that the layer adds after the multiplication
    offset: f32,
}

impl Rescaling {
    /// Creates a new Rescaling layer with an offset of 0
    ///
    /// # Parameters
    ///
    /// - `scale` - Factor that multiplies every element
    ///
    /// # Returns
    ///
    /// - `Self` - New `Rescaling` layer instance
    pub fn new(scale: f32) -> Self {
        Rescaling { scale, offset: 0.0 }
    }

    /// Sets the constant that the layer adds after the multiplication
    ///
    /// The offset is 0 until this method sets it
    ///
    /// # Parameters
    ///
    /// - `offset` - Constant that the layer adds after the multiplication
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }

    /// Checks the rank and the element count of a tensor that enters the layer
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the input has no batch axis
    /// - `Error::EmptyInput` - If any axis has an extent of 0
    fn validate(input: &Tensor) -> Result<(), Error> {
        if input.ndim() == 0 {
            return Err(Error::invalid_input(
                "Rescaling layer expects an input with a batch axis, got a 0D tensor",
            ));
        }
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }
        Ok(())
    }
}

/// Maps `f` over `t` into a new tensor that is in C order
///
/// Every layer here emits C order, so a consumer can read any layer output as 1 contiguous
/// slice. The destination comes from `Tensor::zeros`, which is always in C order, and the
/// source reaches it through a `Zip` that accepts any input layout
fn map_in_c_order(t: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    let mut out = Tensor::zeros(t.raw_dim());
    Zip::from(&mut out).and(t).for_each(|o, &x| *o = f(x));
    out
}

impl Layer for Rescaling {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // The layer keeps no cache and reads no training mode, so the 2 passes agree exactly
        self.predict(input)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        Self::validate(input)?;
        let (scale, offset) = (self.scale, self.offset);
        Ok(map_in_c_order(input, |x| x * scale + offset))
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        Self::validate(grad_output)?;
        // d(x * scale + offset) / dx is scale, so the constant offset drops out
        let scale = self.scale;
        Ok(map_in_c_order(grad_output, |g| g * scale))
    }

    fn layer_type(&self) -> &str {
        "Rescaling"
    }

    no_trainable_parameters_layer_functions!();
}
