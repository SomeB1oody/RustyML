//! Reverse layer that flips the order of the positions along 1 axis

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};
use ndarray::{Axis, Slice};

/// Reverses the order of the positions along 1 axis
///
/// The layer changes no shape and holds no array. Position `k` of the named axis leaves at
/// position `n - 1 - k`, and every other axis stays as it was.
///
/// The main use is the time axis of a recurrent branch that reads its input from last to first.
/// [`SimpleRNN::with_go_backwards`](crate::neural_network::layers::SimpleRNN::with_go_backwards)
/// and its 2 siblings return their states in PROCESSING order, so slot 0 of a returned sequence
/// holds the state that came from the LAST input timestep. A merge layer that joins such a
/// branch to a forward branch would pair mismatched timesteps. This layer puts the backward
/// branch back into input order first. See the module documentation of
/// [`graph`](crate::neural_network::graph) for the whole model.
///
/// # Notes
///
/// **The axis counts against the FULL rank, and axis 0 is the batch axis.** A negative axis
/// counts back from the end, so -1 is the last axis. The layer keeps the axis as given and
/// resolves it against the rank on every call, which follows
/// [`Concatenate`](crate::neural_network::layers::Concatenate). The batch axis is refused,
/// because reversing it would reorder the samples of a batch against their targets.
///
/// **The input must hold at least 3 axes.** A rank-2 input is 1 batch of feature vectors, and
/// it holds no time axis and no spatial axis. Reversing the feature order of such an input is
/// not an operation of this crate, and it is what a caller gets by mistake after leaving
/// `return_sequences` unset on a recurrent branch. The build refuses that input rather than
/// reordering the features without a word.
///
/// **A padded sequence needs care.** This crate carries no mask. Padding that sits at the end
/// of a sequence sits at the FRONT after this layer, so a recurrent layer that reads the
/// reversed sequence starts on the padding. Reverse the values of a ragged batch yourself, per
/// sample, before the model reads them.
///
/// **The reported type carries the axis.** A checkpoint compares the type of every layer of a
/// position, and it compares the shapes that the layer built for. This layer changes no shape
/// and holds no array, so the axis is the only thing that separates 2 of them, and the type is
/// the only field a strict load would see it in. The type is therefore `Reverse(1)` and not
/// `Reverse`, and a load that meets a different axis refuses.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array;
///
/// // 2 samples, 4 timesteps, 3 features. Value 0 of every sequence is its first timestep
/// let x: Array<f32, _> = Array::from_shape_fn((2, 4, 3), |(_, t, _)| t as f32).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(Reverse::new(1))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// let prediction = model.predict(&x).unwrap();
/// assert_eq!(prediction.shape(), &[2, 4, 3]);
/// // The first timestep of the output is the last timestep of the input
/// assert_eq!(prediction[[0, 0, 0]], 3.0);
/// ```
#[derive(Debug)]
pub struct Reverse {
    /// The axis to reverse, as the caller gave it. A negative value counts back from the end
    axis: i32,
    /// The reported type of the layer, which carries the axis
    name: String,
    /// Shape the layer built for, or `None` before the build
    built: Option<Shape>,
}

impl Reverse {
    /// Creates a layer that reverses the order along 1 axis
    ///
    /// The call cannot fail. The axis counts against the full rank of the input, and the layer
    /// resolves it on every call, so 1 layer serves every rank that holds the axis. A layer that
    /// held a resolved index would reverse the wrong axis as soon as the rank changed
    ///
    /// # Parameters
    ///
    /// - `axis` - The axis to reverse. Axis 0 is the batch axis and the build refuses it. A
    ///   negative value counts back from the end, so -1 is the last axis
    ///
    /// # Returns
    ///
    /// - `Self` - The layer
    pub fn new(axis: i32) -> Self {
        Self {
            axis,
            name: format!("Reverse({axis})"),
            built: None,
        }
    }

    /// Resolves the axis against a rank, and refuses a rank or an axis the layer cannot serve
    ///
    /// # Parameters
    ///
    /// - `rank` - Number of axes of the input
    ///
    /// # Returns
    ///
    /// - `Result<usize, Error>` - The axis, counted from 0
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] - If the rank is below 3
    /// - [`Error::InvalidInput`] - If the axis does not name an axis of that rank
    /// - [`Error::InvalidInput`] - If the axis resolves to the batch axis
    fn resolve(&self, rank: usize) -> Result<usize, Error> {
        if rank < 3 {
            return Err(Error::invalid_input(format!(
                "Reverse takes an input of rank 3 or more, and it got rank {rank}. A rank-2 \
                 input holds 1 feature vector per sample, and it holds no axis whose order \
                 carries meaning. A recurrent branch that gives rank 2 returned its last state \
                 alone, and it needs `with_return_sequences(true)` to give a time axis"
            )));
        }
        let resolved = if self.axis < 0 {
            let back = usize::try_from(-self.axis).expect("a negative i32 negates into a usize");
            match rank.checked_sub(back) {
                Some(index) => index,
                None => return Err(self.axis_refusal(rank)),
            }
        } else {
            usize::try_from(self.axis).expect("a positive i32 is a usize")
        };
        if resolved >= rank {
            return Err(self.axis_refusal(rank));
        }
        if resolved == 0 {
            return Err(Error::invalid_input(format!(
                "Reverse cannot reverse axis 0 of an input of rank {rank}, because axis 0 is \
                 the batch axis. Reversing it would put every sample against the target of \
                 another sample"
            )));
        }
        Ok(resolved)
    }

    /// Builds the refusal of an axis that the rank does not hold
    #[cold]
    fn axis_refusal(&self, rank: usize) -> Error {
        Error::invalid_input(format!(
            "Reverse names the axis {}, and the input holds rank {rank}. The axis counts \
             against the full rank, and a negative axis counts back from the end",
            self.axis
        ))
    }

    /// Reverses `input` along the resolved axis, in the standard memory order
    ///
    /// A reversed view carries a negative stride, so it is not the standard order. The output
    /// is a fresh array that the view is assigned into, because every layer of this crate emits
    /// the standard order and the next layer may read the output as 1 slice
    fn flip(&self, input: &Tensor) -> Result<Tensor, Error> {
        let axis = self.resolve(input.ndim())?;
        let mut output = Tensor::zeros(input.raw_dim());
        output.assign(&input.view().slice_axis(Axis(axis), Slice::new(0, None, -1)));
        Ok(output)
    }
}

impl LayerBase for Reverse {
    fn layer_type(&self) -> &str {
        &self.name
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Reverse {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.resolve(input.rank())?;
        let Some(built) = start_build(&self.built, self.layer_type(), input)? else {
            return Ok(());
        };
        self.built = Some(built);
        Ok(())
    }

    /// The reversal needs nothing from the forward pass, so no cache is parked
    ///
    /// The backward pass reverses the same axis of the upstream gradient. The reversal is its
    /// own inverse, and it moves each value without changing it, so the gradient of position
    /// `n - 1 - k` of the output belongs to position `k` of the input
    fn forward(&self, input: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        self.flip(input)
    }

    fn backward(&self, grad_output: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        self.flip(grad_output)
    }

    /// The shape does not move, and the rank and the axis are checked
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.resolve(input.rank())?;
        Ok(input.clone())
    }
}
