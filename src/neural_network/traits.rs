//! Core traits for the neural network module: layers, losses, optimizers, and the named
//! parameter and weight views shared between them

use crate::error::Error;
use crate::neural_network::Shape;
use crate::neural_network::Tensor;
use crate::neural_network::ctx::{Ctx, Grads, StateSlot};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::checkpoint::BuildConfig;
use crate::{Deserialize, Serialize};
use ndarray::{ArrayViewD, ArrayViewMutD};

/// The stable address of 1 parameter tensor inside a model
///
/// A parameter is identified by the layer that holds it and by the name that the layer gives
/// it. Neither half moves while the model trains, so an optimizer can key its per-parameter
/// state on the pair and reach the same buffer on every step
///
/// The scope is the position of the layer in the model that drives the update.
/// [`Sequential`](crate::neural_network::sequential::Sequential) passes the index of the layer,
/// counted from the input. A caller that drives 1 layer directly passes any value it likes,
/// as long as it passes the same value on every step for that layer
///
/// The name is the `&'static str` that [`LayerBase::parameters_mut`] puts in the
/// [`ParamRef`]. It follows the layer, so a layer that stops yielding 1 of its
/// tensors, or that starts yielding a new one, moves no other tensor's address
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId {
    /// Position of the owning layer in the model, counted from the input
    pub scope: usize,
    /// Name the layer gives the tensor, such as `"kernel"`, `"bias"`, or `"gamma"`
    pub name: &'static str,
}

impl ParamId {
    /// Builds the address of the named parameter of the layer at the given position
    ///
    /// # Parameters
    ///
    /// - `scope` - Position of the owning layer in the model, counted from the input
    /// - `name` - Name the layer gives the tensor
    ///
    /// # Returns
    ///
    /// - `ParamId` - The parameter address
    #[inline]
    pub const fn new(scope: usize, name: &'static str) -> Self {
        Self { scope, name }
    }
}

/// A single trainable parameter tensor of a layer, exposed as a flat slice
///
/// Layers yield their trainable tensors (weights, biases, kernels, gamma/beta, ...) as
/// `ParamRef`s. This lets optimizers update any parameter shape with 1 flat-slice kernel,
/// instead of every layer/optimizer pair re-implementing the update
///
/// The entry holds no gradient. A backward pass puts every gradient in the
/// [`Grads`] store of the context, and the optimizer reads
/// it back with the [`ParamId`] that this name and the layer position build
///
/// Construct one with [`ParamRef::weight`] for a tensor that decoupled weight decay applies to
/// (weight matrices, conv/recurrent kernels). Use [`ParamRef::no_decay`] for a tensor it skips
/// (biases and normalization scale/shift `gamma`/`beta`). The `decays` flag tells the optimizer
/// which rule applies, so it never has to guess
///
/// The `name` is the half of the parameter address that the layer owns. It uses the Keras 3
/// name of the tensor, such as `kernel`, `recurrent_kernel`, `depthwise_kernel`, `bias`,
/// `embeddings`, `alpha`, `gamma`, or `beta`. A layer must give the same name to the same
/// storage on every call, and must give 2 different tensors 2 different names
pub struct ParamRef<'a> {
    /// Name the layer gives this tensor. See [`ParamId`]
    pub name: &'static str,
    /// Mutable view of the parameter's contiguous data that the optimizer updates in place
    pub value: &'a mut [f32],
    /// Whether decoupled (AdamW/SGDW-style) weight decay applies to this tensor. `true` for
    /// weight matrices and conv/recurrent kernels, `false` for biases and normalization
    /// scale/shift (`gamma`/`beta`)
    pub decays: bool,
}

impl<'a> ParamRef<'a> {
    /// A weight tensor that decoupled weight decay applies to (dense/conv/recurrent kernels)
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the tensor, such as `"kernel"`
    /// - `value` - Mutable view of the parameter's contiguous data
    ///
    /// # Returns
    ///
    /// - `ParamRef` - The entry, with `decays` set to `true`
    #[inline]
    pub fn weight(name: &'static str, value: &'a mut [f32]) -> Self {
        Self {
            name,
            value,
            decays: true,
        }
    }

    /// A bias or normalization scale/shift (`gamma`/`beta`) tensor that weight decay skips
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the tensor, such as `"bias"`
    /// - `value` - Mutable view of the parameter's contiguous data
    ///
    /// # Returns
    ///
    /// - `ParamRef` - The entry, with `decays` set to `false`
    #[inline]
    pub fn no_decay(name: &'static str, value: &'a mut [f32]) -> Self {
        Self {
            name,
            value,
            decays: false,
        }
    }
}

/// Whether training updates a named array of a layer
///
/// The 2 kinds are what [`ParamCounts`] counts, seen 1 array at a time.
/// A checkpoint holds the kind next to every array, so a load can refuse a file that offers a
/// trainable array where the layer keeps state, and the other way round
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightKind {
    /// An optimizer updates this array. It is a kernel, a bias, or a normalization
    /// scale/shift
    Trainable,
    /// The layer keeps this array and no optimizer writes it. The running statistics of
    /// [`BatchNormalization`](crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization)
    /// are the 1 example today
    NonTrainable,
}

/// 1 named array of a layer, borrowed for reading
///
/// [`LayerBase::weights`] gives 1 of these per array that the layer holds. The name is the layer
/// half of the address of the array, and the checkpoint format writes it. See
/// [`checkpoint`](crate::neural_network::layers::checkpoint)
///
/// The view borrows the live array, so nothing is copied
pub struct WeightRef<'a> {
    /// Name the layer gives this array, such as `"kernel"` or `"moving_mean"`
    pub name: &'static str,
    /// Whether an optimizer updates the array
    pub kind: WeightKind,
    /// Read-only view of the array, at the rank the layer holds it
    pub value: ArrayViewD<'a, f32>,
}

impl<'a> WeightRef<'a> {
    /// A read view of an array that an optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightRef` - The entry, with the kind set to [`WeightKind::Trainable`]
    #[inline]
    pub fn trainable(name: &'static str, value: ArrayViewD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::Trainable,
            value,
        }
    }

    /// A read view of an array that no optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightRef` - The entry, with the kind set to [`WeightKind::NonTrainable`]
    #[inline]
    pub fn non_trainable(name: &'static str, value: ArrayViewD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::NonTrainable,
            value,
        }
    }
}

/// 1 named array of a layer, borrowed for writing
///
/// [`LayerBase::weights_mut`] gives 1 of these per array that the layer holds, under the same
/// names and in the same order as [`LayerBase::weights`]. A checkpoint load writes through these
/// views, so it reaches the storage of the layer itself and keeps the memory order of that
/// storage
pub struct WeightMut<'a> {
    /// Name the layer gives this array, such as `"kernel"` or `"moving_mean"`
    pub name: &'static str,
    /// Whether an optimizer updates the array
    pub kind: WeightKind,
    /// Writable view of the array, at the rank the layer holds it
    pub value: ArrayViewMutD<'a, f32>,
}

impl<'a> WeightMut<'a> {
    /// A write view of an array that an optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightMut` - The entry, with the kind set to [`WeightKind::Trainable`]
    #[inline]
    pub fn trainable(name: &'static str, value: ArrayViewMutD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::Trainable,
            value,
        }
    }

    /// A write view of an array that no optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightMut` - The entry, with the kind set to [`WeightKind::NonTrainable`]
    #[inline]
    pub fn non_trainable(name: &'static str, value: ArrayViewMutD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::NonTrainable,
            value,
        }
    }
}

/// How many inputs a layer takes
///
/// Almost every layer takes 1. A merge layer takes 2 or more, and gives 1 output. A model
/// reads this before it wires a layer, so a node with the wrong fan-in is refused at build
/// time and not in the middle of a pass
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// The layer takes exactly this many inputs
    Exactly(usize),
    /// The layer takes this many inputs or more
    AtLeast(usize),
}

impl Arity {
    /// Whether the layer accepts that many inputs
    ///
    /// # Parameters
    ///
    /// - `count` - How many inputs a caller offers
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when the layer accepts the count
    #[inline]
    pub const fn accepts(self, count: usize) -> bool {
        match self {
            Self::Exactly(n) => count == n,
            Self::AtLeast(n) => count >= n,
        }
    }

    /// Refuses a count of inputs that the layer does not accept
    ///
    /// # Parameters
    ///
    /// - `layer` - The type name of the layer, for the message
    /// - `count` - How many inputs a caller offers
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the layer accepts the count
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer takes another count
    pub fn check(self, layer: &str, count: usize) -> Result<(), Error> {
        if self.accepts(count) {
            return Ok(());
        }
        let wanted = match self {
            Self::Exactly(n) => format!("exactly {n}"),
            Self::AtLeast(n) => format!("{n} or more"),
        };
        Err(Error::invalid_input(format!(
            "layer `{layer}` takes {wanted} inputs, and it received {count}"
        )))
    }
}

/// What every layer holds, whatever number of inputs it takes
///
/// The trait covers the 3 things that do not depend on the arity of a layer: what the layer
/// is, what arrays it owns, and what shape it was built for. The computation itself lives in
/// [`UnaryLayer`] for a layer with 1 input and in [`Layer`] for a layer with several
///
/// A layer holds no gradient and no cache. See [`Ctx`]
pub trait LayerBase: std::any::Any + Send + Sync {
    /// Returns the type name of the layer (e.g. "Dense")
    ///
    /// # Returns
    ///
    /// - `&str` - A string slice representing the layer type
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Returns how many parameters the layer holds, split by whether training updates them
    ///
    /// # Returns
    ///
    /// - `ParamCounts` - The trainable and the non-trainable element counts
    fn param_count(&self) -> ParamCounts;

    /// Exposes the layer's trainable parameters to the optimizer, by name
    ///
    /// Each returned [`ParamRef`] gives a parameter tensor's flat data and names the tensor.
    /// Layers without trainable parameters return the empty vector that the default gives.
    /// A layer yields every trainable tensor it holds on every call, whether a backward pass
    /// gave that tensor a gradient or not. The optimizer looks the gradient up in the store of
    /// the context and skips a tensor that holds none
    ///
    /// The name is the identity of the tensor, and the optimizer keys its per-parameter state
    /// on it (see [`ParamId`]). The same storage must therefore always come back under the same
    /// name, and 2 tensors of 1 layer must never share a name. The order is free, and it is the
    /// order that a global gradient norm reduces in, so a layer must keep it stable
    ///
    /// # Returns
    ///
    /// - `Vec<ParamRef<'_>>` - 1 named entry per trainable tensor
    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        Vec::new()
    }

    /// Every array the layer holds, by name, borrowed for reading
    ///
    /// The set covers the trainable arrays and the non-trainable state alike, which is exactly
    /// what a checkpoint holds. It is therefore a superset of what
    /// [`parameters_mut`](LayerBase::parameters_mut) yields:
    /// [`BatchNormalization`](crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization)
    /// adds its running statistics here
    ///
    /// The name of an array is its address inside the layer, and it follows Keras 3:
    /// `kernel`, `recurrent_kernel`, `depthwise_kernel`, `pointwise_kernel`, `bias`,
    /// `embeddings`, `alpha`, `gamma`, `beta`, `moving_mean`, `moving_variance`. A layer must
    /// give 1 array the same name on every call, and must never give 2 arrays the same name.
    /// A name that [`parameters_mut`](LayerBase::parameters_mut) also uses must reach the same
    /// storage
    ///
    /// The order is free, and it is the order a checkpoint records. Layers without any array
    /// return the empty vector
    ///
    /// # Returns
    ///
    /// - `Vec<WeightRef<'_>>` - 1 named view per array the layer holds
    fn weights(&self) -> Vec<WeightRef<'_>>;

    /// Every array the layer holds, by name, borrowed for writing
    ///
    /// The roster, the names, the kinds, and the order repeat [`weights`](LayerBase::weights)
    /// exactly. A checkpoint load looks a name up here and writes into the view, so the values
    /// reach the storage of the layer and take the memory order that the storage already has
    ///
    /// # Returns
    ///
    /// - `Vec<WeightMut<'_>>` - 1 named view per array the layer holds
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>>;

    /// 1 named array of the layer, or `None` when the layer holds no array of that name
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the array, such as `"kernel"`
    ///
    /// # Returns
    ///
    /// - `Option<ArrayViewD<'_, f32>>` - A read view of the array
    fn weight(&self, name: &str) -> Option<ArrayViewD<'_, f32>> {
        self.weights()
            .into_iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.value)
    }

    /// The input shapes the layer holds, or `None` while it holds none
    ///
    /// A layer that [`build`](UnaryLayer::build) has run on reports the shapes it was built
    /// for, with the batch axis freed. The batch axis is freed because 1 layer serves every
    /// batch size, so a layer built from a tensor of 2 samples still describes itself for
    /// every batch. A layer that needs no build at all, such as
    /// [`Rescaling`](crate::neural_network::layers::rescaling::Rescaling), always reports
    /// `None`. The vector holds 1 shape per input of the layer, so a merge layer reports
    /// several
    ///
    /// [`Layer::output_shape`] runs
    /// [`Layer::compute_output_shape_many`] against these shapes. The method is therefore the
    /// 1 place where the display value reads state, and the shape algebra stays pure
    ///
    /// # Returns
    ///
    /// - `Option<Vec<Shape>>` - The input shapes the layer holds
    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        None
    }

    /// Whether the layer already holds every array it needs
    ///
    /// [`UnaryLayer::forward_mut`] reads this, and builds the layer from the tensor it
    /// received when the answer is `false`. A layer that owns no array and reads no extent of
    /// its input keeps the default `true`, because there is nothing left to allocate
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when no build is outstanding
    fn is_built(&self) -> bool {
        true
    }

    /// The shapes that the layer was built for, when the layer knows them
    ///
    /// A layer that [`build`](UnaryLayer::build) has run on reports the shapes it was built
    /// for, with the batch axis freed. A layer that owns no array and reads no extent of its
    /// input keeps the default `None`, and so does any layer before its build. A checkpoint
    /// records this value, and a load compares it. See [`BuildConfig`]
    ///
    /// The batch axis is freed because 1 layer serves every batch size. A model built for 32
    /// samples and a model built for 1 sample therefore report the same build shape, and a
    /// checkpoint moves between them
    ///
    /// # Returns
    ///
    /// - `Option<BuildConfig>` - The build shapes, or `None` while the layer holds none
    fn build_config(&self) -> Option<BuildConfig> {
        None
    }

    /// Moves the non-trainable state that a forward pass proposed into the layer
    ///
    /// A forward pass takes `&self`, so a layer that changes non-trainable state writes the
    /// new value into the state channel of the context instead. The model calls this after the
    /// forward pass, and the layer takes back every value it recognizes.
    /// [`BatchNormalization`](crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization)
    /// takes its running statistics here, and a dropout layer takes its random stream
    ///
    /// The default does nothing, which is right for every layer whose forward pass changes
    /// nothing outside the context
    ///
    /// # Parameters
    ///
    /// - `state` - The state channel of this layer, for the pass that just ran
    fn apply_state(&mut self, state: &mut StateSlot<'_>) {
        let _ = state;
    }
}

/// A layer that takes 1 input and gives 1 output
///
/// Almost every layer of this crate is one. Implement this trait, and the blanket
/// `impl<T: UnaryLayer> Layer for T` gives the layer the general [`Layer`] interface that a
/// model holds it through. Implement [`Layer`] itself only for a layer that takes several
/// inputs, such as a merge layer
///
/// The forward pass takes `&self`. Everything it needs to remember for its backward pass goes
/// into the [`Ctx`], and so does every gradient it produces.
/// A layer that changes non-trainable state writes it into the state channel of the context.
/// The layer itself is therefore read-only during a pass. A built
/// [`Sequential`](crate::neural_network::sequential::Sequential) is `Send` and `Sync` for the
/// same reason, so several threads can run inference against 1 model
pub trait UnaryLayer: LayerBase {
    /// Runs the forward pass through the layer
    ///
    /// A training pass parks in `ctx` whatever the backward pass needs. An inference pass
    /// parks nothing at all and takes the inference behavior of a mode-dependent layer. Read
    /// [`Ctx::is_training`](crate::neural_network::ctx::Ctx::is_training) to tell the 2 apart
    ///
    /// The method refuses an unbuilt layer. Use [`forward_mut`](UnaryLayer::forward_mut) to
    /// build from the tensor itself, or [`build`](UnaryLayer::build) first
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Tensor` - The layer's output, in the standard memory order
    ///
    /// # Errors
    ///
    /// - `Error` - If the forward pass fails (e.g. shape mismatch)
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer holds no build
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error>;

    /// Runs the backward pass through the layer
    ///
    /// The method takes back what the matching forward pass parked in `ctx`, and it adds the
    /// gradient of every parameter it owns to the gradient store of `ctx`
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    /// - `ctx` - The context of the pass, holding what the forward pass parked
    ///
    /// # Returns
    ///
    /// - `Tensor` - The gradient to pass to the previous layer, in the standard memory order
    ///
    /// # Numerical policy
    ///
    /// Backward is pure math: it does **not** sanitize NaN/Inf (no zeroing, no element-wise
    /// clamping). The backward pass propagates such values instead of masking them. The forward
    /// pass masks nothing either: it validates the rank, the shape, and the layer parameters,
    /// and it never reads the input values to reject them. A NaN or an infinity therefore stays
    /// in the tensor, moves on through every later layer, and shows itself in the output and in
    /// a non-finite loss. [`Embedding`](crate::neural_network::layers::Embedding) is the 1
    /// exception, because it reads its input as a table of row indices and rejects a non-finite
    /// index with `Error::InvalidInput`. To tame large-but-finite gradients, enable
    /// clip-by-global-norm on the optimizer
    /// ([`Optimizer::global_clipnorm`]) instead of clamping inside a layer. Global-norm scaling
    /// preserves gradient direction, unlike per-element clamping
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer, because no forward pass of this layer wrote one
    /// - `Error` - If the layer encountered an error during processing (e.g. shape mismatch)
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error>;

    /// Allocates every array the layer owns, from the shape of the input it will receive
    ///
    /// A constructor takes the configuration of a layer and nothing else. It draws no weight,
    /// because a kernel extent comes from the input and the input is not there yet. `build` is
    /// where the layer learns that shape, checks it against its own configuration, and
    /// allocates. A layer that owns no array still records the shape, so it can check every
    /// later input against it and report an honest output shape
    ///
    /// [`SequentialBuilder::build`](crate::neural_network::sequential::SequentialBuilder::build)
    /// calls this once per layer, threading the output shape of each layer into the next
    /// through [`compute_output_shape`](UnaryLayer::compute_output_shape).
    /// [`forward_mut`](UnaryLayer::forward_mut) builds a layer that a caller drives directly,
    /// from the shape of the tensor it receives
    ///
    /// A second call with the same shape does nothing, and no array is drawn twice. A second
    /// call with another shape is an error, because it would silently replace every weight the
    /// layer holds
    ///
    /// The default does nothing. It is right for every layer that owns no array and reads no
    /// extent of its input, such as an activation
    ///
    /// # Parameters
    ///
    /// - `input` - Shape of the tensor that enters the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the layer holds every array it needs
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept an input of that shape, or if the
    ///   layer is already built for another shape
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let _ = input;
        Ok(())
    }

    /// The output shape the layer gives for an input of the given shape
    ///
    /// The answer is a pure function of the layer configuration and of `input`. The method
    /// reads no cache that a forward pass wrote, so it gives the same answer before any tensor
    /// reaches the layer and after any number of passes. A free axis of the input stays free
    /// in the output wherever the layer passes it through, which is how 1 layer describes
    /// itself for every batch size
    ///
    /// The method refuses an input the layer cannot accept, and the message names the layer
    /// and the axis at fault. A pure shape function is what lets a model walk its layers at
    /// build time, thread each output shape into the next layer, and reject a bad stack before
    /// any data arrives, naming the position of the layer and its type
    ///
    /// The default passes the input through unchanged, which is right for every layer that
    /// changes values and not extents
    ///
    /// # Parameters
    ///
    /// - `input` - Shape of the tensor that enters the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<Shape, Error>` - Shape of the tensor the layer gives back
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept an input of that shape
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        Ok(input.clone())
    }

    /// Builds the layer from the tensor when it holds no build, then runs the forward pass
    ///
    /// This is the entry point for a caller that drives 1 layer by hand and never calls
    /// [`build`](UnaryLayer::build). The whole shape of the tensor goes into the build, batch
    /// extent included, so the layer reports the shape it was really given. A model never uses
    /// this method, because
    /// [`SequentialBuilder::build`](crate::neural_network::sequential::SequentialBuilder::build)
    /// has already built every layer it holds
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The layer's output
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept an input of that shape
    /// - `Error` - If the forward pass fails
    fn forward_mut(&mut self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            self.build(&Shape::known(input.shape()))?;
        }
        self.forward(input, ctx)
    }
}

/// A layer that takes 1 input or several, and gives 1 output
///
/// This is the general interface, and it is the one a model holds a layer through. A layer
/// with 1 input implements [`UnaryLayer`] instead, and the blanket
/// `impl<T: UnaryLayer> Layer for T` gives it this interface for free. Implement this trait by
/// hand only for a layer that takes several inputs
///
/// Every method that ends in `_many` is the same operation as the [`UnaryLayer`] method of the
/// same stem, over every input of the layer
pub trait Layer: LayerBase {
    /// How many inputs the layer takes
    ///
    /// # Returns
    ///
    /// - `Arity` - The accepted input count
    fn arity(&self) -> Arity {
        Arity::Exactly(1)
    }

    /// Runs the forward pass through the layer, over every input
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 tensor per input of the layer, in the order the model wired them
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Tensor` - The layer's output, in the standard memory order
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the count of `inputs` does not match [`arity`](Layer::arity)
    /// - `Error` - If the forward pass fails
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error>;

    /// Runs the backward pass through the layer, giving 1 gradient per input
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    /// - `ctx` - The context of the pass, holding what the forward pass parked
    ///
    /// # Returns
    ///
    /// - `Vec<Tensor>` - 1 gradient per input, in the order the forward pass took them
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer
    /// - `Error` - If the backward pass fails
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error>;

    /// Allocates every array the layer owns, from the shapes of the inputs it will receive
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 shape per input of the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the layer holds every array it needs
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept inputs of those shapes
    fn build_many(&mut self, inputs: &[Shape]) -> Result<(), Error>;

    /// The output shape the layer gives for inputs of the given shapes
    ///
    /// The answer is a pure function of the layer configuration and of `inputs`
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 shape per input of the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<Shape, Error>` - Shape of the tensor the layer gives back
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept inputs of those shapes
    fn compute_output_shape_many(&self, inputs: &[Shape]) -> Result<Shape, Error>;

    /// Builds the layer from the tensors when it holds no build, then runs the forward pass
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 tensor per input of the layer
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The layer's output
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept inputs of those shapes
    /// - `Error` - If the forward pass fails
    fn forward_many_mut(&mut self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            let shapes: Vec<Shape> = inputs.iter().map(|t| Shape::known(t.shape())).collect();
            self.build_many(&shapes)?;
        }
        self.forward_many(inputs, ctx)
    }

    /// Returns a description of the output shape of the layer
    ///
    /// The value is [`compute_output_shape_many`](Layer::compute_output_shape_many) run
    /// against [`known_input_shapes`](LayerBase::known_input_shapes). A layer that holds no
    /// input shape, and a layer whose held shapes the shape algebra refuses, both report
    /// `"Unknown"`
    ///
    /// # Returns
    ///
    /// - `String` - A string describing the output dimensions
    fn output_shape(&self) -> String {
        match self.known_input_shapes() {
            Some(inputs) => match self.compute_output_shape_many(&inputs) {
                Ok(output) => output.to_string(),
                Err(_) => "Unknown".to_string(),
            },
            None => "Unknown".to_string(),
        }
    }
}

impl<T: UnaryLayer> Layer for T {
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::Exactly(1).check(self.layer_type(), inputs.len())?;
        UnaryLayer::forward(self, inputs[0], ctx)
    }

    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        Ok(vec![UnaryLayer::backward(self, grad_output, ctx)?])
    }

    fn build_many(&mut self, inputs: &[Shape]) -> Result<(), Error> {
        Arity::Exactly(1).check(self.layer_type(), inputs.len())?;
        UnaryLayer::build(self, &inputs[0])
    }

    fn compute_output_shape_many(&self, inputs: &[Shape]) -> Result<Shape, Error> {
        Arity::Exactly(1).check(self.layer_type(), inputs.len())?;
        UnaryLayer::compute_output_shape(self, &inputs[0])
    }
}

/// Defines the interface for loss functions used in neural network training
///
/// An implementation computes both the loss value and its gradient with respect to
/// the predicted values
///
/// # Averaging convention
///
/// Each loss normalizes by what is natural for its family, so the conventions differ on
/// purpose. `compute_grad` is always exactly the gradient of `compute_loss`. Switching loss
/// families rescales the gradient magnitude, and thus the effective learning rate:
///
/// - [`MeanSquaredError`](crate::neural_network::losses::MeanSquaredError),
///   [`MeanAbsoluteError`](crate::neural_network::losses::MeanAbsoluteError) and
///   [`BinaryCrossEntropy`](crate::neural_network::losses::BinaryCrossEntropy) average over
///   **every element** (`y.len()`), treating each output as an independent target
/// - [`CategoricalCrossEntropy`](crate::neural_network::losses::CategoricalCrossEntropy) sums
///   over the trailing **class** axis and averages over every **prediction site** (the product
///   of all leading axes). For a `[batch, classes]` target, that divisor is the batch. For the
///   `[batch, height, width, classes]` output of a channels-last softmax conv head, the divisor
///   is `batch * height * width`, 1 site per pixel
/// - [`SparseCategoricalCrossEntropy`](crate::neural_network::losses::SparseCategoricalCrossEntropy)
///   is the same per-sample categorical cross-entropy, but accepts only rank-2
///   `[batch, classes]` predictions, so its divisor is always the batch
///
/// The 2 categorical losses also renormalize `y_pred` along the class axis before
/// clipping when `from_logits` is off, as Keras does. That leaves the loss value alone for an
/// already-normalized head but contributes a row-constant term to the gradient, which a softmax
/// backward annihilates
pub trait Loss: Send + Sync {
    /// Computes the loss between true and predicted values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// - `f32` - The scalar loss value
    ///
    /// # Errors
    ///
    /// - `Error` - If the inputs are inconsistent (e.g. mismatched shapes or, for the
    ///   sparse loss, out-of-range labels)
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error>;

    /// Computes the gradient of the loss with respect to the predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// - `Tensor` - Tensor containing the gradient of the loss with respect to predictions
    ///
    /// # Errors
    ///
    /// - `Error` - If the inputs are inconsistent (see [`compute_loss`](Loss::compute_loss))
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error>;
}

/// Defines the interface for optimization algorithms
///
/// An implementation updates layer parameters during training
pub trait Optimizer: Send + Sync {
    /// Advances the optimizer's global training step
    ///
    /// Called exactly once per batch, before the per-layer [`update`](Optimizer::update) calls.
    /// The method now carries 1 duty only: a step-dependent optimizer advances the counter that
    /// its own math reads. [`Adam`](crate::neural_network::optimizers::Adam) and
    /// [`AdamW`](crate::neural_network::optimizers::AdamW) advance the bias-correction timestep
    /// here, so the correction moves once per batch rather than once per layer. SGD, RMSprop,
    /// and AdaGrad hold no such counter and keep the no-op default
    ///
    /// The method no longer rewinds anything. Per-parameter state is keyed by
    /// [`ParamId`], which is the position of the layer plus the name the layer gives the
    /// tensor, so nothing walks a cursor that a rewind could leave in the wrong place
    fn step(&mut self) {}

    /// The global gradient-norm clip threshold, or `None` (the default) to disable clipping
    ///
    /// When `Some(max_norm)`, the training loop computes the global L2 norm across **all** of
    /// the model's gradients. If the norm exceeds `max_norm`, it scales every gradient by
    /// `max_norm / global_norm` before [`update`](Optimizer::update). This single uniform factor
    /// preserves gradient direction (unlike per-element clamping). When the global norm is
    /// non-finite, the clip leaves gradients unscaled, so divergence still surfaces instead of
    /// being masked
    ///
    /// The name is Keras' `global_clipnorm`, and it is deliberate. Keras also has a `clipnorm`,
    /// which renormalizes each variable's gradient *independently* against the threshold. The 2
    /// give different directions whenever more than 1 tensor is over the limit. A
    /// threshold tuned for one is therefore not a threshold for the other. Only the global form
    /// exists here
    fn global_clipnorm(&self) -> Option<f32> {
        None
    }

    /// Updates the parameters of a layer according to the optimization algorithm
    ///
    /// The optimizer builds a [`ParamId`] from `scope` and the name of each
    /// [`ParamRef`], and keys its per-parameter state on that address. It reads the gradient
    /// of that address out of `grads`, and it skips a parameter that holds none. The caller
    /// must therefore give the same layer the same `scope` on every step
    ///
    /// # Parameters
    ///
    /// - `scope` - Position of this layer in the model, counted from the input. It is the
    ///   layer half of the parameter address
    /// - `layer` - The layer whose parameters should be updated
    /// - `grads` - Every gradient the backward pass produced
    /// - `grad_scale` - Uniform factor that the training loop applies to every gradient before
    ///   the update, to implement clip-by-global-norm. Pass `1.0` for an unscaled update
    fn update(&mut self, scope: usize, layer: &mut dyn LayerBase, grads: &Grads, grad_scale: f32);

    /// The current learning rate
    ///
    /// The read half of the scheduling pair. A schedule (exponential decay, cosine annealing,
    /// warmup restarts) derives its next step size from the current one. It needs this method
    /// for that, rather than a copy of the rate kept alongside the model. A separate copy would
    /// drift the moment anything else retunes the optimizer. Reports whatever was last set.
    /// Unlike the constructors, [`set_learning_rate`](Optimizer::set_learning_rate) does not
    /// validate, so a rate set to 0 or to a negative value comes back unchanged
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate, the hook for external learning-rate scheduling
    ///
    /// Call this between batches or epochs, for step decay or warmup, to retune the step size
    /// without rebuilding the optimizer. The optimizer keeps its accumulated state
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - The new learning rate to use for subsequent updates
    fn set_learning_rate(&mut self, learning_rate: f32);
}
