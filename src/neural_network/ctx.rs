//! The per-pass context that carries the values a forward pass and its backward pass share
//!
//! A layer computes. It does not remember.
//! [`UnaryLayer::forward`](crate::neural_network::traits::UnaryLayer::forward) takes `&self`,
//! so the values that only exist between a forward pass and its backward pass have no place
//! inside the layer. [`Ctx`] is that place. 1 context serves 1 pass, and it holds 4 channels:
//!
//! 1. The training flag, which every mode-dependent layer reads.
//! 2. The cache, which a forward pass writes and the matching backward pass reads.
//! 3. The gradient store, which a backward pass writes and the optimizer reads.
//! 4. The state channel, which carries the non-trainable values that a training pass changes.
//!    Examples include the running statistics of a normalization layer and the random stream
//!    of a dropout layer.
//!
//! The context lives for 1 pass. A gradient therefore cannot survive into the next step, and a
//! cache cannot survive into the next pass.

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::traits::ParamId;
use ahash::AHashMap;
use std::any::Any;
use std::collections::BTreeMap;

/// The position of 1 layer in the model that drives it
///
/// [`Sequential`](crate::neural_network::sequential::Sequential) gives each layer its index,
/// counted from the input. A model that holds a layer arena gives each layer its arena index.
/// That lets 1 layer shared by several positions of the model keep 1 identity. A caller that
/// drives 1 layer by hand takes the default of 0
pub type LayerId = usize;

/// The position of 1 CALL of a layer in the pass that drives it
///
/// A layer and a call of that layer are not the same identity. A model that holds a layer
/// arena can call 1 layer at several positions. Each of those calls has its own input, its
/// own output, and therefore its own cache. The gradients of those calls sum into 1 parameter,
/// so a gradient is addressed by the layer. A cache is not, and it is addressed by the call
///
/// [`Sequential`](crate::neural_network::sequential::Sequential) calls each layer once, so the
/// 2 numbers agree there and [`Ctx::set_owner`] sets both at once
pub type CallId = usize;

/// A value that a layer parks in the context between 2 calls
type Slot = Box<dyn Any + Send + Sync>;

/// 1 parked cache, together with the type name of the layer that parked it
///
/// The name is what makes a mis-addressed take an error. A cache is `dyn Any`, and most layers
/// park a plain tensor or a plain shape. The type alone therefore separates almost nothing.
/// A take that reaches the stack of another layer would find a value of the right type and
/// give back the wrong numbers
struct CacheSlot {
    /// The type name of the layer that parked the value
    layer: &'static str,
    /// The parked value
    value: Slot,
}

/// Every parameter gradient of 1 pass, addressed by [`ParamId`]
///
/// A backward pass adds gradients here, and the optimizer reads them. The store owns the
/// values, so no layer holds a gradient field and no gradient survives the pass that made it
///
/// The order of [`iter`](Grads::iter) follows [`ParamId`], which sorts by layer position and
/// then by name. A caller that needs the canonical order of the model walks the layers instead
/// and looks each name up. The layer order is the order that every reduction of this crate
/// uses
#[derive(Debug, Default)]
pub struct Grads {
    /// 1 entry per parameter that a backward pass gave a gradient
    map: BTreeMap<ParamId, Tensor>,
}

impl Grads {
    /// The gradient of 1 parameter, or `None` when no backward pass gave it one
    ///
    /// # Parameters
    ///
    /// - `id` - The address of the parameter
    ///
    /// # Returns
    ///
    /// - `Option<&Tensor>` - The gradient, in the memory order of the parameter
    #[inline]
    pub fn get(&self, id: ParamId) -> Option<&Tensor> {
        self.map.get(&id)
    }

    /// How many parameters hold a gradient
    ///
    /// # Returns
    ///
    /// - `usize` - The entry count
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no parameter holds a gradient
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when the store is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Every gradient of the store, by address
    ///
    /// # Returns
    ///
    /// - `impl Iterator` - The pairs, sorted by layer position and then by parameter name
    pub fn iter(&self) -> impl Iterator<Item = (ParamId, &Tensor)> {
        self.map.iter().map(|(id, grad)| (*id, grad))
    }

    /// Adds a gradient to the address, and sums it with what the address already holds
    ///
    /// The sum is what makes 1 layer that several positions of a model share correct. Each
    /// position runs its own backward pass and gives its own gradient, and the parameter needs
    /// the total
    ///
    /// # Parameters
    ///
    /// - `id` - The address of the parameter
    /// - `grad` - The gradient to add
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the store holds the sum
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the address already holds a gradient of another shape
    fn add(&mut self, id: ParamId, grad: Tensor) -> Result<(), Error> {
        match self.map.get_mut(&id) {
            Some(total) => {
                if total.shape() != grad.shape() {
                    return Err(Error::invalid_input(format!(
                        "the gradient of parameter `{}` of layer {} has shape {:?}, and the \
                         gradient already in the store has shape {:?}",
                        id.name,
                        id.scope,
                        grad.shape(),
                        total.shape()
                    )));
                }
                *total += &grad;
                Ok(())
            }
            None => {
                self.map.insert(id, standard(grad));
                Ok(())
            }
        }
    }
}

/// Gives the tensor the standard memory order, so a parameter update can read it as 1 slice
///
/// The array comes back untouched when it already has that order
fn standard(grad: Tensor) -> Tensor {
    if grad.is_standard_layout() {
        grad
    } else {
        grad.as_standard_layout().into_owned()
    }
}

/// The non-trainable state of 1 layer that a forward pass proposed to change
///
/// [`Sequential`](crate::neural_network::sequential::Sequential) hands this to
/// [`LayerBase::apply_state`](crate::neural_network::traits::LayerBase::apply_state) after the
/// forward pass, and the layer moves each value it recognizes into its own storage
pub struct StateSlot<'a> {
    /// The state channel of the whole context
    states: &'a mut AHashMap<(LayerId, &'static str), Slot>,
    /// The layer whose state this view reaches
    owner: LayerId,
}

impl StateSlot<'_> {
    /// Removes the named value of this layer and gives it back
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the value, such as `"moving_mean"`
    ///
    /// # Type Parameters
    ///
    /// - `T` - The type the layer stores the value as
    ///
    /// # Returns
    ///
    /// - `Option<T>` - The value, or `None` when the forward pass proposed no change
    pub fn take<T: Any + Send + Sync>(&mut self, name: &'static str) -> Option<T> {
        let slot = self.states.remove(&(self.owner, name))?;
        match slot.downcast::<T>() {
            Ok(value) => Some(*value),
            Err(slot) => {
                self.states.insert((self.owner, name), slot);
                None
            }
        }
    }
}

/// Everything 1 pass of a model needs to carry between its layers
///
/// See the [module documentation](self) for the 4 channels
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::layers::{Activation, Dense};
/// use rustyml::neural_network::traits::{ParamId, UnaryLayer};
/// use rustyml::neural_network::{Ctx, Shape};
///
/// let mut layer = Dense::new(3, Activation::ReLU).unwrap();
/// layer.build(&Shape::known(&[2, 4])).unwrap();
///
/// // A training pass writes a cache, so the backward pass has what it needs
/// let mut ctx = Ctx::training();
/// let output = layer.forward(&Array::ones((2, 4)).into_dyn(), &mut ctx).unwrap();
/// let grad_input = layer.backward(&Array::ones(output.raw_dim()), &mut ctx).unwrap();
/// assert_eq!(grad_input.shape(), &[2, 4]);
///
/// // The kernel gradient is in the store, and not in the layer
/// assert!(ctx.grads().get(ParamId::new(0, "kernel")).is_some());
///
/// // An inference pass writes no cache at all
/// let mut ctx = Ctx::inference();
/// layer.forward(&Array::ones((2, 4)).into_dyn(), &mut ctx).unwrap();
/// assert_eq!(ctx.pending_caches(), 0);
/// ```
#[derive(Default)]
pub struct Ctx {
    /// Whether the pass trains the model
    training: bool,
    /// The layer that the gradient channel and the state channel belong to
    owner: LayerId,
    /// The call of that layer that the cache channel belongs to
    call: CallId,
    /// 1 stack of caches per CALL, in the order the forward pass pushed them
    ///
    /// The key is the call and not the layer. A branch of a model that never reaches the loss
    /// leaves its cache behind. A stack shared with another call of the same layer would then
    /// hand that stale cache to the wrong backward pass
    caches: AHashMap<CallId, Vec<CacheSlot>>,
    /// The non-trainable values that the forward pass proposed to change
    states: AHashMap<(LayerId, &'static str), Slot>,
    /// Every parameter gradient of the pass
    grads: Grads,
}

impl Ctx {
    /// Builds a context for a training pass
    ///
    /// A layer writes its cache and its state changes in this mode
    ///
    /// # Returns
    ///
    /// - `Ctx` - An empty context whose training flag is set
    pub fn training() -> Self {
        Self {
            training: true,
            ..Self::default()
        }
    }

    /// Builds a context for an inference pass
    ///
    /// A layer writes no cache and no state change in this mode, and every mode-dependent
    /// layer takes its inference behavior. The context therefore stays empty, which is what
    /// lets a caller run inference through a shared model
    ///
    /// # Returns
    ///
    /// - `Ctx` - An empty context whose training flag is clear
    pub fn inference() -> Self {
        Self::default()
    }

    /// Whether the pass trains the model
    ///
    /// # Returns
    ///
    /// - `bool` - `true` during training
    #[inline]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// The layer that the gradient channel and the state channel belong to
    ///
    /// # Returns
    ///
    /// - `LayerId` - The position of the layer
    #[inline]
    pub fn owner(&self) -> LayerId {
        self.owner
    }

    /// The call of that layer that the cache channel belongs to
    ///
    /// # Returns
    ///
    /// - `CallId` - The position of the call
    #[inline]
    pub fn call(&self) -> CallId {
        self.call
    }

    /// Points every channel at 1 layer that the model calls once, and gives back the layer
    /// they left
    ///
    /// A model calls this before it calls a layer. The caches, the state, and the gradients of
    /// that call then reach the address of that layer. A caller that drives 1 layer by hand
    /// never calls it, and everything lands at layer 0
    ///
    /// # Parameters
    ///
    /// - `owner` - The position of the layer that the model is about to call
    ///
    /// # Returns
    ///
    /// - `LayerId` - The position that the channels pointed at before the call
    #[inline]
    pub fn set_owner(&mut self, owner: LayerId) -> LayerId {
        self.call = owner;
        std::mem::replace(&mut self.owner, owner)
    }

    /// Points the channels at 1 call of 1 layer
    ///
    /// A model that holds a layer arena calls this, because it can reach 1 layer from several
    /// positions. The gradients and the state of every such position belong to the layer, and
    /// the cache of each belongs to the position
    ///
    /// # Parameters
    ///
    /// - `owner` - The layer that the model is about to call
    /// - `call` - The position of this call of that layer
    #[inline]
    pub fn set_position(&mut self, owner: LayerId, call: CallId) {
        self.owner = owner;
        self.call = call;
    }

    /// Parks a value that the backward pass of this call needs
    ///
    /// The caches of 1 call form a stack, and [`pop_cache`](Ctx::pop_cache) takes the newest
    /// first. The stack belongs to the call alone, so a layer that a model reaches from
    /// several positions keeps 1 stack per position. A branch of a model that never reaches
    /// the loss leaves its cache behind and moves no other call
    ///
    /// # Parameters
    ///
    /// - `layer` - The type name of the layer that parks the value
    /// - `cache` - The value to park
    ///
    /// # Type Parameters
    ///
    /// - `T` - The type the layer parks and takes back
    pub fn push_cache<T: Any + Send + Sync>(&mut self, layer: &'static str, cache: T) {
        debug_assert!(
            self.training,
            "a layer must write no cache in an inference pass"
        );
        self.caches.entry(self.call).or_default().push(CacheSlot {
            layer,
            value: Box::new(cache),
        });
    }

    /// Takes back the newest value that this call parked
    ///
    /// # Parameters
    ///
    /// - `layer` - The type name of the layer, which must match the name of the push
    ///
    /// # Type Parameters
    ///
    /// - `T` - The type the layer parked
    ///
    /// # Returns
    ///
    /// - `Result<T, Error>` - The parked value
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If the call parked nothing
    /// - `Error::Computation` - If the newest value of this call came from another layer, or
    ///   if it holds another type. Both mean that 2 layers share 1 call, or that a layer
    ///   parked 1 type and took back another
    pub fn pop_cache<T: Any + Send + Sync>(&mut self, layer: &'static str) -> Result<T, Error> {
        let stack = self
            .caches
            .get_mut(&self.call)
            .ok_or_else(|| Error::forward_pass_not_run(layer))?;
        let slot = stack
            .pop()
            .ok_or_else(|| Error::forward_pass_not_run(layer))?;
        if slot.layer != layer {
            let found = slot.layer;
            stack.push(slot);
            return Err(Error::computation(format!(
                "the cache of call {} came from layer `{found}`, and layer `{layer}` asked for \
                 it. Give each layer its own position with `Ctx::set_position`",
                self.call
            )));
        }
        match slot.value.downcast::<T>() {
            Ok(cache) => Ok(*cache),
            Err(value) => {
                stack.push(CacheSlot { layer, value });
                Err(Error::computation(format!(
                    "layer `{layer}` parked a cache of another type than the one it asked for"
                )))
            }
        }
    }

    /// How many parked values no backward pass has taken back
    ///
    /// The count is 0 after a full training step of a stack of layers. A model that holds a
    /// branch which never reaches the loss leaves 1 cache per call of that branch
    ///
    /// # Returns
    ///
    /// - `usize` - The total over every call
    pub fn pending_caches(&self) -> usize {
        self.caches.values().map(Vec::len).sum()
    }

    /// The value this layer holds for the name, when the forward pass already proposed one
    ///
    /// A layer reads the channel first and falls back to its own storage. That is what makes
    /// 2 calls of 1 shared layer compose: the second call reads what the first call wrote
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the value
    ///
    /// # Type Parameters
    ///
    /// - `T` - The type the layer stores the value as
    ///
    /// # Returns
    ///
    /// - `Option<&T>` - The proposed value, or `None` when the pass proposed none
    pub fn state<T: Any + Send + Sync>(&self, name: &'static str) -> Option<&T> {
        self.states.get(&(self.owner, name))?.downcast_ref::<T>()
    }

    /// Removes the value this layer holds for the name and gives it back
    ///
    /// A layer whose state is expensive to copy takes it, changes it, and puts it back. The
    /// random stream of a dropout layer works this way
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the value
    ///
    /// # Type Parameters
    ///
    /// - `T` - The type the layer stores the value as
    ///
    /// # Returns
    ///
    /// - `Option<T>` - The proposed value, or `None` when the pass proposed none
    pub fn take_state<T: Any + Send + Sync>(&mut self, name: &'static str) -> Option<T> {
        StateSlot {
            states: &mut self.states,
            owner: self.owner,
        }
        .take(name)
    }

    /// Proposes a new value of the named non-trainable state of this layer
    ///
    /// The value stays in the context until the model applies it with
    /// [`LayerBase::apply_state`](crate::neural_network::traits::LayerBase::apply_state)
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the value
    /// - `value` - The new value
    ///
    /// # Type Parameters
    ///
    /// - `T` - The type the layer stores the value as
    pub fn set_state<T: Any + Send + Sync>(&mut self, name: &'static str, value: T) {
        debug_assert!(
            self.training,
            "a layer must change no state in an inference pass"
        );
        self.states.insert((self.owner, name), Box::new(value));
    }

    /// How many proposed state changes no layer has taken back
    ///
    /// The count is 0 after a full pass, because a model applies the state of every layer it
    /// calls. A value left here is a defect: the layer that wrote it did not take it back.
    /// Its running statistics or its random stream never moved
    ///
    /// # Returns
    ///
    /// - `usize` - The total over every layer
    pub fn pending_states(&self) -> usize {
        self.states.len()
    }

    /// Whether the pass proposed any state change for the layer
    ///
    /// # Parameters
    ///
    /// - `owner` - The position of the layer
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when the layer has something to apply
    pub fn has_state(&self, owner: LayerId) -> bool {
        self.states.keys().any(|(layer, _)| *layer == owner)
    }

    /// A view of the state channel of 1 layer, to move the values into the layer
    ///
    /// # Parameters
    ///
    /// - `owner` - The position of the layer
    ///
    /// # Returns
    ///
    /// - `StateSlot` - The view
    pub fn state_slot(&mut self, owner: LayerId) -> StateSlot<'_> {
        StateSlot {
            states: &mut self.states,
            owner,
        }
    }

    /// Adds the gradient of 1 named parameter of this layer to the store
    ///
    /// The store sums, so a layer that runs twice in 1 pass gives the total
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the parameter, such as `"kernel"`
    /// - `grad` - The gradient, at the shape of the parameter
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the store holds the gradient
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the address already holds a gradient of another shape
    pub fn add_grad(&mut self, name: &'static str, grad: Tensor) -> Result<(), Error> {
        self.grads.add(ParamId::new(self.owner, name), grad)
    }

    /// Every parameter gradient the pass produced
    ///
    /// # Returns
    ///
    /// - `&Grads` - The gradient store
    #[inline]
    pub fn grads(&self) -> &Grads {
        &self.grads
    }

    /// Empties the gradient store, and gives back what it held
    ///
    /// The store SUMS, so a caller that drives several training steps against 1 context must
    /// empty it between the steps. Otherwise step 2 updates every parameter by the total of
    /// step 1 and step 2. A model builds a new context for every step and never needs this
    ///
    /// # Returns
    ///
    /// - `Grads` - Every gradient the store held
    pub fn take_grads(&mut self) -> Grads {
        std::mem::take(&mut self.grads)
    }
}

impl std::fmt::Debug for Ctx {
    /// A cache and a state value are both `dyn Any`, so the report counts them and shows no
    /// value
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ctx")
            .field("training", &self.training)
            .field("owner", &self.owner)
            .field("call", &self.call)
            .field("caches", &self.pending_caches())
            .field("states", &self.states.len())
            .field("grads", &self.grads.len())
            .finish()
    }
}

/// Unit tests for the 4 channels of the context
#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::regularization::dropout::dropout::Dropout;
    use crate::neural_network::traits::UnaryLayer;
    use crate::neural_network::{Shape, Tensor};

    /// A cache is addressed by the call, and 2 layers that share 1 call would otherwise cross
    /// their caches. Most layers park a plain tensor or a plain shape, so the type alone
    /// separates almost nothing. The name of the layer is what makes the take an error
    #[test]
    fn a_cache_of_another_layer_is_refused() {
        let mut ctx = Ctx::training();
        ctx.push_cache("Dense", vec![2_usize, 3]);

        let taken = ctx.pop_cache::<Vec<usize>>("Flatten");
        let message = match taken {
            Ok(_) => panic!("the cache of another layer must not come back"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("Dense"), "{message}");
        assert!(message.contains("Flatten"), "{message}");

        // The refusal leaves the stack as it was, so the owner still finds its cache
        assert_eq!(ctx.pop_cache::<Vec<usize>>("Dense").unwrap(), vec![2, 3]);
    }

    /// A layer that parks 1 type and takes back another is a defect of that layer. The message
    /// says so instead of reporting a missing forward pass
    #[test]
    fn a_cache_of_another_type_is_refused() {
        let mut ctx = Ctx::training();
        ctx.push_cache("Dense", vec![2_usize, 3]);

        let message = match ctx.pop_cache::<Tensor>("Dense") {
            Ok(_) => panic!("a cache of another type must not come back"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("another type"), "{message}");
    }

    /// A backward pass with no forward pass behind it reports exactly that
    #[test]
    fn an_empty_stack_reports_a_missing_forward_pass() {
        let mut ctx = Ctx::training();
        let message = match ctx.pop_cache::<Tensor>("Dense") {
            Ok(_) => panic!("an empty context holds no cache"),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("forward pass has not been run"),
            "{message}"
        );
    }

    /// `forward_mut` completes the pass of a hand-driven layer, so its random stream advances
    ///
    /// Without that step the stream never moves, and every call of a dropout layer draws the
    /// same mask. `forward` takes `&self` and cannot move the stream, which is what makes 2
    /// pure passes agree
    #[test]
    fn a_hand_driven_pass_advances_the_random_stream() {
        let input = Tensor::ones([4, 8].as_slice());

        let mut advancing = Dropout::new(0.5).unwrap().with_random_state(7);
        let mut ctx = Ctx::training();
        let first = advancing.forward_mut(&input, &mut ctx).unwrap();
        assert_eq!(ctx.pending_states(), 0, "the state must reach the layer");
        let mut ctx = Ctx::training();
        let second = advancing.forward_mut(&input, &mut ctx).unwrap();
        assert_ne!(first, second, "the random stream did not advance");

        let mut pure = Dropout::new(0.5).unwrap().with_random_state(7);
        pure.build(&Shape::known(&[4, 8])).unwrap();
        let mut ctx = Ctx::training();
        let one = pure.forward(&input, &mut ctx).unwrap();
        let mut ctx = Ctx::training();
        let two = pure.forward(&input, &mut ctx).unwrap();
        assert_eq!(one, two, "a pure pass must not move the stream");
        assert_eq!(
            one, first,
            "the first draw of the 2 layers is the same draw"
        );
    }

    /// The store sums, so a caller that reuses 1 context across steps must empty it
    #[test]
    fn take_grads_empties_the_store() {
        let mut ctx = Ctx::training();
        ctx.add_grad("kernel", Tensor::ones([2].as_slice()))
            .unwrap();
        ctx.add_grad("kernel", Tensor::ones([2].as_slice()))
            .unwrap();

        let taken = ctx.take_grads();
        assert_eq!(taken.len(), 1);
        assert_eq!(
            taken.get(ParamId::new(0, "kernel")).unwrap(),
            &Tensor::from_elem([2].as_slice(), 2.0),
            "the store sums a gradient that arrives twice"
        );
        assert!(ctx.grads().is_empty(), "the store is empty after the take");
    }
}
