//! A model whose layers form a directed graph, rather than a chain
//!
//! [`Sequential`](crate::neural_network::sequential::Sequential) gives every layer exactly 1
//! input, and that input is the output of the layer before it. A residual connection, a model
//! with 2 towers, a shared encoder, and any model with several inlets or several outlets need
//! more than a chain. [`Graph`](crate::neural_network::graph::Graph) is that model
//!
//! A graph holds a layer ARENA and a node list. A node is a call of 1 layer on the outputs of
//! other nodes, so several nodes can call 1 layer. That is what weight sharing is: 1 set of
//! arrays, read at several positions, and updated by the sum of the gradients of those
//! positions
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array;
//! use rustyml::neural_network::graph::GraphBuilder;
//! use rustyml::neural_network::layers::{Activation, Add, Dense};
//! use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
//! use rustyml::neural_network::optimizers::SGD;
//! use rustyml::neural_network::Shape;
//!
//! // A residual block: the input of the block reaches its output twice
//! let mut builder = GraphBuilder::new();
//! let x = builder.input(Shape::known(&[4, 8]));
//! let hidden = builder.add(Dense::new(8, Activation::ReLU).unwrap(), &[x]);
//! let sum = builder.add(Add::new(), &[x, hidden]);
//! let head = builder.add(Dense::new(2, Activation::Linear).unwrap(), &[sum]);
//! let mut model = builder.build(&[head]).unwrap();
//!
//! model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
//!
//! let inputs = Array::ones((4, 8)).into_dyn();
//! let targets = Array::ones((4, 2)).into_dyn();
//! model.fit(&[&inputs], &[&targets], 3).unwrap();
//!
//! let prediction = model.predict(&[&inputs]).unwrap();
//! assert_eq!(prediction[0].shape(), &[4, 2]);
//! ```

use crate::error::Error;
use crate::math::reduction::det_reduce;
use crate::neural_network::NnError;
use crate::neural_network::Shape;
use crate::neural_network::Tensor;
use crate::neural_network::ctx::{Ctx, Grads, LayerId};
use crate::neural_network::layers::checkpoint::{
    LoadReport, apply, apply_partial, capture, weight_path,
};
use crate::neural_network::sequential::{History, read_checkpoint};
use crate::neural_network::traits::{
    Layer, Loss, Optimizer, ParamId, check_addresses, check_every_gradient_is_claimed,
};
use crate::parallel_gates::sq_sum_f32_parallel_min_elems;
use ahash::AHashMap;
use ndarray::{ArrayViewD, Axis};
use ndarray_rand::rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufWriter, Write};

/// The position of 1 node in the graph that holds it
///
/// A node is 1 call of 1 layer, or 1 inlet of the model. A layer that several nodes call
/// keeps 1 [`LayerId`] and holds 1 set of arrays. Each of its nodes keeps its own `NodeId`
/// and therefore its own cache
pub type NodeId = usize;

/// 1 position of the graph
enum Node {
    /// An inlet of the model. It calls no layer, and the caller supplies its tensor
    Input {
        /// Shape of the tensor that enters here, batch axis first
        shape: Shape,
    },
    /// A call of 1 layer on the outputs of other nodes
    Call {
        /// The layer of the arena that this node calls
        layer: LayerId,
        /// The nodes whose outputs enter the call, in the order the layer takes them
        inputs: Vec<NodeId>,
    },
}

impl Node {
    /// The nodes whose outputs enter this one, which is empty for an inlet
    fn inputs(&self) -> &[NodeId] {
        match self {
            Self::Input { .. } => &[],
            Self::Call { inputs, .. } => inputs,
        }
    }
}

/// Collects the layers and the topology of a graph, and builds them against the inlet shapes
///
/// This is the only way to reach a [`Graph`]. [`input`](GraphBuilder::input),
/// [`layer`](GraphBuilder::layer), [`apply`](GraphBuilder::apply) and
/// [`add`](GraphBuilder::add) never fail, so a whole model reads as a sequence of `let`
/// bindings. [`build`](GraphBuilder::build) is the 1 fallible call
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::graph::GraphBuilder;
/// use rustyml::neural_network::layers::{Activation, Concatenate, Dense};
///
/// // 2 inlets, 1 shared tower, and 1 outlet
/// let mut builder = GraphBuilder::new();
/// let left = builder.input(Shape::known(&[4, 5]));
/// let right = builder.input(Shape::known(&[4, 5]));
///
/// let tower = builder.layer(Dense::new(3, Activation::Tanh).unwrap());
/// let a = builder.apply(tower, &[left]);
/// let b = builder.apply(tower, &[right]);
///
/// let joined = builder.add(Concatenate::new(-1), &[a, b]);
/// let model = builder.build(&[joined]).unwrap();
///
/// // 1 shared layer holds 1 set of arrays, whatever number of nodes call it
/// assert_eq!(model.weight_paths(), vec!["0.kernel", "0.bias"]);
/// assert_eq!(model.output_shapes()[0].to_string(), "(4, 6)");
/// ```
#[derive(Default)]
pub struct GraphBuilder {
    /// The arena. 1 entry per layer, whatever number of nodes call it
    layers: Vec<Box<dyn Layer>>,
    /// The topology, in registration order
    nodes: Vec<Node>,
    /// Optional seed governing the fit-time batch shuffle of the built model
    seed: Option<u64>,
}

impl GraphBuilder {
    /// Creates a builder that holds no layer and no node
    ///
    /// # Returns
    ///
    /// - `GraphBuilder` - The empty builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a builder whose model shuffles its batches from `seed`
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed of the fit-time batch shuffle
    ///
    /// # Returns
    ///
    /// - `GraphBuilder` - The empty builder
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            seed: Some(seed),
            ..Self::default()
        }
    }

    /// Adds an inlet of the model, and gives back the node that carries it
    ///
    /// The inlets keep registration order, and that order is the order that
    /// [`Graph::predict`], [`Graph::fit`] and [`Graph::train_batch`] take their tensors in
    ///
    /// # Parameters
    ///
    /// - `shape` - Shape of the tensor that enters here, batch axis first
    ///
    /// # Returns
    ///
    /// - `NodeId` - The inlet node
    pub fn input(&mut self, shape: Shape) -> NodeId {
        self.nodes.push(Node::Input { shape });
        self.nodes.len() - 1
    }

    /// Puts 1 layer in the arena, and gives back its address
    ///
    /// Call [`apply`](GraphBuilder::apply) with that address as many times as the model reads
    /// the layer. Every such node shares 1 set of arrays and 1 sum of gradients
    ///
    /// # Parameters
    ///
    /// - `layer` - The layer to hold
    ///
    /// # Type Parameters
    ///
    /// - `L` - The concrete layer type
    ///
    /// # Returns
    ///
    /// - `LayerId` - The address of the layer in the arena
    pub fn layer<L: 'static + Layer>(&mut self, layer: L) -> LayerId {
        self.layers.push(Box::new(layer));
        self.layers.len() - 1
    }

    /// Calls a layer of the arena on the outputs of other nodes
    ///
    /// # Parameters
    ///
    /// - `layer` - Address of the layer, from [`layer`](GraphBuilder::layer)
    /// - `inputs` - The nodes whose outputs enter the call, in the order the layer takes them
    ///
    /// # Returns
    ///
    /// - `NodeId` - The node that holds the output of the call
    pub fn apply(&mut self, layer: LayerId, inputs: &[NodeId]) -> NodeId {
        self.nodes.push(Node::Call {
            layer,
            inputs: inputs.to_vec(),
        });
        self.nodes.len() - 1
    }

    /// Puts 1 layer in the arena and calls it once, which is the common case
    ///
    /// # Parameters
    ///
    /// - `layer` - The layer to hold and call
    /// - `inputs` - The nodes whose outputs enter the call
    ///
    /// # Type Parameters
    ///
    /// - `L` - The concrete layer type
    ///
    /// # Returns
    ///
    /// - `NodeId` - The node that holds the output of the call
    pub fn add<L: 'static + Layer>(&mut self, layer: L, inputs: &[NodeId]) -> NodeId {
        let id = self.layer(layer);
        self.apply(id, inputs)
    }

    /// Builds every layer against the shapes that reach it, and gives back the model
    ///
    /// The method walks the graph from its inlets, and refuses a topology that cannot run. It
    /// threads the shape of each node into the nodes that read it, and it builds each layer once
    ///
    /// # Parameters
    ///
    /// - `outputs` - The nodes whose tensors the model gives back, in the order it gives them
    ///
    /// # Returns
    ///
    /// - `Result<Graph, Error>` - The built model
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the graph holds no node
    /// - `Error::InvalidInput` - If `outputs` is empty, if an id names no node, if a node reads
    ///   a node at or after its own position, if the graph holds no input node, if the input
    ///   count of a node does not match the arity of its layer, if a node or an inlet reaches
    ///   no output, if a layer refuses the shapes that reach it, or if a layer gives 2 of its
    ///   arrays the same name
    pub fn build(mut self, outputs: &[NodeId]) -> Result<Graph, Error> {
        if self.nodes.is_empty() {
            return Err(Error::NeuralNetwork(NnError::EmptyModel));
        }
        if outputs.is_empty() {
            return Err(Error::invalid_input(
                "a graph model needs at least 1 output node",
            ));
        }
        for &output in outputs {
            if output >= self.nodes.len() {
                return Err(Error::invalid_input(format!(
                    "node {output} is named as an output, and the graph holds {} node(s)",
                    self.nodes.len()
                )));
            }
        }
        for (id, node) in self.nodes.iter().enumerate() {
            for &input in node.inputs() {
                if input >= self.nodes.len() {
                    return Err(Error::invalid_input(format!(
                        "node {id} reads node {input}, and the graph holds {} node(s)",
                        self.nodes.len()
                    )));
                }
                if input >= id {
                    return Err(Error::invalid_input(format!(
                        "node {id} reads node {input}, which the graph holds at a later \
                         position. A node reads only the nodes that already exist"
                    )));
                }
            }
        }

        let inputs: Vec<NodeId> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node, Node::Input { .. }))
            .map(|(id, _)| id)
            .collect();
        if inputs.is_empty() {
            return Err(Error::invalid_input(
                "a graph model needs at least 1 input node",
            ));
        }

        // Every node an output reaches, found by walking the graph backwards from the outputs.
        // A node reads only earlier nodes, so the position order is already a topological one
        let mut reaches_output = vec![false; self.nodes.len()];
        for &output in outputs {
            reaches_output[output] = true;
        }
        for id in (0..self.nodes.len()).rev() {
            if reaches_output[id] {
                for &input in self.nodes[id].inputs() {
                    reaches_output[input] = true;
                }
            }
        }
        if let Some(dead) = reaches_output.iter().position(|reached| !reached) {
            return Err(Error::invalid_input(format!(
                "node {dead} reaches no output of the model. Name it as an output, or leave it \
                 out of the graph"
            )));
        }

        let order: Vec<NodeId> = (0..self.nodes.len()).collect();

        let mut inlet_of: Vec<Option<NodeId>> = vec![None; self.nodes.len()];
        for (position, &id) in inputs.iter().enumerate() {
            inlet_of[id] = Some(position);
        }

        // Thread the shapes, and build each layer on the first node that reaches it
        let mut node_shapes: Vec<Shape> = vec![Shape::new(Vec::new()); self.nodes.len()];
        for &id in &order {
            match &self.nodes[id] {
                Node::Input { shape } => node_shapes[id] = shape.clone(),
                Node::Call { layer, inputs } => {
                    let (layer, inputs) = (*layer, inputs.clone());
                    let shapes: Vec<Shape> = inputs
                        .iter()
                        .map(|&from| node_shapes[from].clone())
                        .collect();
                    let held = &mut self.layers[layer];
                    let layer_type = held.layer_type().to_string();
                    held.arity()
                        .check(&layer_type, shapes.len())
                        .map_err(|source| node_refusal(id, layer, &layer_type, source))?;
                    held.build_many(&shapes)
                        .map_err(|source| node_refusal(id, layer, &layer_type, source))?;
                    node_shapes[id] = held
                        .compute_output_shape_many(&shapes)
                        .map_err(|source| node_refusal(id, layer, &layer_type, source))?;
                }
            }
        }

        // The arrays are real now, so every roster is the one the model will address. The walk
        // is the arena, because the arena index is the layer half of every address. A layer that
        // several nodes share holds 1 entry, so it is checked once
        for (index, layer) in self.layers.iter_mut().enumerate() {
            check_addresses(index, &mut **layer)?;
        }

        Ok(Graph {
            layers: self.layers,
            nodes: self.nodes,
            order,
            inputs,
            outputs: outputs.to_vec(),
            node_shapes,
            inlet_of,
            optimizer: None,
            losses: Vec::new(),
            loss_weights: Vec::new(),
            seed: self.seed,
        })
    }
}

/// Names the node that refused a shape during a graph build
#[cold]
fn node_refusal(node: NodeId, layer: LayerId, layer_type: &str, source: Error) -> Error {
    Error::invalid_input(format!(
        "node {node} (layer {layer}, `{layer_type}`) refused what reaches it: {source}"
    ))
}

/// A model whose layers form a directed graph
///
/// [`GraphBuilder::build`] is the only way to reach one. See the
/// [module documentation](self)
pub struct Graph {
    /// The arena. 1 entry per layer, whatever number of nodes call it
    layers: Vec<Box<dyn Layer>>,
    /// The topology, in registration order
    nodes: Vec<Node>,
    /// A topological order of every node. A node reads only earlier nodes, so this is the
    /// position order
    order: Vec<NodeId>,
    /// The inlets, in registration order
    inputs: Vec<NodeId>,
    /// The outlets, in the order the build received them
    outputs: Vec<NodeId>,
    /// The shape each node produces
    node_shapes: Vec<Shape>,
    /// Entry `n` is the position of node `n` in the inlet list, or `None` when a layer
    /// computes it. A pass reads an inlet from the tensor the caller holds, so it copies no
    /// input batch
    inlet_of: Vec<Option<NodeId>>,
    /// The optimizer that updates every parameter during training
    optimizer: Option<Box<dyn Optimizer>>,
    /// 1 loss per output
    losses: Vec<Box<dyn Loss>>,
    /// 1 weight per output, which the reported loss and every gradient scale by
    loss_weights: Vec<f32>,
    /// Optional seed governing the fit-time batch shuffle
    seed: Option<u64>,
}

impl Graph {
    /// Sets the optimizer and 1 loss for every output of the model
    ///
    /// Every output takes a weight of 1.0. Use
    /// [`with_loss_weights`](Graph::with_loss_weights) to change them
    ///
    /// # Parameters
    ///
    /// - `optimizer` - The optimizer that updates every parameter
    /// - `losses` - 1 loss per output, in the order the build received the outputs
    ///
    /// # Type Parameters
    ///
    /// - `O` - The concrete optimizer type
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - The compiled model
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the loss count does not match the output count
    pub fn compile_many<O: 'static + Optimizer>(
        &mut self,
        optimizer: O,
        losses: Vec<Box<dyn Loss>>,
    ) -> Result<&mut Self, Error> {
        if losses.len() != self.outputs.len() {
            return Err(Error::invalid_input(format!(
                "the model has {} output(s) and the compile gives {} loss(es). A graph model \
                 takes 1 loss per output",
                self.outputs.len(),
                losses.len()
            )));
        }
        self.loss_weights = vec![1.0; losses.len()];
        self.losses = losses;
        self.optimizer = Some(Box::new(optimizer));
        Ok(self)
    }

    /// Sets the optimizer, and 1 loss that every output of the model takes
    ///
    /// A model with 1 output is the common case, and this is the call for it. A model with
    /// several outputs gives every output the same loss, and the reported loss is the sum.
    /// Use [`compile_many`](Graph::compile_many) to give each output its own loss
    ///
    /// # Parameters
    ///
    /// - `optimizer` - The optimizer that updates every parameter
    /// - `loss` - The loss that every output takes
    ///
    /// # Type Parameters
    ///
    /// - `O` - The concrete optimizer type
    /// - `L` - The concrete loss type
    ///
    /// # Returns
    ///
    /// - `&mut Self` - The compiled model
    pub fn compile<O: 'static + Optimizer, L: 'static + Loss + Clone>(
        &mut self,
        optimizer: O,
        loss: L,
    ) -> &mut Self {
        let losses: Vec<Box<dyn Loss>> = self
            .outputs
            .iter()
            .map(|_| Box::new(loss.clone()) as Box<dyn Loss>)
            .collect();
        self.compile_many(optimizer, losses)
            .expect("the loss list is built from the output list, so the 2 counts agree")
    }

    /// Sets the weight of every output in the total loss
    ///
    /// The reported loss is the weighted sum, and the gradient that seeds each output scales
    /// by the same weight
    ///
    /// # Parameters
    ///
    /// - `weights` - 1 weight per output, in the order the build received the outputs
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - The model
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the weight count does not match the output count
    /// - `Error::InvalidParameter` - If a weight is not finite
    pub fn with_loss_weights(&mut self, weights: &[f32]) -> Result<&mut Self, Error> {
        if weights.len() != self.outputs.len() {
            return Err(Error::invalid_input(format!(
                "the model has {} output(s) and the call gives {} weight(s)",
                self.outputs.len(),
                weights.len()
            )));
        }
        if let Some(bad) = weights.iter().find(|weight| !weight.is_finite()) {
            return Err(Error::invalid_parameter(
                "loss weight",
                format!("must be finite, got {bad}"),
            ));
        }
        self.loss_weights = weights.to_vec();
        Ok(self)
    }

    /// The shape of every inlet, in the order the model takes its tensors
    ///
    /// # Returns
    ///
    /// - `Vec<&Shape>` - 1 shape per inlet
    pub fn input_shapes(&self) -> Vec<&Shape> {
        self.inputs
            .iter()
            .map(|&id| &self.node_shapes[id])
            .collect()
    }

    /// The shape of every outlet, in the order the model gives its tensors
    ///
    /// # Returns
    ///
    /// - `Vec<&Shape>` - 1 shape per outlet
    pub fn output_shapes(&self) -> Vec<&Shape> {
        self.outputs
            .iter()
            .map(|&id| &self.node_shapes[id])
            .collect()
    }

    /// Runs every node once, from the inlets, and gives the tensor each node produced
    ///
    /// The method takes `&self`, so it writes no state into any layer. A training pass takes
    /// the state changes out of `ctx` afterwards
    fn forward_values(&self, xs: &[&Tensor], ctx: &mut Ctx) -> Result<Vec<Option<Tensor>>, Error> {
        if xs.len() != self.inputs.len() {
            return Err(Error::invalid_input(format!(
                "the model has {} inlet(s) and the call gives {} tensor(s)",
                self.inputs.len(),
                xs.len()
            )));
        }
        for tensor in xs {
            if tensor.is_empty() {
                return Err(Error::empty_input("input tensor"));
            }
        }

        // An inlet keeps its tensor where the caller holds it. Only a node that a layer
        // computes owns a tensor here, so a pass copies no input batch
        let mut computed: Vec<Option<Tensor>> = vec![None; self.nodes.len()];
        for &id in &self.order {
            let Node::Call { layer, inputs } = &self.nodes[id] else {
                continue;
            };
            let taken: Vec<&Tensor> = inputs
                .iter()
                .map(|from| match self.inlet_of[*from] {
                    Some(position) => xs[position],
                    None => computed[*from]
                        .as_ref()
                        .expect("a node reads only earlier nodes, which already ran"),
                })
                .collect();
            ctx.set_position(*layer, id);
            computed[id] = Some(self.layers[*layer].forward_many(&taken, ctx)?);
        }

        Ok(computed)
    }

    /// The tensor 1 node produced, whether a layer computed it or the caller supplied it
    fn value_of<'v>(
        &self,
        node: NodeId,
        xs: &[&'v Tensor],
        computed: &'v [Option<Tensor>],
    ) -> &'v Tensor {
        match self.inlet_of[node] {
            Some(position) => xs[position],
            None => computed[node]
                .as_ref()
                .expect("every node of a built graph runs"),
        }
    }

    /// Moves the non-trainable state that the pass proposed into every layer that proposed one
    fn apply_state(&mut self, ctx: &mut Ctx) {
        for (id, layer) in self.layers.iter_mut().enumerate() {
            if ctx.has_state(id) {
                layer.apply_state(&mut ctx.state_slot(id));
                debug_assert!(
                    !ctx.has_state(id),
                    "layer {id} proposed a state change and did not take it back"
                );
            }
        }
    }

    /// Runs the model on 1 batch, and gives back the loss it reports
    ///
    /// # Parameters
    ///
    /// - `xs` - 1 tensor per inlet, in registration order
    /// - `ys` - 1 target per outlet, in the order the build received the outputs
    ///
    /// # Returns
    ///
    /// - `Result<f32, Error>` - The weighted sum of the per-output losses
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the model holds no optimizer or no
    ///   loss
    /// - `Error::InvalidInput` - If the tensor counts do not match the model, or if a tensor
    ///   has no batch axis
    /// - `Error::EmptyInput` - If a tensor is empty
    /// - `Error::DimensionMismatch` - If 2 tensors disagree on the sample count
    /// - `Error` - Whatever a layer reports from its forward or backward pass
    pub fn train_batch(&mut self, xs: &[&Tensor], ys: &[&Tensor]) -> Result<f32, Error> {
        self.check_compiled(true)?;
        self.check_targets(ys)?;
        self.sample_count(xs, ys)?;

        let mut ctx = Ctx::training();
        let values = self.forward_values(xs, &mut ctx)?;
        self.apply_state(&mut ctx);

        let mut total = 0.0_f32;
        let mut grads: Vec<Option<Tensor>> = vec![None; self.nodes.len()];
        for (index, &output) in self.outputs.iter().enumerate() {
            let weight = self.loss_weights[index];
            let value = self.value_of(output, xs, &values);
            total += weight * self.losses[index].compute_loss(ys[index], value)?;
            let mut seed = self.losses[index].compute_grad(ys[index], value)?;
            if weight != 1.0 {
                seed *= weight;
            }
            accumulate(&mut grads, output, seed)?;
        }

        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.step();
        }

        // The walk is the position order reversed, so every consumer of a node runs before
        // that node. The fan-in sum of a node is complete when the walk reaches it
        for &id in self.order.iter().rev() {
            let Node::Call { layer, inputs } = &self.nodes[id] else {
                continue;
            };
            let (layer, inputs) = (*layer, inputs.clone());
            let Some(grad) = grads[id].take() else {
                continue;
            };
            ctx.set_position(layer, id);
            let upstream = self.layers[layer].backward_many(&grad, &mut ctx)?;
            if upstream.len() != inputs.len() {
                return Err(Error::computation(format!(
                    "node {id} (layer {layer}, `{}`) reads {} input(s) and gave back {} \
                     gradient(s)",
                    self.layers[layer].layer_type(),
                    inputs.len(),
                    upstream.len()
                )));
            }
            for (&from, gradient) in inputs.iter().zip(upstream) {
                accumulate(&mut grads, from, gradient)?;
            }
        }

        // Every gradient of the pass must reach a parameter. The optimizer walk below skips an
        // address that holds no gradient. A gradient at an address no parameter reads would
        // otherwise vanish without a word
        check_every_gradient_is_claimed(&mut self.layers, ctx.grads())?;

        let global_clipnorm = self
            .optimizer
            .as_ref()
            .and_then(|optimizer| optimizer.global_clipnorm());
        let grad_scale = match global_clipnorm {
            Some(max_norm) => {
                let norm = global_grad_norm(&mut self.layers, ctx.grads());
                if norm.is_finite() && norm > max_norm {
                    max_norm / norm
                } else {
                    1.0
                }
            }
            None => 1.0,
        };

        // The walk is the arena order, and the arena position is the layer half of every
        // parameter address. A layer that several nodes call holds 1 arena entry. Its
        // gradient is the sum of the gradients of those nodes, and it updates once
        if let Some(ref mut optimizer) = self.optimizer {
            for (scope, layer) in self.layers.iter_mut().enumerate() {
                optimizer.update(scope, &mut **layer, ctx.grads(), grad_scale);
            }
        }

        Ok(total)
    }

    /// Trains the model on the whole data set, 1 full batch per epoch
    ///
    /// # Parameters
    ///
    /// - `xs` - 1 tensor per inlet
    /// - `ys` - 1 target per outlet
    /// - `epochs` - How many epochs to run
    ///
    /// # Returns
    ///
    /// - `Result<History, Error>` - 1 loss per epoch, in epoch order
    ///
    /// # Errors
    ///
    /// - `Error` - Whatever [`train_batch`](Graph::train_batch) reports
    pub fn fit(&mut self, xs: &[&Tensor], ys: &[&Tensor], epochs: u32) -> Result<History, Error> {
        self.check_compiled(true)?;
        self.check_targets(ys)?;
        let mut loss = Vec::with_capacity(epochs as usize);
        for _ in 0..epochs {
            loss.push(self.train_batch(xs, ys)?);
        }
        Ok(History::new(loss))
    }

    /// Trains the model on fixed-size batches, and reshuffles every epoch
    ///
    /// # Parameters
    ///
    /// - `xs` - 1 tensor per inlet
    /// - `ys` - 1 target per outlet
    /// - `epochs` - How many epochs to run
    /// - `batch_size` - How many samples 1 step reads
    ///
    /// # Returns
    ///
    /// - `Result<History, Error>` - 1 mean loss per epoch, in epoch order
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `batch_size` is 0
    /// - `Error::DimensionMismatch` - If 2 tensors disagree on the sample count
    /// - `Error` - Whatever [`train_batch`](Graph::train_batch) reports
    pub fn fit_with_batches(
        &mut self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        epochs: u32,
        batch_size: usize,
    ) -> Result<History, Error> {
        self.check_compiled(true)?;
        self.check_targets(ys)?;
        if batch_size == 0 {
            return Err(Error::invalid_parameter("batch_size", "must be at least 1"));
        }
        let samples = self.sample_count(xs, ys)?;

        let mut rng = crate::random::make_rng(self.seed);
        let mut order: Vec<usize> = (0..samples).collect();
        let mut loss = Vec::with_capacity(epochs as usize);

        for _ in 0..epochs {
            order.shuffle(&mut rng);
            let mut total = 0.0_f32;
            let mut steps = 0_u32;
            for chunk in order.chunks(batch_size) {
                let batch_x: Vec<Tensor> = xs.iter().map(|t| gather(t, chunk)).collect();
                let batch_y: Vec<Tensor> = ys.iter().map(|t| gather(t, chunk)).collect();
                let refs_x: Vec<&Tensor> = batch_x.iter().collect();
                let refs_y: Vec<&Tensor> = batch_y.iter().collect();
                total += self.train_batch(&refs_x, &refs_y)?;
                steps += 1;
            }
            loss.push(if steps == 0 {
                0.0
            } else {
                total / steps as f32
            });
        }
        Ok(History::new(loss))
    }

    /// Runs the model in inference mode, and gives 1 tensor per outlet
    ///
    /// The method takes `&self`, so several threads can run it against 1 model
    ///
    /// # Parameters
    ///
    /// - `xs` - 1 tensor per inlet, in registration order
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Tensor>, Error>` - 1 tensor per outlet
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the tensor count does not match the inlet count
    /// - `Error::EmptyInput` - If a tensor is empty
    /// - `Error` - Whatever a layer reports from its forward pass
    pub fn predict(&self, xs: &[&Tensor]) -> Result<Vec<Tensor>, Error> {
        let mut ctx = Ctx::inference();
        let values = self.forward_values(xs, &mut ctx)?;
        Ok(self
            .outputs
            .iter()
            .map(|&output| self.value_of(output, xs, &values).clone())
            .collect())
    }

    /// The loss the model reports on data it does not train on
    ///
    /// # Parameters
    ///
    /// - `xs` - 1 tensor per inlet
    /// - `ys` - 1 target per outlet
    ///
    /// # Returns
    ///
    /// - `Result<f32, Error>` - The weighted sum of the per-output losses
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the model holds no loss
    /// - `Error` - Whatever the forward pass reports
    pub fn evaluate(&self, xs: &[&Tensor], ys: &[&Tensor]) -> Result<f32, Error> {
        self.check_compiled(false)?;
        self.check_targets(ys)?;
        self.sample_count(xs, ys)?;
        let predicted = self.predict(xs)?;
        let mut total = 0.0_f32;
        for (index, value) in predicted.iter().enumerate() {
            total +=
                self.loss_weights[index] * self.losses[index].compute_loss(ys[index], value)?;
        }
        Ok(total)
    }

    /// Every checkpoint path of the model, in arena order
    ///
    /// A path is `<scope>.<name>`: the position of the layer in the arena, and the name the
    /// layer gives the array. A layer that several nodes call holds 1 arena entry, so it holds
    /// 1 set of paths
    ///
    /// # Returns
    ///
    /// - `Vec<String>` - Every path, in file order
    pub fn weight_paths(&self) -> Vec<String> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(scope, layer)| {
                layer
                    .weights()
                    .into_iter()
                    .map(move |entry| weight_path(scope, entry.name))
            })
            .collect()
    }

    /// 1 named array of the model, or `None` when no layer holds that path
    ///
    /// # Parameters
    ///
    /// - `path` - The address, as [`weight_paths`](Graph::weight_paths) gives it
    ///
    /// # Returns
    ///
    /// - `Option<ArrayViewD<'_, f32>>` - A read view of the array
    pub fn weight(&self, path: &str) -> Option<ArrayViewD<'_, f32>> {
        let (scope, name) = path.split_once('.')?;
        let scope: usize = scope.parse().ok()?;
        self.layers.get(scope)?.weight(name)
    }

    /// Writes every array of the model to a file
    ///
    /// # Parameters
    ///
    /// - `path` - Where to write
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the file holds the model
    ///
    /// # Errors
    ///
    /// - `Error::Io` - If the model cannot be serialized, or the file cannot be written
    pub fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        // `capture` borrows the live arrays, so nothing is copied before postcard reads them
        let bytes = postcard::to_allocvec(&capture(&self.layers))?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes)?;
        writer.flush()?;
        Ok(())
    }

    /// Reads every array of the model from a file, and refuses a file that disagrees
    ///
    /// # Parameters
    ///
    /// - `path` - Where to read from
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when every array of the model holds the value of the file
    ///
    /// # Errors
    ///
    /// - `Error::Io` - If the file cannot be read or decoded, or if the file disagrees with
    ///   the model in any way
    pub fn load_from_path(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        let file = read_checkpoint(path)?;
        apply(&mut self.layers, &file)
    }

    /// Reads what the file and the model agree on, and reports the rest
    ///
    /// # Parameters
    ///
    /// - `path` - Where to read from
    ///
    /// # Returns
    ///
    /// - `Result<LoadReport, Error>` - What the load applied, missed, and left unused
    ///
    /// # Errors
    ///
    /// - `Error::Io` - If the file cannot be read or decoded
    pub fn load_partial_from_path(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<LoadReport, Error> {
        let file = read_checkpoint(path)?;
        Ok(apply_partial(&mut self.layers, &file))
    }

    /// Prints the nodes of the model, their layers, and their shapes
    pub fn summary(&self) {
        let mut output = String::from("Model: \"graph\"\n");
        output.push_str(&format!(
            "{:<6} {:<24} {:<22} {:>10}  {}\n",
            "Node", "Layer (type)", "Output Shape", "Param #", "Reads"
        ));
        output.push_str(&"-".repeat(88));
        output.push('\n');

        let mut counted: Vec<bool> = vec![false; self.layers.len()];
        let mut total = 0_usize;
        let mut trainable = 0_usize;
        let mut names: AHashMap<&str, usize> = AHashMap::new();

        for &id in &self.order {
            let (label, params, reads) = match &self.nodes[id] {
                Node::Input { .. } => ("input".to_string(), String::new(), String::new()),
                Node::Call { layer, inputs } => {
                    let kind = self.layers[*layer].layer_type();
                    let count = names.entry(kind).or_insert(0);
                    let name = if *count == 0 {
                        kind.to_lowercase()
                    } else {
                        format!("{}_{}", kind.to_lowercase(), count)
                    };
                    *count += 1;
                    let counts = self.layers[*layer].param_count();
                    let shown = if counted[*layer] {
                        // A shared layer counts once, which is the whole point of sharing
                        "shared".to_string()
                    } else {
                        counted[*layer] = true;
                        total += counts.trainable + counts.non_trainable;
                        trainable += counts.trainable;
                        (counts.trainable + counts.non_trainable).to_string()
                    };
                    (format!("{name} ({kind})"), shown, format!("{inputs:?}"))
                }
            };
            output.push_str(&format!(
                "{:<6} {:<24} {:<22} {:>10}  {}\n",
                id,
                label,
                self.node_shapes[id].to_string(),
                params,
                reads
            ));
        }

        output.push_str(&"-".repeat(88));
        output.push('\n');
        output.push_str(&format!("Total params: {total}\n"));
        output.push_str(&format!("Trainable params: {trainable}\n"));
        output.push_str(&format!("Non-trainable params: {}\n", total - trainable));
        print!("{output}");
    }

    /// Refuses a model that holds no loss, and no optimizer when one is needed
    fn check_compiled(&self, needs_optimizer: bool) -> Result<(), Error> {
        if needs_optimizer && self.optimizer.is_none() {
            return Err(Error::NeuralNetwork(NnError::NotCompiled("optimizer")));
        }
        if self.losses.is_empty() {
            return Err(Error::NeuralNetwork(NnError::NotCompiled("loss function")));
        }
        Ok(())
    }

    /// Refuses a target list that does not match the outlets
    fn check_targets(&self, ys: &[&Tensor]) -> Result<(), Error> {
        if ys.len() != self.outputs.len() {
            return Err(Error::invalid_input(format!(
                "the model has {} output(s) and the call gives {} target(s)",
                self.outputs.len(),
                ys.len()
            )));
        }
        for tensor in ys {
            if tensor.is_empty() {
                return Err(Error::empty_input("target tensor"));
            }
        }
        Ok(())
    }

    /// The sample count that every tensor of a batch call must agree on
    fn sample_count(&self, xs: &[&Tensor], ys: &[&Tensor]) -> Result<usize, Error> {
        let mut samples: Option<usize> = None;
        for tensor in xs.iter().chain(ys.iter()) {
            if tensor.ndim() == 0 {
                return Err(Error::invalid_input(
                    "a batch call needs a tensor with a batch axis, got a rank-0 tensor",
                ));
            }
            let count = tensor.shape()[0];
            match samples {
                None => samples = Some(count),
                Some(held) if held != count => {
                    return Err(Error::dimension_mismatch(held, count));
                }
                Some(_) => {}
            }
        }
        samples.ok_or_else(|| Error::empty_input("input tensor"))
    }
}

/// Adds a gradient to the slot of a node, and sums it with what the slot already holds
fn accumulate(grads: &mut [Option<Tensor>], node: NodeId, gradient: Tensor) -> Result<(), Error> {
    match &mut grads[node] {
        Some(total) => {
            if total.shape() != gradient.shape() {
                return Err(Error::computation(format!(
                    "node {node} receives a gradient of shape {:?} and already holds one of \
                     shape {:?}",
                    gradient.shape(),
                    total.shape()
                )));
            }
            *total += &gradient;
        }
        slot @ None => *slot = Some(gradient),
    }
    Ok(())
}

/// Takes the rows of `tensor` that `rows` names, in that order
fn gather(tensor: &Tensor, rows: &[usize]) -> Tensor {
    tensor.select(Axis(0), rows)
}

/// The global L2 norm of every gradient of the model
///
/// The walk is the arena order, and the parameter order of each layer. The gradient store
/// sorts by address instead. A sum of `f64` squares is not associative, so reducing in
/// store order would move the last bit of the norm
fn global_grad_norm(layers: &mut [Box<dyn Layer>], grads: &Grads) -> f32 {
    let mut sum_sq = 0.0_f64;
    for (scope, layer) in layers.iter_mut().enumerate() {
        for param in layer.parameters_mut() {
            let Some(grad) = grads.get(ParamId::new(scope, param.name)) else {
                continue;
            };
            let grad = grad
                .as_slice()
                .expect("a stored gradient is in the standard memory order");
            sum_sq += det_reduce(
                grad,
                grad.len() >= sq_sum_f32_parallel_min_elems(),
                |block| block.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>(),
                |a, b| a + b,
                0.0,
            );
        }
    }
    sum_sq.sqrt() as f32
}

/// Unit tests that hold the graph executor against the chain it generalizes
#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::{Activation, BatchNormalization, Dense};
    use crate::neural_network::losses::mean_squared_error::MeanSquaredError;
    use crate::neural_network::optimizers::SGD;
    use crate::neural_network::sequential::SequentialBuilder;
    use ndarray::{Array, IxDyn};

    /// Builds a tensor from a pure formula, so the data never moves
    fn data(shape: &[usize]) -> Tensor {
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|i| ((((i * 37) % 101) as f32) - 50.0) / 25.0)
            .collect();
        Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
    }

    /// A graph that holds 1 chain must agree with the sequential model of that chain
    ///
    /// The agreement must hold bit for bit, on the loss of every epoch and on every trained
    /// array. The 2 models take the same weights, because a fixed seed draws the same values in
    /// the same order. A difference here is a difference in the driver and nowhere else
    #[test]
    fn a_chain_graph_agrees_with_the_sequential_model() {
        let x = data(&[6, 4]);
        let y = data(&[6, 2]);

        let mut chain = SequentialBuilder::new_with_seed(11)
            .add(
                Dense::new(5, Activation::ReLU)
                    .unwrap()
                    .with_random_state(3),
            )
            .add(BatchNormalization::new(0.8, 1e-5).unwrap())
            .add(
                Dense::new(2, Activation::Linear)
                    .unwrap()
                    .with_random_state(4),
            )
            .build(&Shape::known(&[6, 4]))
            .unwrap();
        chain.compile(
            SGD::new(0.05, 0.9, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

        let mut builder = GraphBuilder::new_with_seed(11);
        let input = builder.input(Shape::known(&[6, 4]));
        let first = builder.add(
            Dense::new(5, Activation::ReLU)
                .unwrap()
                .with_random_state(3),
            &[input],
        );
        let norm = builder.add(BatchNormalization::new(0.8, 1e-5).unwrap(), &[first]);
        let head = builder.add(
            Dense::new(2, Activation::Linear)
                .unwrap()
                .with_random_state(4),
            &[norm],
        );
        let mut graph = builder.build(&[head]).unwrap();
        graph.compile(
            SGD::new(0.05, 0.9, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

        let chain_history = chain.fit(&x, &y, 4).unwrap();
        let graph_history = graph.fit(&[&x], &[&y], 4).unwrap();
        for (a, b) in chain_history.loss().iter().zip(graph_history.loss()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "the 2 drivers report a different loss"
            );
        }

        assert_eq!(chain.weight_paths(), graph.weight_paths());
        for path in chain.weight_paths() {
            let from_chain = chain.weight(&path).unwrap();
            let from_graph = graph.weight(&path).unwrap();
            for (a, b) in from_chain.iter().zip(from_graph.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "the array {path} differs");
            }
        }

        let chain_prediction = chain.predict(&x).unwrap();
        let graph_prediction = graph.predict(&[&x]).unwrap();
        for (a, b) in chain_prediction.iter().zip(graph_prediction[0].iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "the 2 predictions differ");
        }
    }

    /// A node that no output reaches is refused, because it would leave a cache behind on
    /// every pass with nothing to ever read it
    #[test]
    fn a_node_that_reaches_no_output_is_refused() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(Shape::known(&[2, 3]));
        let live = builder.add(Dense::new(2, Activation::Linear).unwrap(), &[input]);
        let _dead = builder.add(Dense::new(2, Activation::Linear).unwrap(), &[input]);

        let message = match builder.build(&[live]) {
            Ok(_) => panic!("a dead node must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("reaches no output"), "{message}");
    }

    /// A graph with no inlet has nothing to run on
    #[test]
    fn a_graph_with_no_inlet_is_refused() {
        let mut builder = GraphBuilder::new();
        let orphan = builder.apply(0, &[]);
        assert!(builder.build(&[orphan]).is_err());
    }

    /// A built graph serves inference through a shared reference, for the same reason a
    /// sequential model does. A forward pass takes `&self`, and every part of the model is
    /// `Send` and `Sync`
    #[test]
    fn a_built_graph_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Graph>();
        assert_sync::<Graph>();
    }

    /// A shared layer holds 1 arena entry, so it holds 1 set of checkpoint paths and takes 1
    /// update per step
    #[test]
    fn a_shared_layer_holds_one_set_of_paths() {
        let mut builder = GraphBuilder::new();
        let left = builder.input(Shape::known(&[2, 3]));
        let right = builder.input(Shape::known(&[2, 3]));
        let tower = builder.layer(Dense::new(4, Activation::Tanh).unwrap());
        let a = builder.apply(tower, &[left]);
        let _b = builder.apply(tower, &[right]);
        let model = builder.build(&[a, _b]).unwrap();

        assert_eq!(model.weight_paths(), vec!["0.kernel", "0.bias"]);
        assert_eq!(model.output_shapes().len(), 2);
        assert_eq!(model.input_shapes().len(), 2);
    }
}
