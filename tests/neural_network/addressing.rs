//! The address of an array, and the 2 guards that hold a layer to it
//!
//! The name of an array is its address inside its layer. The gradient store, the per-parameter
//! state of the optimizer, and the path of the checkpoint all key on `(layer position, name)`.
//! [`LayerBase::parameters_mut`] and [`LayerBase::weights`] both state in their own
//! documentation that 2 arrays of 1 layer must never share a name. These guards are what
//! enforces that rule.
//!
//! Both guards exist because `Layer` is public and unsealed, so a layer written outside this
//! crate reaches every path here.
//!
//! The wrappers below are the shape that a layer holding other layers takes. A `Bidirectional`
//! holds 2 recurrent children, and both children name an array `kernel`. Without the build
//! guard that model trains on the sum of 2 different parameters, updates both from 1 momentum
//! buffer, and writes 1 checkpoint path where 2 arrays live, and it reports none of it.

use ndarray::{Array, IxDyn};
use rustyml::error::Error;
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::layers::{Activation, Dense};
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::{LayerBase, ParamRef, UnaryLayer, WeightMut, WeightRef};
use rustyml::neural_network::{Ctx, Shape, Tensor};

/// A layer that holds 2 layers and passes their arrays on under the names they gave
///
/// This is the mistake the build guard refuses. Both children name an array `kernel`, so the
/// wrapper offers `kernel` twice.
#[derive(Debug)]
struct Twin {
    /// The first child
    left: Dense,
    /// The second child
    right: Dense,
}

impl Twin {
    /// A wrapper over 2 dense layers of the same width
    fn new(units: usize) -> Self {
        Self {
            left: Dense::new(units, Activation::Linear)
                .expect("units is above 0")
                .with_random_state(1),
            right: Dense::new(units, Activation::Linear)
                .expect("units is above 0")
                .with_random_state(2),
        }
    }
}

impl LayerBase for Twin {
    fn layer_type(&self) -> &str {
        "Twin"
    }

    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(
            self.left.param_count().trainable + self.right.param_count().trainable,
        )
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let mut all = self.left.parameters_mut();
        all.extend(self.right.parameters_mut());
        all
    }

    fn weights(&self) -> Vec<WeightRef<'_>> {
        let mut all = self.left.weights();
        all.extend(self.right.weights());
        all
    }

    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        let mut all = self.left.weights_mut();
        all.extend(self.right.weights_mut());
        all
    }

    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        self.left.known_input_shapes()
    }

    fn is_built(&self) -> bool {
        self.left.is_built() && self.right.is_built()
    }
}

impl UnaryLayer for Twin {
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.left.build(input)?;
        self.right.build(input)
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(self.left.forward(input, ctx)? + self.right.forward(input, ctx)?)
    }

    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // The cache of 1 call is 1 stack, so the children take theirs back in reverse order
        let right = self.right.backward(grad, ctx)?;
        let left = self.left.backward(grad, ctx)?;
        Ok(left + right)
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.left.compute_output_shape(input)
    }
}

/// A layer whose roster and whose gradient disagree by 1 name
///
/// This is the mistake the gradient guard refuses. The layer offers `kernel` and adds its
/// gradient under `kernal`, which is the typo a layer that spells a name twice can make.
#[derive(Debug)]
struct Misaddressed {
    /// The layer that does the work
    inner: Dense,
}

impl LayerBase for Misaddressed {
    fn layer_type(&self) -> &str {
        "Misaddressed"
    }

    fn param_count(&self) -> ParamCounts {
        self.inner.param_count()
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        self.inner.parameters_mut()
    }

    fn weights(&self) -> Vec<WeightRef<'_>> {
        self.inner.weights()
    }

    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        self.inner.weights_mut()
    }

    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        self.inner.known_input_shapes()
    }

    fn is_built(&self) -> bool {
        self.inner.is_built()
    }
}

impl UnaryLayer for Misaddressed {
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.inner.build(input)
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        self.inner.forward(input, ctx)
    }

    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_grad = self.inner.backward(grad, ctx)?;
        // Park a second gradient at an address the roster does not hold
        ctx.add_grad("kernal", Array::zeros(IxDyn(&[3, 2])))?;
        Ok(input_grad)
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.inner.compute_output_shape(input)
    }
}

/// Builds a tensor from a pure formula
fn data(shape: &[usize]) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| (i % 7) as f32 - 3.0).collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
}

#[test]
fn a_sequential_build_refuses_2_arrays_under_1_name() {
    let Err(error) = SequentialBuilder::new()
        .add(Twin::new(2))
        .build(&Shape::known(&[3, 4]))
    else {
        panic!("2 arrays under 1 name is not an address");
    };
    let text = format!("{error}");
    assert!(
        text.contains("layer 0"),
        "the message names the layer: {text}"
    );
    assert!(text.contains("Twin"), "the message names the type: {text}");
    assert!(
        text.contains("kernel"),
        "the message names the array: {text}"
    );
    assert!(
        text.contains("prefix"),
        "the message says what a layer that holds layers must do: {text}"
    );
}

/// A graph model refuses the same layer, and the walk is the arena
#[test]
fn a_graph_build_refuses_2_arrays_under_1_name() {
    use rustyml::neural_network::graph::GraphBuilder;

    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[3, 4]));
    let out = builder.add(Twin::new(2), &[input]);
    let Err(error) = builder.build(&[out]) else {
        panic!("2 arrays under 1 name is not an address");
    };
    let text = format!("{error}");
    assert!(text.contains("Twin"), "the message names the type: {text}");
    assert!(
        text.contains("kernel"),
        "the message names the array: {text}"
    );
}

/// A training step refuses a gradient that reaches no parameter of the model
#[test]
fn a_training_step_refuses_an_unclaimed_gradient() {
    let mut model = SequentialBuilder::new()
        .add(Misaddressed {
            inner: Dense::new(2, Activation::Linear)
                .unwrap()
                .with_random_state(7),
        })
        .build(&Shape::known(&[3, 3]))
        .expect("1 array under 1 name is a valid address");
    model.compile(
        SGD::new(0.1, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let error = model
        .fit(&data(&[3, 3]), &data(&[3, 2]), 1)
        .expect_err("a gradient that reaches no parameter is not a training step");
    let text = format!("{error}");
    assert!(
        text.contains("kernal"),
        "the message names the address nothing reads: {text}"
    );
    assert!(
        text.contains("0."),
        "the message gives the address in path form: {text}"
    );
}

/// The 2 guards must cost a correct model nothing, so this is the control
#[test]
fn a_model_whose_names_are_addresses_still_trains() {
    let mut model = SequentialBuilder::new()
        .add(
            Dense::new(3, Activation::ReLU)
                .unwrap()
                .with_random_state(1),
        )
        .add(
            Dense::new(2, Activation::Linear)
                .unwrap()
                .with_random_state(2),
        )
        .build(&Shape::known(&[3, 4]))
        .expect("every array holds its own name");
    model.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let history = model
        .fit(&data(&[3, 4]), &data(&[3, 2]), 2)
        .expect("a correct model is not refused");
    assert_eq!(history.loss().len(), 2);
}
