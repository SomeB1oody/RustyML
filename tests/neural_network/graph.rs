//! Tests for [`Graph`]: the topology it refuses, the fan-in gradient, and weight sharing
//!
//! The sharpest tests here compare a graph against a sequential model that computes the same
//! function. A graph whose fan-in accumulator drops a contribution still runs, still trains,
//! and gives a plausible answer. An equality against a model with no fan-in is what catches it.

use ndarray::{Array, IxDyn};
use rustyml::error::Error;
use rustyml::neural_network::Shape;
use rustyml::neural_network::graph::GraphBuilder;
use rustyml::neural_network::layers::{Activation, Add, Concatenate, Dense, Rescaling, Subtract};
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::{Tensor, traits::Layer};

/// Builds a tensor from a pure formula, so the data never moves
fn data(shape: &[usize]) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| ((((i * 37) % 101) as f32) - 50.0) / 25.0)
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
}

/// A node read twice must receive the sum of both gradients
///
/// `Add(h, h)` is `2 * h`, and `Rescaling(2.0)` is the same function with no fan-in at all.
/// The 2 models must therefore agree on every loss and on every trained array. A fan-in
/// accumulator that kept the last contribution instead of the sum would halve the gradient
/// that reaches the layer under the node. The trained kernel would then drift on the first step
#[test]
fn a_node_read_twice_receives_the_sum_of_both_gradients() {
    let x = data(&[6, 4]);
    let y = data(&[6, 3]);

    let mut chain = SequentialBuilder::new()
        .add(
            Dense::new(3, Activation::Tanh)
                .unwrap()
                .with_random_state(5),
        )
        .add(Rescaling::new(2.0))
        .build(&Shape::known(&[6, 4]))
        .unwrap();
    chain.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[6, 4]));
    let hidden = builder.add(
        Dense::new(3, Activation::Tanh)
            .unwrap()
            .with_random_state(5),
        &[input],
    );
    let doubled = builder.add(Add::new(), &[hidden, hidden]);
    let mut graph = builder.build(&[doubled]).unwrap();
    graph.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let chain_history = chain.fit(&x, &y, 5).unwrap();
    let graph_history = graph.fit(&[&x], &[&y], 5).unwrap();
    for (epoch, (a, b)) in chain_history
        .loss()
        .iter()
        .zip(graph_history.loss())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "epoch {epoch}: the fan-in sum differs from the equivalent scale"
        );
    }
    for path in ["0.kernel", "0.bias"] {
        let from_chain = chain.weight(path).unwrap();
        let from_graph = graph.weight(path).unwrap();
        for (a, b) in from_chain.iter().zip(from_graph.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "the array {path} differs");
        }
    }
}

/// A layer that 2 nodes call is updated by the sum of the gradients of those nodes
///
/// The graph reads 1 shared layer twice on the same input and adds the 2 results, which is
/// again `2 * h`. A shared layer that took only 1 of its 2 gradients would train differently
#[test]
fn a_shared_layer_takes_the_sum_of_the_gradients_of_its_nodes() {
    let x = data(&[5, 3]);
    let y = data(&[5, 2]);

    let mut chain = SequentialBuilder::new()
        .add(
            Dense::new(2, Activation::Linear)
                .unwrap()
                .with_random_state(9),
        )
        .add(Rescaling::new(2.0))
        .build(&Shape::known(&[5, 3]))
        .unwrap();
    chain.compile(
        SGD::new(0.02, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[5, 3]));
    let tower = builder.layer(
        Dense::new(2, Activation::Linear)
            .unwrap()
            .with_random_state(9),
    );
    let first = builder.apply(tower, &[input]);
    let second = builder.apply(tower, &[input]);
    let joined = builder.add(Add::new(), &[first, second]);
    let mut graph = builder.build(&[joined]).unwrap();
    graph.compile(
        SGD::new(0.02, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    // 1 shared layer holds 1 arena entry, so it holds 1 set of paths and takes 1 update
    assert_eq!(graph.weight_paths(), vec!["0.kernel", "0.bias"]);

    let chain_history = chain.fit(&x, &y, 4).unwrap();
    let graph_history = graph.fit(&[&x], &[&y], 4).unwrap();
    for (a, b) in chain_history.loss().iter().zip(graph_history.loss()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "the shared layer trained differently"
        );
    }
}

/// A residual connection reaches the output through 2 paths of different length
#[test]
fn a_residual_block_trains_and_predicts() {
    let x = data(&[4, 6]);
    let y = data(&[4, 6]);

    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[4, 6]));
    let hidden = builder.add(
        Dense::new(6, Activation::ReLU)
            .unwrap()
            .with_random_state(2),
        &[input],
    );
    let sum = builder.add(Add::new(), &[input, hidden]);
    let mut model = builder.build(&[sum]).unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let history = model.fit(&[&x], &[&y], 6).unwrap();
    assert!(
        history.loss()[5] < history.loss()[0],
        "the residual block did not train: {:?}",
        history.loss()
    );
    assert_eq!(model.predict(&[&x]).unwrap()[0].shape(), &[4, 6]);
}

/// A model with 2 inlets and 2 outlets takes 1 loss per outlet, and weights them
#[test]
fn a_model_with_several_inlets_and_outlets_trains() {
    let left = data(&[4, 3]);
    let right = data(&[4, 3]);
    let target_a = data(&[4, 2]);
    let target_b = data(&[4, 6]);

    let mut builder = GraphBuilder::new();
    let a = builder.input(Shape::known(&[4, 3]));
    let b = builder.input(Shape::known(&[4, 3]));
    let difference = builder.add(Subtract::new(), &[a, b]);
    let head = builder.add(Dense::new(2, Activation::Linear).unwrap(), &[difference]);
    let joined = builder.add(Concatenate::new(-1), &[a, b]);
    let mut model = builder.build(&[head, joined]).unwrap();

    assert_eq!(model.output_shapes()[0].to_string(), "(4, 2)");
    assert_eq!(model.output_shapes()[1].to_string(), "(4, 6)");

    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    model.with_loss_weights(&[1.0, 0.5]).unwrap();

    let history = model
        .fit(&[&left, &right], &[&target_a, &target_b], 3)
        .unwrap();
    assert_eq!(history.loss().len(), 3);

    let predicted = model.predict(&[&left, &right]).unwrap();
    assert_eq!(predicted.len(), 2);
    assert_eq!(predicted[0].shape(), &[4, 2]);
    assert_eq!(predicted[1].shape(), &[4, 6]);
}

/// A graph saves and loads every array of its arena
#[test]
fn a_graph_round_trips_through_a_checkpoint() {
    let x = data(&[4, 5]);
    let y = data(&[4, 2]);

    let build_model = || {
        let mut builder = GraphBuilder::new();
        let input = builder.input(Shape::known(&[4, 5]));
        let hidden = builder.add(
            Dense::new(3, Activation::Tanh)
                .unwrap()
                .with_random_state(1),
            &[input],
        );
        let head = builder.add(
            Dense::new(2, Activation::Linear)
                .unwrap()
                .with_random_state(2),
            &[hidden],
        );
        builder.build(&[head]).unwrap()
    };

    let mut trained = build_model();
    trained.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    trained.fit(&[&x], &[&y], 3).unwrap();

    let path = std::env::temp_dir().join("rustyml_graph_round_trip.bin");
    trained.save_to_path(&path).unwrap();
    let mut loaded = build_model();
    loaded.load_from_path(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let before = trained.predict(&[&x]).unwrap();
    let after = loaded.predict(&[&x]).unwrap();
    for (a, b) in before[0].iter().zip(after[0].iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "the checkpoint lost a value");
    }
}

/// A node whose input count the arity of its layer refuses is caught at build time
#[test]
fn a_node_with_the_wrong_input_count_is_refused() {
    let mut builder = GraphBuilder::new();
    let a = builder.input(Shape::known(&[2, 3]));
    let b = builder.input(Shape::known(&[2, 3]));
    let c = builder.input(Shape::known(&[2, 3]));
    // Subtract takes exactly 2 inputs
    let bad = builder.add(Subtract::new(), &[a, b, c]);

    let message = match builder.build(&[bad]) {
        Ok(_) => panic!("a node with 3 inputs must not reach a layer that takes 2"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("Subtract"), "{message}");
}

/// A shape that a layer refuses names the node, the layer, and its type
#[test]
fn a_shape_a_layer_refuses_names_the_node() {
    let mut builder = GraphBuilder::new();
    let a = builder.input(Shape::known(&[2, 3]));
    let b = builder.input(Shape::known(&[2, 5]));
    let bad = builder.add(Add::new(), &[a, b]);

    let message = match builder.build(&[bad]) {
        Ok(_) => panic!("2 shapes that do not merge must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("node 2"), "{message}");
    assert!(message.contains("Add"), "{message}");
}

/// A chain gives every layer 1 input, so it refuses a layer that needs more
///
/// `Subtract` takes exactly 2 inputs and a chain can never give it a second one, so the build
/// refuses it and names the graph. `Add` takes 1 input or more, and 1 input is the identity,
/// so a chain accepts it
#[test]
fn a_chain_refuses_a_layer_that_needs_a_second_input() {
    let refused = SequentialBuilder::new()
        .add(Dense::new(3, Activation::Linear).unwrap())
        .add(Subtract::new())
        .build(&Shape::known(&[2, 4]));

    let message = match refused {
        Ok(_) => panic!("a chain cannot hold a layer that takes 2 inputs"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("graph"), "{message}");

    let accepted = SequentialBuilder::new()
        .add(Dense::new(3, Activation::Linear).unwrap())
        .add(Add::new())
        .build(&Shape::known(&[2, 4]));
    assert!(accepted.is_ok(), "Add of 1 input is the identity");
}

/// The summary of a graph names every node, its layer, and the nodes it reads
#[test]
fn a_graph_reports_a_summary() {
    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[2, 3]));
    let hidden = builder.add(Dense::new(4, Activation::ReLU).unwrap(), &[input]);
    let sum = builder.add(Add::new(), &[hidden, hidden]);
    let model = builder.build(&[sum]).unwrap();
    // The call prints, and the test holds it against a panic and nothing more
    model.summary();
    assert_eq!(model.output_shapes()[0].to_string(), "(2, 4)");
}

/// An arity refusal reaches a caller that drives a merge layer by hand
#[test]
fn a_merge_layer_refuses_the_wrong_input_count_by_hand() {
    let layer = Add::new();
    let mut ctx = rustyml::neural_network::Ctx::inference();
    match Layer::forward_many(&layer, &[], &mut ctx) {
        Ok(_) => panic!("a merge layer takes at least 1 input"),
        Err(Error::InvalidInput(message)) => assert!(message.contains("Add"), "{message}"),
        Err(other) => panic!("expected InvalidInput, got {other:?}"),
    }
}
