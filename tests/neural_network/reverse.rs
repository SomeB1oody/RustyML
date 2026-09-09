//! The Reverse layer, and the bidirectional graph it makes correct
//!
//! A recurrent layer with `go_backwards` returns its states in processing order, so slot 0
//! holds the state that came from the last input timestep. Keras puts that branch back into
//! input order before it merges. These tests pin the layer, its 3 refusals, and the invariant
//! that ties it to the recurrent family.

use ndarray::{Array, Axis, IxDyn};
use rustyml::neural_network::graph::GraphBuilder;
use rustyml::neural_network::layers::{Activation, Concatenate, Dense, LSTM, Reverse};
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::Adam;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::UnaryLayer;
use rustyml::neural_network::{Ctx, Shape, Tensor};

/// A tensor whose every value names its own position
fn ramp(shape: &[usize]) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| i as f32).collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
}

/// Reverses 1 tensor through a built layer
fn reversed(axis: i32, input: &Tensor) -> Tensor {
    let mut layer = Reverse::new(axis);
    let mut ctx = Ctx::inference();
    layer
        .forward_mut(input, &mut ctx)
        .expect("the layer serves this rank and axis")
}

/// The layer flips the named axis and leaves every other axis as it was
#[test]
fn reverse_flips_the_named_axis_alone() {
    let x = ramp(&[2, 3, 2]);
    let got = reversed(1, &x);
    assert_eq!(got.shape(), &[2, 3, 2]);
    for batch in 0..2 {
        for step in 0..3 {
            for feature in 0..2 {
                assert_eq!(
                    got[[batch, step, feature]],
                    x[[batch, 2 - step, feature]],
                    "slot {step} must hold what slot {} held",
                    2 - step
                );
            }
        }
    }
}

/// A negative axis counts back from the end of the full rank
#[test]
fn reverse_resolves_a_negative_axis_against_the_rank() {
    let x = ramp(&[2, 3, 4]);
    assert_eq!(reversed(-1, &x), reversed(2, &x));
    assert_eq!(reversed(-2, &x), reversed(1, &x));
}

/// The output holds the standard memory order, so the next layer reads it as 1 slice
#[test]
fn reverse_emits_the_standard_memory_order() {
    let got = reversed(1, &ramp(&[2, 3, 4]));
    assert!(
        got.as_slice().is_some(),
        "a reversed view carries a negative stride, and the output must not"
    );
}

/// A rank-2 input is refused, because its axis 1 holds features and not a sequence
///
/// This is the shape a recurrent branch gives when `return_sequences` stays unset. Without the
/// refusal the layer reorders the features of every sample and reports nothing.
#[test]
fn reverse_refuses_a_rank_2_input() {
    let Err(error) = SequentialBuilder::new()
        .add(Reverse::new(1))
        .build(&Shape::known(&[2, 5]))
    else {
        panic!("a rank-2 input holds no axis whose order carries meaning");
    };
    let text = format!("{error}");
    assert!(text.contains("rank 3 or more"), "got {text}");
    assert!(text.contains("with_return_sequences"), "got {text}");
}

/// The batch axis is refused, because reversing it would move a sample against its target
#[test]
fn reverse_refuses_the_batch_axis() {
    for axis in [0_i32, -3] {
        let Err(error) = SequentialBuilder::new()
            .add(Reverse::new(axis))
            .build(&Shape::known(&[2, 3, 4]))
        else {
            panic!("axis 0 is the batch axis");
        };
        assert!(
            format!("{error}").contains("batch axis"),
            "the message must name the batch axis"
        );
    }
}

/// An axis the rank does not hold is refused, and the message names both
#[test]
fn reverse_refuses_an_axis_the_rank_does_not_hold() {
    let Err(error) = SequentialBuilder::new()
        .add(Reverse::new(7))
        .build(&Shape::known(&[2, 3, 4]))
    else {
        panic!("rank 3 holds no axis 7");
    };
    let text = format!("{error}");
    assert!(text.contains("axis 7"), "got {text}");
    assert!(text.contains("rank 3"), "got {text}");
}

/// A checkpoint refuses a model whose Reverse names another axis
///
/// The layer changes no shape and holds no array, so the recorded build shapes and the empty
/// weight roster are identical for any 2 axes. The reported type carries the axis, and it is
/// the only field a strict load can separate them by.
#[test]
fn a_checkpoint_refuses_a_different_reverse_axis() {
    let x = ramp(&[2, 3, 4]);
    let build = |axis: i32| {
        SequentialBuilder::new()
            .add(Reverse::new(axis))
            .add(
                Dense::new(2, Activation::Linear)
                    .unwrap()
                    .with_random_state(4),
            )
            .build(&Shape::known(&[2, 3, 4]))
            .expect("the stack builds")
    };

    let saved = build(1);
    let dir = std::env::temp_dir().join(format!("rustyml_reverse_axis_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the temporary directory is writable");
    let path = dir.join("m.rustyml");
    saved.save_to_path(&path).expect("the model saves");

    let mut other = build(2);
    let outcome = other.load_from_path(&path);
    let _ = std::fs::remove_dir_all(&dir);

    let error = outcome.expect_err("a load must not accept another axis");
    assert!(
        format!("{error}").contains("Reverse"),
        "the message names the layer: {error}"
    );
    let _ = x;
}

/// A reversed backward branch starts at the first input timestep
///
/// This is the invariant that ties the layer to the recurrent family, and it is what Keras
/// defines. Processing step k of a `go_backwards` layer consumes input timestep `t - 1 - k`, so
/// its last processing state consumed input timestep 0. After the reversal that state sits at
/// slot 0, which is where a forward branch holds the state of input timestep 0. The 2 branches
/// therefore agree timestep for timestep, which is what a merge layer needs.
#[test]
fn a_reversed_backward_branch_starts_at_the_first_input_timestep() {
    let x = ramp(&[2, 5, 3]);
    let mut ctx = Ctx::inference();

    let mut sequence_layer = LSTM::new(2, Activation::Tanh)
        .unwrap()
        .with_go_backwards(true)
        .with_return_sequences(true)
        .with_random_state(11);
    let sequence = sequence_layer.forward_mut(&x, &mut ctx).unwrap();

    let mut last_layer = LSTM::new(2, Activation::Tanh)
        .unwrap()
        .with_go_backwards(true)
        .with_random_state(11);
    let last = last_layer.forward_mut(&x, &mut ctx).unwrap();

    let aligned = reversed(1, &sequence);
    let first_slot = aligned.index_axis(Axis(1), 0);
    assert_eq!(
        first_slot.shape(),
        last.shape(),
        "the returned last state and 1 slot of the sequence hold the same shape"
    );
    for (slot, state) in first_slot.iter().zip(last.iter()) {
        assert_eq!(
            slot.to_bits(),
            state.to_bits(),
            "slot 0 of the reversed sequence is the state the layer returns without \
             `return_sequences`, bit for bit"
        );
    }
}

/// A bidirectional graph builds, aligns its 2 branches, and trains both of them
#[test]
fn a_bidirectional_graph_trains_both_directions() {
    let x = ramp(&[3, 4, 2]);
    let y = ramp(&[3, 4, 4]);

    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[3, 4, 2]));
    let forward = builder.add(
        LSTM::new(2, Activation::Tanh)
            .unwrap()
            .with_return_sequences(true)
            .with_random_state(1),
        &[input],
    );
    let backward = builder.add(
        LSTM::new(2, Activation::Tanh)
            .unwrap()
            .with_return_sequences(true)
            .with_go_backwards(true)
            .with_random_state(2),
        &[input],
    );
    let aligned = builder.add(Reverse::new(1), &[backward]);
    let merged = builder.add(Concatenate::new(-1), &[forward, aligned]);
    let mut model = builder.build(&[merged]).expect("the graph builds");

    // 2 sibling layers are 2 arena entries, so their names never collide
    let paths = model.weight_paths();
    assert_eq!(
        paths.len(),
        6,
        "2 recurrent layers hold 3 arrays each: {paths:?}"
    );
    assert!(paths.contains(&"0.kernel".to_string()), "{paths:?}");
    assert!(paths.contains(&"1.kernel".to_string()), "{paths:?}");

    model.compile(
        Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let before_forward = model.weight("0.kernel").unwrap().sum();
    let before_backward = model.weight("1.kernel").unwrap().sum();

    let history = model.fit(&[&x], &[&y], 3).expect("the model trains");
    assert_eq!(history.loss().len(), 3);
    assert!(
        history.loss()[2] < history.loss()[0],
        "the loss must fall: {:?}",
        history.loss()
    );
    assert_ne!(
        model.weight("0.kernel").unwrap().sum().to_bits(),
        before_forward.to_bits(),
        "the forward branch must train"
    );
    assert_ne!(
        model.weight("1.kernel").unwrap().sum().to_bits(),
        before_backward.to_bits(),
        "the backward branch must train"
    );
}
