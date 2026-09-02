//! Integration tests for [`Shape`] and for `Layer::compute_output_shape`.
//!
//! 2 properties matter here. The first is purity: the answer is a function of the layer
//! configuration and of the shape the caller passes, and of nothing a forward pass wrote. The
//! second is refusal: an input the layer cannot accept comes back as an error that names the
//! layer and the problem.
//!
//! Purity is what lets a later change walk a whole model at build time, thread each output
//! shape into the next layer, and reject a bad stack before any data arrives. The last test of
//! this file threads such a stack by hand, with no tensor anywhere.

use ndarray::{Array, ArrayD, IxDyn};
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::p_relu::PReLU;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::border::cropping_2d::Cropping2D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::pooling::average_pooling_1d::AveragePooling1D;
use rustyml::neural_network::layers::pooling::global_max_pooling_1d::GlobalMaxPooling1D;
use rustyml::neural_network::layers::pooling::max_pooling_1d::MaxPooling1D;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::repeat_vector::RepeatVector;
use rustyml::neural_network::layers::reshape::Reshape;
use rustyml::neural_network::layers::upsampling::Interpolation;
use rustyml::neural_network::layers::upsampling::up_sampling_2d::UpSampling2D;
use rustyml::neural_network::traits::Layer;

// Purity: the answer comes from the configuration and the argument alone

/// Every pooling layer answers before it has seen a single tensor
///
/// Windowed pooling applies its window to each spatial axis. Global pooling drops every
/// spatial axis. Neither layer has run a forward pass at this point, and the global layer
/// holds no input shape at all
#[test]
fn the_pooling_family_answers_before_any_forward_pass() {
    let pool_1d = MaxPooling1D::new(2, vec![1, 8, 2])
        .unwrap()
        .with_stride(2)
        .unwrap();
    let computed = pool_1d
        .compute_output_shape(&Shape::with_free_batch(&[1, 8, 2]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 4, 2)");

    let pool_2d = MaxPooling2D::new((2, 2), vec![1, 6, 6, 3]).unwrap();
    let computed = pool_2d
        .compute_output_shape(&Shape::with_free_batch(&[1, 6, 6, 3]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 3, 3, 3)");

    let average_1d = AveragePooling1D::new(3, vec![1, 6, 1]).unwrap();
    let computed = average_1d
        .compute_output_shape(&Shape::known(&[4, 6, 1]))
        .unwrap();
    assert_eq!(computed.to_string(), "(4, 2, 1)");

    // A global pooling layer keeps no input shape until a forward pass gives it one, so its
    // own `output_shape()` still reads "Unknown" here
    let global = GlobalMaxPooling1D::new();
    assert_eq!(global.output_shape(), "Unknown");
    let computed = global
        .compute_output_shape(&Shape::with_free_batch(&[5, 9, 7]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 7)");
}

/// A layer answers for a shape that differs from the 1 its constructor took
///
/// A pooling layer built for a length of 8 answers correctly for a length of 20. Nothing in
/// the answer comes from the declared shape except the window and the stride
#[test]
fn a_layer_answers_for_a_shape_it_was_not_built_for() {
    let layer = MaxPooling1D::new(2, vec![1, 8, 2])
        .unwrap()
        .with_stride(2)
        .unwrap();
    assert_eq!(layer.output_shape(), "(1, 4, 2)");

    let computed = layer
        .compute_output_shape(&Shape::with_free_batch(&[64, 20, 2]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 10, 2)");
}

/// A forward pass changes no answer that `compute_output_shape` gives
///
/// The layer runs a forward pass on a shape it was not built for. The method still answers
/// from its argument, so both calls agree
#[test]
fn a_forward_pass_changes_no_answer() {
    let mut layer = Conv2D::new(4, (3, 3), vec![2, 8, 8, 3], (1, 1), ReLU::new()).unwrap();
    let asked = Shape::with_free_batch(&[2, 10, 10, 3]);

    let before = layer.compute_output_shape(&asked).unwrap();
    layer.forward(&ArrayD::zeros(IxDyn(&[2, 8, 8, 3]))).unwrap();
    let after = layer.compute_output_shape(&asked).unwrap();

    assert_eq!(before, after);
    assert_eq!(before.to_string(), "(None, 8, 8, 4)");
}

/// A free batch axis stays free, and a fixed batch extent passes straight through
#[test]
fn the_batch_axis_passes_through_as_it_arrives() {
    let layer = Conv2D::new(2, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();

    let free = layer
        .compute_output_shape(&Shape::with_free_batch(&[1, 4, 4, 1]))
        .unwrap();
    assert_eq!(free.axes()[0], None);
    assert_eq!(free.to_string(), "(None, 3, 3, 2)");

    let fixed = layer
        .compute_output_shape(&Shape::known(&[6, 4, 4, 1]))
        .unwrap();
    assert_eq!(fixed.axes()[0], Some(6));
    assert_eq!(fixed.to_string(), "(6, 3, 3, 2)");
}

/// A recurrent layer keeps the time axis free when it returns a sequence, and drops it when it
/// returns the final state
#[test]
fn a_recurrent_layer_keeps_a_free_time_axis() {
    let sequence = LSTM::new(3, 2, Tanh::new())
        .unwrap()
        .with_return_sequences(true);
    let computed = sequence
        .compute_output_shape(&Shape::new(vec![None, None, Some(3)]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, None, 2)");

    let last = LSTM::new(3, 2, Tanh::new()).unwrap();
    let computed = last
        .compute_output_shape(&Shape::new(vec![None, Some(7), Some(3)]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 2)");
}

/// A layer that learns its shape from data still answers before it sees any
///
/// `UpSampling2D` reports "Unknown" until a forward pass, because it holds no input shape.
/// Its shape algebra needs no such state
#[test]
fn a_data_driven_layer_answers_with_no_data() {
    let layer = UpSampling2D::new((2, 3), Interpolation::Nearest).unwrap();
    assert_eq!(layer.output_shape(), "Unknown");

    let computed = layer
        .compute_output_shape(&Shape::with_free_batch(&[1, 5, 6, 2]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 10, 18, 2)");
}

// Refusal: an input the layer cannot accept comes back as a named error

/// A convolution refuses a rank its forward pass would refuse, and names the rank it wanted
#[test]
fn a_convolution_refuses_the_wrong_rank() {
    let layer = Conv2D::new(4, (3, 3), vec![2, 8, 8, 3], (1, 1), ReLU::new()).unwrap();
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[2, 8, 3]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("Conv2D"), "{message}");
    assert!(message.contains("rank 4"), "{message}");
    assert!(message.contains("(None, 8, 3)"), "{message}");
}

/// A dense layer refuses a last axis that does not match its input dimension
#[test]
fn a_dense_layer_refuses_the_wrong_feature_count() {
    let layer = Dense::new(7, 4, Linear::new()).unwrap();
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[2, 5]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("Dense"), "{message}");
    assert!(message.contains("7 elements on the last axis"), "{message}");
    assert!(message.contains("got 5"), "{message}");
}

/// A pooling layer refuses a window that does not fit the length it is given
#[test]
fn a_pooling_layer_refuses_a_window_that_does_not_fit() {
    let layer = MaxPooling1D::new(4, vec![1, 8, 2]).unwrap();
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[1, 3, 2]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("pool_size"), "{message}");
    assert!(
        message.contains("cannot exceed the corresponding input dimension"),
        "{message}"
    );
}

/// A global pooling layer refuses a spatial axis whose extent nothing fixes
///
/// The answer would need the channel extent, and a free axis carries none
#[test]
fn a_global_pooling_layer_refuses_a_free_spatial_axis() {
    let layer = GlobalMaxPooling1D::new();
    let message = layer
        .compute_output_shape(&Shape::new(vec![None, Some(4), None]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("GlobalMaxPooling1D"), "{message}");
    assert!(message.contains("axis 2"), "{message}");
}

/// PReLU refuses an extent on an axis that holds 1 slope per position
#[test]
fn p_relu_refuses_an_axis_it_holds_slopes_for() {
    let layer = PReLU::new(vec![2, 3, 4], 0.25).unwrap();
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[2, 5, 4]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("PReLU"), "{message}");
    assert!(message.contains("axis 1"), "{message}");
    assert!(message.contains("shared_axes"), "{message}");
}

/// `RepeatVector` refuses anything that is not a batch of feature vectors
#[test]
fn repeat_vector_refuses_a_rank_it_cannot_repeat() {
    let layer = RepeatVector::new(3).unwrap();
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[2, 4, 5]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("RepeatVector"), "{message}");
    assert!(message.contains("rank 2"), "{message}");
}

/// `Reshape` refuses an element count its target cannot match
#[test]
fn reshape_refuses_an_element_count_it_cannot_match() {
    let layer = Reshape::new(vec![2, 2]).unwrap();
    assert!(
        layer
            .compute_output_shape(&Shape::with_free_batch(&[2, 5]))
            .is_err()
    );

    // The same layer accepts the count its target names
    let computed = layer
        .compute_output_shape(&Shape::with_free_batch(&[2, 4]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 2, 2)");
}

/// A cropping layer refuses a border that would leave an axis with no position
#[test]
fn a_cropping_layer_refuses_a_border_that_removes_everything() {
    let layer = Cropping2D::new(2);
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[1, 4, 8, 3]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("Cropping2D"), "{message}");
    assert!(message.contains("at least 1 must remain"), "{message}");
}

// The destination: a whole stack checked with no data

/// A stack of 4 layers reports its final shape before any tensor exists
///
/// This is what the pure method buys. The caller threads the output shape of each layer into
/// the next one and learns the shape of the model output, with no forward pass anywhere. A
/// later change moves this walk into the model itself
#[test]
fn a_whole_stack_answers_before_any_tensor_exists() {
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Conv2D::new(4, (3, 3), vec![1, 8, 8, 1], (1, 1), ReLU::new()).unwrap()),
        Box::new(MaxPooling2D::new((2, 2), vec![1, 6, 6, 4]).unwrap()),
        Box::new(Flatten::new(vec![1, 3, 3, 4]).unwrap()),
        Box::new(Dense::new(36, 5, Linear::new()).unwrap()),
    ];

    let mut shape = Shape::with_free_batch(&[1, 8, 8, 1]);
    for (position, layer) in layers.iter().enumerate() {
        shape = layer
            .compute_output_shape(&shape)
            .unwrap_or_else(|error| panic!("layer {position} ({}): {error}", layer.layer_type()));
    }
    assert_eq!(shape.to_string(), "(None, 5)");
}

/// A bad stack names the layer that refuses, and its type
///
/// The pooling window is larger than what the convolution before it emits. A forward pass
/// would report this from inside the pooling kernel, with no layer named
#[test]
fn a_bad_stack_names_the_layer_that_refuses() {
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Conv2D::new(4, (3, 3), vec![1, 4, 4, 1], (1, 1), ReLU::new()).unwrap()),
        Box::new(MaxPooling2D::new((4, 4), vec![1, 8, 8, 4]).unwrap()),
    ];

    let mut shape = Shape::with_free_batch(&[1, 4, 4, 1]);
    let mut refused = None;
    for (position, layer) in layers.iter().enumerate() {
        match layer.compute_output_shape(&shape) {
            Ok(next) => shape = next,
            Err(error) => {
                refused = Some(format!(
                    "layer {position} ({}): {error}",
                    layer.layer_type()
                ));
                break;
            }
        }
    }

    let report = refused.expect("the stack must be refused");
    assert!(report.starts_with("layer 1 (MaxPooling2D)"), "{report}");
}

// The display value stays what it was

/// `output_shape` runs the pure method against the shape the layer holds
///
/// The 3 layers below cover the 3 sources of that shape: a constructor argument, the last
/// forward input, and nothing at all
#[test]
fn output_shape_reads_the_shape_the_layer_holds() {
    // Declared by the constructor, and reported with the declared batch extent
    let declared = AveragePooling1D::new(2, vec![1, 6, 1]).unwrap();
    assert_eq!(declared.output_shape(), "(1, 3, 1)");

    // Learned from the last forward input, and reported with a free batch axis
    let mut learned = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
    assert_eq!(learned.output_shape(), "Unknown");
    learned
        .forward(&Array::zeros(IxDyn(&[1, 3, 3, 2])).into_dyn())
        .unwrap();
    assert_eq!(learned.output_shape(), "(None, 6, 6, 2)");

    // Held by neither, so the layer reports nothing
    let global = GlobalMaxPooling1D::new();
    assert_eq!(global.output_shape(), "Unknown");
}
