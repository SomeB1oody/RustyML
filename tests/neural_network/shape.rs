//! Integration tests for [`Shape`] and for the pure output-shape method of a layer.
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
use rustyml::neural_network::layers::*;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::{Layer, UnaryLayer};
use rustyml::neural_network::{Ctx, Shape};
use std::collections::BTreeSet;

// Purity: the answer comes from the configuration and the argument alone

/// Every pooling layer answers before it has seen a single tensor
///
/// Windowed pooling applies its window to each spatial axis. Global pooling drops every
/// spatial axis. Neither layer has run a forward pass at this point, and the global layer
/// holds no input shape at all
#[test]
fn the_pooling_family_answers_before_any_forward_pass() {
    let pool_1d = MaxPooling1D::new(2).with_stride(2).unwrap();
    let computed = pool_1d
        .compute_output_shape(&Shape::with_free_batch(&[1, 8, 2]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 4, 2)");

    let pool_2d = MaxPooling2D::new((2, 2));
    let computed = pool_2d
        .compute_output_shape(&Shape::with_free_batch(&[1, 6, 6, 3]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 3, 3, 3)");

    let average_1d = AveragePooling1D::new(3);
    let computed = average_1d
        .compute_output_shape(&Shape::known(&[4, 6, 1]))
        .unwrap();
    assert_eq!(computed.to_string(), "(4, 2, 1)");

    // A global pooling layer keeps no input shape until a build gives it one, so its own
    // `output_shape()` still reads "Unknown" here
    let global = GlobalMaxPooling1D::new();
    assert_eq!(global.output_shape(), "Unknown");
    let computed = global
        .compute_output_shape(&Shape::with_free_batch(&[5, 9, 7]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 7)");
}

/// A layer answers for a shape that differs from the 1 its build took
///
/// A pooling layer built for a length of 8 answers correctly for a length of 20. Nothing in
/// the answer comes from the build shape except the window and the stride
#[test]
fn a_layer_answers_for_a_shape_it_was_not_built_for() {
    let mut layer = MaxPooling1D::new(2).with_stride(2).unwrap();
    layer.build(&Shape::known(&[1, 8, 2])).unwrap();
    assert_eq!(layer.output_shape(), "(None, 4, 2)");

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
    let mut layer = Conv2D::new(4, (3, 3), (1, 1), ReLU::new()).unwrap();
    let asked = Shape::with_free_batch(&[2, 10, 10, 3]);

    let before = layer.compute_output_shape(&asked).unwrap();
    layer
        .forward_mut(&ArrayD::zeros(IxDyn(&[2, 8, 8, 3])), &mut Ctx::inference())
        .unwrap();
    let after = layer.compute_output_shape(&asked).unwrap();

    assert_eq!(before, after);
    assert_eq!(before.to_string(), "(None, 8, 8, 4)");
}

/// A free batch axis stays free, and a fixed batch extent passes straight through
#[test]
fn the_batch_axis_passes_through_as_it_arrives() {
    let layer = Conv2D::new(2, (2, 2), (1, 1), Linear::new()).unwrap();

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
    let sequence = LSTM::new(2, Tanh::new())
        .unwrap()
        .with_return_sequences(true);
    let computed = sequence
        .compute_output_shape(&Shape::new(vec![None, None, Some(3)]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, None, 2)");

    let last = LSTM::new(2, Tanh::new()).unwrap();
    let computed = last
        .compute_output_shape(&Shape::new(vec![None, Some(7), Some(3)]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 2)");
}

/// A layer that learns its shape from data still answers before it sees any
///
/// `UpSampling2D` reports "Unknown" until its build, because it holds no input shape. Its
/// shape algebra needs no such state
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
    let layer = Conv2D::new(4, (3, 3), (1, 1), ReLU::new()).unwrap();
    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[2, 8, 3]))
        .unwrap_err()
        .to_string();

    assert!(message.contains("Conv2D"), "{message}");
    assert!(message.contains("rank 4"), "{message}");
    assert!(message.contains("(None, 8, 3)"), "{message}");
}

/// A built dense layer refuses a last axis that its kernel cannot contract
///
/// The kernel is `(input_dim, units)`, and the build reads `input_dim` from the last axis. A
/// layer built for 7 features therefore holds a kernel that 5 features cannot enter
#[test]
fn a_dense_layer_refuses_the_wrong_feature_count() {
    let mut layer = Dense::new(4, Linear::new()).unwrap();
    layer.build(&Shape::with_free_batch(&[2, 7])).unwrap();
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
    let layer = MaxPooling1D::new(4);
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

/// A built PReLU refuses an extent on an axis that holds 1 slope per position
///
/// The build draws 1 slope per position of every axis after the batch axis that `shared_axes`
/// leaves out. A layer built for 3 positions on axis 1 holds 3 slopes, and 5 positions reach
/// past them
#[test]
fn p_relu_refuses_an_axis_it_holds_slopes_for() {
    let mut layer = PReLU::new(0.25).unwrap();
    layer.build(&Shape::with_free_batch(&[2, 3, 4])).unwrap();
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

// The `Valid` convolution family: an oversized kernel is refused, and not reported as 0

/// 1 unbuilt convolution of every type that bounds its input, with a shape too small for it
///
/// Every entry carries a kernel of 3 taps on each spatial axis and an input of 2 positions on
/// each spatial axis, so no complete window fits under `Valid` padding
fn valid_convolutions_that_cannot_fit() -> Vec<(Box<dyn Layer>, Shape, &'static str)> {
    vec![
        (
            Box::new(Conv1D::new(1, 3, 1, Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 1]),
            "Conv1D",
        ),
        (
            Box::new(Conv2D::new(1, (3, 3), (1, 1), Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 2, 1]),
            "Conv2D",
        ),
        (
            Box::new(Conv3D::new(1, (3, 3, 3), (1, 1, 1), Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            "Conv3D",
        ),
        (
            Box::new(DepthwiseConv1D::new(3, 1, Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 1]),
            "DepthwiseConv1D",
        ),
        (
            Box::new(DepthwiseConv2D::new((3, 3), (1, 1), Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 2, 1]),
            "DepthwiseConv2D",
        ),
        (
            Box::new(SeparableConv1D::new(1, 3, 1, 1, Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 1]),
            "SeparableConv1D",
        ),
        (
            Box::new(SeparableConv2D::new(1, (3, 3), (1, 1), 1, Linear::new()).unwrap()),
            Shape::with_free_batch(&[1, 2, 2, 1]),
            "SeparableConv2D",
        ),
    ]
}

/// Every `Valid` convolution refuses an input axis shorter than its effective kernel
///
/// A kernel that does not fit leaves no complete window, so the axis would carry 0 positions.
/// A 0 extent is not an answer. It passes through every later layer of the stack and turns up
/// far from the layer that made it. The refusal names the layer, the axis, and the effective
/// kernel extent that the axis must reach
#[test]
fn every_valid_convolution_refuses_a_kernel_larger_than_the_input() {
    for (layer, input, name) in valid_convolutions_that_cannot_fit() {
        let message = match layer.compute_output_shape_many(std::slice::from_ref(&input)) {
            Ok(answer) => panic!("{name} answered {answer}, and it must refuse the shape {input}"),
            Err(error) => error.to_string(),
        };

        assert!(message.contains(name), "{name}: {message}");
        assert!(
            message.contains("effective kernel extent, which is 3"),
            "{name}: {message}"
        );
        assert!(message.contains("is 2."), "{name}: {message}");
    }
}

/// The same refusal reaches the build, so no oversized layer ever holds a weight
///
/// [`Layer::build_many`] is the step that allocates. It applies the same shape algebra first, so
/// a layer that cannot run draws nothing at all
#[test]
fn every_valid_convolution_refuses_the_build_as_well() {
    for (mut layer, input, name) in valid_convolutions_that_cannot_fit() {
        let message = layer
            .build_many(std::slice::from_ref(&input))
            .unwrap_err()
            .to_string();

        assert!(message.contains(name), "{name}: {message}");
        assert!(
            message.contains("effective kernel extent, which is 3"),
            "{name}: {message}"
        );
        assert!(message.contains("is 2."), "{name}: {message}");
        assert_eq!(layer.param_count().total(), 0, "{name} drew a weight");
    }
}

/// `Same` and `Causal` padding accept the kernel that `Valid` refuses
///
/// The padding supplies the cells the window reaches past, so the same geometry runs. This is
/// why the rule belongs to `Valid` alone, and why no constructor can hold it
#[test]
fn same_and_causal_padding_accept_an_oversized_kernel() {
    let mut same = Conv2D::new(1, (3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    same.build(&Shape::with_free_batch(&[1, 2, 2, 1])).unwrap();
    let computed = same
        .compute_output_shape(&Shape::with_free_batch(&[1, 2, 2, 1]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 2, 2, 1)");

    let mut causal = Conv1D::new(1, 3, 1, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Causal);
    causal.build(&Shape::with_free_batch(&[1, 2, 1])).unwrap();
    let computed = causal
        .compute_output_shape(&Shape::with_free_batch(&[1, 2, 1]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 2, 1)");
}

/// The rule measures the dilated extent, not the tap count
///
/// 2 taps spaced 3 apart span `(2 - 1) * 3 + 1 = 4` cells. An input of 3 positions holds the 2
/// taps and not the extent they span, so the layer refuses it
#[test]
fn the_valid_rule_reads_the_dilated_extent() {
    let layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((3, 3))
        .unwrap();

    let message = layer
        .compute_output_shape(&Shape::with_free_batch(&[1, 3, 3, 1]))
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("effective kernel extent, which is 4"),
        "{message}"
    );

    // 4 positions hold the dilated extent exactly, and give 1 output position
    let computed = layer
        .compute_output_shape(&Shape::with_free_batch(&[1, 4, 4, 1]))
        .unwrap();
    assert_eq!(computed.to_string(), "(None, 1, 1, 1)");
}

/// A transposed convolution puts no lower bound on its input
///
/// It grows its input instead of shrinking it, so a 1-position axis under a 3-tap kernel is a
/// normal first decoder step. The rule above must not reach these 3 layers
#[test]
fn the_transposed_convolutions_accept_a_single_position() {
    let mut conv_1d = Conv1DTranspose::new(1, 3, 1, Linear::new()).unwrap();
    conv_1d.build(&Shape::with_free_batch(&[1, 1, 1])).unwrap();
    assert_eq!(
        conv_1d
            .compute_output_shape(&Shape::with_free_batch(&[1, 1, 1]))
            .unwrap()
            .to_string(),
        "(None, 3, 1)"
    );

    let mut conv_2d = Conv2DTranspose::new(1, (3, 3), (1, 1), Linear::new()).unwrap();
    conv_2d
        .build(&Shape::with_free_batch(&[1, 1, 1, 1]))
        .unwrap();
    assert_eq!(
        conv_2d
            .compute_output_shape(&Shape::with_free_batch(&[1, 1, 1, 1]))
            .unwrap()
            .to_string(),
        "(None, 3, 3, 1)"
    );

    let mut conv_3d = Conv3DTranspose::new(1, (3, 3, 3), (1, 1, 1), Linear::new()).unwrap();
    conv_3d
        .build(&Shape::with_free_batch(&[1, 1, 1, 1, 1]))
        .unwrap();
    assert_eq!(
        conv_3d
            .compute_output_shape(&Shape::with_free_batch(&[1, 1, 1, 1, 1]))
            .unwrap()
            .to_string(),
        "(None, 3, 3, 3, 1)"
    );
}

/// A model build refuses the oversized kernel, and the message names the layer position
///
/// The first convolution takes an 8 by 8 input down to 4 by 4, and the second one takes that
/// to 2 by 2. The third one cannot run, and the build says so before it allocates. Nothing
/// past that position is built
#[test]
fn a_model_build_refuses_an_oversized_valid_kernel() {
    let refused = SequentialBuilder::new()
        .add(Conv2D::new(2, (5, 5), (1, 1), ReLU::new()).unwrap())
        .add(Conv2D::new(2, (3, 3), (1, 1), ReLU::new()).unwrap())
        .add(Conv2D::new(2, (3, 3), (1, 1), ReLU::new()).unwrap())
        .add(Flatten::new())
        .build(&Shape::known(&[2, 8, 8, 1]));

    let message = match refused {
        Ok(_) => panic!("a 3 by 3 Valid kernel does not fit a 2 by 2 input"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("layer 2"), "{message}");
    assert!(message.contains("Conv2D"), "{message}");
    assert!(message.contains("(2, 2, 2, 2)"), "{message}");
    assert!(
        message.contains("effective kernel extent, which is 3"),
        "{message}"
    );
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
        Box::new(Conv2D::new(4, (3, 3), (1, 1), ReLU::new()).unwrap()),
        Box::new(MaxPooling2D::new((2, 2))),
        Box::new(Flatten::new()),
        Box::new(Dense::new(5, Linear::new()).unwrap()),
    ];

    let mut shape = Shape::with_free_batch(&[1, 8, 8, 1]);
    for (position, layer) in layers.iter().enumerate() {
        shape = layer
            .compute_output_shape_many(std::slice::from_ref(&shape))
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
        Box::new(Conv2D::new(4, (3, 3), (1, 1), ReLU::new()).unwrap()),
        Box::new(MaxPooling2D::new((4, 4))),
    ];

    let mut shape = Shape::with_free_batch(&[1, 4, 4, 1]);
    let mut refused = None;
    for (position, layer) in layers.iter().enumerate() {
        match layer.compute_output_shape_many(std::slice::from_ref(&shape)) {
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
/// The 3 layers below cover the 3 sources of that shape: a build the caller ran, a build that
/// `forward_mut` ran from the tensor, and nothing at all
#[test]
fn output_shape_reads_the_shape_the_layer_holds() {
    // Taken from the build, and reported with the batch extent the build named
    let mut declared = AveragePooling1D::new(2);
    declared.build(&Shape::known(&[1, 6, 1])).unwrap();
    assert_eq!(declared.output_shape(), "(None, 3, 1)");

    // Taken from the build that `forward_mut` ran. That build reads the whole shape of the
    // tensor, batch extent included, so the report names the batch extent as well
    let mut learned = UpSampling2D::new(2, Interpolation::Nearest).unwrap();
    assert_eq!(learned.output_shape(), "Unknown");
    learned
        .forward_mut(
            &Array::zeros(IxDyn(&[1, 3, 3, 2])).into_dyn(),
            &mut Ctx::inference(),
        )
        .unwrap();
    assert_eq!(learned.output_shape(), "(None, 6, 6, 2)");

    // Held by neither, so the layer reports nothing
    let global = GlobalMaxPooling1D::new();
    assert_eq!(global.output_shape(), "Unknown");
}

/// The printed output shape is now exactly what the forward pass enforces
#[test]
fn a_built_layer_refuses_the_extents_its_summary_does_not_name() {
    use ndarray::Array4;
    let mut layer = rustyml::neural_network::layers::Conv2D::new(
        3,
        (2, 2),
        (1, 1),
        rustyml::neural_network::layers::Activation::Linear,
    )
    .unwrap();
    layer
        .build(&rustyml::neural_network::Shape::known(&[2, 4, 4, 2]))
        .unwrap();
    assert_eq!(layer.output_shape(), "(None, 3, 3, 3)");

    // The same layer used to accept any spatial extent, and its summary went on printing the
    // declared one
    let wider = Array4::<f32>::ones((2, 6, 6, 2)).into_dyn();
    assert!(layer.forward(&wider, &mut Ctx::inference()).is_err());

    // A different batch size still passes, because the batch axis is never checked
    let bigger_batch = Array4::<f32>::ones((5, 4, 4, 2)).into_dyn();
    assert!(layer.forward(&bigger_batch, &mut Ctx::inference()).is_ok());
}

// The whole roster: every layer type answers before its build

/// 1 unbuilt layer of each covered type, with the shape it takes and the shape it must give
/// back
///
/// The entries hold no built layer and no forward pass, so every answer comes from the
/// constructor arguments and from the input shape.
///
/// This table is the only roster of the layer types that the 2 tests below cover. No item of
/// the crate lists its layer types, so no other file can supply that roster. Add a case here
/// for each new layer type, and raise the count in the test below in the same change.
fn unbuilt_layers() -> Vec<(Shape, Box<dyn Layer>, &'static str)> {
    let flat = Shape::with_free_batch(&[1, 4]);
    let sequence = Shape::with_free_batch(&[1, 5, 2]);
    let signal = Shape::with_free_batch(&[1, 8, 2]);
    let image = Shape::with_free_batch(&[1, 8, 8, 2]);
    let volume = Shape::with_free_batch(&[1, 4, 4, 4, 2]);

    vec![
        // Dense and the shape layers
        (
            flat.clone(),
            Box::new(Dense::new(4, Linear::new()).unwrap()),
            "(None, 4)",
        ),
        (
            Shape::with_free_batch(&[1, 2, 3]),
            Box::new(Flatten::new()),
            "(None, 6)",
        ),
        (flat.clone(), Box::new(Identity::new()), "(None, 4)"),
        (
            flat.clone(),
            Box::new(Reshape::new(vec![2, 2]).unwrap()),
            "(None, 2, 2)",
        ),
        (
            Shape::with_free_batch(&[1, 2, 3]),
            Box::new(Permute::new(vec![2, 1]).unwrap()),
            "(None, 3, 2)",
        ),
        (
            flat.clone(),
            Box::new(RepeatVector::new(3).unwrap()),
            "(None, 3, 4)",
        ),
        (flat.clone(), Box::new(Rescaling::new(2.0)), "(None, 4)"),
        (
            Shape::with_free_batch(&[1, 5]),
            Box::new(Embedding::new(10, 3).unwrap()),
            "(None, 5, 3)",
        ),
        // Activations
        (flat.clone(), Box::new(ReLU::new()), "(None, 4)"),
        (
            flat.clone(),
            Box::new(LeakyReLU::new(0.1).unwrap()),
            "(None, 4)",
        ),
        (flat.clone(), Box::new(ELU::new(1.0).unwrap()), "(None, 4)"),
        (flat.clone(), Box::new(SELU::new()), "(None, 4)"),
        (flat.clone(), Box::new(Softplus::new()), "(None, 4)"),
        (flat.clone(), Box::new(Softsign::new()), "(None, 4)"),
        (flat.clone(), Box::new(HardSigmoid::new()), "(None, 4)"),
        (flat.clone(), Box::new(Exponential::new()), "(None, 4)"),
        (flat.clone(), Box::new(Linear::new()), "(None, 4)"),
        (flat.clone(), Box::new(Sigmoid::new()), "(None, 4)"),
        (flat.clone(), Box::new(Tanh::new()), "(None, 4)"),
        (flat.clone(), Box::new(Softmax::new()), "(None, 4)"),
        (
            flat.clone(),
            Box::new(PReLU::new(0.25).unwrap()),
            "(None, 4)",
        ),
        // Convolution
        (
            signal.clone(),
            Box::new(Conv1D::new(4, 3, 1, ReLU::new()).unwrap()),
            "(None, 6, 4)",
        ),
        (
            image.clone(),
            Box::new(Conv2D::new(4, (3, 3), (1, 1), ReLU::new()).unwrap()),
            "(None, 6, 6, 4)",
        ),
        (
            volume.clone(),
            Box::new(Conv3D::new(3, (2, 2, 2), (1, 1, 1), ReLU::new()).unwrap()),
            "(None, 3, 3, 3, 3)",
        ),
        (
            Shape::with_free_batch(&[1, 4, 2]),
            Box::new(Conv1DTranspose::new(3, 2, 2, Linear::new()).unwrap()),
            "(None, 8, 3)",
        ),
        (
            Shape::with_free_batch(&[1, 4, 4, 2]),
            Box::new(Conv2DTranspose::new(3, (2, 2), (2, 2), Linear::new()).unwrap()),
            "(None, 8, 8, 3)",
        ),
        (
            Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            Box::new(Conv3DTranspose::new(2, (2, 2, 2), (2, 2, 2), Linear::new()).unwrap()),
            "(None, 4, 4, 4, 2)",
        ),
        (
            signal.clone(),
            Box::new(SeparableConv1D::new(4, 3, 1, 1, ReLU::new()).unwrap()),
            "(None, 6, 4)",
        ),
        (
            image.clone(),
            Box::new(SeparableConv2D::new(4, (3, 3), (1, 1), 1, ReLU::new()).unwrap()),
            "(None, 6, 6, 4)",
        ),
        // A depthwise layer reads its channel count from the input, so the unbuilt answer
        // carries the channel count of the argument
        (
            signal.clone(),
            Box::new(DepthwiseConv1D::new(3, 1, ReLU::new()).unwrap()),
            "(None, 6, 2)",
        ),
        (
            image.clone(),
            Box::new(DepthwiseConv2D::new((3, 3), (1, 1), ReLU::new()).unwrap()),
            "(None, 6, 6, 2)",
        ),
        // Pooling
        (
            signal.clone(),
            Box::new(MaxPooling1D::new(2)),
            "(None, 4, 2)",
        ),
        (
            image.clone(),
            Box::new(MaxPooling2D::new((2, 2))),
            "(None, 4, 4, 2)",
        ),
        (
            volume.clone(),
            Box::new(MaxPooling3D::new((2, 2, 2))),
            "(None, 2, 2, 2, 2)",
        ),
        (
            signal.clone(),
            Box::new(AveragePooling1D::new(2)),
            "(None, 4, 2)",
        ),
        (
            image.clone(),
            Box::new(AveragePooling2D::new((2, 2))),
            "(None, 4, 4, 2)",
        ),
        (
            volume.clone(),
            Box::new(AveragePooling3D::new((2, 2, 2))),
            "(None, 2, 2, 2, 2)",
        ),
        (
            signal.clone(),
            Box::new(GlobalMaxPooling1D::new()),
            "(None, 2)",
        ),
        (
            image.clone(),
            Box::new(GlobalMaxPooling2D::new()),
            "(None, 2)",
        ),
        (
            volume.clone(),
            Box::new(GlobalMaxPooling3D::new()),
            "(None, 2)",
        ),
        (
            signal.clone(),
            Box::new(GlobalAveragePooling1D::new()),
            "(None, 2)",
        ),
        (
            image.clone(),
            Box::new(GlobalAveragePooling2D::new()),
            "(None, 2)",
        ),
        (
            volume.clone(),
            Box::new(GlobalAveragePooling3D::new()),
            "(None, 2)",
        ),
        // Resampling and borders
        (
            Shape::with_free_batch(&[1, 4, 2]),
            Box::new(UpSampling1D::new(2).unwrap()),
            "(None, 8, 2)",
        ),
        (
            Shape::with_free_batch(&[1, 4, 4, 2]),
            Box::new(UpSampling2D::new(2, Interpolation::Nearest).unwrap()),
            "(None, 8, 8, 2)",
        ),
        (
            Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            Box::new(UpSampling3D::new(2).unwrap()),
            "(None, 4, 4, 4, 1)",
        ),
        (
            Shape::with_free_batch(&[1, 4, 2]),
            Box::new(ZeroPadding1D::new(1)),
            "(None, 6, 2)",
        ),
        (
            Shape::with_free_batch(&[1, 4, 4, 2]),
            Box::new(ZeroPadding2D::new(1)),
            "(None, 6, 6, 2)",
        ),
        (
            Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            Box::new(ZeroPadding3D::new(1)),
            "(None, 4, 4, 4, 1)",
        ),
        (
            Shape::with_free_batch(&[1, 6, 2]),
            Box::new(Cropping1D::new(1)),
            "(None, 4, 2)",
        ),
        (
            Shape::with_free_batch(&[1, 6, 6, 2]),
            Box::new(Cropping2D::new(1)),
            "(None, 4, 4, 2)",
        ),
        (
            volume.clone(),
            Box::new(Cropping3D::new(1)),
            "(None, 2, 2, 2, 2)",
        ),
        // Recurrent
        (
            sequence.clone(),
            Box::new(SimpleRNN::new(3, Tanh::new()).unwrap()),
            "(None, 3)",
        ),
        (
            sequence.clone(),
            Box::new(LSTM::new(3, Tanh::new()).unwrap()),
            "(None, 3)",
        ),
        (
            sequence.clone(),
            Box::new(GRU::new(3, Tanh::new()).unwrap()),
            "(None, 3)",
        ),
        // Regularization
        (
            flat.clone(),
            Box::new(Dropout::new(0.5).unwrap()),
            "(None, 4)",
        ),
        (
            signal.clone(),
            Box::new(SpatialDropout1D::new(0.5).unwrap()),
            "(None, 8, 2)",
        ),
        (
            image.clone(),
            Box::new(SpatialDropout2D::new(0.5).unwrap()),
            "(None, 8, 8, 2)",
        ),
        (
            volume.clone(),
            Box::new(SpatialDropout3D::new(0.5).unwrap()),
            "(None, 4, 4, 4, 2)",
        ),
        (
            flat.clone(),
            Box::new(GaussianDropout::new(0.3).unwrap()),
            "(None, 4)",
        ),
        (
            flat.clone(),
            Box::new(GaussianNoise::new(0.1).unwrap()),
            "(None, 4)",
        ),
        (
            flat.clone(),
            Box::new(BatchNormalization::new(0.9, 1e-5).unwrap()),
            "(None, 4)",
        ),
        (
            flat.clone(),
            Box::new(LayerNormalization::new(1e-5).unwrap()),
            "(None, 4)",
        ),
        (
            flat.clone(),
            Box::new(GroupNormalization::new(2, 1e-5).unwrap()),
            "(None, 4)",
        ),
        (
            signal.clone(),
            Box::new(InstanceNormalization::new(1e-5).unwrap()),
            "(None, 8, 2)",
        ),
        (
            flat.clone(),
            Box::new(UnitNormalization::new(UnitNormalizationAxis::Default).unwrap()),
            "(None, 4)",
        ),
    ]
}

/// Every layer type answers `compute_output_shape` before its build
///
/// This is the contract that the method exists for. The answer reads the layer configuration
/// and the argument, and nothing that a build or a forward pass wrote, so an unbuilt layer
/// answers. That is what lets `SequentialBuilder::build` walk a stack, thread each output
/// shape into the next layer, and refuse a bad stack before any tensor exists
///
/// A layer that reads a field that only `build` fills breaks this. Such a layer reported the
/// unset value of that field, and refused every input, including the 1 it was about to be
/// built for
#[test]
fn every_layer_type_answers_before_its_build() {
    let mut covered = BTreeSet::new();
    for (input, layer, expected) in unbuilt_layers() {
        let name = layer.layer_type().to_string();
        assert!(
            layer.build_config().is_none(),
            "{name} must hold no build in this table"
        );

        let computed = layer
            .compute_output_shape_many(std::slice::from_ref(&input))
            .unwrap_or_else(|error| {
                panic!("{name} refused the shape {input} before its build: {error}")
            });
        assert_eq!(computed.to_string(), expected, "{name}");

        assert!(
            covered.insert(name.clone()),
            "{name} reaches the table more than once"
        );
    }

    assert_eq!(
        covered.len(),
        66,
        "the table above is the only roster of the covered layer types. A new layer type needs \
         a case here, and a removed one needs its case deleted. Move this count in the same \
         change"
    );
}

/// A build changes no answer that `compute_output_shape` already gave
///
/// The 2 halves of the contract meet here. Every layer of the table above answers for the
/// shape it is about to be built for, then takes that build, then answers again. A layer that
/// reads build state gives 2 different answers, and a layer that reads the argument alone
/// gives 1
#[test]
fn a_build_changes_no_answer() {
    for (input, mut layer, expected) in unbuilt_layers() {
        let name = layer.layer_type().to_string();
        let before = layer
            .compute_output_shape_many(std::slice::from_ref(&input))
            .unwrap();

        layer
            .build_many(std::slice::from_ref(&input))
            .unwrap_or_else(|error| panic!("{name} refused to build for {input}: {error}"));

        let after = layer
            .compute_output_shape_many(std::slice::from_ref(&input))
            .unwrap_or_else(|error| panic!("{name} refused {input} after its build: {error}"));
        assert_eq!(before, after, "{name}");
        assert_eq!(after.to_string(), expected, "{name}");
    }
}

/// A model input with an empty feature axis is refused, and the refusal names the axis
///
/// An axis of 0 elements carries no data. Many layers pass an extent through unchanged, so
/// without this rule a whole stack builds and then produces empty tensors, and the emptiness
/// only shows itself in the output. The batch axis is exempt, because a batch of 0 is an empty
/// dataset that `fit` and `predict` already refuse when the data arrives
#[test]
fn a_model_refuses_an_input_shape_with_an_empty_feature_axis() {
    let error = SequentialBuilder::new()
        .add(Flatten::new())
        .build(&Shape::known(&[1, 0, 4]))
        .err()
        .expect("a 0 extent must not build");
    let text = error.to_string();
    assert!(
        text.contains("axis 1") && text.contains("0 elements"),
        "the refusal must name the empty axis, got {text}"
    );

    // The same stack with a free batch axis and no empty extent builds
    SequentialBuilder::new()
        .add(Flatten::new())
        .build(&Shape::with_free_batch(&[1, 3, 4]))
        .expect("a free batch axis is not an empty axis");
}
