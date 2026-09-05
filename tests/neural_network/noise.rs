//! Integration tests for the GaussianNoise and GaussianDropout layers.
//!
//! GaussianNoise adds N(0, stddev) to the input. GaussianDropout multiplies the input by
//! N(1, stddev), where stddev = sqrt(rate / (1 - rate)).
//!
//! Statistical assertions use large (10,000-element) all-ones tensors, so the sample mean
//! and sample std converge tightly to the population parameters, within a +/-0.05 tolerance.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise;
use rustyml::neural_network::traits::{Layer, LayerBase, UnaryLayer};
use rustyml::{error::Error, neural_network::NnError};

// helpers

/// Sample mean of every element in a Tensor
fn tensor_mean(t: &Tensor) -> f64 {
    let sum: f64 = t.iter().map(|&v| v as f64).sum();
    sum / t.len() as f64
}

/// Sample standard deviation of every element in a Tensor
fn tensor_std(t: &Tensor) -> f64 {
    let n = t.len() as f64;
    let mean = tensor_mean(t);
    let var: f64 = t.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

// GaussianNoise: constructor validation

/// An invalid stddev (negative, or non-finite NaN or +Inf) is rejected at construction
/// instead of panicking later inside `Normal::new`.
#[test]
fn gaussian_noise_invalid_stddev_returns_err() {
    for bad in [-0.1f32, f32::NAN, f32::INFINITY] {
        let result = GaussianNoise::new(bad);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected Err(InvalidParameter) for stddev={bad}, got {:?}",
            result
        );
    }
}

/// stddev = 0 is accepted at construction
#[test]
fn gaussian_noise_zero_stddev_construction_ok() {
    assert!(GaussianNoise::new(0.0).is_ok());
}

/// Positive stddev is accepted at construction
#[test]
fn gaussian_noise_positive_stddev_construction_ok() {
    assert!(GaussianNoise::new(0.5).is_ok());
}

// GaussianNoise: identity paths

/// In eval mode the layer returns a tensor element-wise equal to the input
#[test]
fn gaussian_noise_eval_mode_is_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(1.0).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// An inference pass is always the identity, whatever a training pass would draw
#[test]
fn gaussian_noise_predict_is_always_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    // An inference context gives the identity, at a stddev that a training pass would use
    let mut layer = GaussianNoise::new(1.0).unwrap();
    layer.build(&Shape::known(&[2, 3])).unwrap();
    let output = layer.forward(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// 2 inference passes agree, and both are the identity
#[test]
fn gaussian_noise_predict_equals_forward_in_eval_mode() {
    let input_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 1.0).collect();
    let input = Array::from_shape_vec((3, 4), input_data)
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(0.5).unwrap();
    layer.build(&Shape::known(&[3, 4])).unwrap();

    let forward_out = layer.forward(&input, &mut Ctx::inference()).unwrap();
    let predict_out = layer.forward(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&predict_out, &forward_out, 1e-6f32);
}

/// stddev = 0 is the identity even in training mode (N(0, 0) = 0)
#[test]
fn gaussian_noise_zero_stddev_training_is_identity() {
    let input: Tensor = Array::from_shape_vec((3,), vec![1.0f32, -2.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(0.0).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

// GaussianNoise: statistical properties in training mode

/// E[output] ~= input when training, since output = input + N(0, stddev)
#[test]
fn gaussian_noise_training_preserves_mean() {
    const N: usize = 10_000;
    let input_val = 2.0f32;
    let stddev = 0.5f32;

    let input = Array::from_shape_vec((N,), vec![input_val; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(stddev).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
    let mean = tensor_mean(&output);

    assert!(
        (mean - input_val as f64).abs() < 0.05,
        "GaussianNoise training mean: expected about {:.4}, got {:.6}",
        input_val,
        mean
    );
}

/// std(output - input) ~= stddev when training, since output - input = N(0, stddev)
#[test]
fn gaussian_noise_training_noise_std_matches_stddev() {
    const N: usize = 10_000;

    for &stddev in &[0.5f32, 0.2f32] {
        let input = Array::from_shape_vec((N,), vec![3.0f32; N])
            .unwrap()
            .into_dyn();

        let mut layer = GaussianNoise::new(stddev).unwrap();

        let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
        // Noise = output - input element-wise
        let noise: Tensor = &output - &input;
        let std_val = tensor_std(&noise);

        assert!(
            (std_val - stddev as f64).abs() < 0.05,
            "GaussianNoise stddev={}: expected std about {:.4}, got {:.6}",
            stddev,
            stddev,
            std_val
        );
    }
}

/// Additive model: a large positive input plus large stddev can yield negative outputs
#[test]
fn gaussian_noise_training_can_produce_negative_values() {
    const N: usize = 10_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(5.0).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
    let neg_count = output.iter().filter(|&&v| v < 0.0).count();

    assert!(
        neg_count > 100,
        "GaussianNoise(stddev=5) on input=1: expected >100 negative outputs, got {}",
        neg_count
    );
}

/// 2 consecutive training passes do not produce identical outputs
///
/// The forward pass draws from a copy of the random stream that lives in the context, and
/// `apply_state` moves that copy into the layer. The model does that after every pass, and the
/// next pass therefore starts where the last one stopped
#[test]
fn gaussian_noise_training_consecutive_calls_differ() {
    const N: usize = 10_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(0.5).unwrap();

    let mut ctx = Ctx::training();
    let out1 = layer.forward_mut(&input, &mut ctx).unwrap();
    layer.apply_state(&mut ctx.state_slot(0));

    let mut ctx = Ctx::training();
    let out2 = layer.forward(&input, &mut ctx).unwrap();

    let all_equal = out1.iter().zip(out2.iter()).all(|(a, b)| a == b);
    assert!(
        !all_equal,
        "two consecutive training forward() calls must differ"
    );
}

// GaussianNoise: error paths

/// forward() accepts a shape that the build shape does not name
///
/// The layer owns no array and reads no extent, so nothing about the build shape reaches the
/// transform. Keras 3 marks `GaussianNoise` built at construction for the same reason
#[test]
fn gaussian_noise_forward_accepts_a_shape_the_build_did_not_name() {
    let mut layer = GaussianNoise::new(0.1).unwrap();
    // Build for [4, 4], then supply a [3, 3] tensor
    layer.build(&Shape::known(&[4, 4])).unwrap();
    let other_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();

    let output = layer
        .forward(&other_input, &mut Ctx::training())
        .unwrap_or_else(|e| panic!("forward refused another shape: {e:?}"));
    assert_eq!(output.shape(), &[3, 3]);
}

/// An inference pass takes the same freedom as a training pass, and it refuses an unbuilt
/// layer alone
#[test]
fn gaussian_noise_predict_accepts_a_shape_the_build_did_not_name() {
    let mut layer = GaussianNoise::new(0.1).unwrap();
    let other_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();
    assert!(
        matches!(
            layer.forward(&other_input, &mut Ctx::inference()),
            Err(Error::NeuralNetwork(NnError::NotBuilt("GaussianNoise")))
        ),
        "an inference pass must refuse a layer that holds no build"
    );

    layer.build(&Shape::known(&[4, 4])).unwrap();
    let output = layer
        .forward(&other_input, &mut Ctx::inference())
        .unwrap_or_else(|e| panic!("an inference pass refused another shape: {e:?}"));
    assert_eq!(output.shape(), &[3, 3]);
}

/// 1 layer serves every rank
///
/// The layer owns no array and reads no extent, so a rank cannot be wrong. Keras 3 takes the
/// same position: the layer is built at construction, and 1 instance passes a rank-1, a
/// rank-2, and a rank-3 tensor. The build shape here names rank 1, and the rank-3 tensor
/// passes all the same
#[test]
fn gaussian_noise_accepts_any_rank() {
    let mut layer = GaussianNoise::new(0.1).unwrap();
    layer.build(&Shape::known(&[5])).unwrap();
    // Supply tensors of completely different shapes. Both must succeed.
    let a: Tensor = Array::from_shape_vec((5,), vec![1.0f32; 5])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec((2, 3, 4), vec![1.0f32; 24])
        .unwrap()
        .into_dyn();
    assert!(layer.forward(&a, &mut Ctx::training()).is_ok());
    assert!(layer.forward(&b, &mut Ctx::training()).is_ok());
    assert!(layer.forward(&a, &mut Ctx::inference()).is_ok());
    assert!(layer.forward(&b, &mut Ctx::inference()).is_ok());
}

// GaussianNoise: metadata

#[test]
fn gaussian_noise_layer_type_and_output_shape() {
    let mut layer = GaussianNoise::new(0.3).unwrap();
    layer.build(&Shape::known(&[32, 128])).unwrap();
    assert_eq!(layer.layer_type(), "GaussianNoise");
    assert_eq!(layer.output_shape(), "(None, 128)");
}

#[test]
fn gaussian_noise_unbuilt_output_shape_is_unknown() {
    let layer = GaussianNoise::new(0.3).unwrap();
    assert_eq!(layer.output_shape(), "Unknown");
}

// GaussianDropout

// GaussianDropout: constructor validation

/// An out-of-range rate is rejected: the valid interval is [0, 1). rate = 1.0 (the excluded
/// upper bound), rate > 1.0, and a negative rate all fail.
#[test]
fn gaussian_dropout_invalid_rate_returns_err() {
    for bad in [1.0f32, 1.5, -0.1] {
        let result = GaussianDropout::new(bad);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected Err(InvalidParameter) for rate={bad}, got {:?}",
            result
        );
    }
}

/// rate = 0.0 is accepted at construction
#[test]
fn gaussian_dropout_rate_zero_construction_ok() {
    assert!(GaussianDropout::new(0.0).is_ok());
}

/// rate = 0.999 is accepted (just below 1.0)
#[test]
fn gaussian_dropout_rate_near_one_construction_ok() {
    assert!(GaussianDropout::new(0.999).is_ok());
}

// GaussianDropout: identity paths

/// In eval mode the layer returns a tensor element-wise equal to the input
#[test]
fn gaussian_dropout_eval_mode_is_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// An inference pass is always the identity, whatever a training pass would draw
#[test]
fn gaussian_dropout_predict_is_always_identity() {
    let input_data = vec![2.0f32, -1.5, 0.3, 4.0, -3.0, 1.1];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.3).unwrap();
    layer.build(&Shape::known(&[2, 3])).unwrap();
    let output = layer.forward(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// 2 inference passes agree, and both are the identity
#[test]
fn gaussian_dropout_predict_equals_forward_in_eval_mode() {
    let input_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 1.0).collect();
    let input = Array::from_shape_vec((3, 4), input_data)
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.4).unwrap();
    layer.build(&Shape::known(&[3, 4])).unwrap();

    let forward_out = layer.forward(&input, &mut Ctx::inference()).unwrap();
    let predict_out = layer.forward(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&predict_out, &forward_out, 1e-6f32);
}

/// rate = 0 is the identity even in training mode (stddev = 0, noise = N(1, 0) = 1)
#[test]
fn gaussian_dropout_rate_zero_training_is_identity() {
    let input: Tensor = Array::from_shape_vec((3,), vec![1.0f32, -2.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.0).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

// GaussianDropout: statistical properties in training mode

/// E[output] ~= input when training, since output = input * N(1, stddev)
#[test]
fn gaussian_dropout_training_preserves_mean() {
    const N: usize = 10_000;
    let input_val = 1.0f32;
    // rate = 0.5 => stddev = sqrt(0.5 / 0.5) = 1.0
    let rate = 0.5f32;

    let input = Array::from_shape_vec((N,), vec![input_val; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(rate).unwrap();

    let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
    let mean = tensor_mean(&output);

    assert!(
        (mean - input_val as f64).abs() < 0.05,
        "GaussianDropout training mean: expected about {:.4}, got {:.6}",
        input_val,
        mean
    );
}

/// std(output) ~= sqrt(rate / (1 - rate)) when training with all-ones input,
/// validating the stddev formula defined in the source
#[test]
fn gaussian_dropout_training_noise_std_matches_formula() {
    const N: usize = 10_000;
    // 2 distinct rates pin the formula: rate=0.5 => stddev = 1.0,
    // rate=0.25 => stddev = sqrt(1/3) ~= 0.5774
    let cases: &[(f32, f64)] = &[
        (0.5, 1.0),
        (0.25, (1.0_f64 / 3.0).sqrt()), // ~= 0.5774
    ];

    for &(rate, expected_std) in cases {
        let input = Array::from_shape_vec((N,), vec![1.0f32; N])
            .unwrap()
            .into_dyn();

        let mut layer = GaussianDropout::new(rate).unwrap();

        let output = layer.forward_mut(&input, &mut Ctx::training()).unwrap();
        // With input=1, output = noise, so std(output) = std(noise)
        let std_val = tensor_std(&output);

        assert!(
            (std_val - expected_std).abs() < 0.05,
            "GaussianDropout rate={}: expected std about {:.4}, got {:.6}",
            rate,
            expected_std,
            std_val
        );
    }
}

/// 2 consecutive training passes do not produce identical outputs
///
/// The forward pass draws from a copy of the random stream that lives in the context, and
/// `apply_state` moves that copy into the layer. The model does that after every pass, and the
/// next pass therefore starts where the last one stopped
#[test]
fn gaussian_dropout_training_consecutive_calls_differ() {
    const N: usize = 10_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5).unwrap();

    let mut ctx = Ctx::training();
    let out1 = layer.forward_mut(&input, &mut ctx).unwrap();
    layer.apply_state(&mut ctx.state_slot(0));

    let mut ctx = Ctx::training();
    let out2 = layer.forward(&input, &mut ctx).unwrap();

    let all_equal = out1.iter().zip(out2.iter()).all(|(a, b)| a == b);
    assert!(
        !all_equal,
        "two consecutive training forward() calls must differ"
    );
}

// GaussianDropout: error paths

/// forward() accepts a shape that the build shape does not name
///
/// The layer owns no array and reads no extent, so nothing about the build shape reaches the
/// transform. Keras 3 marks `GaussianDropout` built at construction for the same reason
#[test]
fn gaussian_dropout_forward_accepts_a_shape_the_build_did_not_name() {
    let mut layer = GaussianDropout::new(0.3).unwrap();
    // Build for [4, 4], then supply a [3, 3] tensor
    layer.build(&Shape::known(&[4, 4])).unwrap();
    let other_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();

    let output = layer
        .forward(&other_input, &mut Ctx::training())
        .unwrap_or_else(|e| panic!("forward refused another shape: {e:?}"));
    assert_eq!(output.shape(), &[3, 3]);
}

/// An inference pass takes the same freedom as a training pass, and it refuses an unbuilt
/// layer alone
#[test]
fn gaussian_dropout_predict_accepts_a_shape_the_build_did_not_name() {
    let mut layer = GaussianDropout::new(0.3).unwrap();
    let other_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();
    assert!(
        matches!(
            layer.forward(&other_input, &mut Ctx::inference()),
            Err(Error::NeuralNetwork(NnError::NotBuilt("GaussianDropout")))
        ),
        "an inference pass must refuse a layer that holds no build"
    );

    layer.build(&Shape::known(&[4, 4])).unwrap();
    let output = layer
        .forward(&other_input, &mut Ctx::inference())
        .unwrap_or_else(|e| panic!("an inference pass refused another shape: {e:?}"));
    assert_eq!(output.shape(), &[3, 3]);
}

/// 1 layer serves every rank
///
/// The layer owns no array and reads no extent, so a rank cannot be wrong. Keras 3 takes the
/// same position: the layer is built at construction, and 1 instance passes a rank-1, a
/// rank-2, and a rank-3 tensor. The build shape here names rank 1, and the rank-3 tensor
/// passes all the same
#[test]
fn gaussian_dropout_accepts_any_rank() {
    let mut layer = GaussianDropout::new(0.3).unwrap();
    layer.build(&Shape::known(&[5])).unwrap();
    let a: Tensor = Array::from_shape_vec((5,), vec![1.0f32; 5])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec((2, 3, 4), vec![1.0f32; 24])
        .unwrap()
        .into_dyn();
    assert!(layer.forward(&a, &mut Ctx::training()).is_ok());
    assert!(layer.forward(&b, &mut Ctx::training()).is_ok());
    assert!(layer.forward(&a, &mut Ctx::inference()).is_ok());
    assert!(layer.forward(&b, &mut Ctx::inference()).is_ok());
}

/// In training mode, backward multiplies by the noise that the forward pass parked. Without a
/// prior forward pass, the context holds no noise, so it errors, matching Dense's and Dropout's
/// contract.
#[test]
fn gaussian_dropout_backward_without_forward_errors() {
    let grad: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32; 6])
        .unwrap()
        .into_dyn();
    let layer = GaussianDropout::new(0.3).unwrap();

    let result = layer.backward(&grad, &mut Ctx::training());
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected Err(ForwardPassNotRun) without a prior forward, got {:?}",
        result
    );
}

/// backward reuses the exact multiplicative noise from forward (y = x * noise, so dx = g * noise).
/// With input = ones, output equals noise, so backward(g) must equal g * output.
#[test]
fn gaussian_dropout_backward_multiplies_by_forward_noise() {
    let input: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32; 6])
        .unwrap()
        .into_dyn();
    let mut layer = GaussianDropout::new(0.3).unwrap().with_random_state(42);
    let mut ctx = Ctx::training();

    // With input = ones, output = ones * noise = noise
    let output = layer.forward_mut(&input, &mut ctx).unwrap();

    let grad: Tensor = Array::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&grad, &mut ctx).unwrap();

    // grad_input should be grad * noise == grad * output
    let expected = &grad * &output;
    for (got, exp) in grad_input.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
    }
}

/// 2 passes over the same input give the same gradient
///
/// The context owns the noise, and the backward pass takes it, so a second gradient needs a
/// second pass. The 2 passes agree to the bit, because `forward` takes `&self` and leaves the
/// random stream of the layer where it was
#[test]
fn gaussian_dropout_backward_is_idempotent() {
    let input: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32; 6])
        .unwrap()
        .into_dyn();
    let mut layer = GaussianDropout::new(0.3).unwrap().with_random_state(42);
    layer.build(&Shape::known(&[2, 3])).unwrap();

    let grad: Tensor = Array::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let _ = layer.forward(&input, &mut ctx).unwrap();
    let first = layer.backward(&grad, &mut ctx).unwrap();

    let mut ctx = Ctx::training();
    let _ = layer.forward(&input, &mut ctx).unwrap();
    let second = layer.backward(&grad, &mut ctx).unwrap();
    for (a, b) in first.iter().zip(second.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 0.0);
    }
}

/// During inference (training = false) forward is the identity, so backward passes the gradient
/// through unchanged
#[test]
fn gaussian_dropout_backward_inference_is_passthrough() {
    let grad: Tensor = Array::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        .unwrap()
        .into_dyn();
    let layer = GaussianDropout::new(0.3).unwrap();

    let grad_input = layer.backward(&grad, &mut Ctx::inference()).unwrap();
    for (got, exp) in grad_input.iter().zip(grad.iter()) {
        assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
    }
}

// GaussianDropout: metadata

#[test]
fn gaussian_dropout_layer_type_and_output_shape() {
    let mut layer = GaussianDropout::new(0.3).unwrap();
    layer.build(&Shape::known(&[32, 128])).unwrap();
    assert_eq!(layer.layer_type(), "GaussianDropout");
    assert_eq!(layer.output_shape(), "(None, 128)");
}

#[test]
fn gaussian_dropout_empty_input_shape_output_shape_unknown() {
    let layer = GaussianDropout::new(0.3).unwrap();
    assert_eq!(layer.output_shape(), "Unknown");
}

// Mode-switching symmetry

/// A training context enables the noise, and an inference context disables it
#[test]
fn gaussian_noise_mode_switching_routes_correctly() {
    const N: usize = 1_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(2.0).unwrap();

    // Eval mode is the identity
    let eval_out = layer.forward_mut(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&eval_out, &input, 1e-6f32);

    // Training mode now differs from the input
    let train_out = layer.forward(&input, &mut Ctx::training()).unwrap();
    let any_different = train_out.iter().zip(input.iter()).any(|(a, b)| a != b);
    assert!(any_different, "training mode must add noise");

    // Back to eval mode: the identity again
    let eval_out2 = layer.forward(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&eval_out2, &input, 1e-6f32);
}

/// Same mode-switching symmetry for GaussianDropout
#[test]
fn gaussian_dropout_mode_switching_routes_correctly() {
    const N: usize = 1_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5).unwrap();

    // Eval mode is the identity
    let eval_out = layer.forward_mut(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&eval_out, &input, 1e-6f32);

    // Training mode now differs from the input
    let train_out = layer.forward(&input, &mut Ctx::training()).unwrap();
    let any_different = train_out.iter().zip(input.iter()).any(|(a, b)| a != b);
    assert!(any_different, "training mode must multiply by noise");

    // Back to eval mode: the identity again
    let eval_out2 = layer.forward(&input, &mut Ctx::inference()).unwrap();
    crate::common::assert_allclose(&eval_out2, &input, 1e-6f32);
}

/// GaussianNoise::backward is a pure gradient pass-through, since the noise does not depend on x.
/// d(y)/dx = 1, so the upstream gradient returns unchanged.
#[test]
fn gaussian_noise_backward_passes_gradient_through_unchanged() {
    let mut layer = GaussianNoise::new(1.0).unwrap();
    // Realistic call order: a training-mode forward first, then backward
    let mut ctx = Ctx::training();
    let x: Tensor = Array::from_elem((2, 3), 1.0f32).into_dyn();
    let _ = layer.forward_mut(&x, &mut ctx).unwrap();

    let grad: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad, &mut ctx).unwrap();

    // Pure pass-through => byte-identical gradient (zero epsilon)
    crate::common::assert_allclose(&grad_in, &grad, 0.0f32);
}
