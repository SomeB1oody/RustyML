use crate::error::ModelError;
use crate::neural_network::layer::regularization_layer::noise_injection_layer::gaussian_dropout::GaussianDropout;
use crate::neural_network::neural_network_trait::Layer;
use ndarray::Array;

#[test]
fn test_gaussian_dropout_new() {
    // Test creating a new GaussianDropout layer
    let dropout_layer = GaussianDropout::new(0.3, vec![32, 128]).unwrap();
    assert_eq!(dropout_layer.layer_type(), "GaussianDropout");
    assert_eq!(dropout_layer.output_shape(), "[32, 128]");
    println!("GaussianDropout creation test passed");
}

#[test]
fn test_gaussian_dropout_negative_rate() {
    // Test that negative rate returns an InputValidationError
    let result = GaussianDropout::new(-0.1, vec![32, 128]);
    assert!(result.is_err());
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Dropout rate must be in range [0, 1)"));
        assert!(msg.contains("-0.1"));
    } else {
        panic!("Expected InputValidationError");
    }
}

#[test]
fn test_gaussian_dropout_rate_equals_one() {
    // Test that rate=1.0 returns an InputValidationError
    let result = GaussianDropout::new(1.0, vec![32, 128]);
    assert!(result.is_err());
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Dropout rate must be in range [0, 1)"));
        assert!(msg.contains("1"));
    } else {
        panic!("Expected InputValidationError");
    }
}

#[test]
fn test_gaussian_dropout_rate_over_one() {
    // Test that rate>1.0 returns an InputValidationError
    let result = GaussianDropout::new(1.5, vec![32, 128]);
    assert!(result.is_err());
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Dropout rate must be in range [0, 1)"));
        assert!(msg.contains("1.5"));
    } else {
        panic!("Expected InputValidationError");
    }
}

#[test]
fn test_gaussian_dropout_forward_dimensions() {
    // Test that GaussianDropout preserves input dimensions
    let mut dropout = GaussianDropout::new(0.3, vec![2, 10]).unwrap();
    let input = Array::ones((2, 10)).into_dyn();

    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 10]);
    println!(
        "GaussianDropout dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_gaussian_dropout_training_mode() {
    // Test that GaussianDropout applies multiplicative noise during training
    let mut dropout = GaussianDropout::new(0.5, vec![100]).unwrap();
    dropout.set_training(true);

    let input = Array::from_elem((100,), 10.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Output shape should match input shape
    assert_eq!(output.shape(), input.shape());

    // With rate=0.5, stddev = sqrt(0.5/(1-0.5)) = 1.0
    // Output should be different from input due to multiplicative noise
    let diff = (&output - &input).mapv(|x| x.abs()).sum();
    assert!(diff > 0.0, "Expected noise to be applied during training");

    // Check that values are not set to zero (unlike standard dropout)
    let zero_count = output.iter().filter(|&&x| x == 0.0).count();
    assert_eq!(
        zero_count, 0,
        "GaussianDropout should not set values to zero"
    );

    println!("GaussianDropout training mode test passed: noise applied, no zeros");
}

#[test]
fn test_gaussian_dropout_inference_mode() {
    // Test that GaussianDropout passes through input unchanged during inference
    let mut dropout = GaussianDropout::new(0.5, vec![2, 10]).unwrap();
    dropout.set_training(false);

    let input = Array::from_shape_fn((2, 10), |(i, j)| (i * 10 + j) as f32).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // During inference, output should be identical to input
    assert_eq!(output, input);
    println!("GaussianDropout inference mode test passed: output equals input");
}

#[test]
fn test_gaussian_dropout_rate_zero() {
    // Test that GaussianDropout with rate=0 passes through input unchanged
    let mut dropout = GaussianDropout::new(0.0, vec![2, 10]).unwrap();
    dropout.set_training(true);

    let input = Array::ones((2, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // With rate=0, stddev=0, so no noise is added
    assert_eq!(output, input);
    println!("GaussianDropout rate=0 test passed: no noise added");
}

#[test]
fn test_gaussian_dropout_shape_validation() {
    // Test that shape mismatch is detected
    let mut dropout = GaussianDropout::new(0.3, vec![2, 10]).unwrap();
    let wrong_input = Array::ones((3, 10)).into_dyn();

    let result = dropout.forward(&wrong_input);
    assert!(result.is_err());
    println!("GaussianDropout shape validation test passed");
}

#[test]
fn test_gaussian_dropout_backward_pass() {
    // Test backward pass preserves gradient dimensions
    let mut dropout = GaussianDropout::new(0.3, vec![2, 10]).unwrap();
    dropout.set_training(true);

    let input = Array::ones((2, 10)).into_dyn();
    let _output = dropout.forward(&input).unwrap();

    let grad_output = Array::from_elem((2, 10), 2.0).into_dyn();
    let grad_input = dropout.backward(&grad_output).unwrap();

    // Gradient should have same shape as input
    assert_eq!(grad_input.shape(), input.shape());

    // Gradient should pass through unchanged (expected value of noise is 1)
    assert_eq!(grad_input, grad_output);

    println!("GaussianDropout backward pass test passed");
}

#[test]
fn test_gaussian_dropout_different_rates() {
    // Test GaussianDropout with different rates
    let rates = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    for rate in rates {
        let mut dropout = GaussianDropout::new(rate, vec![1000]).unwrap();
        dropout.set_training(true);

        let input = Array::from_elem((1000,), 5.0).into_dyn();
        let output = dropout.forward(&input).unwrap();

        // Calculate standard deviation of the multiplicative noise
        let expected_stddev = (rate / (1.0 - rate)).sqrt();

        // Output should be different from input
        let diff = (&output - &input).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Rate {}: Expected noise to be applied", rate);

        // Mean of output should be close to mean of input (noise has mean 1)
        let input_mean: f32 = input.iter().sum::<f32>() / 1000.0;
        let output_mean: f32 = output.iter().sum::<f32>() / 1000.0;

        // Allow larger tolerance for higher rates (more variance)
        let tolerance = 1.0 + expected_stddev;
        assert!(
            (output_mean - input_mean).abs() < tolerance,
            "Rate {:.1}: mean should be preserved. Input mean: {:.2}, Output mean: {:.2}",
            rate,
            input_mean,
            output_mean
        );

        println!(
            "GaussianDropout rate={:.1} test passed: stddev={:.2}, mean preserved ({:.2} -> {:.2})",
            rate, expected_stddev, input_mean, output_mean
        );
    }
}

#[test]
fn test_gaussian_dropout_multidimensional() {
    // Test GaussianDropout with different dimensional inputs
    let shapes = vec![vec![10], vec![5, 10], vec![2, 3, 4], vec![2, 3, 4, 5]];

    for shape in shapes {
        let mut dropout = GaussianDropout::new(0.3, shape.clone()).unwrap();
        dropout.set_training(true);

        let input = Array::ones(shape.as_slice()).into_dyn();
        let output = dropout.forward(&input).unwrap();

        assert_eq!(output.shape(), input.shape());

        // Verify noise was applied
        let diff = (&output - &input).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Expected noise for shape {:?}", shape);

        println!(
            "GaussianDropout multidimensional test passed for shape {:?}",
            shape
        );
    }
}

#[test]
fn test_gaussian_dropout_layer_type() {
    // Test that layer type is correctly reported
    let dropout = GaussianDropout::new(0.3, vec![10]).unwrap();
    assert_eq!(dropout.layer_type(), "GaussianDropout");
    println!("GaussianDropout layer type test passed");
}

#[test]
fn test_gaussian_dropout_output_shape() {
    // Test that output shape is correctly reported
    let dropout = GaussianDropout::new(0.3, vec![2, 10]).unwrap();
    let shape_str = dropout.output_shape();
    assert!(shape_str.contains("2") && shape_str.contains("10"));
    println!("GaussianDropout output shape test passed: {}", shape_str);
}

#[test]
fn test_gaussian_dropout_consistency_across_calls() {
    // Test that each forward pass generates different noise
    let mut dropout = GaussianDropout::new(0.5, vec![100]).unwrap();
    dropout.set_training(true);

    let input = Array::from_elem((100,), 5.0).into_dyn();

    let output1 = dropout.forward(&input).unwrap();
    let output2 = dropout.forward(&input).unwrap();

    // The two outputs should be different (different random noise)
    assert_ne!(output1, output2);
    println!("GaussianDropout consistency test passed: different noise on each forward pass");
}

#[test]
fn test_gaussian_dropout_no_zeros() {
    // Test that GaussianDropout never produces exact zeros (unlike standard dropout)
    let mut dropout = GaussianDropout::new(0.9, vec![1000]).unwrap();
    dropout.set_training(true);

    let input = Array::from_elem((1000,), 1.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Count exact zero values - should be 0 (or extremely rare)
    let zero_count = output.iter().filter(|&&x| x == 0.0).count();

    // GaussianDropout multiplies by noise, so zeros are extremely unlikely
    // (only possible if noise value is exactly 0, which has probability 0 for continuous distribution)
    assert!(
        zero_count < 5,
        "GaussianDropout should not produce zeros, got {} zeros",
        zero_count
    );

    println!(
        "GaussianDropout no-zeros test passed: {} zeros out of 1000 (expected ~0)",
        zero_count
    );
}

#[test]
fn test_gaussian_dropout_statistical_properties() {
    // Test that the noise follows expected statistical properties
    let mut dropout = GaussianDropout::new(0.5, vec![10000]).unwrap();
    dropout.set_training(true);

    let input = Array::from_elem((10000,), 10.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Extract the implied noise: output = input * noise => noise = output / input
    let noise: Vec<f32> = output
        .iter()
        .zip(input.iter())
        .map(|(&o, &i)| o / i)
        .collect();

    // Calculate mean and variance of noise
    let mean: f32 = noise.iter().sum::<f32>() / noise.len() as f32;
    let variance: f32 = noise.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / noise.len() as f32;
    let stddev = variance.sqrt();

    // Expected: mean = 1.0, stddev = sqrt(0.5 / 0.5) = 1.0
    let expected_mean = 1.0;
    let expected_stddev = (0.5 / (1.0 - 0.5f32)).sqrt();

    assert!(
        (mean - expected_mean).abs() < 0.1,
        "Mean should be close to {}, got {}",
        expected_mean,
        mean
    );

    assert!(
        (stddev - expected_stddev).abs() < 0.1,
        "Stddev should be close to {}, got {}",
        expected_stddev,
        stddev
    );

    println!(
        "GaussianDropout statistical properties test passed: mean={:.3} (expected {:.3}), stddev={:.3} (expected {:.3})",
        mean, expected_mean, stddev, expected_stddev
    );
}

#[test]
fn test_gaussian_dropout_stddev_formula() {
    // Test that different rates produce expected standard deviations
    let test_cases = vec![
        (0.2, 0.5), // rate=0.2 => stddev = sqrt(0.2/0.8) = 0.5
        (0.5, 1.0), // rate=0.5 => stddev = sqrt(0.5/0.5) = 1.0
        (0.8, 2.0), // rate=0.8 => stddev = sqrt(0.8/0.2) = 2.0
    ];

    for (rate, expected_stddev) in test_cases {
        let mut dropout = GaussianDropout::new(rate, vec![5000]).unwrap();
        dropout.set_training(true);

        let input = Array::from_elem((5000,), 100.0).into_dyn();
        let output = dropout.forward(&input).unwrap();

        // Extract noise and calculate stddev
        let noise: Vec<f32> = output
            .iter()
            .zip(input.iter())
            .map(|(&o, &i)| o / i)
            .collect();

        let mean: f32 = noise.iter().sum::<f32>() / noise.len() as f32;
        let variance: f32 =
            noise.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / noise.len() as f32;
        let stddev = variance.sqrt();

        assert!(
            (stddev - expected_stddev).abs() < 0.15,
            "Rate {}: expected stddev {:.2}, got {:.2}",
            rate,
            expected_stddev,
            stddev
        );

        println!(
            "GaussianDropout stddev formula test passed for rate={}: stddev={:.3} (expected {:.3})",
            rate, stddev, expected_stddev
        );
    }
}
