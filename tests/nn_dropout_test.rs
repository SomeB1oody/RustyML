#![cfg(feature = "neural_network")]

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::layer::regularization_layer::dropout_layer::dropout::Dropout;
use rustyml::neural_network::neural_network_trait::Layer;

#[test]
fn test_dropout_forward_pass_dimensions() {
    // Test that dropout preserves input dimensions
    let mut dropout = Dropout::new(0.5, vec![2, 10]).unwrap();
    let input = Array::ones((2, 10)).into_dyn();

    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 10]);
    println!(
        "Dropout dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_dropout_training_mode() {
    // Test that dropout actually drops values during training
    let mut dropout = Dropout::new(0.5, vec![100]).unwrap();
    dropout.set_training(true);

    let input = Array::ones((100,)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Count non-zero values
    let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();

    // With rate=0.5, approximately 50% should be non-zero
    // Allow some statistical variance (30%-70%)
    assert!(
        non_zero_count > 30 && non_zero_count < 70,
        "Expected ~50 non-zero values, got {}",
        non_zero_count
    );

    // Check that non-zero values are scaled correctly (inverted dropout)
    let scale = 1.0 / (1.0 - 0.5);
    for &val in output.iter() {
        if val != 0.0 {
            assert_abs_diff_eq!(val, scale, epsilon = 0.001);
        }
    }

    println!(
        "Dropout training mode test passed: {} out of {} values retained",
        non_zero_count, 100
    );
}

#[test]
fn test_dropout_inference_mode() {
    // Test that dropout passes through input unchanged during inference
    let mut dropout = Dropout::new(0.5, vec![2, 10]).unwrap();
    dropout.set_training(false);

    let input = Array::from_shape_fn((2, 10), |(i, j)| (i * 10 + j) as f32).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // During inference, output should be identical to input
    assert_eq!(output, input);
    println!("Dropout inference mode test passed: output equals input");
}

#[test]
fn test_dropout_rate_zero() {
    // Test that dropout with rate=0 keeps all values
    let mut dropout = Dropout::new(0.0, vec![2, 10]).unwrap();
    dropout.set_training(true);

    let input = Array::ones((2, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // All values should be retained
    assert_eq!(output, input);
    println!("Dropout rate=0 test passed: all values retained");
}

#[test]
fn test_dropout_invalid_rate() {
    // Test that invalid rates are caught
    let dropout_negative = Dropout::new(-0.1, vec![10]);
    let dropout_over_one = Dropout::new(1.5, vec![10]);

    assert!(dropout_negative.is_err());
    assert!(dropout_over_one.is_err());
    println!("Dropout invalid rate test passed");
}

#[test]
fn test_dropout_shape_validation() {
    // Test that shape mismatch is detected
    let mut dropout = Dropout::new(0.5, vec![2, 10]).unwrap();
    let wrong_input = Array::ones((3, 10)).into_dyn();

    let result = dropout.forward(&wrong_input);
    assert!(result.is_err());
    println!("Dropout shape validation test passed");
}

#[test]
fn test_dropout_backward_pass() {
    // Test backward pass preserves gradient dimensions and applies mask
    let mut dropout = Dropout::new(0.5, vec![2, 10]).unwrap();
    dropout.set_training(true);

    let input = Array::ones((2, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    let grad_output = Array::ones((2, 10)).into_dyn();
    let grad_input = dropout.backward(&grad_output).unwrap();

    // Gradient should have same shape as input
    assert_eq!(grad_input.shape(), input.shape());

    // Gradient should be zero where output was zero
    for i in 0..grad_input.len() {
        let out_val = output.as_slice().unwrap()[i];
        let grad_val = grad_input.as_slice().unwrap()[i];

        if out_val == 0.0 {
            assert_eq!(
                grad_val, 0.0,
                "Gradient should be zero where dropout occurred"
            );
        }
    }

    println!("Dropout backward pass test passed");
}

#[test]
fn test_dropout_different_rates() {
    // Test dropout with different rates
    let rates = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    for rate in rates {
        let mut dropout = Dropout::new(rate, vec![1000]).unwrap();
        dropout.set_training(true);

        let input = Array::ones((1000,)).into_dyn();
        let output = dropout.forward(&input).unwrap();

        let retained_count = output.iter().filter(|&&x| x != 0.0).count();
        let retained_ratio = retained_count as f32 / 1000.0;
        let expected_ratio = 1.0 - rate;

        // Allow 10% variance
        assert!(
            (retained_ratio - expected_ratio).abs() < 0.1,
            "Rate {}: expected ~{:.1}% retained, got {:.1}%",
            rate,
            expected_ratio * 100.0,
            retained_ratio * 100.0
        );

        println!(
            "Dropout rate={:.1} test passed: {:.1}% values retained (expected {:.1}%)",
            rate,
            retained_ratio * 100.0,
            expected_ratio * 100.0
        );
    }
}

#[test]
fn test_dropout_maintains_expected_value() {
    // Test that inverted dropout maintains expected value
    let mut dropout = Dropout::new(0.5, vec![1000]).unwrap();
    dropout.set_training(true);

    let input = Array::from_elem((1000,), 2.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Calculate mean of output
    let sum: f32 = output.iter().sum();
    let mean = sum / 1000.0;

    // Mean should be close to input value (2.0) due to scaling
    assert_abs_diff_eq!(mean, 2.0, epsilon = 0.5);
    println!(
        "Dropout expected value test passed: mean = {:.2} (expected 2.0)",
        mean
    );
}

#[test]
fn test_dropout_multidimensional() {
    // Test dropout with different dimensional inputs
    let shapes = vec![vec![10], vec![5, 10], vec![2, 3, 4], vec![2, 3, 4, 5]];

    for shape in shapes {
        let mut dropout = Dropout::new(0.5, shape.clone()).unwrap();
        dropout.set_training(true);

        let input = Array::ones(shape.as_slice()).into_dyn();
        let output = dropout.forward(&input).unwrap();

        assert_eq!(output.shape(), input.shape());
        println!("Dropout multidimensional test passed for shape {:?}", shape);
    }
}

#[test]
fn test_dropout_layer_type() {
    // Test that layer type is correctly reported
    let dropout = Dropout::new(0.5, vec![10]).unwrap();
    assert_eq!(dropout.layer_type(), "Dropout");
    println!("Dropout layer type test passed");
}

#[test]
fn test_dropout_output_shape() {
    // Test that output shape is correctly reported
    let dropout = Dropout::new(0.5, vec![2, 10]).unwrap();
    let shape_str = dropout.output_shape();
    assert!(shape_str.contains("2") && shape_str.contains("10"));
    println!("Dropout output shape test passed: {}", shape_str);
}

#[test]
fn test_dropout_consistency_across_calls() {
    // Test that each forward pass generates different masks
    let mut dropout = Dropout::new(0.5, vec![100]).unwrap();
    dropout.set_training(true);

    let input = Array::ones((100,)).into_dyn();

    let output1 = dropout.forward(&input).unwrap();
    let output2 = dropout.forward(&input).unwrap();

    // The two outputs should be different (different random masks)
    assert_ne!(output1, output2);
    println!("Dropout consistency test passed: different masks on each forward pass");
}
