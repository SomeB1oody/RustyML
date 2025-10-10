use super::*;

#[test]
fn test_simple_rnn_layer() {
    // Create input with batch_size=2, timesteps=5, input_dim=4,
    // and target with batch_size=2, units=3 (same dimension as the last hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: one SimpleRnn layer with tanh activation
    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(4, 3, Activation::Tanh))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);
    println!("SimpleRnn prediction:\n{:#?}\n", pred);
}

#[test]
fn test_simple_rnn_layer_basic() {
    // Create input data, batch_size=2, timesteps=5, input_dim=4
    // Create target data, batch_size=2, units=3 (same dimension as the last hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: one SimpleRNN layer with tanh activation
    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(4, 3, Activation::Tanh))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);

    // Check output shape
    assert_eq!(pred.shape(), &[2, 3]);
}

#[test]
fn test_simple_rnn_different_activations() {
    // Test different activation functions
    let x = Array::ones((3, 4, 2)).into_dyn();
    let y = Array::ones((3, 6)).into_dyn();

    // Test ReLU activation function
    let mut model_relu = Sequential::new();
    model_relu
        .add(SimpleRNN::new(2, 6, Activation::ReLU))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    model_relu.fit(&x, &y, 3).unwrap();
    let pred_relu = model_relu.predict(&x);

    // Test Sigmoid activation function
    let mut model_sigmoid = Sequential::new();
    model_sigmoid
        .add(SimpleRNN::new(2, 6, Activation::Sigmoid))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    model_sigmoid.fit(&x, &y, 3).unwrap();
    let pred_sigmoid = model_sigmoid.predict(&x);

    // Check output shapes
    assert_eq!(pred_relu.shape(), &[3, 6]);
    assert_eq!(pred_sigmoid.shape(), &[3, 6]);

    // With ReLU, outputs should be greater than or equal to 0
    for v in pred_relu.iter() {
        assert!(*v >= 0.0);
    }

    // With Sigmoid, outputs should be between 0 and 1
    for v in pred_sigmoid.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_simple_rnn_sequential_composition() {
    // Test SimpleRNN combined with other layers
    let x = Array::ones((2, 5, 3)).into_dyn();
    let y = Array::ones((2, 4)).into_dyn();

    // Build model with SimpleRNN and Dense layers
    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(3, 6, Activation::Tanh))
        .add(Dense::new(6, 4, Activation::Sigmoid))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 5).unwrap();

    // Predict
    let pred = model.predict(&x);

    // Check output shape
    assert_eq!(pred.shape(), &[2, 4]);

    // Since the last layer uses Sigmoid activation, all outputs should be between 0 and 1
    for v in pred.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_simple_rnn_overfitting() {
    // Test if the model can overfit a simple dataset
    let x = Array::ones((2, 4, 3)).into_dyn();
    let y = Array::ones((2, 7)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(3, 7, Activation::Tanh))
        .compile(RMSprop::new(0.01, 0.9, 1e-8), MeanSquaredError::new());

    // Train long enough to overfit
    model.fit(&x, &y, 50).unwrap();

    // Predictions should be very close to target values
    let pred = model.predict(&x);
    for (pred_val, target_val) in pred.iter().zip(y.iter()) {
        assert_abs_diff_eq!(*pred_val, *target_val, epsilon = 0.3);
    }
}

#[test]
fn test_simple_rnn_temporal_pattern_recognition() {
    // Test SimpleRNN's ability to recognize temporal patterns
    // Create sequences with specific patterns that require temporal memory
    let batch_size = 6;
    let seq_len = 8;
    let input_dim = 2;

    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y = Array2::<f32>::zeros((batch_size, 1));

    // Pattern 1: Alternating high-low sequence -> output 1.0
    for b in 0..3 {
        for t in 0..seq_len {
            if t % 2 == 0 {
                x[[b, t, 0]] = 0.8;
                x[[b, t, 1]] = 0.2;
            } else {
                x[[b, t, 0]] = 0.2;
                x[[b, t, 1]] = 0.8;
            }
        }
        y[[b, 0]] = 1.0;
    }

    // Pattern 2: Increasing sequence -> output 0.0
    for b in 3..6 {
        for t in 0..seq_len {
            let val = (t as f32) / (seq_len as f32);
            x[[b, t, 0]] = val;
            x[[b, t, 1]] = 1.0 - val;
        }
        y[[b, 0]] = 0.0;
    }

    let x = x.into_dyn();
    let y = y.into_dyn();

    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(input_dim, 8, Activation::Tanh))
        .add(Dense::new(8, 1, Activation::Sigmoid))
        .compile(Adam::new(0.1, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Train the model
    model.fit(&x, &y, 50).unwrap();

    let pred = model.predict(&x);

    // Check pattern recognition accuracy
    let mut correct_predictions = 0;
    for b in 0..batch_size {
        let expected = y[[b, 0]];
        let predicted = pred[[b, 0]];
        let is_correct = if expected > 0.5 {
            predicted > 0.5
        } else {
            predicted <= 0.5
        };

        if is_correct {
            correct_predictions += 1;
        }

        println!(
            "Sample {}: Expected {:.1}, Predicted {:.3}",
            b, expected, predicted
        );
    }

    let accuracy = correct_predictions as f32 / batch_size as f32;
    assert!(
        accuracy >= 0.67,
        "Pattern recognition accuracy too low: {:.2}",
        accuracy
    );

    println!(
        "SimpleRNN Pattern Recognition Accuracy: {:.1}%",
        accuracy * 100.0
    );
}

#[test]
fn test_simple_rnn_sequence_memory() {
    // Test SimpleRNN's short-term memory capacity
    // The first timestep contains the key, later timesteps contain noise
    let batch_size = 8;
    let seq_len = 6;
    let input_dim = 3;

    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y = Array2::<f32>::zeros((batch_size, 2));

    for b in 0..batch_size {
        // Key signal in the first timestep
        if b < 4 {
            x[[b, 0, 0]] = 1.0; // Pattern A
            x[[b, 0, 1]] = 0.0;
            x[[b, 0, 2]] = 0.0;
            y[[b, 0]] = 1.0;
            y[[b, 1]] = 0.0;
        } else {
            x[[b, 0, 0]] = 0.0; // Pattern B
            x[[b, 0, 1]] = 1.0;
            x[[b, 0, 2]] = 0.0;
            y[[b, 0]] = 0.0;
            y[[b, 1]] = 1.0;
        }

        // Add noise in subsequent timesteps
        for t in 1..seq_len {
            x[[b, t, 0]] = 0.1 * ((b * t) as f32).sin();
            x[[b, t, 1]] = 0.1 * ((b * t) as f32).cos();
            x[[b, t, 2]] = 0.05 * (t as f32);
        }
    }

    let x = x.into_dyn();
    let y = y.into_dyn();

    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(input_dim, 12, Activation::Tanh))
        .add(Dense::new(12, 2, Activation::Softmax))
        .compile(Adam::new(0.005, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Train the memory task
    model.fit(&x, &y, 80).unwrap();

    let pred = model.predict(&x);

    // Check memory performance
    let mut correct_predictions = 0;
    for b in 0..batch_size {
        let pred_class = if pred[[b, 0]] > pred[[b, 1]] { 0 } else { 1 };
        let true_class = if y[[b, 0]] > y[[b, 1]] { 0 } else { 1 };

        if pred_class == true_class {
            correct_predictions += 1;
        }

        println!(
            "Sample {}: True class {}, Pred class {}, Confidence [{:.3}, {:.3}]",
            b,
            true_class,
            pred_class,
            pred[[b, 0]],
            pred[[b, 1]]
        );
    }

    let accuracy = correct_predictions as f32 / batch_size as f32;
    assert!(
        accuracy >= 0.6,
        "Memory task accuracy too low: {:.2}",
        accuracy
    );

    println!("SimpleRNN Memory Task Accuracy: {:.1}%", accuracy * 100.0);
}

#[test]
fn test_simple_rnn_vanishing_gradient_susceptibility() {
    // Test SimpleRNN's susceptibility to vanishing gradients with longer sequences
    // Unlike LSTM, SimpleRNN should struggle more with very long sequences
    let batch_size = 3;
    let short_seq_len = 5;
    let long_seq_len = 15;
    let input_dim = 2;
    let units = 4;

    // Create test data for both short and long sequences
    let create_sequence_data = |seq_len: usize| -> (Tensor, Tensor) {
        let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
        let mut y = Array2::<f32>::zeros((batch_size, units));

        for b in 0..batch_size {
            // Important signal at the beginning
            x[[b, 0, 0]] = if b % 2 == 0 { 1.0 } else { -1.0 };
            x[[b, 0, 1]] = 0.5;

            // Noise in middle
            for t in 1..seq_len - 1 {
                x[[b, t, 0]] = 0.1 * (t as f32).sin();
                x[[b, t, 1]] = 0.1 * (t as f32).cos();
            }

            // Target based on first timestep
            for u in 0..units {
                y[[b, u]] = 0.5 + 0.3 * x[[b, 0, 0]];
            }
        }
        (x.into_dyn(), y.into_dyn())
    };

    // Test short sequence
    let (x_short, y_short) = create_sequence_data(short_seq_len);
    let mut model_short = Sequential::new();
    model_short
        .add(SimpleRNN::new(input_dim, units, Activation::Tanh))
        .compile(Adam::new(0.01, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    let initial_loss_short = {
        let pred = model_short.predict(&x_short);
        let diff = &pred - &y_short;
        diff.mapv(|x| x.powi(2)).sum() / pred.len() as f32
    };

    model_short.fit(&x_short, &y_short, 25).unwrap();

    let final_loss_short = {
        let pred = model_short.predict(&x_short);
        let diff = &pred - &y_short;
        diff.mapv(|x| x.powi(2)).sum() / pred.len() as f32
    };

    // Test long sequence
    let (x_long, y_long) = create_sequence_data(long_seq_len);
    let mut model_long = Sequential::new();
    model_long
        .add(SimpleRNN::new(input_dim, units, Activation::Tanh))
        .compile(Adam::new(0.01, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    let initial_loss_long = {
        let pred = model_long.predict(&x_long);
        let diff = &pred - &y_long;
        diff.mapv(|x| x.powi(2)).sum() / pred.len() as f32
    };

    model_long.fit(&x_long, &y_long, 25).unwrap();

    let final_loss_long = {
        let pred = model_long.predict(&x_long);
        let diff = &pred - &y_long;
        diff.mapv(|x| x.powi(2)).sum() / pred.len() as f32
    };

    let improvement_short = (initial_loss_short - final_loss_short) / initial_loss_short;
    let improvement_long = (initial_loss_long - final_loss_long) / initial_loss_long;

    println!("SimpleRNN Vanishing Gradient Analysis:");
    println!(
        "  Short sequence ({} steps): {:.6} -> {:.6} (improvement: {:.1}%)",
        short_seq_len,
        initial_loss_short,
        final_loss_short,
        improvement_short * 100.0
    );
    println!(
        "  Long sequence ({} steps): {:.6} -> {:.6} (improvement: {:.1}%)",
        long_seq_len,
        initial_loss_long,
        final_loss_long,
        improvement_long * 100.0
    );

    // Both should show some improvement, but short sequences should perform better
    assert!(
        improvement_short > 0.05,
        "Short sequence should show significant improvement"
    );

    // Long sequence may struggle more due to vanishing gradients
    if improvement_long < improvement_short * 0.7 {
        println!(
            "  Note: Long sequence shows reduced learning, indicating vanishing gradient effects"
        );
    }
}
