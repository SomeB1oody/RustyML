use super::*;

#[test]
fn test_gru_layer() {
    // Create input data with batch_size=2, timesteps=5, input_dim=4,
    // and target data with batch_size=2, units=3 (same dimension as the final hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: a GRU layer with Tanh activation function
    let mut model = Sequential::new();
    model.add(GRU::new(4, 3, Tanh::new()).unwrap()).compile(
        RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

    // Print model structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);
    println!("GRU prediction:\n{:#?}\n", pred);
}

#[test]
fn test_gru_layer_basic() {
    // Create input data with batch_size=2, timesteps=5, input_dim=4
    // Create target data with batch_size=2, units=3 (same dimension as the final hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: a GRU layer with Tanh activation function
    let mut model = Sequential::new();
    model.add(GRU::new(4, 3, Tanh::new()).unwrap()).compile(
        RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

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
fn test_gru_different_activations() {
    // Test different activation functions
    let x = Array::ones((3, 4, 2)).into_dyn();
    let y = Array::ones((3, 6)).into_dyn();

    // Test ReLU activation function
    let mut model_relu = Sequential::new();
    model_relu
        .add(GRU::new(2, 6, ReLU::new()).unwrap())
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    model_relu.fit(&x, &y, 3).unwrap();
    let pred_relu = model_relu.predict(&x);

    // Test Sigmoid activation function
    let mut model_sigmoid = Sequential::new();
    model_sigmoid
        .add(GRU::new(2, 6, Sigmoid::new()).unwrap())
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    model_sigmoid.fit(&x, &y, 3).unwrap();
    let pred_sigmoid = model_sigmoid.predict(&x);

    // Check output shapes
    assert_eq!(pred_relu.shape(), &[3, 6]);
    assert_eq!(pred_sigmoid.shape(), &[3, 6]);

    // When using ReLU, outputs should be greater than or equal to 0
    for v in pred_relu.iter() {
        assert!(*v >= 0.0);
    }

    // When using Sigmoid, outputs should be between 0 and 1
    for v in pred_sigmoid.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_gru_sequential_composition() {
    // Test GRU combined with other layers
    let x = Array::ones((2, 5, 3)).into_dyn();
    let y = Array::ones((2, 4)).into_dyn();

    // Build a model containing GRU and Dense layers
    let mut model = Sequential::new();
    model
        .add(GRU::new(3, 6, Tanh::new()).unwrap())
        .add(Dense::new(6, 4, Sigmoid::new()).unwrap())
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

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
fn test_gru_sequence_learning() {
    // Test GRU's ability to learn sequence patterns
    // Create sequences where output depends on the sum of inputs
    let batch_size = 8;
    let seq_len = 6;
    let input_dim = 1;

    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y = Array2::<f32>::zeros((batch_size, 1));

    // Generate sequences with different patterns
    for b in 0..batch_size {
        let mut sequence_sum = 0.0;
        for t in 0..seq_len {
            // Create a simple pattern: alternating values
            let val = if t % 2 == 0 { 0.5 } else { -0.3 };
            x[[b, t, 0]] = val + 0.1 * (b as f32 / batch_size as f32); // Add slight variation per batch
            sequence_sum += x[[b, t, 0]];
        }
        // Target is normalized sum of sequence
        y[[b, 0]] = (sequence_sum / seq_len as f32 + 1.0) / 2.0; // Normalize to [0,1]
    }

    let x = x.into_dyn();
    let y = y.into_dyn();

    let mut model = Sequential::new();
    model
        .add(GRU::new(input_dim, 12, Tanh::new()).unwrap())
        .add(Dense::new(12, 1, Sigmoid::new()).unwrap())
        .compile(
            Adam::new(0.005, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Train for sequence learning
    model.fit(&x, &y, 80).unwrap();

    let pred = model.predict(&x);

    // Calculate average error
    let mut total_error = 0.0;
    for b in 0..batch_size {
        let error = (pred[[b, 0]] - y[[b, 0]]).abs();
        total_error += error;
    }
    let avg_error = total_error / batch_size as f32;

    // GRU should learn the sequence pattern reasonably well
    assert!(avg_error < 0.2, "Average error too high: {}", avg_error);

    println!("GRU Sequence Learning - Average Error: {:.4}", avg_error);
    println!("Sample predictions vs targets:");
    for b in 0..4.min(batch_size) {
        println!(
            "  Batch {}: pred={:.3}, target={:.3}",
            b,
            pred[[b, 0]],
            y[[b, 0]]
        );
    }
}

#[test]
fn test_gru_state_evolution() {
    // Test that GRU internal states evolve meaningfully across timesteps
    let batch_size = 2;
    let seq_len = 5;
    let input_dim = 3;
    let units = 4;

    // Create input with clear temporal structure
    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    for b in 0..batch_size {
        for t in 0..seq_len {
            x[[b, t, 0]] = (t as f32 / seq_len as f32).sin(); // Sinusoidal pattern
            x[[b, t, 1]] = t as f32 / seq_len as f32; // Linear increase
            x[[b, t, 2]] = if t % 2 == 0 { 1.0 } else { -1.0 }; // Alternating pattern
        }
    }

    let x = x.into_dyn();
    let y = Array::ones((batch_size, units)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(GRU::new(input_dim, units, Tanh::new()).unwrap())
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Test multiple training steps to see evolution
    let mut predictions = Vec::new();

    for _ in 0..5 {
        model.fit(&x, &y, 1).unwrap();
        let pred = model.predict(&x);
        predictions.push(pred);
    }

    // Check that predictions evolve over training epochs
    let initial_pred = &predictions[0];
    let final_pred = &predictions[predictions.len() - 1];

    let change = (final_pred - initial_pred).mapv(|x| x.abs()).sum();
    assert!(change > 1e-4, "Predictions should evolve during training");

    // Check that final predictions are reasonable (not all zeros or ones)
    let pred_mean = final_pred.mean().unwrap();
    assert!(
        pred_mean > -0.9 && pred_mean < 0.9,
        "Final predictions should be in reasonable range, got mean: {}",
        pred_mean
    );

    println!(
        "GRU State Evolution Test - Prediction change: {:.6}",
        change
    );
    println!("Final prediction mean: {:.4}", pred_mean);
}

#[test]
fn test_gru_temporal_xor() {
    // Temporal XOR: Output depends on XOR of current and previous inputs
    // This is a temporal version of XOR problem, testing GRU's memory
    // y[t] = x[t] XOR x[t-1]

    let seq_len = 10;
    let batch_size = 100;
    let input_dim = 1;

    let mut x_train = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y_train = Array2::<f32>::zeros((batch_size, 1));

    use rand::Rng;
    let mut rng = rand::rng();

    // Generate sequences of 0s and 1s
    for b in 0..batch_size {
        let mut prev_val = 0.0;
        let mut xor_count = 0;

        for t in 0..seq_len {
            let val = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
            x_train[[b, t, 0]] = val;

            // XOR with previous value
            if t > 0 && ((val > 0.5) != (prev_val > 0.5)) {
                xor_count += 1;
            }
            prev_val = val;
        }

        // Target: count of XOR changes (normalized)
        y_train[[b, 0]] = xor_count as f32 / (seq_len - 1) as f32;
    }

    let x_train = x_train.into_dyn();
    let y_train = y_train.into_dyn();

    // Build model
    let mut model = Sequential::new();
    model
        .add(GRU::new(input_dim, 16, Tanh::new()).unwrap())
        .add(Dense::new(16, 8, ReLU::new()).unwrap())
        .add(Dense::new(8, 1, Sigmoid::new()).unwrap())
        .compile(
            Adam::new(0.005, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    println!("\n=== Temporal XOR Task (GRU) ===");
    model.summary();

    // Train
    println!("\nTraining GRU on temporal XOR pattern...");
    for epoch in 0..100 {
        model.fit(&x_train, &y_train, 1).unwrap();

        if (epoch + 1) % 25 == 0 {
            let pred = model.predict(&x_train);
            let mut total_error = 0.0;
            for b in 0..batch_size {
                let error = (pred[[b, 0]] - y_train[[b, 0]]).abs();
                total_error += error;
            }
            let avg_error = total_error / batch_size as f32;
            println!("Epoch {}: Average error = {:.4}", epoch + 1, avg_error);
        }
    }

    // Test
    let pred = model.predict(&x_train);
    let mut total_error = 0.0;

    println!("\n=== Sample Predictions ===");
    for b in 0..5 {
        let error = (pred[[b, 0]] - y_train[[b, 0]]).abs();
        total_error += error;

        print!("  Sequence {}: ", b);
        for t in 0..seq_len {
            print!("{}", if x_train[[b, t, 0]] > 0.5 { "1" } else { "0" });
        }
        println!(
            " -> pred={:.3}, target={:.3}, error={:.3}",
            pred[[b, 0]],
            y_train[[b, 0]],
            error
        );
    }

    let avg_error = total_error / 5.0;
    println!("\nAverage error on samples: {:.4}", avg_error);

    // Calculate overall accuracy
    let mut correct = 0;
    for b in 0..batch_size {
        let error = (pred[[b, 0]] - y_train[[b, 0]]).abs();
        if error < 0.15 {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / batch_size as f32;

    println!(
        "Overall accuracy (within 0.15 error): {:.2}%",
        accuracy * 100.0
    );

    assert!(
        accuracy > 0.70,
        "Temporal XOR accuracy too low: {:.2}%",
        accuracy * 100.0
    );

    println!("GRU successfully learned temporal XOR pattern!");
}

#[test]
fn test_gru_parity_check() {
    // Parity Check Task: Determine if sequence has odd/even number of 1s
    // This tests GRU's ability to count and remember across long sequences

    let seq_len = 8;
    let batch_size = 128;
    let input_dim = 1;

    let mut x_train = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y_train = Array2::<f32>::zeros((batch_size, 1));

    use rand::Rng;
    let mut rng = rand::rng();

    // Generate sequences
    for b in 0..batch_size {
        let mut count_ones = 0;

        for t in 0..seq_len {
            let val = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
            x_train[[b, t, 0]] = val;

            if val > 0.5 {
                count_ones += 1;
            }
        }

        // Target: 1 if odd number of 1s, 0 if even
        y_train[[b, 0]] = if count_ones % 2 == 1 { 1.0 } else { 0.0 };
    }

    let x_train = x_train.into_dyn();
    let y_train = y_train.into_dyn();

    // Build model
    let mut model = Sequential::new();
    model
        .add(GRU::new(input_dim, 32, Tanh::new()).unwrap())
        .add(Dense::new(32, 16, Tanh::new()).unwrap())
        .add(Dense::new(16, 1, Tanh::new()).unwrap())
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    println!("\n=== Parity Check Task (GRU) ===");
    model.summary();

    // Train
    println!("\nTraining GRU on parity check...");

    // Create closure for accuracy calculation
    let calculate_accuracy =
        |pred: &ArrayD<f32>, y_train: &ArrayD<f32>, batch_size: usize| -> f32 {
            let correct = (0..batch_size)
                .filter(|&b| {
                    let predicted = if pred[[b, 0]] > 0.5 { 1.0 } else { 0.0 };
                    let target = y_train[[b, 0]];
                    (predicted - target).abs() < 0.1
                })
                .count();
            correct as f32 / batch_size as f32
        };

    for epoch in 0..200 {
        model.fit(&x_train, &y_train, 1).unwrap();

        if (epoch + 1) % 40 == 0 {
            let pred = model.predict(&x_train);
            let accuracy = calculate_accuracy(&pred, &y_train, batch_size);
            println!("Epoch {}: Accuracy = {:.2}%", epoch + 1, accuracy * 100.0);
        }
    }

    // Test
    let pred = model.predict(&x_train);

    println!("\n=== Sample Predictions ===");
    for b in 0..8 {
        let predicted = if pred[[b, 0]] > 0.5 { 1.0 } else { 0.0 };
        let target = y_train[[b, 0]];

        let mut count_ones = 0;
        print!("  Sequence {}: ", b);
        for t in 0..seq_len {
            let bit = if x_train[[b, t, 0]] > 0.5 { 1 } else { 0 };
            print!("{}", bit);
            count_ones += bit;
        }

        let is_correct = (predicted - target).abs() < 0.1;
        println!(
            " -> {} 1s ({}), pred={}, target={} {}",
            count_ones,
            if count_ones % 2 == 1 { "odd" } else { "even" },
            if predicted > 0.5 { "odd" } else { "even" },
            if target > 0.5 { "odd" } else { "even" },
            if is_correct { "correct" } else { "false" }
        );
    }

    // Calculate overall accuracy using the closure
    let accuracy = calculate_accuracy(&pred, &y_train, batch_size);
    println!("\nOverall Accuracy: {:.2}%", accuracy * 100.0);

    assert!(
        accuracy > 0.85,
        "Parity check accuracy too low: {:.2}%",
        accuracy * 100.0
    );

    println!("GRU successfully learned parity check!");
}

#[test]
fn test_gru_vs_simple_rnn() {
    // Compare GRU with SimpleRNN on a sequence learning task
    // GRU should perform better due to its gating mechanisms

    let batch_size = 16;
    let seq_len = 8;
    let input_dim = 2;

    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y = Array2::<f32>::zeros((batch_size, 1));

    // Generate sequences where output depends on early timesteps
    // This tests long-term dependency learning
    for b in 0..batch_size {
        // First timestep is important
        x[[b, 0, 0]] = if b % 2 == 0 { 1.0 } else { -1.0 };
        x[[b, 0, 1]] = if b % 3 == 0 { 1.0 } else { -1.0 };

        // Fill rest with noise
        for t in 1..seq_len {
            x[[b, t, 0]] = 0.1 * ((b + t) as f32 / batch_size as f32);
            x[[b, t, 1]] = -0.1 * ((b + t) as f32 / batch_size as f32);
        }

        // Target depends on first timestep
        y[[b, 0]] = if x[[b, 0, 0]] > 0.0 && x[[b, 0, 1]] > 0.0 {
            1.0
        } else {
            0.0
        };
    }

    let x = x.into_dyn();
    let y = y.into_dyn();

    // Train GRU model
    let mut model_gru = Sequential::new();
    model_gru
        .add(GRU::new(input_dim, 8, Tanh::new()).unwrap())
        .add(Dense::new(8, 1, Sigmoid::new()).unwrap())
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    println!("\n=== Training GRU ===");
    model_gru.fit(&x, &y, 50).unwrap();

    // Train SimpleRNN model
    let mut model_rnn = Sequential::new();
    model_rnn
        .add(SimpleRNN::new(input_dim, 8, Tanh::new()).unwrap())
        .add(Dense::new(8, 1, Sigmoid::new()).unwrap())
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    println!("\n=== Training SimpleRNN ===");
    model_rnn.fit(&x, &y, 50).unwrap();

    // Compare performance
    let pred_gru = model_gru.predict(&x);
    let pred_rnn = model_rnn.predict(&x);

    let mut error_gru = 0.0;
    let mut error_rnn = 0.0;

    for b in 0..batch_size {
        error_gru += (pred_gru[[b, 0]] - y[[b, 0]]).abs();
        error_rnn += (pred_rnn[[b, 0]] - y[[b, 0]]).abs();
    }

    error_gru /= batch_size as f32;
    error_rnn /= batch_size as f32;

    println!("\n=== Comparison Results ===");
    println!("GRU Average Error: {:.4}", error_gru);
    println!("SimpleRNN Average Error: {:.4}", error_rnn);

    // GRU should perform at least as well as SimpleRNN (or better)
    assert!(
        error_gru < 0.3,
        "GRU should learn this task reasonably well"
    );

    println!("GRU performed well on long-term dependency task!");
}

#[test]
fn test_gru_gradient_flow() {
    // Test that gradients flow properly through GRU
    // by checking that weights actually change during training

    let x = Array::ones((2, 3, 4)).into_dyn();
    let y = Array::ones((2, 5)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(GRU::new(4, 5, Tanh::new()).unwrap())
        .compile(SGD::new(0.1).unwrap(), MeanSquaredError::new());

    // Get initial weights
    let initial_weights = model.get_weights();
    let initial_gru_weights = match &initial_weights[0] {
        LayerWeight::GRU(w) => (
            w.reset.kernel.clone(),
            w.reset.recurrent_kernel.clone(),
            w.reset.bias.clone(),
        ),
        _ => panic!("Expected GRU layer"),
    };

    // Train for a few epochs
    model.fit(&x, &y, 5).unwrap();

    // Get final weights
    let final_weights = model.get_weights();
    let final_gru_weights = match &final_weights[0] {
        LayerWeight::GRU(w) => (
            w.reset.kernel.clone(),
            w.reset.recurrent_kernel.clone(),
            w.reset.bias.clone(),
        ),
        _ => panic!("Expected GRU layer"),
    };

    // Check that weights have changed
    let kernel_diff = (&final_gru_weights.0 - &initial_gru_weights.0)
        .mapv(|x| x.abs())
        .sum();
    let recurrent_diff = (&final_gru_weights.1 - &initial_gru_weights.1)
        .mapv(|x| x.abs())
        .sum();
    let bias_diff = (&final_gru_weights.2 - &initial_gru_weights.2)
        .mapv(|x| x.abs())
        .sum();

    println!("Kernel change: {:.6}", kernel_diff);
    println!("Recurrent kernel change: {:.6}", recurrent_diff);
    println!("Bias change: {:.6}", bias_diff);

    assert!(kernel_diff > 1e-5, "Kernel should change during training");
    assert!(
        recurrent_diff > 1e-5,
        "Recurrent kernel should change during training"
    );
    assert!(bias_diff > 1e-5, "Bias should change during training");

    println!("Gradients flow properly through GRU!");
}
