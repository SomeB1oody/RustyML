//! Integration tests for BatchNormalization and LayerNormalization forward, eval,
//! running-stats, and error-path behavior. Expected values come from the mathematical
//! definition; gradient correctness lives in tests/neural_network/gradient_check.rs

use ndarray::ArrayD;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::regularization::normalization::layer_normalization::{
    LayerNormalization, LayerNormalizationAxis,
};
use rustyml::neural_network::traits::Layer;

use crate::common::assert_allclose;

// Helpers

/// Build a 1-D Tensor (shape [n]) from a flat Vec
fn tensor1(data: Vec<f32>) -> rustyml::neural_network::Tensor {
    let n = data.len();
    ArrayD::from_shape_vec(vec![n], data).expect("tensor1: shape/data mismatch")
}

/// Build a 2-D Tensor (shape [rows, cols]) from a flat Vec (row-major)
fn tensor2(data: Vec<f32>, rows: usize, cols: usize) -> rustyml::neural_network::Tensor {
    ArrayD::from_shape_vec(vec![rows, cols], data).expect("tensor2: shape/data mismatch")
}

// BATCH NORMALIZATION

// Constructor validation

/// BatchNormalization::new returns Err(EmptyInput) when input_shape is empty
#[test]
fn bn_constructor_rejects_empty_input_shape() {
    let result = BatchNormalization::new(vec![], 0.9, 1e-5);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput, got {:?}",
        result
    );
}

/// BatchNormalization::new returns Err(InvalidParameter) for out-of-range momentum
/// (>1.0 or <0.0) and for non-positive epsilon (zero or negative)
#[test]
fn bn_constructor_rejects_invalid_scalar_params() {
    // (momentum, epsilon, label) rows; each must yield InvalidParameter
    let cases = [
        (1.5, 1e-5, "momentum > 1.0"),
        (-0.1, 1e-5, "momentum < 0.0"),
        (0.9, 0.0, "epsilon == 0.0"),
        (0.9, -1e-5, "epsilon < 0.0"),
    ];
    for (momentum, epsilon, label) in cases {
        let result = BatchNormalization::new(vec![4, 3], momentum, epsilon);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label} (momentum={momentum}, epsilon={epsilon}), got {:?}",
            result
        );
    }
}

/// BatchNormalization::new succeeds for boundary momentum values 0.0 and 1.0
#[test]
fn bn_constructor_accepts_boundary_momentum_values() {
    assert!(BatchNormalization::new(vec![4, 3], 0.0, 1e-5).is_ok());
    assert!(BatchNormalization::new(vec![4, 3], 1.0, 1e-5).is_ok());
}

// Shape mismatch on forward

/// BatchNormalization forward returns Err(ShapeMismatch) when input shape differs
/// from the declared [4, 3]
#[test]
fn bn_forward_rejects_wrong_input_shape() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    let wrong = tensor2(vec![1.0f32; 6], 2, 3);
    let result = bn.forward(&wrong);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {:?}",
        result
    );
}

// Training-mode forward: output has batch-mean ~= 0 and batch-var ~= 1

/// BN training-mode forward yields per-feature batch-mean ~= 0 and batch-var ~= 1
/// with default gamma=1, beta=0
#[test]
fn bn_train_output_has_batch_mean_zero_and_var_one() {
    // 4 samples, 3 features; non-uniform so each column has non-trivial mean and variance
    let data = vec![
        2.0f32, 4.0, 6.0, // row 0
        4.0, 2.0, 8.0, // row 1
        6.0, 4.0, 4.0, // row 2
        8.0, 2.0, 6.0, // row 3
    ];
    let input = tensor2(data, 4, 3);

    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    // Default: training=true, gamma=1, beta=0
    let output = bn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // Check batch-mean ~= 0 and batch-variance ~= 1 for each feature column
    for feat in 0..3 {
        let col: Vec<f32> = (0..4).map(|r| output[[r, feat]]).collect();
        let mean: f32 = col.iter().sum::<f32>() / 4.0;
        let var: f32 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "feature {feat}: batch mean too far from 0, got {mean}"
        );
        // var = sigma^2 / (sigma^2 + eps), which is < 1 but very close to 1;
        // tolerance 5e-4 is generous for eps=1e-5 with typical values
        assert!(
            (var - 1.0).abs() < 5e-4,
            "feature {feat}: batch var too far from 1, got {var}"
        );
    }
}

// Training-mode forward: concrete numerical output values

/// BN training-mode forward produces hand-computed normalized values for a 4x1 batch
#[test]
fn bn_train_forward_concrete_values_4x1() {
    let input = tensor2(vec![2.0f32, 4.0, 6.0, 8.0], 4, 1);
    let mut bn = BatchNormalization::new(vec![4, 1], 0.9, 1e-5).unwrap();

    let output = bn.forward(&input).unwrap();

    // Expected values computed analytically
    let std = (5.0f32 + 1e-5f32).sqrt();
    let expected = tensor2(vec![-3.0 / std, -1.0 / std, 1.0 / std, 3.0 / std], 4, 1);
    assert_allclose(&output, &expected, 1e-5);
}

// Running statistics update after training forward

/// One training-mode forward updates running_mean to 0.5 and running_var to 1.4
/// (momentum 0.9), verified through a subsequent eval-mode forward
#[test]
fn bn_running_stats_update_after_one_forward() {
    let input_train = tensor2(vec![2.0f32, 4.0, 6.0, 8.0], 4, 1);
    let mut bn = BatchNormalization::new(vec![4, 1], 0.9, 1e-5).unwrap();

    // Training forward updates running stats
    bn.forward(&input_train).unwrap();

    // Switch to eval and feed the declared shape [4,1]
    bn.set_training_if_mode_dependent(false);
    // All rows = 5.0; each produces the same eval output
    let input_eval = tensor2(vec![5.0f32, 5.0, 5.0, 5.0], 4, 1);
    let output = bn.forward(&input_eval).unwrap();

    // running_mean = 0.5, running_var = 1.4
    let expected_val = 4.5_f32 / (1.4f32 + 1e-5f32).sqrt();
    let expected = tensor2(vec![expected_val; 4], 4, 1);
    assert_allclose(&output, &expected, 1e-4);
}

// Eval mode: uses running stats from set_weights

/// In eval mode, BN normalizes with running_mean and running_var injected via set_weights
#[test]
fn bn_eval_uses_running_stats_from_set_weights() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    // Inject known running statistics and trivial gamma/beta
    let gamma = tensor1(vec![1.0f32, 1.0, 1.0]);
    let beta = tensor1(vec![0.0f32, 0.0, 0.0]);
    let running_mean = tensor1(vec![1.0f32, 2.0, 3.0]);
    let running_var = tensor1(vec![4.0f32, 9.0, 1.0]);
    bn.set_weights(gamma, beta, running_mean, running_var)
        .unwrap();

    bn.set_training_if_mode_dependent(false);

    let input = tensor2(vec![1.0f32, 2.0, 3.0, 5.0, 8.0, 4.0], 2, 3);
    let output = bn.forward(&input).unwrap();

    let eps = 1e-5f32;
    let e00 = 0.0f32;
    let e01 = 0.0f32;
    let e02 = 0.0f32;
    let e10 = 4.0 / (4.0f32 + eps).sqrt();
    let e11 = 6.0 / (9.0f32 + eps).sqrt();
    let e12 = 1.0 / (1.0f32 + eps).sqrt();
    let expected = tensor2(vec![e00, e01, e02, e10, e11, e12], 2, 3);
    assert_allclose(&output, &expected, 1e-4);
}

// predict() == forward() in eval mode

/// In eval mode, BN predict() matches forward() bit-exactly without mutating state
#[test]
fn bn_predict_equals_forward_in_eval_mode() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let gamma = tensor1(vec![1.0f32, 1.0, 1.0]);
    let beta = tensor1(vec![0.0f32, 0.0, 0.0]);
    let running_mean = tensor1(vec![1.0f32, 2.0, 3.0]);
    let running_var = tensor1(vec![4.0f32, 9.0, 1.0]);
    bn.set_weights(gamma, beta, running_mean, running_var)
        .unwrap();
    bn.set_training_if_mode_dependent(false);

    let input = tensor2(vec![1.0f32, 2.0, 3.0, 5.0, 8.0, 4.0], 2, 3);

    let out_forward = bn.forward(&input).unwrap();
    let out_predict = bn.predict(&input).unwrap();

    // Element-wise exact equality
    assert_allclose(&out_forward, &out_predict, 0.0f32);
}

// set_weights rejects wrong shapes

/// set_weights returns Err(NeuralNetwork(WeightShape)) when gamma has the wrong shape
/// (input_shape [4,3] expects gamma [3], supplied [2])
#[test]
fn bn_set_weights_rejects_wrong_gamma_shape() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();

    let gamma_bad = tensor1(vec![1.0f32, 1.0]); // wrong: [2] instead of [3]
    let beta = tensor1(vec![0.0f32, 0.0, 0.0]);
    let rm = tensor1(vec![0.0f32, 0.0, 0.0]);
    let rv = tensor1(vec![1.0f32, 1.0, 1.0]);

    let result = bn.set_weights(gamma_bad, beta, rm, rv);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected NnError::WeightShape, got {:?}",
        result
    );
}

// Custom gamma and beta scale/shift output

/// BN eval mode applies custom gamma and beta to the normalized value
#[test]
fn bn_eval_applies_custom_gamma_and_beta() {
    let mut bn = BatchNormalization::new(vec![1, 1], 0.9, 1e-5).unwrap();

    let gamma = tensor1(vec![2.0f32]);
    let beta = tensor1(vec![1.0f32]);
    let running_mean = tensor1(vec![0.0f32]);
    let running_var = tensor1(vec![1.0f32]);
    bn.set_weights(gamma, beta, running_mean, running_var)
        .unwrap();
    bn.set_training_if_mode_dependent(false);

    let input = tensor2(vec![3.0f32], 1, 1);
    let output = bn.forward(&input).unwrap();

    let eps = 1e-5f32;
    let x_norm = 3.0_f32 / (1.0f32 + eps).sqrt();
    let expected_val = x_norm * 2.0 + 1.0;
    let expected = tensor2(vec![expected_val], 1, 1);
    assert_allclose(&output, &expected, 1e-4);
}

// Multiple training forwards accumulate running stats

/// Two training forwards accumulate running_mean to 0.75 and running_var to 1.0
/// (momentum 0.5), verified through an eval-mode forward
#[test]
fn bn_running_stats_accumulate_over_multiple_forwards() {
    let mut bn = BatchNormalization::new(vec![2, 1], 0.5, 1e-5).unwrap();

    let x = tensor2(vec![0.0f32, 2.0], 2, 1);
    bn.forward(&x).unwrap();
    bn.forward(&x).unwrap();

    bn.set_training_if_mode_dependent(false);

    // running_mean ~= 0.75, running_var ~= 1.0; feed shape [2,1] to match input_shape
    let x_eval = tensor2(vec![1.0f32, 5.0], 2, 1);
    let output = bn.forward(&x_eval).unwrap();

    let denom = (1.0f32 + 1e-5f32).sqrt();
    let e0 = (1.0 - 0.75) / denom;
    let e1 = (5.0 - 0.75) / denom;
    let expected = tensor2(vec![e0, e1], 2, 1);
    assert_allclose(&output, &expected, 1e-4);
}

// Uniform batch: BN output is all zeros

/// When every row of the batch is identical, BN output is all zeros
#[test]
fn bn_uniform_batch_output_is_zero() {
    let data = vec![
        3.0f32, 7.0, -2.0, // row 0, same as all others
        3.0, 7.0, -2.0, // row 1
        3.0, 7.0, -2.0, // row 2
        3.0, 7.0, -2.0, // row 3
    ];
    let input = tensor2(data, 4, 3);
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();

    let output = bn.forward(&input).unwrap();
    let zeros = tensor2(vec![0.0f32; 12], 4, 3);
    assert_allclose(&output, &zeros, 1e-6);
}

// Training mode vs eval mode produce different outputs

/// For a non-trivial input, BN training mode (batch stats) and eval mode (running stats)
/// produce different outputs
#[test]
fn bn_training_and_eval_modes_produce_different_outputs() {
    let input = tensor2(vec![1.0f32, 1.0, 3.0, 3.0], 2, 2);

    let mut bn_train = BatchNormalization::new(vec![2, 2], 0.9, 1e-5).unwrap();
    // Training mode (default), running stats untouched (mean=0, var=1)
    let out_train = bn_train.forward(&input).unwrap();

    let mut bn_eval = BatchNormalization::new(vec![2, 2], 0.9, 1e-5).unwrap();
    bn_eval.set_training_if_mode_dependent(false);
    let out_eval = bn_eval.forward(&input).unwrap();

    // The two outputs must differ in at least one element
    let differs = out_train
        .iter()
        .zip(out_eval.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(
        differs,
        "training and eval outputs are unexpectedly identical"
    );
}

// LAYER NORMALIZATION

// Constructor validation

/// LayerNormalization::new returns Err(InvalidParameter) for non-positive epsilon
/// (zero or negative)
#[test]
fn ln_constructor_rejects_invalid_epsilon() {
    // (epsilon, label) rows; each must yield InvalidParameter
    let cases = [(0.0, "epsilon == 0.0"), (-1e-5, "epsilon < 0.0")];
    for (epsilon, label) in cases {
        let result = LayerNormalization::new(vec![4, 3], epsilon);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label} (epsilon={epsilon}), got {:?}",
            result
        );
    }
}

/// LayerNormalization::new with Multiple([]) returns Err(InvalidParameter)
#[test]
fn ln_constructor_rejects_multiple_empty_axes() {
    let result = LayerNormalization::new(vec![4, 3], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![]));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LayerNormalization::new with Multiple(axes) rejects an out-of-bounds axis
/// (axis 2 against input_shape [4,3] of ndim 2)
#[test]
fn ln_constructor_rejects_multiple_out_of_bounds_axis() {
    let result = LayerNormalization::new(vec![4, 3], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![0, 2]));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LayerNormalization::new with Multiple(axes) rejects a duplicate axis (e.g. [0, 0])
#[test]
fn ln_constructor_rejects_multiple_duplicate_axes() {
    let result = LayerNormalization::new(vec![4, 3], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![0, 0]));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

// Default axis: each sample has mean ~= 0 and variance ~= 1

/// LN Default normalizes along the last dimension, so each row has mean ~= 0 and var ~= 1
#[test]
fn ln_default_each_sample_has_mean_zero_and_var_one() {
    let data = vec![
        1.0f32, 3.0, 5.0, 7.0, // row 0: mean=4, var=5
        2.0, -2.0, 0.0, 4.0, // row 1: mean=1, var=5
    ];
    let input = tensor2(data, 2, 4);
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    for row in 0..2 {
        let vals: Vec<f32> = (0..4).map(|c| output[[row, c]]).collect();
        let mean: f32 = vals.iter().sum::<f32>() / 4.0;
        let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row {row}: mean {mean} too far from 0");
        assert!(
            (var - 1.0).abs() < 5e-4,
            "row {row}: var {var} too far from 1"
        );
    }
}

// Default axis: concrete hand-computed values

/// LN Default produces hand-computed normalized values for two rows of four features
#[test]
fn ln_default_forward_concrete_values() {
    let data = vec![
        1.0f32, 3.0, 5.0, 7.0, // row 0
        2.0, -2.0, 0.0, 4.0, // row 1
    ];
    let input = tensor2(data, 2, 4);
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();

    let std_val = (5.0f32 + 1e-5f32).sqrt();
    let expected = tensor2(
        vec![
            -3.0 / std_val,
            -1.0 / std_val,
            1.0 / std_val,
            3.0 / std_val,
            1.0 / std_val,
            -3.0 / std_val,
            -1.0 / std_val,
            3.0 / std_val,
        ],
        2,
        4,
    );
    assert_allclose(&output, &expected, 1e-5);
}

// Custom axis

/// LN Custom(axis=0) normalizes across rows for each column, matching hand-computed values
#[test]
fn ln_custom_axis0_concrete_values() {
    // rows=3, cols=2; normalize across rows for each column (axis=0)
    let data = vec![1.0f32, 4.0, 3.0, 2.0, 5.0, 6.0];
    let input = tensor2(data, 3, 2);
    let mut ln = LayerNormalization::new(vec![3, 2], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Custom(0))
        .unwrap();

    let output = ln.forward(&input).unwrap();

    let std_val = (8.0f32 / 3.0 + 1e-5f32).sqrt();
    // Col 0: centered = [-2, 0, 2], col 1: centered = [0, -2, 2]
    let expected = tensor2(
        vec![
            -2.0 / std_val,
            0.0 / std_val,
            0.0 / std_val,
            -2.0 / std_val,
            2.0 / std_val,
            2.0 / std_val,
        ],
        3,
        2,
    );
    assert_allclose(&output, &expected, 1e-5);
}

// Custom axis: each column has zero mean and unit variance

/// LN Custom(axis=0) on a [5, 3] input gives each output column mean ~= 0 and var ~= 1
#[test]
fn ln_custom_axis0_each_column_has_mean_zero_and_var_one() {
    let data: Vec<f32> = (0..15).map(|v| v as f32 * 1.3 - 4.0).collect();
    let input = tensor2(data, 5, 3);
    let mut ln = LayerNormalization::new(vec![5, 3], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Custom(0))
        .unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[5, 3]);

    for col in 0..3 {
        let vals: Vec<f32> = (0..5).map(|r| output[[r, col]]).collect();
        let mean: f32 = vals.iter().sum::<f32>() / 5.0;
        let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-5, "col {col}: mean {mean} too far from 0");
        assert!(
            (var - 1.0).abs() < 5e-4,
            "col {col}: var {var} too far from 1"
        );
    }
}

// Multiple axes

/// LN Multiple([0, 1]) on a 2D input normalizes all 12 elements jointly, giving the whole
/// output mean ~= 0 and var ~= 1
#[test]
fn ln_multiple_axes_output_has_mean_zero_and_var_one() {
    let data: Vec<f32> = (0..12).map(|i| 0.5 * i as f32 - 2.75).collect();
    let input = tensor2(data, 3, 4);
    let mut ln = LayerNormalization::new(vec![3, 4], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![0, 1]))
        .unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[3, 4]);

    // With Multiple([0,1]) on a 2D tensor, the entire output is normalized jointly
    let flat: Vec<f32> = output.iter().cloned().collect();
    let n = flat.len() as f32;
    let mean: f32 = flat.iter().sum::<f32>() / n;
    let var: f32 = flat.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    assert!(mean.abs() < 1e-5, "global mean {mean} too far from 0");
    assert!((var - 1.0).abs() < 5e-4, "global var {var} too far from 1");
}

/// LN Multiple([1]) on a [2, 3, 4] input normalizes the 3 elements of axis=1 for each
/// (batch, spatial) position, giving mean ~= 0 and var ~= 1 along axis=1
#[test]
fn ln_multiple_single_axis_on_3d_input() {
    // shape [2, 3, 4]: 2 batches, 3 channels, 4 spatial
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 5.0).collect();
    let shape = vec![2, 3, 4];
    let input = ArrayD::from_shape_vec(shape.clone(), data).unwrap();

    let mut ln = LayerNormalization::new(vec![2, 3, 4], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![1]))
        .unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Axis=1 has 3 elements; for each (batch, spatial) pair those 3 values are normalized
    for b in 0..2 {
        for s in 0..4 {
            let vals: Vec<f32> = (0..3).map(|c| output[[b, c, s]]).collect();
            let mean: f32 = vals.iter().sum::<f32>() / 3.0;
            let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5, "b={b} s={s}: mean {mean} too far from 0");
            assert!(
                (var - 1.0).abs() < 5e-4,
                "b={b} s={s}: var {var} too far from 1"
            );
        }
    }
}

// Zero-variance row: epsilon prevents NaN

/// LN Default: a constant row produces finite, all-zero output thanks to epsilon
#[test]
fn ln_default_constant_row_is_finite_and_zero() {
    let data = vec![
        5.0f32, 5.0, 5.0, 5.0, // constant row
        1.0, 2.0, 3.0, 4.0, // non-constant row
    ];
    let input = tensor2(data, 2, 4);
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();

    // Row 0 (constant): all outputs should be exactly 0
    for c in 0..4 {
        assert!(
            output[[0, c]].is_finite(),
            "output[0,{c}] = {} is not finite",
            output[[0, c]]
        );
        assert_eq!(
            output[[0, c]],
            0.0,
            "output[0,{c}] should be 0 for constant row"
        );
    }

    // Row 1 (non-constant) should be finite
    for c in 0..4 {
        assert!(
            output[[1, c]].is_finite(),
            "output[1,{c}] = {} is not finite",
            output[[1, c]]
        );
    }
}

// predict() == forward() for LN

/// LN predict() matches forward() since LN always computes stats from the current input
#[test]
fn ln_predict_equals_forward() {
    let data = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0];
    let input = tensor2(data, 2, 4);
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();

    // Run forward first (writes caches)
    let out_forward = ln.forward(&input).unwrap();
    // predict() computes the same values without touching the cache
    let out_predict = ln.predict(&input).unwrap();

    assert_allclose(&out_forward, &out_predict, 1e-6);
}

// set_weights: custom gamma/beta applied

/// LN set_weights: custom gamma and beta scale and shift the normalized output
#[test]
fn ln_set_weights_custom_gamma_beta() {
    let input = tensor2(vec![0.0f32, 4.0], 1, 2);
    let mut ln = LayerNormalization::new(vec![1, 2], 1e-5).unwrap();

    let gamma = tensor1(vec![3.0f32, 3.0]);
    let beta = tensor1(vec![10.0f32, 10.0]);
    ln.set_weights(gamma, beta).unwrap();

    let output = ln.forward(&input).unwrap();

    let eps = 1e-5f32;
    let std = (4.0f32 + eps).sqrt();
    let x_norm_0 = -2.0_f32 / std;
    let x_norm_1 = 2.0_f32 / std;
    let e0 = x_norm_0 * 3.0 + 10.0;
    let e1 = x_norm_1 * 3.0 + 10.0;
    let expected = tensor2(vec![e0, e1], 1, 2);
    assert_allclose(&output, &expected, 1e-4);
}

/// LN set_weights rejects a wrong gamma shape ([3] against expected [4])
#[test]
fn ln_set_weights_rejects_wrong_gamma_shape() {
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();
    let bad_gamma = tensor1(vec![1.0f32, 1.0, 1.0]); // [3] instead of [4]
    let beta = tensor1(vec![0.0f32; 4]);
    let result = ln.set_weights(bad_gamma, beta);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected NnError::WeightShape, got {:?}",
        result
    );
}

// Shape-mismatch on LN forward

/// LN forward rejects input whose shape ([3, 4]) differs from the declared [2, 4]
#[test]
fn ln_forward_rejects_wrong_input_shape() {
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();
    let wrong = tensor2(vec![1.0f32; 12], 3, 4);
    let result = ln.forward(&wrong);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {:?}",
        result
    );
}

// LN eval mode: set_training_if_mode_dependent

/// Toggling training mode does not change LN forward output (mode only affects backward)
#[test]
fn ln_mode_switch_does_not_change_forward_output() {
    let data = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0];
    let input = tensor2(data, 2, 4);
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();

    // Training mode output
    let out_train = ln.forward(&input).unwrap();

    // Switch to eval mode
    ln.set_training_if_mode_dependent(false);
    let out_eval = ln.forward(&input).unwrap();

    // LN forward result must be identical in both modes
    assert_allclose(&out_train, &out_eval, 1e-6);
}

// LN predict() in eval mode equals forward() in eval mode

/// In eval mode, LN predict() and forward() agree exactly
#[test]
fn ln_predict_equals_forward_in_eval_mode() {
    let data = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0];
    let input = tensor2(data, 2, 4);
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();
    ln.set_training_if_mode_dependent(false);

    let out_forward = ln.forward(&input).unwrap();
    let out_predict = ln.predict(&input).unwrap();

    assert_allclose(&out_forward, &out_predict, 0.0f32);
}

// LN Multiple axes: forward-time axis validation

/// LN forward with valid Multiple([0, 1]) axes succeeds
#[test]
fn ln_multiple_valid_axes_forward_succeeds() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let input = tensor2(data, 3, 4);
    let mut ln = LayerNormalization::new(vec![3, 4], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![0, 1]))
        .unwrap();
    let result = ln.forward(&input);
    assert!(
        result.is_ok(),
        "forward with valid Multiple axes failed: {:?}",
        result
    );
}

// backward() before forward() must error

/// BatchNormalization::backward called before any forward returns ForwardPassNotRun
#[test]
fn bn_backward_before_forward_errors() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();
    let grad = tensor2(vec![1.0f32; 6], 2, 3);
    let err = bn.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("BatchNormalization"))
        ),
        "expected ForwardPassNotRun(\"BatchNormalization\"), got {:?}",
        err
    );
}

/// LayerNormalization::backward called before any forward returns ForwardPassNotRun
#[test]
fn ln_backward_before_forward_errors() {
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();
    let grad = tensor2(vec![1.0f32; 8], 2, 4);
    let err = ln.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("LayerNormalization"))
        ),
        "expected ForwardPassNotRun(\"LayerNormalization\"), got {:?}",
        err
    );
}

// LN runtime axis validation on forward / predict

/// LN Custom(axis) with axis >= input.ndim() is rejected at forward time with InvalidParameter
#[test]
fn ln_custom_axis_out_of_bounds_forward_errors() {
    let mut ln = LayerNormalization::new(vec![4], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Custom(5))
        .unwrap();
    let input = tensor1(vec![1.0f32, 2.0, 3.0, 4.0]); // ndim = 1, axis 5 is out of bounds
    let result = ln.forward(&input);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LN Custom(axis) out-of-bounds is also rejected by predict() with InvalidParameter
#[test]
fn ln_custom_axis_out_of_bounds_predict_errors() {
    let ln = LayerNormalization::new(vec![4], 1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Custom(5))
        .unwrap();
    let input = tensor1(vec![1.0f32, 2.0, 3.0, 4.0]); // ndim = 1, axis 5 is out of bounds
    let result = ln.predict(&input);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LN Default on a 0-dim (scalar) tensor is rejected at forward time with InvalidInput
#[test]
fn ln_default_scalar_input_forward_errors() {
    let mut ln = LayerNormalization::new(vec![], 1e-5).unwrap();
    let scalar = ArrayD::from_shape_vec(vec![], vec![3.0f32]).unwrap(); // 0-dim tensor
    let result = ln.forward(&scalar);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        result
    );
}

/// LN Default on a 0-dim (scalar) tensor is also rejected by predict() with InvalidInput
#[test]
fn ln_default_scalar_input_predict_errors() {
    let ln = LayerNormalization::new(vec![], 1e-5).unwrap();
    let scalar = ArrayD::from_shape_vec(vec![], vec![3.0f32]).unwrap(); // 0-dim tensor
    let result = ln.predict(&scalar);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        result
    );
}

// Eval-mode backward: gradient passes through unchanged

/// BatchNormalization::backward in eval mode returns grad_output unchanged (bit-exact)
#[test]
fn bn_backward_eval_mode_passes_gradient_through() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();
    bn.set_training_if_mode_dependent(false);

    // A forward in eval mode (uses running stats); does not affect the passthrough
    let input = tensor2(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    bn.forward(&input).unwrap();

    // Distinct gradient values so an accidental transform would change them
    let grad = tensor2(vec![0.5f32, -1.5, 2.0, -3.0, 4.5, -6.0], 2, 3);
    let grad_input = bn.backward(&grad).unwrap();

    // Eval-mode backward returns grad_output.clone(): bit-exact equality
    assert_allclose(&grad_input, &grad, 0.0f32);
}

/// LayerNormalization::backward in eval mode returns grad_output unchanged (bit-exact)
#[test]
fn ln_backward_eval_mode_passes_gradient_through() {
    let mut ln = LayerNormalization::new(vec![2, 4], 1e-5).unwrap();
    ln.set_training_if_mode_dependent(false);

    let input = tensor2(vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0], 2, 4);
    ln.forward(&input).unwrap();

    let grad = tensor2(vec![0.5f32, -1.5, 2.0, -3.0, 4.5, -6.0, 7.0, -8.5], 2, 4);
    let grad_input = ln.backward(&grad).unwrap();

    assert_allclose(&grad_input, &grad, 0.0f32);
}

// BN constructor with 1-D input_shape: scalar-parameter branch

/// BatchNormalization with a 1-D input_shape uses length-1 scalar params and normalizes
/// every element as its own sample, preserving the input shape
#[test]
fn bn_new_scalar_param_branch_forward_1d() {
    let mut bn = BatchNormalization::new(vec![4], 0.9, 1e-5).unwrap();

    let input = tensor1(vec![1.0f32, 2.0, 3.0, 4.0]);
    let output = bn.forward(&input).unwrap();

    // Shape is preserved through the scalar-param broadcast
    assert_eq!(output.shape(), &[4]);

    let std = (1.25f32 + 1e-5f32).sqrt();
    let expected = tensor1(vec![-1.5 / std, -0.5 / std, 0.5 / std, 1.5 / std]);
    assert_allclose(&output, &expected, 1e-5);
}

/// Spatial batch norm (rank > 2): a [1, 2, 2, 2] input normalizes each channel over its (N, H, W)
/// values, and the parameters are per-channel (2*C), not per spatial element
#[test]
fn bn_spatial_4d_normalizes_per_channel() {
    use rustyml::neural_network::Tensor;
    use rustyml::neural_network::layers::TrainingParameters;

    let mut bn = BatchNormalization::new(vec![1, 2, 2, 2], 0.9, 1e-5).unwrap();
    // Per-channel (C=2) parameters: gamma[2] + beta[2] = 4 trainable (not 2*C*H*W = 16)
    assert!(matches!(bn.param_count(), TrainingParameters::Trainable(4)));

    // channel 0 = {1,2,3,4}, channel 1 = {5,6,7,8} (row-major [N, C, H, W])
    let x: Tensor =
        ArrayD::from_shape_vec(vec![1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
    let out = bn.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Each channel has variance 1.25, so the normalized values repeat per channel
    let inv = 1.0 / (1.25f32 + 1e-5).sqrt();
    let expected: Tensor = ArrayD::from_shape_vec(
        vec![1, 2, 2, 2],
        vec![
            -1.5 * inv,
            -0.5 * inv,
            0.5 * inv,
            1.5 * inv,
            -1.5 * inv,
            -0.5 * inv,
            0.5 * inv,
            1.5 * inv,
        ],
    )
    .unwrap();
    assert_allclose(&out, &expected, 1e-4);
}

/// Spatial batch-norm backward returns a gradient matching the input shape (folds/unfolds correctly)
#[test]
fn bn_spatial_4d_backward_shape() {
    use rustyml::neural_network::Tensor;
    let mut bn = BatchNormalization::new(vec![2, 3, 2, 2], 0.9, 1e-5).unwrap();
    let x: Tensor =
        ArrayD::from_shape_fn(vec![2, 3, 2, 2], |idx| (idx[1] + idx[2] + idx[3]) as f32);
    bn.forward(&x).unwrap();
    let grad: Tensor = ArrayD::ones(vec![2, 3, 2, 2]);
    let grad_in = bn.backward(&grad).unwrap();
    assert_eq!(grad_in.shape(), &[2, 3, 2, 2]);
    assert!(grad_in.iter().all(|v| v.is_finite()));
}
