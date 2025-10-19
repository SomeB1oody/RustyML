use super::*;

#[test]
fn test_tsne_default_creation() {
    // Test that TSNE can be created with default parameters
    let tsne = TSNE::default();

    assert_eq!(tsne.get_perplexity(), 30.0);
    assert_eq!(tsne.get_learning_rate(), 200.0);
    assert_eq!(tsne.get_actual_iterations(), 1000);
    assert_eq!(tsne.get_dimensions(), 2);
    assert_eq!(tsne.get_random_state(), 42);
    assert_eq!(tsne.get_early_exaggeration(), 12.0);
    assert_eq!(tsne.get_initial_momentum(), 0.5);
    assert_eq!(tsne.get_final_momentum(), 0.8);
}

#[test]
fn test_tsne_custom_creation() {
    // Test TSNE creation with custom parameters
    let tsne = TSNE::new(
        Some(20.0),  // perplexity
        Some(100.0), // learning_rate
        Some(500),   // n_iter
        3,           // dim
        Some(123),   // random_state
        Some(8.0),   // early_exaggeration
        Some(50),    // exaggeration_iter
        Some(0.6),   // initial_momentum
        Some(0.9),   // final_momentum
        Some(150),   // momentum_switch_iter
    )
    .unwrap();

    assert_eq!(tsne.get_perplexity(), 20.0);
    assert_eq!(tsne.get_learning_rate(), 100.0);
    assert_eq!(tsne.get_actual_iterations(), 500);
    assert_eq!(tsne.get_dimensions(), 3);
    assert_eq!(tsne.get_random_state(), 123);
    assert_eq!(tsne.get_early_exaggeration(), 8.0);
    assert_eq!(tsne.get_exaggeration_iterations(), 50);
    assert_eq!(tsne.get_initial_momentum(), 0.6);
    assert_eq!(tsne.get_final_momentum(), 0.9);
    assert_eq!(tsne.get_momentum_switch_iterations(), 150);
}

#[test]
fn test_tsne_invalid_perplexity() {
    // Test that negative perplexity is rejected
    let result = TSNE::new(
        Some(-1.0), // invalid negative perplexity
        None,
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_invalid_learning_rate() {
    // Test that negative learning rate is rejected
    let result = TSNE::new(
        None,
        Some(0.0), // invalid zero learning rate
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_invalid_dim() {
    // Test that zero dimension is rejected
    let result = TSNE::new(
        None, None, None, 0, // invalid zero dimension
        None, None, None, None, None, None,
    );

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_invalid_momentum() {
    // Test that momentum outside [0, 1] range is rejected
    let result = TSNE::new(
        None,
        None,
        None,
        2,
        None,
        None,
        None,
        Some(1.5), // invalid momentum > 1.0
        None,
        None,
    );

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_invalid_early_exaggeration() {
    // Test that early_exaggeration <= 1.0 is rejected
    let result = TSNE::new(
        None,
        None,
        None,
        2,
        None,
        Some(0.5), // invalid early_exaggeration <= 1.0
        None,
        None,
        None,
        None,
    );

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_fit_transform_simple() {
    // Test basic fit_transform with simple data
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ]);

    let tsne = TSNE::new(
        Some(2.0), // small perplexity for small dataset
        Some(100.0),
        Some(100), // fewer iterations for test
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_ok());

    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[5, 2]);

    // Check that all values are finite
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_tsne_empty_input() {
    // Test that empty input is rejected
    let data = Array2::<f64>::zeros((0, 0));
    let tsne = TSNE::default();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_nan_input() {
    // Test that NaN values in input are rejected
    let data = arr2(&[[1.0, 2.0, f64::NAN], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let tsne = TSNE::new(
        Some(1.0),
        None,
        Some(50),
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_infinite_input() {
    // Test that infinite values in input are rejected
    let data = arr2(&[[1.0, 2.0, 3.0], [4.0, f64::INFINITY, 6.0], [7.0, 8.0, 9.0]]);

    let tsne = TSNE::new(
        Some(1.0),
        None,
        Some(50),
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_perplexity_exceeds_samples() {
    // Test that perplexity >= n_samples is rejected
    let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

    let tsne = TSNE::new(
        Some(5.0), // perplexity >= n_samples (3)
        None,
        Some(50),
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_dim_exceeds_samples() {
    // Test that dim > n_samples is rejected
    let data = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let tsne = TSNE::new(
        Some(1.0),
        None,
        Some(50),
        3, // dim > n_samples (2)
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::InputValidationError(_)
    ));
}

#[test]
fn test_tsne_reproducibility() {
    // Test that same random_state produces same results
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ]);

    let tsne1 = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(100),
        2,
        Some(42), // same random state
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let tsne2 = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(100),
        2,
        Some(42), // same random state
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let embedding1 = tsne1.fit_transform(&data.view()).unwrap();
    let embedding2 = tsne2.fit_transform(&data.view()).unwrap();

    // Results should be identical with same random state
    assert_eq!(embedding1.shape(), embedding2.shape());
    for (a, b) in embedding1.iter().zip(embedding2.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_tsne_different_random_states() {
    // Test that different random_states produce different results
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ]);

    let tsne1 = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(100),
        2,
        Some(42), // different random state
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let tsne2 = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(100),
        2,
        Some(123), // different random state
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let embedding1 = tsne1.fit_transform(&data.view()).unwrap();
    let embedding2 = tsne2.fit_transform(&data.view()).unwrap();

    // Results should be different with different random states
    let mut has_difference = false;
    for (a, b) in embedding1.iter().zip(embedding2.iter()) {
        if (a - b).abs() > 1e-6 {
            has_difference = true;
            break;
        }
    }
    assert!(has_difference);
}

#[test]
fn test_tsne_3d_embedding() {
    // Test 3D embedding
    let data = arr2(&[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
        [17.0, 18.0, 19.0, 20.0],
        [21.0, 22.0, 23.0, 24.0],
    ]);

    let tsne = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(100),
        3, // 3D embedding
        Some(42),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_ok());

    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[6, 3]);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_tsne_centered_output() {
    // Test that output is approximately centered
    let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]);

    let tsne = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(100),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let embedding = tsne.fit_transform(&data.view()).unwrap();

    // Calculate mean along each dimension
    let mean_0: f64 = embedding.column(0).iter().sum::<f64>() / embedding.nrows() as f64;
    let mean_1: f64 = embedding.column(1).iter().sum::<f64>() / embedding.nrows() as f64;

    // Mean should be close to zero
    assert!(mean_0.abs() < 1e-10);
    assert!(mean_1.abs() < 1e-10);
}

#[test]
fn test_tsne_larger_dataset() {
    // Test with a moderately larger dataset
    let n_samples = 50;
    let n_features = 10;
    let mut data = Array2::<f64>::zeros((n_samples, n_features));

    // Create structured data with some patterns
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = (i as f64 * 0.1) + (j as f64 * 0.5);
        }
    }

    let tsne = TSNE::new(
        Some(10.0),
        Some(200.0),
        Some(250), // moderate iterations for test
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = tsne.fit_transform(&data.view());
    assert!(result.is_ok());

    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[n_samples, 2]);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}
