use super::*;

#[test]
fn test_tsne_basic_functionality() {
    // Create a simple dataset with distinct clusters
    let data = array![
        [1.0, 1.0, 1.0],
        [1.1, 0.9, 1.0],
        [0.9, 1.1, 1.0],
        [5.0, 5.0, 5.0],
        [5.1, 4.9, 5.0],
        [4.9, 5.1, 5.0],
    ];

    let tsne = TSNE::new(
        Some(2.0),
        Some(100.0),
        Some(200),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_ok());
    let embedding = result.unwrap();

    // Check output shape
    assert_eq!(embedding.shape(), &[6, 2]);

    // Check that the embedding is not all zeros
    let sum = embedding.iter().map(|x| x.abs()).sum::<f64>();
    assert!(sum > 0.0);
}

#[test]
fn test_tsne_default_parameters() {
    // Test TSNE with default parameters
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ];

    let tsne = TSNE::default();
    let result = tsne.fit_transform(data.view());

    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[4, 2]);
}

#[test]
fn test_tsne_custom_dimensions() {
    // Test with 3D embedding
    let data = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
        [17.0, 18.0, 19.0, 20.0],
    ];

    let tsne = TSNE::new(
        None,
        None,
        Some(100),
        3,
        Some(123),
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[5, 3]);
}

#[test]
fn test_tsne_getters() {
    // Test all getter methods
    let tsne = TSNE::new(
        Some(25.0),
        Some(150.0),
        Some(500),
        2,
        Some(99),
        Some(10.0),
        Some(50),
        Some(0.6),
        Some(0.9),
        Some(200),
    );

    assert_eq!(tsne.get_perplexity(), 25.0);
    assert_eq!(tsne.get_learning_rate(), 150.0);
    assert_eq!(tsne.get_actual_iterations(), 500);
    assert_eq!(tsne.get_dimensions(), 2);
    assert_eq!(tsne.get_random_state(), 99);
    assert_eq!(tsne.get_early_exaggeration(), 10.0);
    assert_eq!(tsne.get_exaggeration_iterations(), 50);
    assert_eq!(tsne.get_initial_momentum(), 0.6);
    assert_eq!(tsne.get_final_momentum(), 0.9);
    assert_eq!(tsne.get_momentum_switch_iterations(), 200);
}

#[test]
fn test_tsne_with_none_parameters() {
    // Test that None parameters use defaults correctly
    let tsne = TSNE::new(None, None, None, 2, None, None, None, None, None, None);

    assert_eq!(tsne.get_perplexity(), 30.0);
    assert_eq!(tsne.get_learning_rate(), 200.0);
    assert_eq!(tsne.get_actual_iterations(), 1000);
    assert_eq!(tsne.get_random_state(), 42);
    assert_eq!(tsne.get_early_exaggeration(), 12.0);
    assert_eq!(tsne.get_exaggeration_iterations(), 1000 / 12);
    assert_eq!(tsne.get_initial_momentum(), 0.5);
    assert_eq!(tsne.get_final_momentum(), 0.8);
    assert_eq!(tsne.get_momentum_switch_iterations(), 1000 / 3);
}

#[test]
fn test_tsne_invalid_perplexity() {
    // Test with invalid perplexity (0)
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let tsne = TSNE::new(
        Some(0.0),
        None,
        Some(10),
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_invalid_learning_rate() {
    // Test with invalid learning rate (negative)
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let tsne = TSNE::new(
        None,
        Some(-10.0),
        Some(10),
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_invalid_n_iter() {
    // Test with invalid n_iter (0)
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let tsne = TSNE::new(None, None, Some(0), 2, None, None, None, None, None, None);
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_invalid_early_exaggeration() {
    // Test with invalid early exaggeration (less than 1.0)
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let tsne = TSNE::new(
        None,
        None,
        Some(10),
        2,
        None,
        Some(0.5),
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_invalid_momentum() {
    // Test with invalid momentum (outside 0-1 range)
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let tsne = TSNE::new(
        None,
        None,
        Some(10),
        2,
        None,
        None,
        None,
        Some(1.5),
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_invalid_dimension_zero() {
    // Test with zero dimension
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let tsne = TSNE::new(None, None, Some(10), 0, None, None, None, None, None, None);
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_invalid_dimension_exceeds_samples() {
    // Test with dimension exceeding number of samples
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let tsne = TSNE::new(None, None, Some(10), 5, None, None, None, None, None, None);
    let result = tsne.fit_transform(data.view());

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ModelError::InputValidationError(_)));
    }
}

#[test]
fn test_tsne_reproducibility() {
    // Test that same random state produces same results
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ];

    let tsne1 = TSNE::new(
        None,
        None,
        Some(50),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let result1 = tsne1.fit_transform(data.view()).unwrap();

    let tsne2 = TSNE::new(
        None,
        None,
        Some(50),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let result2 = tsne2.fit_transform(data.view()).unwrap();

    // Results should be identical with same random state
    assert_eq!(result1.shape(), result2.shape());
    for (a, b) in result1.iter().zip(result2.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_tsne_different_random_states() {
    // Test that different random states produce different results
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ];

    let tsne1 = TSNE::new(
        None,
        None,
        Some(50),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let result1 = tsne1.fit_transform(data.view()).unwrap();

    let tsne2 = TSNE::new(
        None,
        None,
        Some(50),
        2,
        Some(123),
        None,
        None,
        None,
        None,
        None,
    );
    let result2 = tsne2.fit_transform(data.view()).unwrap();

    // Results should be different with different random states
    let mut different = false;
    for (a, b) in result1.iter().zip(result2.iter()) {
        if (a - b).abs() > 1e-5 {
            different = true;
            break;
        }
    }
    assert!(different);
}

#[test]
fn test_tsne_small_dataset() {
    // Test with a very small dataset
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

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
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[3, 2]);
}

#[test]
fn test_tsne_embedding_centered() {
    // Test that embedding is centered (mean approximately zero)
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ];

    let tsne = TSNE::new(
        None,
        None,
        Some(100),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_ok());
    let embedding = result.unwrap();

    // Calculate mean for each dimension
    let mean0 = embedding.column(0).mean().unwrap();
    let mean1 = embedding.column(1).mean().unwrap();

    // Mean should be close to zero (centered)
    assert!(mean0.abs() < 1e-10);
    assert!(mean1.abs() < 1e-10);
}

#[test]
fn test_tsne_high_dimensional_input() {
    // Test with high-dimensional input
    let mut data = Array2::<f64>::zeros((10, 50));
    for i in 0..10 {
        for j in 0..50 {
            data[[i, j]] = (i * 50 + j) as f64;
        }
    }

    let tsne = TSNE::new(
        Some(3.0),
        None,
        Some(100),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne.fit_transform(data.view());

    assert!(result.is_ok());
    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[10, 2]);
}
