use crate::utility::t_sne::*;
use approx::assert_relative_eq;
use ndarray::{Array2, Axis};

#[test]
fn test_tsne_default() {
    let tsne = TSNE::default();
    assert_eq!(tsne.get_perplexity(), 30.0);
    assert_eq!(tsne.get_learning_rate(), 200.0);
    assert_eq!(tsne.get_n_iter(), 1000);
    assert_eq!(tsne.get_dim(), 2);
    assert_eq!(tsne.get_random_state(), 42);
    assert_eq!(tsne.get_early_exaggeration(), 12.0);
    assert_eq!(tsne.get_exaggeration_iter(), 1000 / 12);
    assert_eq!(tsne.get_initial_momentum(), 0.5);
    assert_eq!(tsne.get_final_momentum(), 0.8);
    assert_eq!(tsne.get_momentum_switch_iter(), 1000 / 3);
}

#[test]
fn test_tsne_new() {
    let tsne = TSNE::new(
        Some(50.0),
        Some(300.0),
        Some(500),
        3,
        Some(123),
        Some(20.0),
        Some(50),
        Some(0.3),
        Some(0.9),
        Some(200),
    );

    assert_eq!(tsne.get_perplexity(), 50.0);
    assert_eq!(tsne.get_learning_rate(), 300.0);
    assert_eq!(tsne.get_n_iter(), 500);
    assert_eq!(tsne.get_dim(), 3);
    assert_eq!(tsne.get_random_state(), 123);
    assert_eq!(tsne.get_early_exaggeration(), 20.0);
    assert_eq!(tsne.get_exaggeration_iter(), 50);
    assert_eq!(tsne.get_initial_momentum(), 0.3);
    assert_eq!(tsne.get_final_momentum(), 0.9);
    assert_eq!(tsne.get_momentum_switch_iter(), 200);
}

#[test]
fn test_tsne_partial_params() {
    let tsne = TSNE::new(
        Some(40.0),
        None,
        Some(200),
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    assert_eq!(tsne.get_perplexity(), 40.0);
    assert_eq!(tsne.get_learning_rate(), 200.0); // default
    assert_eq!(tsne.get_n_iter(), 200);
    assert_eq!(tsne.get_dim(), 2);
    assert_eq!(tsne.get_random_state(), 42); // default
    assert_eq!(tsne.get_early_exaggeration(), 12.0); // default
    assert_eq!(tsne.get_exaggeration_iter(), 1000 / 12); // default
    assert_eq!(tsne.get_initial_momentum(), 0.5); // default
    assert_eq!(tsne.get_final_momentum(), 0.8); // default
    assert_eq!(tsne.get_momentum_switch_iter(), 1000 / 3); // default
}

#[test]
fn test_fit_transform_dimensions() {
    let n_samples = 20;
    let n_features = 10;
    let output_dim = 2;

    let x = Array2::<f64>::ones((n_samples, n_features));
    let tsne = TSNE::new(
        None,
        None,
        Some(10),
        output_dim,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let result = tsne.fit_transform(x.view());
    assert!(result.is_ok());

    let embedding = result.unwrap();
    assert_eq!(embedding.shape(), &[n_samples, output_dim]);
}

#[test]
fn test_fit_transform_validation() {
    let n_samples = 20;
    let n_features = 10;

    let x = Array2::<f64>::ones((n_samples, n_features));

    // Test invalid perplexity
    let tsne_invalid_perplexity = TSNE::new(
        Some(-1.0),
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
    let result = tsne_invalid_perplexity.fit_transform(x.view());
    assert!(result.is_err());

    // Test invalid learning_rate
    let tsne_invalid_lr = TSNE::new(
        None,
        Some(-0.1),
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let result = tsne_invalid_lr.fit_transform(x.view());
    assert!(result.is_err());

    // Test invalid n_iter
    let tsne_invalid_iter = TSNE::new(None, None, Some(0), 2, None, None, None, None, None, None);
    let result = tsne_invalid_iter.fit_transform(x.view());
    assert!(result.is_err());

    // Test invalid momentum
    let tsne_invalid_momentum =
        TSNE::new(None, None, None, 2, None, None, None, Some(1.5), None, None);
    let result = tsne_invalid_momentum.fit_transform(x.view());
    assert!(result.is_err());
}

#[test]
fn test_result_is_zero_mean() {
    let n_samples = 10;
    let n_features = 5;
    let output_dim = 2;

    let x = Array2::<f64>::ones((n_samples, n_features));
    let tsne = TSNE::new(
        None,
        None,
        Some(10),
        output_dim,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );

    let result = tsne.fit_transform(x.view()).unwrap();

    // Check if result has zero mean
    let mean = result.mean_axis(Axis(0)).unwrap();
    for &m in mean.iter() {
        assert_relative_eq!(m, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_reproducibility() {
    let n_samples = 10;
    let n_features = 5;

    let x = Array2::<f64>::ones((n_samples, n_features));

    // Using the same random seed should produce the same results
    let tsne1 = TSNE::new(
        None,
        None,
        Some(10),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );
    let tsne2 = TSNE::new(
        None,
        None,
        Some(10),
        2,
        Some(42),
        None,
        None,
        None,
        None,
        None,
    );

    let result1 = tsne1.fit_transform(x.view()).unwrap();
    let result2 = tsne2.fit_transform(x.view()).unwrap();

    // Results should be identical
    for (v1, v2) in result1.iter().zip(result2.iter()) {
        assert_eq!(*v1, *v2);
    }

    // Using different random seeds should produce different results
    let tsne3 = TSNE::new(
        None,
        None,
        Some(10),
        2,
        Some(24),
        None,
        None,
        None,
        None,
        None,
    );
    let result3 = tsne3.fit_transform(x.view()).unwrap();

    // At least one value should be different
    let mut any_different = false;
    for (v1, v3) in result1.iter().zip(result3.iter()) {
        if *v1 != *v3 {
            any_different = true;
            break;
        }
    }
    assert!(any_different);
}
