//! Integration tests for `rustyml::utils::StandardScaler`, covering the fit/transform
//! contract, incremental fitting, persistence, and the error paths

use crate::common::assert_allclose;
use ndarray::{Array2, array};
use rustyml::error::Error;
use rustyml::traits::{Fit, FitTransform, Transform};
use rustyml::utils::StandardScaler;
use rustyml::utils::standardize::{StandardizationAxis, standardize};

// Fitted statistics

/// The fitted statistics are the population mean and standard deviation per feature
#[test]
fn test_fit_learns_population_statistics() {
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).unwrap();

    assert_allclose(scaler.get_mean().unwrap(), &array![2.5, 25.0], 1e-12);
    // Population variance (ddof = 0): mean of squared deviations
    assert_allclose(scaler.get_var().unwrap(), &array![1.25, 125.0], 1e-12);
    assert_allclose(
        scaler.get_scale().unwrap(),
        &array![1.25_f64.sqrt(), 125.0_f64.sqrt()],
        1e-12,
    );
    assert_eq!(scaler.get_n_samples_seen(), 4);
    assert_eq!(scaler.get_n_features(), Some(2));
}

/// An unfitted scaler exposes no statistics
#[test]
fn test_unfitted_scaler_has_no_statistics() {
    let scaler = StandardScaler::new();

    assert!(scaler.get_mean().is_none());
    assert!(scaler.get_var().is_none());
    assert!(scaler.get_scale().is_none());
    assert_eq!(scaler.get_n_features(), None);
    assert_eq!(scaler.get_n_samples_seen(), 0);
}

/// Refitting replaces the previous statistics instead of merging into them
#[test]
fn test_fit_replaces_previous_statistics() {
    let mut scaler = StandardScaler::new();
    scaler.fit(&array![[1.0], [2.0]]).unwrap();
    scaler.fit(&array![[10.0], [20.0], [30.0]]).unwrap();

    assert_eq!(scaler.get_n_samples_seen(), 3);
    assert_allclose(scaler.get_mean().unwrap(), &array![20.0], 1e-12);
}

// Transform semantics

/// `fit_transform` on the training matrix reproduces the stateless column standardization
#[test]
fn test_fit_transform_matches_stateless_standardize() {
    let x = array![[1.0, 2000.0], [2.0, 3000.0], [3.0, 4000.0], [4.0, 5000.0]];

    let scaled = StandardScaler::new().fit_transform(&x).unwrap();
    let stateless = standardize(&x, StandardizationAxis::Column).unwrap();

    // Same Welford pass and the same constant-feature rule, so the two agree exactly
    assert_eq!(scaled, stateless);
}

/// A later batch is scaled by the training statistics, not by its own
#[test]
fn test_transform_uses_frozen_training_statistics() {
    let x_train = array![[1.0], [2.0], [3.0]];
    let x_test = array![[4.0], [5.0]];

    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train).unwrap();
    let z_test = scaler.transform(&x_test).unwrap();

    // Training mean 2.0, population std sqrt(2/3)
    let std = (2.0_f64 / 3.0).sqrt();
    assert_allclose(&z_test, &array![[2.0 / std], [3.0 / std]], 1e-12);

    // Standardizing the test batch on its own would have centered it on its own mean
    let self_scaled = standardize(&x_test, StandardizationAxis::Column).unwrap();
    assert!((self_scaled[[0, 0]] - z_test[[0, 0]]).abs() > 1.0);
}

/// A single sample transforms correctly, which the stateless function cannot do
#[test]
fn test_transform_single_sample() {
    let mut scaler = StandardScaler::new();
    scaler.fit(&array![[0.0, 10.0], [2.0, 20.0]]).unwrap();

    let z = scaler.transform(&array![[1.0, 15.0]]).unwrap();
    assert_allclose(&z, &array![[0.0, 0.0]], 1e-12);
}

/// The input array is never modified
#[test]
fn test_transform_does_not_mutate_input() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let original = x.clone();

    let mut scaler = StandardScaler::new();
    let _ = scaler.fit_transform(&x).unwrap();
    let _ = scaler.transform(&x).unwrap();

    assert_eq!(x, original, "StandardScaler must not modify its input");
}

/// `inverse_transform` recovers the original matrix
#[test]
fn test_inverse_transform_round_trip() {
    let x = array![[1.0, -5.0], [2.0, 7.5], [3.0, 0.5], [4.0, 100.0]];

    let mut scaler = StandardScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    let restored = scaler.inverse_transform(&z).unwrap();

    assert_allclose(&restored, &x, 1e-9);
}

/// The `with_mean` and `with_std` flags switch off centering and scaling independently
#[test]
fn test_with_mean_and_with_std_flags() {
    let x = array![[1.0], [2.0], [3.0]];

    let centered = StandardScaler::new()
        .with_std(false)
        .fit_transform(&x)
        .unwrap();
    assert_allclose(&centered, &array![[-1.0], [0.0], [1.0]], 1e-12);

    let scaled = StandardScaler::new()
        .with_mean(false)
        .fit_transform(&x)
        .unwrap();
    let std = (2.0_f64 / 3.0).sqrt();
    assert_allclose(
        &scaled,
        &array![[1.0 / std], [2.0 / std], [3.0 / std]],
        1e-12,
    );

    let untouched = StandardScaler::new()
        .with_mean(false)
        .with_std(false)
        .fit_transform(&x)
        .unwrap();
    assert_eq!(untouched, x);

    // The statistics are still computed and inspectable with both flags off
    let mut scaler = StandardScaler::new().with_mean(false).with_std(false);
    scaler.fit(&x).unwrap();
    assert_allclose(scaler.get_mean().unwrap(), &array![2.0], 1e-12);

    // Inverse transform honours the same flags, so the round trip still holds
    let restored = scaler.inverse_transform(&untouched).unwrap();
    assert_eq!(restored, x);
}

// Incremental fitting

/// `partial_fit` over batches matches a single `fit` over their concatenation
#[test]
fn test_partial_fit_matches_single_fit() {
    let full = array![
        [1.0, 10.0],
        [2.0, 25.0],
        [3.0, 30.0],
        [4.0, 15.0],
        [5.0, 50.0],
        [6.0, 5.0]
    ];

    let mut incremental = StandardScaler::new();
    incremental
        .partial_fit(&full.slice(ndarray::s![0..2, ..]))
        .unwrap();
    incremental
        .partial_fit(&full.slice(ndarray::s![2..5, ..]))
        .unwrap();
    incremental
        .partial_fit(&full.slice(ndarray::s![5..6, ..]))
        .unwrap();

    let mut single = StandardScaler::new();
    single.fit(&full).unwrap();

    assert_eq!(incremental.get_n_samples_seen(), 6);
    assert_allclose(
        incremental.get_mean().unwrap(),
        single.get_mean().unwrap(),
        1e-9,
    );
    assert_allclose(
        incremental.get_var().unwrap(),
        single.get_var().unwrap(),
        1e-9,
    );
    assert_allclose(
        incremental.get_scale().unwrap(),
        single.get_scale().unwrap(),
        1e-9,
    );
}

/// `partial_fit` on an unfitted scaler behaves like `fit`
#[test]
fn test_partial_fit_on_unfitted_scaler() {
    let x = array![[1.0], [2.0], [3.0]];

    let mut incremental = StandardScaler::new();
    incremental.partial_fit(&x).unwrap();
    let mut fitted = StandardScaler::new();
    fitted.fit(&x).unwrap();

    assert_eq!(incremental.get_mean(), fitted.get_mean());
    assert_eq!(incremental.get_scale(), fitted.get_scale());
}

/// A batch with a different feature count is rejected
#[test]
fn test_partial_fit_feature_mismatch() {
    let mut scaler = StandardScaler::new();
    scaler.partial_fit(&array![[1.0, 2.0]]).unwrap();

    let err = scaler.partial_fit(&array![[1.0, 2.0, 3.0]]).unwrap_err();
    match err {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 3);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

// Edge cases

/// A constant feature is divided by 1.0 and maps to zeros, leaving other features alone
#[test]
fn test_constant_feature() {
    let x = array![[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]];

    let mut scaler = StandardScaler::new();
    let z = scaler.fit_transform(&x).unwrap();

    assert_eq!(scaler.get_scale().unwrap()[0], 1.0);
    assert_allclose(&z.column(0).to_owned(), &array![0.0, 0.0, 0.0], 1e-12);
    assert!(z.iter().all(|v| v.is_finite()));
}

/// A single training sample gives zero variance everywhere, so every feature is constant
#[test]
fn test_single_sample_fit() {
    let mut scaler = StandardScaler::new();
    scaler.fit(&array![[5.0, -2.0]]).unwrap();

    assert_allclose(scaler.get_mean().unwrap(), &array![5.0, -2.0], 1e-12);
    assert_allclose(scaler.get_scale().unwrap(), &array![1.0, 1.0], 1e-12);
    assert_eq!(scaler.get_n_samples_seen(), 1);
}

// Traits

/// The scaler is usable through the shared `Fit` / `Transform` / `FitTransform` traits
#[test]
fn test_scaler_through_shared_traits() {
    let x_train = array![[1.0, 100.0], [2.0, 150.0], [3.0, 200.0]];
    let x_test = array![[4.0, 250.0]];

    let mut scaler = StandardScaler::new();
    Fit::fit(&mut scaler, &x_train).unwrap();
    let by_trait = Transform::transform(&scaler, &x_test).unwrap();
    let by_method = scaler.transform(&x_test).unwrap();
    assert_eq!(by_trait, by_method);

    let mut other = StandardScaler::new();
    let fitted = FitTransform::fit_transform(&mut other, &x_train).unwrap();
    assert_eq!(fitted.dim(), (3, 2));
}

// Persistence

/// A fitted scaler survives a save/load round trip and keeps transforming identically
#[test]
fn test_save_and_load_round_trip() {
    let x_train = array![[1.0, 100.0], [2.0, 150.0], [3.0, 200.0]];
    let x_test = array![[4.0, 250.0]];

    let mut scaler = StandardScaler::new().with_std(false);
    scaler.fit(&x_train).unwrap();
    let expected = scaler.transform(&x_test).unwrap();

    let path = std::env::temp_dir().join("rustyml_standard_scaler_round_trip.bin");
    let path = path.to_str().unwrap();
    scaler.save_to_path(path).unwrap();
    let loaded = StandardScaler::load_from_path(path).unwrap();
    std::fs::remove_file(path).unwrap();

    assert!(loaded.get_with_mean());
    assert!(!loaded.get_with_std());
    assert_eq!(loaded.get_n_samples_seen(), 3);
    assert_eq!(loaded.transform(&x_test).unwrap(), expected);
}

// Error paths

/// Transforming before fitting reports NotFitted
#[test]
fn test_transform_before_fit() {
    let scaler = StandardScaler::new();

    match scaler.transform(&array![[1.0, 2.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "StandardScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }
    match scaler.inverse_transform(&array![[1.0, 2.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "StandardScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }
}

/// A feature-count mismatch at transform time reports DimensionMismatch
#[test]
fn test_transform_feature_mismatch() {
    let mut scaler = StandardScaler::new();
    scaler.fit(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();

    match scaler.transform(&array![[1.0]]).unwrap_err() {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

/// Empty input is rejected at fit and at transform time
#[test]
fn test_empty_input() {
    let empty: Array2<f64> = Array2::zeros((0, 3));
    let featureless: Array2<f64> = Array2::zeros((3, 0));

    match StandardScaler::new().fit(&empty).unwrap_err() {
        Error::EmptyInput(_) => {}
        other => panic!("expected EmptyInput, got {other:?}"),
    }
    match StandardScaler::new().fit(&featureless).unwrap_err() {
        Error::EmptyInput(_) => {}
        other => panic!("expected EmptyInput, got {other:?}"),
    }

    let mut scaler = StandardScaler::new();
    scaler.fit(&array![[1.0, 2.0, 3.0]]).unwrap();
    match scaler.transform(&empty).unwrap_err() {
        Error::EmptyInput(_) => {}
        other => panic!("expected EmptyInput, got {other:?}"),
    }
}

/// Non-finite input is rejected at fit and at transform time
#[test]
fn test_non_finite_input() {
    match StandardScaler::new()
        .fit(&array![[1.0, f64::NAN], [3.0, 4.0]])
        .unwrap_err()
    {
        Error::NonFinite(_) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }

    let mut scaler = StandardScaler::new();
    scaler.fit(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
    match scaler.transform(&array![[1.0, f64::INFINITY]]).unwrap_err() {
        Error::NonFinite(_) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}
