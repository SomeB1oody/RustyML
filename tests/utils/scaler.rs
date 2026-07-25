//! Integration tests for the `rustyml::utils::scaler` family - `StandardScaler`,
//! `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and `Normalizer` - covering the
//! fit/transform contract, incremental fitting, persistence, and the error paths

use crate::common::assert_allclose;
use ndarray::{Array2, array};
use rustyml::error::Error;
use rustyml::traits::{Fit, FitTransform, Transform};
use rustyml::utils::normalize::{NormalizationAxis, NormalizationOrder, normalize};
use rustyml::utils::standardize::{StandardizationAxis, standardize};
use rustyml::utils::{MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler};

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

// ============================================================================
// MinMaxScaler
// ============================================================================

/// The training extrema are recorded and mapped onto the default `[0, 1]` range
#[test]
fn test_min_max_default_range() {
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

    let mut scaler = MinMaxScaler::new();
    let z = scaler.fit_transform(&x).unwrap();

    assert_allclose(&z, &array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], 1e-12);
    assert_allclose(scaler.get_data_min().unwrap(), &array![1.0, 10.0], 1e-12);
    assert_allclose(scaler.get_data_max().unwrap(), &array![3.0, 30.0], 1e-12);
    assert_allclose(&scaler.get_data_range().unwrap(), &array![2.0, 20.0], 1e-12);
    assert_eq!(scaler.get_feature_range(), (0.0, 1.0));
    assert_eq!(scaler.get_n_samples_seen(), 3);
    assert_eq!(scaler.get_n_features(), Some(2));
}

/// A later batch is mapped by the training extrema, so it may leave the target range
#[test]
fn test_min_max_uses_frozen_training_extrema() {
    let x_train = array![[1.0], [3.0]];

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&x_train).unwrap();

    // 5.0 is twice the training span above the training minimum
    assert_allclose(
        &scaler.transform(&array![[5.0], [-1.0]]).unwrap(),
        &array![[2.0], [-1.0]],
        1e-12,
    );
}

/// A custom feature range retargets the map, including on an already-fitted scaler
#[test]
fn test_min_max_custom_feature_range() {
    let x = array![[0.0], [10.0]];

    let mut scaler = MinMaxScaler::new().with_feature_range(-1.0, 1.0).unwrap();
    assert_allclose(
        &scaler.fit_transform(&x).unwrap(),
        &array![[-1.0], [1.0]],
        1e-12,
    );

    // Retargeting after fitting keeps the learned extrema and only moves the destination
    let retargeted = scaler.clone().with_feature_range(0.0, 100.0).unwrap();
    assert_allclose(
        &retargeted.transform(&x).unwrap(),
        &array![[0.0], [100.0]],
        1e-12,
    );
    assert_allclose(retargeted.get_data_max().unwrap(), &array![10.0], 1e-12);
}

/// An invalid feature range is rejected with InvalidParameter
#[test]
fn test_min_max_invalid_feature_range() {
    for (low, high) in [(1.0, 1.0), (2.0, 1.0), (0.0, f64::INFINITY)] {
        match MinMaxScaler::new()
            .with_feature_range(low, high)
            .unwrap_err()
        {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "feature_range"),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }
}

/// Clipping clamps out-of-range values into the target interval
#[test]
fn test_min_max_clip() {
    let x_train = array![[1.0], [3.0]];
    let x_new = array![[5.0], [-1.0]];

    let mut open = MinMaxScaler::new();
    open.fit(&x_train).unwrap();
    assert!(!open.get_clip());
    assert_allclose(
        &open.transform(&x_new).unwrap(),
        &array![[2.0], [-1.0]],
        1e-12,
    );

    let mut clipped = MinMaxScaler::new().with_clip(true);
    clipped.fit(&x_train).unwrap();
    assert!(clipped.get_clip());
    assert_allclose(
        &clipped.transform(&x_new).unwrap(),
        &array![[1.0], [0.0]],
        1e-12,
    );
}

/// A constant feature lands on the low end of the range instead of producing NaN
#[test]
fn test_min_max_constant_feature() {
    let x = array![[7.0, 1.0], [7.0, 3.0], [7.0, 5.0]];

    let mut scaler = MinMaxScaler::new().with_feature_range(-2.0, 2.0).unwrap();
    let z = scaler.fit_transform(&x).unwrap();

    assert!(z.iter().all(|v| v.is_finite()));
    assert_allclose(&z.column(0).to_owned(), &array![-2.0, -2.0, -2.0], 1e-12);
    assert_allclose(&z.column(1).to_owned(), &array![-2.0, 0.0, 2.0], 1e-12);
}

/// `partial_fit` widens the stored interval and matches a single fit over the concatenation
#[test]
fn test_min_max_partial_fit() {
    let full = array![[2.0, 5.0], [3.0, 4.0], [1.0, 9.0], [6.0, 7.0]];

    let mut incremental = MinMaxScaler::new();
    incremental
        .partial_fit(&full.slice(ndarray::s![0..2, ..]))
        .unwrap();
    incremental
        .partial_fit(&full.slice(ndarray::s![2..4, ..]))
        .unwrap();

    let mut single = MinMaxScaler::new();
    single.fit(&full).unwrap();

    assert_allclose(
        incremental.get_data_min().unwrap(),
        single.get_data_min().unwrap(),
        1e-12,
    );
    assert_allclose(
        incremental.get_data_max().unwrap(),
        single.get_data_max().unwrap(),
        1e-12,
    );
    assert_eq!(incremental.get_n_samples_seen(), 4);

    // A batch of a different width is rejected
    match incremental.partial_fit(&array![[1.0]]).unwrap_err() {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

/// `inverse_transform` round-trips back to the original values
#[test]
fn test_min_max_inverse_transform() {
    let x = array![[1.0, -5.0], [2.0, 7.5], [3.0, 0.5], [4.0, 100.0]];

    let mut scaler = MinMaxScaler::new().with_feature_range(-3.0, 6.0).unwrap();
    let z = scaler.fit_transform(&x).unwrap();
    let restored = scaler.inverse_transform(&z).unwrap();

    assert_allclose(&restored, &x, 1e-9);
}

/// A fitted MinMaxScaler survives a save/load round trip
#[test]
fn test_min_max_save_and_load() {
    let x_train = array![[1.0, 10.0], [3.0, 30.0]];
    let x_test = array![[2.0, 25.0]];

    let mut scaler = MinMaxScaler::new()
        .with_feature_range(-1.0, 1.0)
        .unwrap()
        .with_clip(true);
    scaler.fit(&x_train).unwrap();
    let expected = scaler.transform(&x_test).unwrap();

    let path = std::env::temp_dir().join("rustyml_min_max_scaler_round_trip.bin");
    let path = path.to_str().unwrap();
    scaler.save_to_path(path).unwrap();
    let loaded = MinMaxScaler::load_from_path(path).unwrap();
    std::fs::remove_file(path).unwrap();

    assert_eq!(loaded.get_feature_range(), (-1.0, 1.0));
    assert!(loaded.get_clip());
    assert_eq!(loaded.transform(&x_test).unwrap(), expected);
}

/// MinMaxScaler error paths
#[test]
fn test_min_max_errors() {
    let unfitted = MinMaxScaler::new();
    match unfitted.transform(&array![[1.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "MinMaxScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }
    match unfitted.inverse_transform(&array![[1.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "MinMaxScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
    match scaler.transform(&array![[1.0]]).unwrap_err() {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
    match scaler.transform(&array![[1.0, f64::NAN]]).unwrap_err() {
        Error::NonFinite(_) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }

    let empty: Array2<f64> = Array2::zeros((0, 2));
    match MinMaxScaler::new().fit(&empty).unwrap_err() {
        Error::EmptyInput(_) => {}
        other => panic!("expected EmptyInput, got {other:?}"),
    }
}

// ============================================================================
// MaxAbsScaler
// ============================================================================

/// Each feature is divided by its own magnitude, leaving zeros and signs intact
#[test]
fn test_max_abs_divides_by_magnitude() {
    let x = array![[1.0, -4.0], [0.0, 2.0], [-2.0, 0.0]];

    let mut scaler = MaxAbsScaler::new();
    let z = scaler.fit_transform(&x).unwrap();

    assert_allclose(scaler.get_max_abs().unwrap(), &array![2.0, 4.0], 1e-12);
    assert_allclose(&z, &array![[0.5, -1.0], [0.0, 0.5], [-1.0, 0.0]], 1e-12);
    assert_eq!(scaler.get_n_samples_seen(), 3);
    assert_eq!(scaler.get_n_features(), Some(2));
}

/// Structural zeros stay zero - the property that separates MaxAbs from MinMax
#[test]
fn test_max_abs_preserves_zeros_where_min_max_does_not() {
    let x = array![[0.0, 1.0], [4.0, 2.0], [8.0, 3.0]];

    let max_abs = MaxAbsScaler::new().fit_transform(&x).unwrap();
    let min_max = MinMaxScaler::new().fit_transform(&x).unwrap();

    // MaxAbs never shifts, so a zero entry stays exactly zero
    assert_eq!(max_abs[[0, 0]], 0.0);
    // MinMax shifts by the column minimum, so the zero in column 1 (min 1.0) moves
    assert_eq!(min_max[[0, 1]], 0.0);
    assert!((max_abs[[0, 1]] - 1.0 / 3.0).abs() < 1e-12);
}

/// Training values all land inside [-1, 1]
#[test]
fn test_max_abs_bounds_training_values() {
    let x = array![[10.0, -3.0], [-7.0, 1.0], [4.0, 2.0]];

    let z = MaxAbsScaler::new().fit_transform(&x).unwrap();

    assert!(z.iter().all(|v| v.abs() <= 1.0));
}

/// An all-zero feature keeps a divisor of 1.0 and stays zero
#[test]
fn test_max_abs_all_zero_feature() {
    let x = array![[0.0, 1.0], [0.0, 2.0]];

    let mut scaler = MaxAbsScaler::new();
    let z = scaler.fit_transform(&x).unwrap();

    assert_eq!(scaler.get_scale().unwrap()[0], 1.0);
    assert!(z.iter().all(|v| v.is_finite()));
    assert_allclose(&z.column(0).to_owned(), &array![0.0, 0.0], 1e-12);
}

/// `partial_fit` keeps the larger magnitude and matches a single fit
#[test]
fn test_max_abs_partial_fit() {
    let full = array![[1.0, -9.0], [-5.0, 2.0], [3.0, 4.0]];

    let mut incremental = MaxAbsScaler::new();
    incremental
        .partial_fit(&full.slice(ndarray::s![0..1, ..]))
        .unwrap();
    incremental
        .partial_fit(&full.slice(ndarray::s![1..3, ..]))
        .unwrap();

    let mut single = MaxAbsScaler::new();
    single.fit(&full).unwrap();

    assert_allclose(
        incremental.get_max_abs().unwrap(),
        single.get_max_abs().unwrap(),
        1e-12,
    );
    assert_eq!(incremental.get_n_samples_seen(), 3);
}

/// `inverse_transform` round-trips back to the original values
#[test]
fn test_max_abs_inverse_transform() {
    let x = array![[1.0, -5.0], [2.0, 7.5], [0.0, 0.5]];

    let mut scaler = MaxAbsScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    let restored = scaler.inverse_transform(&z).unwrap();

    assert_allclose(&restored, &x, 1e-9);
}

/// A fitted MaxAbsScaler survives a save/load round trip
#[test]
fn test_max_abs_save_and_load() {
    let x_train = array![[1.0, -4.0], [-2.0, 2.0]];
    let x_test = array![[1.0, 1.0]];

    let mut scaler = MaxAbsScaler::new();
    scaler.fit(&x_train).unwrap();
    let expected = scaler.transform(&x_test).unwrap();

    let path = std::env::temp_dir().join("rustyml_max_abs_scaler_round_trip.bin");
    let path = path.to_str().unwrap();
    scaler.save_to_path(path).unwrap();
    let loaded = MaxAbsScaler::load_from_path(path).unwrap();
    std::fs::remove_file(path).unwrap();

    assert_eq!(loaded.get_n_samples_seen(), 2);
    assert_eq!(loaded.transform(&x_test).unwrap(), expected);
}

/// MaxAbsScaler error paths
#[test]
fn test_max_abs_errors() {
    match MaxAbsScaler::new().transform(&array![[1.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "MaxAbsScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }

    let mut scaler = MaxAbsScaler::new();
    scaler.fit(&array![[1.0, 2.0]]).unwrap();
    match scaler.transform(&array![[1.0, 2.0, 3.0]]).unwrap_err() {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 3);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }

    match MaxAbsScaler::new()
        .fit(&array![[1.0, f64::INFINITY]])
        .unwrap_err()
    {
        Error::NonFinite(_) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

// ============================================================================
// Normalizer
// ============================================================================

/// Each row is scaled to unit norm under the configured order
#[test]
fn test_normalizer_orders() {
    let x = array![[3.0, 4.0]];

    let l2 = Normalizer::default().fit_transform(&x).unwrap();
    assert_allclose(&l2, &array![[0.6, 0.8]], 1e-12);

    let l1 = Normalizer::new(NormalizationOrder::L1)
        .unwrap()
        .fit_transform(&x)
        .unwrap();
    assert_allclose(&l1, &array![[3.0 / 7.0, 4.0 / 7.0]], 1e-12);

    let max = Normalizer::new(NormalizationOrder::Max)
        .unwrap()
        .fit_transform(&x)
        .unwrap();
    assert_allclose(&max, &array![[0.75, 1.0]], 1e-12);

    let lp = Normalizer::new(NormalizationOrder::Lp(3.0))
        .unwrap()
        .fit_transform(&x)
        .unwrap();
    let expected = normalize(&x, NormalizationAxis::Row, NormalizationOrder::Lp(3.0)).unwrap();
    assert_eq!(lp, expected);
}

/// The normalizer agrees with the free function on the Row axis
#[test]
fn test_normalizer_matches_free_function() {
    let x = array![[3.0, 4.0], [1.0, 2.0], [0.0, 0.0]];

    let by_object = Normalizer::default().fit_transform(&x).unwrap();
    let by_function = normalize(&x, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();

    assert_eq!(by_object, by_function);
}

/// Rows are scaled independently, so batching cannot change a row's result
#[test]
fn test_normalizer_is_batch_independent() {
    let x = array![[3.0, 4.0], [1.0, 1.0], [0.0, 5.0]];

    let mut normalizer = Normalizer::default();
    normalizer.fit(&x.slice(ndarray::s![0..1, ..])).unwrap();

    let whole = normalizer.transform(&x).unwrap();
    for (i, row) in x.rows().into_iter().enumerate() {
        let single = normalizer
            .transform(&row.to_owned().insert_axis(ndarray::Axis(0)))
            .unwrap();
        assert_eq!(whole.row(i).to_owned(), single.row(0).to_owned());
    }
}

/// A zero row has no direction and is left exactly as-is
#[test]
fn test_normalizer_zero_row() {
    let x = array![[3.0, 4.0], [0.0, 0.0]];

    let z = Normalizer::default().fit_transform(&x).unwrap();

    assert_allclose(&z.row(1).to_owned(), &array![0.0, 0.0], 1e-12);
}

/// A fitted Normalizer survives a save/load round trip, keeping its order
#[test]
fn test_normalizer_save_and_load() {
    let x = array![[3.0, 4.0], [1.0, 2.0]];

    let mut normalizer = Normalizer::new(NormalizationOrder::Lp(3.0)).unwrap();
    let expected = normalizer.fit_transform(&x).unwrap();

    let path = std::env::temp_dir().join("rustyml_normalizer_round_trip.bin");
    let path = path.to_str().unwrap();
    normalizer.save_to_path(path).unwrap();
    let loaded = Normalizer::load_from_path(path).unwrap();
    std::fs::remove_file(path).unwrap();

    assert_eq!(loaded.get_order(), NormalizationOrder::Lp(3.0));
    assert_eq!(loaded.get_n_features(), Some(2));
    assert_eq!(loaded.transform(&x).unwrap(), expected);
}

/// Normalizer error paths
#[test]
fn test_normalizer_errors() {
    match Normalizer::new(NormalizationOrder::Lp(-1.0)).unwrap_err() {
        Error::InvalidParameter { name, .. } => assert_eq!(name, "p"),
        other => panic!("expected InvalidParameter, got {other:?}"),
    }

    match Normalizer::default().transform(&array![[1.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "Normalizer"),
        other => panic!("expected NotFitted, got {other:?}"),
    }

    let mut normalizer = Normalizer::default();
    normalizer.fit(&array![[1.0, 2.0]]).unwrap();
    match normalizer.transform(&array![[1.0]]).unwrap_err() {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

// ============================================================================
// The family through the shared traits
// ============================================================================

/// Every scaler in the family is usable through `Fit` / `Transform` / `FitTransform`
#[test]
fn test_scaler_family_through_shared_traits() {
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

    let mut standard = StandardScaler::new();
    Fit::fit(&mut standard, &x).unwrap();
    assert_eq!(
        Transform::transform(&standard, &x).unwrap(),
        standard.transform(&x).unwrap()
    );

    let mut min_max = MinMaxScaler::new();
    Fit::fit(&mut min_max, &x).unwrap();
    assert_eq!(
        Transform::transform(&min_max, &x).unwrap(),
        min_max.transform(&x).unwrap()
    );
    assert_eq!(
        FitTransform::fit_transform(&mut MinMaxScaler::new(), &x)
            .unwrap()
            .dim(),
        (3, 2)
    );

    let mut max_abs = MaxAbsScaler::new();
    Fit::fit(&mut max_abs, &x).unwrap();
    assert_eq!(
        Transform::transform(&max_abs, &x).unwrap(),
        max_abs.transform(&x).unwrap()
    );
    assert_eq!(
        FitTransform::fit_transform(&mut MaxAbsScaler::new(), &x)
            .unwrap()
            .dim(),
        (3, 2)
    );

    let mut normalizer = Normalizer::default();
    Fit::fit(&mut normalizer, &x).unwrap();
    assert_eq!(
        Transform::transform(&normalizer, &x).unwrap(),
        normalizer.transform(&x).unwrap()
    );
    assert_eq!(
        FitTransform::fit_transform(&mut Normalizer::default(), &x)
            .unwrap()
            .dim(),
        (3, 2)
    );
}

// ============================================================================
// RobustScaler
// ============================================================================

/// Reproduces the worked example from scikit-learn's `RobustScaler` documentation
#[test]
fn test_robust_matches_scikit_learn_reference() {
    let x = array![[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]];

    let z = RobustScaler::new().fit_transform(&x).unwrap();

    assert_allclose(
        &z,
        &array![[0.0, -2.0, 0.0], [-1.0, 0.0, 0.4], [1.0, 0.0, -1.6]],
        1e-12,
    );
}

/// The quantiles use NumPy's linear interpolation, so they match a ported pipeline exactly
#[test]
fn test_robust_quantile_interpolation() {
    // Quantiles of [1, 2, 3, 4]: q25 = 1.75, median = 2.5, q75 = 3.25
    let x = array![[1.0], [2.0], [3.0], [4.0]];

    let mut scaler = RobustScaler::new();
    scaler.fit(&x).unwrap();

    assert_allclose(scaler.get_center().unwrap(), &array![2.5], 1e-12);
    assert_allclose(scaler.get_scale().unwrap(), &array![1.5], 1e-12);
    assert_eq!(scaler.get_quantile_range(), (25.0, 75.0));
    assert_eq!(scaler.get_n_samples_seen(), 4);
    assert_eq!(scaler.get_n_features(), Some(1));

    // An odd count puts the median on an exact index instead of between two
    let odd = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let mut scaler = RobustScaler::new();
    scaler.fit(&odd).unwrap();
    assert_allclose(scaler.get_center().unwrap(), &array![3.0], 1e-12);
    assert_allclose(scaler.get_scale().unwrap(), &array![2.0], 1e-12);
}

/// The robust statistics ignore an extreme value that wrecks the mean and standard deviation
#[test]
fn test_robust_resists_outliers() {
    let clean = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0]
    ];
    let spoiled = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [1e6]
    ];

    let mut robust_clean = RobustScaler::new();
    robust_clean.fit(&clean).unwrap();
    let mut robust_spoiled = RobustScaler::new();
    robust_spoiled.fit(&spoiled).unwrap();

    // Identical statistics: the outlier sits outside both quantile positions
    assert_eq!(robust_clean.get_center(), robust_spoiled.get_center());
    assert_eq!(robust_clean.get_scale(), robust_spoiled.get_scale());

    // StandardScaler on the same data moves a long way
    let mut standard_clean = StandardScaler::new();
    standard_clean.fit(&clean).unwrap();
    let mut standard_spoiled = StandardScaler::new();
    standard_spoiled.fit(&spoiled).unwrap();
    assert!(
        standard_spoiled.get_mean().unwrap()[0] > 100.0 * standard_clean.get_mean().unwrap()[0]
    );

    // And the eight in-distribution rows stay comparable under robust scaling
    let z = robust_spoiled.transform(&spoiled).unwrap();
    assert!(z.rows().into_iter().take(8).all(|row| row[0].abs() <= 1.0));
}

/// A custom quantile range widens the divisor, and setting it discards any previous fit
#[test]
fn test_robust_custom_quantile_range() {
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0]
    ];

    let mut iqr = RobustScaler::new();
    iqr.fit(&x).unwrap();

    let wide = RobustScaler::new().with_quantile_range(10.0, 90.0).unwrap();
    assert!(wide.get_center().is_none(), "changing the range must unfit");
    let mut wide = wide;
    wide.fit(&x).unwrap();

    assert!(wide.get_scale().unwrap()[0] > iqr.get_scale().unwrap()[0]);
    assert_eq!(wide.get_quantile_range(), (10.0, 90.0));

    // Refitting an already-fitted scaler through the builder also clears the old numbers
    let retargeted = iqr.clone().with_quantile_range(5.0, 95.0).unwrap();
    assert!(retargeted.get_scale().is_none());
    assert_eq!(retargeted.get_n_samples_seen(), 0);
}

/// An invalid quantile range is rejected with InvalidParameter
#[test]
fn test_robust_invalid_quantile_range() {
    for (low, high) in [
        (75.0, 25.0),
        (50.0, 50.0),
        (-0.1, 75.0),
        (25.0, 100.1),
        (f64::NAN, 75.0),
    ] {
        match RobustScaler::new()
            .with_quantile_range(low, high)
            .unwrap_err()
        {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "quantile_range"),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }
}

/// A constant feature has no spread, so it centers to zeros rather than dividing by zero
#[test]
fn test_robust_constant_feature() {
    let x = array![[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]];

    let mut scaler = RobustScaler::new();
    let z = scaler.fit_transform(&x).unwrap();

    assert_eq!(scaler.get_scale().unwrap()[0], 1.0);
    assert_allclose(&z.column(0).to_owned(), &array![0.0, 0.0, 0.0], 1e-12);
    assert!(z.iter().all(|v| v.is_finite()));
}

/// The flags switch centering and scaling off independently
#[test]
fn test_robust_flags() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];

    let centered = RobustScaler::new()
        .with_scaling(false)
        .fit_transform(&x)
        .unwrap();
    assert_allclose(&centered, &array![[-1.5], [-0.5], [0.5], [1.5]], 1e-12);

    let scaled = RobustScaler::new()
        .with_centering(false)
        .fit_transform(&x)
        .unwrap();
    assert_allclose(
        &scaled,
        &array![[1.0 / 1.5], [2.0 / 1.5], [3.0 / 1.5], [4.0 / 1.5]],
        1e-12,
    );

    // Statistics stay inspectable with both flags off
    let mut both_off = RobustScaler::new()
        .with_centering(false)
        .with_scaling(false);
    let untouched = both_off.fit_transform(&x).unwrap();
    assert_eq!(untouched, x);
    assert_allclose(both_off.get_center().unwrap(), &array![2.5], 1e-12);
}

/// `inverse_transform` round-trips back to the original values
#[test]
fn test_robust_inverse_transform() {
    let x = array![[1.0, -5.0], [2.0, 7.5], [3.0, 0.5], [4.0, 100.0]];

    let mut scaler = RobustScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    let restored = scaler.inverse_transform(&z).unwrap();

    assert_allclose(&restored, &x, 1e-9);
}

/// A fitted RobustScaler survives a save/load round trip
#[test]
fn test_robust_save_and_load() {
    let x_train = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
    let x_test = array![[2.5, 25.0]];

    let mut scaler = RobustScaler::new().with_quantile_range(20.0, 80.0).unwrap();
    scaler.fit(&x_train).unwrap();
    let expected = scaler.transform(&x_test).unwrap();

    let path = std::env::temp_dir().join("rustyml_robust_scaler_round_trip.bin");
    let path = path.to_str().unwrap();
    scaler.save_to_path(path).unwrap();
    let loaded = RobustScaler::load_from_path(path).unwrap();
    std::fs::remove_file(path).unwrap();

    assert_eq!(loaded.get_quantile_range(), (20.0, 80.0));
    assert_eq!(loaded.get_n_samples_seen(), 4);
    assert_eq!(loaded.transform(&x_test).unwrap(), expected);
}

/// RobustScaler error paths
#[test]
fn test_robust_errors() {
    let unfitted = RobustScaler::new();
    match unfitted.transform(&array![[1.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "RobustScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }
    match unfitted.inverse_transform(&array![[1.0]]).unwrap_err() {
        Error::NotFitted(model) => assert_eq!(model, "RobustScaler"),
        other => panic!("expected NotFitted, got {other:?}"),
    }

    let mut scaler = RobustScaler::new();
    scaler.fit(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
    match scaler.transform(&array![[1.0]]).unwrap_err() {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }

    match RobustScaler::new()
        .fit(&array![[1.0, f64::NAN]])
        .unwrap_err()
    {
        Error::NonFinite(_) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }

    let empty: Array2<f64> = Array2::zeros((0, 2));
    match RobustScaler::new().fit(&empty).unwrap_err() {
        Error::EmptyInput(_) => {}
        other => panic!("expected EmptyInput, got {other:?}"),
    }
}

/// RobustScaler is usable through the shared traits like the rest of the family
#[test]
fn test_robust_through_shared_traits() {
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];

    let mut scaler = RobustScaler::new();
    Fit::fit(&mut scaler, &x).unwrap();
    assert_eq!(
        Transform::transform(&scaler, &x).unwrap(),
        scaler.transform(&x).unwrap()
    );
    assert_eq!(
        FitTransform::fit_transform(&mut RobustScaler::new(), &x)
            .unwrap()
            .dim(),
        (4, 2)
    );
}
