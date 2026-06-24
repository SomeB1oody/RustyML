//! Integration tests for `rustyml::machine_learning::LDA`
//!
//! Expected values come from the problem design or closed-form definitions,
//! never from running the model and recording its output

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::{LDA, Shrinkage, Solver};

use crate::common::assert_allclose;

// Local helpers

/// Three well-separated classes in 2D (3 samples per class), centred near
/// (1,1), (5,5) and (9,1); ~4 units apart so every solver reaches 100% accuracy
fn make_three_class_2d() -> (Array2<f64>, Array1<i32>) {
    let x = array![
        [1.0, 1.0],
        [1.5, 0.8],
        [0.8, 1.2],
        [5.0, 5.0],
        [5.2, 4.8],
        [4.8, 5.2],
        [9.0, 1.0],
        [9.2, 0.8],
        [8.8, 1.2]
    ];
    let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
    (x, y)
}

/// Holdout test set: one sample clearly within each of the three clusters
fn make_three_class_holdout() -> (Array2<f64>, Array1<i32>) {
    let x = array![
        [1.1, 1.0], // -> class 0
        [5.1, 5.0], // -> class 1
        [9.1, 1.0], // -> class 2
    ];
    let y = array![0, 1, 2];
    (x, y)
}

/// Minimal 2-class, 1-feature dataset with hand-verifiable projections: class 0
/// is {1,2,3} (mean 2.0), class 1 is {7,8,9} (mean 8.0), grand mean 5.0
fn make_two_class_1d() -> (Array2<f64>, Array1<i32>) {
    let x = array![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]];
    let y = array![0, 0, 0, 1, 1, 1];
    (x, y)
}

/// Fraction of predictions that match `true_labels`
fn accuracy(predicted: &Array1<i32>, true_labels: &Array1<i32>) -> f64 {
    let correct = predicted
        .iter()
        .zip(true_labels.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f64 / true_labels.len() as f64
}

// 1. Constructor validation

#[test]
fn test_new_default_values() {
    let lda = LDA::new(2).expect("default construction should succeed");
    assert_eq!(lda.get_n_components(), Some(2));
    assert_eq!(lda.get_solver(), Solver::SVD);
    assert!(lda.get_shrinkage().is_none());
    // Pre-fit getters return None
    assert!(lda.get_classes().is_none());
    assert!(lda.get_priors().is_none());
    assert!(lda.get_means().is_none());
    assert!(lda.get_projection().is_none());
}

#[test]
fn test_default_impl() {
    let lda = LDA::default();
    // The default is now `None` (auto): the component count is resolved at fit time to
    // min(n_classes - 1, n_features)
    assert_eq!(lda.get_n_components(), None);
    assert_eq!(lda.get_solver(), Solver::SVD);
    assert!(lda.get_shrinkage().is_none());
}

#[test]
fn test_new_explicit_solver_and_shrinkage() {
    let lda = LDA::new(1)
        .expect("valid construction")
        .with_solver(Solver::Eigen)
        .with_shrinkage(Shrinkage::Manual(0.3))
        .unwrap();
    assert_eq!(lda.get_solver(), Solver::Eigen);
    assert_eq!(lda.get_shrinkage(), Some(Shrinkage::Manual(0.3)));
}

#[test]
fn test_new_zero_components_errors() {
    let err = LDA::new(0).expect_err("n_components=0 must fail");
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn test_new_shrinkage_manual_invalid_alpha_errors() {
    // Manual shrinkage requires alpha in [0, 1] and finite: out-of-range and
    // non-finite alphas must all be rejected with InvalidParameter
    for alpha in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = LDA::new(1)
            .unwrap()
            .with_shrinkage(Shrinkage::Manual(alpha))
            .expect_err(&format!("alpha={alpha:?} must fail"));
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for alpha={alpha:?}, got {err:?}"
        );
    }
}

#[test]
fn test_new_shrinkage_boundary_zero_succeeds() {
    LDA::new(1)
        .expect("Manual(0.0) must be valid")
        .with_shrinkage(Shrinkage::Manual(0.0))
        .unwrap();
}

#[test]
fn test_new_shrinkage_boundary_one_succeeds() {
    LDA::new(1)
        .expect("Manual(1.0) must be valid")
        .with_shrinkage(Shrinkage::Manual(1.0))
        .unwrap();
}

// 2. fit() error paths

#[test]
fn test_fit_empty_rows_errors() {
    let mut lda = LDA::new(1).unwrap();
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<i32> = Array1::zeros(0);
    let err = lda.fit(&x, &y).expect_err("empty x must fail");
    assert!(
        matches!(err, Error::EmptyInput(..)),
        "expected EmptyInput, got {err:?}"
    );
}

#[test]
fn test_fit_x_y_length_mismatch_errors() {
    let mut lda = LDA::new(1).unwrap();
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    // y has 2 entries but x has 3 rows
    let y = array![0, 1];
    let err = lda.fit(&x, &y).expect_err("x/y length mismatch must fail");
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

#[test]
fn test_fit_single_class_errors() {
    // All samples belong to class 0: LDA requires >= 2 classes
    let mut lda = LDA::new(1).unwrap();
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let y = array![0, 0, 0];
    let err = lda.fit(&x, &y).expect_err("single class must fail");
    assert!(
        matches!(err, Error::InvalidInput(..)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn test_fit_n_samples_le_n_classes_errors() {
    // 2 samples, 2 classes: n_samples (2) == n_classes (2), which is not > n_classes
    let mut lda = LDA::new(1).unwrap();
    let x = array![[1.0, 2.0], [5.0, 6.0]];
    let y = array![0, 1];
    let err = lda
        .fit(&x, &y)
        .expect_err("n_samples <= n_classes must fail");
    assert!(
        matches!(err, Error::InvalidInput(..)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn test_fit_class_with_single_sample_errors() {
    // Class 1 has only 1 sample, must get InvalidInput ("each class must have at least 2")
    let mut lda = LDA::new(1).unwrap();
    let x = array![
        [1.0, 1.0],
        [1.5, 1.0],
        [1.0, 1.5],
        [5.0, 5.0] // class 1 with exactly 1 sample
    ];
    let y = array![0, 0, 0, 1];
    let err = lda.fit(&x, &y).expect_err("class with 1 sample must fail");
    assert!(
        matches!(err, Error::InvalidInput(..)),
        "expected InvalidInput (per-class min), got {err:?}"
    );
}

#[test]
fn test_fit_n_components_exceeds_max_errors() {
    // 3 classes: max_components = min(n_classes-1, n_features) = min(2, 2) = 2, so
    // requesting n_components=3 should fail at fit time
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(3).unwrap(); // constructor allows any n_components > 0
    let err = lda
        .fit(&x, &y)
        .expect_err("n_components > n_classes-1 must fail at fit");
    assert!(
        matches!(err, Error::InvalidInput(..)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn test_fit_non_finite_in_x_errors() {
    // A non-finite entry anywhere in x must trip the finiteness guard in fit
    for sentinel in [f64::NAN, f64::INFINITY] {
        let mut lda = LDA::new(1).unwrap();
        let x = array![
            [1.0, sentinel],
            [2.0, 1.0],
            [3.0, 1.0],
            [7.0, 5.0],
            [8.0, 5.0],
            [9.0, 5.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];
        let err = lda
            .fit(&x, &y)
            .expect_err(&format!("{sentinel:?} in x must fail"));
        assert!(
            matches!(err, Error::NonFinite(..) | Error::InvalidInput(..)),
            "expected NonFinite or InvalidInput for sentinel={sentinel:?}, got {err:?}"
        );
    }
}

// 3. predict / transform NotFitted errors

#[test]
fn test_predict_before_fit_errors() {
    let lda = LDA::new(1).unwrap();
    let x = array![[1.0, 2.0]];
    let err = lda.predict(&x).expect_err("predict before fit must fail");
    assert!(
        matches!(err, Error::NotFitted(..)),
        "expected NotFitted, got {err:?}"
    );
}

#[test]
fn test_transform_before_fit_errors() {
    let lda = LDA::new(1).unwrap();
    let x = array![[1.0, 2.0]];
    let err = lda
        .transform(&x)
        .expect_err("transform before fit must fail");
    assert!(
        matches!(err, Error::NotFitted(..)),
        "expected NotFitted, got {err:?}"
    );
}

// 4. predict / transform dimension-mismatch errors (after fit)

#[test]
fn test_predict_wrong_feature_count_errors() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    // 3 features instead of 2
    let x_bad = array![[1.0, 2.0, 3.0]];
    let err = lda
        .predict(&x_bad)
        .expect_err("feature count mismatch in predict must fail");
    assert!(
        matches!(
            err,
            Error::DimensionMismatch { .. } | Error::InvalidInput(..)
        ),
        "expected DimensionMismatch or InvalidInput, got {err:?}"
    );
}

#[test]
fn test_transform_wrong_feature_count_errors() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let x_bad = array![[1.0, 2.0, 3.0]];
    let err = lda
        .transform(&x_bad)
        .expect_err("feature count mismatch in transform must fail");
    assert!(
        matches!(
            err,
            Error::DimensionMismatch { .. } | Error::InvalidInput(..)
        ),
        "expected DimensionMismatch or InvalidInput, got {err:?}"
    );
}

// 5. Correctness: fit -> predict on training data (3-class, SVD solver)

#[test]
fn test_fit_predict_train_100pct_svd() {
    // Clusters 4 units apart, so perfect classification is guaranteed
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let preds = lda.predict(&x).unwrap();
    assert_eq!(
        accuracy(&preds, &y),
        1.0,
        "training accuracy on well-separated 3-class data must be 100 %"
    );
}

#[test]
fn test_fit_predict_holdout_svd() {
    // Holdout samples sit clearly in each cluster, so must be classified correctly
    let (x_train, y_train) = make_three_class_2d();
    let (x_test, y_test) = make_three_class_holdout();

    let mut lda = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda.fit(&x_train, &y_train).unwrap();

    let preds = lda.predict(&x_test).unwrap();
    assert_eq!(
        accuracy(&preds, &y_test),
        1.0,
        "holdout accuracy must be 100 % on well-separated clusters"
    );
}

#[test]
fn test_predict_labels_are_i32_domain() {
    // Contract: predict emits i32 labels identical to the training labels {0, 1, 2}
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let preds = lda.predict(&x).unwrap();
    for p in preds.iter() {
        assert!(
            *p == 0 || *p == 1 || *p == 2,
            "predicted label {p} not in {{0,1,2}}"
        );
    }
}

#[test]
fn test_classes_sorted_after_fit() {
    // classes_vec is sorted_unstable before storing; confirm the getter reflects this
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let classes = lda.get_classes().expect("classes must be set after fit");
    assert_eq!(classes.as_slice().unwrap(), &[0, 1, 2]);
}

// 6. Correctness: all three solvers

#[test]
fn test_all_solvers_classify_correctly() {
    let (x, y) = make_three_class_2d();
    for solver in [Solver::SVD, Solver::Eigen, Solver::LSQR] {
        let mut lda = LDA::new(2).unwrap().with_solver(solver);
        lda.fit(&x, &y).unwrap();

        let preds = lda.predict(&x).unwrap();
        assert_eq!(
            accuracy(&preds, &y),
            1.0,
            "solver {solver:?} must achieve 100 % on well-separated data"
        );
    }
}

// 7. transform() output shape and finiteness

#[test]
fn test_transform_output_shape_2d_3class() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let out = lda.transform(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[9, 2],
        "transform shape must be [n_samples, n_components]"
    );
}

#[test]
fn test_transform_output_finite() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let out = lda.transform(&x).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert!(v.is_finite(), "transform output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_transform_n_components_1() {
    // Reduce to 1 component (n_classes-1 = 2, so 1 is valid)
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(1).unwrap();
    lda.fit(&x, &y).unwrap();

    let out = lda.transform(&x).unwrap();
    assert_eq!(out.shape(), &[9, 1]);
    for v in out.iter() {
        assert!(v.is_finite());
    }
}

#[test]
fn test_transform_single_sample() {
    // Single-sample inputs always take the sequential path (< 500)
    let (x_train, y_train) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x_train, &y_train).unwrap();

    let x_one = array![[5.0, 5.0]]; // should map to class 1
    let out = lda.transform(&x_one).unwrap();
    assert_eq!(out.shape(), &[1, 2]);
    for v in out.iter() {
        assert!(v.is_finite());
    }
}

// 8. Hand-computed reference for 2-class 1D case

#[test]
fn test_two_class_1d_classification_correctness() {
    // class 0 ~ {1,2,3}, class 1 ~ {7,8,9}: any solver must classify all 6 samples
    let (x, y) = make_two_class_1d();
    let mut lda = LDA::new(1).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let preds = lda.predict(&x).unwrap();
    assert_eq!(
        accuracy(&preds, &y),
        1.0,
        "2-class 1D LDA must classify training data with 100 % accuracy"
    );
}

#[test]
fn test_two_class_1d_projected_class_separation() {
    // After projecting onto 1 component the class means must separate: with a unit
    // vector w, |8w - 2w| = 6|w| = 6.0
    let (x, y) = make_two_class_1d();
    let mut lda = LDA::new(1).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let proj = lda.transform(&x).unwrap(); // shape [6, 1]

    // Collect projected values by known class membership
    let proj_class0: Vec<f64> = (0..3).map(|i| proj[[i, 0]]).collect();
    let proj_class1: Vec<f64> = (3..6).map(|i| proj[[i, 0]]).collect();

    let mean0: f64 = proj_class0.iter().sum::<f64>() / 3.0;
    let mean1: f64 = proj_class1.iter().sum::<f64>() / 3.0;

    // |projected mean diff| = |8 - 2| * |w| should be close to 6.0
    assert_abs_diff_eq!((mean1 - mean0).abs(), 6.0, epsilon = 0.05);

    // Within each class the projections stay monotone (strictly increasing or
    // strictly decreasing), matching the original ordering
    let class0_sorted_asc = proj_class0[0] < proj_class0[1] && proj_class0[1] < proj_class0[2];
    let class0_sorted_desc = proj_class0[0] > proj_class0[1] && proj_class0[1] > proj_class0[2];
    assert!(
        class0_sorted_asc || class0_sorted_desc,
        "class-0 projections {proj_class0:?} must be monotone"
    );
}

#[test]
fn test_two_class_1d_unit_projection_vector() {
    // For a 1-feature dataset with n_components=1 the projection is a single unit
    // vector (L2-norm = 1.0)
    let (x, y) = make_two_class_1d();
    let mut lda = LDA::new(1).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let w = lda.get_projection().expect("projection must be set");
    // w shape: [n_features=1, n_components=1]
    assert_eq!(w.shape(), &[1, 1]);
    let norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
}

// 9. fit_transform consistency

#[test]
fn test_fit_transform_equals_fit_then_transform() {
    // fit_transform(x, y) must match fit(x, y) then transform(x)
    let (x, y) = make_three_class_2d();

    let mut lda_a = LDA::new(2).unwrap().with_solver(Solver::SVD);
    let out_a = lda_a.fit_transform(&x, &y).unwrap();

    let mut lda_b = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda_b.fit(&x, &y).unwrap();
    let out_b = lda_b.transform(&x).unwrap();

    assert_allclose(&out_a, &out_b, 1e-12);
}

#[test]
fn test_fit_transform_sets_projection() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit_transform(&x, &y).unwrap();
    assert!(
        lda.get_projection().is_some(),
        "projection must be set after fit_transform"
    );
}

// 10. Shrinkage variants

#[test]
fn test_shrinkage_auto_classifies_correctly() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2)
        .unwrap()
        .with_shrinkage(Shrinkage::Auto)
        .unwrap();
    lda.fit(&x, &y).unwrap();
    let preds = lda.predict(&x).unwrap();
    assert_eq!(accuracy(&preds, &y), 1.0);
}

#[test]
fn test_shrinkage_manual_half_classifies_correctly() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2)
        .unwrap()
        .with_shrinkage(Shrinkage::Manual(0.5))
        .unwrap();
    lda.fit(&x, &y).unwrap();
    let preds = lda.predict(&x).unwrap();
    assert_eq!(accuracy(&preds, &y), 1.0);
}

#[test]
fn test_shrinkage_manual_zero_matches_no_shrinkage() {
    // Manual(0.0): the alpha <= 0.0 early-return returns the unmodified covariance,
    // so the result must be identical to None shrinkage
    let (x, y) = make_three_class_2d();

    let mut lda_none = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda_none.fit(&x, &y).unwrap();
    let out_none = lda_none.transform(&x).unwrap();

    let mut lda_zero = LDA::new(2)
        .unwrap()
        .with_solver(Solver::SVD)
        .with_shrinkage(Shrinkage::Manual(0.0))
        .unwrap();
    lda_zero.fit(&x, &y).unwrap();
    let out_zero = lda_zero.transform(&x).unwrap();

    assert_allclose(&out_none, &out_zero, 1e-10);
}

#[test]
fn test_shrinkage_manual_boundary_one_produces_finite_output() {
    // Manual(1.0): shrinks the covariance entirely to a scaled identity; fit and
    // transform should still succeed on well-separated data
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2)
        .unwrap()
        .with_shrinkage(Shrinkage::Manual(1.0))
        .unwrap();
    lda.fit(&x, &y).unwrap();
    let out = lda.transform(&x).unwrap();
    for v in out.iter() {
        assert!(v.is_finite());
    }
}

// 11. Projection columns are unit-norm

#[test]
fn test_projection_columns_are_unit_norm_svd() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let w = lda.get_projection().unwrap(); // shape [2, 2]
    for col_idx in 0..w.ncols() {
        let col = w.column(col_idx);
        let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "column {col_idx} of projection must be a unit vector"
        );
    }
}

#[test]
fn test_projection_columns_are_unit_norm_eigen() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap().with_solver(Solver::Eigen);
    lda.fit(&x, &y).unwrap();

    let w = lda.get_projection().unwrap();
    for col_idx in 0..w.ncols() {
        let col = w.column(col_idx);
        let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_projection_columns_are_unit_norm_lsqr() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap().with_solver(Solver::LSQR);
    lda.fit(&x, &y).unwrap();

    let w = lda.get_projection().unwrap();
    for col_idx in 0..w.ncols() {
        let col = w.column(col_idx);
        let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
    }
}

// 12. Priors sum to 1 after fit

#[test]
fn test_priors_sum_to_one() {
    // Priors are n_class/n_samples per class and must sum to 1.0
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let priors = lda.get_priors().expect("priors must be set after fit");
    let sum: f64 = priors.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
}

#[test]
fn test_priors_equal_for_balanced_classes() {
    // 9 samples, 3 balanced classes: prior for each class = 3/9 = 1/3
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let priors = lda.get_priors().unwrap();
    for p in priors.iter() {
        assert_abs_diff_eq!(*p, 1.0 / 3.0, epsilon = 1e-12);
    }
}

// 13. Class means after fit

#[test]
fn test_class_means_correct_2class_1d() {
    // Class 0: [1,2,3] -> mean = 2.0; Class 1: [7,8,9] -> mean = 8.0
    let (x, y) = make_two_class_1d();
    let mut lda = LDA::new(1).unwrap();
    lda.fit(&x, &y).unwrap();

    let means = lda.get_means().expect("means must be set");
    // classes are sorted: [0, 1] -> row 0 = class 0, row 1 = class 1
    assert_abs_diff_eq!(means[[0, 0]], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(means[[1, 0]], 8.0, epsilon = 1e-12);
}

// 14. Determinism (same seed -> identical result)

#[test]
fn test_determinism_svd_same_data() {
    // LDA is deterministic, so fitting twice on the same data yields identical
    // transforms
    let (x, y) = make_three_class_2d();

    let mut lda1 = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda1.fit(&x, &y).unwrap();
    let out1 = lda1.transform(&x).unwrap();

    let mut lda2 = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda2.fit(&x, &y).unwrap();
    let out2 = lda2.transform(&x).unwrap();

    // Eigenvector sign may flip between runs, so compare up to a per-column sign
    assert_eq!(out1.shape(), out2.shape());
    for col in 0..out1.ncols() {
        let col1: Vec<f64> = out1.column(col).iter().copied().collect();
        let col2: Vec<f64> = out2.column(col).iter().copied().collect();
        let sign = if col1[0] * col2[0] >= 0.0 { 1.0 } else { -1.0 };
        for (a, b) in col1.iter().zip(col2.iter()) {
            assert_abs_diff_eq!(a, &(sign * b), epsilon = 1e-12);
        }
    }
}

// 15. save_to_path / load_from_path round-trip

#[test]
fn test_save_load_round_trip() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let preds_before = lda.predict(&x).unwrap();
    let proj_before = lda.transform(&x).unwrap();

    let path = "/tmp/rustyml_lda_roundtrip_test.json";
    lda.save_to_path(path).expect("save must succeed");

    let lda_loaded = LDA::load_from_path(path).expect("load must succeed");

    let preds_after = lda_loaded.predict(&x).unwrap();
    let proj_after = lda_loaded.transform(&x).unwrap();

    assert_eq!(
        preds_before, preds_after,
        "predictions must survive JSON round-trip"
    );
    // Projections must survive the round-trip to within floating-point tolerance
    assert_allclose(&proj_before, &proj_after, 1e-12);
}

#[test]
fn test_save_load_preserves_hyperparameters() {
    let mut lda = LDA::new(1)
        .unwrap()
        .with_solver(Solver::Eigen)
        .with_shrinkage(Shrinkage::Manual(0.3))
        .unwrap();
    let (x, y) = make_two_class_1d();
    lda.fit(&x, &y).unwrap();

    let path = "/tmp/rustyml_lda_hp_roundtrip_test.json";
    lda.save_to_path(path).unwrap();
    let loaded = LDA::load_from_path(path).unwrap();

    assert_eq!(loaded.get_n_components(), Some(1));
    assert_eq!(loaded.get_solver(), Solver::Eigen);
    assert_eq!(loaded.get_shrinkage(), Some(Shrinkage::Manual(0.3)));
}

#[test]
fn test_load_preserves_fit_state() {
    // After round-trip, get_classes / get_priors / get_means / get_projection must
    // all be Some and match the pre-save state
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::new(2).unwrap();
    lda.fit(&x, &y).unwrap();

    let path = "/tmp/rustyml_lda_state_roundtrip_test.json";
    lda.save_to_path(path).unwrap();
    let loaded = LDA::load_from_path(path).unwrap();

    assert!(
        loaded.get_classes().is_some(),
        "classes must survive round-trip"
    );
    assert!(
        loaded.get_priors().is_some(),
        "priors must survive round-trip"
    );
    assert!(
        loaded.get_means().is_some(),
        "means must survive round-trip"
    );
    assert!(
        loaded.get_projection().is_some(),
        "projection must survive round-trip"
    );
}

// 16. Large-batch Rayon paths (fit `use_parallel` + predict parallel scoring)

/// 600-sample (200 per class, > LDA_PARALLEL_THRESHOLD = 500) 3-class 2D dataset
/// that is linearly separable by construction
///
/// Centroids (0,0), (10,10), (20,0) each get a deterministic `sin`-based jitter
/// bounded by ~1.131, far below the >= 7.071 margin to any class boundary, so
/// every sample lies on the correct side and expected accuracy = 1.0
fn make_large_separable_3class_2d() -> (Array2<f64>, Array1<i32>) {
    const PER_CLASS: usize = 200;
    let centroids: [(f64, f64); 3] = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)];
    let n = PER_CLASS * centroids.len();

    let mut data: Vec<f64> = Vec::with_capacity(n * 2);
    let mut labels: Vec<i32> = Vec::with_capacity(n);

    for (class, &(cx, cy)) in centroids.iter().enumerate() {
        for i in 0..PER_CLASS {
            // Deterministic, bounded jitter in [-0.8, 0.8] on each axis
            let t = (class * PER_CLASS + i) as f64;
            let jx = (t * 0.7).sin() * 0.8;
            let jy = (t * 1.3 + 0.5).sin() * 0.8;
            data.push(cx + jx);
            data.push(cy + jy);
            labels.push(class as i32);
        }
    }

    let x = Array2::from_shape_vec((n, 2), data).unwrap();
    let y = Array1::from_vec(labels);
    (x, y)
}

#[test]
fn test_fit_predict_large_separable_parallel_paths() {
    // 600 samples (> 500) exercises both Rayon branches (fit per-class par_iter and
    // predict parallel scoring); separable data must still reach 100% accuracy
    let (x, y) = make_large_separable_3class_2d();
    assert!(
        x.nrows() > 500,
        "dataset must exceed the 500-sample parallel threshold, got {}",
        x.nrows()
    );

    let mut lda = LDA::new(2).unwrap().with_solver(Solver::SVD);
    lda.fit(&x, &y).unwrap();

    let preds = lda.predict(&x).unwrap();
    assert_eq!(
        preds.len(),
        y.len(),
        "prediction count must match the number of input rows"
    );
    assert_eq!(
        accuracy(&preds, &y),
        1.0,
        "linearly-separable 3-class data must be classified perfectly on the parallel path"
    );
}

// Auto n_components (default = None): the cap is min(n_classes-1, n_features), applied
// automatically so the default model fits any valid dataset (binary included)

/// `LDA::default()` must fit binary data: the previous default (n_components=2) wrongly
/// errored because the cap for 2 classes is 1. The corrected default auto-caps to 1.
#[test]
fn test_default_fits_binary_data_via_auto_n_components() {
    let (x, y) = make_two_class_1d();
    let mut lda = LDA::default();
    lda.fit(&x, &y)
        .expect("default LDA must fit binary data via auto n_components");
    let transformed = lda.transform(&x).unwrap();
    assert_eq!(
        transformed.ncols(),
        1,
        "binary LDA must auto-project to min(n_classes-1, n_features)=1 component"
    );
}

/// The auto default also caps correctly for multiclass: 3 classes, 2 features ->
/// min(2, 2) = 2 components
#[test]
fn test_default_auto_caps_multiclass_components() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::default();
    lda.fit(&x, &y).unwrap();
    let transformed = lda.transform(&x).unwrap();
    assert_eq!(
        transformed.ncols(),
        2,
        "3-class/2-feature LDA caps to 2 components"
    );
}

// decision_function (per-class discriminant scores) and predict_proba (softmax posteriors)

/// Helper: index of the max element in a 1D view
fn argmax(row: ndarray::ArrayView1<f64>) -> usize {
    let mut best_j = 0;
    let mut best = f64::NEG_INFINITY;
    for (j, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_j = j;
        }
    }
    best_j
}

/// decision_function returns an (n_samples, n_classes) matrix whose per-row argmax equals
/// the label that predict returns
#[test]
fn test_decision_function_shape_and_argmax_matches_predict() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::default();
    lda.fit(&x, &y).unwrap();

    let scores = lda.decision_function(&x).unwrap();
    assert_eq!(
        scores.shape(),
        &[9, 3],
        "shape must be (n_samples, n_classes)"
    );

    let preds = lda.predict(&x).unwrap();
    let classes = lda.get_classes().unwrap();
    for i in 0..scores.nrows() {
        assert_eq!(
            preds[i],
            classes[argmax(scores.row(i))],
            "predict must equal argmax of the decision function at row {i}"
        );
    }
}

/// predict_proba rows are valid probability distributions: every entry in [0,1] and each
/// row sums to 1
#[test]
fn test_predict_proba_rows_are_distributions() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::default();
    lda.fit(&x, &y).unwrap();

    let proba = lda.predict_proba(&x).unwrap();
    assert_eq!(proba.shape(), &[9, 3]);
    for row in proba.outer_iter() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        for &p in row.iter() {
            assert!((0.0..=1.0).contains(&p), "probability {p} outside [0,1]");
        }
    }
}

/// Ground truth: predict_proba is exactly the row-wise softmax of decision_function
#[test]
fn test_predict_proba_equals_softmax_of_decision_function() {
    let (x, y) = make_three_class_2d();
    let mut lda = LDA::default();
    lda.fit(&x, &y).unwrap();

    let scores = lda.decision_function(&x).unwrap();
    let proba = lda.predict_proba(&x).unwrap();
    assert_eq!(scores.shape(), proba.shape());

    for i in 0..scores.nrows() {
        let row = scores.row(i);
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..row.len() {
            assert_abs_diff_eq!(proba[[i, j]], exps[j] / sum, epsilon = 1e-12);
        }
    }
}

/// On well-separated classes the predicted class receives a posterior close to 1
#[test]
fn test_predict_proba_confident_on_separated_data() {
    let (x, y) = make_three_class_2d();
    let (xh, yh) = make_three_class_holdout();
    let mut lda = LDA::default();
    lda.fit(&x, &y).unwrap();

    let proba = lda.predict_proba(&xh).unwrap();
    let classes = lda.get_classes().unwrap();
    for (i, &true_label) in yh.iter().enumerate() {
        let j = classes.iter().position(|&c| c == true_label).unwrap();
        assert!(
            proba[[i, j]] > 0.99,
            "expected a confident posterior for the true class, got {}",
            proba[[i, j]]
        );
    }
}

/// decision_function and predict_proba require a fitted model
#[test]
fn test_decision_function_and_predict_proba_not_fitted_error() {
    let lda = LDA::default();
    let (x, _) = make_three_class_2d();
    assert!(matches!(
        lda.decision_function(&x),
        Err(Error::NotFitted("LDA"))
    ));
    assert!(matches!(
        lda.predict_proba(&x),
        Err(Error::NotFitted("LDA"))
    ));
}
