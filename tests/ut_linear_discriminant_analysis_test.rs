#![cfg(feature = "utility")]

use ndarray::*;
use rustyml::utility::linear_discriminant_analysis::*;
use std::error::Error;

fn make_three_class_dataset() -> (Array2<f64>, Array1<i32>) {
    let x = arr2(&[
        [0.0, 0.2],
        [0.2, -0.1],
        [-0.1, 0.1],
        [5.0, 5.1],
        [5.2, 4.8],
        [4.8, 5.0],
        [10.0, 0.2],
        [9.9, -0.1],
        [10.1, 0.1],
    ]);
    let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
    (x, y)
}

fn classification_accuracy(pred: &Array1<i32>, y: &Array1<i32>) -> f64 {
    let correct = pred.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    correct as f64 / y.len() as f64
}

#[test]
fn test_lda_new_and_default() {
    let lda = LDA::new(2, None, None).unwrap();
    assert_eq!(lda.get_n_components(), 2);
    assert_eq!(lda.get_solver(), Solver::SVD);
    assert!(lda.get_shrinkage().is_none());
    assert!(lda.get_classes().is_none());
    assert!(lda.get_priors().is_none());
    assert!(lda.get_means().is_none());
    assert!(lda.get_cov_inv().is_none());
    assert!(lda.get_projection().is_none());

    let default_lda = LDA::default();
    assert_eq!(default_lda.get_n_components(), 2);
    assert_eq!(default_lda.get_solver(), Solver::SVD);
}

#[test]
fn test_lda_new_validation() {
    assert!(LDA::new(0, None, None).is_err());
    assert!(LDA::new(2, None, Some(Shrinkage::Manual(-0.1))).is_err());
    assert!(LDA::new(2, None, Some(Shrinkage::Manual(1.1))).is_err());
    assert!(LDA::new(2, None, Some(Shrinkage::Manual(f64::NAN))).is_err());
    assert!(LDA::new(2, None, Some(Shrinkage::Manual(0.25))).is_ok());
}

#[test]
fn test_fit_predict_transform() -> Result<(), Box<dyn Error>> {
    let (x, y) = make_three_class_dataset();
    let mut lda = LDA::new(2, Some(Solver::SVD), None)?;

    lda.fit(&x.view(), &y.view())?;

    assert!(lda.get_projection().is_some());
    assert_eq!(lda.get_projection().unwrap().shape(), &[2, 2]);

    let predictions = lda.predict(&x.view())?;
    assert_eq!(predictions.len(), y.len());
    assert!(classification_accuracy(&predictions, &y) >= 0.9);

    let transformed = lda.transform(&x.view())?;
    assert_eq!(transformed.shape(), &[x.nrows(), 2]);

    Ok(())
}

#[test]
fn test_fit_transform() -> Result<(), Box<dyn Error>> {
    let (x, y) = make_three_class_dataset();

    let mut lda = LDA::new(2, Some(Solver::SVD), Some(Shrinkage::Auto))?;
    let transformed = lda.fit_transform(&x.view(), &y.view())?;
    assert_eq!(transformed.shape(), &[x.nrows(), 2]);
    assert!(lda.get_projection().is_some());

    Ok(())
}

#[test]
fn test_fit_with_solvers() -> Result<(), Box<dyn Error>> {
    let (x, y) = make_three_class_dataset();
    let solvers = [Solver::SVD, Solver::Eigen, Solver::LSQR];

    for solver in solvers {
        let mut lda = LDA::new(2, Some(solver), None)?;
        lda.fit(&x.view(), &y.view())?;
        let predictions = lda.predict(&x.view())?;
        assert_eq!(predictions.len(), y.len());
        assert!(classification_accuracy(&predictions, &y) >= 0.7);
        let transformed = lda.transform(&x.view())?;
        assert_eq!(transformed.shape(), &[x.nrows(), 2]);
    }

    Ok(())
}

#[test]
fn test_errors_and_validation_paths() {
    let (x, y) = make_three_class_dataset();

    let lda = LDA::new(2, None, None).unwrap();
    assert!(lda.predict(&x.view()).is_err());
    assert!(lda.transform(&x.view()).is_err());

    let mut lda_bad_components = LDA::new(3, None, None).unwrap();
    assert!(lda_bad_components.fit(&x.view(), &y.view()).is_err());

    let mut lda_mismatch = LDA::new(2, None, None).unwrap();
    let y_short = Array1::from_vec(vec![0, 0, 1]);
    assert!(lda_mismatch.fit(&x.view(), &y_short.view()).is_err());

    let x_bad = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let y_bad = Array1::from_vec(vec![0, 1]);
    let mut lda_single = LDA::new(1, None, None).unwrap();
    assert!(lda_single.fit(&x_bad.view(), &y_bad.view()).is_err());
}

#[test]
fn test_lda_actual_performance() -> Result<(), Box<dyn Error>> {
    let x_train = arr2(&[
        [-0.2, 0.1],
        [0.1, 0.2],
        [-0.1, -0.2],
        [0.2, -0.1],
        [3.8, 4.1],
        [4.2, 3.9],
        [4.0, 4.2],
        [4.1, 3.8],
        [-0.1, 5.1],
        [0.2, 4.9],
        [0.1, 5.2],
        [-0.2, 4.8],
    ]);
    let y_train = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);

    let x_test = arr2(&[
        [0.0, 0.0],
        [0.15, -0.05],
        [-0.15, 0.05],
        [4.05, 4.05],
        [3.9, 4.0],
        [4.2, 4.1],
        [0.0, 5.0],
        [0.15, 4.95],
        [-0.15, 5.05],
    ]);
    let y_test = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);

    let mut lda = LDA::new(2, Some(Solver::SVD), Some(Shrinkage::Manual(0.1)))?;
    lda.fit(&x_train.view(), &y_train.view())?;

    let predictions = lda.predict(&x_test.view())?;
    let accuracy = classification_accuracy(&predictions, &y_test);
    assert!(accuracy >= 0.9);

    let transformed = lda.transform(&x_test.view())?;
    assert_eq!(transformed.shape(), &[x_test.nrows(), 2]);

    Ok(())
}
