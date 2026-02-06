use ndarray::prelude::*;
use rustyml::utility::principal_component_analysis::*;
use std::error::Error;

fn make_pca_dataset() -> Array2<f64> {
    arr2(&[
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 0.5],
        [2.0, 0.5, 1.0],
        [3.0, 1.5, 2.5],
        [4.0, 3.0, 0.0],
        [5.0, 2.5, 1.5],
        [6.0, 4.0, 3.0],
        [7.0, 5.0, 2.0],
    ])
}

#[test]
fn test_pca_default_and_new() {
    let pca = PCA::default();
    assert_eq!(pca.get_n_components(), 2);
    assert_eq!(pca.get_svd_solver(), SVDSolver::Full);
    assert!(pca.get_mean().is_none());
    assert!(pca.get_components().is_none());
    assert!(pca.get_explained_variance().is_none());
    assert!(pca.get_explained_variance_ratio().is_none());
    assert!(pca.get_singular_values().is_none());

    let custom = PCA::new(3, SVDSolver::Randomized(42)).unwrap();
    assert_eq!(custom.get_n_components(), 3);
    assert_eq!(custom.get_svd_solver(), SVDSolver::Randomized(42));
}

#[test]
fn test_pca_new_validation() {
    assert!(PCA::new(0, SVDSolver::Full).is_err());
}

#[test]
fn test_pca_fit_and_transform() -> Result<(), Box<dyn Error>> {
    let data = make_pca_dataset();
    let mut pca = PCA::new(2, SVDSolver::Full)?;

    pca.fit(&data.view())?;

    assert!(pca.get_mean().is_some());
    assert!(pca.get_components().is_some());
    assert_eq!(pca.get_components().unwrap().shape(), &[2, 3]);
    assert_eq!(pca.get_explained_variance().unwrap().len(), 2);
    assert_eq!(pca.get_explained_variance_ratio().unwrap().len(), 2);

    let ratio_sum: f64 = pca.get_explained_variance_ratio().unwrap().iter().sum();
    assert!(ratio_sum >= 0.0);
    assert!(ratio_sum <= 1.00001);

    let transformed = pca.transform(&data.view())?;
    assert_eq!(transformed.shape(), &[data.nrows(), 2]);
    assert!(transformed.iter().all(|v| v.is_finite()));

    Ok(())
}

#[test]
fn test_pca_fit_transform_and_inverse() -> Result<(), Box<dyn Error>> {
    let data = make_pca_dataset();
    let mut pca = PCA::new(3, SVDSolver::Full)?;

    let transformed = pca.fit_transform(&data.view())?;
    assert_eq!(transformed.shape(), &[data.nrows(), 3]);

    let reconstructed = pca.inverse_transform(&transformed.view())?;
    assert_eq!(reconstructed.shape(), data.shape());

    let max_diff = reconstructed
        .iter()
        .zip(data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_diff < 1e-6);

    Ok(())
}

#[test]
fn test_pca_solver_variants() -> Result<(), Box<dyn Error>> {
    let data = make_pca_dataset();
    let solvers = [SVDSolver::Full, SVDSolver::Randomized(7), SVDSolver::ARPACK];

    for solver in solvers {
        let mut pca = PCA::new(2, solver)?;
        pca.fit(&data.view())?;
        let transformed = pca.transform(&data.view())?;
        assert_eq!(transformed.shape(), &[data.nrows(), 2]);
        assert!(pca.get_components().is_some());
    }

    Ok(())
}

#[test]
fn test_pca_validation_and_errors() {
    let data = make_pca_dataset();

    let mut pca = PCA::new(2, SVDSolver::Full).unwrap();
    assert!(pca.transform(&data.view()).is_err());
    assert!(pca.inverse_transform(&data.view()).is_err());

    let empty = Array2::<f64>::zeros((0, 3));
    assert!(pca.fit(&empty.view()).is_err());

    let single = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    assert!(pca.fit(&single.view()).is_err());

    let mut pca_bad = PCA::new(4, SVDSolver::Full).unwrap();
    assert!(pca_bad.fit(&data.view()).is_err());

    let data_with_nan =
        Array2::from_shape_vec((2, 3), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
    assert!(pca.fit(&data_with_nan.view()).is_err());

    let mut pca_fit = PCA::new(2, SVDSolver::Full).unwrap();
    pca_fit.fit(&data.view()).unwrap();

    let wrong_features = Array2::<f64>::zeros((2, 4));
    assert!(pca_fit.transform(&wrong_features.view()).is_err());

    let wrong_components = Array2::<f64>::zeros((2, 3));
    assert!(pca_fit.inverse_transform(&wrong_components.view()).is_err());
}

#[test]
fn test_pca_effectiveness_on_linear_data() -> Result<(), Box<dyn Error>> {
    let data = arr2(&[
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
        [6.0, 12.0],
        [7.0, 14.0],
    ]);

    let mut pca = PCA::new(1, SVDSolver::Full)?;
    let transformed = pca.fit_transform(&data.view())?;
    assert_eq!(transformed.shape(), &[data.nrows(), 1]);

    let ratio = pca.get_explained_variance_ratio().unwrap()[0];
    assert!(ratio > 0.999);

    let reconstructed = pca.inverse_transform(&transformed.view())?;
    let max_diff = reconstructed
        .iter()
        .zip(data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_diff < 1e-6);

    Ok(())
}
