use crate::error::ModelError;
use crate::utility::kernel_pca::*;
use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use std::error::Error;

fn make_kernel_pca_dataset() -> Array2<f64> {
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

fn make_concentric_circle_dataset(
    inner_radius: f64,
    outer_radius: f64,
    n_angles: usize,
) -> (Array2<f64>, Vec<usize>) {
    let mut data = Vec::with_capacity(n_angles * 2 * 2);
    let mut labels = Vec::with_capacity(n_angles * 2);

    for k in 0..n_angles {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (n_angles as f64);
        data.push(inner_radius * angle.cos());
        data.push(inner_radius * angle.sin());
        labels.push(0);
    }

    for k in 0..n_angles {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (n_angles as f64);
        data.push(outer_radius * angle.cos());
        data.push(outer_radius * angle.sin());
        labels.push(1);
    }

    let dataset = Array2::from_shape_vec((n_angles * 2, 2), data).expect("Failed to build dataset");
    (dataset, labels)
}

fn centroid_distance(transformed: &Array2<f64>, labels: &[usize]) -> f64 {
    let n_components = transformed.ncols();
    let mut sum0 = vec![0.0; n_components];
    let mut sum1 = vec![0.0; n_components];
    let mut count0 = 0usize;
    let mut count1 = 0usize;

    for (idx, &label) in labels.iter().enumerate() {
        if label == 0 {
            count0 += 1;
            for j in 0..n_components {
                sum0[j] += transformed[[idx, j]];
            }
        } else {
            count1 += 1;
            for j in 0..n_components {
                sum1[j] += transformed[[idx, j]];
            }
        }
    }

    if count0 == 0 || count1 == 0 {
        return 0.0;
    }

    let mut dist_sq = 0.0;
    for j in 0..n_components {
        let mean0 = sum0[j] / count0 as f64;
        let mean1 = sum1[j] / count1 as f64;
        let diff = mean0 - mean1;
        dist_sq += diff * diff;
    }

    dist_sq.sqrt()
}

#[test]
fn test_kernel_pca_default_and_new() {
    let kpca = KernelPCA::default();
    assert_eq!(kpca.get_n_components(), 2);
    assert_eq!(kpca.get_eigen_solver(), EigenSolver::Dense);
    match kpca.get_kernel() {
        KernelType::RBF { gamma } => assert_abs_diff_eq!(gamma, 0.1, epsilon = 1e-12),
        _ => panic!("Expected RBF kernel"),
    }
    assert!(kpca.get_eigenvalues().is_none());
    assert!(kpca.get_eigenvectors().is_none());
    assert!(kpca.get_kernel_row_means().is_none());
    assert!(kpca.get_kernel_all_mean().is_none());
    assert!(kpca.get_n_samples().is_none());
    assert!(kpca.get_n_features().is_none());

    let custom = KernelPCA::new(
        KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 1.0,
        },
        3,
        EigenSolver::ARPACK,
    )
    .unwrap();
    assert_eq!(custom.get_n_components(), 3);
    assert_eq!(custom.get_eigen_solver(), EigenSolver::ARPACK);
    match custom.get_kernel() {
        KernelType::Poly {
            degree,
            gamma,
            coef0,
        } => {
            assert_eq!(degree, 2);
            assert_abs_diff_eq!(gamma, 1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(coef0, 1.0, epsilon = 1e-12);
        }
        _ => panic!("Expected Poly kernel"),
    }

    let cosine = KernelPCA::new(KernelType::Cosine, 2, EigenSolver::Dense).unwrap();
    match cosine.get_kernel() {
        KernelType::Cosine => (),
        _ => panic!("Expected Cosine kernel"),
    }
}

#[test]
fn test_kernel_pca_new_validation() {
    assert!(matches!(
        KernelPCA::new(KernelType::Linear, 0, EigenSolver::Dense),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        KernelPCA::new(KernelType::RBF { gamma: 0.0 }, 2, EigenSolver::Dense),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        KernelPCA::new(
            KernelType::Poly {
                degree: 0,
                gamma: 1.0,
                coef0: 0.0
            },
            2,
            EigenSolver::Dense
        ),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        KernelPCA::new(
            KernelType::Poly {
                degree: 2,
                gamma: -1.0,
                coef0: 0.0
            },
            2,
            EigenSolver::Dense
        ),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        KernelPCA::new(
            KernelType::Sigmoid {
                gamma: f64::NAN,
                coef0: 0.0
            },
            2,
            EigenSolver::Dense
        ),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        KernelPCA::new(
            KernelType::Sigmoid {
                gamma: 1.0,
                coef0: f64::INFINITY
            },
            2,
            EigenSolver::Dense
        ),
        Err(ModelError::InputValidationError(_))
    ));
}

#[test]
fn test_kernel_pca_fit_and_transform() -> Result<(), Box<dyn Error>> {
    let data = make_kernel_pca_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense)?;

    kpca.fit(&data.view())?;

    assert_eq!(kpca.get_n_samples(), Some(data.nrows()));
    assert_eq!(kpca.get_n_features(), Some(data.ncols()));
    assert!(kpca.get_eigenvalues().is_some());
    assert!(kpca.get_eigenvectors().is_some());
    assert_eq!(kpca.get_eigenvalues().unwrap().len(), 2);
    assert_eq!(kpca.get_eigenvectors().unwrap().shape(), &[data.nrows(), 2]);
    assert!(kpca.get_kernel_row_means().is_some());
    assert_eq!(kpca.get_kernel_row_means().unwrap().len(), data.nrows());
    assert!(kpca.get_kernel_all_mean().is_some());
    assert!(
        kpca.get_eigenvalues()
            .unwrap()
            .iter()
            .all(|v| *v > 0.0 && v.is_finite())
    );

    let transformed = kpca.transform(&data.view())?;
    assert_eq!(transformed.shape(), &[data.nrows(), 2]);
    assert!(transformed.iter().all(|v| v.is_finite()));

    let new_data = arr2(&[[1.0, 1.5, 2.5], [2.0, 2.5, 3.5]]);
    let new_transformed = kpca.transform(&new_data.view())?;
    assert_eq!(new_transformed.shape(), &[2, 2]);
    assert!(new_transformed.iter().all(|v| v.is_finite()));

    Ok(())
}

#[test]
fn test_kernel_pca_cosine_kernel() -> Result<(), Box<dyn Error>> {
    let data = make_kernel_pca_dataset();
    let mut kpca = KernelPCA::new(KernelType::Cosine, 2, EigenSolver::Dense)?;

    let transformed = kpca.fit_transform(&data.view())?;
    assert_eq!(transformed.shape(), &[data.nrows(), 2]);
    assert!(transformed.iter().all(|v| v.is_finite()));

    Ok(())
}

#[test]
fn test_kernel_pca_fit_transform_consistency() -> Result<(), Box<dyn Error>> {
    let data = make_kernel_pca_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.8 }, 2, EigenSolver::Dense)?;

    let transformed_fit = kpca.fit_transform(&data.view())?;
    let transformed_again = kpca.transform(&data.view())?;

    assert_eq!(transformed_fit.shape(), transformed_again.shape());
    for (a, b) in transformed_fit.iter().zip(transformed_again.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }

    Ok(())
}

#[test]
fn test_kernel_pca_solver_variants() -> Result<(), Box<dyn Error>> {
    let data = make_kernel_pca_dataset();
    let solvers = [EigenSolver::Dense, EigenSolver::ARPACK];

    for solver in solvers {
        let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.6 }, 2, solver)?;
        kpca.fit(&data.view())?;
        let transformed = kpca.transform(&data.view())?;
        assert_eq!(transformed.shape(), &[data.nrows(), 2]);
        assert!(kpca.get_eigenvectors().is_some());
    }

    Ok(())
}

#[test]
fn test_kernel_pca_validation_and_errors() {
    let data = make_kernel_pca_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();

    assert!(matches!(
        kpca.transform(&data.view()),
        Err(ModelError::NotFitted)
    ));

    let empty = Array2::<f64>::zeros((0, 3));
    assert!(matches!(
        kpca.fit(&empty.view()),
        Err(ModelError::InputValidationError(_))
    ));

    let single = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    assert!(matches!(
        kpca.fit(&single.view()),
        Err(ModelError::InputValidationError(_))
    ));

    let mut kpca_bad = KernelPCA::new(
        KernelType::RBF { gamma: 0.5 },
        data.nrows() + 1,
        EigenSolver::Dense,
    )
    .unwrap();
    assert!(matches!(
        kpca_bad.fit(&data.view()),
        Err(ModelError::InputValidationError(_))
    ));

    let mut data_with_nan = data.clone();
    data_with_nan[[0, 1]] = f64::NAN;
    assert!(matches!(
        kpca.fit(&data_with_nan.view()),
        Err(ModelError::InputValidationError(_))
    ));

    let mut kpca_fit =
        KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    kpca_fit.fit(&data.view()).unwrap();

    let wrong_features = Array2::<f64>::zeros((2, 4));
    assert!(matches!(
        kpca_fit.transform(&wrong_features.view()),
        Err(ModelError::InputValidationError(_))
    ));
}

#[test]
fn test_kernel_pca_effectiveness_on_nonlinear_data() -> Result<(), Box<dyn Error>> {
    let (data, labels) = make_concentric_circle_dataset(1.0, 3.0, 16);

    let mut kpca_rbf = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense)?;
    let transformed_rbf = kpca_rbf.fit_transform(&data.view())?;
    let rbf_distance = centroid_distance(&transformed_rbf, &labels);

    let mut kpca_linear = KernelPCA::new(KernelType::Linear, 2, EigenSolver::Dense)?;
    let transformed_linear = kpca_linear.fit_transform(&data.view())?;
    let linear_distance = centroid_distance(&transformed_linear, &labels);

    assert!(rbf_distance > linear_distance + 1e-2);
    assert!(rbf_distance > 0.1);

    Ok(())
}
