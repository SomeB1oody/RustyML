use crate::ModelError;
use crate::utility::KernelType;
use crate::utility::kernel_pca::{KernelPCA, compute_kernel};
use approx::assert_abs_diff_eq;
use ndarray::{Array2, ArrayView1};

#[test]
fn test_kernel_pca_default() {
    let kpca = KernelPCA::default();
    assert!(matches!(kpca.get_kernel(), KernelType::Linear));
    assert_eq!(kpca.get_n_components(), 2);
    assert!(kpca.get_eigenvalues().is_err());
    assert!(kpca.get_eigenvectors().is_err());
    assert!(kpca.get_x_fit().is_err());
    assert!(kpca.get_row_means().is_err());
    assert!(kpca.get_total_mean().is_err());
}

#[test]
fn test_kernel_pca_new() {
    let kernel = KernelType::RBF { gamma: 0.1 };
    let n_components = 3;
    let kpca = KernelPCA::new(kernel.clone(), n_components);

    assert!(matches!(kpca.get_kernel(), KernelType::RBF { gamma: 0.1 }));
    assert_eq!(kpca.get_n_components(), 3);
    assert!(kpca.get_eigenvalues().is_err());
    assert!(kpca.get_eigenvectors().is_err());
    assert!(kpca.get_x_fit().is_err());
    assert!(kpca.get_row_means().is_err());
    assert!(kpca.get_total_mean().is_err());
}

#[test]
fn test_compute_kernel_linear() {
    let x = ArrayView1::from(&[1.0, 2.0, 3.0]);
    let y = ArrayView1::from(&[4.0, 5.0, 6.0]);
    let kernel = KernelType::Linear;

    let result = compute_kernel(&x, &y, &kernel);
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_compute_kernel_rbf() {
    let x = ArrayView1::from(&[1.0, 2.0, 3.0]);
    let y = ArrayView1::from(&[1.0, 2.0, 3.0]);
    let kernel = KernelType::RBF { gamma: 1.0 };

    let result = compute_kernel(&x, &y, &kernel);
    assert_eq!(result, 1.0); // Same vectors should have kernel value 1.0
}

#[test]
fn test_compute_kernel_poly() {
    let x = ArrayView1::from(&[1.0, 2.0]);
    let y = ArrayView1::from(&[3.0, 4.0]);
    let kernel = KernelType::Poly {
        degree: 2,
        gamma: 1.0,
        coef0: 0.0,
    };

    let result = compute_kernel(&x, &y, &kernel);
    assert_eq!(result, 121.0); // (1*3 + 2*4)^2 = 11^2 = 121
}

#[test]
fn test_compute_kernel_sigmoid() {
    let x = ArrayView1::from(&[1.0, 2.0]);
    let y = ArrayView1::from(&[3.0, 4.0]);
    let kernel = KernelType::Sigmoid {
        gamma: 1.0,
        coef0: 0.0,
    };

    let result = compute_kernel(&x, &y, &kernel);
    assert_abs_diff_eq!(result, 11.0_f64.tanh(), epsilon = 1e-10);
}

#[test]
fn test_getters_not_fitted() {
    let kpca = KernelPCA::default();

    assert!(matches!(kpca.get_eigenvalues(), Err(ModelError::NotFitted)));
    assert!(matches!(
        kpca.get_eigenvectors(),
        Err(ModelError::NotFitted)
    ));
    assert!(matches!(kpca.get_x_fit(), Err(ModelError::NotFitted)));
    assert!(matches!(kpca.get_row_means(), Err(ModelError::NotFitted)));
    assert!(matches!(kpca.get_total_mean(), Err(ModelError::NotFitted)));
}

#[test]
fn test_getter_methods() {
    let kernel = KernelType::RBF { gamma: 0.1 };
    let n_components = 2;
    let kpca = KernelPCA::new(kernel.clone(), n_components);

    assert!(matches!(*kpca.get_kernel(), KernelType::RBF { gamma: 0.1 }));
    assert_eq!(kpca.get_n_components(), 2);
}

#[test]
fn test_fit_invalid_inputs() {
    let mut kpca = KernelPCA::default();

    // Test empty input
    let empty = Array2::<f64>::zeros((0, 5));
    assert!(kpca.fit(empty.view()).is_err());

    // Test case where n_components is 0
    let mut kpca_zero = KernelPCA::new(KernelType::Linear, 0);
    let data = Array2::<f64>::zeros((10, 3));
    assert!(kpca_zero.fit(data.view()).is_err());

    // Test case where sample count is less than n_components
    let mut kpca_large = KernelPCA::new(KernelType::Linear, 5);
    let small_data = Array2::<f64>::zeros((3, 3));
    assert!(kpca_large.fit(small_data.view()).is_err());
}

#[test]
fn test_fit_simple_case() {
    let mut kpca = KernelPCA::new(KernelType::Linear, 2);
    let data = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let result = kpca.fit(data.view());
    assert!(result.is_ok());

    // Verify that the model is correctly fitted
    assert!(kpca.get_eigenvalues().is_ok());
    assert!(kpca.get_eigenvectors().is_ok());
    assert!(kpca.get_x_fit().is_ok());
    assert!(kpca.get_row_means().is_ok());
    assert!(kpca.get_total_mean().is_ok());

    // Verify the number of eigenvalues
    assert_eq!(kpca.get_eigenvalues().unwrap().len(), 2);
}

#[test]
fn test_transform_not_fitted() {
    let kpca = KernelPCA::default();
    let data = Array2::<f64>::zeros((5, 3));

    assert!(kpca.transform(data.view()).is_err());
}

#[test]
fn test_fit_transform() {
    let mut kpca = KernelPCA::new(KernelType::Linear, 2);
    let data = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let result = kpca.fit_transform(data.view());
    assert!(result.is_ok());

    let transformed = result.unwrap();
    // Verify the shape of transformed data
    assert_eq!(transformed.shape(), &[4, 2]);
}

#[test]
fn test_fit_and_transform() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.1 }, 2);
    let train_data = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    // First fit
    let fit_result = kpca.fit(train_data.view());
    assert!(fit_result.is_ok());

    // Then transform new data
    let test_data = Array2::from_shape_vec((2, 3), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

    let transform_result = kpca.transform(test_data.view());
    assert!(transform_result.is_ok());

    let transformed = transform_result.unwrap();
    // Verify the shape of transformed data
    assert_eq!(transformed.shape(), &[2, 2]);
}

#[test]
fn test_different_kernel_types() {
    // Test different kernel function types
    let kernels = vec![
        KernelType::Linear,
        KernelType::RBF { gamma: 0.1 },
        KernelType::Poly {
            degree: 2,
            gamma: 0.1,
            coef0: 1.0,
        },
        KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 1.0,
        },
    ];

    let data = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    for kernel in kernels {
        let mut kpca = KernelPCA::new(kernel, 2);
        let result = kpca.fit_transform(data.view());
        assert!(result.is_ok());
    }
}
