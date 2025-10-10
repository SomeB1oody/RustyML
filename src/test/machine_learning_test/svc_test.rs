use super::*;
#[test]
fn test_svc_constructor() {
    // Test basic constructor
    let svc = SVC::new(KernelType::Linear, 1.0, 0.001, 100);

    assert_eq!(svc.get_regularization_parameter(), 1.0);
    assert_eq!(svc.get_tolerance(), 0.001);
    assert_eq!(svc.get_max_iterations(), 100);

    // Match kernel type
    match svc.get_kernel() {
        KernelType::Linear => (), // Should be linear kernel
        _ => panic!("Expected linear kernel"),
    }
}

#[test]
fn test_svc_default() {
    let svc = SVC::default();

    // Check default values
    match svc.get_kernel() {
        KernelType::RBF { gamma } => {
            assert!((gamma - 0.1).abs() < 1e-10);
        }
        _ => panic!("Expected RBF kernel with gamma=1.0"),
    }
    assert_eq!(svc.get_regularization_parameter(), 1.0);
    assert_eq!(svc.get_tolerance(), 0.001);
    assert_eq!(svc.get_max_iterations(), 1000);
    assert_eq!(svc.get_epsilon(), 1e-8);
}

#[test]
fn test_getters_before_fit() {
    // Test that getters return errors before model is fitted
    let svc = SVC::default();

    assert!(svc.get_alphas().is_none());
    assert!(svc.get_support_vectors().is_none());
    assert!(svc.get_support_vector_labels().is_none());
    assert!(svc.get_bias().is_none());
}

#[test]
fn test_fit_and_predict_linear() {
    // Create a simple linearly separable dataset
    let x = arr2(&[
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [2.0, 1.0],
        [-1.0, -2.0],
        [-2.0, -3.0],
        [-3.0, -2.0],
        [-2.0, -1.0],
    ]);

    let y = arr1(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);

    let mut svc = SVC::new(KernelType::Linear, 10.0, 0.001, 10000);

    // Train the model
    let fit_result = svc.fit(x.view(), y.view());
    assert!(fit_result.is_ok());

    // Check that model parameters are available after training
    assert!(svc.get_alphas().is_some());
    assert!(svc.get_support_vectors().is_some());
    assert!(svc.get_support_vector_labels().is_some());
    assert!(svc.get_bias().is_some());

    let predictions = svc.predict(x.view()).unwrap();

    let mut correct_count = 0;

    for (pre_val, act_val) in predictions.iter().zip(y.iter()) {
        if pre_val == act_val {
            correct_count += 1;
        }
    }

    assert!(correct_count >= 4);
}

#[test]
fn test_fit_and_predict_rbf() {
    // Create a non-linear dataset (XOR-like problem)
    let x = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [1.5, 1.5],
        [-0.5, -0.5],
        [-1.0, -1.0],
    ]);

    // XOR-like problem: (0,0)->-1, (0,1)->1, (1,0)->1, (1,1)->-1
    let y = arr1(&[-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);

    let mut svc = SVC::new(KernelType::RBF { gamma: 10.0 }, 10.0, 0.001, 10000);

    // Train the model
    let fit_result = svc.fit(x.view(), y.view());
    assert!(fit_result.is_ok());

    let predictions = svc.predict(x.view()).unwrap();

    let mut correct_count = 0;

    for (pre_val, act_val) in predictions.iter().zip(y.iter()) {
        if pre_val == act_val {
            correct_count += 1;
        }
    }

    assert!(correct_count >= 5);
}

#[test]
fn test_error_handling() {
    // Test error handling for various cases
    let mut svc = SVC::default();

    // Test mismatched dimensions input
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let y = arr1(&[1.0, -1.0, 1.0]); // Dimension mismatch

    let result = svc.fit(x.view(), y.view());
    assert!(result.is_err());

    // Test prediction error before training
    let test_x = arr2(&[[1.0, 2.0]]);
    let predict_result = svc.predict(test_x.view());
    assert!(predict_result.is_err());

    // Test decision function error before training
    let decision_result = svc.decision_function(test_x.view());
    assert!(decision_result.is_err());
}

#[test]
fn test_different_kernels() {
    // Create various kernel types
    let poly_svc = SVC::new(
        KernelType::Poly {
            degree: 3,
            gamma: 0.1,
            coef0: 0.0,
        },
        1.0,
        0.001,
        100,
    );

    let sigmoid_svc = SVC::new(
        KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        },
        1.0,
        0.001,
        100,
    );

    // Ensure kernel types are correctly matched
    match poly_svc.get_kernel() {
        KernelType::Poly {
            degree,
            gamma,
            coef0,
        } => {
            assert_eq!(degree, 3);
            assert!((gamma - 0.1).abs() < 1e-10);
            assert!((coef0 - 0.0).abs() < 1e-10);
        }
        _ => panic!("Expected polynomial kernel"),
    }

    match sigmoid_svc.get_kernel() {
        KernelType::Sigmoid { gamma, coef0 } => {
            assert!((gamma - 0.1).abs() < 1e-10);
            assert!((coef0 - 0.0).abs() < 1e-10);
        }
        _ => panic!("Expected sigmoid kernel"),
    }
}

#[test]
fn test_decision_function() {
    // Test decision function after fit
    let x = arr2(&[[1.0, 2.0], [2.0, 3.0], [-1.0, -2.0], [-2.0, -3.0]]);

    let y = arr1(&[1.0, 1.0, -1.0, -1.0]);

    let mut svc = SVC::new(KernelType::Linear, 1.0, 0.001, 100);
    svc.fit(x.view(), y.view()).unwrap();

    // Get decision scores
    let test_point = arr2(&[[0.0, 0.0]]);
    let decisions = svc.decision_function(test_point.view()).unwrap();

    // Decision function should return a single score for the test point
    assert_eq!(decisions.len(), 1);
}
