use ndarray::{Array1, Array2};
use rustyml::machine_learning::linear_regression::LinearRegression;

#[test]
fn test_default_constructor() {
    let model = LinearRegression::default();

    assert!(matches!(model.get_coefficients(), None));
    assert!(matches!(model.get_intercept(), None));
    assert!(matches!(model.get_actual_iterations(), None));
    // Check if default values meet expectations
    assert_eq!(model.get_fit_intercept(), true); // Assuming default is true
    assert!(model.get_learning_rate() > 0.0);
    assert!(model.get_max_iterations() > 0);
    assert!(model.get_tolerance() > 0.0);
}

#[test]
fn test_new_constructor() {
    let model = LinearRegression::new(false, 0.01, 1000, 1e-5, None).unwrap();
    assert!(!model.get_fit_intercept());
    assert_eq!(model.get_learning_rate(), 0.01);
    assert_eq!(model.get_max_iterations(), 1000);
    assert_eq!(model.get_tolerance(), 1e-5);
    assert!(matches!(model.get_actual_iterations(), None));
}

#[test]
fn test_not_fitted_error() {
    let model = LinearRegression::default();

    // Try to get coefficients and intercept from an unfitted model
    let coef_result = model.get_coefficients();
    let intercept_result = model.get_intercept();

    assert!(matches!(coef_result, None));
    assert!(matches!(intercept_result, None));
}

#[test]
fn test_fit_and_predict() {
    // Create a simple dataset: y = 2*x + 1
    let x_vec = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y_vec = vec![3.0, 5.0, 7.0, 9.0];

    // 将Vec转换为ndarray的Array
    let x = Array2::from_shape_vec((4, 1), x_vec.into_iter().flatten().collect()).unwrap();
    let y = Array1::from_vec(y_vec);

    let mut model = LinearRegression::new(true, 0.01, 10000, 1e-8, None).unwrap();
    model.fit(&x.view(), &y.view()).unwrap();

    // Check if coefficients and intercept are close to expected values
    let coefficients = match model.get_coefficients() {
        Some(coefficients) => coefficients,
        None => panic!("Model coefficients are not initialized"),
    };
    let intercept = model.get_intercept().unwrap();

    assert!((coefficients[0] - 2.0).abs() < 0.1);
    assert!((intercept - 1.0).abs() < 0.1);

    // Test predictions
    let test_x_vec = vec![vec![5.0], vec![6.0]];
    let test_x =
        Array2::from_shape_vec((2, 1), test_x_vec.into_iter().flatten().collect()).unwrap();
    let predictions = model.predict(&test_x.view()).unwrap();

    assert!((predictions[0] - 11.0).abs() < 0.2);
    assert!((predictions[1] - 13.0).abs() < 0.2);
}

#[test]
fn test_multivariate_regression() {
    // Create multivariate dataset: y = 2*x1 + 3*x2 + 1
    let x_vec = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 4.0],
        vec![4.0, 3.0],
        vec![2.5, 3.5],
        vec![3.5, 2.5],
        vec![1.5, 3.0],
        vec![3.0, 1.5],
    ];

    // Calculate y values based on y = 2*x1 + 3*x2 + 1
    let mut y_vec = Vec::new();
    for sample in &x_vec {
        y_vec.push(2.0 * sample[0] + 3.0 * sample[1] + 1.0);
    }

    let x = Array2::from_shape_vec((8, 2), x_vec.into_iter().flatten().collect()).unwrap();
    let y = Array1::from_vec(y_vec);

    // Create model and train
    let mut model = LinearRegression::new(true, 0.005, 20000, 1e-10, None).unwrap();
    model.fit(&x.view(), &y.view()).unwrap();

    // Check if coefficients and intercept are close to expected values
    let coefficients = match model.get_coefficients() {
        Some(coefficients) => coefficients,
        None => panic!("Model coefficients are not initialized"),
    };
    let intercept = model.get_intercept().unwrap();

    println!("Learned coefficients: {:?}", coefficients);
    println!("Learned intercept: {}", intercept);

    assert!(
        (coefficients[0] - 2.0).abs() < 0.2,
        "First coefficient {} differs too much from expected 2.0",
        coefficients[0]
    );
    assert!(
        (coefficients[1] - 3.0).abs() < 0.2,
        "Second coefficient {} differs too much from expected 3.0",
        coefficients[1]
    );
    assert!(
        (intercept - 1.0).abs() < 0.2,
        "Intercept {} differs too much from expected 1.0",
        intercept
    );

    // Test predictions
    let test_x_vec = vec![vec![5.0, 5.0], vec![2.0, 4.0]];
    let test_x =
        Array2::from_shape_vec((2, 2), test_x_vec.into_iter().flatten().collect()).unwrap();
    let predictions = model.predict(&test_x.view()).unwrap();

    assert!(
        (predictions[0] - 26.0).abs() < 0.5,
        "Prediction {} differs too much from expected 26.0",
        predictions[0]
    );
    assert!(
        (predictions[1] - 17.0).abs() < 0.5,
        "Prediction {} differs too much from expected 17.0",
        predictions[1]
    );
}

#[test]
fn test_no_intercept() {
    // Test without intercept: y = 2*x
    let x_vec = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y_vec = vec![2.0, 4.0, 6.0, 8.0];

    let x = Array2::from_shape_vec((4, 1), x_vec.into_iter().flatten().collect()).unwrap();
    let y = Array1::from_vec(y_vec);

    let mut model = LinearRegression::new(false, 0.01, 10000, 1e-8, None).unwrap();
    model.fit(&x.view(), &y.view()).unwrap();

    // Check if coefficient is close to expected value (around 2.0)
    let coefficients = match model.get_coefficients() {
        Some(coefficients) => coefficients,
        None => panic!("Model coefficients are not initialized"),
    };

    assert!((coefficients[0] - 2.0).abs() < 0.1);

    // Intercept should be close to 0 or not available
    if let Some(intercept) = model.get_intercept() {
        assert!(intercept.abs() < 0.1);
    }
}

#[test]
fn test_linear_regression_fit_predict() {
    // Create a simple linear regression model instance
    let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();

    // Create simple training data
    let x_vec = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let y_vec = vec![5.0, 7.0, 9.0, 11.0, 13.0];

    let x = Array2::from_shape_vec((5, 1), x_vec.into_iter().flatten().collect()).unwrap();
    let y = Array1::from_vec(y_vec);

    // Use the fit_predict method
    let predictions = model.fit_predict(&x.view(), &y.view()).unwrap();

    // Verify predictions are close to actual values
    assert_eq!(predictions.len(), y.len());

    for i in 0..y.len() {
        assert!(
            (predictions[i] - y[i]).abs() < 0.5,
            "Prediction {} differs too much from actual value {}",
            predictions[i],
            y[i]
        );
    }

    // Additional check for model parameters
    let coefficients = match model.get_coefficients() {
        Some(coefficients) => coefficients,
        None => panic!("Model coefficients are not initialized"),
    };

    let intercept = model.get_intercept().unwrap();

    // Verify coefficient is close to 2 and intercept is close to 3
    assert!(
        (coefficients[0] - 2.0).abs() < 0.5,
        "Coefficient {} differs too much from expected value 2.0",
        coefficients[0]
    );
    assert!(
        (intercept - 3.0).abs() < 0.5,
        "Intercept {} differs too much from expected value 3.0",
        intercept
    );
}
