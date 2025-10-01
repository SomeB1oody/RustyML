use super::*;

// Test default initialization of KNN
#[test]
fn test_knn_default() {
    let knn: KNN<i32> = KNN::default();
    assert_eq!(knn.get_k(), 5); // Default K value should be 5
    assert!(matches!(
        knn.get_weighting_strategy(),
        WeightingStrategy::Uniform
    )); // Default weight strategy should be uniform
    assert!(matches!(
        knn.get_metric(),
        DistanceCalculationMetric::Euclidean
    )); // Default metric should be Euclidean
    assert!(matches!(knn.get_x_train(), None)); // Should not have training data by default
    assert!(matches!(knn.get_y_train(), None)); // Should not have training labels by default
}

// Test custom initialization of KNN
#[test]
fn test_knn_new() {
    let knn: KNN<i32> = KNN::new(
        3,
        WeightingStrategy::Distance,
        DistanceCalculationMetric::Manhattan,
    );
    assert_eq!(knn.get_k(), 3);
    assert!(matches!(
        knn.get_weighting_strategy(),
        WeightingStrategy::Distance
    ));
    assert!(matches!(
        knn.get_metric(),
        DistanceCalculationMetric::Manhattan
    ));
}

// Test fit method of KNN
#[test]
fn test_knn_fit() {
    let mut knn: KNN<i32> = KNN::default();

    // Create simple training data
    let x_train = Array2::<f64>::from_shape_vec(
        (7, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0,
        ],
    )
    .unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 1, 1, 1, 1, 1]);

    knn.fit(x_train.view(), y_train.view()).unwrap();

    // Verify training data is stored
    assert!(matches!(knn.get_x_train(), Some(_)));
    assert!(matches!(knn.get_y_train(), Some(_)));

    // Verify training data content is correct
    let stored_x = match knn.get_x_train() {
        Some(x) => x,
        None => panic!("Training data should be available after fitting"),
    };
    let stored_y = match knn.get_y_train() {
        Some(y) => y,
        None => panic!("Training labels should be available after fitting"),
    };

    assert_eq!(stored_x.shape(), x_train.shape());
    assert_eq!(stored_y.len(), y_train.len());

    // Compare elements one by one
    for i in 0..x_train.nrows() {
        for j in 0..x_train.ncols() {
            assert_eq!(stored_x[[i, j]], x_train[[i, j]]);
        }
        assert_eq!(stored_y[i], y_train[i]);
    }
}

// Test predict method with euclidean distance and uniform weights
#[test]
fn test_knn_predict_euclidean_uniform() {
    let mut knn: KNN<i32> = KNN::new(
        1,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Euclidean,
    );

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec(
        (4, 2),
        vec![
            1.0, 1.0, // Class 0
            2.0, 2.0, // Class 0
            5.0, 5.0, // Class 1
            6.0, 6.0, // Class 1
        ],
    )
    .unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 1, 1]);

    knn.fit(x_train.view(), y_train.view()).unwrap();

    // Test data: should be classified as class 0
    let x_test1 = Array2::<f64>::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();
    let predictions1 = knn.predict(x_test1.view()).unwrap();
    assert_eq!(predictions1, array![0]);

    // Test data: should be classified as class 1
    let x_test2 = Array2::<f64>::from_shape_vec((1, 2), vec![5.5, 5.5]).unwrap();
    let predictions2 = knn.predict(x_test2.view()).unwrap();
    assert_eq!(predictions2, array![1]);
}

// Test predict method with manhattan distance
#[test]
fn test_knn_predict_manhattan() {
    let mut knn: KNN<i32> = KNN::new(
        1,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Manhattan,
    );

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec(
        (4, 2),
        vec![
            1.0, 1.0, // Class 0
            2.0, 2.0, // Class 0
            5.0, 5.0, // Class 1
            6.0, 6.0, // Class 1
        ],
    )
    .unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 1, 1]);

    knn.fit(x_train.view(), y_train.view()).unwrap();

    // Test data: should still be classified as class 0 with manhattan distance
    let x_test1 = Array2::<f64>::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();
    let predictions1 = knn.predict(x_test1.view()).unwrap();
    assert_eq!(predictions1, array![0]);
}

// Test KNN with k=3
#[test]
fn test_knn_with_k3() {
    let mut knn: KNN<i32> = KNN::new(
        3,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Euclidean,
    );

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec(
        (5, 2),
        vec![
            1.0, 1.0, // Class 0
            1.5, 1.5, // Class 0
            2.0, 2.0, // Class 0
            5.0, 5.0, // Class 1
            6.0, 6.0, // Class 1
        ],
    )
    .unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 0, 1, 1]);

    knn.fit(x_train.view(), y_train.view()).unwrap();

    // With k=3, this point has 2 nearest neighbors of class 0 and 1 of class 1
    // So it should predict class 0
    let x_test = Array2::<f64>::from_shape_vec((1, 2), vec![3.0, 3.0]).unwrap();
    let predictions = knn.predict(x_test.view()).unwrap();
    assert_eq!(predictions, array![0]);
}

// Test KNN with distance weights
#[test]
fn test_knn_distance_weights() {
    let mut knn: KNN<i32> = KNN::new(
        3,
        WeightingStrategy::Distance,
        DistanceCalculationMetric::Euclidean,
    );

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, // Class 0
            8.0, 8.0, // Class 0 (far from test point)
            9.0, 9.0, // Class 0 (far from test point)
            3.9, 3.9, // Class 1 (close to test point)
            4.1, 4.1, // Class 1 (close to test point)
            5.0, 5.0, // Class 1 (relatively close to test point)
        ],
    )
    .unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 0, 1, 1, 1]);

    knn.fit(x_train.view(), y_train.view()).unwrap();

    // With distance weights, this point should be predicted as class 1
    // because the nearest neighbors are of class 1
    let x_test = Array2::<f64>::from_shape_vec((1, 2), vec![4.0, 4.0]).unwrap();
    let predictions = knn.predict(x_test.view()).unwrap();
    assert_eq!(predictions, array![1]);
}

// Test case with empty training set
#[test]
fn test_knn_empty_train() {
    let knn: KNN<i32> = KNN::default();

    let x_test = Array2::<f64>::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
    assert!(matches!(
        knn.predict(x_test.view()),
        Err(ModelError::NotFitted)
    ));
}

// Test with string labels instead of integers
#[test]
fn test_knn_string_labels() {
    let mut knn: KNN<String> = KNN::new(
        2,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Euclidean,
    );

    // Create simple training data
    let x_train =
        Array2::<f64>::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0])
            .unwrap();

    let y_train = Array1::<String>::from_vec(vec![
        "cat".to_string(),
        "cat".to_string(),
        "dog".to_string(),
        "dog".to_string(),
    ]);

    knn.fit(x_train.view(), y_train.view()).unwrap();

    // Test predictions
    let x_test = Array2::<f64>::from_shape_vec(
        (2, 2),
        vec![
            1.5, 1.5, // Should be "cat"
            5.5, 5.5, // Should be "dog"
        ],
    )
    .unwrap();

    let predictions = knn.predict(x_test.view()).unwrap();
    assert_eq!(predictions, array!["cat".to_string(), "dog".to_string()]);
}

#[test]
fn test_fit_predict() {
    // Create a new KNN model with k=3, uniform weights and euclidean metric
    let mut knn = KNN::<i32>::new(
        3,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Euclidean,
    );

    // Create training data
    // Features: 2D points
    let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0]];

    // Labels: classes 0 and 1
    let y_train = array![0, 0, 0, 1, 1];

    // Test data
    let x_test = array![
        [1.5, 2.5], // Should be classified as 0 (closer to first 3 points)
        [5.5, 6.5]  // Should be classified as 1 (closer to last 2 points)
    ];

    // Use fit_predict to get predictions
    let predictions = knn
        .fit_predict(x_train.view(), y_train.view(), x_test.view())
        .unwrap();

    // Verify predictions
    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0], 0);
    assert_eq!(predictions[1], 1);

    // Verify the model has been fitted
    assert!(knn.get_x_train().is_some());
    assert!(knn.get_y_train().is_some());
}

#[test]
fn test_fit_predict_empty_data() {
    // Test with empty test data
    let mut knn = KNN::<i32>::new(
        3,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Euclidean,
    );

    let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y_train = array![0, 0, 1];

    // Empty test data
    let x_test = Array2::<f64>::zeros((0, 2));

    // Should return empty predictions vector
    assert_eq!(
        knn.fit_predict(x_train.view(), y_train.view(), x_test.view()),
        Err(ModelError::InputValidationError(
            "Input data is empty".to_string()
        ))
    );
}
