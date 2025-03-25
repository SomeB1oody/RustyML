use ndarray::{Array1, Array2};
use crate::machine_learning::knn::KNN;
use crate::ModelError;

// Test default initialization of KNN
#[test]
fn test_knn_default() {
    let knn: KNN<i32> = KNN::default();
    assert_eq!(knn.get_k(), 5); // Default K value should be 5
    assert_eq!(knn.get_weights(), "uniform"); // Default weight strategy should be uniform
    assert_eq!(knn.get_metric(), "euclidean"); // Default metric should be euclidean
    assert!(matches!(knn.get_x_train(), Err(ModelError::NotFitted))); // Should not have training data by default
    assert!(matches!(knn.get_y_train(), Err(ModelError::NotFitted))); // Should not have training labels by default
}

// Test custom initialization of KNN
#[test]
fn test_knn_new() {
    let knn: KNN<i32> = KNN::new(3, "distance", "manhattan");
    assert_eq!(knn.get_k(), 3);
    assert_eq!(knn.get_weights(), "distance");
    assert_eq!(knn.get_metric(), "manhattan");
}

// Test fit method of KNN
#[test]
fn test_knn_fit() {
    let mut knn: KNN<i32> = KNN::default();

    // Create simple training data
    let x_train = Array2::<f64>::from_shape_vec((4, 2), vec![
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0,
        4.0, 5.0,
    ]).unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 1, 1]);

    knn.fit(x_train.clone(), y_train.clone());

    // Verify training data is stored
    assert!(matches!(knn.get_x_train(), Ok(_)));
    assert!(matches!(knn.get_y_train(), Ok(_)));

    // Verify training data content is correct
    let stored_x = knn.get_x_train().unwrap();
    let stored_y = knn.get_y_train().unwrap();

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
    let mut knn: KNN<i32> = KNN::new(1, "uniform", "euclidean");

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec((4, 2), vec![
        1.0, 1.0,  // Class 0
        2.0, 2.0,  // Class 0
        5.0, 5.0,  // Class 1
        6.0, 6.0,  // Class 1
    ]).unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 1, 1]);

    knn.fit(x_train, y_train);

    // Test data: should be classified as class 0
    let x_test1 = Array2::<f64>::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();
    let predictions1 = knn.predict(x_test1.view());
    assert_eq!(predictions1, vec![0]);

    // Test data: should be classified as class 1
    let x_test2 = Array2::<f64>::from_shape_vec((1, 2), vec![5.5, 5.5]).unwrap();
    let predictions2 = knn.predict(x_test2.view());
    assert_eq!(predictions2, vec![1]);
}

// Test predict method with manhattan distance
#[test]
fn test_knn_predict_manhattan() {
    let mut knn: KNN<i32> = KNN::new(1, "uniform", "manhattan");

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec((4, 2), vec![
        1.0, 1.0,  // Class 0
        2.0, 2.0,  // Class 0
        5.0, 5.0,  // Class 1
        6.0, 6.0,  // Class 1
    ]).unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 1, 1]);

    knn.fit(x_train, y_train);

    // Test data: should still be classified as class 0 with manhattan distance
    let x_test1 = Array2::<f64>::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();
    let predictions1 = knn.predict(x_test1.view());
    assert_eq!(predictions1, vec![0]);
}

// Test KNN with k=3
#[test]
fn test_knn_with_k3() {
    let mut knn: KNN<i32> = KNN::new(3, "uniform", "euclidean");

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec((5, 2), vec![
        1.0, 1.0,  // Class 0
        1.5, 1.5,  // Class 0
        2.0, 2.0,  // Class 0
        5.0, 5.0,  // Class 1
        6.0, 6.0,  // Class 1
    ]).unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 0, 1, 1]);

    knn.fit(x_train, y_train);

    // With k=3, this point has 2 nearest neighbors of class 0 and 1 of class 1
    // So it should predict class 0
    let x_test = Array2::<f64>::from_shape_vec((1, 2), vec![3.0, 3.0]).unwrap();
    let predictions = knn.predict(x_test.view());
    assert_eq!(predictions, vec![0]);
}

// Test KNN with distance weights
#[test]
fn test_knn_distance_weights() {
    let mut knn: KNN<i32> = KNN::new(3, "distance", "euclidean");

    // Training data: 2D points
    let x_train = Array2::<f64>::from_shape_vec((6, 2), vec![
        1.0, 1.0,  // Class 0
        8.0, 8.0,  // Class 0 (far from test point)
        9.0, 9.0,  // Class 0 (far from test point)
        3.9, 3.9,  // Class 1 (close to test point)
        4.1, 4.1,  // Class 1 (close to test point)
        5.0, 5.0,  // Class 1 (relatively close to test point)
    ]).unwrap();

    let y_train = Array1::<i32>::from_vec(vec![0, 0, 0, 1, 1, 1]);

    knn.fit(x_train, y_train);

    // With distance weights, this point should be predicted as class 1
    // because the nearest neighbors are of class 1
    let x_test = Array2::<f64>::from_shape_vec((1, 2), vec![4.0, 4.0]).unwrap();
    let predictions = knn.predict(x_test.view());
    assert_eq!(predictions, vec![1]);
}

// Test case with empty training set
#[test]
#[should_panic]
fn test_knn_empty_train() {
    let knn: KNN<i32> = KNN::default();

    // Don't call fit method

    // This should panic because there's no training data
    let x_test = Array2::<f64>::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
    knn.predict(x_test.view());
}

// Test with string labels instead of integers
#[test]
fn test_knn_string_labels() {
    let mut knn: KNN<String> = KNN::new(2, "uniform", "euclidean");

    // Create simple training data
    let x_train = Array2::<f64>::from_shape_vec((4, 2), vec![
        1.0, 1.0,
        2.0, 2.0,
        5.0, 5.0,
        6.0, 6.0,
    ]).unwrap();

    let y_train = Array1::<String>::from_vec(vec![
        "cat".to_string(),
        "cat".to_string(),
        "dog".to_string(),
        "dog".to_string()
    ]);

    knn.fit(x_train, y_train);

    // Test predictions
    let x_test = Array2::<f64>::from_shape_vec((2, 2), vec![
        1.5, 1.5,  // Should be "cat"
        5.5, 5.5,  // Should be "dog"
    ]).unwrap();

    let predictions = knn.predict(x_test.view());
    assert_eq!(predictions, vec!["cat".to_string(), "dog".to_string()]);
}