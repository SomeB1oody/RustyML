use crate::ModelError;
use crate::math::*;
use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::{Array, Array1, array};

#[test]
fn test_sum_of_square_total() {
    // Standard case
    let values = array![2.0, 4.0, 6.0, 8.0];
    // Mean is 5.0, so SST = (2-5)² + (4-5)² + (6-5)² + (8-5)² = 9 + 1 + 1 + 9 = 20
    let expected = 20.0;
    assert_abs_diff_eq!(
        sum_of_square_total(values.view()).unwrap(),
        expected,
        epsilon = f64::EPSILON
    );

    // Case where all values are identical
    let same_values = array![5.0, 5.0, 5.0, 5.0];
    // Mean is 5.0, so SST = 0
    assert_abs_diff_eq!(
        sum_of_square_total(same_values.view()).unwrap(),
        0.0,
        epsilon = f64::EPSILON
    );

    // Single value case
    let single = array![10.0];
    assert_abs_diff_eq!(
        sum_of_square_total(single.view()).unwrap(),
        0.0,
        epsilon = f64::EPSILON
    );
}

#[test]
fn test_sum_of_squared_errors() {
    // Standard case
    let predicted = array![1.0, 2.0, 3.0, 4.0];
    let actual = array![1.2, 1.8, 3.3, 3.9];
    // SSE = (1-1.2)² + (2-1.8)² + (3-3.3)² + (4-3.9)²
    // SSE = 0.04 + 0.04 + 0.09 + 0.01 = 0.18
    let expected = 0.18;
    assert_abs_diff_eq!(
        sum_of_squared_errors(predicted.view(), actual.view()).unwrap(),
        expected,
        epsilon = 0.0001
    );

    // Case with perfect prediction
    let perfect_prediction = array![5.0, 10.0, 15.0];
    let actual_values = array![5.0, 10.0, 15.0];
    assert_abs_diff_eq!(
        sum_of_squared_errors(perfect_prediction.view(), actual_values.view()).unwrap(),
        0.0,
        epsilon = f64::EPSILON
    );

    // Empty arrays case
    let empty_pred = Array::from(Vec::<f64>::new());
    let empty_actual = Array::from(Vec::<f64>::new());
    assert!(matches!(
        sum_of_squared_errors(empty_pred.view(), empty_actual.view()),
        Err(_)
    ));
}

#[test]
#[should_panic]
fn test_sum_of_squared_errors_different_lengths() {
    // Different length arrays should panic
    let predicted = array![1.0, 2.0, 3.0];
    let actual = array![1.2, 1.8];
    sum_of_squared_errors(predicted.view(), actual.view()).unwrap();
}

#[test]
fn test_sigmoid() {
    // Test the midpoint - sigmoid(0) should be exactly 0.5
    assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = f64::EPSILON);

    // Test extreme positive input - should approach 1.0
    assert_abs_diff_eq!(sigmoid(10.0), 1.0, epsilon = 0.0001);

    // Test extreme negative input - should approach 0.0
    assert_abs_diff_eq!(sigmoid(-10.0), 0.0, epsilon = 0.0001);

    // Test symmetry property: sigmoid(-z) = 1 - sigmoid(z)
    let z = 2.5;
    assert_abs_diff_eq!(sigmoid(-z), 1.0 - sigmoid(z), epsilon = f64::EPSILON);

    // Test a few known values
    assert_abs_diff_eq!(sigmoid(1.0), 0.7310585786300049, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(sigmoid(-1.0), 0.2689414213699951, epsilon = f64::EPSILON);
}

#[test]
fn test_logistic_loss() {
    // Test case 1: Very good predictions
    let predictions1 = array![10.0, -10.0]; // Very confident predictions
    let labels1 = array![1.0, 0.0]; // Actual labels
    let loss1 = logistic_loss(predictions1.view(), labels1.view()).unwrap();
    println!("Loss1: {}", loss1);
    assert!(loss1 < 0.01); // Loss should be small

    // Test case 2: Completely wrong predictions
    let predictions2 = array![10.0, -10.0];
    let labels2 = array![0.0, 1.0];
    let loss2 = logistic_loss(predictions2.view(), labels2.view()).unwrap();
    println!("Loss2: {}", loss2);
    assert!(loss2 > 9.0);

    // Test case 3: Uncertain predictions
    let predictions3 = array![0.1, -0.1, 0.0];
    let labels3 = array![1.0, 0.0, 1.0];
    let loss3 = logistic_loss(predictions3.view(), labels3.view()).unwrap();
    println!("Loss3: {}", loss3);
    assert!(loss3 > 0.5 && loss3 < 1.0);
}

#[test]
#[should_panic]
fn test_logistic_loss_invalid_labels() {
    let predictions = array![1.0, 2.0];
    let labels = array![1.0, 0.5]; // Invalid label value
    logistic_loss(predictions.view(), labels.view()).unwrap(); // Should panic
}

#[test]
#[should_panic]
fn test_logistic_loss_different_lengths() {
    // Different length arrays should panic
    let predictions = array![0.1, 0.2, 0.3];
    let labels = array![0.0, 1.0];
    logistic_loss(predictions.view(), labels.view()).unwrap();
}

#[test]
fn test_squared_euclidean_distance_row() {
    // Basic test: Calculate squared Euclidean distance between two vectors
    let v1 = array![1.0, 2.0, 3.0];
    let v2 = array![4.0, 5.0, 6.0];
    let dist = squared_euclidean_distance_row(v1.view(), v2.view()).unwrap();
    assert_relative_eq!(dist, 27.0, epsilon = 1e-10);

    // Test with identical vectors
    let v3 = array![1.5, 2.5, 3.5];
    let dist = squared_euclidean_distance_row(v3.view(), v3.view()).unwrap();
    assert_relative_eq!(dist, 0.0, epsilon = 1e-10);

    // Test with zero vector
    let v4 = array![0.0, 0.0, 0.0];
    let v5 = array![1.0, 1.0, 1.0];
    let dist = squared_euclidean_distance_row(v4.view(), v5.view()).unwrap();
    assert_relative_eq!(dist, 3.0, epsilon = 1e-10);

    // Test with negative values
    let v6 = array![-1.0, -2.0, -3.0];
    let v7 = array![1.0, 2.0, 3.0];
    let dist = squared_euclidean_distance_row(v6.view(), v7.view()).unwrap();
    assert_relative_eq!(dist, 56.0, epsilon = 1e-10);
}

#[test]
fn test_manhattan_distance_row() {
    // Basic test: Calculate Manhattan distance between two vectors
    let v1 = array![1.0, 2.0];
    let v2 = array![4.0, 6.0];
    let dist = manhattan_distance_row(v1.view(), v2.view()).unwrap();
    assert_relative_eq!(dist, 7.0, epsilon = 1e-6);

    // Test with identical vectors
    let v3 = array![1.5, 2.5, 3.5];
    let dist = manhattan_distance_row(v3.view(), v3.view()).unwrap();
    assert_relative_eq!(dist, 0.0, epsilon = 1e-6);

    // Test with negative values
    let v4 = array![-1.0, -2.0, -3.0];
    let v5 = array![1.0, 2.0, 3.0];
    let dist = manhattan_distance_row(v4.view(), v5.view()).unwrap();
    assert_relative_eq!(dist, 12.0, epsilon = 1e-6);

    // Test with zero vector
    let v6 = array![0.0, 0.0, 0.0];
    let v7 = array![1.0, 2.0, 3.0];
    let dist = manhattan_distance_row(v6.view(), v7.view()).unwrap();
    assert_relative_eq!(dist, 6.0, epsilon = 1e-6);
}

#[test]
fn test_minkowski_distance_row() {
    // Basic test: Calculate Minkowski distance between two vectors with p=3
    let v1 = array![1.0, 2.0];
    let v2 = array![4.0, 6.0];
    let dist = minkowski_distance_row(v1.view(), v2.view(), 3.0).unwrap();
    assert_relative_eq!(dist, 4.497, epsilon = 1e-3);

    // When p=1, should equal Manhattan distance
    let dist_p1 = minkowski_distance_row(v1.view(), v2.view(), 1.0).unwrap();
    let manhattan_dist = manhattan_distance_row(v1.view(), v2.view()).unwrap();
    assert_relative_eq!(dist_p1, manhattan_dist, epsilon = 1e-6);

    // When p=2, should equal Euclidean distance
    let dist_p2 = minkowski_distance_row(v1.view(), v2.view(), 2.0).unwrap();
    let euclidean_dist = squared_euclidean_distance_row(v1.view(), v2.view())
        .unwrap()
        .sqrt();
    assert_relative_eq!(dist_p2, euclidean_dist, epsilon = 1e-6);

    // Test with identical vectors
    let v3 = array![1.5, 2.5, 3.5];
    let dist = minkowski_distance_row(v3.view(), v3.view(), 4.0).unwrap();
    assert_relative_eq!(dist, 0.0, epsilon = 1e-6);

    // Test with negative values
    let v4 = array![-1.0, -2.0, -3.0];
    let v5 = array![1.0, 2.0, 3.0];
    let dist = minkowski_distance_row(v4.view(), v5.view(), 2.0).unwrap();
    assert_relative_eq!(dist, 7.483314773547883, epsilon = 1e-6);
}

#[test]
#[should_panic(expected = "p must be greater than or equal to 1.0")]
fn test_minkowski_with_invalid_p() {
    // Test that p < 1.0 triggers a panic
    let v1 = array![1.0, 2.0];
    let v2 = array![3.0, 4.0];
    minkowski_distance_row(v1.view(), v2.view(), 0.5).unwrap();
}

#[test]
fn test_entropy_empty() {
    let empty = Array::from(Vec::<f64>::new());
    assert_eq!(entropy(empty.view()), 0.0);
}

#[test]
fn test_entropy_homogeneous() {
    // Dataset with only one class, entropy should be 0
    let homogeneous = array![1.0, 1.0, 1.0, 1.0, 1.0];
    assert_abs_diff_eq!(entropy(homogeneous.view()), 0.0, epsilon = f64::EPSILON);
}

#[test]
fn test_entropy_balanced_binary() {
    // Balanced binary dataset, entropy should be 1
    let balanced = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    assert_abs_diff_eq!(entropy(balanced.view()), 1.0, epsilon = f64::EPSILON);
}

#[test]
fn test_entropy_unbalanced() {
    // Unbalanced dataset
    let unbalanced = array![0.0, 1.0, 1.0, 1.0];
    // Probabilities: 0.0 -> 1/4, 1.0 -> 3/4
    // Entropy = -(0.25*log2(0.25) + 0.75*log2(0.75))
    let expected = -(0.25 * 0.25f64.log2() + 0.75 * 0.75f64.log2());
    assert_abs_diff_eq!(entropy(unbalanced.view()), expected, epsilon = f64::EPSILON);
}

#[test]
fn test_entropy_multiclass() {
    // Multi-class dataset
    let multiclass = array![0.0, 1.0, 2.0, 3.0];
    // Each class has probability 0.25
    // Entropy = -(4 * 0.25*log2(0.25))
    let expected = -(4.0 * 0.25 * 0.25f64.log2());
    assert_abs_diff_eq!(entropy(multiclass.view()), expected, epsilon = f64::EPSILON);
}

#[test]
fn test_gini_empty() {
    let empty = Array::from(Vec::<f64>::new());
    assert_eq!(gini(empty.view()), 0.0);
}

#[test]
fn test_gini_homogeneous() {
    // Homogeneous dataset, Gini impurity should be 0
    let homogeneous = array![1.0, 1.0, 1.0, 1.0];
    assert_abs_diff_eq!(gini(homogeneous.view()), 0.0, epsilon = f64::EPSILON);
}

#[test]
fn test_gini_balanced_binary() {
    // Balanced binary dataset, Gini impurity should be 0.5
    let balanced = array![0.0, 0.0, 1.0, 1.0];
    // Gini = 1 - (0.5^2 + 0.5^2) = 0.5
    assert_abs_diff_eq!(gini(balanced.view()), 0.5, epsilon = f64::EPSILON);
}

#[test]
fn test_gini_unbalanced() {
    // Unbalanced dataset
    let unbalanced = array![0.0, 1.0, 1.0, 1.0];
    // Probabilities: 0.0 -> 1/4, 1.0 -> 3/4
    // Gini = 1 - (0.25^2 + 0.75^2) = 1 - (0.0625 + 0.5625) = 0.375
    let expected = 1.0 - (0.25f64.powi(2) + 0.75f64.powi(2));
    assert_abs_diff_eq!(gini(unbalanced.view()), expected, epsilon = f64::EPSILON);
}

#[test]
fn test_information_gain() {
    let parent = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 zeros, 3 ones
    let left = array![0.0, 0.0, 0.0]; // 3 zeros
    let right = array![1.0, 1.0, 1.0]; // 3 ones

    // Parent entropy = 1.0
    // Left child entropy = 0.0
    // Right child entropy = 0.0
    // Information gain = 1.0 - (3/6)*0.0 - (3/6)*0.0 = 1.0
    assert_abs_diff_eq!(
        information_gain(parent.view(), left.view(), right.view()),
        1.0,
        epsilon = f64::EPSILON
    );
}

#[test]
fn test_information_gain_no_improvement() {
    let parent = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 zeros, 3 ones
    let left = array![0.0, 0.0, 1.0]; // 2 zeros, 1 one
    let right = array![0.0, 1.0, 1.0]; // 1 zero, 2 ones

    // Left and right child nodes have similar entropy distribution to parent
    // So information gain should be close to 0
    let gain = information_gain(parent.view(), left.view(), right.view());
    assert!(gain < 0.1); // Allow for small error
}

#[test]
fn test_gain_ratio() {
    let parent = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 zeros, 3 ones
    let left = array![0.0, 0.0, 0.0]; // 3 zeros
    let right = array![1.0, 1.0, 1.0]; // 3 ones

    // Information gain = 1.0
    // Split info = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
    // Gain ratio = 1.0/1.0 = 1.0
    assert_abs_diff_eq!(
        gain_ratio(parent.view(), left.view(), right.view()),
        1.0,
        epsilon = f64::EPSILON
    );
}

#[test]
fn test_gain_ratio_zero_split_info() {
    // Test case where split info is zero
    let parent = array![0.0, 0.0, 1.0, 1.0];
    let left = array![0.0, 0.0, 1.0, 1.0]; // All samples in left node
    let right: Array1<f64> = array![]; // Right node empty

    // Split info should be 0, function should return 0
    assert_eq!(gain_ratio(parent.view(), left.view(), right.view()), 0.0);
}

#[test]
fn test_variance_empty() {
    let empty: Array1<f64> = array![];
    assert_eq!(variance(empty.view()), 0.0);
}

#[test]
fn test_variance_constant() {
    // All values are the same, variance should be 0
    let constant = array![5.0, 5.0, 5.0, 5.0];
    assert!(variance(constant.view()).abs() < f64::EPSILON);
}

#[test]
fn test_variance_simple() {
    // Simple case: [1, 2, 3]
    // Mean = 2
    // MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
    let values = array![1.0, 2.0, 3.0];
    let expected = 2.0 / 3.0;
    assert_abs_diff_eq!(variance(values.view()), expected, epsilon = f64::EPSILON);
}

#[test]
fn test_variance_with_variance() {
    // Case with larger variance
    let values = array![10.0, 20.0, 30.0, 40.0, 50.0];
    // Mean = 30
    // variance = ((10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2) / 5
    //     = (400 + 100 + 0 + 100 + 400) / 5 = 1000 / 5 = 200
    let expected = 200.0;
    assert_abs_diff_eq!(variance(values.view()), expected, epsilon = f64::EPSILON);
}

#[test]
fn test_standard_deviation() {
    // Test empty array
    let empty: Array1<f64> = array![];
    assert_eq!(
        standard_deviation(empty.view()),
        Err(ModelError::InputValidationError(
            "Cannot calculate standard deviation with empty input".to_string(),
        ))
    );

    // Test array with single element
    let single = array![5.0];
    assert_eq!(standard_deviation(single.view()), Ok(0.0));

    // Test array with all same values
    let same_values = array![2.0, 2.0, 2.0, 2.0];
    let result = standard_deviation(same_values.view()).unwrap();
    assert_abs_diff_eq!(result, 0.0, epsilon = f64::EPSILON);

    // Test general case
    let values = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let expected = 2.0; // population standard deviation is 2.0
    assert_abs_diff_eq!(
        standard_deviation(values.view()).unwrap(),
        expected,
        epsilon = f64::EPSILON
    );

    // Test with negative numbers
    let negative_values = array![-5.0, -3.0, 0.0, 3.0, 5.0];
    // Correct expected population standard deviation = sqrt(68/5)
    let expected = f64::sqrt(68.0 / 5.0);
    assert_abs_diff_eq!(
        standard_deviation(negative_values.view()).unwrap(),
        expected,
        epsilon = f64::EPSILON
    );
}
