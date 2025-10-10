use super::*;

#[test]
fn test_train_test_split_valid_input() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

    let result = train_test_split(x.clone(), y.clone(), Some(0.4), Some(42));
    assert!(result.is_ok());

    let (x_train, x_test, y_train, y_test) = result.unwrap();

    // Verify split sizes
    assert_eq!(x_train.nrows(), 3); // 5 * (1-0.4) = 3
    assert_eq!(x_test.nrows(), 2); // 5 * 0.4 = 2
    assert_eq!(y_train.len(), 3);
    assert_eq!(y_test.len(), 2);

    // Verify all original data is included in the split
    let all_x_rows = x_train
        .rows()
        .into_iter()
        .chain(x_test.rows().into_iter())
        .collect::<Vec<_>>();
    let all_y_values = y_train.iter().chain(y_test.iter()).collect::<Vec<_>>();

    assert_eq!(all_x_rows.len(), 5);
    assert_eq!(all_y_values.len(), 5);
}

#[test]
fn test_default_parameters() {
    let x = Array2::from_shape_vec((10, 2), (1..=20).map(|x| x as f64).collect()).unwrap();
    let y = Array1::from_vec((0..10).map(|i| (i % 2) as f64).collect());

    let result = train_test_split(x.clone(), y.clone(), None, Some(42));
    assert!(result.is_ok());

    let (x_train, x_test, _y_train, _y_test) = result.unwrap();

    // Default test_size is 0.3, so test set should have 3 samples (10 * 0.3)
    assert_eq!(x_train.nrows(), 7);
    assert_eq!(x_test.nrows(), 3);
}

#[test]
fn test_same_random_state_gives_same_split() {
    let x = Array2::from_shape_vec((10, 2), (1..=20).map(|x| x as f64).collect()).unwrap();
    let y = Array1::from_vec((0..10).map(|i| (i % 2) as f64).collect());

    let result1 = train_test_split(x.clone(), y.clone(), Some(0.3), Some(42)).unwrap();
    let result2 = train_test_split(x.clone(), y.clone(), Some(0.3), Some(42)).unwrap();

    // Same random seed should produce identical splits
    assert_eq!(result1.0, result2.0); // x_train
    assert_eq!(result1.1, result2.1); // x_test
    assert_eq!(result1.2, result2.2); // y_train
    assert_eq!(result1.3, result2.3); // y_test
}

#[test]
fn test_different_random_states_give_different_splits() {
    let x = Array2::from_shape_vec((100, 2), (1..=200).map(|x| x as f64).collect()).unwrap();
    let y = Array1::from_vec((0..100).map(|i| (i % 2) as f64).collect());

    let result1 = train_test_split(x.clone(), y.clone(), Some(0.3), Some(42)).unwrap();
    let result2 = train_test_split(x.clone(), y.clone(), Some(0.3), Some(43)).unwrap();

    // Different random seeds should likely produce different splits
    // Note: There's a tiny theoretical chance they could be the same by random chance,
    // but it's highly unlikely with a large dataset
    assert!(result1.0 != result2.0 || result1.1 != result2.1);
}

#[test]
fn test_error_different_sample_sizes() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = Array1::from(vec![0.0, 1.0, 0.0]); // Only 3 samples while x has 5

    let result = train_test_split(x, y, Some(0.4), Some(42));
    assert!(result.is_err());

    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("x and y must have the same number of samples"));
    } else {
        panic!("Expected InputValidationError");
    }
}

#[test]
fn test_error_invalid_test_size() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

    // Test too small test_size
    let result = train_test_split(x.clone(), y.clone(), Some(-0.1), Some(42));
    assert!(result.is_err());

    // Test too large test_size
    let result = train_test_split(x.clone(), y.clone(), Some(1.5), Some(42));
    assert!(result.is_err());
}

#[test]
fn test_consistent_x_y_split() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = Array1::from(vec![100.0, 200.0, 300.0, 400.0, 500.0]); // Use different values for easier validation

    let (x_train, x_test, y_train, y_test) =
        train_test_split(x.clone(), y.clone(), Some(0.4), Some(42)).unwrap();

    // Verify that x and y splits are consistent (rows from x and corresponding values from y
    // should go together into either train or test set)
    for i in 0..x.nrows() {
        let x_row = x.row(i);
        let y_val = y[i];

        // Check whether this row is in train or test set
        let in_train_x = x_train.rows().into_iter().any(|r| r == x_row);
        let in_train_y = y_train.iter().any(|&val| val == y_val);

        let in_test_x = x_test.rows().into_iter().any(|r| r == x_row);
        let in_test_y = y_test.iter().any(|&val| val == y_val);

        // x and y should both be either in train or test set
        assert_eq!(in_train_x, in_train_y);
        assert_eq!(in_test_x, in_test_y);
    }
}
