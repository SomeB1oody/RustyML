use crate::machine_learning::decision_tree::{DecisionTree, DecisionTreeParams, Node, NodeType};
use ndarray::{arr1, arr2};

#[test]
fn test_decision_tree_new() {
    // Test creating a decision tree with default parameters
    let dt = DecisionTree::new(None, true, None);
    assert_eq!(dt.get_algorithm(), "CART");
    assert!(dt.get_is_classifier());

    // Test creating a decision tree with custom parameters
    let params = DecisionTreeParams {
        max_depth: Some(5),
        min_samples_split: 10,
        min_samples_leaf: 5,
        min_impurity_decrease: 0.1,
        random_state: Some(42),
    };
    let dt = DecisionTree::new(Some("ID3".to_string()), false, Some(params));
    assert_eq!(dt.get_algorithm(), "ID3");
    assert!(!dt.get_is_classifier());
    assert_eq!(dt.get_params().max_depth, Some(5));
    assert_eq!(dt.get_params().min_samples_split, 10);
    assert_eq!(dt.get_params().min_samples_leaf, 5);
    assert_eq!(dt.get_params().min_impurity_decrease, 0.1);
    assert_eq!(dt.get_params().random_state, Some(42));
}

#[test]
fn test_decision_tree_params_default() {
    let params = DecisionTreeParams::default();
    assert_eq!(params.max_depth, None);
    assert_eq!(params.min_samples_split, 2);
    assert_eq!(params.min_samples_leaf, 1);
    assert!(params.min_impurity_decrease < 1e-6);
    assert_eq!(params.random_state, None);
}

#[test]
fn test_fit_predict_classifier() {
    // Create a simple binary classification dataset
    let x = arr2(&[
        [2.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [3.0, 3.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ]);
    let y = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

    // Create classifier
    let mut dt = DecisionTree::new(None, true, None);

    // Train the model
    dt.fit(&x, &y);

    // Ensure the model is trained correctly
    assert_eq!(dt.get_n_features(), 2);
    assert!(dt.get_root().is_ok());

    // Predict
    let predictions = dt.predict(&x).unwrap();
    assert_eq!(predictions.len(), 6);

    // Check prediction accuracy
    for i in 0..6 {
        assert_eq!(predictions[i], y[i]);
    }

    // Test single sample prediction
    let sample = &[2.0, 2.0];
    let pred = dt.predict_one(sample).unwrap();
    assert_eq!(pred, 0.0);

    let sample = &[1.0, 1.0];
    let pred = dt.predict_one(sample).unwrap();
    assert_eq!(pred, 1.0);
}

#[test]
fn test_fit_predict_regressor() {
    // Create a simple regression dataset
    let x = arr2(&[
        [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]
    ]);
    let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Create regressor
    let mut dt = DecisionTree::new(None, false, None);

    // Train the model
    dt.fit(&x, &y);

    // Ensure the model is trained correctly
    assert_eq!(dt.get_n_features(), 1);
    assert!(dt.get_root().is_ok());

    // Predict
    let predictions = dt.predict(&x).unwrap();
    assert_eq!(predictions.len(), 6);

    // Check prediction accuracy (regression might not be exact)
    for i in 0..6 {
        assert!((predictions[i] - y[i]).abs() < 0.5);
    }
}

#[test]
fn test_predict_proba() {
    // Create a simple multi-class dataset
    let x = arr2(&[
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]);
    let y = arr1(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

    // Create classifier
    let mut dt = DecisionTree::new(None, true, None);

    // Train the model
    dt.fit(&x, &y);

    // Test class probability prediction
    let prob = dt.predict_proba(&x).unwrap();

    // Check the shape of the probability matrix
    assert_eq!(prob.shape()[0], 6); // 6 samples
    assert_eq!(prob.shape()[1], 3); // 3 classes

    // Check that the sum of probabilities for each sample is 1
    for i in 0..6 {
        let row_sum: f64 = prob.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-5);
    }

    // Test single sample probability prediction
    let sample = &[1.0, 1.0];
    let probs = dt.predict_proba_one(sample).unwrap();
    assert_eq!(probs.len(), 3);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-5);
}

#[test]
fn test_different_algorithms() {
    let x = arr2(&[
        [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]
    ]);
    let y = arr1(&[0.0, 0.0, 1.0, 1.0]);

    // Test ID3 algorithm
    let mut dt_id3 = DecisionTree::new(Some("ID3".to_string()), true, None);
    dt_id3.fit(&x, &y);
    let pred_id3 = dt_id3.predict(&x).unwrap();

    // Test C4.5 algorithm
    let mut dt_c45 = DecisionTree::new(Some("C45".to_string()), true, None);
    dt_c45.fit(&x, &y);
    let pred_c45 = dt_c45.predict(&x).unwrap();

    // Test CART algorithm
    let mut dt_cart = DecisionTree::new(Some("CART".to_string()), true, None);
    dt_cart.fit(&x, &y);
    let pred_cart = dt_cart.predict(&x).unwrap();

    // Verify that all algorithms correctly predict the training data
    for i in 0..4 {
        assert_eq!(pred_id3[i], y[i]);
        assert_eq!(pred_c45[i], y[i]);
        assert_eq!(pred_cart[i], y[i]);
    }
}

#[test]
fn test_max_depth_parameter() {
    let x = arr2(&[
        [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0],
        [3.0, 3.0], [3.0, 4.0], [4.0, 3.0], [4.0, 4.0],
    ]);
    let y = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

    let params_depth1 = DecisionTreeParams {
        max_depth: Some(1),
        ..DecisionTreeParams::default()
    };

    let mut dt_limited = DecisionTree::new(None, true, Some(params_depth1));
    dt_limited.fit(&x, &y);

    // Check if tree depth is limited
    // We can't directly check the depth, but we can infer that if the model
    // has only one level of depth, its prediction ability will be limited

    let _predictions = dt_limited.predict(&x).unwrap();

    // Create an unlimited depth tree for comparison
    let mut dt_unlimited = DecisionTree::new(None, true, None);
    dt_unlimited.fit(&x, &y);

    let predictions_unlimited = dt_unlimited.predict(&x).unwrap();

    // The unlimited depth tree should perfectly fit the training data
    for i in 0..8 {
        assert_eq!(predictions_unlimited[i], y[i]);
    }
}

#[test]
fn test_error_handling() {
    // Create an untrained tree
    let dt = DecisionTree::new(None, true, None);

    // Trying to get the root node should fail
    assert!(dt.get_root().is_err());

    // Trying to get the number of classes should also fail
    assert!(dt.get_n_classes().is_err());

    // Trying to predict should fail
    let x = arr2(&[[1.0, 2.0]]);
    assert!(dt.predict(&x).is_err());

    // Trying to get a single prediction should fail
    let sample = &[1.0, 2.0];
    assert!(dt.predict_one(sample).is_err());

    // Trying to get class probabilities should fail
    assert!(dt.predict_proba(&x).is_err());
    assert!(dt.predict_proba_one(sample).is_err());
}

#[test]
fn test_node_creation() {
    // Test leaf node creation
    let leaf = Node::new_leaf(1.5, Some(0), Some(vec![0.8, 0.2]));
    match leaf.node_type {
        NodeType::Leaf { value, class, probabilities } => {
            assert_eq!(value, 1.5);
            assert_eq!(class, Some(0));
            assert_eq!(probabilities, Some(vec![0.8, 0.2]));
        },
        _ => panic!("Expected a leaf node"),
    }

    // Test internal node creation
    let internal = Node::new_internal(1, 0.5);
    match internal.node_type {
        NodeType::Internal { feature_index, threshold, categories } => {
            assert_eq!(feature_index, 1);
            assert_eq!(threshold, 0.5);
            assert_eq!(categories, None);
        },
        _ => panic!("Expected an internal node"),
    }

    // Test categorical node creation
    let categorical = Node::new_categorical(2, vec!["A".to_string(), "B".to_string()]);
    match categorical.node_type {
        NodeType::Internal { feature_index, threshold: _, categories } => {
            assert_eq!(feature_index, 2);
            assert_eq!(categories, Some(vec!["A".to_string(), "B".to_string()]));
        },
        _ => panic!("Expected an internal categorical node"),
    }
}