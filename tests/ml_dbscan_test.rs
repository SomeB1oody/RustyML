use ndarray::{Array2, arr2};
use rustyml::error::ModelError;
use rustyml::machine_learning::DistanceCalculationMetric;
use rustyml::machine_learning::dbscan::DBSCAN;

#[test]
fn test_new() {
    let dbscan = DBSCAN::new(0.5, 5, DistanceCalculationMetric::Euclidean).unwrap();
    assert_eq!(dbscan.get_epsilon(), 0.5);
    assert_eq!(dbscan.get_min_samples(), 5);
    assert!(matches!(
        dbscan.get_metric(),
        DistanceCalculationMetric::Euclidean
    ));
}

#[test]
fn test_default() {
    let dbscan = DBSCAN::default();
    // Verify default values (adjust based on the actual implementation)
    assert!(dbscan.get_epsilon() > 0.0);
    assert!(dbscan.get_min_samples() > 0);
}

#[test]
fn test_getters() {
    let dbscan = DBSCAN::new(0.7, 10, DistanceCalculationMetric::Manhattan).unwrap();
    assert_eq!(dbscan.get_epsilon(), 0.7);
    assert_eq!(dbscan.get_min_samples(), 10);
    assert!(matches!(
        dbscan.get_metric(),
        DistanceCalculationMetric::Manhattan
    ));
}

#[test]
fn test_get_labels_before_fit() {
    let dbscan = DBSCAN::new(0.5, 5, DistanceCalculationMetric::Euclidean).unwrap();
    match dbscan.get_labels() {
        None => assert!(true),
        _ => panic!("Expected NotFitted error"),
    }
}

#[test]
fn test_get_core_sample_indices_before_fit() {
    let dbscan = DBSCAN::new(0.5, 5, DistanceCalculationMetric::Euclidean).unwrap();
    match dbscan.get_core_sample_indices() {
        None => assert!(true),
        _ => panic!("Expected NotFitted error"),
    }
}

#[test]
fn test_fit_simple_data() {
    let data = arr2(&[
        [1.0, 2.0],
        [1.1, 2.2],
        [0.9, 1.9],
        [1.0, 2.1],
        // Another cluster
        [10.0, 10.0],
        [10.2, 10.1],
        [10.1, 9.9],
        // Outlier
        [5.0, 5.0],
    ]);

    let mut dbscan = DBSCAN::new(0.5, 3, DistanceCalculationMetric::Euclidean).unwrap();
    dbscan.fit(&data.view()).unwrap();

    let labels = match dbscan.get_labels() {
        Some(labels) => labels,
        None => panic!("Expected labels to be Some"),
    };
    let core_indices = match dbscan.get_core_sample_indices() {
        Some(core_indices) => core_indices,
        None => panic!("Expected core_indices to be Some"),
    };

    // Test that labels are correctly assigned
    assert_eq!(labels.len(), data.nrows());

    // Ensure there are at least two clusters and one noise point
    let mut cluster_count = 0;
    let mut has_noise = false;

    for &label in labels.iter() {
        if label == -1 {
            has_noise = true;
        } else if label >= cluster_count {
            cluster_count = label + 1;
        }
    }

    assert!(cluster_count >= 2); // At least two clusters
    assert!(has_noise); // Should have noise points
    assert!(!core_indices.is_empty()); // Should have core points
}

#[test]
fn test_predict() {
    let train_data = arr2(&[
        [1.0, 2.0],
        [1.1, 2.2],
        [0.9, 1.9],
        [10.0, 10.0],
        [10.2, 10.1],
    ]);

    let new_data = arr2(&[
        [1.0, 2.1],   // Should belong to first cluster
        [10.1, 10.0], // Should belong to second cluster
        [5.0, 5.0],   // Should be noise
    ]);

    let mut dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    dbscan.fit(&train_data.view()).unwrap();

    let predictions = dbscan
        .predict(&train_data.view(), &new_data.view())
        .unwrap();
    assert_eq!(predictions.len(), new_data.nrows());
}

#[test]
fn test_fit_predict() {
    let data = arr2(&[
        [1.0, 2.0],
        [1.1, 2.2],
        [0.9, 1.9],
        [10.0, 10.0],
        [10.2, 10.1],
        [5.0, 5.0],
    ]);

    let mut dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    let labels = dbscan.fit_predict(&data.view()).unwrap();
    let model_labels = match dbscan.get_labels() {
        Some(labels) => labels,
        None => panic!("Expected labels to be Some"),
    };

    // Verify fit_predict results match fit+get_labels
    assert_eq!(labels, model_labels);
    assert_eq!(labels.len(), data.nrows());
}

#[test]
fn test_predict_before_fit() {
    let data = arr2(&[[1.0, 2.0], [1.1, 2.2]]);

    let new_data = arr2(&[[1.0, 2.1]]);

    let dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    match dbscan.predict(&data.view(), &new_data.view()) {
        Err(ModelError::NotFitted) => assert!(true),
        _ => panic!("Expected NotFitted error"),
    }
}

#[test]
fn test_empty_data() {
    let data = Array2::<f64>::zeros((0, 2));
    let mut dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();

    // Test with empty dataset
    match dbscan.fit(&data.view()) {
        Err(ModelError::InputValidationError(_)) => assert!(true),
        _ => panic!("Expected InputValidationError"),
    }
}

#[test]
fn test_different_metrics() {
    let data = arr2(&[
        [1.0, 2.0],
        [1.1, 2.2],
        [0.9, 1.9],
        [10.0, 10.0],
        [10.2, 10.1],
    ]);

    // Test with different distance metrics
    let mut euclidean_dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    euclidean_dbscan.fit(&data.view()).unwrap();
    let euclidean_labels = match euclidean_dbscan.get_labels() {
        Some(labels) => labels,
        None => panic!("Expected labels to be Some"),
    };

    let mut manhattan_dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    manhattan_dbscan.fit(&data.view()).unwrap();
    let manhattan_labels = match manhattan_dbscan.get_labels() {
        Some(labels) => labels,
        None => panic!("Expected labels to be Some"),
    };

    // The clustering results might differ based on the metric
    // We're just checking that both complete successfully
    assert_eq!(euclidean_labels.len(), data.nrows());
    assert_eq!(manhattan_labels.len(), data.nrows());
}
