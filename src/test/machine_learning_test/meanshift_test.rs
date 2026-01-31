use crate::machine_learning::meanshift::{MeanShift, estimate_bandwidth};
use ndarray::{Array2, arr2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn create_test_data() -> Array2<f64> {
    // Create a simple test dataset with three clusters
    let mut rng = StdRng::seed_from_u64(42);

    let n_samples = 300;
    let mut data = Array2::zeros((n_samples, 2));

    // First cluster
    for i in 0..100 {
        data[[i, 0]] = rng.random_range(-20.0..=-10.0);
        data[[i, 1]] = rng.random_range(-20.0..=-10.0);
    }

    // Second cluster
    for i in 100..200 {
        data[[i, 0]] = rng.random_range(0.0..=10.0);
        data[[i, 1]] = rng.random_range(0.0..=10.0);
    }

    // Third cluster
    for i in 200..300 {
        data[[i, 0]] = rng.random_range(20.0..=30.0);
        data[[i, 1]] = rng.random_range(20.0..=30.0);
    }

    data
}

#[test]
fn test_meanshift_default() {
    let ms = MeanShift::default();
    assert_eq!(ms.get_bandwidth(), 1.0);
    assert_eq!(ms.get_max_iterations(), 300);
    assert_eq!(ms.get_tolerance(), 1e-3);
    assert_eq!(ms.get_bin_seeding(), false);
    assert_eq!(ms.get_cluster_all(), true);
}

#[test]
fn test_meanshift_new() {
    let ms = MeanShift::new(2.0, Some(200), Some(1e-4), Some(true), Some(true)).unwrap();
    assert_eq!(ms.get_bandwidth(), 2.0);
    assert_eq!(ms.get_max_iterations(), 200);
    assert_eq!(ms.get_tolerance(), 1e-4);
    assert_eq!(ms.get_bin_seeding(), true);
    assert_eq!(ms.get_cluster_all(), true);
}

#[test]
fn test_meanshift_getters_before_fit() {
    let ms = MeanShift::default();

    assert!(ms.get_cluster_centers().is_none());
    assert!(ms.get_labels().is_none());
    assert!(ms.get_actual_iterations().is_none());
    assert!(ms.get_n_samples_per_center().is_none());
}

#[test]
fn test_meanshift_fit() {
    let data = create_test_data();

    let mut ms = MeanShift::new(2.0, None, None, None, Some(true)).unwrap();
    ms.fit(&data.view()).unwrap();

    // Check that all attributes are accessible after fitting
    assert!(ms.get_cluster_centers().is_some());
    assert!(ms.get_labels().is_some());
    assert!(ms.get_actual_iterations().is_some());
    assert!(ms.get_n_samples_per_center().is_some());

    // Check the shape of cluster centers
    let centers = match ms.get_cluster_centers() {
        Some(centers) => centers,
        None => panic!("Cluster centers should be available after fitting"),
    };
    assert_eq!(centers.dim().1, 2); // Number of features should be 2

    // Check labels
    let labels = match ms.get_labels() {
        Some(labels) => labels,
        None => panic!("Labels should be available after fitting"),
    };
    assert_eq!(labels.len(), data.dim().0); // Number of labels should equal number of samples
}

#[test]
fn test_meanshift_predict() {
    let data = create_test_data();

    let mut ms = MeanShift::new(2.0, None, None, None, Some(true)).unwrap();
    ms.fit(&data.view()).unwrap();

    // Create some new test points
    let test_points = arr2(&[
        [-15.0, -15.0], // Should belong to the first cluster
        [5.0, 5.0],     // Should belong to the second cluster
        [25.0, 25.0],   // Should belong to the third cluster
    ]);

    let predictions = ms.predict(test_points.view()).unwrap();
    assert_eq!(predictions.len(), 3);

    // Check that the predicted labels match the expected labels
    // Note: Since the specific label values are algorithm-determined,
    // we don't check the exact values, but rather that different points
    // are assigned to different clusters
    assert_ne!(predictions[0], predictions[1]);
    assert_ne!(predictions[1], predictions[2]);
    assert_ne!(predictions[0], predictions[2]);
}

#[test]
fn test_meanshift_fit_predict() {
    let data = create_test_data();

    let mut ms = MeanShift::new(2.0, None, None, None, Some(true)).unwrap();
    let labels = ms.fit_predict(&data.view()).unwrap();

    assert_eq!(labels.len(), data.dim().0);

    // Check that there are multiple different labels (should have at least 3 clusters)
    let mut unique_labels = std::collections::HashSet::new();
    for label in labels.iter() {
        unique_labels.insert(*label);
    }

    assert!(unique_labels.len() >= 3);
}

#[test]
fn test_bin_seeding() {
    let data = create_test_data();

    // Compare with and without bin_seeding
    let mut ms1 = MeanShift::new(2.0, None, None, Some(false), None).unwrap();
    let mut ms2 = MeanShift::new(2.0, None, None, Some(true), None).unwrap();

    ms1.fit(&data.view()).unwrap();
    ms2.fit(&data.view()).unwrap();

    // Both methods should fit successfully
    assert!(ms1.get_cluster_centers().is_some());
    assert!(ms2.get_cluster_centers().is_some());

    // With bin_seeding we typically expect fewer initial points, but we can't make strong assertions about the final number of clusters
}

#[test]
fn test_estimate_bandwidth() {
    let data = create_test_data();

    // Default parameters
    let bw1 = estimate_bandwidth(&data.view(), None, None, None).unwrap();
    assert!(bw1 > 0.0);

    // Specified quantile
    let bw2 = estimate_bandwidth(&data.view(), Some(0.3), None, None).unwrap();
    assert!(bw2 > 0.0);

    // Specified n_samples
    let bw3 = estimate_bandwidth(&data.view(), None, Some(50), None).unwrap();
    assert!(bw3 > 0.0);

    // Specified random_state
    let bw4 = estimate_bandwidth(&data.view(), None, None, Some(42)).unwrap();
    assert!(bw4 > 0.0);

    // Using the same random seed should yield the same result
    let bw5 = estimate_bandwidth(&data.view(), None, None, Some(42)).unwrap();
    assert_eq!(bw4, bw5);
}

#[test]
fn test_cluster_all_parameter() {
    let data = create_test_data();

    // With cluster_all = false, some points may not be assigned to clusters
    let mut ms1 = MeanShift::new(1.0, None, None, None, Some(false)).unwrap();
    ms1.fit(&data.view()).unwrap();
    let labels1 = match ms1.get_labels() {
        Some(labels) => labels,
        None => panic!("Labels should be available after fitting"),
    };

    // With cluster_all = true, all points should be assigned to clusters
    let mut ms2 = MeanShift::new(1.0, None, None, None, Some(true)).unwrap();
    ms2.fit(&data.view()).unwrap();
    let labels2 = match ms2.get_labels() {
        Some(labels) => labels,
        None => panic!("Labels should be available after fitting"),
    };

    // Both should have the same number of labels
    assert_eq!(labels1.len(), labels2.len());
}

#[test]
fn test_fit_with_max_iterations() {
    let data = create_test_data();

    // Set a very low max_iter to force early stopping
    let mut ms = MeanShift::new(2.0, Some(1), None, None, None).unwrap();
    ms.fit(&data.view()).unwrap();

    // Should complete successfully and n_iter should be 1
    assert_eq!(ms.get_actual_iterations().unwrap(), 1);
}
