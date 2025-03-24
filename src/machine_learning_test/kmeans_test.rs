use crate::machine_learning::kmeans::KMeans;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Helper function: Create a simple test dataset
fn create_test_data() -> Array2<f64> {
    // Create a simple dataset with two distinct clusters
    let mut data = Array2::zeros((20, 2));

    // First cluster (10 points, centered around (0,0))
    for i in 0..10 {
        let mut rng = StdRng::seed_from_u64(i as u64);
        data[[i, 0]] = rng.random_range(-1.0..1.0);
        data[[i, 1]] = rng.random_range(-1.0..1.0);
    }

    // Second cluster (10 points, centered around (5,5))
    for i in 10..20 {
        let mut rng = StdRng::seed_from_u64(i as u64);
        data[[i, 0]] = 5.0 + rng.random_range(-1.0..1.0);
        data[[i, 1]] = 5.0 + rng.random_range(-1.0..1.0);
    }

    data
}

#[test]
fn test_new_and_default() {
    // Test new method
    let kmeans = KMeans::new(3, 100, 0.0001, Some(42));

    // Verify parameters of instance created with new()
    assert_eq!(kmeans.get_centroids(), None);
    assert_eq!(kmeans.get_labels(), None);
    assert_eq!(kmeans.get_inertia(), None);
    assert_eq!(kmeans.get_n_iter(), None);


    // Test default method
    let default_kmeans = KMeans::default();

    // Verify default parameters
    assert_eq!(default_kmeans.get_centroids(), None);
    assert_eq!(default_kmeans.get_labels(), None);
    assert_eq!(default_kmeans.get_inertia(), None);
    assert_eq!(default_kmeans.get_n_iter(), None);
}

#[test]
fn test_fit() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // Test fit method
    kmeans.fit(&data);

    // Verify state after fitting
    assert!(kmeans.get_centroids().is_some());
    assert_eq!(kmeans.get_centroids().unwrap().shape(), &[2, 2]);
    assert!(kmeans.get_inertia().is_some());
    assert!(kmeans.get_n_iter().is_some());
}

#[test]
fn test_predict() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // Fit first
    kmeans.fit(&data);

    // Test prediction
    let predictions = kmeans.predict(&data);

    // Verify prediction results
    assert_eq!(predictions.len(), 20);

    // Check if clustering is reasonable (first 10 points should be in one cluster, last 10 in another)
    let first_label = predictions[0];
    let expected_first_half = Array1::from_elem(10, first_label);
    let expected_second_half = Array1::from_elem(10, 1 - first_label); // Other cluster

    for i in 0..10 {
        assert_eq!(predictions[i], expected_first_half[i]);
    }

    for i in 10..20 {
        assert_eq!(predictions[i], expected_second_half[i-10]);
    }
}

#[test]
fn test_fit_predict() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // Test fit_predict method
    let predictions = kmeans.fit_predict(&data);

    // Verify results
    assert_eq!(predictions.len(), 20);
    assert!(kmeans.get_centroids().is_some());
    assert!(kmeans.get_labels().is_some());
    assert!(kmeans.get_inertia().is_some());
    assert!(kmeans.get_n_iter().is_some());

    // Verify labels are the same as predictions
    assert_eq!(predictions, *kmeans.get_labels().unwrap());
}

#[test]
fn test_getters() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // State before fitting
    assert_eq!(kmeans.get_centroids(), None);
    assert_eq!(kmeans.get_labels(), None);
    assert_eq!(kmeans.get_inertia(), None);
    assert_eq!(kmeans.get_n_iter(), None);

    // State after fitting
    kmeans.fit(&data);
    assert!(kmeans.get_centroids().is_some());
    assert!(kmeans.get_labels().is_some());
    assert!(kmeans.get_inertia().is_some());
    assert!(kmeans.get_n_iter().is_some());
}

#[test]
fn test_different_cluster_counts() {
    let data = create_test_data();

    // Test with k=1
    let mut kmeans_k1 = KMeans::new(1, 100, 0.0001, Some(42));
    kmeans_k1.fit(&data);
    assert_eq!(kmeans_k1.get_centroids().unwrap().shape(), &[1, 2]);

    // Test with k=3
    let mut kmeans_k3 = KMeans::new(3, 100, 0.0001, Some(42));
    kmeans_k3.fit(&data);
    assert_eq!(kmeans_k3.get_centroids().unwrap().shape(), &[3, 2]);
}
