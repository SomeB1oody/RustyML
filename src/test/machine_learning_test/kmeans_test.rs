use super::*;

// Helper function: Create a simple test dataset
fn create_test_data() -> Array2<f64> {
    // Create a simple dataset with two distinct clusters
    let mut data = Array2::zeros((20, 2));

    // First cluster (10 points, centered around (0,0))
    for i in 0..10 {
        let mut rng = StdRng::seed_from_u64(i as u64);
        data[[i, 0]] = rng.random_range(-30.0..-20.0);
        data[[i, 1]] = rng.random_range(-30.0..-20.0);
    }

    // Second cluster (10 points, centered around (5,5))
    for i in 10..20 {
        let mut rng = StdRng::seed_from_u64(i as u64);
        data[[i, 0]] = rng.random_range(20.0..30.0);
        data[[i, 1]] = rng.random_range(20.0..30.0);
    }

    data
}

#[test]
fn test_new_and_default() {
    // Test new method
    let kmeans = KMeans::new(3, 100, 0.0001, Some(42));

    // Verify parameters of instance created with new()
    assert!(matches!(kmeans.get_centroids(), None));
    assert!(matches!(kmeans.get_labels(), None));
    assert!(matches!(kmeans.get_inertia(), None));
    assert!(matches!(kmeans.get_actual_iterations(), None));

    // Test default method
    let default_kmeans = KMeans::default();

    // Verify default parameters
    assert!(matches!(default_kmeans.get_centroids(), None));
    assert!(matches!(default_kmeans.get_labels(), None));
    assert!(matches!(default_kmeans.get_inertia(), None));
    assert!(matches!(default_kmeans.get_actual_iterations(), None));
}

#[test]
fn test_fit() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // Test fit method
    kmeans.fit(data.view()).unwrap();

    let centroids = match kmeans.get_centroids() {
        Some(centroids) => centroids,
        None => panic!("Centroids should be available after fitting"),
    };

    // Verify state after fitting
    assert!(matches!(kmeans.get_centroids(), Some(_)));
    assert_eq!(centroids.shape(), &[2, 2]);
    assert!(matches!(kmeans.get_inertia(), Some(_)));
    assert!(matches!(kmeans.get_actual_iterations(), Some(_)));
}

#[test]
fn test_predict() {
    let mut kmeans = KMeans::new(2, 1000, 1e-7, Some(42));
    let data = create_test_data();

    // Fit first
    kmeans.fit(data.view()).unwrap();

    // Test prediction
    let predictions = kmeans.predict(data.view()).unwrap();

    // Verify prediction results
    assert_eq!(predictions.len(), 20);

    // Check if clustering is reasonable (first 10 points should be in one cluster, last 10 in another)
    let first_label = predictions[0];
    let expected_first_half = Array1::from_elem(10, first_label);
    let expected_second_half = Array1::from_elem(10, 1 - first_label); // Other cluster

    let mut correct_count = 0;

    for i in 0..10 {
        if predictions[i] == expected_first_half[i] {
            correct_count += 1;
        }
    }
    assert!(correct_count >= 8);

    correct_count = 0;

    for i in 10..20 {
        if predictions[i] == expected_second_half[i - 10] {
            correct_count += 1;
        }
    }
    assert!(correct_count >= 8)
}

#[test]
fn test_fit_predict() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // Test fit_predict method
    let predictions = kmeans.fit_predict(data.view()).unwrap();

    // Verify results
    assert_eq!(predictions.len(), 20);
    assert!(matches!(kmeans.get_centroids(), Some(_)));
    assert!(matches!(kmeans.get_labels(), Some(_)));
    assert!(matches!(kmeans.get_inertia(), Some(_)));
    assert!(matches!(kmeans.get_actual_iterations(), Some(_)));

    let labels = match kmeans.get_labels() {
        Some(labels) => labels,
        None => panic!("Labels should be available after fitting"),
    };

    // Verify labels are the same as predictions
    assert_eq!(predictions, labels);
}

#[test]
fn test_getters() {
    let mut kmeans = KMeans::new(2, 100, 0.0001, Some(42));
    let data = create_test_data();

    // State before fitting
    assert!(matches!(kmeans.get_centroids(), None));
    assert!(matches!(kmeans.get_labels(), None));
    assert!(matches!(kmeans.get_inertia(), None));
    assert!(matches!(kmeans.get_actual_iterations(), None));

    // State after fitting
    kmeans.fit(data.view()).unwrap();
    assert!(matches!(kmeans.get_centroids(), Some(_)));
    assert!(matches!(kmeans.get_labels(), Some(_)));
    assert!(matches!(kmeans.get_inertia(), Some(_)));
    assert!(matches!(kmeans.get_actual_iterations(), Some(_)));
}

#[test]
fn test_different_cluster_counts() {
    let data = create_test_data();

    // Test with k=1
    let mut kmeans_k1 = KMeans::new(1, 100, 0.0001, Some(42));
    kmeans_k1.fit(data.view()).unwrap();
    let centroids = match kmeans_k1.get_centroids() {
        Some(centroids) => centroids,
        None => panic!("Centroids should be available after fitting"),
    };
    assert_eq!(centroids.shape(), &[1, 2]);

    // Test with k=3
    let mut kmeans_k3 = KMeans::new(3, 100, 0.0001, Some(42));
    kmeans_k3.fit(data.view()).unwrap();
    let centroids = match kmeans_k3.get_centroids() {
        Some(centroids) => centroids,
        None => panic!("Centroids should be available after fitting"),
    };
    assert_eq!(centroids.shape(), &[3, 2]);
}
