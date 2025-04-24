/// Represents different distance calculation methods used in various machine learning algorithms.
///
/// This enum defines common distance metrics that can be used in clustering algorithms,
/// nearest neighbor searches, and other applications where distance between points is relevant.
///
/// # Variants
///
/// - `Euclidean` - Euclidean distance (L2 norm), calculated as the square root of the sum of squared differences between corresponding coordinates.
/// - `Manhattan` - Manhattan distance (L1 norm), calculated as the sum of absolute differences between corresponding coordinates.
/// - `Minkowski` - A generalized metric that includes both Euclidean and Manhattan distances as special cases. Requires an additional parameter p (not implemented in this enum).
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceCalculationMetric {
    /// Euclidean distance (L2 norm) - the straight-line distance between two points.
    Euclidean,
    /// Manhattan distance (L1 norm) - the sum of absolute differences between coordinates.
    Manhattan,
    /// Minkowski distance - a generalized metric that includes both Euclidean and Manhattan distances.
    Minkowski,
}

/// Linear regression module implementing the ordinary least squares method.
///
/// This module provides functionality to fit a linear relationship between independent
/// variables and a dependent variable, making predictions based on this relationship.
/// It implements gradient descent optimization for finding the optimal coefficients.
///
/// # Features
///
/// * Configurable fit intercept to include/exclude bias term
/// * Gradient descent optimization with configurable learning rate
/// * Early stopping based on convergence tolerance
/// * Maximum iteration limit for optimization
///
/// # Examples
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::linear_regression::LinearRegression;
/// use rustyml::machine_learning::RegularizationType;
///
/// // Create a new linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None);
///
/// // Example data
/// let x = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
/// let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
///
/// // Fit the model and make predictions
/// model.fit(x.view(), y.view()).unwrap();
/// let predictions = model.predict(x.view()).unwrap();
/// ```
///
/// # Performance Considerations
///
/// The algorithm's convergence and performance depend on appropriate values for learning rate, tolerance, and maximum iterations, which may need to be tuned for specific datasets.
pub mod linear_regression;

/// Logistic regression module for binary classification problems.
///
/// This module provides implementation of logistic regression, a statistical model that uses
/// a logistic function to model a binary dependent variable. It is widely used for
/// classification problems where the goal is to determine the probability that an instance
/// belongs to a particular class.
///
/// # Features
///
/// - Binary classification using gradient descent optimization
/// - Configurable intercept term (bias)
/// - Adjustable learning rate and convergence tolerance
/// - Maximum iteration limit to prevent excessive computation
/// - Probability estimation via sigmoid function
/// - Polynomial feature transformation support
///
/// # Examples
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::logistic_regression::LogisticRegression;
///
/// // Create a new logistic regression model
/// let mut model = LogisticRegression::new(true, 0.1, 1000, 1e-5, None);
///
/// // Example data (features and binary labels)
/// let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
///
/// // Fit the model and predict
/// model.fit(x.view(), y.view()).unwrap();
/// let predictions = model.predict(x.view()).unwrap();
/// ```
///
/// # Note
///
/// - The model uses gradient descent optimization which may require tuning of learning rate
///   and other parameters for optimal performance.
/// - For non-linearly separable data, consider using `generate_polynomial_features` to
///   transform the input data before training.
pub mod logistic_regression;

/// K-means clustering implementation for unsupervised learning.
///
/// This module provides an implementation of the K-means clustering algorithm, which partitions
/// data points into k clusters, each represented by the mean (centroid) of the points within
/// the cluster. The algorithm aims to minimize the within-cluster sum-of-squares (inertia).
///
/// # Features
///
/// - Configurable number of clusters (k)
/// - Centroid initialization using random selection
/// - Iterative optimization with customizable maximum iterations
/// - Early stopping based on convergence tolerance
/// - Optional random seed for reproducible results
/// - Calculation of cluster inertia (within-cluster sum of squared distances)
///
/// # Examples
/// ```rust
/// use ndarray::Array2;
/// use rustyml::machine_learning::kmeans::KMeans;
///
/// // Create a new KMeans model with 3 clusters
/// let mut model = KMeans::new(3, 100, 1e-4, Some(42));
///
/// // Example data
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     8.0, 7.5,
///     8.2, 8.0,
///     15.0, 15.0,
///     15.5, 14.5,
/// ]).unwrap();
///
/// // Fit the model and get cluster labels
/// model.fit(data.view()).unwrap();
/// let labels = model.get_labels().unwrap();
/// let centroids = model.get_centroids().unwrap();
/// let inertia = model.get_inertia().unwrap();
/// ```
///
/// # Algorithm Details
///
/// The algorithm works as follows:
/// 1. Initialize k centroids (randomly or using specific methods)
/// 2. Assign each data point to the nearest centroid
/// 3. Update centroids as the mean of all points assigned to that centroid
/// 4. Repeat steps 2-3 until convergence or maximum iterations
///
/// # Notes
///
/// - The algorithm may converge to a local optimum, so multiple runs with different
///   initializations might be necessary for optimal results
/// - The quality of clustering depends heavily on the initial centroid placement
/// - Performance scales with the number of data points, dimensions, and clusters
pub mod kmeans;

/// K-Nearest Neighbors (KNN) implementation for classification and regression.
///
/// This module provides an implementation of the k-nearest neighbors algorithm, a non-parametric
/// method used for classification and regression. The output depends on the k closest training
/// examples in the feature space, with predictions made based on a majority vote (classification)
/// or average (regression) of the nearest neighbors.
///
/// # Features
///
/// - Configurable number of neighbors (k)
/// - Multiple distance metric options through the `Metric` enum
/// - Two weighting strategies: uniform and distance-based
/// - Support for any hashable and equatable label type
/// - Efficient prediction for single and multiple test samples
///
/// # Weighting Strategies
///
/// - `Uniform` - All neighbors contribute equally to the prediction
/// - `Distance` - Closer neighbors contribute more than distant ones, weighted by inverse distance
///
/// # Examples
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::knn::{KNN, WeightingStrategy};
/// use rustyml::machine_learning::DistanceCalculationMetric as Metric;
///
/// // Create KNN classifier with 3 neighbors, distance-based weighting and Euclidean distance
/// let mut model = KNN::new(3, WeightingStrategy::Distance, Metric::Euclidean);
///
/// // Training data
/// let x_train = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
/// let y_train = Array1::from_vec(vec!["A", "A", "B", "B"]);
///
/// // Fit the model and make predictions
/// model.fit(x_train.view(), y_train.view()).unwrap();
///
/// // Predict for new data
/// let x_test = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 3.5, 3.5]).unwrap();
/// let predictions = model.predict(x_test.view()).unwrap();
/// ```
///
/// # Algorithm Details
///
/// For each query point, the algorithm:
/// 1. Calculates distances to all training points
/// 2. Selects the k nearest neighbors
/// 3. Assigns weights based on the chosen weighting strategy
/// 4. Determines the output label based on weighted voting
///
/// # Performance Considerations
///
/// - KNN has minimal training cost but higher prediction cost as distances to all training
///   points must be computed for each prediction
/// - Performance degrades with high-dimensional data (curse of dimensionality)
/// - Memory usage scales with the size of the training set, as all training data must be retained
pub mod knn;

/// Mean Shift clustering algorithm implementation.
///
/// This module provides an implementation of the Mean Shift algorithm, a non-parametric
/// clustering technique that does not require specifying the number of clusters in advance.
/// It works by updating candidates for centroids to be the mean of points within a given region
/// (determined by the bandwidth parameter) until convergence.
///
/// # Features
///
/// - Automatic discovery of cluster centers without specifying cluster count
/// - Bandwidth parameter to control cluster size
/// - Optional bin seeding for improved performance on large datasets
/// - Configurable convergence tolerance and maximum iterations
/// - Option to assign all points to clusters or leave some as outliers
/// - Automatic bandwidth estimation via `estimate_bandwidth` function
///
/// # Examples
/// ```rust
/// use ndarray::Array2;
/// use rustyml::machine_learning::meanshift::{MeanShift, estimate_bandwidth};
///
/// // Example data
/// let data = Array2::from_shape_vec((8, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     8.0, 7.5,
///     8.2, 8.0,
///     8.1, 7.8,
///     15.0, 15.0,
///     15.2, 14.8,
///     15.5, 14.5,
/// ]).unwrap();
///
/// // Estimate bandwidth and create model
/// let bandwidth = estimate_bandwidth(data.view(), Some(0.2), None, None);
/// let mut model = MeanShift::new(bandwidth, None, None, Some(false), Some(true));
///
/// // Fit the model and get cluster labels
/// model.fit(data.view()).unwrap();
/// let labels = model.get_labels().unwrap();
/// let centers = model.get_cluster_centers().unwrap();
/// ```
///
/// # Algorithm Details
///
/// The algorithm works by:
/// 1. Initializing candidate centroids (all points or bin seeds if enabled)
/// 2. For each candidate centroid, calculating the mean of all points within the bandwidth radius
/// 3. Moving the centroid to this mean position
/// 4. Repeating steps 2-3 until convergence or maximum iterations
/// 5. Merging centroids that are within the bandwidth of each other
/// 6. Assigning each data point to its nearest centroid
///
/// # Performance Considerations
///
/// - The algorithm can be computationally expensive for large datasets
/// - Bin seeding can significantly improve performance with minimal impact on results
/// - The bandwidth parameter critically affects results - use `estimate_bandwidth` when uncertain
/// - Unlike k-means, it can find irregularly shaped clusters and doesn't need a predefined number of clusters
///
/// # References
///
/// - Comaniciu, D., & Meer, P. (2002). Mean shift: A robust approach toward feature space analysis.
///   IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(5), 603-619.
pub mod meanshift;

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm implementation.
///
/// This module provides an implementation of DBSCAN, a density-based clustering algorithm
/// that groups together points that are closely packed (points with many nearby neighbors),
/// marking points that lie alone in low-density regions as outliers or noise.
///
/// # Features
///
/// - Discovers clusters of arbitrary shape
/// - Automatically identifies outliers as noise points
/// - Does not require specifying the number of clusters a priori
/// - Configurable distance metric through the `Metric` enum
/// - Customizable density parameters via `eps` and `min_samples`
/// - Efficient prediction for new data points based on trained model
///
/// # Parameters
///
/// - `eps` - The maximum distance between two samples for them to be considered as in the same neighborhood
/// - `min_samples` - The minimum number of samples in a neighborhood for a point to be considered a core point
/// - `metric` - The distance metric to use for finding neighbors
///
/// # Examples
/// ```rust
/// use ndarray::Array2;
/// use rustyml::machine_learning::dbscan::DBSCAN;
/// use rustyml::machine_learning::DistanceCalculationMetric as Metric;
///
/// // Create a DBSCAN model
/// let mut model = DBSCAN::new(0.5, 5, Metric::Euclidean);
///
/// // Example data
/// let data = Array2::from_shape_vec((10, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     0.2, 0.0,
///     0.0, 0.2,
///     0.1, 0.2,
///     9.0, 9.0,
///     9.1, 9.1,
///     9.0, 9.2,
///     9.2, 9.0,
///     5.0, 5.0,  // Noise point
/// ]).unwrap();
///
/// // Fit the model and get cluster assignments
/// model.fit(data.view()).unwrap();
/// let labels = model.get_labels().unwrap();
/// let core_indices = model.get_core_sample_indices().unwrap();
///
/// // Noise points are labeled as -1
/// ```
///
/// # Algorithm Details
///
/// DBSCAN works by:
/// 1. Finding all points within distance `eps` of each point (its neighborhood)
/// 2. Identifying core points that have at least `min_samples` points in their neighborhood
/// 3. Forming clusters by connecting core points that are neighbors
/// 4. Assigning non-core points to clusters if they're in a core point's neighborhood
/// 5. Labeling remaining points as noise (with label -1)
///
/// # Advantages and Limitations
///
/// ## Advantages
/// - Does not require knowing the number of clusters beforehand
/// - Can find arbitrarily shaped clusters
/// - Robust to outliers
/// - Only requires two parameters
///
/// ## Limitations
/// - Not entirely deterministic: border points can be assigned to different clusters
/// - Struggles with clusters of varying densities
/// - Sensitive to parameter selection
/// - Can be computationally expensive for large datasets
///
/// # References
///
/// Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for
/// discovering clusters in large spatial databases with noise. In Proceedings of the Second
/// International Conference on Knowledge Discovery and Data Mining (KDD-96) (pp. 226-231).
pub mod dbscan;

/// Decision Tree implementation for classification and regression tasks.
///
/// This module provides an implementation of Decision Tree algorithms, which create models that
/// predict the value of a target variable by learning simple decision rules inferred from data features.
/// The implementation supports multiple algorithms (ID3, C4.5, and CART) and can be used for both
/// classification and regression tasks.
///
/// # Features
///
/// - Multiple decision tree algorithms: ID3, C4.5, and CART
/// - Support for both classification and regression tasks
/// - Customizable tree parameters (depth, minimum samples, etc.)
/// - Probability predictions for classification
/// - Categorical feature support
/// - Configurable splitting criteria
/// - Tree structure visualization
///
/// # Decision Tree Algorithms
///
/// - `ID3` - Iterative Dichotomiser 3, uses information gain for splitting
/// - `C45` - An extension of ID3 that uses gain ratio and can handle continuous attributes
/// - `CART` - Classification And Regression Trees, uses Gini impurity (classification) or MSE (regression)
///
/// # Parameters
///
/// The tree behavior can be customized through `DecisionTreeParams`:
///
/// - `max_depth` - Maximum depth of the tree (None for unlimited)
/// - `min_samples_split` - Minimum samples required to split an internal node
/// - `min_samples_leaf` - Minimum samples required in a leaf node
/// - `min_impurity_decrease` - Minimum impurity decrease required for splitting
/// - `random_state` - Seed for random number generator (for reproducibility)
///
/// # Examples
///
/// ## Classification Example
/// ``` rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::decision_tree::{DecisionTree, Algorithm, DecisionTreeParams};
///
/// // Create a decision tree classifier
/// let params = DecisionTreeParams {
///     max_depth: Some(5),
///     min_samples_split: 2,
///     min_samples_leaf: 1,
///     min_impurity_decrease: 0.0,
///     random_state: Some(42),
/// };
///
/// let mut clf = DecisionTree::new(Algorithm::CART, true, Some(params));
///
/// // Example data
/// let x = Array2::from_shape_vec((6, 2), vec![
///     2.0, 2.0,
///     1.0, 3.0,
///     4.0, 3.0,
///     3.0, 4.0,
///     5.0, 1.0,
///     3.0, 2.0,
/// ]).unwrap();
///
/// let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
///
/// // Train the model
/// clf.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let predictions = clf.predict(x.view()).unwrap();
/// let probabilities = clf.predict_proba(x.view()).unwrap();
/// ```
///
/// ## Regression Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::decision_tree::{DecisionTree, Algorithm};
///
/// // Create a decision tree regressor
/// let mut regressor = DecisionTree::new(Algorithm::CART, false, None);
///
/// // Example data
/// let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
///
/// // Train and predict
/// regressor.fit(x.view(), y.view()).unwrap();
/// let predictions = regressor.predict(x.view()).unwrap();
/// ```
///
/// # Tree Structure
///
/// The decision tree is composed of `Node` structures which can be either:
/// - Internal nodes with a feature index and threshold (or categories for categorical features)
/// - Leaf nodes with a prediction value and optional class/probability information
///
/// # Performance Considerations
///
/// - Decision trees are prone to overfitting, especially with deep trees
/// - The `max_depth` parameter can help control overfitting
/// - Training time increases with dataset size and feature count
/// - Trees can handle both numerical and categorical data without preprocessing
///
/// # References
///
/// - Quinlan, J. R. (1986). Induction of decision trees. Machine learning, 1(1), 81-106.
/// - Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
/// - Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and
///   regression trees. CRC press.
pub mod decision_tree;

/// Isolation Forest algorithm implementation for anomaly detection.
///
/// This module provides an implementation of the Isolation Forest algorithm, an unsupervised
/// learning method that efficiently detects anomalies by isolating observations. The algorithm
/// works on the principle that anomalies are more susceptible to isolation than normal instances
/// in the data.
///
/// # Features
///
/// - Efficient anomaly detection without requiring a labeled training set
/// - Configurable number of trees (estimators) in the ensemble
/// - Support for different sample sizes through `max_samples` parameter
/// - Adjustable tree height through `max_depth` parameter
/// - Anomaly score calculation for each data point
/// - Binary anomaly prediction (normal/anomaly) based on a contamination threshold
/// - Randomized feature selection for improved robustness
///
/// # Parameters
///
/// - `n_estimators` - Number of isolation trees in the ensemble
/// - `max_samples` - Number of samples to draw for each tree
/// - `contamination` - Expected proportion of anomalies in the dataset
/// - `max_depth` - Maximum depth of the isolation trees (None for unlimited)
/// - `random_state` - Seed for random number generation (for reproducibility)
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::isolation_forest::IsolationForest;
/// use ndarray::Array2;
///
/// // Create a new Isolation Forest model
/// let mut model = IsolationForest::new(100, 256, None, Some(42));
///
/// // Assuming we have some data
/// // Create a sample dataset with 100 points, each with 2 features
/// let mut sample_data = Vec::with_capacity(100 * 2);
/// for _ in 0..100 {
///     sample_data.push(1.0);  // First feature
///     sample_data.push(2.0);  // Second feature
/// }
/// let data = Array2::from_shape_vec((100, 2), sample_data).unwrap();
///
/// // Fit the model
/// model.fit(data.view()).unwrap();
///
/// // Or get binary predictions (true for inliers, false for outliers)
/// let predictions = model.predict(data.view());
/// ```
///
/// # Algorithm Details
///
/// Isolation Forest works by:
/// 1. Building an ensemble of isolation trees, each constructed by recursively partitioning
///    the data until instances are isolated
/// 2. Randomly selecting a feature and a split value between the min and max values of that feature
/// 3. Measuring anomaly scores based on the average path length (number of edges) an observation
///    traverses in the trees until termination
/// 4. Anomalies typically have shorter path lengths as they are easier to isolate
///
/// # Performance Considerations
///
/// - The algorithm excels with high-dimensional data where distance-based methods struggle
/// - Performs well when anomalies are isolated points rather than small clusters
/// - Subsampling makes it efficient for large datasets
/// - Time complexity is roughly O(t * n * log(n)) where t is the number of trees and n is sample size
/// - More trees generally improve stability but increase computation cost
///
/// # References
///
/// Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 Eighth
/// IEEE International Conference on Data Mining (pp. 413-422). IEEE.
pub mod isolation_forest;

/// # Support Vector Classification (SVC)
///
/// This module provides an implementation of Support Vector Classification, a powerful
/// supervised machine learning algorithm used for classification tasks.
///
/// ## Overview
///
/// Support Vector Classification works by finding the hyperplane that best separates different
/// classes in the feature space. It aims to maximize the margin between the closest data points
/// (support vectors) of different classes.
///
/// ## Features
///
/// - Multiple kernel functions: Linear, Polynomial, RBF (Radial Basis Function), and Sigmoid
/// - Configurable regularization parameter (C)
/// - SMO (Sequential Minimal Optimization) algorithm for efficient training
/// - Support for binary classification tasks
///
/// ## Example
/// ```rust
/// use rustyml::machine_learning::svc::{SVC, KernelType};
/// use ndarray::{Array1, Array2};
///
/// // Create a new SVC with RBF kernel
/// let mut classifier = SVC::new(
///     KernelType::RBF { gamma: 0.1 },
///     1.0,  // regularization parameter
///     1e-3, // tolerance
///     100   // maximum iterations
/// );
///
/// // Training data (features)
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
/// ]).unwrap();
///
/// // Labels (XOR problem)
/// let y = Array1::from_vec(vec![-1.0, 1.0, 1.0, -1.0]);
///
/// // Train the model
/// classifier.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let predictions = classifier.predict(x.view()).unwrap();
/// ```
///
/// ## References
///
/// 1. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.
/// 2. Platt, J. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines.
pub mod svc;

/// This module contains the implementation of Linear Support Vector Classification (SVC).
///
/// Linear SVC is a popular machine learning algorithm used for binary classification problems.
/// It works by finding the hyperplane that best separates the classes with the maximum margin.
/// This implementation uses gradient descent optimization with regularization options.
///
/// # Features
///
/// - Supports both L1 and L2 regularization penalties
/// - Configurable learning rate and maximum iterations
/// - Optional intercept (bias) fitting
/// - Provides access to model parameters and convergence information
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::linear_svc::{LinearSVC, PenaltyType};
/// use ndarray::{Array1, Array2};
///
/// // Create a new Linear SVC model
/// let mut model = LinearSVC::new(
///     1000,            // max_iter
///     0.01,            // learning_rate
///     0.1,             // regularization_param
///     PenaltyType::L2, // penalty
///     true,            // fit_intercept
///     1e-4,            // tolerance
/// );
///
/// // Fit the model to training data
/// let x_train = Array2::<f64>::zeros((100, 5)); // Replace with actual training data
/// let y_train = Array1::<f64>::zeros(100);      // Replace with actual labels
/// model.fit(x_train.view(), y_train.view()).unwrap();
///
/// // Make predictions
/// let x_test = Array2::<f64>::zeros((10, 5));   // Replace with actual test data
/// let predictions = model.predict(x_test.view()).unwrap();
/// ```
pub mod linear_svc;

/// # Linear Discriminant Analysis (LDA)
///
/// This module provides an implementation of Linear Discriminant Analysis, a supervised
/// dimensionality reduction technique that finds a linear combination of features that
/// characterizes or separates two or more classes.
///
/// LDA works by finding a projection that maximizes the distance between the means of different
/// classes while minimizing the variance within each class.
///
/// ## Features
///
/// - Dimensionality reduction for classification problems
/// - Projection of data onto a lower-dimensional space
/// - Statistical classification based on the Bayes rule
///
/// ## Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::linear_discriminant_analysis::LDA;
///
/// // Create feature matrix and class labels
/// let x = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 5.0, 5.0, 5.5, 4.5, 6.0, 5.0]).unwrap();
/// let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// // Create and fit LDA model
/// let mut lda = LDA::new();
/// lda.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let x_new = Array2::from_shape_vec((2, 2), vec![1.2, 2.2, 5.2, 4.8]).unwrap();
/// let predictions = lda.predict(x_new.view()).unwrap();
///
/// // Transform data to lower dimension
/// let x_transformed = lda.transform(x.view(), 1).unwrap();
/// ```
///
/// ## References
///
/// - Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems"
/// - Rao, C. R. (1948). "The utilization of multiple measurements in problems of biological classification"
pub mod linear_discriminant_analysis;

/// Performs validation checks on the input data matrices.
///
/// This function validates that:
/// - The input data matrix is not empty
/// - The input data does not contain NaN or infinite values
/// - When a target vector is provided:
///   - The target vector is not empty
///   - The target vector length matches the number of rows in the input data
///
/// # Parameters
///
/// * `x` - A 2D array of feature values where rows represent samples and columns represent features
/// * `y` - An optional 1D array representing the target variables or labels corresponding to each sample
///
/// # Returns
///
/// - `Ok(())` - If all validation checks pass
/// - `Err(ModelError::InputValidationError)` - If any validation check fails, with an informative error message
pub fn preliminary_check(
    x: ndarray::ArrayView2<f64>,
    y: Option<ndarray::ArrayView1<f64>>,
) -> Result<(), crate::ModelError> {
    if x.nrows() == 0 {
        return Err(crate::ModelError::InputValidationError(
            "Input data is empty".to_string(),
        ));
    }

    for (i, row) in x.outer_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(crate::ModelError::InputValidationError(format!(
                    "Input data contains NaN or infinite value at position [{}][{}]",
                    i, j
                )));
            }
        }
    }

    if let Some(y) = y {
        if y.len() == 0 {
            return Err(crate::ModelError::InputValidationError(
                "Target vector is empty".to_string(),
            ));
        }

        if y.len() != x.nrows() {
            return Err(crate::ModelError::InputValidationError(format!(
                "Input data and target vector have different lengths, x columns: {}, y length: {}",
                x.nrows(),
                y.len()
            )));
        }
    }
    Ok(())
}

/// Represents different types of regularization techniques used in machine learning models.
///
/// Regularization helps prevent overfitting by adding a penalty term to the model's loss function
/// during training. This enum defines common regularization approaches that can be applied to
/// various learning algorithms.
///
/// # Variants
///
/// - `L1` - L1 regularization (Lasso) that adds the sum of absolute values of parameters
///   multiplied by the specified coefficient. Promotes sparse solutions by driving some
///   parameters to exactly zero.
/// - `L2` - L2 regularization (Ridge) that adds the sum of squared parameter values
///   multiplied by the specified coefficient. Discourages large parameter values but
///   typically does not produce sparse solutions.
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso) with the specified regularization strength coefficient
    L1(f64),
    /// L2 regularization (Ridge) with the specified regularization strength coefficient
    L2(f64),
}
