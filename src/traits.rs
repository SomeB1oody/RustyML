//! Estimator traits shared by every model and stateful transformer in the crate
//!
//! These traits give the otherwise independent estimators a single uniform contract
//! so that generic code can train, transform, and run predictions over any of them:
//!
//! - [`Fit`](crate::traits::Fit) - train a model from input data (`(&X, &Y)` for supervised
//!   models, `&X` for unsupervised ones)
//! - [`Predict`](crate::traits::Predict) - run inference on a fitted model, with the output
//!   type chosen by each model (labels, probabilities, cluster ids, anomaly scores, ...)
//! - [`Transform`](crate::traits::Transform) - project new data through a fitted transformer
//!   (`PCA`, `KernelPCA`, `StandardScaler`)
//! - [`FitTransform`](crate::traits::FitTransform) - fit and transform the training data in
//!   one call (`PCA`, `KernelPCA`, `TSNE`, `StandardScaler`)
//!
//! The traits live at the crate root because implementors span features: the
//! estimators in [`machine_learning`](crate::machine_learning) and the preprocessing
//! transformers in [`utils`](crate::utils). Each module also re-exports them
//! (`machine_learning::Fit`, `utils::Fit`), and both preludes bring them into scope
//!
//! Every implementor also exposes the same operations as inherent methods
//! (`model.fit(..)`, `model.predict(..)`); the trait implementations forward to them,
//! and Rust's method resolution keeps the inherent methods first, so existing call
//! sites are unaffected. Bring the traits into scope (directly or via a prelude) to
//! write code that is generic over the concrete estimator
//!
//! # Examples
//!
//! Writing one routine that trains and scores any supervised estimator, without naming
//! a concrete model type:
//!
//! ```rust
//! use rustyml::traits::{Fit, Predict};
//! use ndarray::{Array1, Array2};
//!
//! fn train_and_predict<M>(model: &mut M, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64>
//! where
//!     M: for<'a> Fit<(&'a Array2<f64>, &'a Array1<f64>)>
//!         + for<'a> Predict<&'a Array2<f64>, Output = Array1<f64>>,
//! {
//!     model.fit((x, y)).unwrap();
//!     model.predict(x).unwrap()
//! }
//!
//! // Call it with any estimator that fits the bounds, e.g.:
//! // let preds = train_and_predict(&mut LinearRegression::default(), &x, &y);
//! ```

use crate::error::Error;
#[cfg(feature = "machine_learning")]
use crate::machine_learning::{
    DBSCAN, DecisionTree, IsolationForest, KMeans, KNN, KernelPCA, LDA, LinearRegression,
    LinearSVC, LogisticRegression, MeanShift, PCA, SVC, TSNE,
};
#[cfg(feature = "utils")]
use crate::utils::{MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler};
#[cfg(feature = "machine_learning")]
use ndarray::{Array1, Ix1};
use ndarray::{Array2, ArrayBase, Data, Ix2};
#[cfg(feature = "machine_learning")]
use std::hash::Hash;

/// Trains an estimator from input data
///
/// `D` is the shape of the training data: a `(&features, &targets)` tuple for
/// supervised models and `&features` for unsupervised ones
///
/// # Type Parameters
///
/// - `D` - the training data accepted by this estimator
pub trait Fit<D> {
    /// Fits the model to `data`, returning a mutable reference to `self` for chaining
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the input is invalid or training fails
    fn fit(&mut self, data: D) -> Result<&mut Self, Error>;
}

/// Runs inference with a fitted estimator
///
/// `X` is the input type, the feature matrix `&ArrayBase<S, Ix2>` for every model,
/// and [`Predict::Output`] is the prediction type produced by the model
///
/// # Type Parameters
///
/// - `X` - the input accepted by this estimator's `predict`
pub trait Predict<X> {
    /// The prediction type produced by this estimator
    type Output;

    /// Predicts outputs for `input`
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the model is not fitted or the input is invalid
    fn predict(&self, input: X) -> Result<Self::Output, Error>;
}

/// Projects new data through a fitted transformer (out-of-sample transform)
///
/// `X` is the input type (a feature matrix `&ArrayBase<S, Ix2>`), and
/// [`Transform::Output`] is the projected representation. Implemented by the
/// decomposition estimators that learn a reusable projection ([`PCA`],
/// [`KernelPCA`]); manifold methods such as [`TSNE`], which embed only the data
/// they are fitted on, implement [`FitTransform`] instead
///
/// # Type Parameters
///
/// - `X` - the input accepted by this transformer's `transform`
pub trait Transform<X> {
    /// The transformed representation produced by this transformer
    type Output;

    /// Transforms `input` using the fitted transformer
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the transformer is not fitted or the input is invalid
    fn transform(&self, input: X) -> Result<Self::Output, Error>;
}

/// Fits a transformer and transforms the training data in a single call
///
/// `D` is the shape of the training data (`&features` for the unsupervised
/// transformers), and [`FitTransform::Output`] is the transformed training data.
/// It is the only entry point for transformers without an out-of-sample
/// projection, such as [`TSNE`]
///
/// # Type Parameters
///
/// - `D` - the training data accepted by this transformer
pub trait FitTransform<D> {
    /// The transformed representation produced by this transformer
    type Output;

    /// Fits the transformer to `data` and returns the transformed `data`
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the input is invalid or fitting fails
    fn fit_transform(&mut self, data: D) -> Result<Self::Output, Error>;
}

/// Trait implementations for the [`machine_learning`](crate::machine_learning) estimators,
/// each forwarding to the model's inherent method of the same name
#[cfg(feature = "machine_learning")]
mod machine_learning_impls {
    use super::*;

    // Supervised estimators: Fit<(&X, &Y)>

    impl<'a, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for LinearRegression
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    impl<'a, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for LogisticRegression
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    impl<'a, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for DecisionTree
    where
        S1: Data<Elem = f64> + Send + Sync,
        S2: Data<Elem = f64> + Send + Sync,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    impl<'a, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for LinearSVC
    where
        S1: Data<Elem = f64> + Send + Sync,
        S2: Data<Elem = f64> + Send + Sync,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    impl<'a, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for SVC
    where
        S1: Data<Elem = f64> + Send + Sync,
        S2: Data<Elem = f64> + Send + Sync,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    impl<'a, T, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for KNN<T>
    where
        T: Clone + Hash + Eq,
        S1: Data<Elem = f64>,
        S2: Data<Elem = T>,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    impl<'a, S1, S2> Fit<(&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>)> for LDA
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = i32>,
    {
        fn fit(
            &mut self,
            data: (&'a ArrayBase<S1, Ix2>, &'a ArrayBase<S2, Ix1>),
        ) -> Result<&mut Self, Error> {
            let (x, y) = data;
            self.fit(x, y)
        }
    }

    // Unsupervised estimators: Fit<&X>

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for KMeans
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for DBSCAN
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for MeanShift
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for IsolationForest
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for PCA
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for KernelPCA
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    // Predict<&X>

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for LinearRegression
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<f64>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for LogisticRegression
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<f64>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for DecisionTree
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<f64>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for LinearSVC
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<f64>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for SVC
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        type Output = Array1<f64>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for KMeans
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<isize>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for MeanShift
    where
        S: Data<Elem = f64> + Sync,
    {
        type Output = Array1<isize>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for IsolationForest
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<i32>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, T, S> Predict<&'a ArrayBase<S, Ix2>> for KNN<T>
    where
        T: Clone + Hash + Eq,
        S: Data<Elem = f64>,
    {
        type Output = Array1<T>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for LDA
    where
        S: Data<Elem = f64>,
    {
        type Output = Array1<i32>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    impl<'a, S> Predict<&'a ArrayBase<S, Ix2>> for DBSCAN
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        type Output = Array1<isize>;
        fn predict(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.predict(input)
        }
    }

    // Transformers: Transform<&X>

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for PCA
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for KernelPCA
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    // Transformers: FitTransform<&X>

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for PCA
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for KernelPCA
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for TSNE
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            // t-SNE's inherent `fit_transform` takes `&self` (it stores no fitted state), while this
            // trait method takes `&mut self`. Reborrow as shared so method resolution selects the
            // inherent method here rather than recursing into this trait impl
            let this: &TSNE = self;
            this.fit_transform(data)
        }
    }
} // mod machine_learning_impls

/// Trait implementations for the stateful [`utils`](crate::utils) transformers, each
/// forwarding to the transformer's inherent method of the same name
#[cfg(feature = "utils")]
mod utils_impls {
    use super::*;

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for StandardScaler
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for StandardScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for StandardScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for MinMaxScaler
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for MinMaxScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for MinMaxScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for MaxAbsScaler
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for MaxAbsScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for MaxAbsScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for RobustScaler
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for RobustScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for RobustScaler
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }

    impl<'a, S> Fit<&'a ArrayBase<S, Ix2>> for Normalizer
    where
        S: Data<Elem = f64>,
    {
        fn fit(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<&mut Self, Error> {
            self.fit(data)
        }
    }

    impl<'a, S> Transform<&'a ArrayBase<S, Ix2>> for Normalizer
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn transform(&self, input: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.transform(input)
        }
    }

    impl<'a, S> FitTransform<&'a ArrayBase<S, Ix2>> for Normalizer
    where
        S: Data<Elem = f64>,
    {
        type Output = Array2<f64>;
        fn fit_transform(&mut self, data: &'a ArrayBase<S, Ix2>) -> Result<Self::Output, Error> {
            self.fit_transform(data)
        }
    }
}

#[cfg(all(test, feature = "machine_learning"))]
mod tests {
    use super::*;
    use crate::machine_learning::linear_model::LeastSquaresSolver;
    use ndarray::{Array1, Array2};

    #[test]
    fn fit_and_predict_through_traits() {
        // Trait methods forward to the inherent methods, so they do not recurse
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);

        let mut model = LinearRegression::new(true)
            .with_solver(LeastSquaresSolver::GradientDescent {
                learning_rate: 0.05,
                max_iter: 5000,
                tol: 1e-9,
            })
            .unwrap();
        Fit::fit(&mut model, (&x, &y)).unwrap();
        let preds = Predict::predict(&model, &x).unwrap();

        assert_eq!(preds.len(), 3);
        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 0.5);
        }
    }

    #[test]
    fn transformers_through_traits() {
        // The transformer trait methods forward to the inherent methods. For t-SNE the inherent
        // method takes `&self` while the trait takes `&mut self`, so this also guards against the
        // forwarding recursing (which would overflow the stack rather than just fail an assert)
        use crate::machine_learning::{KernelPCA, KernelType, PCA, TSNE, TSNEMethod};

        let x = ndarray::array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]];

        // PCA: Fit, then Transform, both through the traits
        let mut pca = PCA::new(2).unwrap();
        Fit::fit(&mut pca, &x).unwrap();
        let projected = Transform::transform(&pca, &x).unwrap();
        assert_eq!(projected.ncols(), 2);

        // PCA / KernelPCA: FitTransform through the trait
        let mut pca2 = PCA::new(2).unwrap();
        assert_eq!(
            FitTransform::fit_transform(&mut pca2, &x).unwrap().ncols(),
            2
        );
        let mut kpca = KernelPCA::new(KernelType::Linear, 2).unwrap();
        assert_eq!(
            FitTransform::fit_transform(&mut kpca, &x).unwrap().ncols(),
            2
        );

        // TSNE: FitTransform through the trait (forwards `&mut self` -> inherent `&self`)
        let mut tsne = TSNE::new(2, 2.0, 200.0, 50)
            .unwrap()
            .with_random_state(42)
            .with_method(TSNEMethod::Exact)
            .unwrap();
        let embedding = FitTransform::fit_transform(&mut tsne, &x).unwrap();
        assert_eq!(embedding.ncols(), 2);
    }
}

#[cfg(all(test, feature = "utils"))]
mod utils_tests {
    use super::*;
    use crate::utils::StandardScaler;

    /// The scaler's trait methods forward to its inherent methods without recursing, and the
    /// fitted statistics survive the `Fit` -> `Transform` hand-off
    #[test]
    fn scaler_through_traits() {
        let x_train = ndarray::array![[1.0, 100.0], [2.0, 150.0], [3.0, 200.0]];
        let x_test = ndarray::array![[4.0, 250.0]];

        let mut scaler = StandardScaler::new();
        Fit::fit(&mut scaler, &x_train).unwrap();
        let scaled = Transform::transform(&scaler, &x_test).unwrap();
        assert_eq!(scaled.dim(), (1, 2));
        // Scaled by the training statistics: (4 - 2) / sqrt(2/3)
        assert!((scaled[[0, 0]] - 2.449_489_742_783_178).abs() < 1e-12);

        let mut scaler2 = StandardScaler::new();
        assert_eq!(
            FitTransform::fit_transform(&mut scaler2, &x_train)
                .unwrap()
                .dim(),
            (3, 2)
        );
    }
}
