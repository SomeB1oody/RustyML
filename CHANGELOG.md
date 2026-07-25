# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24, summarized per version — each entry lists only the significant changes for that release.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [Unreleased]
### Added
- **New `utils::StandardScaler`: the stateful counterpart to `standardize`, which remembers the training mean and standard deviation.** `fit` learns the per-feature statistics, `transform` applies those frozen numbers to any later batch (a test split, a validation fold, or a single row at inference time), and `inverse_transform` maps standardized data back to the original units — the scikit-learn `StandardScaler` contract, so a train/test boundary no longer depends on the caller hand-rolling `(x - mean) / std` and threading the vectors around. Also provides `partial_fit` for fitting over batches that never exist in memory at once (the moments merge via Chan et al.), the `with_mean` / `with_std` builder flags, `get_mean` / `get_var` / `get_scale` / `get_n_samples_seen` / `get_n_features` accessors (scikit-learn's `mean_` / `var_` / `scale_` / `n_samples_seen_` / `n_features_in_`), `save_to_path` / `load_from_path` persistence, and the shared `Fit` / `Transform` / `FitTransform` traits. It shares the constant-feature rule and the Welford pass with the free function, so `StandardScaler::default().fit_transform(&x)` is bit-for-bit identical to `standardize(&x, StandardizationAxis::Column)`. Two deliberate deviations from scikit-learn: non-finite input is rejected up front rather than skipped as missing data, and `mean_` / `var_` / `scale_` stay populated even when the corresponding flag is `false`. The stateless `standardize` is unchanged and remains the right call for `Row` / `Global` axes and N-D arrays.

### Fixed
- **`IsolationForest`'s inlier/outlier threshold is now fitted from the training scores instead of recomputed per prediction call.** The old `predict_labels(&x, contamination)` took the `ceil(contamination * n)`-th highest score *of the batch it was handed*, so the decision was transductive: a single-row call always returned `-1` (the count clamps to 1), and splitting a test set in half changed its labels. `contamination` is now a builder-set [`Contamination`] rule resolved into a stored `offset` at fit time, so a sample gets the same label whether it is scored alone, in a slice, or in the full batch.

### Changed
- **Breaking: the estimator traits move from `machine_learning::traits` to the crate root as `rustyml::traits`.** `Fit`, `Predict`, `Transform`, and `FitTransform` are no longer owned by one feature: `utils::StandardScaler` implements three of them, and `utils` does not depend on `machine_learning`. Replace `use rustyml::machine_learning::traits::{...}` with `use rustyml::traits::{...}`; `rustyml::machine_learning::{Fit, Predict, ...}` and both preludes still resolve unchanged, and `prelude::utils` now exports the traits too. The `machine_learning::traits` *module path* is the only thing that disappears.
- **Breaking: the matrix-multiply backend switches from the `gemm` crate to [`gemmkit`](https://crates.io/crates/gemmkit) (via its zero-copy `gemmkit-ndarray` adapter).** All GEMM/GEMV scheduling that this crate used to hand-roll — the per-dtype FLOPs serial-vs-rayon gates, the thin-output row split, and the GEMV row split — is deleted; gemmkit gates serial-vs-parallel on problem size, picks worker counts on persistent exact-fit pools, and parallelizes matvecs on its own bandwidth-bound path. Consequences:
  - `tuning::matmul` loses `set/get_gemm_min_flops_{f32,f64}`, `set/get_gemv_min_flops_{f32,f64}`, and `set/get_colpar_min_cols_per_thread`. The backend's knobs are reachable as `GEMMKIT_*` environment variables, programmatically via the new `tuning::matmul::backend` re-export (gemmkit's tuning module), or as a machine profile emitted by the `gemmkit-tune` autotuner. `chunk_elems` and `cache_resident_max_bytes` (caller-side tiling policy) are unchanged.
  - The `benches/calibrations/gemm_calibrate.rs` sweep is removed — it calibrated the deleted gates; use `gemmkit-tune` to tune the backend per machine instead.
  - Reproducibility is unchanged in practice and stronger for matvecs: for a fixed machine and configuration, results are identical regardless of worker count (gemmkit's blocking is thread-count-independent), and GEMV is bit-identical at any worker count.
- **The hot layers fuse their bias adds (and ReLU) into the GEMM epilogue.** `Dense` forward/predict, the conv engine's forward blocks (per-filter bias, now on the gemmkit kernel instead of ndarray's `matrixmultiply`), the SimpleRNN/LSTM/GRU timestep loops (the `+ x_t@kernel` accumulate and the bias ride the kernel's store, deleting the per-timestep allocating broadcast adds), and `LDA::decision_scores` (per-class intercept) now write their pre-activations in a single pass. ReLU additionally fuses on the backend's vectorized `Relu` epilogue; the exp-based sigmoid/tanh maps deliberately stay separate vectorized sweeps (a per-element closure epilogue measures far slower). gemmkit's fused epilogues are bitwise-identical to the unfused product followed by the same scalar ops, so results are unchanged — with one deliberate exception: where a layer uses the fused `Relu` (`Dense` and SimpleRNN with `Activation::ReLU`), a `NaN` pre-activation now maps to `0.0` instead of propagating.
- **Breaking: `SVC` takes and returns `{0.0, 1.0}` labels instead of `±1.0`,** matching `LinearSVC`, `DecisionTree`, and `LogisticRegression`, so the two SVM classifiers are interchangeable without rewriting the label array. The ±1 encoding the SMO dual needs is now internal to `fit`; `get_support_vector_labels` still exposes it, and `decision_function` and the serialized format are unchanged. Passing `±1.0` to `fit` is now `Error::InvalidInput`.
- **Breaking: `IsolationForest` prediction API renamed to match scikit-learn.** `predict` now returns `Array1<i32>` (`-1` outlier / `+1` inlier) using the fitted offset, the old score-returning `predict` becomes `score_samples`, `predict_labels` is removed, and `fit_predict` returns labels. New `Contamination` enum (`Auto`, the paper's `0.5` cutoff, or `Fraction(f64)`), builder `with_contamination`, and getters `get_contamination` / `get_offset`. `Predict::Output` for `IsolationForest` becomes `Array1<i32>`, so generic code over the trait now gets labels from every estimator. **The serialized format gained two fields — re-save any persisted forests.**
- **Breaking: `LogisticRegression::predict` and `fit_predict` return `Array1<f64>` labels (`0.0`/`1.0`) instead of `Array1<i32>`.** The model is trained on `f64` labels, so predictions now round-trip back into `fit` and feed the `f64` classification metrics (`ConfusionMatrix`, `accuracy`) without a `mapv(|v| v as f64)` bridge, matching `DecisionTree`, `SVC`, and `LinearSVC`. Drop any such conversion at the call site; comparisons against integer literals become `0.0`/`1.0`. `LDA` is unchanged (`i32` in, `i32` out — its labels are arbitrary class ids).
- **Every supervised estimator now accepts `x` and `y` with different storage types.** `LinearRegression` (`fit`, `fit_predict`, `score`), `LogisticRegression`, `DecisionTree`, `SVC`, and `LinearSVC` take independent `S1`/`S2` parameters, so an owned matrix can be paired with a borrowed label view (`KNN` and `LDA` already did). `DecisionTree::predict` also drops its unnecessary `Send + Sync` bound, matching `predict_proba`. Purely relaxations — existing calls still compile.
- **Breaking: the dimensionality-reduction estimators move from `utils` to `machine_learning`.** `PCA` and `KernelPCA` now live in `machine_learning::decomposition`, and `TSNE` in `machine_learning::manifold` (mirroring scikit-learn), so they are gated by the `machine_learning` feature instead of `utils`. Update imports and prelude paths accordingly. `utils` keeps the stateless preprocessing (`normalize`, `standardize`, label encoding) and `train_test_split`. The reducers now also implement shared `Transform` / `FitTransform` estimator traits.
- **Removed the `nalgebra` runtime dependency; SVD, symmetric eigendecomposition, and QR are now hand-rolled in pure Rust.** A new crate-internal `machine_learning::linalg` module provides `symmetric_eigen`, a one-sided Jacobi `svd` (with `solve` / `pseudo_inverse`), and a Gram-Schmidt `qr_q`, all operating directly on `ndarray` arrays; the dense factorizations and iterative top-`k` eigensolvers are consolidated here. PCA, KernelPCA, LDA, and the ridge/linear-regression solvers call these instead of `nalgebra`, which drops to a dev-dependency used only to cross-check the routines in tests. Numerically equivalent up to the usual sign/rounding freedom.

## [v0.13.0] - 2026-06-23
### Added
- **Breaking: kernel `gamma` is now a `Gamma` enum supporting data-dependent rules instead of a bare `f64`.** New `Gamma` type with `Gamma::Value(f64)`, `Gamma::Scale` (scikit-learn `'scale'`) and `Gamma::Auto` (`'auto'`), resolved at fit time. Every `KernelType::{Poly, RBF, Sigmoid}` construction site must switch (e.g. `KernelType::RBF { gamma: 0.5 }` → `KernelType::RBF { gamma: Gamma::Value(0.5) }`).
- **New public `tuning` module: the crate's parallel/serial gate thresholds are now overridable at runtime.** A flat `set_*`/`get_*` facade (grouped into `matmul`, `elementwise`, `reduction`, `tree`, `conv`, `pool`, `norm`, `metrics`) lets a program retune serial-vs-rayon crossovers per machine without recompiling. Defaults and numerical results are unchanged.
- **`LDA` gains `decision_function` and `predict_proba`** (per-class discriminant scores and their row-wise softmax); `predict` labels are unchanged.
- **`LinearRegression` gains a closed-form normal-equation solver and a `score` (R²) method** via a new `Solver` enum (`GradientDescent` default, `Normal` solving the ridge least-squares system through SVD).
- **`IsolationForest::predict_labels(x, contamination)`** classifies samples as inlier (`+1`) or outlier (`-1`), mirroring scikit-learn's `IsolationForest.predict`.
- **`LinearSVC` gains a squared-hinge loss and inverse-scaling learning-rate decay** via a new `Loss` enum and a `with_learning_rate_decay` builder.
- **`TSNE` gains `min_grad_norm` early stopping** (default `1e-7`) after the early-exaggeration phase, for scikit-learn parity; pass `0.0` to disable.

### Changed
- **Breaking: model persistence switched from JSON to a compact binary format (`postcard`).** `save_to_path`/`load_from_path` on every classical-ML model, `PCA`/`KernelPCA`, and `Sequential` now use postcard (a fitted `KMeans` shrinks ~5x on disk). **Old `.json` model files can no longer be loaded — re-save any persisted models.** `IoError::Json` is renamed `IoError::Serialization`.
- **Breaking: the matrix-product backend switches from `matrixmultiply` to the pure-Rust `gemm` crate**, with runtime-dispatched SIMD kernels and shape-aware parallelism. Matrix products stay reproducible across runs on the same machine but are **no longer bit-for-bit identical** to the old backend; the public `gemm`/`gemv`/`gemm_par`/`gemv_par` API in `math::matmul` is removed (products are now reached only through crate-internal wrappers).
- **Breaking: `standardize` drops its `epsilon` parameter** and now matches `StandardScaler` exactly (`standardize(&data, axis)`), detecting constant lanes via scikit-learn's `_is_constant_feature` rule.
- **Breaking: `LDA::get_n_components` returns `Option<usize>` and the default is now `None` (auto)**, resolving at fit time to `min(n_classes - 1, n_features)`.
- **Breaking: the serialized format for `LinearRegression` and `LinearSVC` changed** (new `solver` / `learning_rate_decay` / `loss` fields) — re-save any persisted models.
- **KMeans now declares convergence on centroid shift rather than inertia change**, matching scikit-learn; this can change the iteration count and final centroids for a given `tol`.
- **`DecisionTree` scales the minimum-impurity-decrease threshold by the node's sample fraction**, and enforces `min_samples_leaf` during the split search, matching scikit-learn.
- **`LogisticRegression` regularization penalty is no longer divided by the sample count** (`alpha * R(w)` rather than `alpha * R(w) / n_samples`), matching scikit-learn's SGD convention; this changes fitted coefficients for any regularized model.
- **`PCA` now flips principal-axis signs deterministically** so all SVD solvers and repeated runs agree on axis orientation; reconstructions are unaffected.
- **`silhouette_score` now evaluates each unordered pair once** (symmetric upper-triangle fill), ~33–45% faster; the public signature is unchanged.
- Aligned numerical guards and constants with scikit-learn/PyTorch/TensorFlow conventions: `math::sigmoid` and the softmax forward drop their input clamp / denominator floor; `utils::normalize` leaves near-zero lanes unchanged; `log_loss`/`mean_absolute_percentage_error` raise their epsilon to `f64::EPSILON`; and t-SNE's numerical constants match scikit-learn.

### Fixed
- **`LinearSVC::fit` now rejects labels other than `0.0`/`1.0`** instead of silently mishandling more than two classes.
- **MeanShift keeps the current center on a zero-weight window** instead of resetting it to the origin (which previously injected a spurious cluster center).
- **`IsolationForest::predict` handles non-contiguous input rows** instead of panicking on sliced/transposed matrices.

## [v0.12.0] - 2026-06-14
### Added
- **Reproducible pseudo-random number generation.** A new crate-level `random` module with `set_global_seed`/`clear_global_seed`, plus a `random_state: Option<u64>` parameter on every `neural_network` layer and every seedable estimator, giving one-call whole-crate reproducibility with per-component override (local-over-global, mirroring Keras).
- **Major neural-network training features:** the `AdamW` optimizer; opt-in clip-by-global-norm gradient clipping across all optimizers; SGD momentum / Nesterov and decoupled weight decay; external learning-rate scheduling (`Optimizer::set_learning_rate`); `from_logits` fused softmax-cross-entropy; and `padding='same'` for the windowed pooling layers.
- **Major metrics expansion:** multi-class classification (`MulticlassConfusionMatrix`, `log_loss`, `cohen_kappa`, `top_k_accuracy`, `average_precision`, `roc_curve`, `precision_recall_curve`); the regression metrics `explained_variance_score`, `median_absolute_error`, `mean_absolute_percentage_error`; and the clustering metrics `adjusted_rand_index`, `silhouette_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`, `fowlkes_mallows_score`, `davies_bouldin_score`, `calinski_harabasz_score`. `ConfusionMatrix` gains `mcc` and `balanced_accuracy`.
- **`train_test_split_stratified`**, which splits each class independently so both subsets keep the input's class proportions.
- **Public block-parallel matrix products and deterministic blocked reductions** (`math::matmul`, `math::reduction`), reproducible across runs on the same machine.
- An internal kd-tree accelerating DBSCAN/KNN neighbor queries, and benchmark infrastructure under `benches/` (criterion) with a calibration suite for the parallel-gate thresholds.

### Changed
- **Breaking: module renames for consistency** — `metric` → `metrics`, `utility` → `utils` (both the modules **and** their Cargo features), and under `neural_network` `layer`/`optimizer`/`loss_function` → `layers`/`optimizers`/`losses` (the `LossFunction` trait becomes `Loss`).
- **Breaking: unified `Error` type built on `thiserror`**, replacing the stringly-typed `ModelError` and separate `IoError`, with domain-specific `NnError`/`TreeError`/`IoError` sub-enums, smart constructors, and a `Context` extension trait.
- **Breaking: `machine_learning`'s models are regrouped by algorithm family** into `clustering`, `linear_model`, `svm`, `tree`, `neighbors`, `discriminant_analysis`, and `ensemble` submodules (mirroring scikit-learn); every estimator is still re-exported flat, so only leaf-path imports break.
- **Breaking: constructors keep only primary hyperparameters; secondary settings move to chainable `with_*` builders** across every `machine_learning`/`utils` estimator and every `neural_network` layer and optimizer (mirroring scikit-learn's argument ordering). Defaults and serde formats are unchanged.
- **Full `neural_network` refactor** (~1360 fewer lines while adding features): a serializable `Activation` enum replacing the `T: ActivationLayer` generic; a generic optimizer interface (`Layer::parameters()` + flat-slice kernels) removing all per-layer update code; an inference-mode `predict`; dimension-generic convolution/pooling engines; channel-last Instance/Group normalization and multi-axis `LayerNorm`; and a loss trait returning `Result` instead of panicking.
- **Breaking: `BatchNormalization` is now genuine spatial batch norm for rank > 2 inputs** (per-channel parameters, statistics reducing over batch and all spatial positions, matching Keras/PyTorch).
- **Breaking: clone-free model saving** — `LayerWeight` borrows the live layer arrays via `Cow` and is the single weight type for both inspection and serialization; on-disk format is unchanged.
- **Breaking: LSTM and GRU store their gates fused** (per-gate weights packed into single kernel/recurrent/bias matrices), collapsing each projection to one wide GEMM; older saved models no longer load.
- **Barnes-Hut t-SNE (now the default, `O(n log n)`) and PCA initialization for t-SNE**, replacing the random-init/exact default; LDA's projection is rewritten to solve the true generalized eigenproblem, and its `Solver::LSQR` becomes a genuine iterative solve.
- **GEMM-based hot paths:** the ML/utils matrix products, kernel matrices, and per-sample/per-pair distance loops (KMeans, KNN, MeanShift, t-SNE) are rewritten in batched GEMM form via the shared block-parallel helpers, with all parallel/serial gates recalibrated from measurement.
- **Behavior change: NaN/Inf values now propagate instead of being silently sanitized.** The `±500`/`±1e6`/`±5` clamps and eager non-finite scans are removed from the activations, standalone activation layers, and recurrent gradients (use the new clip-by-global-norm instead).
- **Breaking: the prelude root now flattens every category** (so `use rustyml::prelude::*;` brings the actual items into scope) and the `prelude::math` submodule is dropped.

### Fixed
- **`SimpleRNN::backward` no longer accumulates its gradient across batches** (it previously summed over all prior batches without a `zero_grad`, drifting the direction).
- **`GaussianDropout` backward now uses the sampled forward noise** (it previously passed the gradient straight through).
- Numerous scikit-learn-alignment and NaN-handling fixes: the ranking metrics no longer hang or misrank on `NaN` scores; `ConfusionMatrix::recall` returns `0.0` (not `1.0`) with no actual positives; `normalized_mutual_info` uses the arithmetic-mean normalization; Minkowski `p` is validated against `p ≥ 1`; ReLU and max pooling propagate `NaN`; and several layers return recoverable errors where they previously panicked.

## [v0.11.0] - 2026-02-14
### Added
- Add `Cosine` kernel support to `KernelType`.

### Changed
- Refactor and reimplement the `PCA`, `LDA`, `KernelPCA`, and `t-SNE` estimators in the `utility` module.
- Reorganize the module layout: move each module's prelude into a dedicated `*_prelude` module, relocate `KernelType`, and move the integration tests from `./src/test/` to `./tests/`.

### Removed
- Remove the `rand` dependency in favor of `ndarray_rand`'s built-in random module.

## [v0.10.0] - 2026-01-19
### Changed
- Introduce comprehensive input validation for the neural-network optimizers and layers.
- Refactor the metric functions to use a generic `ArrayBase` for greater flexibility.

### Removed
- Remove the `statrs` dependency, replacing it with custom hypergeometric PMF / log-binomial calculations.
- Remove the `rand_distr` dependency.

## [v0.9.1] - 2026-01-16
### Added
- Add the Gaussian Dropout, Gaussian Noise, Group Normalization, and Instance Normalization layers.

## [v0.9.0] - 2025-10-22
### Added
- Add the `LayerNormalization`, `BatchNormalization`, Dropout, and SpatialDropout layers.
- Add the `AdaGrad` optimizer and the `GRU` recurrent layer.
- The activation function now implements the `Layer` trait and can be used directly as a layer.
- Introduce adaptive parallel-processing thresholds across the neural-network layers.

### Changed
- Enhance error handling and input validation across the ML models and the `utility` module.
- Include `machine_learning` and `neural_network` in the default feature set.

### Removed
- Remove the `Result` return type from the numerical functions.

## [v0.8.0] - 2025-10-11
### Added
- Add serialization/deserialization support (`save_to_path` / `load_from_path`) across the ML models, the utility module, and the `Sequential` neural network.
- Add the `normalize` (L1/L2/Lp/Max) and `standardize` (Row/Column/Global) utilities.
- Introduce progress-bar support across the ML models, utility module, and neural network.

### Changed
- Introduce parallelization thresholds across the machine-learning implementations, and reconstruct the `DecisionTree` and `IsolationForest` implementations.
- Refactor the distance-computation methods to return `Result`.

## [v0.7.0] - 2025-09-26
### Added
- Add feature flags for selective compilation.
- Add batch processing for `fit` in the `Sequential` model.
- Add the `Linear` activation function and the `label_encoding` module (sparse ↔ categorical conversions).
- Add the raw-data dataset loaders (Boston housing, Titanic, diabetes) and cost calculation/reporting for the ML models.

## [v0.6.3] - 2025-09-16
### Changed
- Improve input validation, edge-case handling, and error reporting across the `Sequential` model, the mathematical utilities, and the clustering/classification algorithms.
- Refactor the `utility` and `machine_learning` models for efficiency and maintainability.

## [v0.6.2] - 2025-06-05
### Added
- Add the `Conv1D`, `Conv3D`, `DepthwiseConv2D`, and `SeparableConv2D` convolutional layers.
- Add the `MaxPooling1D/3D`, `AveragePooling1D/3D`, `GlobalMaxPooling1D/3D`, and `GlobalAveragePooling1D/3D` pooling layers.
- Add input-dimensionality checks for the convolutional and pooling layers, and Flatten support for 3D/4D/5D tensors.

### Changed
- Replace `HashMap`/`HashSet` with `AHashMap`/`AHashSet`.

## [v0.6.1] - 2025-05-22
### Added
- Add the `Conv2D`, `MaxPooling2D`, `AveragePooling2D`, `GlobalMaxPooling2D`, `GlobalAveragePooling2D`, and `Flatten` layers.
- Add comprehensive weight structs for the neural-network layers.

### Changed
- Parallelize the `Conv2D` and pooling-layer parameter updates.

## [v0.6.0] - 2025-05-05
### Added
- Add the `LSTM` layer and the `get_weights` method with the `LayerWeight` enum.
- Add L1/L2 regularization support to linear and logistic regression.

### Changed
- Refactor the layers to enforce explicit activation usage, and unify the optimizer state handling into a single cache.

## [v0.5.1] - 2025-04-23
### Added
- Add the `SimpleRNN` layer.

### Changed
- Modularize the activation functions, optimizers, and loss functions into separate modules, and parallelize the neural-network computations.

## [v0.5.0] - 2025-04-13
### Added
- Add activation-function support to the `Dense` layer, plus getter methods for key struct properties.

### Changed
- Replace `ndarray-linalg` with `nalgebra` for `PCA`, `LDA`, and `KernelPCA`.
- Refactor the metrics API to remove `Result` in favor of panics, and encapsulate previously public struct fields.

## [v0.4.0] - 2025-04-09
### Added
- Add the neural-network module (initial implementation).
- Add the `Adam` and `RMSprop` optimizers.
- Add the `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`, and MAE loss functions.

## [v0.3.0] - 2025-04-06
### Added
- Add the `dataset` module with the iris and diabetes datasets.

### Changed
- Refactor the data handling to use `ArrayView` for memory efficiency.

## [v0.2.1] - 2025-04-04
### Added
- Add the `SVC`, `LinearSVC`, `KernelPCA`, and `LDA` models.
- Add the t-SNE (t-Distributed Stochastic Neighbor Embedding) implementation.
- Integrate Rayon for parallel computation across modules.

## [v0.2.0] - 2025-04-01
### Added
- Add the `train_test_split` utility, the AUC-ROC calculation, and the `normalized_mutual_info` / `adjusted_mutual_info` metrics.

### Changed
- Split the algorithm functions (`math`) from the model-evaluation functions (`metric`) into separate modules.

## [v0.1.1] - 2025-03-31
### Added
- Add the `preliminary_check` input-validation helper and the confusion matrix in the `math` module.

## [v0.1.0] - 2025-03-30
Initial release (crate renamed from `rust_ai` to `rustyml`).
### Added
- Add the core machine-learning models: `KMeans`, `MeanShift`, `DBSCAN`, `KNN`, `DecisionTree`, `IsolationForest`, and `PCA`.
- Add the `math` module (entropy, Gini, MSE, variance, standard deviation, Gaussian/RBF kernel) and the `metric` module.
- Add the unified `fit` / `predict` / `fit_predict` API and the prelude module.
