# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24, summarized per version — each entry lists only the significant changes for that release.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [Unreleased]
### Added
- **New `utils::scaler` module: `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and `Normalizer` join `StandardScaler`,** completing the scikit-learn-style transformer family with the same `fit` / `transform` / `fit_transform` contract, `Fit`/`Transform`/`FitTransform` impls, and `save_to_path` persistence. `MinMaxScaler` maps each feature onto a target interval (default `[0, 1]`, retargetable with `with_feature_range` even after fitting, with optional `with_clip` clamping) and exposes `data_min_`/`data_max_`/`data_range_`/`scale_`/`min_`. `MaxAbsScaler` divides by each feature's `max(|x|)`, so structural zeros stay zero and signs survive — the one to use where min-max's shift would destroy that meaning. Both support `partial_fit` (extrema merge exactly) and `inverse_transform`. `Normalizer` is the stateful face of row-wise `normalize`: it learns only the feature count, since a sample's norm depends on that sample alone, and exists so row normalization composes with the same estimator traits. A degenerate divisor (a constant column, an all-zero column) is replaced by `1.0` per scikit-learn's `_handle_zeros_in_scale`; `StandardScaler` keeps its finer variance-based constant-feature test. `RobustScaler` centers on the median and divides by the spread of `quantile_range` (default the IQR, retargetable with `with_quantile_range`, which discards any previous fit since the stored quantiles were read at the old positions), so a few extreme values move neither statistic; its quantiles interpolate linearly between order statistics exactly as NumPy's default `linear` method does, so they match a ported scikit-learn pipeline. It has no `partial_fit` (quantiles do not merge across batches, and scikit-learn has none either) and no `unit_variance` option (that needs an inverse normal CDF the crate does not provide). `NormalizationOrder` gained `Serialize`/`Deserialize` so a fitted `Normalizer` round-trips through postcard.
- **New `utils::StandardScaler`: the stateful counterpart to `standardize`, which remembers the training mean and standard deviation.** `fit` learns the per-feature statistics, `transform` applies those frozen numbers to any later batch (a test split, a validation fold, or a single row at inference time), and `inverse_transform` maps standardized data back to the original units — the scikit-learn `StandardScaler` contract, so a train/test boundary no longer depends on the caller hand-rolling `(x - mean) / std` and threading the vectors around. Also provides `partial_fit` for fitting over batches that never exist in memory at once (the moments merge via Chan et al.), the `with_mean` / `with_std` builder flags, `get_mean` / `get_var` / `get_scale` / `get_n_samples_seen` / `get_n_features` accessors (scikit-learn's `mean_` / `var_` / `scale_` / `n_samples_seen_` / `n_features_in_`), `save_to_path` / `load_from_path` persistence, and the shared `Fit` / `Transform` / `FitTransform` traits. It shares the constant-feature rule and the Welford pass with the free function, so `StandardScaler::default().fit_transform(&x)` is bit-for-bit identical to `standardize(&x, StandardizationAxis::Column)`. Two deliberate deviations from scikit-learn: non-finite input is rejected up front rather than skipped as missing data, and `mean_` / `var_` / `scale_` stay populated even when the corresponding flag is `false`. The stateless `standardize` is unchanged and remains the right call for `Row` / `Global` axes and N-D arrays.

### Fixed
- **`r2_score` and `explained_variance_score` no longer report a perfect score for a genuinely varying target.** Both guarded the undefined constant-`y_true` case with an absolute `1e-10` threshold on an unnormalized sum of squares, so any target whose entire spread fell below it — `[1e-6, 2e-6, 3e-6]` has an SST of `2e-12` — was mistaken for a constant and scored `1.0`. Constancy is now decided from the values themselves. The exact-zero test that would be the obvious replacement does not work here either: `sum / n` does not round-trip every constant (eight copies of `0.1` leave the variance at `1.9e-34`, not `0.0`), so an `== 0.0` guard would miss real constants and return garbage of order `1e31`. Both functions keep scikit-learn's convention for a constant target: `1.0` for an exact fit, `0.0` otherwise.
- **`precision_recall_curve` no longer attaches its closing point to the wrong end of the curve.** It returned thresholds in *decreasing* order with recall *increasing*, then appended the `(precision = 1, recall = 0)` point last — where recall was `1.0` — producing a recall array (`[0.5, 0.5, 1.0, 1.0, 0.0]`) that was monotone in neither direction. The points now run in scikit-learn's order: ascending thresholds, descending recall, closing point last. Verified element for element against scikit-learn 1.9.0.
- **`KMeans` no longer leaves `labels_` and `inertia_` describing different centroids than `cluster_centers_`.** The Lloyd loop labels against the current centroids and only then installs the updated ones, so a fit that stopped at `max_iter` recorded the *previous* iteration's labels and `predict(x) != get_labels()`. A final assignment pass now runs on that exit path, as scikit-learn's final E-step does for the same reason.

### Changed
- **The default feature set is now `full`, so `cargo add rustyml` gives the whole crate.** It was `["machine_learning", "neural_network"]`, which left out exactly the two modules a ported scikit-learn script reaches for first: `train_test_split` lives in `utils` and `accuracy` in `metrics`, so the shortest possible first program did not compile and the fix — discovering that features exist and naming two more — came before any result. The change costs nothing in dependencies: `utils` and `metrics` pull no crate that `machine_learning` did not already pull, so a default build's dependency graph is unchanged. Nothing is removed; any build that already set `default-features = false` is unaffected, and one that named `features = ["metrics"]` while leaving the default on was compiling `machine_learning` + `neural_network` anyway.
- **Breaking: the two `Solver` enums are renamed to say which solver they select.** `linear_model::Solver` becomes **`LeastSquaresSolver`** and `discriminant_analysis::Solver` becomes **`DiscriminantSolver`**, matching the `SVDSolver` / `EigenSolver` naming the decomposition module already uses. The crate previously defined two distinct enums both called `Solver`, and only LDA's was re-exported at the `machine_learning` and prelude level - so after `use rustyml::prelude::machine_learning::*`, writing `Solver::GradientDescent` resolved to LDA's enum and failed with "no variant named `GradientDescent`". Both are now re-exported under their new, unambiguous names, so `LeastSquaresSolver` no longer needs a fully-qualified path.
- **Breaking: `LinearRegression` is exact OLS by default, and each solver now carries its own settings.** `LeastSquaresSolver` is a payload-carrying enum - `LeastSquaresSolver::Normal` (the default, no fields) and `LeastSquaresSolver::GradientDescent { learning_rate, max_iter, tol }` - so the iteration knobs live on the only strategy that reads them and cannot be handed to the closed form at all. This is the shape `SVDSolver::Randomized(u64)` and `TSNEMethod::BarnesHut { .. }` already had; `LinearRegression` was the last solver-selecting enum whose settings sat loose on the estimator. Consequences: `LinearRegression::new(fit_intercept)` takes one argument and returns `Self` rather than `Result`, because there is nothing left for it to validate; `with_solver` returns `Result` instead, validating the variant's payload (the same shape `with_regularization` already has for `RegularizationType::L1`); `LinearRegression::default()` is now literally `new(true)`, so the two constructors can no longer build different algorithms; and `get_learning_rate` / `get_max_iterations` / `get_tolerance` are removed - read them back by matching on `get_solver()`. The default being `LeastSquaresSolver::Normal` means the analogue of Python's `LinearRegression()` reproduces scikit-learn's coefficients (verified to ~1e-15) instead of an approximation that could also fail outright with `Error::NonFinite` on unscaled data. Migration: `LinearRegression::new(fi, lr, mi, tol)?` becomes `LinearRegression::new(fi).with_solver(LeastSquaresSolver::GradientDescent { learning_rate: lr, max_iter: mi, tol })?`, and dropping the `with_solver` call altogether gives you OLS. **The serialized layout changed - three loose fields folded into the solver payload - so re-fit and re-save any persisted models.**
- **Breaking: `RegularizationType::L1` now produces exact zeros.** `LinearRegression` and `LogisticRegression` apply L1 through a proximal (soft-thresholding / ISTA) step after the gradient step instead of adding `alpha * sign(w)` to the gradient. A sub-gradient step can only approach zero asymptotically, so "Lasso" delivered no sparsity and no feature selection however long it ran; an unsupported coefficient now lands on exactly `0.0`. The intercept stays unpenalized. `LinearSVC`'s L1 is unchanged for now — its minibatch schedule needs its own treatment.
- **`RegularizationType` documents what `alpha` means and how to convert one from scikit-learn.** Both variants measure an undivided penalty against a *mean* data term, which is exactly scikit-learn's `SGDRegressor`/`SGDClassifier` objective, so `alpha` transfers 1:1 from those. Against the closed-form estimators: `Lasso(alpha=a)` → `L1(a)`, `Ridge(alpha=a)` → `L2(a / n)` (scikit-learn's `Ridge` does not divide its data term by `n`; verified to 6.7e-16), `LogisticRegression(C=c)` → `alpha = 1 / (c * n)`. No numbers changed.
- **Breaking: `ConfusionMatrix::new` requires hard `{0.0, 1.0}` labels and no longer thresholds anything.** It binarized *both* arguments at a hardcoded `0.5`, so a probabilistic ground truth was silently coerced, an unbounded decision-function score was cut at a meaningless point, and `NaN` counted as negative. scikit-learn's `confusion_matrix` likewise requires hard labels. Threshold your scores before the call; for a different label pair — the `-1`/`+1` a margin classifier emits — use the new `ConfusionMatrix::new_with_labels(y_true, y_pred, negative_label, positive_label)`, the binary form of scikit-learn's `labels=[neg, pos]`. `new` also takes independent storage types for its two arguments now, so `ConfusionMatrix::new(&y_test, &model.predict(&x)?.view())` compiles.
- **Breaking: `IsolationForest` scores adopt scikit-learn's sign, and `decision_function` joins the API.** `score_samples` returns `-(2^(-E[h(x)] / c(n)))` — values in `[-1, 0)` where **lower** is more anomalous — instead of the raw `[0, 1]` score where higher was. Flip every comparison that ranks or thresholds scores. The per-sample slice form `anomaly_score` is renamed **`score_sample`** and flipped with it, so the estimator has one score orientation. New `decision_function(&x)` returns `score_samples(&x) - offset`, and `predict` is exactly its sign (`-1` where strictly negative, `+1` otherwise) — so a sample landing *exactly* on the cutoff is now an inlier, which changes the labels of a completely undifferentiated dataset. `get_offset()` under `Contamination::Auto` returns `-0.5`; `Contamination::Fraction(c)` now resolves through NumPy's linear-interpolating `percentile(scores, 100 * c)`, matching scikit-learn's `offset_` numerically rather than merely flagging the same count. **The serialized `offset` changes sign — re-save any persisted forests.**
- **Breaking: `LDA::transform` centers by the training mean and keeps the whitened axis scale.** scikit-learn computes `(X - xbar_) @ scalings_`; RustyML skipped the centering and rescaled each discriminant axis to unit L2 norm, so projected coordinates disagreed with a ported pipeline on both offset and scale. `fit` now stores the overall training mean (new `get_overall_mean`, scikit-learn's `xbar_`) and `transform` subtracts it, and the axes are left at the scale the whitening produced — which is what makes the projected data have unit within-class covariance. The projection is no longer a unit vector; the degenerate-axis guard became relative to compensate. **The serialized format gained a field — re-save any persisted models.**
- **Breaking: `MeanShift` uses scikit-learn's flat kernel and merge rule, and matches it element for element.** Three changes together: the shift step uses a **flat** kernel (the plain mean of the points within one bandwidth) instead of a Gaussian one weighted over every sample; converged modes are merged by scikit-learn's intensity-ordered greedy suppression instead of a running average in seed order, so a kept center *is* a density mode rather than a blend; and unassigned points under `cluster_all = false` are labelled `-1` instead of `n_clusters`, which used to read as a real extra cluster. On the reference 10-point set the cluster centers, their numbering, and every label now equal scikit-learn 1.9.0's exactly. The Gaussian kernel is **removed** rather than kept as an option: it has no scikit-learn counterpart, so nothing validates it, and the new intensity-ordered merge ranks modes by a window's point count — a quantity the Gaussian weighting does not produce. `MeanShift` now has one kernel and no `kernel` parameter. **A model fitted before this change would reload and then cluster differently — re-fit and re-save.**
- **Breaking: `estimate_bandwidth` returns a local-density statistic, matching scikit-learn.** It returned a quantile of the *whole* pairwise-distance distribution — a global spread measure that runs far larger than a bandwidth should on clustered data, collapsing everything into one cluster. It now averages each point's distance to its `(k - 1)`-th nearest neighbour with `k = floor(n * quantile)`, reproducing scikit-learn's off-by-one (its neighbour query counts the query point itself). Agrees with scikit-learn 1.9.0 to 1e-14. A neighbourhood of one now yields `0.0`, as it does in scikit-learn.
- **Breaking: cluster labels are `isize` everywhere, and `-1` is the noise value.** `KMeans` and `MeanShift` join `DBSCAN` in returning `Array1<isize>` from `predict`/`fit_predict` (and `get_labels`), and all ten clustering metrics in `metrics::clustering` take `Data<Elem = isize>` instead of `usize`. scikit-learn's `labels_` is likewise a signed integer array. This is what lets any clustering estimator's output feed any clustering metric — previously `DBSCAN`'s `-1` noise labels did not type-check against a single one of them, and giving `MeanShift` its `-1` would have broken it the same way.
- **`KMeans` gains `n_init`, defaulting to 10 restarts.** Each restart re-seeds k-means++ from a deterministically derived sub-seed, runs the full Lloyd iteration, and the lowest-inertia run wins, so one unlucky seeding no longer decides the result. scikit-learn's `n_init='auto'` is 1 for k-means++, but that rests on its *greedy* seeding (`2 + ln(k)` candidates per center); RustyML's is plain k-means++, which is the higher-variance seeding restarts exist to compensate for. Pass `with_n_init(1)` for literal scikit-learn parity. A seeded fit stays reproducible; fitted results move for every seeded model, and the serialized format gained a field.
- **`roc_curve`'s origin threshold is `f64::INFINITY`** instead of `max_score + 1.0`, matching scikit-learn and keeping the sentinel distinguishable from the top real threshold for large scores (`1e17 + 1.0 == 1e17`). `roc_curve` still returns the full sweep — scikit-learn's `drop_intermediate=False` — so point counts can differ from its default while the curve and `roc_auc` do not.
- **`IsolationForest`'s inlier/outlier threshold is now fitted from the training scores instead of recomputed per prediction call.** The old `predict_labels(&x, contamination)` took the `ceil(contamination * n)`-th highest score *of the batch it was handed*, so the decision was transductive: a single-row call always returned `-1` (the count clamps to 1), and splitting a test set in half changed its labels. `contamination` is now a builder-set [`Contamination`] rule resolved into a stored `offset` at fit time, so a sample gets the same label whether it is scored alone, in a slice, or in the full batch.
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
