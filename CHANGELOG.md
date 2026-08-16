# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24, summarized per version — each entry lists only the significant changes for that release.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [Unreleased]
### Added
- **7 new `Activation` variants: `LeakyReLU`, `ELU`, `SELU`, `Softplus`, `Softsign`, `HardSigmoid`, and `Exponential`.** Each one also ships as a thin standalone layer of the same name in `neural_network::layers::activation`. `LeakyReLU::new(negative_slope)` and `ELU::new(alpha)` return a `Result`, the other 5 take no argument, and the 2 defaults are `0.3` and `1.0`.
- **New `Activation::validate`, and all 9 trainable layer constructors call it** (`Dense`, `Conv1D`, `Conv2D`, `Conv3D`, `DepthwiseConv2D`, `SeparableConv2D`, `SimpleRNN`, `LSTM`, and `GRU`). A `negative_slope` or `alpha` that is not finite and greater than 0 is now `Error::InvalidParameter` at construction, not at the first forward pass. The bound is strict because the backward pass separates the 2 branches by the sign of the activated output. Use `Activation::ReLU` for a slope of 0.
- **`LeakyReLU` takes the positive branch at `x >= 0`, while `ELU` and `SELU` take it at `x > 0`.** The derivative at exactly 0 is therefore 1 for `LeakyReLU`, `alpha` for `ELU`, and `scale * alpha` for `SELU`. The tests pin the forward and derivative tables as reference values.
- **6 new border layers: `ZeroPadding1D/2D/3D` and `Cropping1D/2D/3D`.** A zero-padding layer adds zero positions at the ends of the spatial axes, and a cropping layer removes them. Both leave the batch axis and the channel axis untouched, and neither holds a parameter. Each half is the backward pass of the other half, so a pad and a crop with the same amounts cancel. A `Cropping*` amount that leaves a spatial axis with no positions is `Error::InvalidInput` at forward time.
- **New `Border1D`, `Border2D`, and `Border3D` argument types.** Every border constructor takes `impl Into<..>`, and it returns the layer directly with no `Result`. A border amount is a `usize`, so no value is invalid at construction. An integer sets every end. A tuple sets 1 equal amount per axis. A tuple of `(before, after)` pairs names every end on its own.
  - Read the 2D and 3D tuple form with care. `ZeroPadding2D::new((1, 2))` gives 1 row at the top and the bottom, plus 2 columns at the left and the right. It does not give 1 row at the top and 2 at the bottom.
- **New `Identity` layer.** It passes its input through unchanged at every rank, and its backward pass returns the gradient it receives. `Identity::new()` takes no argument and returns the layer directly with no `Result`. Its use is a placeholder, for a builder that must return a layer when the choice is "no operation". It also fits a stack whose depth is a runtime value. It copies rather than borrows, because a layer returns an owned tensor.
- **New `Permute` layer.** `dims` names the new order of the axes after the batch axis, counting from 1 and never including the batch axis itself. The entries must be a permutation of `1..=dims.len()`, and the input rank must be `dims.len() + 1`. Unlike `Reshape`, a permute moves data rather than relabeling it, so the layer copies. The output is always in C order, so a consumer can read it as 1 contiguous slice. `Permute::new(vec![])` returns `Error::InvalidParameter`, because an empty `dims` reorders nothing.
- **New `RepeatVector` layer.** It turns a `[batch, features]` input into `[batch, n, features]`, with the same vector at every step. The backward pass sums over the step axis. This bridges 1 recurrent layer to the next, because a recurrent layer here returns a rank-2 state and needs a rank-3 input.
- **New `Reshape` layer.** `target_shape` names the axes after the batch axis, which stays free, so 1 layer instance serves every batch size. At most 1 entry may be `-1`, and that axis takes the extent that makes the element count match. `Reshape::new(vec![-1])` is exactly `Flatten`, but unlike `Flatten` the constructor takes no input shape. The constructor rejects a `0` entry at construction, not at the first forward pass.
- **New `UnitNormalization` layer.** It scales each group of elements to an L2 norm of 1. The new `UnitNormalizationAxis` picks the axes the norm reduces over. `Default` names the last axis, `Custom(a)` names 1 axis, and `Multiple(axes)` names a joint norm over several axes. It holds no parameter and learns no `gamma` or `beta`. It behaves the same in training and in inference, which makes it the first layer in `regularization` that is not mode-dependent.
  - The scale is capped at `1e12`, so an all-zero group comes back all zero instead of dividing by zero. `UnitNormalization::new` returns `Error::InvalidParameter` for an empty axis list or a duplicate axis, at construction.
- **3 new upsampling layers: `UpSampling1D`, `UpSampling2D`, and `UpSampling3D`.** Each one multiplies the extent of every spatial axis by its factor, leaves the batch axis and the channel axis untouched, and holds no parameter. The family is the decoder counterpart of the pooling family, so an autoencoder and a segmentation decoder become expressible without a transposed convolution. New `Factor2D` and `Factor3D` argument types take an integer for a shared factor or a tuple for 1 factor per axis. `UpSampling1D::new` takes a plain integer, because a rank-3 input has 1 spatial axis. All 3 constructors return a `Result` and reject a factor of 0 with `Error::InvalidParameter`.
- **New `Interpolation` argument on `UpSampling2D`, with 5 modes: `Nearest`, `Bilinear`, `Bicubic`, `Lanczos3`, and `Lanczos5`.** `Nearest` is the default, and it repeats each pixel into a block. The other 4 resample with a separable kernel, so a new pixel is a weighted sum of its neighbors along each axis. The weights of 1 output pixel always add up to 1, including at an edge where part of the kernel falls outside the image. A constant image therefore stays constant.
  - Only `UpSampling2D` takes this argument: `UpSampling1D` and `UpSampling3D` repeat and take no interpolation. Verified over 250 shape, factor, and mode combinations and 51025 scalars, forward and backward.
- **New `tuning::upsampling::set_parallel_min_ops` gate, defaulting to 2,000,000.** It gates 1 axis pass of an upsampling layer on `destination elements * taps`, not on the element count. The taps per output position run from 1 for the repeat mode up to 11 for `Lanczos5`. The default is the point at which every mode at least breaks even against its own serial path. Moving the gate never changes a result, because the taps are added in a fixed order.
- **`UpSampling2D` computes its resample weight table in `f64`, not `f32`.** The Lanczos kernels have large lobes that cancel, so a `f32` table loses precision before it is ever applied. This layer rounds to `f32` once, at the end. Measured against an independent `f64` reference over the same 250 cases, the worst deviation of `Lanczos5` is `2.5e-7`. Nearest and bilinear agree to the last bit that `f32` can hold.
- **`UnitNormalization` returns a real gradient at a group the cap took over.** The derivative of the reciprocal square root itself overflows at such a group, so differentiating through it directly would return `NaN` for every element. This layer returns the derivative of the function it computed, which is the cap. The 2 approaches agree everywhere else, checked over 106 shape and axis combinations and 57648 scalars.

### Fixed
- **`Activation::Softmax` no longer fails on an input that is not in C order.** It called `to_owned` and then `into_shape_with_order`, and `to_owned` keeps the strides of such an input, so the reshape refused it. A hand-transposed tensor, such as `x.t().to_owned()`, therefore returned `Error::Computation` from `Softmax` while every other layer accepted it. The layer now settles the layout first, at the same 1 copy it already made. Results for an input already in C order are unchanged.

### Changed
- **Breaking: `Activation` no longer derives `Eq`,** because `LeakyReLU` and `ELU` carry an `f32` parameter. It still derives `PartialEq`. Only code that uses `Activation` as a `HashMap` key or in a `HashSet` needs an edit.
- **`Flatten` caches only the input shape, not the input tensor.** Its backward pass never read the values, since a flatten moves no data, so every forward pass was copying the whole activation for nothing. `Flatten::forward` now does 1 pass over the activation instead of 2, and the layer holds 1 fewer copy of it. The public API and every result are unchanged.

## [v0.14.0] - 2026-07-29
### Added
- **New `utils::scaler` module: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and `Normalizer`** — the scikit-learn transformer family, with a shared `fit` / `transform` / `inverse_transform` contract, the `Fit`/`Transform`/`FitTransform` traits, `partial_fit` wherever the statistics merge exactly, and `save_to_path` persistence. `StandardScaler` reproduces `standardize` bit-for-bit.
- **`fit` and `fit_with_batches` return a `History`** carrying one loss entry per epoch whether or not `show_progress` is on, so early stopping, learning-rate schedules, and best-checkpoint loops become expressible as user code.
- **`Sequential::evaluate` scores a model without training it** (Keras' `evaluate`): one inference-mode forward pass through the compiled loss, updating nothing and drawing from no RNG.
- **`Sequential::train_batch` is public** (Keras' `train_on_batch`), so a caller can own the epoch structure; it now validates its own inputs.
- **The learning rate is readable as well as writable** through the new `Optimizer::learning_rate` and `Sequential::learning_rate()`, so a schedule is a read-scale-write instead of a shadow copy that goes stale.

### Fixed
- **The per-epoch loss `fit_with_batches` reports no longer over-weights a short trailing batch.** It is now `sum(loss_i * n_i) / n_samples` as in Keras, rather than a mean over the batch count.
- **A rank-0 input tensor no longer panics `fit`, `fit_with_batches`, `train_batch`, or `evaluate`;** it is rejected with `Error::InvalidInput`.
- **`r2_score` and `explained_variance_score` no longer score `1.0` for a genuinely varying target.** An absolute `1e-10` threshold on the sum of squares mistook any low-spread target for a constant one; constancy is now decided from the values themselves.
- **`precision_recall_curve` no longer appends its closing point to the wrong end of the curve,** which left recall monotone in neither direction. The points now run in scikit-learn's order.
- **`KMeans` no longer leaves `labels_` and `inertia_` describing different centroids than `cluster_centers_`** after a fit that stopped at `max_iter`; a final assignment pass now runs on that exit path.
- **`CategoricalCrossEntropy` no longer computes the wrong function above rank 2.** It divided by the batch axis alone and, under `from_logits`, softmaxed across every (position, class) pair at once. The last axis is now the class axis, every leading axis an independent prediction site, and the divisor their product — Keras' `sum_over_batch_size`. **Rank-2 input is bit-for-bit unchanged.**
- **`fit_with_batches` no longer fails on its first mini-batch for a model containing a dropout, noise, or normalization layer.** The shape check compared axis 0 against the shape declared at construction, so only `batch_size == n_samples` survived; the batch axis is no longer compared.
- **`DepthwiseConv2D` and `SeparableConv2D` no longer omit the channel count from `fan_in`,** which widened the Glorot bound by roughly `sqrt(channels)`. **Weights drawn from a given seed change.**

### Changed
- **Breaking: gradient clipping is renamed `global_clipnorm`** (`Optimizer::clip_norm`, and `with_clip_norm` → `with_global_clipnorm`), because the crate has always clipped by *global* norm while Keras' `clipnorm` is the per-variable knob. No behaviour changed.
- **Breaking: `fit` and `fit_with_batches` return `Result<History, Error>`** instead of `Result<&mut Self, Error>`. Statement-position callers still compile; only code chaining another method off `fit` needs an edit.
- **Breaking: `Optimizer` requires `learning_rate(&self) -> f32`, and `set_learning_rate` lost its default no-op body,** which had let a custom optimizer swallow every schedule call in silence. Only out-of-crate impls need an edit.
- **Breaking: both categorical cross-entropies renormalize `y_pred` along the class axis before clipping, matching Keras.** The loss is unchanged for an already-normalized head but the gradient is not, since the divisor is differentiated; `Softmax` + `CategoricalCrossEntropy::new(false)` still trains bit-for-bit as before. `SparseCategoricalCrossEntropy`'s probability gradient becomes dense.
- **Breaking: `RMSprop` and `AdaGrad` move `epsilon` inside the square root,** matching Keras, while `Adam` keeps it outside as Keras' `Adam` does. **Retune it rather than porting your value across** — roughly `eps_inside = eps_outside²`.
- **`Adam` keeps coupled L2 weight decay as a deliberate, now-documented divergence from Keras 3,** matching `torch.optim.Adam` / `AdamW` instead. No code changed.
- **Breaking: `GRU`'s gate order and update-gate convention now match Keras:** the fused tensors pack as `[z | r | h]`, and `h_t = z_t * h_{t-1} + (1 - z_t) * n_t`. `set_gate_weights` keeps its argument order and repacks internally. **Re-save any GRU checkpoint** — the tensors change meaning, not shape.
- **Breaking: the whole `neural_network` module is channels-last, matching Keras.** Tensors are `[batch, spatial..., channels]`, kernels `(spatial..., in_channels, filters)`, and every convolution and `Dense` bias drops to rank 1. The conversion is native, so im2col becomes a contiguous copy and the module lost roughly 500 lines. **Migration: permute your input tensors and saved kernels, and re-save every checkpoint.** Recurrent layers, `Dense`, and `LayerNormalization` are unaffected.
  - **`DepthwiseConv2D::new` drops `filters` and gains `with_depth_multiplier`;** the output channel count is `channels * depth_multiplier`.
  - **`GroupNormalization::new` and `InstanceNormalization::new` drop `channel_axis`,** which had been implemented by the very permute the new layout avoids.
  - **`Activation::Softmax` on a convolution now normalizes over channels** rather than image width, so such a layer computes something different — and correct.
- **Breaking: saved `Sequential` models carry a magic tag and format version,** validated before anything else is decoded and reported as the new `IoError::UnsupportedModelFormat`. The old layer-count and weight-extent checks can be satisfied by coincidence across an incompatible release. **Every existing `.bin` checkpoint must be re-saved.**
- **The default feature set is now `full`, so `cargo add rustyml` gives the whole crate.** The old default omitted `utils` and `metrics`, so `train_test_split` and `accuracy` did not compile. No dependency is added.
- **Breaking: the two `Solver` enums are renamed** to **`LeastSquaresSolver`** and **`DiscriminantSolver`**. Only LDA's was re-exported, so after a prelude glob import `Solver::GradientDescent` resolved to the wrong enum; both are now re-exported.
- **Breaking: `LinearRegression` is exact OLS by default, and its iteration knobs move into the solver payload** (`LeastSquaresSolver::GradientDescent { learning_rate, max_iter, tol }`). `new(fit_intercept)` takes one argument and returns `Self`, `with_solver` returns `Result`, and `get_learning_rate` / `get_max_iterations` / `get_tolerance` are removed — match on `get_solver()`. **The serialized layout changed — re-fit and re-save.**
- **Breaking: `RegularizationType::L1` now produces exact zeros** through a proximal (soft-thresholding) step, so Lasso finally delivers sparsity. The intercept stays unpenalized; `LinearSVC`'s L1 is unchanged for now.
- **`RegularizationType` documents how to convert `alpha` from scikit-learn:** 1:1 from `SGDRegressor`/`SGDClassifier`, `Ridge(alpha=a)` → `L2(a / n)`, `LogisticRegression(C=c)` → `alpha = 1 / (c * n)`. No numbers changed.
- **Breaking: `ConfusionMatrix::new` requires hard `{0.0, 1.0}` labels instead of thresholding both arguments at a hardcoded `0.5`,** as scikit-learn's `confusion_matrix` does. Threshold your scores first, or use the new `new_with_labels(y_true, y_pred, negative_label, positive_label)` for another label pair. Its arguments now take independent storage types.
- **Breaking: `IsolationForest` adopts scikit-learn's scoring and prediction API.** `predict` returns `-1`/`+1` labels, the old score-returning `predict` becomes `score_samples` with scikit-learn's sign (**lower** is more anomalous), `anomaly_score` becomes `score_sample`, `predict_labels` is removed, and `decision_function` is new. Contamination is a builder-set `Contamination` rule resolved into a stored `offset` at fit time, so a label no longer depends on the batch the sample was scored in. Flip every comparison that ranks or thresholds scores. **Re-save any persisted forests.**
- **Breaking: `LDA::transform` centers by the training mean and keeps the whitened axis scale,** computing scikit-learn's `(X - xbar_) @ scalings_`; the mean is exposed as `get_overall_mean`. **The format gained a field — re-save any persisted models.**
- **Breaking: `MeanShift` matches scikit-learn element for element:** a flat kernel for the shift step, intensity-ordered greedy merging of converged modes, and `-1` for unassigned points under `cluster_all = false`. The Gaussian kernel is **removed** — nothing validated it, and the new merge ranks modes by a point count it does not produce. **Re-fit and re-save.**
- **Breaking: `estimate_bandwidth` returns a local-density statistic,** the mean distance to each point's `(k - 1)`-th nearest neighbour, instead of a quantile of the whole pairwise-distance distribution, which ran far too large on clustered data.
- **Breaking: cluster labels are `isize` everywhere, and `-1` is the noise value.** `KMeans` and `MeanShift` join `DBSCAN`, and all ten `metrics::clustering` functions take `Data<Elem = isize>`, so any clustering estimator's output now feeds any clustering metric.
- **`KMeans` gains `n_init`, defaulting to 10 restarts** with the lowest-inertia run winning, so one unlucky k-means++ seeding no longer decides the result. Pass `with_n_init(1)` for literal scikit-learn parity. Fitted results move for seeded models, and the format gained a field.
- **`roc_curve`'s origin threshold is `f64::INFINITY`** instead of `max_score + 1.0`, matching scikit-learn and staying distinguishable from the top real threshold at large scores.
- **Breaking: the estimator traits move from `machine_learning::traits` to the crate root as `rustyml::traits`,** since `utils::StandardScaler` implements three of them and `utils` does not depend on `machine_learning`. Both preludes still resolve; only the old module path disappears.
- **Breaking: the matrix-multiply backend switches from the `gemm` crate to [`gemmkit`](https://crates.io/crates/gemmkit) 0.1.2,** through its zero-copy `gemmkit-ndarray` adapter. Every hand-rolled serial-vs-rayon gate and row split is deleted in favour of the backend's own scheduling.
  - `tuning::matmul` loses its GEMM/GEMV FLOPs thresholds and `colpar_min_cols_per_thread`; tune through `GEMMKIT_*` environment variables, the new `tuning::matmul::backend` re-export, or a `gemmkit-tune` profile. `chunk_elems` and `cache_resident_max_bytes` are unchanged.
  - The `gemm_calibrate` bench is removed, since it calibrated the deleted gates.
  - Results no longer depend on worker count, and GEMV is bit-identical at any worker count.
  - The repaired fused epilogue makes `Dense::forward` 32–49% faster, an MLP training epoch 17% faster, and the matvec-bound estimators 13–21% faster; `PCA::fit_transform` regresses 1–2%.
- **The hot layers fuse their bias adds (and ReLU) into the GEMM epilogue:** `Dense`, the conv forward blocks, the SimpleRNN/LSTM/GRU timestep loops, and `LDA::decision_scores` write their pre-activations in one pass. Bitwise-identical to the unfused form, except that a `NaN` pre-activation under the fused `Relu` now maps to `0.0`.
- **Breaking: `SVC` takes and returns `{0.0, 1.0}` labels instead of `±1.0`,** matching the other classifiers; the SMO dual's ±1 encoding is now internal to `fit`. Passing `±1.0` is `Error::InvalidInput`, and the serialized format is unchanged.
- **Breaking: `LogisticRegression::predict` and `fit_predict` return `Array1<f64>` labels** instead of `Array1<i32>`, so predictions round-trip into `fit` and the `f64` metrics without a conversion. `LDA` is unchanged.
- **Every supervised estimator accepts `x` and `y` with different storage types,** so an owned matrix pairs with a borrowed label view. `DecisionTree::predict` also drops an unnecessary `Send + Sync` bound. Purely relaxations.
- **Breaking: `PCA` and `KernelPCA` move to `machine_learning::decomposition` and `TSNE` to `machine_learning::manifold`,** mirroring scikit-learn, so they are gated by `machine_learning` rather than `utils`. They also implement `Transform` / `FitTransform`.
- **Removed the `nalgebra` runtime dependency.** A crate-internal `machine_learning::linalg` module provides `symmetric_eigen`, a one-sided Jacobi `svd` (with `solve` / `pseudo_inverse`), and a Gram-Schmidt `qr_q` over `ndarray` arrays; `nalgebra` drops to a dev-dependency that only cross-checks them in tests.

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
