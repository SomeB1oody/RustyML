# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [Unreleased] - 2026-06-24 (UTC-7)
### Changed
- **Removed the `nalgebra` runtime dependency; SVD, symmetric eigendecomposition, and QR are now hand-rolled in pure Rust.** A new crate-internal `math::decomposition` module (gated on `machine_learning`/`utils`) provides `symmetric_eigen` (Householder `tred2` + implicit-shift QL `tql2`), a one-sided Jacobi `svd` with `solve` / `pseudo_inverse`, and a modified-Gram-Schmidt `qr_q`, all operating directly on `ndarray` arrays. The PCA (`Full` / `Randomized`), KernelPCA (`Dense`), LDA, and ridge/linear-regression solvers now call these instead of `nalgebra`; `PCA`'s `SVDSolver::Full` routes through the covariance eigendecomposition, reusing `symmetric_eigen`. `nalgebra` moves from an optional runtime dependency to a `dev-dependency` used only to cross-check the new routines in tests, dropping `nalgebra` + `simba` + `safe_arch` + `wide` from the normal dependency tree. Numerically equivalent up to the usual sign/rounding freedom; all existing solver tests pass unchanged.

## [v0.13.0] - 2026-06-23 (UTC-7)
### Added
- **`TSNE` gains `min_grad_norm` early stopping after early exaggeration, for scikit-learn parity.** New builder `TSNE::with_min_grad_norm` and getter `get_min_grad_norm` set a gradient infinity-norm threshold (default `1e-7`, `DEFAULT_MIN_GRAD_NORM`); past the exaggeration phase, optimization stops once the largest absolute gradient drops below it. Pass `0.0` to disable and always run the full `n_iter` (the previous behavior). The field's `#[serde(default = "default_min_grad_norm")]` makes pre-field serialized instances deserialize with `1e-7`; the `new` signature is unchanged.
- New `benches/benchmarks/eigensolver.rs` criterion suite (bench `eigensolver`, `required-features = ["utils"]`) isolating `utils::linalg::top_eigenpairs_power_iteration` via its two callers — PCA's `SVDSolver::PowerIteration` and KernelPCA's `EigenSolver::PowerIteration` — sized so the deflated iterative inner loop, not the one-off GEMM, dominates.
- New `benches/benchmarks/silhouette.rs` criterion suite (`required-features = ["metrics"]`) isolating the `O(n^2 * d)` pairwise-distance fill of `silhouette_score`, sizing every config above the 262,144 parallel gate and benchmarking `Euclidean`, `Manhattan`, and `Minkowski(3)` separately.

### Changed
- **Breaking: `standardize` drops its `epsilon` parameter and now matches `StandardScaler` exactly.** The signature changes from `standardize(data, axis, epsilon)` to `standardize(&data, axis)` — drop the third argument (e.g. `standardize(&data, StandardizationAxis::Column, 1e-8)` becomes `standardize(&data, StandardizationAxis::Column)`). Constant lanes are detected via scikit-learn's `_is_constant_feature` rule and divided by `1.0` (centered values stay zero); non-constant lanes divide by raw `sqrt(variance)`. The non-positive/non-finite epsilon `Error::InvalidParameter` case is gone. Verified against scikit-learn 1.9.0.
- **`math::sigmoid` no longer clamps its input to `[-500, 500]`.** It is now plainly `1.0 / (1.0 + (-z).exp())`, since IEEE-754 already saturates at the extremes, matching scipy/PyTorch/TensorFlow. All inputs produce the same saturated values as before.
- **The softmax layer forward pass drops its `1e-8` denominator floor.** The `max(SOFTMAX_EPSILON)` and `SOFTMAX_EPSILON` constant are removed from `softmax_forward` (used by `Activation::Softmax`); the row-max shift already guarantees the sum is `>= 1.0`, so the floor was dead code that masked NaN rows. NaN now propagates through a NaN row; all finite rows are numerically unchanged.
- `metrics::log_loss` and `mean_absolute_percentage_error` raise their stability epsilon from `1e-15` to `f64::EPSILON` (scikit-learn's `finfo(float64).eps`) — the floor on the per-row probability sum and `clamp` bound in the former, the floor on `|y_true|` in the latter.
- `utils::normalize` now leaves a (near-)zero lane unchanged instead of zeroing it. The threshold `NORM_ZERO_THRESHOLD` (`1e-15`) is renamed `NORM_CONSTANT_THRESHOLD` (`10 * f64::EPSILON`) and the zeroing branches are removed: a lane below it keeps its values (norm treated as `1.0`), matching scikit-learn's `normalize`. Lanes previously zeroed in the `1e-15`..`10*f64::EPSILON` band now retain their original values.
- t-SNE aligns its numerical constants to scikit-learn: `MIN_Q` (lower bound on `q_ij` and Barnes-Hut normalizer `Z`) changes `1e-12` -> `f64::EPSILON`, and the perplexity binary-search `sum_p` floor changes `1e-12` -> `1e-8` (`EPSILON_DBL`).
- **Power-iteration eigensolver now does one matvec per step and deflates the Hotelling term in place.** `dominant_eigenpair` estimates `lambda` from the single matvec already computed for the next step, halving the per-iteration GEMV count. Hotelling deflation in `top_eigenpairs_power_iteration` drops the dense `n x n` outer product for a new in-place `deflate_rank_one` helper (`row_i -= value * v_i * v`), parallel above the `cheap_map_f64_parallel_threshold` gate and serial below. Numerically equivalent up to rounding; still deterministic. Feeds the PCA and KernelPCA `PowerIteration` solvers.
- **Behavior change: `PCA` now flips principal-axis signs deterministically so all SVD solvers and repeated runs agree on axis orientation.** A new `flip_component_signs` step in `PCA::fit` negates each `components` row whose largest-magnitude loading is negative (ties keep the lowest index). Decided from the component vectors themselves (not `U`), so `SVDSolver::Full`, `Randomized`, and `PowerIteration` return sign-identical components, though not byte-matching scikit-learn's `svd_flip`. Reconstructions are unaffected (scores flip in step). Sign of returned components may flip relative to the previous release.
- **`silhouette_score` now fills its `dist_to_cluster` table over the symmetric upper triangle, evaluating each unordered pair once instead of twice.** A new `pairwise_cluster_distances` helper computes `metric.distance` once per `j > i` and adds it to both `dist_to_cluster[[i, cluster[j]]]` and `dist_to_cluster[[j, cluster[i]]]`, halving the metric calls. Above the `SILHOUETTE_PARALLEL_MIN_ELEMS` (262,144) gate the fill runs in parallel with a fixed bucket grouping, so the result matches the old full scan and stays reproducible across runs on the same machine. ~33% faster for `Euclidean`/`Manhattan`, ~45% for `Minkowski(3)`. The public signature is unchanged.
- Tooling/CI only: the `benches/benchmarks/ml_end_to_end.rs` suite wraps its RBF `gamma` literals in `Gamma::Value(..)` (e.g. `KernelType::RBF { gamma: Gamma::Value(0.5) }`), catching the `--all-targets`-only bench up to the `Gamma` enum so `cargo clippy --all-features --all-targets` compiles again; also rustfmt-wraps the new `pairwise_cluster_distances` tests in `src/metrics/clustering.rs`. No library behavior change.

### Fixed
- **The lightweight `metrics`-only build now compiles.** `Gamma::resolve` returns `Result<f64, crate::error::Error>`, but `crate::error` only compiles under `machine_learning`/`utils`, so a `metrics`-only build failed on the missing module. `Gamma::resolve` and its four `resolve`/`resolve_gamma` tests are now `#[cfg(any(feature = "machine_learning", feature = "utils"))]`, matching where `crate::error` exists.

## [v0.13.0] - 2026-06-22 (UTC-7)
### Added
- **Breaking: kernel `gamma` is now a `Gamma` enum supporting data-dependent rules instead of a bare `f64`.** New public `Gamma` type (`machine_learning::Gamma`, `utils::Gamma`) with variants `Gamma::Value(f64)`, `Gamma::Scale` (scikit-learn `'scale'`: `1 / (n_features * X.var())`) and `Gamma::Auto` (`'auto'`: `1 / n_features`), resolved to a concrete value at fit time. The `gamma` field of `KernelType::Poly`, `RBF`, and `Sigmoid` changes from `f64` to `Gamma`, so every construction site must switch (e.g. `KernelType::RBF { gamma: 0.5 }` -> `KernelType::RBF { gamma: Gamma::Value(0.5) }`); `SVC` and `KernelPCA` resolve `Scale`/`Auto` against training data at `fit`. `Gamma::Scale` errors on zero-variance data.
- **`LDA` gains `decision_function` and `predict_proba`.** `decision_function` returns the `(n_samples, n_classes)` matrix of per-class discriminant scores (columns aligned to `get_classes`); `predict_proba` returns posterior probabilities as a numerically-stable row-wise softmax of those scores. `predict` shares the same `decision_scores` core, so its labels are unchanged.
- **`LinearRegression` gains a closed-form normal-equation solver and a `score` (R²) method.** New `linear_model::Solver` enum (`Solver::GradientDescent`, default, and `Solver::Normal`) via the `with_solver` builder; `Solver::Normal` solves the ridge least-squares system via SVD on `[Xc; sqrt(lambda) I]` (minimum-norm even for collinear/wide data), ignores learning rate/iterations/tolerance, and errors with L1. `score(x, y)` returns `R² = 1 - SS_res / SS_tot`, with the constant-target case matching scikit-learn's `r2_score`. Adds a `get_solver` getter.
- **`IsolationForest::predict_labels(x, contamination)` classifies samples as inlier (`+1`) or outlier (`-1`).** The `ceil(contamination * n_samples)` highest-scoring samples (ties included) are flagged `-1`, the rest `+1`, mirroring scikit-learn's `IsolationForest.predict` except the threshold is taken on the provided batch. `contamination` must be finite and in `(0.0, 0.5]`.
- **`LinearSVC` gains a squared-hinge loss and inverse-scaling learning-rate decay.** New `machine_learning::Loss` enum (`Loss::Hinge`, default, and `Loss::SquaredHinge` = `max(0, 1 - y·f(x))²`) via `with_loss`; new `with_learning_rate_decay(decay)` applies `learning_rate / (1 + learning_rate_decay * t)` per epoch (`0.0` = constant default; negative or non-finite errors). Adds `get_loss` and `get_learning_rate_decay` getters. `SquaredHinge` matches scikit-learn's `LinearSVC` default.

### Changed
- **Breaking: `LDA::get_n_components` returns `Option<usize>` and the default `n_components` is now `None` (auto).** `None` resolves at fit time to `min(n_classes - 1, n_features)`; `LDA::default()` now uses `None` instead of `2`. Callers reading `get_n_components()` must handle `Option<usize>` (previously `usize`).
- **Breaking: the serialized model format for `LinearRegression` and `LinearSVC` changed.** `LinearRegression` gains a `solver` field and `LinearSVC` gains `learning_rate_decay` and `loss` fields, so previously saved models can no longer be loaded — re-save any persisted models. Other estimators are unchanged.
- **KMeans now declares convergence on centroid shift rather than inertia change, matching scikit-learn.** It stops when the summed squared centroid shift is `<= tol * mean(var(X))`, replacing the previous `|prev_inertia - inertia| < tol * prev` test. This can change the iteration count and final centroids for a given `tol`.
- **`DecisionTree` scales the minimum-impurity-decrease threshold by the node's sample fraction.** A split is accepted only when `(N_t / N_total) * impurity_decrease >= min_impurity_decrease`, matching scikit-learn; previously the raw `impurity_decrease` was compared directly, holding smaller nodes to the root's absolute threshold.
- **`LogisticRegression` regularization penalty is no longer divided by the sample count.** The L1/L2 penalty is added as `alpha * R(w)` rather than `alpha * R(w) / n_samples`, in both cost and gradient, matching scikit-learn's SGD convention and keeping `alpha` consistent across sample sizes. The intercept remains unpenalized. This changes fitted coefficients for any regularized model at a fixed `alpha`.

### Fixed
- **`LinearSVC::fit` now rejects labels other than `0.0`/`1.0` instead of silently collapsing more than two classes.** Any other label produces an `Error::InvalidInput` (LinearSVC is binary), whereas previously non-binary labels were silently mishandled.
- **MeanShift keeps the current center on a zero-weight window instead of resetting it to the origin.** When every RBF weight underflows to zero (`weight_sum == 0`), the new `resolve_shifted_center` leaves the center in place rather than teleporting it to the origin, which previously injected a spurious cluster center.
- **`DecisionTree` enforces `min_samples_leaf` during the split search and no longer discards a whole categorical split for one rare branch.** Numeric candidates whose left (`pos`) or right (`n - pos`) child falls below `min_samples_leaf` are now skipped during search, not caught as a post-hoc early stop; a categorical split is retained as long as at least two branches each meet the constraint (previously one undersized branch rejected the entire split).
- **`IsolationForest::predict` handles non-contiguous input rows.** Each row is scored via `row.as_slice()` with a `row.to_vec()` fallback instead of `row.as_slice().unwrap()`, which previously panicked on non-contiguous (e.g. sliced or transposed) matrices. Scores for contiguous input are unchanged.

## [v0.13.0] - 2026-06-20 (UTC-7)
### Changed
- **LinearRegression's L1 sub-gradient now always runs serially.** The optional `cheap_map_f64_parallel_threshold` gate on the `+= alpha * w.signum()` accumulation is removed (it stayed serial at realistic feature counts anyway). No API or numerical change.

## [v0.13.0] - 2026-06-19 (UTC-7)
### Added
- New regression benches for nested-parallelism hotspots: `kmeans_high_k` (`bench_kmeans_fit_high_k`, k=256 over 4096x128) and `lda_fit` (`bench_lda_fit`, few-class vs many-class) in `benches/benchmarks/ml_end_to_end.rs`, plus `conv2d_backward` (forward and forward+backward at batch 16/32) in `benches/benchmarks/nn_end_to_end.rs`.
- New `svc_predict`, `mean_shift_fit`, and `poly_features` cases in `benches/benchmarks/ml_end_to_end.rs`, covering the SVC prediction GEMV, MeanShift's per-seed matvecs, and the small (degree-3) and large (degree-1) polynomial-feature paths.

### Changed
- **Nested GEMM parallelism is now bounded by outer-axis breadth in three hotspots that previously forked rayon inside an already-parallel outer loop.** `conv_backward`, `LDA::fit`'s per-class scatter, and `KMeans` centroid averaging each re-forked the pool for an inner GEMM. The new rule parallelizes the outer axis and forces the inner GEMM serial once the outer count reaches the thread count (`batch >= rayon::current_num_threads()` for conv, `n_classes >= rayon::current_num_threads()` for LDA). Measured: conv backward **~13% faster** at batch=32, LDA many-class fit **~6% faster**, KMeans high-k fit **~6% faster**. Numerically unchanged and still deterministic.
- **Forced-serial GEMMs now run the `gemm` crate's serial SIMD kernel instead of ndarray's `matrixmultiply` `.dot()`, which was ~11-13% slower.** `gemm_internal` is renamed to `gemm_par_auto` (same gated behavior), and new `gemm_par_switch(a, b, parallel)` gives explicit control: `false` runs one serial `gemm` call (`Parallelism::None`), `true` the parallel strategy. Nested callers pass `gemm_par_switch(_, false)` to avoid re-forking. Both are `pub(crate)`; no public API change. Numerically unchanged.
- **`KMeans` centroid averaging drops the nested per-centroid `par_mapv_inplace` for a single outer parallel fan gated on the cheap-map threshold.** The cluster-mean update now uses a serial inner `mapv_inplace` and parallelizes the outer centroid loop only when `n_clusters * n_features >= cheap_map_f64_parallel_threshold()`. KMeans high-k fit **~6% faster**; numerically unchanged.
- In `LDA::fit` the per-class within-class scatter GEMM keys its serial-vs-parallel choice off the class-fan breadth (`n_classes >= threads`) rather than the total-data-size gate; the between-class outer product and small products stay on `gemm_par_auto`. Many-class fits avoid oversubscribing the pool; few-class fits are numerically identical.
- **Leftover ndarray `.dot()` matvecs now run on the `gemm`-crate backend.** The GEMV wrapper is split into `gemv_par_strategy` (row-split, no gate), `gemv_par_switch(a, x, parallel)`, and `gemv_par_auto` (FLOPs-gated above `MatmulElem::gemv_rayon_min_flops`); `gemv_internal` is renamed to `gemv_par_auto` (`pub(crate)`, no signature change). `SVC::predict`'s `decision_values_batch` matvec calls `gemv_par_switch(.., false)`: **svc_predict -5%**. KNN's brute-force projection in `predict_parallel` switches to `gemv_par_switch(.., false)`: **knn_predict -22%**. `MeanShift::fit`'s per-seed matvecs use a `serial_gemv` flag forcing serial when `seeds >= rayon::current_num_threads()`: **mean_shift_fit -47%**. `LinearSVC`, `LogisticRegression`, `LinearRegression`, `LDA`'s LSQR, and the `utils::linalg` eigensolvers are likewise rerouted to `gemv_par_auto`. Numerically unchanged.
- **`generate_polynomial_features` now gates its per-sample parallel maps on `cheap_map_f64_parallel_threshold` instead of always forking rayon.** Since the monomial map fires once per output column, small inputs paid fork-join overhead repeatedly. Both maps run serial below the gate (a `[2000, 12]` degree-3 expansion is **-94.5%**), while the large first-order path stays parallel above it. Numerically unchanged.
- **KNN's `predict_parallel` label decode now runs serially.** The final index-to-label step switches from `into_par_iter()` to `into_iter()` since the per-query clone never beats fork overhead: **knn_predict -11.6%**. Numerically unchanged.
- The `gemm_calibrate` bench's GEMV section now measures the real path. It previously compared serial against column parallelism (a no-op at `n == 1`); it now times `gemv_par_switch(false)` vs `gemv_par_switch(true)` over an expanded `m`-sweep, plus an f64 block-count cap sweep. Both investigations landed on **no code change**: the flop gate is well-placed (a shape-aware gate would regress logistic by ~31%) and the bandwidth knee is shape-dependent. Calibration tooling only; no library code changes.

## [v0.13.0] - 2026-06-18 (UTC-7)
### Added
- **New public `tuning` module: the crate's parallel/serial gate thresholds are now overridable at runtime.** `rustyml::tuning` is a flat `set_*`/`get_*` facade grouped into submodules by subsystem — `matmul`, `elementwise`, `reduction`, `tree`, `conv`, `pool`, `norm`, and `metrics` — letting a program retune the serial-vs-rayon (and GEMM strategy) crossovers per machine without recompiling. Each former calibrated gate is now an `AtomicUsize` defaulting to its old value, so **defaults and numerical results are unchanged**: gates only select an execution strategy, and results stay reproducible across runs on the same machine. Feature-gated to `any(machine_learning, neural_network, utils, metrics, math)`. Examples: `tuning::matmul::set_gemm_min_flops_f32(4_000_000)`, `tuning::matmul::get_chunk_elems()`.

### Changed
- **`math::matmul`'s GEMM dispatch now routes column-starved shapes to the rayon row-split instead of the `gemm` crate's column parallelism.** Above `MatmulElem::GEMM_RAYON_MIN_FLOPS`, `gemm_internal` now splits rows whenever `m >= n` *and* `n < threads * GEMM_COLPAR_MIN_COLS_PER_THREAD` (a new internal constant `= 16`), since the backend only parallelizes over output columns `n` and starves on thin/medium-tall outputs. Wide outputs (`n > m`) still go to the `gemm` crate. Recovers `mlp_fit_epoch` to v0.12.0 parity (was 0.86x) and speeds the small/thin kernel and PCA/SVC, with no regressions on large square/wide GEMMs. The `gemv_internal` path is unchanged. Results stay reproducible across runs on the same machine (not necessarily bit-for-bit).
- The `benches/calibrations/gemm_calibrate.rs` tool gains a `grow_us` row-split column, a per-row `best` marker over `{gnone, grow, gpar}`, and new threshold-deriving shapes: the three ~33.5M-flop MLP training-loop GEMMs (`mlp_fwd_512x256x128`, `mlp_dx_512x128x256`, `mlp_dw_256x512x128`) and a medium-band ladder (`med_512x512x128`, `med_512x512x256`), replacing the old `mlp_512x256x128` shape.
- `MatmulElem`'s per-dtype FLOPs crossovers move from associated consts to runtime-gate methods: `GEMM_RAYON_MIN_FLOPS` -> `gemm_rayon_min_flops()` and `GEMV_RAYON_MIN_FLOPS` -> `gemv_rayon_min_flops()`. The GEMM crossover keeps **separate f32 and f64 thresholds** (`gemm_rayon_min_flops_f32` = 8,000,000, `gemm_rayon_min_flops_f64` = 1,000,000) — which the `gemm` crate's single global threshold cannot express — and GEMV stays 524,288 for both. The knobs `gemm_colpar_min_cols_per_thread` (16), `gemm_chunk_elems` (33,554,432), and `cache_resident_max_bytes` (64 MiB) likewise become runtime-tunable via `tuning::matmul`. `MatmulElem` stays `pub(crate)`; values and behavior unchanged at defaults. Override f32 GEMM via `tuning::matmul::set_gemm_min_flops_f32`, and set `cache_resident_max_bytes` to a machine's actual shared-L3 size.
- Every elementwise, reduction, tree, conv, pool, and norm gate constant in `crate::parallel_gates` (plus `math::exp_reduce_min_elems`, the layer-local norm/conv/pool gates, and the silhouette gate) is replaced by a runtime-tunable atomic via a new crate-root `tunable_gate!` macro, which expands a `const X = V` into a private `AtomicUsize` plus a `pub(crate)` getter/setter preserving docs and `#[cfg(...)]` gating. The hot-path read is a single relaxed atomic load. Defaults are the previously calibrated values, so this is **No behavior change** — pure infrastructure for the new `crate::tuning` facade. Safety floors, `DET_REDUCE_BLOCK`, and algorithm constants stay plain consts.

## [v0.13.0] - 2026-06-16 (UTC-7)
### Added
- New `benches/benchmarks/matmul_kernels.rs` criterion suite sweeps `Dense::forward` across GEMM shapes to isolate the matmul backend over the public API, and a new `benches/calibrations/gemm_calibrate.rs` harness calibrates the `GEMM_RAYON_MIN_FLOPS` / `GEMV_RAYON_MIN_FLOPS` crossovers. The `benches/` tree is reorganized into `benchmarks/` and `calibrations/`, with explicit `[[bench]]` `path` entries and `autobenches = false` in `Cargo.toml`; the old `benches/parallel_gates/matmul.rs` and `benches/RESULTS.md` are deleted.

### Changed
- **Breaking: model persistence switched from JSON to a compact binary format (`postcard`).** `save_to_path`/`load_from_path` on every classical-ML model, `PCA`/`KernelPCA`, and `Sequential` now use [postcard](https://docs.rs/postcard/) instead of `serde_json` (on a fitted `KMeans` the on-disk size drops ~5x). **Old `.json` model files can no longer be loaded** — re-save any persisted models. Signatures are unchanged; only the byte format differs. `serde_json` is dropped in favor of `postcard`.
- **Breaking: `IoError::Json(serde_json::Error)` renamed to `IoError::Serialization(postcard::Error)`** to match the new backend. Code matching on `IoError::Json` must switch to `IoError::Serialization`.
- `LayerWeight`'s on-disk representation changed from internally tagged (`#[serde(tag = "type")]`) to the default externally tagged form, required by postcard's non-self-describing format. Affects only the serialized bytes, not the enum or any code using it.
- Removed the now-unused `IoError::load_in_buf_reader` helper (the binary load path reads the whole file via `std::fs::read`).
- **Breaking: the matrix-product backend switches from `matrixmultiply` to the pure-Rust `gemm` crate.** `gemm_internal` / `gemv_internal` now call `gemm::gemm` with runtime-dispatched SIMD kernels over `gemm`'s rayon pool, via new private `gemm_kernel` / `gemm_rowsplit` helpers. Parallelism is shape-aware: serial below the per-type FLOP gate, column-parallel for wide outputs, row-split for thin outputs and GEMV (`n == 1`, which `gemm` never parallelizes). `gemm = { version = "0.19.0", features = ["rayon"] }` is added (always-on via `math`); `matrixmultiply` is dropped. Signatures of dependent models (`Dense`, `PCA`, `KMeans`, SVM, LDA, kernels) are unchanged; only the kernel and its numerics differ.
- **Behavior change: matrix products stay reproducible across runs on the same machine but are no longer bit-for-bit identical to the old `matrixmultiply` backend (or across different machines).** Tests or pipelines asserting bit-exact matmul output against the old backend must switch to an approximate comparison.
- **Breaking (crate-internal): the `MatmulElem` parallelism crossover constants are renamed and retuned.** `PAR_GEMM_MIN_FLOPS` becomes `GEMM_RAYON_MIN_FLOPS` and `PAR_GEMV_MIN_FLOPS` becomes `GEMV_RAYON_MIN_FLOPS`. The GEMM gate retunes for the in-loop RNN/LSTM regime: `f32` `4,000,000 -> 8,000,000` and `f64` `2,000,000 -> 1,000,000`; the GEMV gate is unchanged at 524,288. The old `PAR_GEMM_MIN_BLOCK = 64` / `PAR_GEMV_MIN_BLOCK = 8` floors are replaced by a single `PAR_ROWSPLIT_MIN_BLOCK = 8` for the row-split path.

### Removed
- **Breaking: the public matrix-product API `gemm`, `gemv`, `gemm_par`, and `gemv_par` in `math::matmul` is removed.** These block-parallel products took a caller-supplied threshold or `min_block` floor over the `matrixmultiply` kernels; they (and their test suite) no longer exist. There is no public replacement — matrix products are now reached only through the crate-internal `gemm_internal` / `gemv_internal` wrappers. Callers that imported `rustyml::math::matmul::{gemm, gemv, gemm_par, gemv_par}` must use `ndarray`'s `dot` or the `gemm` crate directly.

## [v0.12.0] - 2026-06-14 (UTC-7)
### Added
- **`AdamW` optimizer** (`neural_network::optimizers::AdamW`): Adam with decoupled weight decay, shrinking the parameter by `(1 - learning_rate * weight_decay)` before the gradient step (Loshchilov & Hutter) instead of folding it into the gradient — generally the better choice with adaptive optimizers. Honors the same weights-only exclusion and `with_clip_norm` builder as the other optimizers.

### Changed
- **Breaking: `Adam`'s `weight_decay` is now classic coupled L2, not decoupled** (matching PyTorch's `Adam`); the old decoupled behavior moved to `AdamW`. Behavior changes only for `Adam` with `weight_decay > 0`; the default `0.0` is unchanged. Switch to `AdamW` for the old behavior.
- `Sequential::fit_with_batches` no longer keeps two near-identical copies of its epoch/batch training loop (one per `show_progress` feature state); the loop is written once, with only the progress-bar bookkeeping gated behind `#[cfg(feature = "show_progress")]`. No behavior change.
- **Breaking: weight decay now applies only to weight tensors, excluding biases and normalization `gamma`/`beta`, matching the PyTorch/Keras convention.** `ParamGrad` gains a `decays: bool` field with two constructors — `ParamGrad::weight` (`decays = true`, for kernels) and `ParamGrad::no_decay` (for biases and norm `gamma`/`beta`) — and the optimizers gate `apply_weight_decay` on it. **Breaking for code that constructs `ParamGrad` directly** (use the constructors). No numerical change when `weight_decay == 0.0`; otherwise only biases/norm parameters change (no longer shrunk).
- `CategoricalCrossEntropy`'s `from_logits` path drops two redundant `to_owned()` clones of the reshaped one-hot labels, operating straight through the `CowArray` view. Saves one `[batch, classes]` allocation each in `compute_loss` and `compute_grad`; no behavior change.

### Fixed
- **`MeanAbsoluteError` now propagates NaN through its gradient instead of silently zeroing it.** A NaN difference is neither `> 0` nor `< 0`, so it previously fell into the scaled-sign zero branch while `compute_loss` already returned NaN; the `else` (only reachable for NaN) now returns `f32::NAN`, matching numpy/PyTorch `sign(NaN) = NaN`. **Behavior change for NaN inputs only**; every finite input's gradient is unchanged.

## [v0.12.0] - 2026-06-13 (UTC-7)
### Added
- **More common trait derives across `machine_learning`, `utils`, and `neural_network` public types** (additive, no behavior change). `KernelType`/`DecisionTreeParams` gain `PartialEq`; `Solver`, `WeightingStrategy`, `Algorithm`, `EigenSolver`, `SVDSolver` gain `Eq`; `LayerNormalizationAxis` gains `PartialEq`/`Eq`; `PoolKind` gains `Debug`; and the five loss types now derive `Debug, Clone, Copy, PartialEq, Eq, Default`. Layer/activation structs stay `Debug`-only (they hold weights/caches and persist via weights-only save/load).

### Changed
- **Breaking: every `machine_learning`/`utils` estimator constructor keeps only its primary hyperparameters; secondary "optional / has-a-default" settings move to chainable `with_*` builders** (mirroring scikit-learn's argument ordering). Validating setters return `Result<Self, Error>`, trivial ones return `Self`, so a chain reads `Model::new(req…)?.with_a(x)?.with_b(y)`. Affected: `regularization_type`, `random_state`, `max_depth`, `metric`, `weighting_strategy`, `solver`/`shrinkage`, `svd_solver`, `eigen_solver`, `init`/`method`, MeanShift's `max_iter`/`tol`/`bin_seeding`/`cluster_all`, and DecisionTree's `Option<DecisionTreeParams>` bundle (dissolved into per-field setters). **No numerical change** — defaults and serde formats are unchanged. `DecisionTree`'s cross-field `min_samples_leaf <= min_samples_split` check moves to `fit` time, and `TSNE`'s default method is now Barnes-Hut for `n_components <= 3` and Exact otherwise. The defaulted strategy enums gain `#[derive(Default)]` on their canonical variant.
- **Breaking: the same constructor refactor extends to `neural_network` layers and optimizers** — `random_state`, pooling `stride`/`strides`/`padding`, conv `padding`, `LayerNormalization`'s `normalized_axis`, and every optimizer's `clip_norm` move to `with_*` builders, e.g. `Conv2D::new(32, (3,3), shape, (1,1), ReLU).with_padding(Same).with_random_state(42)` and `Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).with_clip_norm(1.0)`. Validating setters return `Result<Self, Error>`; `with_random_state` re-runs the layer's exact initialization, so weights are byte-identical to the old `Some(seed)` form. **Numerically unchanged** for any fixed seed (one subtlety: with both a thread-global seed and `with_random_state`, `new`'s default draw now advances the global stream once, which can shift a later unseeded layer's sub-seed). `PaddingType`/`LayerNormalizationAxis` gain `#[derive(Default)]`.
- **Breaking: clone-free model saving — `LayerWeight` borrows the live layer arrays via `Cow` and is now the single weight type for both inspection and serialization.** `get_weights` builds it with `Cow::Borrowed` (zero clone), saving serializes the borrowed view directly, and loading deserializes to `Cow::Owned`; this also merges the former borrowing `LayerWeight` view and owned `SerializableLayerWeight` DTO into one type. **On-disk format is unchanged.** **Breaking for code that names the weight types:** `get_weights` payloads are now `Cow<'a, _>` instead of `&'a _`.
- The per-layer weight containers move into the `layer_weight` module as one file per layer (`layer_weight/dense_weight.rs`, ...), each holding the `Cow`-backed struct and its `ApplyWeights` impl. The `LayerWeight` enum and the public import paths are unchanged.
- **Breaking: the `serialize_weight` module is renamed `serialize_model`.** It now holds only the whole-model serialization scaffolding (`LayerInfo`, `SerializableLayer`, `SerializableSequential`, `apply_weights_to_layer`); symbol names are unchanged, only the module path moves.
- **SpatialDropout1D/2D/3D apply their per-channel mask as a single fused scale pass and never materialize a full-size mask.** Each `(batch, channel)` is a contiguous `spatial`-length segment, so the shared `spatial_dropout_scale` helper scales it by `channel_mask[i] / (1 - rate)` straight into the output; only the small `[batch, channels]` mask is stored (e.g. 8KB instead of 33MB at conv scale), and backward reuses the same helper. **Bit-for-bit identical** to the previous `input * broadcast(mask) * scale`. At 8.4M-element conv scale the forward drops ~12-14x (1D 25.6 -> 2.4 ms, 2D 29.4 -> 2.4 ms, 3D 33.0 -> 2.4 ms) and backward ~3x (6.9 -> 2.4 ms).
- `GaussianDropout::backward` borrows its cached forward noise (`Option::as_ref`) instead of consuming it (`Option::take`), so backward is now idempotent, matching the other dropout layers. No behavior change for the normal training loop.
- `softmax` backward borrows a contiguous `[batch, features]` view of its read-only `output`/`grad_output` operands (via `to_shape`) instead of always cloning them, cloning only when the source is non-standard-layout. No behavior change.
- **`DepthwiseConv2D` and `SeparableConv2D`'s depthwise stage now convolve directly over flat contiguous buffers, dropping the per-position temporaries and 4-D dynamic indexing of the old loop nests.** The per-channel kernels slice channel and kernel to `&[f32]` once and run a multiply-accumulate over the kernel window with plain offset arithmetic — same math, no allocations, no dynamic indexing. At MobileNet scale (`[8, 64, 56, 56]`, 3x3, `Same`): depthwise forward **-74%** (5.7 -> 1.5 ms), forward+backward **-47%** (8.7 -> 4.7 ms); separable forward **-16%**, forward+backward **-30%**. Numerically equivalent (backward fuses the weight/input gradient passes). Also folds `DepthwiseConv2D::forward`/`::predict` into a shared `convolve` helper.
- The three engine-backed conv layers report a missing input cache in `backward` consistently as `Error::forward_pass_not_run("Conv{1,2,3}D")` (Conv1D/Conv3D previously raised ad-hoc `Error::computation`). Cosmetic — the branch is unreachable in practice.
- **`SimpleRNN`, `LSTM`, and `GRU` no longer duplicate their timestep recurrence between `forward` and `predict`** — both delegate to a shared private `run(&self, x3, caches: Option<&mut Caches>)`, where `forward` passes `Some` (recording per-timestep values) and `predict` passes `None`. The per-timestep caches also collapse into a single `Option<LstmCaches>`/`Option<GruCaches>`. No behavior change; net ~80 fewer lines.

### Removed
- **Breaking:** the separate owned serialization types — the `SerializableLayerWeight` enum, the `Serializable*Weight` structs, and `SerializableLayerWeight::from_layer_weight` — subsumed by the now-`Cow`-backed `LayerWeight` family that serves both inspection and serialization.

### Fixed
- `GaussianNoise::new` now rejects a non-finite `stddev` (NaN / +inf) with `Error::InvalidParameter` at construction, instead of passing validation and later **panicking** in `Normal::new(0.0, stddev).unwrap()` on the first forward. `GaussianDropout` was never affected.
- **ReLU now propagates NaN instead of silently mapping it to 0.** The `Activation::ReLU` forward was `if x > 0 { x } else { 0 }`, zeroing a NaN input; it is now `if x <= 0 { 0 } else { x }`, so NaN propagates, matching the sibling activations and PyTorch/TensorFlow/NumPy and removing a forward/backward asymmetry. **Behavior change for NaN inputs only** (`ReLU(NaN)`: 0 → NaN); every finite/infinite input is unchanged.
- **Max pooling now propagates NaN instead of silently dropping it** (`MaxPooling1D/2D/3D`, `GlobalMaxPooling1D/2D/3D`). Since `NaN > x` is always false, the old `if v > max_val` reduction returned the max of the *other* elements; it now takes a NaN as the window result and records its arg-max position (routing the backward gradient to it), matching PyTorch/TensorFlow/NumPy. Average pooling was never affected. **Behavior change for NaN inputs only**; every finite/infinite input is unchanged.

## [v0.12.0] - 2026-06-12 (UTC-7)
### Added
- **Public deterministic blocked reductions: `math::reduction`.** The crate-internal `reduction` module moves into `math` as a public API: `det_par_fold` folds a `&[f64]` slice and the new `det_par_fold_range` folds an index range `0..n`, for reductions that zip several arrays or accumulate compound state. Both fold fixed `DET_REDUCE_BLOCK` blocks and merge in block order, so re-running on the same machine reproduces the result — and 2-3x faster than a bare rayon reduction on L3-resident inputs.
- Two new calibration sections in `cargo bench --bench parallel_gates`: the exp-heavy logistic-loss reduction (14.3x at 1M) and the k-means assign-accumulate fold (3.2-3.9x at 1-2M).
- New calibration sections in `cargo bench --bench parallel_gates` for the `f32` reduction classes: the f32-elements/f64-accumulator square-sum (12.7x at 1M), an f32 `DET_REDUCE_BLOCK` validation sweep, and the BatchNorm column-stats row-block fold (2.8-4.5x at 1-4M, 12x for narrow channel counts). `cargo bench --bench nn_end_to_end` gains `batchnorm_forward_32x64x64x64`.
- A `parallel_gates` calibration section for the native-layout BatchNorm plane fold (2.8-3.8x at 1M, 11.7x at conv-scale 8.4M) and a `batchnorm_backward_32x64x64x64` end-to-end tracker.
- A `parallel_gates` calibration section for the LayerNorm fused row pass (2.5-4.1x at 1M) and three `nn_end_to_end` trackers: `layernorm_forward_32x512x768`, `layernorm_backward_32x512x768`, and `layernorm_forward_multi_32x64x64x64`.
- Three `nn_end_to_end` trackers for the group-normalization family: `groupnorm_forward_32x64x64x64_8g`, `groupnorm_backward_32x64x64x64_8g`, and `instancenorm_forward_32x64x64x64`, plus new core/finite-difference tests for the GroupNorm, LayerNorm, and BatchNorm row/plane passes.

### Changed
- **Breaking: `machine_learning`'s models are regrouped by algorithm family into submodules** (`clustering`, `linear_model`, `svm`, `tree`, `neighbors`, `discriminant_analysis`, `ensemble`), mirroring scikit-learn; shared internals (`traits`, `parallel`/`validation`, the `spatial` kd-tree) stay at the root. **Behavior change: none** — every estimator is still re-exported flat as `machine_learning::<Model>`. **Breaking only for leaf-path imports:** `machine_learning::dbscan::DBSCAN` is now `machine_learning::clustering::dbscan::DBSCAN`, and `LinearSVC` moves under `svm`.
- `math::logistic_loss` runs as a deterministic blocked parallel reduction above a measured exp-class gate. **Behavior change:** above the gate the summation grouping changes, so results differ from previous versions. End-to-end, `logistic_fit_50000x64_100it` drops 120 ms → 105 ms (~12%).
- K-means' per-iteration assignment accumulation (cluster sums, counts, inertia) runs as a deterministic blocked range fold above the sum gate, replacing the serial per-sample scatter; the labels store becomes a chunked parallel write, and k-means++ seeding uses the same blocked fold. **Behavior change:** above the gates the accumulation grouping — and therefore centroid/inertia low bits — changes versus previous versions. The win is large-dataset scaling.
- Linear regression's per-iteration SSE and intercept-gradient sums use the deterministic blocked fold above the sum gate (`SUM_F64_PARALLEL_MIN_ELEMS`). **Behavior change:** the intercept gradient is on the optimization path, so large trainings follow a different trajectory in the low bits.
- The remaining serial `f64` reductions in `machine_learning` (SVC SMO scans, linear SVC minibatch gradients, mean-shift/decision-tree node sums, isolation-forest path average, LDA shrinkage stats) now carry brief why-serial comments, and the stale `# Performance` doc sections on `KMeans::fit`/`LinearRegression::fit`/`LogisticRegression::fit` are updated to the actual gate classes.
- A determinism audit over every rayon call site in `neural_network` found one violation: `SparseCategoricalCrossEntropy::compute_loss`'s probability path summed its per-sample `ln` terms with a bare parallel `sum`, so the reported loss varied with thread scheduling (`compute_grad`, which drives training, was unaffected). It now sums serially. Everything else checked out.
- A determinism audit of every rayon call site in `machine_learning`/`utils` found one violation: t-SNE's `show_progress`-only KL-divergence readout used a bare parallel `sum`, so the displayed loss could vary with thread scheduling (embeddings were unaffected). It now sums in row order. Everything else checked out.
- **`math::reduction::det_par_fold` is generic over the element type** (`&[T] where T: Sync`, was `&[f64]`), so widening reductions (f32 elements, f64 accumulator) need no conversion pass. Existing `f64` call sites are unaffected.
- **Breaking: `math::matmul`'s public API takes the serial/parallel threshold as a parameter.** `par_matmul`/`par_matvec` are renamed to `gemm`/`gemv` with a trailing `min_parallel_flops: usize` (`0` = always split, `usize::MAX` = always serial); both paths are numerically a pure refactor. The `MatmulElem` trait becomes `pub(crate)` and the element bounds relax to `LinalgScalar + Send + Sync` (any `LinalgScalar`, not just `f32`/`f64`); crate-internal call sites use new `gemm_internal`/`gemv_internal` wrappers. The forced-split hooks become documented public `gemm_par`/`gemv_par`, exposing a per-block `min_block` floor instead of the FLOPs threshold.
- **Breaking: `math::reduction` internalizes the serial/parallel switch.** `det_par_fold`/`det_par_fold_range` are renamed to `det_reduce`/`det_reduce_range` and take a trailing `parallel: bool`; the serial path folds the same fixed blocks sequentially, so the flag never changes the result. Every internal call site passes its calibrated gate and drops its hand-written serial branch (PCA, standardize, `logistic_loss`, k-means, linear regression, `global_grad_norm`, t-SNE, and BatchNorm's seven column folds). **Behavior change:** below their gates these sites use the blocked grouping instead of a flat fold, so sub-gate sizes above one block see a one-time low-bit change; clip-norm models change regardless of size.
- **BatchNorm's per-channel statistics run as row-block deterministic folds** above a measured gate: the forward mean/variance and all five backward column reductions, with the fused product folds also eliminating the `[M, C]` temporaries. **Behavior change:** above the gate the per-channel accumulation grouping changes versus previous versions; below it the serial paths are unchanged. Spatial BatchNorm forward at `[32, 64, 64, 64]` drops 72.3 -> 54.1 ms (~25%); the remainder is the fold/unfold transposes.
- **Spatial BatchNorm (rank >= 3) runs entirely on the native `[batch, channels, *spatial]` layout, dropping the fold/unfold transpose copies.** Per-channel statistics are computed as plane folds straight off the native layout, and every elementwise pass streams per plane with the channel scalars hoisted, sharing one kernel between its serial and parallel paths. **Behavior change:** spatial *training* statistics shift in the low bits; spatial *inference* and the 2-D path are bit-for-bit unchanged. End-to-end at `[32, 64, 64, 64]`: forward 54.0 -> 9.6 ms (-82%), backward 55.8 -> 9.6 ms (-83%).
- **LayerNorm runs a fused parallel row path for every trailing-axis configuration, and trailing in-order `Multiple` axes no longer transpose at all.** Each trailing-block group is one row of a logical `[R, N]` matrix, so forward and backward fold the per-row stats and fuse normalize + scale-shift into one streaming sweep (kernels shared with BatchNorm via `normalization::folds`); for in-order `Multiple` the merge/unmerge transposes are skipped, while a genuinely permuted list keeps one copy each way. **Behavior change:** outputs/gradients change in the low bits at every size — still deterministic. End-to-end: `Default` forward `[32, 512, 768]` 48.8 -> 13.5 ms (-72%), backward 92.7 -> 7.4 ms (-92%), trailing-`Multiple` forward at `[32, 64, 64, 64]` 44.2 -> 9.1 ms (-79%).
- **GroupNorm and InstanceNorm run on a fused parallel per-instance row path, dropping the channels-first core's reshape copies and broadcast temporaries.** Each instance (a contiguous `channels_per_group * spatial` block; `InstanceNormalization` is the `num_groups == channels` case) folds its stats with the fixed-order kernels and writes the affine output in one sweep, with gamma/beta gradients as the same plane folds BatchNorm uses; `channel_axis == 1` now borrows with zero full-tensor copies. **Behavior change:** outputs/gradients change in the low bits at every size — still deterministic. End-to-end at `[32, 64, 64, 64]`: GroupNorm (8 groups) forward 46.8 -> 4.9 ms (-89%), backward 90.5 -> 4.1 ms (-95%), InstanceNorm forward 46.5 -> 4.9 ms (-89%).
- **`global_grad_norm` (clip-by-global-norm) folds large parameter tensors in parallel:** tensors at or above the `SQ_SUM_F32_PARALLEL_MIN_ELEMS` gate use the deterministic blocked fold (f32 elements, f64 accumulator), saving ~0.35 ms per optimizer step per million parameters. **Behavior change:** for such models the clip scale's low bits change; models whose tensors are all below the gate reproduce the previous norm.

## [v0.12.0] - 2026-06-11 (UTC-7)
### Added
- **Public block-parallel matrix products: `math::matmul`.** The crate-internal `matmul` helper moves into `math` as a public API: `par_matmul` (`C = A.B`) and `par_matvec` (`y = A.x`) over any `f32`/`f64` ndarray operands (owned, views, transposes pass by reference). Both keep matrixmultiply's serial inner kernels, split the output across rayon above calibrated per-type FLOPs gates, and are reproducible across runs on the same machine. The `MatmulElem` trait is public but sealed to `f32`/`f64`. The `math` feature gains `dep:rayon` (a dependency-tree no-op) and `neural_network` now depends on `math`.
- **Benchmark infrastructure under `benches/`** (criterion added as a dev-dependency). `cargo bench --bench parallel_gates` is a custom-harness calibration suite that times forced-serial vs forced-parallel for every parallel-gated kernel class across size ladders and rewrites `benches/RESULTS.md` — the source of truth for the `*_MIN_FLOPS`/`*_MIN_OPS`/`*_PARALLEL_THRESHOLD` constants. `cargo bench --bench nn_end_to_end` tracks public-API performance with saved baselines. The forced-path entry points live in a `#[doc(hidden)] pub mod bench_internals`; the conv/pooling engines gained `force_parallel: Option<bool>` plumbing.
- **Clip-by-global-norm** gradient clipping for the neural-network optimizers, opt-in via a trailing `clip_norm: Option<f32>` on `SGD`/`Adam`/`RMSprop`/`AdaGrad::new` (`Some` must be positive and finite). When enabled, the training loop computes the global L2 norm across all gradients (in `f64`) and, if it exceeds `max_norm`, scales every gradient by `max_norm / global_norm`, preserving direction; a non-finite norm is left unscaled so it surfaces. Exposed via `Optimizer::clip_norm()` and a `grad_scale` parameter on `Optimizer::update`, with zero overhead when off.
- **`padding='same'` for the windowed pooling layers** (`MaxPooling`/`AveragePooling` 1D/2D/3D). The pooling engine gains a `PaddingType` argument and a `pool_geometry` helper: `Same` rounds the output up to `ceil(in/stride)` and keeps padded cells virtual (out-of-bounds positions skipped). Average pooling divides by the count of real in-bounds elements (Keras's `count_include_pad=False`); max pooling's backward is unchanged.
- **SGD momentum / Nesterov and decoupled weight decay across all optimizers.** `SGD::new` gains `momentum` and `nesterov` (a `sgd_momentum_step` kernel with per-parameter velocity buffers; `momentum = 0` stays plain SGD), and every optimizer gains a `weight_decay` applied as decoupled AdamW/SGDW-style decay (`apply_weight_decay` shrinks the parameter by `1 - lr*wd` before the gradient step).
- **External learning-rate scheduling.** New `Optimizer::set_learning_rate` trait hook (overridden by all four optimizers) and a `Sequential::set_learning_rate` forwarder, so a schedule can retune the step size between epochs or batches without rebuilding the optimizer or losing its state.
- **`from_logits` fused softmax-cross-entropy.** `CategoricalCrossEntropy::new`/`SparseCategoricalCrossEntropy::new` take a `from_logits: bool`. When `true`, the loss applies a stable log-softmax internally and `compute_grad` returns the fused `(softmax(z) − y) / batch` gradient w.r.t. the logits, skipping a separate softmax-layer backward.

### Changed
- Make the `math` distance-row functions (`squared_euclidean_distance_row`, `manhattan_distance_row`, `minkowski_distance_row`) single-pass and allocation-free by accumulating over an `ndarray::Zip` instead of materializing intermediate arrays, removing two allocations per call on the KNN/DBSCAN/silhouette `O(n²)` hot paths.
- `DistanceCalculationMetric::within` now decides the threshold test in the metric's order-preserving "comparable" space for every variant, so `Minkowski(p)` compares `Σ|a−b|ᵖ` against `thresholdᵖ` instead of taking a per-pair `p`-th root — matching the root-free Euclidean path it already used.
- Evaluate the `Poly` kernel with `powi(degree)` instead of `powf(degree as f64)` in both `KernelType::compute` and the batched `compute_matrix`; the exponent is integral, so `powi` is faster and more accurate.
- `silhouette_score` accumulates its pairwise distances over independent per-sample rows, switching between serial and parallel `Zip::par_for_each` fills at a sample-count threshold, replacing the serial `O(n²)` double loop. **Breaking:** it now takes a trailing `metric: DistanceCalculationMetric` argument (pass `Euclidean` for the conventional silhouette) and tightens its storage bound to `S1: Data + Sync`. The `types` module is now compiled under `metrics` too, with serde gated to `machine_learning`/`utils`, so `metrics` still pulls in no serde dependency.
- **Breaking:** the regression metrics take two independent storage parameters `S1`/`S2` instead of a single shared `S`, so `y_true` and `y_pred` may now mix owned arrays and views (matching `classification`/`math`).
- Document the deliberate NaN divergence between `r2_score` (plain sums, so a `NaN`/`inf` propagates to a `NaN` result) and `explained_variance_score` (routes through `math::variance`, which silently skips non-finite samples).
- The prelude root now flattens every category, so `use rustyml::prelude::*;` brings the actual items (traits, models, metrics, ...) into scope instead of only the category module names. The per-category modules (`rustyml::prelude::machine_learning::*`, etc.) remain available for narrower imports.
- **Breaking:** every neural-network optimizer constructor (`SGD::new`, `Adam::new`, `RMSprop::new`, `AdaGrad::new`) takes a new trailing `clip_norm: Option<f32>` argument (pass `None` to keep the previous behavior), and the `Optimizer::update` trait method gains a trailing `grad_scale: f32` parameter. Putting this optional setting in the constructor matches the crate's existing `random_state: Option<u64>` convention rather than a separate builder method.
- **Behavior change:** the embedded activation derivatives (`Activation::backward` for Sigmoid/Tanh/ReLU/Softmax) are now exact — the `clip_grad`/`GRAD_CLIP_VALUE` sanitization (clamp to ±1e6, NaN/Inf → 0) is removed, so a non-finite gradient now propagates rather than being silently zeroed. The standalone activation layers likewise drop their backward NaN/Inf check on `grad_output`. The pure-math contract is documented on `Layer::backward`.
- **Behavior change:** the recurrent layers (SimpleRNN, GRU/LSTM via `store_gate_gradients`) no longer clamp their stored gradients element-wise to `±5`; the hardcoded `GRADIENT_CLIP_VALUE` is gone. Use the new opt-in clip-by-global-norm to tame exploding gradients instead.
- **Breaking:** `BatchNormalization` is now genuine *spatial* batch norm for rank > 2 inputs. Parameters are per-channel (length `input_shape[1]`, was the full `input_shape[1..]`) and statistics reduce over the batch **and** all spatial positions (matching Keras/PyTorch), implemented by folding `[N, C, *spatial]` to `[M, C]`; the 2-D path is byte-identical. CNN models relying on the old per-element parameterization see different (correct) results and a smaller parameter count.
- **Breaking:** every optimizer constructor gains a trailing `weight_decay: f32`, and `SGD::new` additionally gains `momentum: f32`/`nesterov: bool` (so `SGD::new(lr, clip_norm, momentum, nesterov, weight_decay)`; the others take `..., clip_norm, weight_decay`). Pass `0.0`/`false` to keep the previous behavior. The `Optimizer` trait gains a defaulted `set_learning_rate`.
- **Breaking:** `CategoricalCrossEntropy::new` and `SparseCategoricalCrossEntropy::new` take a `from_logits: bool` (pass `false` for the previous probability-input behavior).
- **Breaking:** the windowed pooling constructors (`MaxPooling`/`AveragePooling` 1D/2D/3D `::new`) take a trailing `padding: PaddingType` argument (pass `PaddingType::Valid` for the previous behavior); the shared pooling engine functions gain the same parameter.
- **Breaking:** the convolution engine (`conv_geometry`/`conv_forward`/`conv_backward`) now returns `Result`: under `Valid` padding an input spatial dimension smaller than the kernel returns `Error::InvalidInput` instead of underflowing `usize` and panicking. This covers Conv1D/2D/3D and the SeparableConv2D pointwise stage uniformly, and `validate_input_shape_3d` gains a kernel parameter so Conv3D rejects an oversized kernel at construction like Conv1D/2D already did.
- **Behavior change:** removed the eager NaN/Inf scan from the standalone `ReLU`/`Sigmoid`/`Tanh`/`Softmax`/`Linear` layers' `forward`/`predict`, matching the embedded-activation path; a non-finite input now propagates instead of being rejected with `Error::NonFinite`.
- Removed the `±500` input clamp from sigmoid/tanh (`Activation` and the recurrent `apply_sigmoid`). `1/(1 + e^-x)` is finite for any finite `x` (an overflowing `e^-x` yields the exact limit `0`) and tanh self-saturates, so the clamp never prevented overflow and was misleadingly documented as doing so.
- `Sequential::fit_with_batches` builds each mini-batch with `ndarray::select(Axis(0), &indices)` (a bulk per-sample gather) instead of an element-by-element `extend`, for any input rank.
- **Breaking:** LSTM and GRU store their gates **fused** — per-gate weights packed side by side into single `kernel`/`recurrent_kernel`/`bias` matrices (column blocks `[i|f|g|o]` for LSTM, `[r|z|h]` for GRU), so the input projection, BPTT reductions, input-gradient GEMM, and per-timestep recurrent projection each collapse to one wide GEMM. `set_weights` now takes the three fused matrices (so older saved models no longer load); a new `set_gate_weights` keeps the per-gate signature. `LSTMGateWeight`/`GRUGateWeight`/`SerializableGateWeight` and the `Gate` struct are replaced by `FusedGates`, and the optimizer sees 3 parameter tensors per layer instead of 12/9. Initialization semantics are unchanged but RNG stream consumption differs, so same-seed inits differ from previous versions. The per-step `rayon::join` and the `LSTM`/`GRU_PARALLEL_THRESHOLD` constants are gone.
- The Dense and recurrent (SimpleRNN/LSTM/GRU) GEMMs run **block-parallel** through a new crate-internal `par_matmul` helper: large products split their longer output axis into row/column blocks across rayon, each block run by the serial kernel. Splitting the `m`/`n` axes never reorders the `k`-accumulation, so the result is reproducible across runs on the same machine; products below an estimated-FLOPs gate fall through to serial. Sanity check on `[2048,1024] × [1024,1024]`: 30.6 ms → 5.5 ms (~5.6×).
- The convolution engine's forward pass parallelizes over `(batch item, output-position block)` tasks instead of batch items alone, so a single large image saturates every core even at `batch == 1` (previously serial). Block boundaries never change accumulation order, so results are reproducible across runs on the same machine; the backward keeps its batch-order reduction and routes its per-item GEMMs through `par_matmul`. Sanity check on `[1, 64, 128, 128]` with 64 `3x3` filters: 97.4 ms → 9.3 ms (~10.5×) for five forward passes.
- The parallel/serial gates across the convolution and pooling layers estimate actual per-pass work instead of counting output elements: the conv/`DepthwiseConv2D`/`SeparableConv2D` gates count FLOPs including kernel-tap/channel multipliers, and the pooling engine gates on `planes × per-plane work`. The new constants (`CONV_PARALLEL_MIN_FLOPS`, `DEPTHWISE_CONV_2D_PARALLEL_MIN_FLOPS`, `SEPARABLE_CONV_2D_PARALLEL_MIN_FLOPS`, `POOL_PARALLEL_MIN_OPS`) are initial estimates pending calibration.
- **Every parallel/serial gate constant recalibrated from measurement.** The elementwise thresholds were severely *too low* (small tensors paid a ~20-25 µs rayon fork/join for maps that finish in microseconds; sigmoid at its old 1000-element gate ran ~25× slower than serial) while the GEMM gates were *too high*. New values: `PAR_GEMM_MIN_FLOPS`/`CONV_PARALLEL_MIN_FLOPS` 1e7 → 4M; `POOL_PARALLEL_MIN_OPS` 1M → 12K taps; `SIGMOID`/`TANH` 1000/2048 → 131,072; `SOFTMAX_PARALLEL_THRESHOLD` now on `batch * classes`; `RELU`/`DROPOUT`/`SPATIAL_DROPOUT` → 4M; optimizer `kernels::PARALLEL_THRESHOLD` 1024 → 1M; `BATCH_NORM_PARALLEL_THRESHOLD` 1024 → 262,144. End-to-end: Dense forward −34%, Conv2D batch-1 forward −24%, LSTM forward −57%, MLP train epoch −46%.
- **Finer task granularity for the remaining batch-bound passes**, so small batches (especially `batch == 1`) keep every thread busy. `DepthwiseConv2D` backward parallelizes over `(batch item, channel)`; `SeparableConv2D`'s depthwise forward over `(batch item, output channel)` and its backward gradients over `(depth multiplier, channel)`/`(batch item, channel)` (the old depth-multiplier-only split degenerated to a single task at `depth_multiplier == 1`), and both gained the serial gate they lacked. Windowed pooling's forward splits each plane into output-position chunks (2.7× → 7.8× on `[1, 3, 1024, 1024]` max pool). Deliberately *not* re-granulated: pooling's backward and global pooling (both would break per-chunk reproducibility). The unused `merge_results` helper is removed.
- The elementwise gate constants are consolidated into a single crate-internal `neural_network::parallel_gates` module, one constant per calibrated **cost class** — `CHEAP_MAP_PARALLEL_THRESHOLD`, `EXP_MAP_PARALLEL_THRESHOLD`, `FUSED_SLICE_PARALLEL_THRESHOLD`, and `NAIVE_CONV_PARALLEL_MIN_FLOPS` — replacing nine per-layer duplicates. The engine-specific gates (`PAR_GEMM_*`, `CONV_*`, `POOL_*`, `BATCH_NORM_*`) keep their engine-local constants.
- The block-parallel matmul helper moves from `neural_network::matmul` to a crate-level `matmul` module and is **generic over the element type** through a new `MatmulElem` trait (`f32`/`f64`), so the `f64` classical-ML and utils modules can route their GEMMs through the same gated path. The FLOPs gates live on the trait as per-type constants (`f64` values are conservative placeholders pending calibration). Also adds `par_matvec` (`y = A·x`) with its own `PAR_GEMV_MIN_FLOPS` gate, keeping each block on ndarray's matrix-vector kernel so it stays reproducible across runs on the same machine.
- **The ML/utils matrix products run block-parallel** through the shared `matmul` helpers. `KernelType::compute_matrix` (the dominant cost of SVC and KernelPCA) computes its cross-Gram matrix as one block-parallel GEMM plus a parallel kernel transform; PCA, KernelPCA, and LDA route their projections, scatter/covariance GEMMs, and LSQR/scoring products through `par_matmul`/`par_matvec` (retiring the hand-rolled `project_parallel`/`reconstruct_parallel` loops); `utils::linalg`'s power-iteration/Lanczos matvecs and Hotelling deflation, and the full-batch GEMVs of LogisticRegression/LinearRegression/LinearSVC follow. The GEMV call sites are numerically unchanged; the kernel-matrix/projection GEMMs swap to the matrix-matrix kernel, so entries may differ at rounding level (still deterministic). All golden tests pass.
- **The ML/utils per-sample and per-pair distance loops are rewritten in batched GEMM form.** KMeans computes each Lloyd iteration's sample-centroid projection as one `data · centroidsᵀ` GEMM plus a per-row arg-min scan (ranking by `‖c‖² − 2·x·c`), and KMeans++ folds each new center into a running per-sample minimum (`O(n·k·d)` instead of `O(n·k²·d)`, bitwise identical). KNN's brute-force path, t-SNE's exact pairwise distances and gradient (`grad = 4·(diag(W·1)·Y − W·Y)`), the Barnes-Hut neighbor search, and MeanShift's `estimate_bandwidth` all use the `‖x_i‖² + ‖x_j‖² − 2·x_i·x_j` identity with chunked GEMMs (governed by `matmul::gemm_chunk_rows`). **Numerical note:** the norms-identity distances round differently than the old scalar loops, so KMeans/KNN outcomes can shift on exact ties and t-SNE embeddings differ at rounding level (still deterministic). All golden tests pass.

- **Every ML/utils parallel/serial gate is now calibrated, work-metric-based, and class-shared.** The calibration bench gains `f64` sections, landing `MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS` 4M -> 2M, `PAR_GEMV_MIN_FLOPS` 4M -> 524,288, a new `PAR_GEMV_MIN_BLOCK = 8` matvec row floor, and crate-root `f64` classes (`CHEAP_MAP_F64_PARALLEL_THRESHOLD = 4M`, `EXP_MAP_F64_PARALLEL_THRESHOLD = 65,536`, `SCAN_F64`/`SUM_F64_PARALLEL_MIN_ELEMS = 262,144`). **Fifteen per-model magic-number thresholds are deleted**, their sites re-gated on **total work** (items × per-item cost, e.g. `n*k` for the KMeans scan, `n*n` for the t-SNE fills) against the matching class constant. `map_collect` takes the caller's parallel decision; `compute_matrix`'s kernel transforms now gate per class. `metrics`' silhouette gate is restated locally as `SILHOUETTE_PARALLEL_MIN_ELEMS = 262_144`; the decision-tree/isolation-forest gates keep their uncalibrated values.
- **Eight scheduling-dependent f64 reductions are fixed for reproducibility.** A bare rayon `sum`/`fold().reduce()` groups partials by work-stealing, so the result varied with the thread count. Serialized outright (the work sits below the sum crossover at any realistic size): SVC's `compute_error`/error-cache, `select_second_alpha` max-reduce, LinearRegression's L1 penalty, LinearSVC's per-batch gradient, KernelPCA's kernel-mean, KNN's weighted-vote fold, and SVC's support-vector filter. PCA's total-variance sum and standardization's Welford moments instead go through the new `reduction::det_par_fold`, which parallelizes above the calibrated gate while staying deterministic.
- **The tiled-GEMM strategy from the batched-rewrite round is corrected by measurement: it only wins once the shared matrix overflows the L3 cache.** While cache-resident, a per-row GEMV swarm beats both the scalar loops (2.8x) and a tiled GEMM (~2.7x); on a cache-overflowing 256 MB training set the tiled GEMM wins ~2x instead. KNN's brute-force path, t-SNE's neighbor search, and MeanShift's bandwidth estimation now pick per shape via a new `matmul::cache_resident` helper (64 MB boundary), and `GEMM_CHUNK_ELEMS` is recalibrated 4M -> 32Mi for the overflow regime.
- **The last uncalibrated ML thresholds are now measured.** New `parallel_gates` classes: `TREE_TRAVERSAL_MIN_VISITS = 262_144` and `SORT_SCAN_MIN_ELEMS = 8_192`. DecisionTree's `DECISION_TREE_PARALLEL_THRESHOLD = 1000` is deleted (measured at a 0.33x *loss*); prediction now gates on traversal work and the split search on sort work. IsolationForest's `DEFAULT_PARALLEL_THRESHOLD_SAMPLES = 100` becomes the same traversal-work gate, its tree-build gate keeping 10. The kd-tree ceilings `KNN/DBSCAN_KD_TREE_MAX_DIMS` drop 16 -> 8 (the kd-tree loses to brute force by 2.2-2.6x over d=12-16). `reduction::DET_REDUCE_BLOCK` moves 8192 -> 16384. The remaining unmeasured ML/utils constants are algorithm semantics, not performance gates.
- New `benches/ml_end_to_end.rs` criterion suite over the classical-ML/utils public API (KMeans/SVC/LogisticRegression fits, KNN brute-force predict, PCA/KernelPCA fit+transform, exact t-SNE) for performance-regression tracking with saved baselines, complementing `nn_end_to_end`.

### Removed
- **Breaking:** drop the `rustyml::prelude::math` submodule. The `math` items are low-level numeric primitives (distances, losses, variance/SST, gini/entropy) that don't belong in a prelude by convention - traits and high-level entry points do. Import them directly instead, e.g. `use rustyml::math::variance;`. The other category preludes (`machine_learning`, `metrics`, `utils`, `neural_network`) are unchanged.

### Fixed
- The recurrent layers' batched input projection and input-gradient reshapes no longer panic with `IncompatibleLayout` when a GEMM operand has row stride 1 (which `ndarray`'s `dot` returns column-major, e.g. for `units == 1` gradient matrices, and the bare `into_shape_with_order` rejected). A shared `gate::reshape_2d_to_3d` helper normalizes the layout first; this was a pre-existing latent panic in `SimpleRNN::backward` for `units == 1`.
- **Breaking:** `math::{sum_of_squared_errors, logistic_loss, hinge_loss}` now panic on a length mismatch (matching `metrics::validate_pair`) instead of silently truncating to the shorter input through `zip`, and panic on empty input — so `logistic_loss` no longer returns `0.0/0.0 = NaN` and `hinge_loss` no longer disagrees with it by returning `0.0`.
- **Breaking:** `ConfusionMatrix::recall` returns `0.0` (was `1.0`) when there are no actual positives, matching scikit-learn's `zero_division=0` default and `MulticlassConfusionMatrix::per_class_recall`; the old `1.0` also spuriously inflated `balanced_accuracy`.
- **Breaking:** `normalized_mutual_info` normalizes by the **arithmetic** mean of the two entropies (was the geometric mean `√(H_true·H_pred)`), matching scikit-learn (≥ 0.22), `adjusted_mutual_info`, and `v_measure_score` (which is exactly the arithmetic-mean-normalized NMI), so the family of metrics now agrees on the same data.
- **Breaking:** the ranking metrics (`roc_auc`, `average_precision`, `roc_curve`, `precision_recall_curve`) now panic on a `NaN` score instead of letting `f64::total_cmp` silently rank it as the most-confident prediction; `top_k_accuracy` likewise panics on a `NaN` in `y_prob` instead of miscounting a `NaN` true-class probability as a hit.
- `log_loss` renormalizes each probability row to sum to 1 before scoring (matching scikit-learn), so rows that do not already sum to 1 are comparable to scikit-learn's output.
- `average_path_length_factor` includes the next harmonic-expansion term `1/(2(n−1))` in its `n > 50` branch, cutting the approximation error from ~2×10⁻² to ~10⁻⁵ at the cost of one division.
- **Breaking:** Minkowski `p` is now validated against its documented `p ≥ 1` contract: `minkowski_distance_row` panics for `p < 1` (or `NaN`), `KNN::new` rejects such `p` (it performed no validation before), and `DBSCAN::new` tightens its check from `p > 0` to `p ≥ 1`. Orders below 1 are not valid metrics (the triangle inequality fails) and `p ≤ 0` additionally yields a meaningless `sumᶦⁿᶠ`.
- **`SimpleRNN` gradient accumulated across batches.** `SimpleRNN::backward` seeded its gradient buffers from the previous call, so with no `zero_grad` step the gradient at batch *n* was the running sum over batches `1..=n`, drifting from the correct direction. It now starts each backward from zero, matching `Dense` and the LSTM/GRU gates.
- **`GaussianDropout` backward ignored the sampled noise.** The forward pass applies multiplicative noise `y = x * noise`, but backward passed the gradient straight through; the correct gradient is `grad_output * noise` using the *same* noise drawn in forward. The layer now caches the forward noise and multiplies it back in backward (inference and `rate == 0` remain a pass-through). The additive `GaussianNoise` pass-through was already correct.
- **`backward` panicked on a wrong-rank gradient.** `Dense`, `SimpleRNN`, `LSTM`, and `GRU` converted the upstream gradient with `into_dimensionality().unwrap()`, panicking through the public `Layer::backward` on a mismatched rank. They now return a recoverable `Error` (`Dense` also gained an up-front shape guard, since its activation backward would otherwise panic inside an element-wise `Zip`).
- **`DepthwiseConv2D` panicked on a wrong channel count.** Its forward/predict used `assert_eq!(channels, filters, ...)`; this is now an `Error::DimensionMismatch`, matching the recoverable-error convention of the other layers.
- **Optimizer state could silently corrupt across a parameter-shape change.** Adam/RMSprop/AdaGrad (and now momentum-SGD) address per-parameter state by cursor position; if a parameter's length changed between steps, `*_step`'s `zip` silently truncated. They now detect a length mismatch and rebuild that state slot.
- **`CategoricalCrossEntropy` accepted 1-D inputs.** With a 1-D tensor, `shape()[0]` is the total element count rather than the batch size, silently rescaling the loss and its gradient; it now requires `ndim ≥ 2` (`[batch, classes]`).

## [v0.12.0] - 2026-06-10 (UTC-7)
### Added
- Add `KernelType::compute_matrix(x, y)`, which builds the full kernel matrix `K[i, j] = K(x_i, y_j)` in one batched, row-parallel pass (a GEMV per row with the kernel's elementwise transform fused in) instead of looping the scalar `compute` over every pair.
- Add Barnes-Hut t-SNE as the default optimization method via a new `TSNEMethod { BarnesHut { angle }, Exact }` selector. `BarnesHut` (default, `angle = 0.5`) summarizes repulsive forces through a space-partitioning tree, cutting per-iteration cost from `O(n²)` to ~`O(n log n)`; `Exact` keeps the dense all-pairs gradient. `BarnesHut` requires `n_components ≤ 3`.
- Add PCA initialization for t-SNE through a new `Init { PCA, Random }` selector. `Init::PCA` (the default) seeds the embedding from the input's top principal components, giving a deterministic, seed-independent start; `Init::Random` keeps the seeded small-noise start. Falls back to random when the input has fewer features than components or the leading component is degenerate.
- Add `train_test_split_stratified`, which splits each class independently so both subsets keep the input's class proportions. Requires `A: Clone + Eq + Hash` and errors on any class with fewer than 2 samples. Re-exported from `utils` and the prelude.
- Add an internal `machine_learning::spatial` kd-tree (`KdTree`) to accelerate neighbor queries in DBSCAN and KNN. Pruning runs in a metric "comparable space" so a single tree serves every `DistanceCalculationMetric`; it exposes `build`, `radius_neighbors`, and `k_nearest`. Adds the supporting `DistanceCalculationMetric::{comparable_scalar, comparable_distance, distance_from_comparable, within}` methods.

### Changed
- `SVC` (training Gram matrix) and Kernel PCA (fit and transform kernel matrices) now build their kernel matrices through `KernelType::compute_matrix`, replacing their private per-pair `compute` loops with one parallel GEMV-per-row pass.
- Reformulate the hot distance loops in `KMeans`, `KNN`, and `MeanShift` around the `‖x − y‖² = ‖x‖² + ‖y‖² − 2·x·y` identity, turning each per-pair inner loop into a single GEMV (`KMeans` recomputes the winning centroid's true distance directly so inertia is unchanged). `LinearSVC` likewise evaluates its hinge-loss margins as a single `X·w + b` GEMV, matching `predict`.
- t-SNE's gradient-descent optimizer gains adaptive per-parameter gains (Jacobs' delta-bar-delta), making the embedding markedly more robust on well-separated clusters. **Behavior change:** embeddings now differ from earlier versions. The pairwise distance is kept as the exact per-pair form rather than the squared-norm identity, which loses precision to catastrophic cancellation on uncentered data.
- **Breaking:** `TSNE::new` takes two new trailing arguments, `init: Init` and `method: TSNEMethod`; the `Init::Pca` variant is renamed `Init::PCA`; and the chained `with_init` builder is removed. **Behavior change:** the default embedding now uses PCA initialization and the Barnes-Hut gradient, so results differ from the previous random-init/exact default and no longer depend on `random_state`.
- Replace LDA's `Shrinkage::Auto` heuristic with the closed-form Ledoit-Wolf optimal shrinkage intensity `δ = b² / d²`, clamped to `[0, 1]`. The previous `n_features / (n_samples + n_features)` ratio was labeled "Ledoit-Wolf style" but was not the optimal estimator.
- Make LDA's `Solver::LSQR` a genuine iterative least-squares solve (Paige-Saunders LSQR) of each class's scoring system `Σ · coef = μ_c`, instead of a relabeled SVD pseudo-inverse identical to `Solver::SVD`. Dispatch now yields the per-class scoring coefficients directly, so the three solvers are genuinely distinct; the `transform` projection stays solver-independent.
- Document that `standardize` uses the population variance (divides by `n`), matching scikit-learn's `StandardScaler`, with no sample-variance (`n − 1`) option.
- DBSCAN and KNN use the new kd-tree for neighbor search when `n_features ≤ 16`, falling back to the linear scan above that. KNN caches the tree in a serde-skipped `OnceLock` invalidated on each `fit`. Both unify their neighbor tie-break to a `(distance, index)` total order, so the tree and brute-force paths return identical results.
- Rewrite the decision tree's numeric-split search to sort each candidate feature once and sweep thresholds incrementally, scoring each split in `O(1)` via a shared `impurity_from_counts`, dropping per-feature cost from `O(n²)` to `O(n log n)`.
- `SVC`'s decision function builds the support-vector kernel matrix in one batched `compute_matrix` pass and evaluates `K · (α ⊙ y) + b`, replacing the per-`(sample, support vector)` scalar-kernel loop.
- Rewrite the LDA projection to solve the generalized eigenproblem `S_b w = λ S_w w` through a whitening transform plus a symmetric eigendecomposition, giving the correct discriminant directions; the previous code took the left singular vectors of the non-symmetric `S_w⁻¹ S_b`, which are not the discriminant axes. Per-class scoring parameters are cached at fit so `predict` scores in parallel over rows.
- PCA's randomized SVD re-orthonormalizes the subspace with a QR step between each power iteration (Halko et al.), so the iterates no longer collapse toward the dominant singular vector and corrupt the trailing components.
- Standardization computes its mean and variance in a single numerically-stable pass via Welford's online algorithm (with Chan's parallel merge on the parallel path), replacing the prior less-stable computation.
- `LinearSVC` accumulates each minibatch gradient in place with `scaled_add` into per-thread accumulators (rayon `fold`/`reduce`), removing a temporary allocation per sample.
- t-SNE's parallel pairwise-distance and Student-t affinity matrices compute the upper triangle once and scatter it symmetrically, instead of computing every pair twice.
- Kernel PCA tolerates non-positive eigenvalues instead of failing the fit, since a centered Gram matrix is only PSD up to round-off and non-Mercer kernels (e.g. `Sigmoid`) legitimately yield slightly negative trailing eigenvalues. Validation now rejects only non-finite eigenvalues, and degenerate components are zeroed at projection time, matching scikit-learn.

### Fixed
- DBSCAN: tighten the cluster-expansion guards so an already-labeled point is not reprocessed and only still-unlabeled neighbors are enqueued (skip when `labels[q] >= 0`, enqueue when `labels[r] < 0`), preventing two touching clusters from bleeding into each other.
- Isolation Forest: normalize anomaly scores by the average path length of the actual subsample size `c(sample_size)` rather than `c(max_samples)`, and return `1.0` when `c(sample_size) ≤ 0` (e.g. a single-sample subsample) instead of producing `NaN`. Adds a `sample_size` field and `get_sample_size` getter.
- K-means++: skip zero-distance candidates in the roulette-wheel center selection (`dist > 0.0 && cumulative_dist >= choice`), so an already-chosen or coincident point cannot be picked again as a new center.
- Linear regression: report the L2 penalty in the cost as `0.5·α·‖w‖²` to match the `α·w` gradient and the half-MSE data term (it was `α·‖w‖²`, inconsistent with the gradient). Fitted models are unchanged; only the reported cost is corrected.
- Logistic regression: evaluate the reported training cost from the same weights as the logits (and apply the update in place with `scaled_add`), rather than mixing pre-update logits with post-update weights.
- Mean-Shift: seed from every point instead of a capped 100-point random subset (which could miss clusters on larger data), and report `n_samples_per_center` as the samples actually assigned to each converged center. Fitting is now deterministic.

### Removed
- **Breaking:** remove `MeanShift`'s `random_state` constructor parameter, field, and `get_random_state` getter, since seeding from every point made `fit` deterministic (`MeanShift::new` now takes five arguments). The standalone `estimate_bandwidth` keeps its own independent `random_state`.
- Remove LDA's internal `cov_inv` field (the cached covariance inverse), now that the solver produces per-class scoring coefficients directly. The field was never publicly exposed; removing it changes the serialized model layout.
- Remove the now-unused `machine_learning::parallel::try_map_collect` helper (and its tests); `SVC`'s batched kernel path no longer needs the per-pair fallible parallel map.

## [v0.12.0] - 2026-06-09 (UTC-7)
### Added
- Add a crate-level `random` module for reproducible pseudo-random number generation. `set_global_seed(u64)` / `clear_global_seed()` (re-exported at the crate root) set a thread-local global seed, and an internal `make_rng` resolves each component's `random_state: Option<u64>` against it: an explicit `Some(seed)` is used as-is, `None` derives a sub-seed from the global stream when one is set, otherwise it falls back to entropy. This gives one-call whole-crate reproducibility with per-component override (local-over-global, mirroring Keras).
- Add a `random_state: Option<u64>` parameter to every `neural_network` layer constructor (`Dense`, `Conv*`, `DepthwiseConv2D`, `SeparableConv2D`, `SimpleRNN`, `LSTM`, `GRU`, the `Dropout`/`SpatialDropout*` and `GaussianNoise`/`GaussianDropout` layers) for reproducible weight init and dropout/noise masks, plus `Sequential::new_with_seed(u64)` and `Sequential::set_seed(u64)` for a reproducible fit-time minibatch shuffle.
- Add `tests/neural_network/reproducibility.rs` covering same-seed determinism, seed divergence, global-seed reproducibility, and local-overrides-global precedence.
- Rewrite the entire integration test suite under a per-feature "Route C" layout — `tests/<feature>/main.rs` crate roots with per-topic submodules and a shared `common.rs`, gated by `required-features` (`autotests = false`) — replacing the flat `*_test.rs` files. Expected values are derived from independent ground truth rather than traced from the implementation, and coverage now spans error paths, data-size-dependent parallel branches, and private numerical kernels.

### Changed
- **Breaking:** every `neural_network` layer constructor listed above takes a new trailing `random_state: Option<u64>` argument. Stochastic layers now own a seeded RNG, and all weight init, dropout/noise masks, and the `Sequential` minibatch shuffle draw through `crate::random::make_rng` instead of an unseeded thread RNG; `None` preserves the previous non-deterministic behavior.
- Route the `machine_learning` / `utils` estimators that own a `random_state` (`KMeans`, `MeanShift`, `IsolationForest`, `SVC`, `LinearSVC`, t-SNE, `train_test_split`) through `crate::random::make_rng` too, replacing their inline `match random_state` blocks. A `None` seed now honors `set_global_seed`, so one call governs the whole crate's randomness. PCA and Kernel PCA are left as-is: their only randomness is in `linalg`'s iterative eigensolvers (seeded with a fixed constant for determinism) and PCA's always-user-supplied `SVDSolver::Randomized(u64)`.
- **Breaking:** rename `KMeans`'s `random_seed` field, `new()` parameter, and `get_random_seed()` getter to `random_state` / `get_random_state()`, matching every other estimator and scikit-learn. (The constructor argument is positional, so only the getter name changes for callers.)
- Make `DecisionTree`'s `random_state` functional instead of a reserved no-op: it now seeds scikit-learn-style random tie-breaking among equally-scoring splits. With a seed in effect, tied splits are chosen randomly but reproducibly; with `None` and no global seed the tree stays deterministic. Adds the internal `crate::random::make_rng_opt` helper for "randomize only when a seed is in effect" callers.

### Fixed
- `roc_auc`, `roc_curve`, `precision_recall_curve`, and `average_precision` no longer hang or exhaust memory on a `NaN` score: the equal-score tie-grouping loop used `==`, which never advances past a `NaN` (since `NaN == NaN` is `false`). Ties are now grouped with a `NaN`-aware equality, ordering a `NaN` score deterministically.
- `variance` and `standard_deviation` (`math`) now skip non-finite (`NaN`/`±∞`) values and compute over the finite subset, instead of `variance` short-circuiting to `0.0` and `standard_deviation` propagating `NaN`. `standard_deviation` is now defined as `sqrt(variance)`; an input with no finite values returns `0.0`, and all-finite inputs are unchanged.

## [v0.12.0] - 2026-06-08 (UTC-7)
### Added
- Add the `adjusted_rand_index` (Adjusted Rand Index) and `silhouette_score` (mean silhouette coefficient, Euclidean) clustering metrics, completing the clustering set the crate documentation already advertised.
- Add multi-class classification support: a `MulticlassConfusionMatrix` (K x K counts, per-class precision/recall/F1/support, an `Average` enum for macro/micro/weighted aggregation, and a `classification_report`-style `summary()`), plus `log_loss`, `cohen_kappa`, `top_k_accuracy`, `average_precision`, `roc_curve`, and `precision_recall_curve`. The binary `ConfusionMatrix` gains `mcc` and `balanced_accuracy`.
- Add the regression metrics `explained_variance_score`, `median_absolute_error`, and `mean_absolute_percentage_error`.
- Add the clustering metrics `homogeneity_score`, `completeness_score`, `v_measure_score`, `fowlkes_mallows_score` (entropy- and pairwise-based external metrics), and the internal indices `davies_bouldin_score` and `calinski_harabasz_score`.
- Export all of the above from `metric_prelude`.

### Changed
- Split the `metric` module into public `regression`, `classification`, and `clustering` submodules, with every item also re-exported at the module root — so each metric is reachable both by category and flat, and existing flat paths are unchanged.
- Standardize every paired metric on `(y_true, y_pred)` argument order (ground truth first, matching scikit-learn and the clustering metrics). **Breaking:** this swaps the argument order of `r2_score` and `ConfusionMatrix::new`; for the symmetric metrics (MSE, RMSE, MAE, accuracy) only the parameter names change.
- Rename `calculate_auc` to `roc_auc` and reorder its arguments to `(labels, scores)`.
- Make metric panic messages mirror the crate's `Error` wording (`dimension mismatch: expected .., found ..`, `input is empty: ..`), and panic uniformly on empty input — the regression metrics previously returned `0.0` for empty arrays.
- `r2_score` now returns `1.0` for a perfect fit on zero-variance ground truth (previously always `0.0`), matching scikit-learn.
- `roc_auc` sorts scores with `total_cmp`, so it no longer panics on `NaN` scores; `normalized_mutual_info` / `adjusted_mutual_info` no longer panic on non-contiguous array views.
- Optimize the Adjusted Mutual Information's expected-MI term with a shared log-factorial table, turning each binomial coefficient into an `O(1)` lookup.
- Derive `Debug, Clone, Copy, PartialEq, Eq` for `ConfusionMatrix`, and render `ConfusionMatrix::summary` as an aligned table.
- Rename modules for naming consistency (plural collections, abbreviations, de-stuttering). **Breaking:** `metric` → `metrics`, `utility` → `utils`; under `neural_network`, `layer` → `layers`, `optimizer` → `optimizers`, `loss_function` → `losses`, `neural_network_trait` → `traits`, and the `layers` category submodules drop their `_layer` suffix; `machine_learning::meanshift` → `mean_shift`, `linear_discriminant_analysis` → `lda`, `utils::principal_component_analysis` → `pca`; the `prelude` submodules drop their `_prelude` suffix. The two Cargo features gating a renamed module are renamed to match: **`utility` → `utils`** and **`metric` → `metrics`**, so downstream crates must enable `features = ["utils"]` / `["metrics"]`.
- Rename the `LossFunction` trait to `Loss` (now in `neural_network::traits`), aligning the three core abstractions with their now-plural modules: `layers`/`Layer`, `optimizers`/`Optimizer`, `losses`/`Loss`. **Breaking.**

### Removed
- Remove the now-unused `ActivationLayer` trait from `neural_network`; its forward/derivative dispatch is fully served by the serializable `Activation` enum.

## [v0.12.0] - 2026-06-07 (UTC-7)
### Changed
- Refactor error handling into a single unified `Error` type built on `thiserror`, replacing the stringly-typed `ModelError` and separate `IoError`. Adds structured shared variants in place of the `InputValidationError` / `ProcessingError` catch-alls, with domain-specific failures grouped into nested `NnError`, `TreeError`, and `IoError` sub-enums, plus smart constructors, a `Context` extension trait (`.context()` / `.with_context()`), and a `RustymlResult<T>` alias. **Breaking:** `ModelError` is renamed `Error`, its variants are restructured, and the type is now `#[non_exhaustive]` and no longer derives `PartialEq` / `Clone`.
- Refactor the entire `neural_network` module for quality, correctness, and consistency (a net reduction of ~1360 lines while adding features). Replaces the `T: ActivationLayer` generic with a serializable `Activation` enum; adds a generic optimizer interface (`Layer::parameters() -> Vec<ParamGrad>` plus flat-slice `sgd`/`adam`/`rmsprop`/`adagrad` kernels) that removes all per-layer/per-optimizer update code; adds an inference-mode `Layer::predict(&self)` / `Sequential::predict(&self)`; replaces per-rank convolution/pooling code with dimension-generic engines; adds channel-last (NHWC) Instance/Group normalization and multi-axis `LayerNorm`; uses a Gram-Schmidt orthogonal recurrent-kernel initializer; makes the loss trait return `Result` instead of `assert!` panics; and splits `helper_function.rs` into `shape_helpers` / `conv_op_helpers` / `validation`.
- Make `cargo doc` warning-free: drop 37 redundant explicit targets on `[`Layer::predict`]` intra-doc links, and fully-qualify the unresolved `RegularizationType` / `DistanceCalculationMetric` links in the `types` module-level docs.

## [v0.12.0] - 2026-06-06 (UTC-7)
### Changed
- Refactor the `utility` module: add shared `validation` and `linalg` (power iteration plus a new pure-Rust Lanczos solver) submodules, removing duplicated validation and power-iteration copies across PCA and Kernel PCA, and move per-variant computation onto the config enums. Kernel PCA gains an `EigenSolver::Lanczos` variant and renames the mislabeled `ARPACK` solver to `PowerIteration` (likewise for PCA); t-SNE drops save/load on a stateless model and vectorizes its momentum update; `label_encoding` now returns `Result` instead of panicking; `train_test_split` gains a generic label type; and `utility` switches to explicit re-exports.
- Move `LinearDiscriminantAnalysis` from `utility` to `machine_learning`, where as a supervised classifier it belongs. It now implements the shared `Fit` / `Predict` traits, reuses `machine_learning::validation`, and moves per-solver logic onto the `Solver` enum. **Breaking:** LDA's import path changes from `utility` to `machine_learning`, and the `machine_learning` feature now enables `nalgebra`.
- Collapse nested `if`s into edition-2024 let-chains (`clippy::collapsible_if`) in `isolation_forest`, `kmeans`, and `knn`.

## [v0.12.0] - 2026-06-05 (UTC-7)
### Added
- Add `hinge_loss` to the `math` module (mean hinge loss for margin-based classifiers), alongside `logistic_loss`, and export it from the math prelude.

### Changed
- Encapsulate decision tree per-algorithm behaviour as methods on the `Algorithm` enum (impurity criterion, split-selection score, and capability checks), replacing the `match self.algorithm` branches scattered across `DecisionTree`. The now-exhaustive matches turn adding a new algorithm into a compile-time checklist.
- Document the membership rule for the `math` module (pure, model-agnostic, reusable primitives that are shared by more than one caller) in its module-level docs.
- `LinearSVC` now computes its training cost through `math::hinge_loss` instead of an inline hinge sum.
- `MeanShift` now computes its RBF neighbour weights through `KernelType::RBF`, sharing the single kernel-dispatch implementation in `types` (mirroring how the distance metrics are already dispatched).
- `metric::r2_score` now reuses `math::sum_of_squared_errors` and `math::sum_of_square_total` instead of recomputing SSE/SST inline. The `metric` feature now enables `math`.
- Update dependencies and raise the minimum supported Rust version to 1.89.0: `nalgebra` 0.34.1 -> 0.35.0 (required by the PCA / LDA / Kernel PCA solvers), `rayon` 1.11 -> 1.12, and `serde_json` 1.0.149 -> 1.0.150. 1.89 is the true minimum (via nalgebra 0.35 -> simba 0.10).

### Removed
- Remove the unused `information_gain` and `gain_ratio` functions from the `math` module; the decision tree computes its split criteria directly.
- Move `binary_search_sigma` out of the public `math` API into the t-SNE module as an internal helper. It is a t-SNE-specific perplexity solver, not a reusable primitive.

## [v0.12.0] - 2026-02-16 (UTC-7)
### Added
- Add Chinese README.zh-CN.md, separating it from English README.md.
- Add bilingual language switch links to README.md and README.zh-CN.md

### Removed
- dataset module is moved to [dataset-core](http://crates.io/crates/dataset-core/0.1.0) crate

## [v0.12.0] - 2026-02-15 (UTC-7)
### Added
- add conditional progress bar support using the `show_progress` feature flag

## [v0.11.0] - 2026-02-14 (UTC-7)
### Removed
- remove rand dependency and use the built-in rand module in ndarray_rand

## [v0.11.0] - 2026-02-12 (UTC-7)
### Changed
- move preludes of `dataset` module to `dataset_prelude`
- move preludes of `neural_network` module to `neural_network_prelude`

## [v0.11.0] - 2026-02-11 (UTC-7)
### Changed
- move preludes of `metric` module to `metric_prelude`

## [v0.11.0] - 2026-02-10 (UTC-7)
### Changed
- move preludes of `math` module to `math_prelude`

## [v0.11.0] - 2026-02-09 (UTC-7)
### Changed
- move preludes of `utility` module to `utility_prelude`

## [v0.11.0] - 2026-02-08 (UTC-7)
### Changed
- move preludes of `machine_learning` module to `machine_learning_prelude`

## [v0.11.0] - 2026-02-07 (UTC-7)
### Changed
- Move test modules for `math` and `metric` from `./src/test/` to `./tests/`

## [v0.11.0] - 2026-02-06 (UTC-7)
### Changed
- Move test modules for `utility` from `./src/test/` to `./tests/`

## [v0.11.0] - 2026-02-05 (UTC-7)
### Changed
- Move test modules for `neural_network` from `./src/test/` to `./tests/`

## [v0.11.0] - 2026-02-04 (UTC-7)
### Changed
- Move test modules for `machine_learning` from `./src/test/` to `./tests/`

## [v0.11.0] - 2026-02-03 (UTC-7)
### Changed
- Refactor imports in `neural_network`, `math`, and `metric`

## [v0.11.0] - 2026-02-02 (UTC-7)
### Changed
- Refactor imports in `utility`

## [v0.11.0] - 2026-02-01 (UTC-7)
### Changed
- Refactor imports in `dataset`

## [v0.11.0] - 2026-01-31 (UTC-7)
### Changed
- Refactor imports in `machine_learning`

## [v0.11.0] - 2026-01-30 (UTC-7)
### Added
- Add `Cosine` kernel support to `KernelType`

## [v0.11.0] - 2026-01-29 (UTC-7)
### Changed
- Relocate `KernelType` and update relevant modules

## [v0.11.0] - 2026-01-28 (UTC-7)
### Changed
- Refactor and simplify `LDA` and `t-SNE` implementations

### Removed
- Remove `linear_discriminant_analysis` module from `machine_learning`

## [v0.11.0] - 2026-01-27 (UTC-7)
### Changed
- Refactor and update `KernelPCA` implementation in `utility` module

## [v0.11.0] - 2026-01-26 (UTC-7)
### Changed
- Refactor and update `PCA` implementation in `utility` module

## [v0.11.0] - 2026-01-25 (UTC-7)
### Changed
- Refactor and update `LDA` implementation in `utility` module

## [v0.11.0] - 2026-01-24 (UTC-7)
### Changed
- Refactor and update `t_sne` implementation in `utility` module

## [v0.11.0] - 2026-01-23 (UTC-7)
### Changed
- Update documentation of `neural_network` module
- Refactor `serialize_weight` module for better maintainability

## [v0.11.0] - 2026-01-22 (UTC-7)
### Changed
- Restrict visibility of validation helper functions
- Update documentation of `math` module
- Update documentation of `metric` module

## [v0.11.0] - 2026-01-21 (UTC-7)
### Changed
- Update documentation and error handling of `utility` module

## [v0.11.0] - 2026-01-20 (UTC-7)
### Changed
- Update documentation of `machine_learning` module

## [v0.10.0] - 2026-01-19 (UTC-7)
### Changed
- Introduce comprehensive input validation for optimizers and layers

## [v0.10.0] - 2026-01-18 (UTC-7)
### Removed
- Remove `statrs` dependency and replace with custom hypergeometric PMF/log-binomial calculations.

## [v0.10.0] - 2026-01-17 (UTC-7)
### Changed
- update usage examples in `lib.rs` and `README.md`
- Refactor metric functions to use generic `ArrayBase` for increased flexibility

### Removed
- Remove `rand_distr` dependency

## [v0.9.1] - 2026-01-16 (UTC-7)
### Added
- Add Gaussian Dropout layer

## [v0.9.1] - 2026-01-15 (UTC-7)
### Added
- Add Gaussian Noise layer

### Changed
- Refactor weight serialization with dedicated macros for activation-based and simple layers.

## [v0.9.1] - 2026-01-13 (UTC-7)
### Added
- Add Group Normalization layer

### Changed
- Refactor AdaGrad parameter update logic in convolutional layers by introducing a reusable macro to reduce duplication and improve maintainability
- Refactor forward pass in `train_batch` and `forward` to prevent unnecessary cloning.

## [v0.9.1] - 2026-01-12 (UTC-7)
### Added
- Add Instance Normalization layer

### Changed
- Update dependencies and improve random initialization error handling

## [v0.9.0] - 2025-10-22 (UTC-7)
### Added
- Introduce `LayerNormalization` layer

## [v0.9.0] - 2025-10-21 (UTC-7)
### Changed
- Modularize weight serialization by introducing dedicated files for each layer type

## [v0.9.0] - 2025-10-20 (UTC-7)
### Added
- Add Batch Normalization layer

## [v0.9.0] - 2025-10-19 (UTC-7)
### Changed
- Change the parameters that require arrays in reference to use a more general solution for `utility` module

## [v0.9.0] - 2025-10-18 (UTC-7)
### Added
- Introduce `AdaGrad` (Adaptive Gradient Algorithm) optimizer

### Changed
- Change optimizer computation to adaptive parallel thresholds
- Change the parameters that require arrays in reference to use a more general solution for `math` module
- Change the parameters that require arrays in reference to use a more general solution for `machine_learning` module
- Streamline tensor handling and improve training stability in recurrent and dense layers

## [v0.9.0] - 2025-10-17 (UTC-7)
### Added
- Introduce regularization layers with Dropout and SpatialDropout support

## [v0.9.0] - 2025-10-16 (UTC-7)
### Added
- The activation function implements the `Layer` trait and can be called as a layer

## [v0.9.0] - 2025-10-15 (UTC-7)
### Added
- Add support for GRU layers in the neural network module

### Changed
- Improve handling of empty arrays in `sum_of_square_total` and `standard_deviation` functions, and optimize probability distribution normalization
- Organize the pooling layers, recurrent layers, and the convolutional layers under their respective modules

## [v0.9.0] - 2025-10-14 (UTC-7)
### Removed
- Remove `Result` type from numerical functions

## [v0.9.0] - 2025-10-13 (UTC-7)
### Added
- Introduce adaptive parallel processing thresholds in layers of `neural_network` module

### Changed
- Enhance error handling and input validation across ML models
- Enhance error handling and input validation in `utility` module
- Refactor gradient merging in pooling layers by introducing macros for 1D, 2D, and 3D operations

## [v0.9.0] - 2025-10-12 (UTC-7)
### Changed
- Update default feature set to include `machine_learning` and `neural_network`

## [v0.8.0] - 2025-10-11 (UTC-7)
### Added
- Add serialization and deserialization support across ML models, add `save_to_path` and `load_from_path` functions for ML models
- Add serialization and deserialization support in the utility module, add `save_to_path` and `load_from_path` functions in the utility module
- Add serialization and deserialization support for neural network layers, add `save_to_path` and `load_from_path` functions for `Sequential` model

## [v0.8.0] - 2025-10-9 (UTC-7)
### Added
- Introduce progress bar support across ML models
- Introduce progress bar support for the utility module
- Introduce progress bar support to neural network

### Changed
- Refactor ML models
- Refactor utilities

## [v0.8.0] - 2025-10-8 (UTC-7)
### Changed
- Refactor convolutional and pooling layers

## [v0.8.0] - 2025-10-7 (UTC-7)
### Changed
- Refactor the implementation code of the `DecisionTree`
- Improve module-level documentations
- Reconstruct the implementation of `IsolationForest`
- Encode labels as indices in `KNN` for efficient computation
- Introduce parallelization thresholds across machine learning implementations
- Refactor getter methods using macros for consistency and reduced duplication in utility modules
- Refactor LSTM layer and tests

## [v0.8.0] - 2025-10-6 (UTC-7)
### Removed
- Remove `#[doc(cfg)]` because it is still experimental

## [v0.8.0] - 2025-10-1 (UTC-7)
### Changed
- Refactor getter methods across machine learning models using `get_field` and `get_field_as_ref` macros for consistency and reduced code duplication

## [v0.8.0] - 2025-9-30 (UTC-7)
### Changed
- Refactor distance computation methods to return `Result`

## [v0.8.0] - 2025-9-27 (UTC-7)
### Added
- Implement `normalize` utility with support for L1, L2, Lp, and Max normalization

### Changed
- Refactor `param_count` method across layers to use `TrainingParameters` enum
- Refactor dataset loading functions to return merged headers and rows as single `&'static str`
- Update `normalize` function to accept `ArrayBase`
- Implement `standardize` function with flexible axis-based standardization (Row/Column/Global)

## [v0.7.0] - 2025-9-26 (UTC-7)
### Changed
- Refactor imports across neural network and utility modules to use `super::*` format, streamline dependencies, and clean up unused imports. Update import paths to simplify code structure and improve maintainability

## [v0.7.0] - 2025-9-25 (UTC-7)
### Added
- Add `load_boston_housing_raw_data` function to provide static raw Boston housing dataset and headers
- add batch processing for `fit` in Sequential model
- Add `Linear` activation function to support identity transformations
- Add `sequential_test` module with comprehensive tests for `Sequential` neural network functionality
- Add `#[inline]` attribute to performance-critical functions for potential inlining optimization in `metric.rs`
- Add comprehensive tests for `apply_activation_inplace` and `activation_derivative_inplace`, including edge cases and panics for unsupported operations
- Add `label_encoding` module for conversions between sparse and categorical formats
- Add comprehensive tests for LSTM, Dense, SimpleRNN, and Sequential modules
- Derive `Copy` and `Clone` for `Activation` enum to enable value duplication without ownership transfer

### Changed
- Refactor imports and expand wildcard usage
- improve modular organization of utility and dataset modules
- Refactor activation handling: consolidate `apply_activation` and `activation_derivative` logic into specialized in-place methods
- Consolidate `helper_functions` into `layer.rs` for improved modularity and maintainability. Refactor layer imports to remove dependency on `helper_functions`
- Update README for features clarification

## [v0.7.0] - 2025-9-24 (UTC-7)
### Changed
- Refactor dataset loading functions to use `OnceLock` for thread-safe memoization and add owned copy variants

### Added
- Add `load_titanic_raw_data` function to provide static raw Titanic dataset and headers
- Add cost calculation and reporting for ML models

## [v0.7.0] - 2025-9-21 (UTC-7)
### Added
- Add `load_diabetes_raw_data` function to provide static raw diabetes dataset and headers

## [v0.7.0] - 2025-9-19 (UTC-7)
### Added
- Annotate mathematical utility functions with `#[inline]` for potential performance improvements during compilation
- add feature flags for selective compilation

### Changed
- Replace trait-based getter implementation with macros for improved reusability and consistency
- Refactor imports
- modularize traits
- Replace `HashSet` with `AHashSet` in DBSCAN and LDA

## [v0.6.3] - 2025-9-16 (UTC-7)
### Changed
- Refactor models in utility and machine_learning for efficiency, maintainability, and clarity

## [v0.6.3] - 2025-9-13 (UTC-7)
### Changed
- Standardize doc comments

## [v0.6.3] - 2025-8-30 (UTC-7)
### Changed
- Improve input validation and error handling in `Sequential` model
- Improve input validation and error handling across mathematical utilities
- Enhance input validation, edge case handling, and error reporting across clustering and classification algorithms

## [v0.6.3] - 2025-8-23 (UTC-7)
### Added
- Add doc comments for `SeparableConv2D` and `DepthwiseConv2D` layer weights
- Add lifetime parameter to `get_weights` return type across layers

### Changed
- Update dependencies(`rand`, `rayon`, and `nalgebra`)

## [v0.6.2] - 2025-6-5 (UTC-7)
### Added
- Refactor activation handling for convolutional layers

## [v0.6.2] - 2025-6-4 (UTC-7)
### Added
- Add `DepthwiseConv2D` layer and related utilities

## [v0.6.2] - 2025-6-3 (UTC-7)
### Added
- Add `GlobalMaxPooling3D` layer and tests
- Add `GlobalAveragePooling3D` layer and tests
- Add input dimensionality checks for pooling layers
- Add `Conv3D` layer and optimizer support
- Add input dimensionality checks for `Conv1D` and `Conv2D` layers
- Add `SeparableConv2D` layer and related utilities

### Changed
- Refactor to use `layer_functions_global_pooling` macro
- Separate `layer_weight` and `padding_type` into dedicated modules
- Add support for Flatten layer with 3D, 4D, and 5D tensors
- Replace `HashMap` and `HashSet` with `AHashMap` and `AHashSet`

## [v0.6.2] - 2025-6-2 (UTC-7)
### Added
- Add `AveragePooling3D` layer and tests

### Changed
- Refactor pooling layers to use macros for output shape calculation

## [v0.6.2] - 2025-6-1 (UTC-7)
### Changed
- Refactor layers without trainable parameters to use `no_trainable_parameters_layer_functions!` macro

## [v0.6.2] - 2025-5-31 (UTC-7)
### Added
- Add `MaxPooling3D` layer and tests

## [v0.6.2] - 2025-5-30 (UTC-7)
### Added
- Add `Conv1D` layer implementation and tests

### Changed
- Refactor tests to reuse `generate_data` function for pooling layers
- Rename `OptimizerCacheFEL` to `OptimizerCacheConv2D`
- Extracted the SGD parameter update logic into a `update_sgd_conv` macro to eliminate redundancy. Updated `Conv1D` and `Conv2D` layers to use the macro.
- Extracted the Adam parameter update logic into a `update_adam_conv` function to eliminate redundancy. Updated `Conv1D` and `Conv2D` layers to use the new method for weight and bias updates.

## [v0.6.2] - 2025-5-29 (UTC-7)
### Added
- Add `GlobalAveragePooling1D` layer implementation

## [v0.6.2] - 2025-5-28 (UTC-7)
### Added
- Add `GlobalMaxPooling1D` layer and corresponding tests

### Changed
- Change function `preliminary_check` from public into private
- PCA no longer requires `preliminary_check` function and integrates input validation functionality

## [v0.6.2] - 2025-5-27 (UTC-7)
### Added
- Add tests for `MaxPooling1D` layer
- Add more doc comments for `MaxPooling1D` layer

### Changed
- Introduced a shared `compute_output_shape` function to streamline output shape calculations across pooling layers.
- Centralized 1D and 2D pooling output shape logic into reusable helper functions (`calculate_output_shape_1d_pooling` and `calculate_output_shape_2d_pooling`)

## [v0.6.2] - 2025-5-26 (UTC-7)
### Added
- Add `MaxPooling1D` layer implementation

## [v0.6.2] - 2025-5-25 (UTC-7)
### Added
- Add more test functions for neural network

### Changed
- move all test functions to `test` module

## [v0.6.2] - 2025-5-24 (UTC-7)
### Changed
- Refactor and document `AveragePooling1D` module

## [v0.6.2] - 2025-5-23 (UTC-7)
### Added
- Add `AveragePooling1D` layer and corresponding tests

## [v0.6.1] - 2025-5-22 (UTC-7)
### Added
- Add doc comments for modules

### Changed
- Optimize parameter updates of struct `Conv2D` with parallelization

## [v0.6.1] - 2025-5-21 (UTC-7)
### Added
- Add tests for `GlobalAveragePooling2D` layer

### Changed
- Refactor `GlobalAveragePooling2D` with improved comments and docs
- Refactor global max pooling to leverage parallel processing

## [v0.6.1] - 2025-5-20 (UTC-7)
### Added
- Add `GlobalAveragePooling2D` layer implementation

## [v0.6.1] - 2025-5-19 (UTC-7)
### Added
- Add comprehensive weight structs for neural network layers

## [v0.6.1] - 2025-5-18 (UTC-7)
### Changed
- Refactor `GlobalMaxPooling2D` with improved comments and docs

## [v0.6.1] - 2025-5-17 (UTC-7)
### Added
- Add `GlobalMaxPooling2D` layer initial implementation

## [v0.6.1] - 2025-5-16 (UTC-7)
### Changed
- Rename `AveragePooling` to `AveragePooling2D` for clarity

## [v0.6.1] - 2025-5-15 (UTC-7)
### Added
- Add detailed comments and example to `AveragePooling` layer

## [v0.6.1] - 2025-5-14 (UTC-7)
### Added
- Add `AveragePooling` layer implementation and corresponding tests

## [v0.6.1] - 2025-5-13 (UTC-7)
### Added
- Update comments and documentation in `Flatten` layer

## [v0.6.1] - 2025-5-12 (UTC-7)
### Added
- Add `Flatten` layer initial implementation and associated tests

## [v0.6.1] - 2025-5-11 (UTC-7)
### Added
- Add detailed documentation and usage example for `MaxPooling2D`

## [v0.6.1] - 2025-5-10 (UTC-7)
### Added
- Add `MaxPooling2D` layer initial implementation

## [v0.6.1] - 2025-5-9 (UTC-7)
### Changed
- Rename `OptimizerCacheFEX` to `OptimizerCacheFEL` (FEL stands for feature extraction layer)

## [v0.6.1] - 2025-5-8 (UTC-7)
### Added
- Add complete test function for `Conv2D`
- Add comprehensive docstrings for `Conv2D` layer and methods

## [v0.6.1] - 2025-5-7 (UTC-7)
### Added
- Add `Debug`, `Clone`, and `Default` traits to optimizer structs

### Changed
- Use parallelized computation for performance improvement
- Replaced multiple optimizer-specific fields with a unified `optimizer_cache` structure
- Refactor optimizer cache initialization and parameter flattening
- Refactor optimizer caching to support feature extraction layers
- Refactor SGD parameter updates with parallelized helper methods

## [v0.6.1] - 2025-5-6 (UTC-7)
### Added
- Add Conv2D layer(initial implementation) support to neural_network module

### Changed
- Rename layer naming convention in Sequential model

### Removed
- Remove some default implementations in `Layer` trait

## [v0.6.0] - 2025-5-5 (UTC-7)
### Added
- Add `get_weights` method across layers and `LayerWeight` enum

## [v0.6.0] - 2025-5-4 (UTC-7)
### Changed
- Refactor optimizer state handling with unified cache

## [v0.6.0] - 2025-5-3 (UTC-7)
### Added
- Ensure `fit` validates optimizer and layers before training

### Changed
- Refactor parameter updates to use parallel processing

### Removed
- Remove method `update_parameters` in `Layer` trait
- Remove getter methods from Dense layer

## [v0.6.0] - 2025-5-2 (UTC-7)
### Changed
- Refactor RMSprop implementation with unified cache structure

## [v0.6.0] - 2025-5-1 (UTC-7)
### Added
- Add detailed documentation and examples to neural network layers

### Changed
- Refactor layers to enforce explicit activation usage
- Optimize LSTM computations with parallel processing using Rayon
- Refactor LSTM to consolidate gate logic into reusable structures
- Refactor Adam optimizer state management into `AdamStates`

## [v0.6.0] - 2025-4-30 (UTC-7)
### Added
- Add doc comment for `LSTM`

### Removed
- Remove redundant method documentation comments

## [v0.6.0] - 2025-4-29 (UTC-7)
### Added
- Add LSTM layer initial implementation and associated tests

## [v0.6.0] - 2025-4-28 (UTC-7)
### Changed
Refactor traits in neural_network module definitions into traits module

## [v0.6.0] - 2025-4-27 (UTC-7)
### Added
- Add doc comment for traits module

## [v0.6.0] - 2025-4-26 (UTC-7)
### Changed
- Refactor `LinearSVC` to use `RegularizationType` for penalties

## [v0.6.0] - 2025-4-25 (UTC-7)
### Changed
- Refactor regressors to use shared `RegressorCommonGetterFunctions` trait

## [v0.6.0] - 2025-4-24 (UTC-7)
### Added
- Add regularization support to linear and logistic regression

### Changed
- Refactor `KernelType` into machine_learning module

## [v0.5.1] - 2025-4-23 (UTC-7)
### Changed
- Modularize activation functions into a separate module

## [v0.5.1] - 2025-4-22 (UTC-7)
### Changed
- Optimize activation functions with parallel processing

## [v0.5.1] - 2025-4-21 (UTC-7)
### Changed
- Refactor optimizer implementations into separate files

## [v0.5.1] - 2025-4-20 (UTC-7)
### Changed
- Refactor loss functions into separate modules

## [v0.5.1] - 2025-4-19 (UTC-7)
### Added
- Add doc comments for layer `SimpleRNN`

## [v0.5.1] - 2025-4-18 (UTC-7)
### Changed
- Modularize `layer` into separate `dense` and `simple_rnn` modules

## [v0.5.1] - 2025-4-17 (UTC-7)
### Added
- Add `SimpleRNN` layer initial implementation

### Changed
- Update to rand v0.9.1

## [v0.5.1] - 2025-4-16 (UTC-7)
### Changed
- Optimize neural network computations with parallelism

## [v0.5.1] - 2025-4-15 (UTC-7)
### Changed
- `ModelError` is used in the neural network implementation

## [v0.5.1] - 2025-4-14 (UTC-7)
### Changed
- Standardize documentation comments across modules

## [v0.5.0] - 2025-4-13 (UTC-7)
### Changed
- Replace `ndarray-linalg` with `nalgebra` for `PCA`, `LDA`, and `KernelPCA`

## [v0.5.0] - 2025-4-12 (UTC-7)
### Changed
- Refactor metrics API to remove `Result` usage and add panics

## [v0.5.0] - 2025-4-11 (UTC-7)
### Added
- Added getter methods for accessing key properties

### Changed
- Replaced public fields with private ones in `Dense`, `Adam`, and `RMSprop` structs to improve encapsulation.

## [v0.5.0] - 2025-4-10 (UTC-7)
### Added
- Add activation functions support to `Dense` layer

## [v0.4.0] - 2025-4-9 (UTC-7)
### Added
- Add RMSprop optimizer support to the neural network
- Add MAE(Mean Absolute Error) support in loss function computation
- Add doc comments for neural_network module

## [v0.4.0] - 2025-4-8 (UTC-7)
### Added
- Add `Adam` optimizer for neural network
- Added new loss functions: `CategoricalCrossEntropy` and `SparseCategoricalCrossEntropy`

### Changed
- Replaced slices with `ndarray`'s `ArrayView` to improve consistency and compatibility with numerical operations in metric module
- Refactor polynomial feature generation to exclude constant term

## [v0.4.0] - 2025-4-7 (UTC-7)
### Added
- Add neural network support(initial implementation)

## [v0.3.0] - 2025-4-6 (UTC-7)
### Added
- Add dataset module and put iris and diabetes datasets in it
- Enhance SVM/LDA documentation

## [v0.3.0] - 2025-4-5 (UTC-7)
### Changed
- Refactor to use `ArrayView` for memory-efficient data handling
- Refactor outputs to use `Array` instead of `Vec` in DBSCAN and KNN

## [v0.2.1] - 2025-4-4 (UTC-7)
### Added
- Add variance calculation in math module
- Add MSE calculation in metric module
- Print info after training completes for `fit` functions of struct `LDA`, `SVC`, `LinearSVC` and `PCA`
- Integrate Rayon for parallel computation across modules (**7 s faster** in `cargo test`!!!)

### Changed
- Refactor functions in math module and metric module to use `ArrayView1` for improved efficiency

### Removed
- Remove MSE calculation(named mean_squared_error, but actually calculate variance) in math module

## [v0.2.1] - 2025-4-3 (UTC-7)
### Added
- Add t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation

### Changed
- Remove gaussian kernel calculation function, put gaussian kernel calculation directly in function `fit` of struct `MeanShift`

## [v0.2.1] - 2025-4-2 (UTC-7)
### Added
- Add LinearSVC support
- Add KernelPCA support
- Add `ProcessingError(String)` to `crate::ModelError` 
- Add LDA(Linear Discriminant Analysis) support

### Changed
- Change the location of function `standardize` from `crate::utility::principal_component_analysis` to `crate::utility`

## [v0.2.1] - 2025-4-1 (UTC-7)
### Added
- Add SVC(Support Vector Classification) support

## [v0.2.0] - 2025-4-1 (UTC-7)
### Added
- Add `train_test_split` function in utility module to split dataset for training and dataset for test
- Add function `normalized_mutual_info` and `adjusted_mutual_info` to metric module to calculate NMI and AMI info
- Add AUC-ROC value calculation in metric module

## [v0.2.0] - 2025-3-31 (UTC-7)
### Changed
- Change `principal_component_analysis` module to `utility` module, change `principal_component_analysis_test` module to `utility_test` module
- Keep the algorithm functions in the math module, and move the functions that evaluate the model's performance (such as R-square values) and structures (confusion matrices) to the metric module. Some of them are used in both ways, then keep them in both modules.
- Change the output of some of the functions in math module and metric module from `T` to `Result<T, crate::ModelError>`

## [v0.1.1] - 2025-3-31 (UTC-7)
### Added
- Add function `preliminary_check` in machine_learning module to performs validation checks on the input data matrices
- Add confusion matrix in math module

### Changed
- Change type of field `coefficients` of struct `LinearRegression` from `Option<Vec<f64>>` to `Option<Array1<f64>>`
- Change the output of some methods of struct `LinearRegression` from `Vec<f64>` to `Array1<f64>`
- Change variant `InputValidationError` of enum type `ModelError` from `InputValidationError(&str)` to `InputValidationError(String)`

## [v0.1.0] - 2025-3-30 (UTC-7)
### Added
- Add function `fit_predict` for some models
- Add examples for functions in math.rs
- Add input validation
- Add doc comments for machine learning modules
- Add prelude module(all re-exports are there)

### Changed
- Change input types of function `fit`, `predict` and `fit_predict` to `Array1` and `Array2`
- Rename the crate from `rust_ai` to `rustyml`
- Change the output of function `fit` from `&mut Self` to `Result<&mut Self, ModelError>` or `Result<&mut Self, Box<dyn std::error::Error>>`

## [v0.1.0] - 2025-3-29 (UTC-7)
### Added
- Add function `generate_tree_structure` for `DecisionTree` to generate tree structure as string
- Add isolation forest implementation
- Add PCA(Principal Component Analysis) implementation
- Add function `standard_deviation` in math module to calculates the standard deviation of a set of values

## [v0.1.0] - 2025-3-28 (UTC-7)
### Added
- Add Decision Tree model
- Add following functions to math.rs:
    - `entropy`: Calculates the entropy of a label set
    - `gini`: Calculates the Gini impurity of a label set
    - `information_gain`: Calculates the information gain when splitting a dataset
    - `gain_ratio`: Calculates the gain ratio for a dataset split
    - `mean_squared_error`: Calculates the Mean Squared Error (MSE) of a set of values

### Changed
- Replaced string-based distance calculation method options with an enum `crate::machine_learning::DistanceCalculation`
- For KNN model: replaced string-based weight function options with an enum `crate::machine_learning::knn::WeightingStrategy`
- For decision tree: replaced string-based algorithm options with an enum `crate::machine_learning::decision_tree::Algorithm`

## [v0.1.0] - 2025-3-27 (UTC-7)
### Added
- Add changelog.md to record updates
- Add DBSCAN model
- Add function `fit_predict` to fit and predict in one step
- Add doc comments to tell user `p` value of function `minkowski_distance` in model is always 3

## [v0.1.0] - 2025-3-26 (UTC-7)
### Added
- Add "xx model converged at iteration x, cost: x" when finishing `fit`
- Add description for `n_iter` field
- Add getter functions for `KMeans`
- implement `Default` trait for `KMeans`

### Changed
- Rename `max_iteration` and `tolerance` to `max_iter` and `tol`
- Change doc comments to enhanced consistency

### Removed
- Remove examples in math.rs(add them back later)

## [v0.1.0] - 2025-3-25 (UTC-7)
### Added
- Add MeanShift model
- Add `InputValidationError` in `ModelError`, indicating the input data provided  does not meet the expected format, type, or validation rules
- Add `gaussian_kernel` in math module, calculate the Gaussian kernel (RBF kernel)

### Changed
- Change the output of all `predict` functions(except KNN) from `T` to `Result<T, crate::ModelError>`
- Correct doc comments
