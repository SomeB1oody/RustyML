# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [Unreleased] - 2026-06-13 (UTC-7)
### Added
- **More of the common trait derives across `machine_learning`, `utils`, and `neural_network` public types** (a derive-coverage audit; all additive, no behavior change). `KernelType` and `DecisionTreeParams` gain `PartialEq` (kernels and tree hyperparameters are now `==`-comparable; `KernelType` was the only strategy enum without it); the unit/integer strategy enums `Solver`, `WeightingStrategy`, `Algorithm`, `EigenSolver`, and `SVDSolver` gain `Eq` (aligning them with `Init`/`NormalizationAxis`/`StandardizationAxis`); `LayerNormalizationAxis` gains `PartialEq`/`Eq`; the pooling-engine `PoolKind` gains the `Debug` it was missing; and the five loss types — `MeanSquaredError`, `BinaryCrossEntropy`, `MeanAbsoluteError`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy` — which previously derived nothing, now derive `Debug, Clone, Copy, PartialEq, Eq, Default` (their hand-written `Default` impls, which only delegated to `new()`/`new(false)`, are folded into the derive). Layer and activation structs are intentionally left at `Debug` only (they hold weights/caches and are persisted via the weights-only save/load rather than cloned), and config enums keep no `serde` derives beyond what already existed, since layer hyperparameters are reconstructed in code, not serialized.

### Changed
- **Breaking: every `machine_learning` and `utils` estimator constructor keeps only its primary hyperparameters; secondary "optional / has-a-default" settings move to chainable `with_*` builder methods.** `new` now mirrors the parameters scikit-learn lists first (the ones tuned per dataset), so a new user is no longer confronted with a wall of `Option`/strategy arguments; the rest are opt-in. The split rule: a parameter stays in `new` if it is a primary tuning knob (`n_clusters`, `eps`/`min_samples`, `kernel` + `C`, `k`, `n_components`, `bandwidth`, `max_iter`/`tol`/`learning_rate`, `algorithm` + `is_classifier`, …); it moves to a `with_*` setter if it was an `Option` meaning "default/disabled" or a strategy enum with a near-universal default. Setters that validate their input return `Result<Self, Error>` (so construction stays fail-fast); trivially-valid setters return `Self`, so a typical chain reads `Model::new(req…)?.with_a(x)?.with_b(y)`. Per model — **LinearRegression**/**LogisticRegression**: `regularization_type` → `with_regularization`; **KMeans**/**SVC**/**LinearSVC**/**IsolationForest**/**TSNE**: `random_state` → `with_random_state`; **IsolationForest**: also `max_depth` → `with_max_depth`; **MeanShift**: `max_iter`/`tol`/`bin_seeding`/`cluster_all` → setters (only `bandwidth` stays — it dominates tuning); **DBSCAN**/**KNN**: `metric` → `with_metric` (and KNN `weighting_strategy` → `with_weighting_strategy`); **LDA**: `solver`/`shrinkage` → `with_solver`/`with_shrinkage`; **PCA**: `svd_solver` → `with_svd_solver`; **KernelPCA**: `eigen_solver` → `with_eigen_solver`; **TSNE**: also `init`/`method` → `with_init`/`with_method`; **DecisionTree**: the `Option<DecisionTreeParams>` bundle is dissolved into `with_max_depth`/`with_min_samples_split`/`with_min_samples_leaf`/`with_min_impurity_decrease`/`with_random_state`. Each `new`'s doc gains a `# Notes` section listing the available `with_*` methods and their defaults. **Behavior change: none numerically** — defaults are identical to the previous implicit ones and on-disk/serde formats are unchanged; this is a pure constructor-ergonomics refactor. Two validation timings shift to accommodate independent setters: `DecisionTree`'s cross-field `min_samples_leaf <= min_samples_split` check now runs at `fit` time (the two are set separately), and `TSNE`'s default gradient method is now Barnes-Hut for `n_components <= 3` and Exact otherwise (so `new` is always valid for any dimensionality), with the Barnes-Hut angle/dimensionality validation moving to `with_method`. The strategy enums these builders default to now also `#[derive(Default)]` on their canonical variant (`DistanceCalculationMetric::Euclidean`, `WeightingStrategy::Uniform`, `Solver::SVD`, `SVDSolver::Full`, `EigenSolver::Dense`), matching the existing `Init`/`TSNEMethod` defaults and enabling `..Default::default()` / `#[serde(default)]` use.
- **Breaking: clone-free model saving — `LayerWeight` borrows the live layer arrays via `Cow` and is now the single weight type for both inspection and serialization.** `save_to_path` previously cloned every weight array to build an owned serializable mirror, but serialization only needs read access. The per-layer weight structs and the `LayerWeight` enum now hold their arrays as `Cow<'a, _>` and derive `Serialize`/`Deserialize`: `Layer::get_weights` / `Sequential::get_weights` build them with `Cow::Borrowed` (zero clone, borrowing straight from the layer), saving serializes the borrowed view directly (no intermediate owned tree), and loading deserializes to `Cow::Owned` (`LayerWeight<'static>`). This also collapses what were two parallel 13-variant enums — the borrowing `LayerWeight` view and the owned `SerializableLayerWeight` serialization DTO became isomorphic once both used `Cow`, so they merge into one type that serves both directions, removing the "add a layer, update both enums plus the conversion" maintenance tax. **On-disk format is unchanged** — the `#[serde(tag = "type")]` variant names and field names are identical, so round-trip tests pass and previously-saved JSON loads unchanged; this is a pure refactor for saved models that pays off most under a future binary format, where the clone would dominate. **Breaking for code that names the weight types:** the `get_weights` variant payloads are now `Cow<'a, _>` instead of `&'a _` (read through the `Cow`'s `Deref`).
- The per-layer weight containers move into the `layer_weight` module as one file per layer (`layer_weight/dense_weight.rs`, `layer_weight/conv_2d_weight.rs`, ...), each holding the `Cow`-backed struct and its `ApplyWeights` impl. The `LayerWeight` enum and the public import paths (`layers::layer_weight::{LayerWeight, DenseLayerWeight, ...}`) are unchanged, so the 13 weight-carrying layers are untouched apart from `get_weights` wrapping its borrows in `Cow::Borrowed`, and the ~30 parameter-free layers (which return `LayerWeight::Empty`) are untouched entirely.
- **Breaking: the `serialize_weight` module is renamed `serialize_model`.** After the merge it no longer holds any weight types (those live in `layer_weight`); it keeps only the whole-model serialization scaffolding — `LayerInfo`, `SerializableLayer`, `SerializableSequential`, and the load-time `apply_weights_to_layer`. The symbol names are unchanged; only the module path moves (`layers::serialize_weight::*` → `layers::serialize_model::*`).

### Removed
- **Breaking:** the separate owned serialization types — the `SerializableLayerWeight` enum and the `Serializable*Weight` structs (`SerializableDenseWeight`, `SerializableConv2DWeight`, ...) — along with `SerializableLayerWeight::from_layer_weight`. They are subsumed by the now-`Cow`-backed `LayerWeight` family (`DenseLayerWeight`, `Conv2DLayerWeight`, ...), which serves both weight inspection and serialization; `get_weights` constructs it directly, so the old view→DTO conversion step is gone.

## [Unreleased] - 2026-06-12 (UTC-7)
### Added
- **Public deterministic blocked reductions: `math::reduction`.** The crate-internal `reduction` module moves into `math` as a public API alongside `math::matmul`: `det_par_fold` folds a `&[f64]` slice and the new `det_par_fold_range` folds an index range `0..n` — for reductions that zip several arrays or accumulate compound state (per-cluster sums, Welford moments) rather than a single scalar over one slice. Both cut the input into fixed `DET_REDUCE_BLOCK` (16Ki-element) blocks, fold each block serially, and merge the per-block results in block order, so the float result is **bitwise identical at any rayon thread count** (a bare rayon `sum`/`fold().reduce()` groups partials by work-stealing, making the rounding run-to-run nondeterministic). Measured against the bare rayon reductions the fixed grouping costs nothing — it is 2-3x *faster* on L3-resident inputs (uniform blocks balance better than rayon's adaptive splitting) and identical once memory-bandwidth-bound. Internal call sites address it as `crate::math::reduction::...` (no crate-root re-export).
- Two new calibration sections in `cargo bench --bench parallel_gates` (tables in `benches/RESULTS.md`): the exp-heavy logistic-loss reduction (crossover bracket 16K-32K elements, 14.3x at 1M) and the k-means assign-accumulate fold (work metric samples x features; never loses at 262_144 across d=8/d=32, 3.2-3.9x at 1-2M).
- New calibration sections in `cargo bench --bench parallel_gates` for the `f32` reduction classes: the f32-elements/f64-accumulator square-sum (crossover 32K-64K elements, 12.7x at 1M -> `SQ_SUM_F32_PARALLEL_MIN_ELEMS = 65_536`), an f32 `DET_REDUCE_BLOCK` validation sweep (16Ki elements sits mid-plateau for f32 too, so the constant stays shared across element types), and the BatchNorm column-stats shootout - including a kept **negative result**: a channel-chunked variant that would have preserved the serial accumulation order bitwise measured 0.3-0.9x everywhere and was rejected in favor of the row-block fold (crossover 64K-256K elements, 2.8-4.5x at 1-4M, 12x for narrow channel counts). `cargo bench --bench nn_end_to_end` gains `batchnorm_forward_32x64x64x64` to track spatial BatchNorm.
- A `parallel_gates` calibration section for the native-layout BatchNorm plane fold (forced serial vs parallel of the same fold; crossover bracket 64K-256K elements, 2.8-3.8x at 1M, 11.7x at the conv-scale 8.4M) and a `batchnorm_backward_32x64x64x64` end-to-end tracker alongside the forward one.
- A `parallel_gates` calibration section for the LayerNorm fused row pass (forced serial vs parallel of the same per-row sweep; crossover bracket 64K-256K elements, 2.5-4.1x at 1M, fading toward memory bandwidth at 12.6M -> `LN_ROW_PARALLEL_MIN_ELEMS = 262_144`) and three `nn_end_to_end` trackers: `layernorm_forward_32x512x768`, `layernorm_backward_32x512x768`, and `layernorm_forward_multi_32x64x64x64`.
- Three `nn_end_to_end` trackers for the group-normalization family: `groupnorm_forward_32x64x64x64_8g`, `groupnorm_backward_32x64x64x64_8g`, and `instancenorm_forward_32x64x64x64`. New core tests pin the group-norm row passes bitwise across the parallel flag (sub-8 spatial sizes, chunk-boundary-exact group sizes, instance-norm-like single-channel groups), the forward core to the reshape + broadcast reference exactly on integer data with a power-of-two group size, and both cores to the reference formulas to rounding on float data (including the plane-fold parameter gradients); a `channel_axis = 2` GroupNorm finite-difference check covers the permute bracket around the new core. New LayerNorm tests pin the row passes bitwise across the parallel flag (including `predict` == training output bit for bit), the row path to the broadcast reference exactly on integer data with power-of-two row lengths, identity-perm `Multiple` to a `Default`-axis layer on the reshaped input bit for bit (forward, backward, and parameter gradients), and permuted `Multiple` to an explicit merge -> `Default` -> unmerge composition bit for bit; five finite-difference gradient checks join the integration suite (rank-3 `Default`, trailing `Custom`, trailing and permuted `Multiple`). New BatchNorm tests pin the plane folds bitwise to a straight-line blocked reference for both flag values (including plane-crossing blocks and sub-plane ranges) and exactly to ndarray's axis reductions on integer data, spatial `predict` bit for bit to the fold -> broadcast -> unfold reference both below and above the parallel threshold, the spatial training forward bit for bit to the folded reference on exact (integer, power-of-two-count) statistics, and the spatial path to an equivalent 2-D layer to rounding; rank-4 BatchNorm finite-difference gradient checks join the integration suite.

### Changed
- **Breaking: `machine_learning`'s models are regrouped by algorithm family into submodules.** The eleven flat model modules move under seven family modules — `clustering` (`dbscan`, `kmeans`, `mean_shift`), `linear_model` (`linear_regression`, `logistic_regression`), `svm` (`svc`, `linear_svc`), `tree` (`decision_tree`), `neighbors` (`knn`), `discriminant_analysis` (`lda`), and `ensemble` (`isolation_forest`) — mirroring scikit-learn's module layout, which groups by family rather than by task precisely so multi-task estimators (`DecisionTree` does classification *and* regression, `LDA` classifies *and* reduces dimensionality) have one unambiguous home. The shared internals stay at the `machine_learning` root, since they are cross-family: the `Fit`/`Predict` `traits`, the `parallel`/`validation` helpers, and the `spatial` kd-tree (used by both `neighbors::knn` and `clustering::dbscan`). **Behavior change: none** — this is a pure module reorganization; every estimator is still re-exported flat as `machine_learning::<Model>` (and `prelude::machine_learning::*` is unchanged), so the recommended flat import paths and all runtime behavior are identical. **Breaking only for code that imported through a model's leaf path:** `machine_learning::dbscan::DBSCAN` is now `machine_learning::clustering::dbscan::DBSCAN` (or, preferred, the unchanged flat `machine_learning::DBSCAN`); `LinearSVC` moves under `svm` (matching `sklearn.svm.LinearSVC`), not `linear_model`. Models within a family are also reachable through the family namespace, e.g. `machine_learning::clustering::KMeans`.
- `math::logistic_loss` runs as a deterministic blocked parallel reduction above a measured exp-class gate (`EXP_REDUCE_MIN_ELEMS = 32_768` elements). **Behavior change:** at or above the gate the summation grouping changes, so results differ in the last bits from previous versions (and from the serial path below the gate) — but are bitwise reproducible at any thread count. End-to-end, `logistic_fit_50000x64_100it` drops 120 ms → 105 ms (~12%).
- K-means' per-iteration assignment accumulation (cluster sums, counts, inertia) runs as a deterministic blocked range fold above the sum gate on samples x features, replacing the serial per-sample scatter; the labels store becomes a chunked parallel write (a per-index map, no ordering involved), and the k-means++ seeding total-distance sum uses the same blocked fold above the gate (its roulette prefix walk is inherently serial and stays so). **Behavior change:** above the gates the accumulation grouping — and therefore centroid/inertia low bits — changes versus previous versions; results remain bitwise reproducible at any thread count. The benched 20000x32 fit sits at the measured break-even (unchanged, 20.4 ms); the win is large-dataset scaling. A new integration test pins the parallel fold to exact serial means on integer-valued data above the gate.
- Linear regression's per-iteration SSE and intercept-gradient sums use the deterministic blocked fold above the sum gate (`SUM_F64_PARALLEL_MIN_ELEMS`, now compiled for `machine_learning` as well as `utils`). **Behavior change:** the intercept gradient is on the optimization path, so trainings with >= 262_144 samples follow a (deterministically) different trajectory in the low bits.
- The remaining serial `f64` reductions in `machine_learning` now carry brief why-serial comments instead of bare loops: SVC's SMO error/alpha scans (the O(n^2) kernel matrix bounds n far below any gate), linear SVC's minibatch gradient accumulation (batch below the gate, vector accumulator), mean-shift's per-seed weight sum and the decision-tree node sums (nested inside an already-parallel loop), the isolation-forest path average (~10 terms), and LDA's shrinkage statistics (n_features^2 scale). Stale `# Performance` doc sections on `KMeans::fit`, `LinearRegression::fit`, and `LogisticRegression::fit` (one still referenced a long-removed 1000-sample threshold) now describe the actual gate classes.
- The same determinism audit over every rayon call site in `neural_network` found one violation: `SparseCategoricalCrossEntropy::compute_loss`'s probability path summed its per-sample `ln` terms with a bare parallel `sum`, so the reported loss value varied with thread scheduling (its logits path and every sibling loss already summed serially, and `compute_grad` — what actually drives training — was fully serial and unaffected). It now sums serially like the rest. Everything else checked out: the losses and batch-norm statistics reduce serially (`.sum()`/`mean_axis`/`sum_axis`), clip-by-global-norm accumulates in serial f64, dropout samples its mask serially from the seeded RNG before the parallel threshold map, pooling's per-window arg-max is a serial first-wins scan inside disjoint plane tasks, the conv/depthwise/separable backward passes compute per-item partials in parallel but merge them in batch/task order, and the optimizer kernels and activation derivatives are pure elementwise zips.
- A determinism audit of every rayon call site in `machine_learning`/`utils` found one violation: t-SNE's `show_progress`-only KL-divergence readout used a bare parallel `sum`, so the displayed loss digits could vary with thread scheduling (embeddings were unaffected). It now collects per-row terms and sums them in row order, matching the exact-gradient `z`-normalizer pattern. Everything else checked out: parallel argbest/tie sites resolve ties explicitly (decision-tree split selection collects candidates in feature order, kNN votes break ties by class index, t-SNE neighbor selection orders by `(distance, index)`), unordered intermediates (mean-shift's mutex-binned seeding, t-SNE's hash-map adjacency) are canonicalized by min/sort before use, and the order-preserving guarantee of rayon's `collect` covers the `filter`/`flat_map` gathers (DBSCAN region queries, mean-shift bandwidth distances).
- **`math::reduction::det_par_fold` is generic over the element type** (`&[T] where T: Sync`, previously `&[f64]`): the accumulator type was already independent, so widening reductions (f32 elements, f64 accumulator) need no conversion pass. Existing `f64` call sites compile unchanged via inference and their results are bit-for-bit unaffected.
- **Breaking: `math::matmul`'s public API takes the serial/parallel threshold as a parameter.** `par_matmul` / `par_matvec` are renamed to `gemm` / `gemv` (the "par" prefix no longer fit - whether they parallelize now depends on the argument) and gain a trailing `min_parallel_flops: usize` compared against the product's estimated FLOPs (`0` = always split, `usize::MAX` = always serial). The threshold is a pure performance knob - both paths remain bitwise identical to the serial `dot`, so unlike `det_reduce`'s narrow-band note this is numerically a pure refactor: every result in the crate is bit-for-bit unchanged. The `MatmulElem` trait leaves the public API (now `pub(crate)`, unsealed - the seal is redundant for an invisible trait; its machine-calibrated constants keep their provenance docs internally), and the element bounds relax from the sealed trait to plain `LinalgScalar + Send + Sync`, so the functions now accept any `LinalgScalar` element type, not just `f32`/`f64`. Crate-internal call sites go through new `pub(crate)` wrappers `gemm_internal` / `gemv_internal` that pass the calibrated per-type trait constants; the calibration bench mirrors the f64 gate as an explicit literal since the trait is no longer visible to it. The formerly `#[doc(hidden)]` forced-split hooks `split_matmul` / `split_matvec` are promoted to documented public API as `gemm_par` / `gemv_par` - the always-split forms `gemm`/`gemv` dispatch to, exposing the per-block size floor (`min_block`) instead of the FLOPs threshold, with full docs on the block-floor economics (GEMM blocks amortize operand re-packing, pass 64 when in doubt; matvec blocks only amortize scheduling, pass 8) and the same any-knob bitwise guarantee.
- **Breaking: `math::reduction` internalizes the serial/parallel switch.** `det_par_fold` / `det_par_fold_range` are renamed to `det_reduce` / `det_reduce_range` (matching `DET_REDUCE_BLOCK`) and take a `parallel: bool` after the data argument. The serial path folds the **same fixed blocks sequentially** instead of a flat chain, so the flag is a pure performance hint - flipping it never changes the result bits (test-enforced), and the size gate is no longer part of a caller's reproducibility surface. Every internal call site passes its calibrated gate as the flag and drops its hand-written serial branch: PCA's total variance, standardize's Welford pass, `logistic_loss`, k-means++'s seeding sum, k-means' assignment fold (its fused serial arm is gone; the label store became its own small pass, and the now-redundant per-iteration buffer zeroing went with it), linear regression's SSE/intercept sums (ndarray's `dot`/`sum` kernels remain only as the non-contiguous-storage fallback), `global_grad_norm` (always per-tensor blocked totals), and t-SNE's `show_progress` KL readout. BatchNorm's column-fold helpers gained the same flag with the same semantics (serial = same row blocks sequentially), collapsing all seven of its call-site gates. **Behavior change:** below their gates these sites now use the blocked grouping instead of a flat serial fold (or ndarray's unrolled kernels), so sub-gate sizes above one block (16Ki elements; for BatchNorm one row block) see a one-time low-bit change, and clip-norm models change low bits regardless of size (per-tensor subtotals replace the flat cross-tensor chain); inputs within a single block are unchanged bit for bit.
- **BatchNorm's per-channel statistics run as row-block deterministic folds** above a measured gate (`BN_COL_STATS_PARALLEL_MIN_ELEMS = 262_144` folded elements): the forward mean/variance and all five backward column reductions (`d_gamma`, `d_beta`, the variance/mean gradient sums). The fused product folds also eliminate the `[M, C]` temporaries the serial forms materialized (the forward squared-diff buffer and the backward product temps). **Behavior change:** above the gate the per-channel accumulation grouping changes versus previous versions (still bitwise identical at any thread count); below it (and for the 1-D scalar-parameter branch) the serial `mean_axis`/`sum_axis` paths are unchanged. End-to-end, the spatial BatchNorm forward at `[32, 64, 64, 64]` drops 72.3 ms -> 54.1 ms (~25%); the remaining cost is dominated by the fold/unfold transposes, not the statistics.
- **Spatial BatchNorm (rank >= 3) runs entirely on the native `[batch, channels, *spatial]` layout — the channel-last fold/unfold transpose copies are gone.** Previously every spatial training step paid four full-tensor permutation copies (fold the input to `[M, C]` and unfold the output in forward, the same pair for the gradients in backward), all single-threaded strided gathers — after the statistics parallelization above they dominated the layer's cost. Forward/backward/eval now compute the per-channel statistics as **plane folds** straight off the native layout: each channel's logical sequence (its contiguous `[P]` planes in batch order) folds in `DET_REDUCE_BLOCK`-element blocks whose segments accumulate in eight fixed-order lanes, partials merging in block order — grouping depends only on the input shape, and a new measured gate (`BN_PLANE_STATS_PARALLEL_MIN_ELEMS = 262_144` elements, crossover bracket 64K-256K) moves the (channel, block) tasks onto rayon without changing the bits. Every elementwise pass (center, the fused normalize + scale-shift, backward's `d_xhat` and input-gradient composition) streams per plane with the channel scalars hoisted, sharing one kernel between its serial and parallel paths (which also retires the old elementwise paths' multiplication-association mismatch across `BATCH_NORM_PARALLEL_THRESHOLD`). The row-block column folds above now serve only 2-D inputs. **Behavior change:** spatial *training* statistics use the plane grouping, so batch mean/variance — and everything downstream — change in the low bits versus previous versions at every spatial size (still bitwise identical at any thread count); spatial *inference* (`predict` and eval-mode `forward`) is pure per-element arithmetic and reproduces previous outputs **bit for bit** (test-enforced), as does the entire 2-D path. End-to-end at `[32, 64, 64, 64]`: forward 54.0 ms -> 9.6 ms (-82%; 7.5x against the 72.3 ms start of this optimization line), backward 55.8 ms -> 9.6 ms (-83%).
- **LayerNorm runs a fused parallel row path for every trailing-axis configuration, and trailing in-order `Multiple` axes no longer transpose at all.** A normalization group whose elements form the contiguous trailing block — `Default`, `Custom` on the last axis, and any `Multiple` list — is one row of a logical `[R, N]` matrix, so forward now folds each row's mean/variance with fixed-order eight-lane kernels and fuses center + normalize + scale-shift into one streaming sweep; backward fuses `grad_x_normalized` as `g * gamma` per term (the full-size temporary is gone), folds the three per-row reduction scalars, and composes the input gradient in the same sweep, while the gamma/beta gradients become row-block column folds over `[R, N]` (the kernels now live in a `normalization::folds` module shared with BatchNorm). For `Multiple`, the merge permutation is computed first: when the listed axes are already the trailing axes in order (the common Keras configuration) the permutation is the identity and the **merge/unmerge transpose copies are skipped entirely** — previously every forward, backward, and predict paid two full-tensor permutation copies even in that no-op case; a genuinely permuted axis list keeps one transpose copy each way (it is what makes the groups contiguous) around the same row core. A non-trailing `Custom` axis stays on the broadcast ndarray path, where ndarray reduces the strided mid-axis groups in place and a transpose would add the very copies the row path exists to avoid. `predict` allocates only the output (the variance folds deviations in registers) and is **bit for bit equal to the training-mode output**, test-enforced. Determinism is stronger than a grouped fold: each row is computed entirely inside one task, so the results are bitwise identical at any thread count *and* on either side of the parallel gate (`LN_ROW_PARALLEL_MIN_ELEMS`, measured); only the gamma/beta column folds carry a grouping surface (`LN_COL_STATS_PARALLEL_MIN_ELEMS`, mapped from the BatchNorm row-block measurement of the same kernel class). **Behavior change:** outputs and gradients change in the low bits versus previous versions at every size (the fixed-order segment kernels replace ndarray's `mean_axis`/`sum_axis` accumulation patterns) — still deterministic and thread-count-invariant. End-to-end: `Default` forward `[32, 512, 768]` 48.8 ms -> 13.5 ms (-72%), backward 92.7 ms -> 7.4 ms (-92%), and the trailing-`Multiple` forward at `[32, 64, 64, 64]` 44.2 ms -> 9.1 ms (-79%).
- **GroupNorm and InstanceNorm run on a fused parallel per-instance row path — the shared channels-first core's reshape copies and broadcast temporaries are gone.** In channels-first layout a normalization instance (one sample's channel group) is a contiguous `channels_per_group * spatial` block, so the shared core (`InstanceNormalization` is the `num_groups == channels` case of the same code) now folds each instance's mean/variance with the fixed-order segment kernels (deviations square in registers — no centered temporary) and writes `x_normalized` plus the per-channel affine output in one streaming sweep; backward fuses `g * gamma` per term into the two per-instance reductions (the `grad_x_normalized` temporary is gone) and composes the input gradient in one sweep, while the per-channel gamma/beta gradients become the same deterministic plane folds BatchNorm uses on the native `[batch, channels, spatial]` layout. The old core materialized roughly four avoidable full-tensor copies/temporaries in forward and six in backward (`.to_shape(...).to_owned()` where slice views suffice, broadcast products, the flat reshape of already-contiguous data), all serial. The `to_channels_first` bracket is unchanged: `channel_axis == 1` still borrows (now zero full-tensor copies end to end), other positions keep their justified permute copies. Instances are independent and each is computed entirely inside one task, so results are bitwise identical at any thread count and on either side of the row gate (`GN_ROW_PARALLEL_MIN_ELEMS`, mapped from the LayerNorm row-pass measurement of the same kernel class; the plane folds map `BN_PLANE_STATS_PARALLEL_MIN_ELEMS`). **Behavior change:** outputs and gradients change in the low bits versus previous versions at every size (fixed-order kernels replace ndarray's `mean_axis`/`sum_axis` patterns) — still deterministic and thread-count-invariant. End-to-end at `[32, 64, 64, 64]`: GroupNorm (8 groups) forward 46.8 ms -> 4.9 ms (-89%), backward 90.5 ms -> 4.1 ms (-95%), InstanceNorm forward 46.5 ms -> 4.9 ms (-89%).
- **`global_grad_norm` (clip-by-global-norm) folds large parameter tensors in parallel:** tensors at or above the measured `SQ_SUM_F32_PARALLEL_MIN_ELEMS` gate (65_536 elements) use the deterministic blocked fold (f32 elements, f64 accumulator), saving ~0.35 ms per optimizer step per million parameters when clipping is enabled. **Behavior change:** for models with such tensors the squared-sum grouping - and therefore the clip scale's low bits - changes versus previous versions (still bitwise identical at any thread count); models whose tensors are all below the gate reproduce the previous norm exactly, bit for bit.

## [Unreleased] - 2026-06-11 (UTC-7)
### Added
- **Public block-parallel matrix products: `math::matmul`.** The crate-internal `matmul` helper moves into the `math` module as a public API: `par_matmul` (`C = A.B`) and `par_matvec` (`y = A.x`) over any `f32`/`f64` ndarray operands - the signatures are generic `&ArrayBase<S, Ix2>` / `&ArrayBase<S, Ix1>` (`S: Data<Elem = T>`, `T: MatmulElem`), so owned arrays, views, and transposes all pass directly by reference. Both functions keep matrixmultiply's serial kernels for the inner work, split the output across rayon above the calibrated per-type FLOPs gates, and guarantee a result **bitwise identical to the serial `dot` at any thread count** plus a standard-layout output. The `MatmulElem` trait (carrying the calibrated gate constants, documented as machine-specific defaults rather than semantic contracts) is public but sealed to `f32`/`f64`; the calibration-only forced paths (`split_matmul`/`split_matvec`) and the internal tiling/cache-residency policy helpers stay `#[doc(hidden)]`, and `bench_internals` no longer re-exports any of them. The `math` feature gains `dep:rayon` - a no-op for the dependency tree, since the hardwired `ndarray/rayon` feature already pulled rayon into every ndarray build - and `neural_network` now depends on `math`. Internal call sites address the helpers as `crate::math::matmul::...` (no crate-root re-export); the module's unit tests drop their `ndarray-rand`/`crate::random` dependence so they run under a `math`-only build.
- **Benchmark infrastructure under `benches/`** (criterion added as a dev-dependency). `cargo bench --bench parallel_gates` is a custom-harness calibration suite that times the forced-serial and forced-parallel implementation of every parallel-gated kernel class (block-parallel GEMM, conv engine, pooling engine, and the relu/sigmoid/dropout/optimizer elementwise classes) across size ladders, computes each crossover with a noise margin, and rewrites `benches/RESULTS.md` with the full tables plus CPU/thread-count/date provenance — the source of truth for setting the `*_MIN_FLOPS`/`*_MIN_OPS`/`*_PARALLEL_THRESHOLD` constants. `cargo bench --bench nn_end_to_end` is a criterion suite over the public API (Dense/Conv2D/LSTM forwards and an MLP training epoch) for performance-regression tracking with saved baselines. The forced-path entry points the calibration needs are exposed through a `#[doc(hidden)] pub mod bench_internals` (explicitly not part of the public API, no stability guarantees); the convolution and pooling engines gained internal `force_parallel: Option<bool>` plumbing for it.
- **Clip-by-global-norm** gradient clipping for the neural-network optimizers, opt-in through a new trailing `clip_norm: Option<f32>` argument on `SGD::new`, `Adam::new`, `RMSprop::new`, and `AdaGrad::new` (a `Some` value must be positive and finite; `None` disables clipping). When enabled, the `Sequential` training loop runs every layer's backward, computes the global L2 norm across **all** of the model's gradients (accumulated in `f64`), and — if it exceeds `max_norm` — scales every gradient by the single factor `max_norm / global_norm` before the optimizer step, preserving gradient direction (unlike per-element clamping). A norm at or below the threshold applies no scaling, and a non-finite norm (which signals upstream divergence) is deliberately left unscaled so it still surfaces. Exposed on the `Optimizer` trait via a new `clip_norm()` method; the scale itself flows through a new `grad_scale` parameter on `Optimizer::update` and a `kernels::scaled_grad` helper that borrows the gradient unchanged when the scale is `1.0`, so there is zero overhead when clipping is off.
- **`padding='same'` for the windowed pooling layers** (`MaxPooling`/`AveragePooling` 1D/2D/3D). The dimension-generic pooling engine gains a `PaddingType` argument and a `pool_geometry` helper: `Same` rounds the output up to `ceil(in/stride)` and offsets each window by the leading pad, while padded cells stay virtual (out-of-bounds positions are skipped). Average pooling therefore divides by the count of real in-bounds elements — Keras's `count_include_pad=False` behavior — and max pooling's recorded arg-max already carries the correct input index, so its backward is unchanged. The output-shape calculators are padding-aware.
- **SGD momentum / Nesterov and decoupled weight decay across all optimizers.** `SGD::new` gains `momentum` and `nesterov` (a `sgd_momentum_step` kernel with per-parameter velocity buffers; `momentum = 0` stays plain stateless SGD), and every optimizer gains a `weight_decay` coefficient applied as decoupled AdamW/SGDW-style decay (`kernels::apply_weight_decay` shrinks the parameter by `1 - lr*wd` *before* the gradient step, so the penalty acts on the weights rather than being rescaled by an adaptive denominator).
- **External learning-rate scheduling.** New `Optimizer::set_learning_rate` trait hook (overridden by all four optimizers) and a `Sequential::set_learning_rate` forwarder, so a step-decay / warmup schedule can retune the step size between epochs or batches without rebuilding the optimizer or losing its accumulated state.
- **`from_logits` fused softmax-cross-entropy.** `CategoricalCrossEntropy::new` and `SparseCategoricalCrossEntropy::new` take a `from_logits: bool`. When `true`, the loss applies a numerically stable log-softmax internally (via a shared `stable_log_softmax_softmax` helper) and `compute_grad` returns the fused `(softmax(z) − y) / batch` gradient directly w.r.t. the logits — skipping a separate softmax-layer backward, and more stable than feeding a clipped softmax through `ln`.

### Changed
- Make the `math` distance-row functions (`squared_euclidean_distance_row`, `manhattan_distance_row`, `minkowski_distance_row`) single-pass and allocation-free: they accumulate over an `ndarray::Zip` instead of materializing `x1 - x2` and then a second `mapv` array, removing two allocations per call on the KNN/DBSCAN/silhouette `O(n²)` hot paths (the `Zip` still panics on a length mismatch, where a plain iterator `zip` would silently truncate).
- `DistanceCalculationMetric::within` now decides the threshold test in the metric's order-preserving "comparable" space for every variant, so `Minkowski(p)` compares `Σ|a−b|ᵖ` against `thresholdᵖ` instead of taking a per-pair `p`-th root — matching the root-free Euclidean path it already used.
- Evaluate the `Poly` kernel with `powi(degree)` instead of `powf(degree as f64)` in both `KernelType::compute` and the batched `compute_matrix`; the exponent is integral, so `powi` is faster and more accurate.
- `silhouette_score` accumulates its pairwise distances over independent per-sample rows, switching between a serial fill and a parallel `Zip::par_for_each` fill at a sample-count threshold (`SILHOUETTE_PARALLEL_THRESHOLD = 128`) so small inputs avoid rayon's task-spawn overhead, instead of the previous serial `O(n²)` double loop. **Breaking:** it now takes a trailing `metric: DistanceCalculationMetric` argument (pass `DistanceCalculationMetric::Euclidean` for the conventional silhouette) instead of hardcoding Euclidean, routing every pairwise distance through the same dispatch point the estimators use; its feature-matrix storage bound also tightens to `S1: Data + Sync` (satisfied by owned arrays and views). To support this, the `types` module (and `DistanceCalculationMetric`) is now compiled under the `metrics` feature as well — its serde derives are gated to the `machine_learning`/`utils` configurations, so `metrics` still pulls in no serde dependency.
- **Breaking:** the regression metrics (`mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`, `explained_variance_score`, `median_absolute_error`, `mean_absolute_percentage_error`) take two independent storage parameters `S1`/`S2` instead of a single shared `S`, so `y_true` and `y_pred` may now mix owned arrays and views (matching `classification`/`math`).
- Document the deliberate NaN divergence between `r2_score` (plain sums, so a `NaN`/`inf` propagates to a `NaN` result) and `explained_variance_score` (routes through `math::variance`, which silently skips non-finite samples).
- The prelude root now flattens every category, so `use rustyml::prelude::*;` brings the actual items (traits, models, metrics, ...) into scope instead of only the category module names. The per-category modules (`rustyml::prelude::machine_learning::*`, etc.) remain available for narrower imports.
- **Breaking:** every neural-network optimizer constructor (`SGD::new`, `Adam::new`, `RMSprop::new`, `AdaGrad::new`) takes a new trailing `clip_norm: Option<f32>` argument (pass `None` to keep the previous behavior), and the `Optimizer::update` trait method gains a trailing `grad_scale: f32` parameter. Putting this optional setting in the constructor matches the crate's existing `random_state: Option<u64>` convention rather than a separate builder method.
- **Behavior change:** the embedded activation derivatives (`Activation::backward` for Sigmoid/Tanh/ReLU/Softmax, used by Dense and the recurrent layers) are now exact — the `clip_grad`/`GRAD_CLIP_VALUE` sanitization (clamp to ±1e6, NaN/Inf → 0) is removed. Each derivative is bounded for finite inputs, so a non-finite gradient can only come from already-diverged upstream values; it is now propagated rather than silently zeroed, surfacing at the next forward pass (which rejects non-finite input) or as a NaN loss. The standalone `Sigmoid`/`Tanh`/`ReLU`/`Softmax`/`Linear` layers likewise drop their backward NaN/Inf check on `grad_output` (forward-side input validation is unchanged). This pure-math contract is now documented on `Layer::backward`.
- **Behavior change:** the recurrent layers (SimpleRNN, and GRU/LSTM via `store_gate_gradients`) no longer clamp their stored gradients element-wise to `±5`; the hardcoded `GRADIENT_CLIP_VALUE` is gone. Gradients are stored exactly as computed — use the new opt-in clip-by-global-norm on the optimizer to tame exploding gradients instead, which scales by the global norm and preserves direction rather than distorting each component independently.
- **Breaking:** `BatchNormalization` is now genuine *spatial* batch norm for rank > 2 (convolutional) inputs. Parameters are per-channel (`gamma`/`beta`/running stats have length `input_shape[1]`, was the full `input_shape[1..]`), and statistics reduce over the batch **and** all spatial positions (matching Keras/PyTorch), instead of keeping a separate mean/variance/scale/shift per spatial element. Implemented by folding `[N, C, *spatial]` to `[M, C]` (`M = batch·spatial`) so the existing, gradient-checked per-feature math runs unchanged; the 2-D (Dense) path is byte-identical to before. CNN models that relied on the old per-element parameterization will see different (correct) results and a much smaller parameter count.
- **Breaking:** every optimizer constructor gains a trailing `weight_decay: f32` argument, and `SGD::new` additionally gains `momentum: f32` and `nesterov: bool` (so `SGD::new(lr, clip_norm, momentum, nesterov, weight_decay)`; `Adam`/`RMSprop`/`AdaGrad` take `..., clip_norm, weight_decay`). Pass `0.0`/`false` to keep the previous behavior. The `Optimizer` trait gains a `set_learning_rate` method (defaulted to a no-op).
- **Breaking:** `CategoricalCrossEntropy::new` and `SparseCategoricalCrossEntropy::new` take a `from_logits: bool` (pass `false` for the previous probability-input behavior).
- **Breaking:** the windowed pooling constructors (`MaxPooling`/`AveragePooling` 1D/2D/3D `::new`) take a trailing `padding: PaddingType` argument (pass `PaddingType::Valid` for the previous behavior); the shared pooling engine functions gain the same parameter.
- **Breaking:** the convolution engine (`conv_geometry`/`conv_forward`/`conv_backward`) now returns `Result`: under `Valid` padding an input spatial dimension smaller than the kernel returns `Error::InvalidInput` instead of underflowing `usize` and panicking. This covers Conv1D/2D/3D and the SeparableConv2D pointwise stage uniformly, and `validate_input_shape_3d` gains a kernel parameter so Conv3D rejects an oversized kernel at construction like Conv1D/2D already did.
- **Behavior change:** removed the eager NaN/Inf scan from the standalone `ReLU`/`Sigmoid`/`Tanh`/`Softmax`/`Linear` layers' `forward`/`predict`. They now match the embedded-activation path (no per-call `O(n)` sanitization); a non-finite input propagates (sigmoid/tanh saturate `±inf` to finite values and only a `NaN` propagates; ReLU maps `NaN`/`-inf` to 0) instead of being rejected with `Error::NonFinite`.
- Removed the `±500` input clamp from sigmoid/tanh (`Activation` and the recurrent `apply_sigmoid`). `1/(1 + e^-x)` is correct and finite for any finite `x` (an overflowing `e^-x` yields the exact limit `0`) and tanh self-saturates, so the clamp — at `±500`, far past the `~88.7` f32 `exp` overflow point — never prevented overflow and was misleadingly documented as doing so.
- `Sequential::fit_with_batches` builds each mini-batch with `ndarray::select(Axis(0), &indices)` (a bulk per-sample gather) instead of an element-by-element `extend`, for any input rank.
- **Breaking:** LSTM and GRU store their gates **fused**: the per-gate weights are packed side by side into single matrices (`kernel [input_dim, n_gates*units]`, `recurrent_kernel [units, n_gates*units]`, `bias [1, n_gates*units]`), gate column blocks `[i | f | g | o]` for LSTM (the Keras layout) and `[r | z | h]` for GRU. The batched input projection, the BPTT kernel/recurrent/bias reductions, and the input-gradient GEMM each collapse from one GEMM per gate into a single wide GEMM, and the per-timestep recurrent projection becomes one `[batch, units] × [units, n_gates·units]` GEMM (GRU fuses reset+update per step; its candidate projection stays separate because it consumes the freshly computed `r_t .* h_{t-1}`). `set_weights` now takes the three fused matrices — this is the form serialization uses, so LSTM/GRU models saved by older versions no longer load — and a new `set_gate_weights` convenience keeps the old per-gate signature, packing internally. `LayerWeight::LSTM`/`LayerWeight::GRU` and the serializable weights expose the fused matrices (`LSTMGateWeight`/`GRUGateWeight`/`SerializableGateWeight` are removed), the shared `Gate` struct is replaced by `FusedGates`, and the optimizer sees 3 parameter tensors per layer instead of 12/9. Initialization semantics are unchanged (Xavier with the per-gate fan, an independent orthogonal recurrent block per gate, forget-gate bias 1.0), though the RNG stream consumption differs, so same-seed initializations differ from previous versions. The per-step `rayon::join` over gate GEMMs and the `LSTM_PARALLEL_THRESHOLD`/`GRU_PARALLEL_THRESHOLD` constants are gone — one fused GEMM replaces the joined small ones.
- The Dense and recurrent (SimpleRNN/LSTM/GRU) GEMMs run **block-parallel** through a new crate-internal `par_matmul` helper. Without a BLAS backend, `ndarray`'s `dot` runs on the single-threaded `matrixmultiply` kernel; large products — the Dense forward/predict transform and both backward GEMMs, the fused batched input projections, the per-timestep recurrent projections, and every BPTT weight/input-gradient reduction — now split their longer output axis into row/column blocks across rayon, each block computed by the serial kernel (keeping its cache blocking + SIMD). Splitting the `m`/`n` axes never reorders any output element's `k`-accumulation, so the result is **bitwise identical** to the serial product at any thread count — the parallelism costs no reproducibility (a `k`-split reduction would). Products below an estimated-FLOPs gate (`2·m·k·n < 1e7`, an initial estimate pending benchmark calibration) fall through to the plain serial `dot`, so small models see no scheduling overhead. The helper also guarantees a standard-layout result, centralizing the column-major `dot`-output normalization. Sanity check on a `[2048,1024] × [1024,1024]` product (9950X, 16C/32T): 30.6 ms serial → 5.5 ms block-parallel (~5.6×), outputs bitwise equal.
- The convolution engine's forward pass parallelizes over `(batch item, output-position block)` tasks instead of batch items alone: each task gathers its own im2col column block and runs its GEMM + bias add, so a single large image saturates every core even at `batch == 1` — previously the common single-sample inference case ran fully serial. Block boundaries never change an output element's accumulation order, so results stay bitwise independent of the thread count. The backward pass keeps its batch-order partial reduction (also thread-count independent) and routes its two per-item GEMMs through `par_matmul`, so a small batch still spreads the GEMM work. Sanity check on a `[1, 64, 128, 128]` input with 64 `3x3` filters (9950X, 16C/32T): 97.4 ms → 9.3 ms (~10.5×) for five forward passes.
- The parallel/serial gates across the convolution and pooling layers estimate actual per-pass work instead of counting output elements. The conv engine, `DepthwiseConv2D`, and `SeparableConv2D` gates count FLOPs including the kernel-tap/channel multipliers the old element counts ignored (an element count rates a `7x7`-kernel, 512-channel convolution the same as a `3x3`, 3-channel one), and the pooling engine gates on `planes × per-plane work` rather than the plane count alone, so `batch == 1` on a large image can parallelize while many tiny planes stay serial. The replacement constants (`CONV_PARALLEL_MIN_FLOPS`, `DEPTHWISE_CONV_2D_PARALLEL_MIN_FLOPS`, `SEPARABLE_CONV_2D_PARALLEL_MIN_FLOPS`, `POOL_PARALLEL_MIN_OPS`) are initial estimates pending a benchmark-calibration pass.
- **Every parallel/serial gate constant recalibrated from measurement** (forced-path crossovers on a 9950X, 16C/32T; full tables in `benches/RESULTS.md`, provenance noted on each constant). The headline finding: the elementwise thresholds were severely *too low* — at the old gates, small tensors paid a ~20-25 µs rayon fork/join for maps that finish in microseconds serially (sigmoid at its old gate of 1000 elements ran ~25× slower than serial) — while the GEMM gates were *too high*, leaving 2-10M-FLOP products (e.g. an LSTM's per-timestep recurrent GEMM) needlessly serial. New values: `PAR_GEMM_MIN_FLOPS`/`CONV_PARALLEL_MIN_FLOPS` 1e7 → 4M (measured crossover brackets 2.1-4.2M and 2.1-8.3M); `POOL_PARALLEL_MIN_OPS` 1M → 12K taps; `SIGMOID`/`TANH` thresholds 1000/2048 → 131,072 elements; `SOFTMAX_PARALLEL_THRESHOLD` now gates on `batch * classes` total elements (was a bare row count of 8) at the same exp-class crossover; `RELU`/`DROPOUT`/`SPATIAL_DROPOUT` → 4M elements (these memory-bound maps never beat serial up to 1M elements - dropout's rng sampling stays a single serial stream for seed reproducibility, so only the cheap compare map was ever gated); optimizer `kernels::PARALLEL_THRESHOLD` 1024 → 1M elements (fused multi-stream updates cross at 256K-1M); `BATCH_NORM_PARALLEL_THRESHOLD` 1024 → 262,144 by analogy with that class; `PAR_GEMM_MIN_BLOCK = 64` was validated as measured-optimal. End-to-end effect (criterion, same machine): Dense forward −34%, Conv2D batch-1 forward −24%, LSTM forward −57%, MLP train epoch −46%.
- **Finer task granularity for the remaining batch-bound passes**, so small batches (especially `batch == 1`) keep every thread busy. `DepthwiseConv2D` backward parallelizes over `(batch item, channel)` instead of batch items (the channels of a depthwise convolution are independent; the per-channel weight partials are still reduced in ascending-batch order, so the f32 result stays thread-count independent). `SeparableConv2D`'s depthwise stage parallelizes its forward over `(batch item, output channel)` and its backward weight gradients over `(depth multiplier, channel)` — the previous depth-multiplier-only split degenerated to a *single task* at the common `depth_multiplier == 1` — and its backward input gradients over `(batch item, channel)` (a pure gather into disjoint planes); both backward passes also gained the FLOPs-based serial gate they previously lacked. Windowed pooling's forward splits each plane into output-position chunks like the conv engine (bitwise thread-count independent; measured 2.7× → 7.8× on a `[1, 3, 1024, 1024]` max pool — see `benches/RESULTS.md`, crossover unchanged). Deliberately *not* re-granulated: windowed pooling's backward (overlapping windows scatter into shared input cells, so finer chunks would need per-chunk partial buffers) and global pooling (splitting a single reduction reorders f32 summation and would break reproducibility); both keep their per-plane tasks. The unused `merge_results` helper is removed.
- The elementwise gate constants are consolidated into a single crate-internal `neural_network::parallel_gates` module with one constant per calibrated **cost class** — `CHEAP_MAP_PARALLEL_THRESHOLD` (ReLU and the dropout layers' mask thresholding), `EXP_MAP_PARALLEL_THRESHOLD` (sigmoid/tanh/softmax), `FUSED_SLICE_PARALLEL_THRESHOLD` (optimizer kernels), and `NAIVE_CONV_PARALLEL_MIN_FLOPS` (DepthwiseConv2D and SeparableConv2D's depthwise stage) — replacing nine per-layer duplicates of the same class values, so each calibration result lives in exactly one place and a recalibration touches one file. The engine-specific gates keep their engine-local constants (`PAR_GEMM_*`, `CONV_*`, `POOL_*`, `BATCH_NORM_*`), since their work metrics are engine-specific rather than class-shared.
- The block-parallel matmul helper moves from `neural_network::matmul` to a crate-level `matmul` module (compiled under any of `machine_learning`/`neural_network`/`utils`) and is **generic over the element type** through a new `MatmulElem` trait (implemented for `f32` and `f64`), so the classical-ML and utils modules — whose linear algebra is `f64` — can route their GEMMs through the same gated block-parallel path the neural-network layers use. The serial/parallel FLOPs gates live on the trait as per-type associated constants: the `f32` values keep their calibration, the `f64` values adopt them as conservative (serial-leaning) placeholders pending an `f64` calibration sweep. Also adds `par_matvec` (`y = A·x`), the matrix-vector counterpart with its own `PAR_GEMV_MIN_FLOPS` gate class: it splits rows across rayon but keeps each block on ndarray's matrix-*vector* kernel — deliberately not reusing the matrix-matrix kernel on a `[k, 1]` operand, whose different accumulation order would change the result — so it is bitwise identical to the serial `a.dot(&x)` at any thread count, and call sites migrated onto it see no numerical change at all.
- **The ML/utils matrix products run block-parallel** through the shared `matmul` helpers. `KernelType::compute_matrix` — the `O(n·m·d)` dominant cost of SVC fit/predict and KernelPCA fit/transform — computes its cross-Gram matrix as one block-parallel GEMM followed by a parallel elementwise kernel transform, instead of the previous one-GEMV-per-output-row sweep (the RBF/Cosine per-row `||x_i||²` norms are now precomputed once; the method and its tests are feature-gated to `machine_learning`/`utils`, since `metrics` compiles `types` without the `rayon` dependency). PCA routes its fit covariance `Xᵀ·X`, transform projection, and inverse-transform reconstruction through `par_matmul`, retiring the hand-rolled `project_parallel`/`reconstruct_parallel` row loops (whose per-row scalar dots strided down the component columns); KernelPCA's transform projection likewise replaces its dual serial/`project_parallel` paths. LDA routes its transform projection, per-class scatter GEMMs, scoring-coefficient products, and LSQR bidiagonalization matvecs through the helpers, and `predict` scores all samples against all classes in one `X·coefᵀ` GEMM plus a per-row arg-max instead of per-row, per-class scalar dots. The `utils::linalg` power-iteration and Lanczos solvers' per-iteration matvecs (`A·v` on the `[n, n]` covariance/Gram matrix) and the Hotelling-deflation outer product go through `par_matvec`/`par_matmul`. The full-batch GEMVs of LogisticRegression (fit predictions + gradient, predict decision values), LinearRegression (fit predictions + gradient, predict), and LinearSVC (cost margins, decision function) go through `par_matvec`, which is bitwise identical to the serial `dot` — those call sites see zero numerical change. The kernel-matrix and projection GEMMs swap the GEMV/scalar-dot kernels for the matrix-matrix kernel, so individual entries may differ from previous versions at rounding level (results remain deterministic and thread-count independent); all golden tests pass unchanged.
- **The ML/utils per-sample and per-pair distance loops are rewritten in batched GEMM form.** KMeans computes every sample-centroid projection per Lloyd iteration as one block-parallel `data · centroidsᵀ` GEMM followed by a cheap per-row arg-min scan (ranking by `‖c‖² − 2·x·c`; the assigned cluster's inertia distance is still measured exactly), replacing the per-sample GEMV swarm; `predict` uses the same identity (it previously compared exact distances — labels can differ from older versions only on exact ties) and the per-sample `closest_centroid` helper is gone. KMeans++ initialization also folds each newly selected center into a **running** per-sample minimum instead of re-scanning all `k` selected centers every round — `O(n·k·d)` total instead of `O(n·k²·d)`, bitwise identical since the running `min` is exact. KNN's brute-force Euclidean path (the kd-tree fallback for > 16 features) computes query projections in chunked block-parallel GEMMs (`X_chunk · X_trainᵀ`) shared by both `predict` and `predict_parallel`, so each query's work drops to a scan + selection over a precomputed projection row instead of re-streaming the whole training set per query. t-SNE's exact path builds its pairwise squared distances as one GEMM plus the `‖x_i‖² + ‖x_j‖² − 2·x_i·x_j` identity (negatives from cancellation clamped at zero, diagonal forced to exactly 0, result exactly symmetric), the Barnes-Hut neighbor search computes its distances from chunked GEMM projections the same way, and the exact gradient factors into GEMM form — `W = (P − Q)∘num`, `grad = 4·(diag(W·1)·Y − W·Y)` — turning the dominant `O(n²·d)` force summation into one block-parallel GEMM. MeanShift's `estimate_bandwidth` computes its upper-triangle pairwise distances from chunked GEMM projections with the same identity. The chunking is governed by a shared `matmul::gemm_chunk_rows` helper (a ~4M-element / ~32 MB buffer budget per chunk, initial estimate pending calibration). **Numerical note:** the norms-identity distances and the GEMM-form gradient round differently than the old per-pair scalar loops, so KMeans/KNN outcomes can shift on exact distance ties and t-SNE embeddings differ from previous versions at rounding level amplified by t-SNE's chaotic optimization (results remain deterministic and thread-count independent); all golden tests pass unchanged.

- **Every ML/utils parallel/serial gate is now calibrated, work-metric-based, and class-shared** (forced-path crossovers on a 9950X, 16C/32T; tables in `benches/RESULTS.md`). The calibration bench gains `f64` sections (GEMM, matvec, block floors, cheap-map/exp-map/arg-min-scan/sum classes, tiled-chunk budgets, and a pairwise-distance strategy shoot-out), and the measured constants land as follows. `MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS` drops 4M -> 2M (bracket 524K-1.77M; f64 crosses earlier than f32, as the halved SIMD width predicts); `PAR_GEMV_MIN_FLOPS` drops 4M -> 524,288 for both element types (bracket 131K-524K - the placeholder left an order of magnitude of parallel matvec wins on the table); a new `PAR_GEMV_MIN_BLOCK = 8` row floor replaces the GEMM block floor for matvec splits (no operand packing to amortize; the tall shape plateaus over 8-64 rows, the short-wide one over 1-16). The former `neural_network::parallel_gates` module moves to the crate root with new calibrated `f64` classes - `CHEAP_MAP_F64_PARALLEL_THRESHOLD = 4M` (bracket 1M-4.2M), `EXP_MAP_F64_PARALLEL_THRESHOLD = 65,536` (bracket 16K-32K with a thin 1.09x; the gate sits at the solid-win rung), `SCAN_F64_PARALLEL_MIN_ELEMS = 262,144` (bracket 65K-262K), `SUM_F64_PARALLEL_MIN_ELEMS = 262,144` (bracket 131K-262K) - and **fifteen per-model magic-number thresholds are deleted** (`KMEANS`/`DBSCAN`/`MEANSHIFT`/`SVC`/`LDA`/`LOGISTIC_REGRESSION`/`LINEAR_REGRESSION`/`LINEAR_SVC`/`PCA`/`KERNEL_PCA`/`TSNE`/`NORMALIZE x2`/`STANDARDIZE x2` `_PARALLEL_THRESHOLD`/`_LANES`, plus KNN's function-local voting thresholds), their sites re-gated on **total work** (items x per-item cost, e.g. `n*k` for the KMeans arg-min scan, `n*centers*d` for MeanShift labeling, `n*n` for the t-SNE fills and SVC error-cache build) against the matching class constant - a bare item count rated a 2-feature scan the same as a 2000-feature one. `map_collect` accordingly takes the caller's parallel decision instead of an item-count threshold. The kernel transforms in `compute_matrix`, previously unconditionally parallel, now gate per class (RBF/Sigmoid at the exp-map gate, Poly/Cosine at the cheap-map gate). `metrics`' silhouette gate is restated locally as `SILHOUETTE_PARALLEL_MIN_ELEMS = 262_144` scanned elements (the module deliberately imports no crate internals); the decision-tree and isolation-forest gates are documented as an uncalibrated coarse-task class and keep their values.
- **Eight scheduling-dependent f64 reductions are fixed for reproducibility.** A bare rayon `sum`/`fold().reduce()` groups partial float sums by work-stealing, so the rounded result varied run to run and with the thread count. Serialized outright (the work sits below the measured ~262K-element sum crossover at any realistic size, so this also costs nothing): SVC's `compute_error` decision sum and error-cache update (whose nondeterminism fed the whole SMO trajectory), SVC's `select_second_alpha` parallel max-reduce (which also resolved |E1-E2| ties by scheduling), LinearRegression's L1 penalty sum, LinearSVC's per-batch gradient fold/reduce (every batch's f64 gradient depended on thread grouping), KernelPCA's overall-kernel-mean sum, KNN's distance-weighted-vote fold, and SVC's support-vector extraction filter. PCA's total-variance sum and global standardization's Welford moments instead go through a new crate-internal `reduction::det_par_fold` - fixed-size blocks folded serially, merged in block order - which parallelizes above the calibrated sum gate while staying **bitwise identical at any thread count**.
- **The tiled-GEMM strategy from the batched-rewrite round is corrected by measurement: it only wins once the shared matrix overflows the L3 cache.** The strategy shoot-out showed a per-row GEMV swarm beating both the per-pair scalar loops (2.8x) and a one-shot/tiled GEMM (~2.7x) while the matrix is cache-resident - each task re-reads X from L3 for free, and the GEMM's packing and output materialization buy nothing - while on a cache-overflowing 256 MB training set the tiled GEMM wins ~2x instead (each chunk streams X once, the swarm re-streams it per row). KNN's brute-force Euclidean path, t-SNE's neighbor search, and MeanShift's bandwidth estimation now pick per shape via a new `matmul::cache_resident` helper (boundary at 64 MB, the full measured L3; only 25 MB and 256 MB bracket it so the band between is uncalibrated), and `GEMM_CHUNK_ELEMS` is recalibrated 4M -> 32Mi elements for the overflow regime, where bigger chunks are strictly better (1.23x at 8M -> 1.93x at 64M) and the budget caps the transient buffer at 256 MB.
- **The last uncalibrated ML thresholds are now measured** (new calibration sections in `benches/parallel_gates.rs`; tables in `benches/RESULTS.md`). New measured classes in `parallel_gates`: `TREE_TRAVERSAL_MIN_VISITS = 262_144` node visits (synthetic depth-16 root-to-leaf walks; crossover bracket 65K-262K) and `SORT_SCAN_MIN_ELEMS = 8_192` sorted elements (per-feature copy+sort+scan; bracket 2K-8K). DecisionTree's `DECISION_TREE_PARALLEL_THRESHOLD = 1000` is deleted: at 1000 samples the parallel prediction path was measured at a 0.33x *loss* - prediction now gates on traversal work (`samples x assumed depth 16`, i.e. ~16K samples) and the split search on sort work (`node samples x features`). IsolationForest's `DEFAULT_PARALLEL_THRESHOLD_SAMPLES = 100` is replaced by the same traversal-work gate (`samples x trees x c(256)~=10`); its tree-build gate keeps 10, now backed by the measured 16-32-task crossover at ~0.7 us/task (a real iTree build costs far more per task, so 10 clears the fork cost with margin). The kd-tree dimensionality ceilings `KNN/DBSCAN_KD_TREE_MAX_DIMS` drop 16 -> 8: measured on uniform data (20k points, k=8), the kd-tree wins up to d=8 (2.6x) but *loses* to the brute-force scan by 2.2-2.6x over d=12-16, where the old ceiling still chose it (single-shape calibration; the boundary shifts with distribution and size, as documented). `reduction::DET_REDUCE_BLOCK` moves 8192 -> 16384, the measured optimum of a 4K-64K plateau (note: the block size is part of the deterministic-grouping reproducibility surface). The kd-tree-vs-brute comparison drives the crate-private `KdTree` through `bench_internals` (now compiled under `machine_learning` too); the `parallel_gates` bench requires both features. The remaining unmeasured constants in ML/utils are algorithm semantics (t-SNE's paper hyperparameters, convergence/tolerance epsilons), not performance gates.
- New `benches/ml_end_to_end.rs` criterion suite over the classical-ML/utils public API (KMeans/SVC/LogisticRegression fits, KNN brute-force predict, PCA/KernelPCA fit+transform, exact t-SNE) for performance-regression tracking with saved baselines, complementing `nn_end_to_end`.

### Removed
- **Breaking:** drop the `rustyml::prelude::math` submodule. The `math` items are low-level numeric primitives (distances, losses, variance/SST, gini/entropy) that don't belong in a prelude by convention - traits and high-level entry points do. Import them directly instead, e.g. `use rustyml::math::variance;`. The other category preludes (`machine_learning`, `metrics`, `utils`, `neural_network`) are unchanged.

### Fixed
- The recurrent layers' batched input projection and input-gradient reshapes no longer panic with `IncompatibleLayout` when a GEMM operand has a row stride of 1: `ndarray`'s `dot` returns a **column-major** result whenever both operands have row stride 1 — which arrays with a length-1 axis can exhibit while still passing `is_standard_layout` (e.g. a `[1, w]` kernel assembled by `concatenate`, or `units == 1` gradient matrices) — and the previous bare `into_shape_with_order` rejected that layout. A shared layout-tolerant reshape helper (`gate::reshape_2d_to_3d`) normalizes the layout first; for `SimpleRNN` with `units == 1` and `input_dim > 1` this was a pre-existing latent panic in `backward`.
- **Breaking:** `math::{sum_of_squared_errors, logistic_loss, hinge_loss}` now panic on a length mismatch (matching `metrics::validate_pair`) instead of silently truncating to the shorter input through `zip`, and panic on empty input — so `logistic_loss` no longer returns `0.0/0.0 = NaN` and `hinge_loss` no longer disagrees with it by returning `0.0`.
- **Breaking:** `ConfusionMatrix::recall` returns `0.0` (was `1.0`) when there are no actual positives, matching scikit-learn's `zero_division=0` default and `MulticlassConfusionMatrix::per_class_recall`; the old `1.0` also spuriously inflated `balanced_accuracy`.
- **Breaking:** `normalized_mutual_info` normalizes by the **arithmetic** mean of the two entropies (was the geometric mean `√(H_true·H_pred)`), matching scikit-learn (≥ 0.22), `adjusted_mutual_info`, and `v_measure_score` (which is exactly the arithmetic-mean-normalized NMI), so the family of metrics now agrees on the same data.
- **Breaking:** the ranking metrics (`roc_auc`, `average_precision`, `roc_curve`, `precision_recall_curve`) now panic on a `NaN` score instead of letting `f64::total_cmp` silently rank it as the most-confident prediction; `top_k_accuracy` likewise panics on a `NaN` in `y_prob` instead of miscounting a `NaN` true-class probability as a hit.
- `log_loss` renormalizes each probability row to sum to 1 before scoring (matching scikit-learn), so rows that do not already sum to 1 are comparable to scikit-learn's output.
- `average_path_length_factor` includes the next harmonic-expansion term `1/(2(n−1))` in its `n > 50` branch, cutting the approximation error from ~2×10⁻² to ~10⁻⁵ at the cost of one division.
- **Breaking:** Minkowski `p` is now validated against its documented `p ≥ 1` contract: `minkowski_distance_row` panics for `p < 1` (or `NaN`), `KNN::new` rejects such `p` (it performed no validation before), and `DBSCAN::new` tightens its check from `p > 0` to `p ≥ 1`. Orders below 1 are not valid metrics (the triangle inequality fails) and `p ≤ 0` additionally yields a meaningless `sumᶦⁿᶠ`.
- **`SimpleRNN` gradient accumulated across batches.** `SimpleRNN::backward` seeded its gradient buffers from the previous call (`grad_kernel.take().unwrap_or_else(zeros)` then `+=`), so with no `zero_grad` step in the training loop the gradient applied at batch *n* was the running sum over batches `1..=n` (then saturated by the old ±5 clamp), steadily drifting from the correct direction. It now starts each backward from zero, matching the replace semantics of `Dense` and the LSTM/GRU gates.
- **`GaussianDropout` backward ignored the sampled noise.** The forward pass applies multiplicative noise `y = x * noise`, but backward passed the gradient straight through; the correct gradient is `grad_output * noise` using the *same* noise drawn in forward. The layer now caches the forward noise and multiplies it back in backward (inference and `rate == 0` remain a pass-through). The additive `GaussianNoise` pass-through was already correct.
- **`backward` panicked on a wrong-rank gradient.** `Dense`, `SimpleRNN`, `LSTM`, and `GRU` converted the upstream gradient with `into_dimensionality().unwrap()`, panicking through the public `Layer::backward` on a mismatched rank. They now return a recoverable `Error` (`Dense` also gained an up-front shape guard, since its activation backward would otherwise panic inside an element-wise `Zip`).
- **`DepthwiseConv2D` panicked on a wrong channel count.** Its forward/predict used `assert_eq!(channels, filters, ...)`; this is now an `Error::DimensionMismatch`, matching the recoverable-error convention of the other layers.
- **Optimizer state could silently corrupt across a parameter-shape change.** Adam/RMSprop/AdaGrad (and now momentum-SGD) address their per-parameter state by cursor position; if a parameter's length changed between steps, `*_step`'s `zip` would silently truncate to the shorter length. They now detect a length mismatch and rebuild that state slot instead.
- **`CategoricalCrossEntropy` accepted 1-D inputs.** With a 1-D tensor, `shape()[0]` is the total element count rather than the batch size, silently rescaling the loss and its gradient; it now requires `ndim ≥ 2` (`[batch, classes]`).

## [v0.12.0] - 2026-06-10 (UTC-7)
### Added
- Add `KernelType::compute_matrix(x, y)`, which builds the full kernel matrix `K[i, j] = K(x_i, y_j)` in one batched pass instead of looping the scalar `compute` over every pair. Each output row is `K[i, :] = x_i · Yᵀ` evaluated as a single GEMV with the kernel's elementwise transform (`Poly`/`Sigmoid`/`RBF`/`Cosine`) fused in, parallelized over rows; `RBF`/`Cosine` reuse each sample's precomputed squared norm and the `‖x − y‖² = ‖x‖² + ‖y‖² − 2·x·y` identity (the `RBF` distance is clamped at zero to absorb the tiny negatives cancellation can leave). The result is numerically equivalent to the per-pair fill up to floating-point rounding.
- Add Barnes-Hut t-SNE as the default optimization method via a new `TSNEMethod { BarnesHut { angle }, Exact }` selector. `BarnesHut` (default, `angle = 0.5`) keeps the joint affinities sparse over each point's `k = min(n − 1, ⌈3·perplexity⌉ + 1)` nearest neighbors and summarizes the repulsive forces together with their normalizer `Z` through a `d`-dimensional space-partitioning tree (a cell collapses to its center-of-mass summary when `cell_width / dist < angle`), cutting the per-iteration cost from `O(n²)` to roughly `O(n log n)`; `Exact` keeps the dense all-pairs gradient. Both share the same factor-4 gradient scale, so one learning rate fits either, and `Z` is reduced sequentially so a fixed seed stays bit-reproducible (otherwise t-SNE's chaotic dynamics amplify a differing summation order into a different embedding). `BarnesHut` requires `n_components ≤ 3` (else `Exact`). The nearest-neighbor search is a one-time brute-force `O(n²)` pass, matching the old distance-matrix cost; the asymptotic win is in the per-iteration forces.
- Add PCA initialization for t-SNE through a new `Init { PCA, Random }` selector. `Init::PCA` (the default) seeds the embedding from the top principal components of the input (rescaled to a small leading-component spread), giving a deterministic, well-spread, seed-independent start; `Init::Random` keeps the seeded small-noise start. Falls back to random when the input has fewer features than components or the leading component is degenerate.
- Add `train_test_split_stratified`, which splits each class independently so both subsets keep the input's class proportions — preventing a class from being absent on one side with imbalanced data. Requires `A: Clone + Eq + Hash`, groups sample indices in first-appearance order for determinism under a fixed seed, and errors on any class with fewer than 2 samples. Re-exported from `utils` and the prelude.
- Add an internal `machine_learning::spatial` kd-tree (`KdTree`) to accelerate neighbor queries in DBSCAN and KNN. Pruning is performed in a metric "comparable space" (Euclidean → squared distance, Manhattan → the distance itself, Minkowski-`p` → the `tᵖ` form) so a single tree serves every `DistanceCalculationMetric`; it exposes `build`, `radius_neighbors`, and `k_nearest` (a `BinaryHeap` over a `(distance, index)` total order, with count-median splits on the max-spread axis). Adds the supporting `DistanceCalculationMetric::{comparable_scalar, comparable_distance, distance_from_comparable, within}` methods.

### Changed
- `SVC` (training Gram matrix) and Kernel PCA (the fit and transform (cross-)kernel matrices) now build their kernel matrices through `KernelType::compute_matrix`, replacing their private per-pair `compute` loops — collapsing an `n·m` swarm of scalar dot products into one parallel GEMV-per-row pass.
- Reformulate the hot distance loops in `KMeans`, `KNN`, and `MeanShift` around the `‖x − y‖² = ‖x‖² + ‖y‖² − 2·x·y` identity, turning each per-pair inner loop into a single GEMV: `KMeans` assignment picks the nearest centroid via `argmin_j (‖c_j‖² − 2·c_j·x)` from one `C·x` GEMV per sample, then recomputes the winner's true squared distance directly so reported inertia stays bit-identical; `KNN` computes one query's Euclidean distances to every training row from a single `X_train·x` GEMV; and `MeanShift`'s per-iteration RBF weights and weighted mean become two GEMVs (`X·c` and `Xᵀ·w`). `LinearSVC` likewise evaluates its hinge-loss margins as a single `X·w + b` GEMV, matching `predict`.
- t-SNE's gradient-descent optimizer gains adaptive per-parameter gains (Jacobs' delta-bar-delta: each coordinate's gain grows by a fixed increment while the gradient keeps its direction — the current gradient and the previous update increment disagree in sign — and decays multiplicatively when the step oscillates — they agree in sign — floored at a small minimum), making the embedding markedly more robust on well-separated clusters. **Behavior change:** embeddings now differ from earlier versions. The pairwise distance is intentionally kept as the exact per-pair form rather than the squared-norm GEMM identity, which loses precision by catastrophic cancellation on uncentered data where the cross term dwarfs the true distances.
- **Breaking:** `TSNE::new` takes two new trailing arguments, `init: Init` and `method: TSNEMethod`; the `Init::Pca` variant is renamed `Init::PCA`; and the chained `with_init` builder is removed in favor of the constructor argument. `Default` and the doctest now use `Init::PCA` + `BarnesHut { angle: 0.5 }`. **Behavior change:** the default embedding now uses PCA initialization and the Barnes-Hut gradient, so results differ from the previous random-init/exact default (and, since PCA init is deterministic, no longer depend on `random_state`).
- Replace LDA's `Shrinkage::Auto` heuristic with the closed-form Ledoit-Wolf optimal shrinkage intensity `δ = b² / d²`, computed from the pooled within-class scatter and the per-sample dispersion term `Σ ‖z_k‖⁴` (threaded out of the per-class statistics) and clamped to `[0, 1]`. The previous `n_features / (n_samples + n_features)` ratio was labeled "Ledoit-Wolf style" but was not the optimal estimator.
- Make LDA's `Solver::LSQR` a genuine iterative least-squares solve (the Paige-Saunders LSQR algorithm) of each class's scoring system `Σ · coef = μ_c`, instead of a relabeled SVD pseudo-inverse that produced numerically identical results to `Solver::SVD`. Solver dispatch now yields the per-class scoring coefficients directly: `Eigen`/`SVD` form a symmetric inverse and multiply it by the class means, while `LSQR` never materializes an inverse — so the three solvers are now genuinely distinct methods. The discriminant projection used by `transform` stays solver-independent.
- Document that `standardize` uses the population variance (divides by `n`), matching scikit-learn's `StandardScaler`, with no sample-variance (`n − 1`) option.
- DBSCAN and KNN use the new kd-tree for neighbor search when `n_features ≤ 16`, falling back to the linear scan above that (where kd-tree pruning stops paying off). DBSCAN's `region_query` calls `radius_neighbors`; KNN lazily builds and caches the tree in a serde-skipped `OnceLock` that is invalidated on each `fit`. Both unify their neighbor tie-break to a `(distance, index)` total order, so the tree and brute-force paths return identical neighbor sets and predictions stay deterministic.
- Rewrite the decision tree's numeric-split search to sort each candidate feature once and sweep thresholds incrementally — carrying running class counts (classification) or a running sum / sum-of-squares (regression) and scoring each split in `O(1)` through a shared `impurity_from_counts` — instead of recomputing both child impurities from scratch at every threshold (`O(n log n)` per feature rather than `O(n²)`).
- `SVC`'s decision function builds the support-vector kernel matrix in one batched `compute_matrix` pass and evaluates `K · (α ⊙ y) + b`, replacing the per-`(sample, support vector)` scalar-kernel loop.
- Rewrite the LDA projection to solve the generalized eigenproblem `S_b w = λ S_w w` through a whitening transform — `S_w = U diag(d) Uᵀ` gives the whitening `W = U diag(d^{−1/2})`, then a symmetric eigendecomposition of `Wᵀ S_b W` whose eigenvectors map back as `w = W v`. These are the correct discriminant directions; the previous code took the left singular vectors of the non-symmetric `S_w⁻¹ S_b`, which are not the discriminant axes. The per-class linear-scoring parameters are cached at fit so `predict` scores in parallel over rows without recomputing them or cloning the input.
- PCA's randomized SVD re-orthonormalizes the subspace with a QR step between each power iteration (Halko et al.), so the iterates no longer collapse toward the dominant singular vector and corrupt the trailing components.
- Standardization computes its mean and variance in a single numerically-stable pass via Welford's online algorithm (with Chan's parallel merge on the parallel path), replacing the prior less-stable computation.
- `LinearSVC` accumulates each minibatch gradient in place with `scaled_add` into per-thread accumulators (rayon `fold`/`reduce`), removing a temporary allocation per sample.
- t-SNE's parallel pairwise-distance and Student-t affinity matrices compute the upper triangle once and scatter it symmetrically, instead of computing every pair twice.
- Kernel PCA tolerates non-positive eigenvalues instead of failing the fit: a centered Gram matrix is only PSD up to round-off and non-Mercer kernels (e.g. `Sigmoid`) legitimately yield slightly negative trailing eigenvalues. Validation now rejects only non-finite eigenvalues, and the degenerate components are zeroed at projection time (scale `0`), matching scikit-learn.

### Fixed
- DBSCAN: tighten the cluster-expansion guards so an already-labeled point is not reprocessed and only still-unlabeled neighbors are enqueued (skip when `labels[q] >= 0`, enqueue when `labels[r] < 0`), preventing two touching clusters from bleeding into each other.
- Isolation Forest: normalize anomaly scores by the average path length of the actual subsample size `c(sample_size)` rather than `c(max_samples)`, and return `1.0` when `c(sample_size) ≤ 0` (e.g. a single-sample subsample) instead of producing `NaN`. Adds a `sample_size` field and `get_sample_size` getter.
- K-means++: skip zero-distance candidates in the roulette-wheel center selection (`dist > 0.0 && cumulative_dist >= choice`), so an already-chosen or coincident point cannot be picked again as a new center.
- Linear regression: report the L2 penalty in the cost as `0.5·α·‖w‖²` to match the `α·w` gradient and the half-MSE data term (it was `α·‖w‖²`, inconsistent with the gradient). Fitted models are unchanged; only the reported cost is corrected.
- Logistic regression: evaluate the reported training cost from the same weights as the logits (and apply the update in place with `scaled_add`), rather than mixing pre-update logits with post-update weights.
- Mean-Shift: seed from every point instead of a capped 100-point random subset (which could miss clusters on larger data), and report `n_samples_per_center` as the number of samples actually assigned to each converged center. Fitting is now fully deterministic as a result.

### Removed
- **Breaking:** remove `MeanShift`'s `random_state` constructor parameter, field, and `get_random_state` getter. Seeding from every point made `fit` deterministic, so the seed no longer influenced anything (`MeanShift::new` now takes five arguments). The standalone `estimate_bandwidth` keeps its own independent `random_state`.
- Remove LDA's internal `cov_inv` field (the cached covariance inverse). The per-class scoring coefficients are now produced directly by the solver, so a full inverse is never materialized or stored. The field was never publicly exposed; removing it changes the serialized model layout.
- Remove the now-unused `machine_learning::parallel::try_map_collect` helper (and its tests); `SVC`'s batched kernel path no longer needs the per-pair fallible parallel map.

## [v0.12.0] - 2026-06-09 (UTC-7)
### Added
- Add a crate-level `random` module for reproducible pseudo-random number generation. `set_global_seed(u64)` / `clear_global_seed()` (re-exported at the crate root) set a thread-local global seed, and an internal `make_rng` resolves every component's `random_state: Option<u64>` against it: an explicit `Some(seed)` is used as-is and never consumes the global stream, `None` derives an independent sub-seed from the global stream when one is set, and otherwise falls back to entropy. This gives one-call whole-crate reproducibility with per-component override (the local-over-global rule mirrors Keras), reusing the `random_state` convention the machine-learning estimators already expose.
- Add a `random_state: Option<u64>` parameter to every `neural_network` layer constructor (`Dense`, `Conv1D`/`Conv2D`/`Conv3D`, `DepthwiseConv2D`, `SeparableConv2D`, `SimpleRNN`, `LSTM`, `GRU`, `Dropout`, `SpatialDropout1D`/`2D`/`3D`, `GaussianNoise`, `GaussianDropout`) for reproducible weight initialization and dropout/noise masks, plus `Sequential::new_with_seed(u64)` and `Sequential::set_seed(u64)` to make the fit-time minibatch shuffle reproducible.
- Add `tests/neural_network/reproducibility.rs` covering same-seed determinism, seed divergence, global-seed reproducibility, local-overrides-global precedence, and reproducible end-to-end training.
- Rewrite the entire integration test suite under a per-feature "Route C" layout — `tests/<feature>/main.rs` crate roots with per-topic submodules and a shared `common.rs`, gated by `required-features` (`autotests = false`) — replacing the flat `*_test.rs` files. Every expected value is derived from independent ground truth (math definitions / hand calculations) rather than traced from the implementation, and coverage now spans constructor and runtime error paths, data-size-dependent parallel branches, and private numerical kernels (the latter via inline `#[cfg(test)]` unit tests in their source files).

### Changed
- **Breaking:** every `neural_network` layer constructor listed above takes a new trailing `random_state: Option<u64>` argument. Stochastic layers (`Dropout`, `SpatialDropout*`, `GaussianNoise`, `GaussianDropout`) now own a seeded RNG, and all weight initialization, dropout/noise masks, and the `Sequential` minibatch shuffle draw through `crate::random::make_rng` instead of an unseeded thread RNG — so a seed makes them fully reproducible while `None` preserves the previous non-deterministic behavior.
- Route the `machine_learning` / `utils` estimators that own a `random_state` (`KMeans`, `MeanShift`, `IsolationForest`, `SVC`, `LinearSVC`, t-SNE, and `train_test_split`) through `crate::random::make_rng` as well, replacing their inline `match random_state` blocks. An explicit `Some(seed)` is unchanged, but a `None` seed now honors `set_global_seed` — so one `set_global_seed` call governs the whole crate's randomness, not just `neural_network`. PCA and Kernel PCA are intentionally left as-is: their only randomness is in `linalg`'s iterative eigensolvers — which converge regardless of the random start, so they seed it with a fixed constant purely for determinism — and PCA's `SVDSolver::Randomized(u64)`, whose seed is always user-supplied with no unseeded fallback.
- **Breaking:** rename `KMeans`'s `random_seed` field, `new()` parameter, and `get_random_seed()` getter to `random_state` / `get_random_state()`, matching every other estimator and scikit-learn. (The constructor argument is positional, so only the getter name changes for callers.)
- Make `DecisionTree`'s `random_state` functional instead of a reserved no-op: it now seeds random tie-breaking among equally-scoring splits (scikit-learn-style — features achieving the same impurity decrease). With `Some(seed)` or an active global seed, tied splits are chosen randomly but reproducibly; with `None` and no global seed the tree stays fully deterministic (trees without ties, and unseeded usage, are unchanged). Adds the internal `crate::random::make_rng_opt` helper for "randomize only when a seed is in effect" callers.

### Fixed
- `roc_auc`, `roc_curve`, `precision_recall_curve`, and `average_precision` no longer hang or exhaust memory on a `NaN` score. They sort scores with `total_cmp` (a NaN-safe total order) but grouped equal-score ties with `==`; because `NaN == NaN` is `false`, the tie-grouping loop never advanced past a `NaN` element — an infinite loop, and unbounded allocation in the curve builders. Ties are now grouped with a `NaN`-aware equality, so a `NaN` score is ordered deterministically as the docs already promised, and all finite results are byte-identical.
- `variance` and `standard_deviation` (`math`) now skip non-finite (`NaN`/`±∞`) values and compute over the finite subset (dividing by the finite count), instead of `variance` short-circuiting to `0.0` (its NaN-skipping fold was unreachable dead code) and `standard_deviation` propagating `NaN`. `standard_deviation` is now defined as `sqrt(variance)`, keeping the two consistent; an input with no finite values returns `0.0`, and all-finite inputs are unchanged.

## [v0.12.0] - 2026-06-08 (UTC-7)
### Added
- Add the `adjusted_rand_index` (Adjusted Rand Index) and `silhouette_score` (mean silhouette coefficient, Euclidean) clustering metrics, completing the clustering set the crate documentation already advertised.
- Add multi-class classification support: a `MulticlassConfusionMatrix` (K x K counts, per-class precision/recall/F1/support, an `Average` enum for macro/micro/weighted aggregation, and a `summary()` that prints the matrix grid followed by a `classification_report`-style table, mirroring `ConfusionMatrix::summary()`), plus `log_loss`, `cohen_kappa`, `top_k_accuracy`, `average_precision`, `roc_curve`, and `precision_recall_curve`. The binary `ConfusionMatrix` gains `mcc` (Matthews correlation coefficient) and `balanced_accuracy`, both now also shown in its `summary()`.
- Add the regression metrics `explained_variance_score`, `median_absolute_error`, and `mean_absolute_percentage_error`.
- Add the clustering metrics `homogeneity_score`, `completeness_score`, `v_measure_score`, `fowlkes_mallows_score` (entropy- and pairwise-based external metrics that reuse the existing contingency/entropy machinery), and the internal indices `davies_bouldin_score` and `calinski_harabasz_score`.
- Export all of the above from `metric_prelude`.

### Changed
- Split the `metric` module into public `regression`, `classification`, and `clustering` submodules, with every item also re-exported at the module root — so each metric is reachable both by category (`metric::regression::mean_squared_error`) and flat (`metric::mean_squared_error`), and the existing flat paths are unchanged.
- Standardize every paired metric on `(y_true, y_pred)` argument order (ground truth first, matching scikit-learn and the clustering metrics). **Breaking:** this swaps the argument order of `r2_score` and `ConfusionMatrix::new`; for the symmetric metrics (MSE, RMSE, MAE, accuracy) only the parameter names change.
- Rename `calculate_auc` to `roc_auc` and reorder its arguments to `(labels, scores)`.
- Make metric panic messages mirror the crate's `Error` wording (`dimension mismatch: expected .., found ..`, `input is empty: ..`), and panic uniformly on empty input — the regression metrics previously returned `0.0` for empty arrays.
- `r2_score` now returns `1.0` for a perfect fit on zero-variance ground truth (previously always `0.0`), matching scikit-learn.
- `roc_auc` sorts scores with `total_cmp`, so it no longer panics on `NaN` scores; `normalized_mutual_info` / `adjusted_mutual_info` no longer panic on non-contiguous array views.
- Optimize the Adjusted Mutual Information's expected-MI term with a shared log-factorial table, turning each binomial coefficient into an `O(1)` lookup.
- Derive `Debug, Clone, Copy, PartialEq, Eq` for `ConfusionMatrix`, and render `ConfusionMatrix::summary` as an aligned table.
- Rename modules for naming consistency — plural for collection modules, abbreviations, and de-stuttering. **Breaking:** `metric` → `metrics` and `utility` → `utils`; under `neural_network`, `layer` → `layers`, `optimizer` → `optimizers`, `loss_function` → `losses`, and `neural_network_trait` → `traits`; the `layers` category submodules drop their `_layer` suffix (`activation_layer` → `activation`, `convolution_layer` → `convolution`, `pooling_layer` → `pooling`, `recurrent_layer` → `recurrent`, `regularization_layer` → `regularization`, and the nested `dropout_layer`/`noise_injection_layer`/`normalization_layer` → `dropout`/`noise_injection`/`normalization`); `machine_learning::meanshift` → `mean_shift` and `machine_learning::linear_discriminant_analysis` → `lda`; `utils::principal_component_analysis` → `pca`; and the `prelude` submodules drop their `_prelude` suffix (`machine_learning_prelude` → `machine_learning`, `math_prelude` → `math`, `metric_prelude` → `metrics`, `neural_network_prelude` → `neural_network`, `utility_prelude` → `utils`). All public re-export paths, doc examples, and tests are updated accordingly. The two Cargo features that gate a renamed module are renamed to match (preserving the crate's feature-name-equals-module-name convention): **`utility` → `utils`** and **`metric` → `metrics`** — so downstream crates must now enable `features = ["utils"]` / `["metrics"]` (and `full` is updated accordingly).
- Rename the `LossFunction` trait to `Loss` (now in `neural_network::traits`), aligning the three core abstractions with their now-plural modules: `layers`/`Layer`, `optimizers`/`Optimizer`, `losses`/`Loss`. **Breaking.**

### Removed
- Remove the now-unused `ActivationLayer` trait from `neural_network`; its forward/derivative dispatch is fully served by the serializable `Activation` enum.

## [v0.12.0] - 2026-06-07 (UTC-7)
### Changed
- Refactor error handling into a single unified `Error` type built on `thiserror`, replacing the stringly-typed `ModelError` and the separate `IoError`. Adds structured shared variants (`EmptyInput`, `DimensionMismatch`, `ShapeMismatch`, `NonFinite`, `InvalidParameter`, `InvalidInput`, `NotFitted`, `NotConverged`, `Computation { context, source }`) in place of the `InputValidationError` / `ProcessingError` catch-alls, with domain-specific failures grouped into nested `NnError`, `TreeError`, and `IoError` sub-enums. Adds smart constructors (`Error::dimension_mismatch`, etc.), a `Context` extension trait (`.context()` / `.with_context()`) that wraps foreign errors while preserving the source chain, and a `RustymlResult<T>` alias. **Breaking:** `ModelError` is renamed `Error`, its variants are restructured, and the type is now `#[non_exhaustive]` and no longer derives `PartialEq` / `Clone`.
- Refactor the entire `neural_network` module for quality, correctness, and consistency (a net reduction of ~1360 lines while adding features). Replaces the `T: ActivationLayer` generic on `Dense` / `Conv*` / `RNN` with a serializable `Activation` enum (eliminating the load-time downcast cascade and monomorphization bloat); adds a generic optimizer interface (`Layer::parameters() -> Vec<ParamGrad>` plus flat-slice `sgd` / `adam` / `rmsprop` / `adagrad` kernels) that removes all per-layer/per-optimizer update code; adds an inference-mode `Layer::predict(&self)` / `Sequential::predict(&self)` that writes no caches and borrows `&self` for concurrent inference; replaces the per-rank convolution/pooling code with dimension-generic engines; adds channel-last (NHWC) Instance/Group normalization and multi-axis `LayerNorm`; uses a real Gram-Schmidt orthogonal recurrent-kernel initializer for SimpleRNN/GRU/LSTM; makes the loss trait return `Result` instead of `assert!` panics; and splits `helper_function.rs` into `shape_helpers` / `conv_op_helpers` / `validation`.
- Make `cargo doc` warning-free: drop 37 redundant explicit targets on `[`Layer::predict`]` intra-doc links, and fully-qualify the unresolved `RegularizationType` / `DistanceCalculationMetric` links in the `types` module-level docs.

## [v0.12.0] - 2026-06-06 (UTC-7)
### Changed
- Refactor the `utility` module: add shared `validation` (input checks) and `linalg` (power iteration plus a new pure-Rust Lanczos solver) submodules, removing duplicated validation and the two near-identical power-iteration copies across PCA and Kernel PCA; move per-variant computation onto the config enums (`SVDSolver`, `EigenSolver`, the normalization/standardization axes). Kernel PCA gains an `EigenSolver::Lanczos` variant and renames the mislabeled `ARPACK` solver to `PowerIteration` (likewise for PCA); t-SNE drops the meaningless save/load on a stateless model and vectorizes its momentum update; `label_encoding` now returns `Result` instead of panicking (NaN-safe argmax); `train_test_split` gains a generic label type; and `utility` switches to explicit re-exports.
- Move `LinearDiscriminantAnalysis` from `utility` to `machine_learning` — as a supervised classifier (`fit(x, y)` + `predict`) it belongs with the estimators. It now implements the shared `Fit` / `Predict` traits, reuses `machine_learning::validation`, and moves per-solver logic onto the `Solver` enum (within-class scatter computed as a single GEMM). **Breaking:** LDA's import path changes from `utility` to `machine_learning`, and the `machine_learning` feature now enables `nalgebra`.
- Collapse nested `if`s into edition-2024 let-chains (`clippy::collapsible_if`) in `isolation_forest`, `kmeans`, and `knn`.

## [v0.12.0] - 2026-06-05 (UTC-7)
### Added
- Add `hinge_loss` to the `math` module (mean hinge loss for margin-based classifiers), alongside `logistic_loss`, and export it from the math prelude.

### Changed
- Encapsulate decision tree per-algorithm behaviour as methods on the `Algorithm` enum: the impurity criterion (Gini/entropy), the split-selection score (C4.5 gain ratio vs. raw impurity decrease), and the regression / multi-way-categorical capability checks. This replaces the `match self.algorithm` branches scattered across `DecisionTree`, and the now-exhaustive matches turn adding a new algorithm into a compile-time checklist instead of a silent fall-through.
- Document the membership rule for the `math` module (pure, model-agnostic, reusable primitives that are shared by more than one caller) in its module-level docs.
- `LinearSVC` now computes its training cost through `math::hinge_loss` instead of an inline hinge sum.
- `MeanShift` now computes its RBF neighbour weights through `KernelType::RBF`, sharing the single kernel-dispatch implementation in `types` (mirroring how the distance metrics are already dispatched).
- `metric::r2_score` now reuses `math::sum_of_squared_errors` and `math::sum_of_square_total` instead of recomputing SSE/SST inline. The `metric` feature now enables `math`.
- Update dependencies and raise the minimum supported Rust version to 1.89.0: `nalgebra` 0.34.1 -> 0.35.0 (source-compatible; required by the PCA / LDA / Kernel PCA solvers), `rayon` 1.11 -> 1.12, and `serde_json` 1.0.149 -> 1.0.150, plus refreshed transitive dependencies. 1.89 is the true minimum (nalgebra 0.35 -> simba 0.10 -> wide / safe_arch require it).

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
