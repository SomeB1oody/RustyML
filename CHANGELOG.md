# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24, summarized per version — each entry lists only the significant changes for that release.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [Unreleased]
### Added
- **A checkpoint now addresses every array by name, and `Sequential` gained 3 methods for it.** `weight_paths()` lists every address of a model in file order, and `weight(path)` reads 1 of them as a view that borrows the live array. `load_partial_from_path` is the opt-in lenient load: it applies what the file and the model agree on and returns a `LoadReport { applied, missing, unused }` of checkpoint paths. `load_from_path` stays strict and refuses on any disagreement.
  - An address is `<scope>.<name>`, the position of the layer counted from the input and the name the layer gives the array. It is the same pair the optimizer keys its state on, so 1 address serves the training loop and the file alike.
  - Strict is the default because a name and a shape together are a weaker key than the enum they replace. `InstanceNormalization`, `GroupNormalization`, and a rank-2 `LayerNormalization` all expose `gamma [C]` and `beta [C]`, identically, and nothing but the per-layer type name separates them. The load therefore compares the layer type of every position, and the name, the kind, and the shape of every array.
  - The strict load runs in 2 passes and the first one writes nothing, so a refusal leaves the model exactly as it was. The message names the path, for example ``model structure mismatch: `0.kernel` has shape [3, 3] in the model, and [2, 2] in the file``.
- **`WeightKind`, and the `Layer::weights`, `Layer::weights_mut`, and `Layer::weight` accessors that carry it.** `weights()` gives 1 `WeightRef` per array the layer holds, with the name of the array, its kind, and a read view. `weights_mut()` gives the same list as `WeightMut` for writing, under the same names and in the same order. `WeightKind::Trainable` marks a kernel, a bias, or a normalization scale or shift. `WeightKind::NonTrainable` marks state that the layer keeps and no optimizer writes.
  - The 2 running statistics of `BatchNormalization` are the only non-trainable arrays today. A checkpoint holds the kind next to every array, so a file that offers a trainable array where the layer keeps state is refused instead of applied.
- **`ParamCounts`, which replaces `TrainingParameters`.** It carries the trainable count and the non-trainable count together, rather than 1 of the 2. `BatchNormalization` is both at once: for `C` channels it holds `2 * C` trainable elements and `2 * C` non-trainable ones, so the third column of `summary` is now honest and agrees with Keras.
- **`use_bias` on `Dense` and on all 10 convolution layers, and `center` and `scale` on the 4 normalization layers.** Each arrives as a builder method that consumes and returns `self`, and each defaults to `true`, which is the behavior every model had before. All 3 mirror the Keras arguments of the same names.
  - A dropped array leaves the layer, and not only the sum. The layer stops reporting it in `param_count`, stops listing it in `weights`, and stops writing it to a checkpoint. Every other path stays where it was, because a path names an array and never a position inside a layer.
  - The matching `set_weights` argument became optional. Pass `None` for an array the layer does not hold, and the array for one it does. The other order returns `Error::InvalidParameter`, so a bias that reaches a layer holding none is refused rather than dropped in silence.
- **A reserved `BuildConfig` slot in the checkpoint, for the input shape a layer was built for.** No layer fills it today, because every constructor allocates its own arrays and a live model already knows every extent. A load compares the field only when the file and the layer both carry a shape. The slot exists now so that the change moving allocation out of the constructors costs 1 version bump and not 2.
- **`return_sequences` and `go_backwards` on `SimpleRNN`, `LSTM`, and `GRU`.** Both arrive as builder methods that consume and return `self`, and both default to false, which is the behavior every model had before. `with_return_sequences(true)` keeps the time axis, so the layer returns `[batch, timesteps, units]` instead of `[batch, units]`, and slot `k` holds the state after processing step `k`. The backward pass then expects a gradient of that same rank-3 shape. Verified against Keras 3.15.1 to f32 round-off.
  - The last slot of a returned sequence always equals the output of the same layer with the flag off. An `LSTM` keeps its cell state internal in both settings, because there is still no `return_state`.
  - `with_go_backwards(true)` reverses the reading order alone, and it changes no shape. Processing step 0 consumes input timestep `timesteps - 1`, and the output stays in processing order. The layer does not turn the sequence back into input order, which is what Keras does as well.
  - A stack of recurrent layers no longer needs a `RepeatVector` between its members. Set `return_sequences` on every layer of the stack except the last one. `RepeatVector` stays the right bridge for an encoder-decoder model, whose 2 sequences have different lengths.
- **`dilation_rate` on all 10 convolution layers,** through a new `with_dilation_rate` builder on the plain, transposed, depthwise, and separable layers. A dilation of `d` spaces the kernel taps `d` cells apart, so `k` taps span an effective kernel of `(k - 1) * d + 1` cells. The offset of tap `t` at output position `o` is `o * stride + t * d`, and never `(o * stride + t) * d`. The default is 1 on every axis, and it gives the solid kernel every layer had before. The 1D layers take a scalar, and the 2D and 3D layers take a tuple.
  - The method returns a `Result`. A dilation of 0 is `Error::InvalidParameter`, and so is a dilation above 1 together with a stride above 1. Keras refuses the same pair.
  - No layer grew a loop nest. Every geometry helper reads the effective kernel where it read the kernel size, so the padded buffer, the offset table, and both output rules cover the dilated case with no new branch.
- **`padding="causal"` on `Conv1D`,** through the new `ConvPadding` enum, which adds `Causal` beside `Valid` and `Same`. `Conv1D::with_padding` takes `impl Into<ConvPadding>`, so a `PaddingType` still passes unchanged. Causal padding puts all `keff - 1` pad cells on the leading edge and none on the trailing edge, so an output position never reads a later input position. Its output length is the `Same` one, `ceil(in / stride)`.
  - Only `Conv1D` can build it. Every other convolution keeps a `PaddingType` field and converts at the engine boundary, so a causal pass cannot reach a layer that carries more than 1 spatial axis.
- **`Dense` accepts an input of rank 2 or more.** The layer contracts the last axis alone and replaces it with `output_dim`, so a `[N, T, input_dim]` input gives `[N, T, output_dim]`. 1 kernel serves every leading position, which is what lets 1 `Dense` layer transform every timestep of a sequence. Rank 3 is the common case.
  - The extra rank costs nothing. The leading axes fold into 1 row axis, so the pass stays the same single matrix product a rank-2 input runs, and the 2 forms agree bit for bit. The gradient goes back at the rank of the input, and `output_shape` reports the leading axes after the first forward pass.
- **New `Rescaling` layer, which applies `y = x * scale + offset` to every element.** `Rescaling::new(scale)` leaves the offset at 0, and `with_offset(offset)` sets it. Neither call returns a `Result`, because no value is invalid here. A `scale` of 0 and a negative `scale` are both legal. The layer holds no parameter and has no training mode, so `predict` returns exactly what `forward` returns. The backward pass multiplies the incoming gradient by `scale`, and the `offset` has no part in the gradient.
  - The layer keeps no cache, not even the shape of the last input. `backward` therefore runs correctly before any forward pass, and `output_shape` keeps the default answer of "Unknown".
  - Its use is input normalization in front of a model, which keeps that step inside the model instead of in a serving path that can forget it.
- **`noise_shape` on `Dropout`,** through the new `with_noise_shape` builder. It takes 1 entry per axis. `Some(1)` gives the whole axis 1 shared draw, so the same units drop at every position of that axis. `None` takes the extent of the input on that axis, which keeps the draws there independent. Any other entry must equal that extent. A vector shorter than the rank of the input lines up against the last axes, which is the usual right-aligned broadcast rule, so the axes it leaves out share 1 draw.
  - The layer samples the mask at that shape and keeps it at that shape for the backward pass. It broadcasts in both directions and never builds a full-size copy. `Some(0)` is `Error::InvalidParameter`, and an empty vector is `Error::EmptyInput`.

### Removed
- **The `LayerWeight` enum, its 20 container structs, the `ApplyWeights` trait, and the 21-arm downcast dispatch that applied them.** 1001 lines across 22 files. The `layer_weight` and `serialize_model` modules, `SerializableSequential`, `Layer::get_weights`, and `Sequential::get_weights` are gone with them. A checkpoint holds a name and a kind per array now, so nothing needs a closed set of per-layer containers, and the load no longer downcasts a layer to its concrete type to write into it.
  - The append-only variant rule is gone as well. Postcard wrote the variant index and not the variant name, so a new layer had to append its variant at the end of the enum and no earlier index could move. A new layer now adds a type name and some array names, and both travel in the file as strings.
- **5 public tuning items, none of which had a caller.** `tuning::norm::set_/get_bn_plane_stats` governed a plane-fold path that was never written, so the gate had no effect on any input. `tuning::reduction::set_/get_exp_reduce` and the whole `tuning::metrics` module (`set_/get_silhouette`) are gone as well; both gates stay, as plain constants.
  - The silhouette gate must stay a constant. Its parallel fill matches the serial fill numerically but not bit for bit, so a public setter there changed a returned score. Every gate `tuning` exposes selects an execution strategy only, and never changes a result. That promise now holds with no exception.
  - The exp-reduction gate is fixed by the block size. The fold splits its range into `DET_REDUCE_BLOCK` blocks, so a value below 2 blocks gives 1 task and no parallelism. A setter could not improve it.
- **`math::matmul::gemm_chunk_rows` and `math::matmul::cache_resident` are now crate-internal.** Both carried `#[doc(hidden)]` and a written notice that they were not part of the public API. Use the `tuning::matmul` knobs that govern them, `set_/get_chunk_elems` and `set_/get_cache_resident_max_bytes`, which are unchanged.
- **6 unreachable `match` arms in `BatchNormalization`.** Each guard discriminated on `Some`/`None` for a buffer the surrounding code already guarantees is contiguous, so only the rank >= 2 half was ever live, and the other half was a plain rank test computed 2 lines earlier. Verified with a temporary assertion over the whole test suite before the change.
- **`apply_spatial_dropout_threshold`,** whose parallel arm was gated at 4,000,000 elements on a mask that holds 1 value per `(batch, channel)`. The largest shape in the repository reaches 131,072, which is 30x below the gate. The 1 surviving line moves to its 3 call sites.
- **`bench_internals::KdTree`**, whose only consumer was a deleted calibration section. `KdTree` itself is unchanged and still serves DBSCAN and KNN.
- **The `matmul_kernels` benchmark target**, and 4 calibration sections that measured nothing the crate can act on: the channel-chunked BatchNorm fold and the reserved plane fold, which have no path in `src`; the f32 `DET_REDUCE_BLOCK` sweep, which repeats 1 serial timing across all its rows; and the kd-tree ladder, which ran rayon in both of its columns and therefore reported 2 parallel timings as a crossover.

### Fixed
- **10 activation forward passes no longer copy the whole tensor before mapping it.** Each parallel arm ran `z.clone()` and then `par_mapv_inplace`, so it paid a serial full-tensor copy before the parallel map started, while its serial twin did 1 fused allocate-and-map pass. `Zip::par_map_collect` does the whole thing in 1 parallel pass. Measured on a 9950X: the parallel arm is 1.9x to 3.9x faster (Tanh at 262,144 elements goes from 230.8 to 68.3 us, Sigmoid from 203.4 to 52.7 us). ReLU and Sigmoid also drop a copy from their serial arm. No result bit changes, and the output layout is what it always was.
- **k-means gated 2 reductions on a product that does not count their tasks.** Both compared `n_samples * n_features` (or a cluster-count equivalent) against the sum gate, while `det_reduce_range` blocks the sample axis alone. A wide, short input cleared the gate while the fold still had 1 task, so the parallel path was pure overhead. Both sites now also require the sample count to reach 2 blocks of `DET_REDUCE_BLOCK`. The bench doc and guide 6.3 asserted that the product is what the fold walks, which the bench's own table contradicts; both are corrected.
- **Pooling backward could fan out into exactly 1 task.** The gate is calibrated on the forward pass, which splits output positions, but the backward pass splits channels and its slab has a floor of 16. A 1-item batch with no more than 16 channels therefore produced 1 task however much work it held. The backward path now also checks the task count. The check only narrows, so no shape that already won can regress.
- **Moving the BatchNormalization gate changed the result.** The parallel and the serial arm of the input-gradient pass associated 1 term differently: the parallel arm computed `(grad_var * x_centered * 2) / batch_size` and the serial arm computed `grad_var * ((x_centered * 2) / batch_size)`. The 2 forms round in different places, so the arms disagreed in the last bit whenever the row count was not a power of 2. `crate::tuning` promises that a gate selects an execution strategy and never changes a result, so this was a broken contract and not a tolerance question. The parallel arm now takes the serial arm's association, and a new test asserts bit-for-bit equality across the gate for the output, the input gradient, and the gamma and beta gradients.
- **BatchNormalization's 4 parallel passes no longer do an integer division per element.** Each one indexed its per-channel table with `i % feature_size` over a flat parallel iterator. `feature_size` is a runtime value, so the compiler cannot strength-reduce it and had to emit a hardware division for every element. The passes now walk row chunks of exactly `feature_size` elements and index the table by position inside the row. Measured on a 9950X: 1.47x faster at 5 channels, and 2 to 5 percent at 64 and 128 channels, where memory bandwidth dominates.
- **Sigmoid, tanh, and softmax backward no longer fork rayon about 30x too early.** All 3 reuse the cached activation and call no `exp`, so they belong to the cheap-map class at 4,000,000 elements and not the exp-map class at 131,072. The exp-free arms beside them (LeakyReLU, ELU, SELU, Softsign, HardSigmoid, Exponential) were already classed correctly. Softplus backward stays in the exp class, because its derivative calls `exp_m1`.
- **Softmax forward and backward no longer fan out over a single row.** Both spread work 1 row per task, but both compared the total element count against the gate, so a `[1, 200000]` input cleared it and forked a 1-item rayon iterator. Both now also require more than 1 row.

### Changed
- **Behavior change: `MODEL_FORMAT_VERSION` is 2, and every checkpoint written by an earlier release stops loading.** Version 1 held the closed enum of per-layer weight containers and wrote its variant index in place of any name. Version 2 is the named checkpoint. No byte of the 2 layouts agrees, so the header check refuses a version 1 file with `IoError::UnsupportedModelFormat`, and the message names both versions. Rebuild the model under this release, re-run training, and save again.
- **Behavior change: `BatchNormalization` renamed its 2 running arrays to the Keras names `moving_mean` and `moving_variance`,** and both now carry `WeightKind::NonTrainable`. The `set_weights` parameters follow the same 2 names. The values and the update rule do not move.
- **Behavior change: the optimizers key their per-parameter state on a name, not on a cursor.** All 5 kept that state in a flat vector indexed by a cursor that walked the whole model, and the only self-check was a length comparison. So 2 tensors of equal length that changed places silently exchanged their moments, and `gamma` and `beta` of a normalization layer always have equal lengths and sit in adjacent slots. The arity was data-dependent as well, because most parameter lists gated on a tuple of gradients and yielded nothing before the first backward pass.
  - Every parameter now carries a name that the layer supplies, and the optimizers key on `ParamId`, which pairs that name with the index of the layer counted from the input. A layer that changes how many parameters it yields, or that is added or reordered, no longer disturbs any other tensor. The length check remains, and now means that a named tensor was resized.
  - The names follow Keras. `kernel` replaces `weight` on `Dense` and on the 8 convolution layers that had it, and `depthwise_kernel` and `pointwise_kernel` replace `depthwise_weight` and `pointwise_weight` on the 2 separable layers.
  - The update walk now runs from the input, which is the order `global_grad_norm` already used. Each update reads only its own value, gradient, and state, so no result moves.
- **Behavior change: the rule that an input must not be shorter than the kernel now applies under `Valid` padding only, and it runs in the forward pass.** A configuration that used to fail in the constructor now builds. `Conv2D::new(1, (3, 3), vec![1, 2, 2, 1], (1, 1), Activation::Linear)` returns `Ok`. Under `Valid` its `forward` returns `Error::InvalidInput`, which is where the old constructor error moved to. Under `Same`, and under `Causal` on a `Conv1D`, the same layer runs and returns the size those 2 rules give. That is what Keras does. Measured across every convolution layer type and every padding mode.
  - The check cannot stay in a constructor. `with_padding` and `with_dilation_rate` both run after the constructor, and both decide whether the rule applies at all, so the padding mode is not final until the layer runs.
  - Only `Valid` bounds the kernel by the input, because it reads complete windows alone. `Same` and `Causal` add the missing cells on the borders, so every extent stays legal there. The rule reads the effective kernel, so a dilated layer needs `(k - 1) * d + 1` cells and not `k`.
  - A transposed convolution grows its input and puts no such bound on it, so this rule never applied to those 3 layers.
- **`CACHE_RESIDENT_MAX_BYTES` drops from 64 MiB to 32 MiB.** The old value was the calibration machine's package L3 total, but a core on that part reaches only its own die's half, and each GEMV sweep runs inside its own rayon task. `lscpu` reports "L3 cache: 64 MiB (2 instances)", and the backend sets 32 MiB for the same part with the contract "Every backend must report the per-core-reachable slice, not a package aggregate". Both guides already told users to write 32 MiB on this CPU. The wording that produced the error, "set this to the machine's actual shared-L3 size", is corrected everywhere. This is an alignment with the backend's own topology figure, not a measured crossover: no calibration section exists for this gate.
- **4 more normalization gates, merged rather than deleted.** `tuning::norm` had 7 gates for 3 layers, all shipping the same value. Below `set_/get_batch_norm` the 3 layers share only 2 kernel shapes, so 2 gates now cover them.
  - `set_/get_col_fold` replaces `set_/get_bn_col_stats`, `set_/get_ln_col_stats`, and `set_/get_gn_param_grad`. All 3 fed the same 2 fold kernels with the same `[M, C]` view, in the same argument position.
  - `set_/get_row_pass` replaces `set_/get_ln_row` and `set_/get_gn_row`. Both gated a row-block sweep of the same shape.
  - No result bits move. The fold kernels take their block boundaries from the input shape alone, so the flag decides where the work runs and never what it computes. Both merged gates keep 262_144.
- **The calibration takeaway rule no longer erases or misplaces a crossover.** It keyed on the LAST losing rung, so 1 slow top rung cancelled a whole ladder and 1 interior dip pushed the bracket past a real win. It now takes the first winning rung that has at most 1 losing rung above it, names that rung instead of hiding it, and collapses tied work values to their slowest rung. `benches/calibrations/RESULTS.md` is regenerated.
  - The LayerNorm row ladder reported no crossover at all. It crosses between 65,536 and 262,144, which is where the gate sits. Its largest rung holds 4 times its own size in buffers, so both columns fall to memory bandwidth; that rung is now labelled.
  - The pooling bracket read 25,088 to 49,152 and hid a measured 2.8x win at 16,384 taps. It is 12,288 to 16,384, so `POOL_PARALLEL_MIN_OPS` at 12,000 now agrees with its own measurement instead of overriding it.
  - Re-measurement moved 1 gate out of its bracket: the serial convolution path is more than twice as fast as at the previous calibration, so `CONV_PARALLEL_MIN_FLOPS` at 4,000,000 now sits below a bracket of 8,294,400 to 35,426,304 rather than inside it.
- **`cargo bench --bench parallel_gates` no longer writes into the source tree.** The report goes to `target/parallel_gates/RESULTS.md`. Set `RUSTYML_WRITE_RESULTS=1` to refresh the tracked copy at `benches/calibrations/RESULTS.md`. A write failure now prints a warning instead of a panic.
- **The `nn_end_to_end` Dense benchmark times `predict`, not `forward`.** `forward` writes an input cache and an output cache, so the old loop held 2 extra tensors live per iteration and measured the allocator as much as the kernel.

## [v0.15.0] - 2026-08-20
### Added
- **7 new `Activation` variants: `LeakyReLU`, `ELU`, `SELU`, `Softplus`, `Softsign`, `HardSigmoid`, and `Exponential`.** Each one also ships as a thin standalone layer of the same name in `neural_network::layers::activation`. `LeakyReLU::new(negative_slope)` and `ELU::new(alpha)` return a `Result` and default to `0.3` and `1.0`, and the other 5 take no argument. The new `Activation::validate` runs in all 14 trainable layer constructors. A `negative_slope` or `alpha` that is not finite and above 0 is now `Error::InvalidParameter` at construction, not at the first forward pass.
  - `LeakyReLU` takes the positive branch at `x >= 0`, while `ELU` and `SELU` take it at `x > 0`. The derivative at exactly 0 is therefore 1 for `LeakyReLU`, `alpha` for `ELU`, and `scale * alpha` for `SELU`.
- **3 new transposed convolution layers: `Conv1DTranspose`, `Conv2DTranspose`, and `Conv3DTranspose`.** Each one runs a convolution backwards over its spatial axes, so it grows a tensor instead of shrinking one. The constructors mirror their plain counterparts, plus `with_padding`, `with_random_state`, and `set_weights`. Output size is `input * stride + max(kernel - stride, 0)` under `PaddingType::Valid` and `input * stride` under `PaddingType::Same`. Unlike the plain convolution layers, these put no lower bound on the input spatial size. Verified against Keras 3.15, forward and backward.
  - The kernel is `[k..., filters, channels]`, with the filter axis **before** the input-channel axis. That is the reverse of the plain convolution kernel, so `set_weights` refuses a kernel laid out for the matching `Conv2D` whenever the 2 counts differ.
  - There is no `output_padding` argument, so a convolution followed by its transpose recovers the original size only under a condition on each axis. Under `PaddingType::Valid` the kernel must be at least the stride, and the stride must divide `input - kernel`. Under `PaddingType::Same` the stride must divide the input. Resize afterwards with a border layer when a condition does not hold.
  - The shared `conv_transpose_engine` is the existing `convolution_engine` with its 2 halves exchanged. Every geometry helper, the offset table, and the block copy are reused, and the existing `tuning::conv::set_parallel_min_flops` gate covers both engines.
- **2 new convolution layers: `DepthwiseConv1D` and `SeparableConv1D`,** which complete the depthwise and separable family at rank 1. The constructors mirror their 2D counterparts with scalar arguments instead of tuples. The depthwise kernel is `[kernel_size, channels, depth_multiplier]` and the pointwise kernel is `[1, channels * depth_multiplier, filters]`. Verified against Keras 3.15, forward and backward.
  - Neither layer adds a loop nest. Both set the height terms of the shared depthwise geometry to 1 and pass the flat slices of their own rank-3 arrays. Nothing is repacked, and the existing `tuning::conv::set_naive_parallel_min_flops` gate covers them.
- **New `Embedding` layer, a trainable lookup table that turns whole-number indices into dense vectors.** `Embedding::new(input_dim, output_dim)` appends `output_dim` as a trailing axis, so `[N, T]` indices give `[N, T, output_dim]`, at any input rank of 1 or more. The table starts from a uniform draw over `[-0.05, 0.05]`, and `.with_random_state(seed)` makes that draw reproducible. The backward pass adds the upstream gradient into the row each index selected, so a repeated index accumulates and the result reproduces bit for bit. Verified against Keras 3.15, forward and backward.
  - A `Tensor` holds `f32`, so an index arrives as a floating-point value. It is truncated toward zero and then range-checked. An index outside `0..input_dim`, and any non-finite value, gives an `Error::InvalidInput` that names it.
  - `mask_zero` has no counterpart, because this crate propagates no mask. The flag would change nothing, so it is omitted rather than accepted and ignored.
- **New `PReLU` layer, a ReLU whose negative-side slope is trainable.** `PReLU::new(input_shape, alpha)` holds 1 slope per position of `input_shape` with the batch axis removed, all starting at `alpha`. The slopes are ordinary trainable parameters, so `parameters()`, `get_weights()`, `set_weights(alpha)`, and `param_count()` all cover them. Decoupled weight decay skips them, because a slope of 0 turns the layer back into `ReLU` and would erase what it learns. Verified against Keras 3.15, forward and backward.
  - **`with_shared_axes(axes)` cuts the slope count, and it is what makes the layer usable after a convolution.** Each named axis drops to extent 1 in the slope array and broadcasts back over the input. `[batch, height, width, channels]` with `[1, 2]` holds 1 slope per channel instead of 1 per pixel. A shared axis is also no longer checked at forward time, so 1 such layer serves images of several sizes.
  - **The derivative at exactly 0 is 0, and neither 1 nor `alpha`.** That is what makes an `alpha` of 0 give the exact `ReLU` gradient as well as the exact `ReLU` transform. `LeakyReLU` differs here, so a `PReLU` with a frozen uniform slope matches it everywhere except at exactly 0.
  - There is no `Activation::PReLU` variant, and there cannot be one. An `Activation` value carries no state, and this layer carries a trainable array. Place `PReLU` after the layer whose output it activates.
- **3 new upsampling layers: `UpSampling1D`, `UpSampling2D`, and `UpSampling3D`.** Each one multiplies the extent of every spatial axis by its factor, leaves the batch axis and the channel axis untouched, and holds no parameter. The family is the decoder counterpart of the pooling family, so an autoencoder and a segmentation decoder become expressible without a transposed convolution. New `Factor2D` and `Factor3D` argument types take an integer for a shared factor or a tuple for 1 factor per axis. All 3 constructors return a `Result` and reject a factor of 0 with `Error::InvalidParameter`.
  - **`UpSampling2D` takes a new `Interpolation` argument with 5 modes: `Nearest`, `Bilinear`, `Bicubic`, `Lanczos3`, and `Lanczos5`.** `Nearest` is the default, and it repeats each pixel into a block. The other 4 resample with a separable kernel. The weights of 1 output pixel always sum to 1, including at an edge, so a constant image stays constant. The weight table is computed in `f64` and rounded to `f32` once, because the Lanczos kernels have large lobes that cancel. `UpSampling1D` and `UpSampling3D` repeat and take no interpolation.
  - **New `tuning::upsampling::set_parallel_min_ops` gate, defaulting to 2,000,000.** It gates 1 axis pass on `destination elements * taps` rather than on the element count. The taps run from 1 for the repeat mode up to 11 for `Lanczos5`. Moving the gate never changes a result.
- **6 new border layers: `ZeroPadding1D/2D/3D` and `Cropping1D/2D/3D`.** A zero-padding layer adds zero positions at the ends of the spatial axes, and a cropping layer removes them. Both leave the batch axis and the channel axis untouched, and neither holds a parameter. Each half is the backward pass of the other half, so a pad and a crop with the same amounts cancel. A `Cropping*` amount that leaves a spatial axis with no positions is `Error::InvalidInput` at forward time.
  - New `Border1D`, `Border2D`, and `Border3D` argument types, taken as `impl Into<..>` and returned with no `Result`. An integer sets every end. A tuple sets 1 equal amount per axis. A tuple of `(before, after)` pairs names every end on its own.
  - Read the 2D and 3D tuple form with care. `ZeroPadding2D::new((1, 2))` gives 1 row at the top and the bottom, plus 2 columns at the left and the right. It does not give 1 row at the top and 2 at the bottom.
- **New `UnitNormalization` layer, which scales each group of elements to an L2 norm of 1.** The new `UnitNormalizationAxis` picks the axes the norm reduces over. `Default` names the last axis, `Custom(a)` names 1 axis, and `Multiple(axes)` names a joint norm. It holds no parameter and behaves the same in training and in inference, which makes it the first layer in `regularization` that is not mode-dependent. `UnitNormalization::new` returns `Error::InvalidParameter` for an empty axis list or a duplicate axis, at construction.
  - The scale is capped at `1e12`, so an all-zero group stays all zero instead of dividing by zero. The backward pass returns the derivative of the cap at such a group, which is a real number. The derivative of the reciprocal square root itself overflows there and would give `NaN`. The 2 approaches agree everywhere else.
- **4 new shape layers: `Reshape`, `Permute`, `RepeatVector`, and `Identity`.** None of them holds a parameter, and each constructor names only the axes after the batch axis, so 1 instance serves every batch size.
  - `Reshape::new(target_shape)` accepts at most 1 `-1` entry, which takes the extent that makes the element count match. `vec![-1]` is exactly `Flatten`, but the constructor takes no input shape. A `0` entry is rejected at construction, not at the first forward pass.
  - `Permute::new(dims)` names the new order of the axes, counting from 1 and never including the batch axis. A permute moves data rather than relabeling it, so the layer copies, and the output is always in C order. An empty `dims` is `Error::InvalidParameter`.
  - `RepeatVector::new(n)` turns a `[batch, features]` input into `[batch, n, features]`, with the same vector at every step, and sums over the step axis on the way back. This bridges 1 recurrent layer to the next, since a recurrent layer here returns a rank-2 state and needs a rank-3 input.
  - `Identity::new()` passes its input through unchanged at every rank, and returns the gradient it receives. Its use is a placeholder, for a builder that must return a layer when the choice is "no operation".
- **New `LayerWeight` variants for every new trainable layer:** `Embedding`, `Conv1DTranspose`, `Conv2DTranspose`, `Conv3DTranspose`, `PReLU`, `DepthwiseConv1D`, and `SeparableConv1D`, each with its weight container. The variants are appended at the end of the enum. postcard writes the variant index, so every existing index stays put, `MODEL_FORMAT_VERSION` stays at 1, and a model file written before this release still loads.

### Fixed
- **`DepthwiseConv2D::output_shape` reported the input channel count instead of `channels * depth_multiplier`.** It read the trailing axis of the shared 2D output-shape calculator, which is right for a plain convolution but not for a depthwise one. `model.summary()` therefore printed the wrong output width above the default multiplier of 1, while the forward pass emitted the correct one. Only the printed description was wrong, so no stored weight, forward value, or gradient changes.
- **`SeparableConv2D` returned a panic, not an error, for an input carrying more channels than it was built for.** Its depthwise kernel is sized from the declared channel count, so the shared kernel indexed past the end of it. Both separable layers now check the rank and the channel count at the layer boundary, the way `DepthwiseConv2D` already did. An input with the declared channel count is unaffected.
- **`Activation::Softmax` no longer fails on an input that is not in C order.** It called `to_owned` and then `into_shape_with_order`, and `to_owned` keeps the strides, so the reshape refused a hand-transposed tensor such as `x.t().to_owned()`. The layer now settles the layout in the 1 copy it already made. Results for an input already in C order are unchanged.

### Changed
- **Breaking: `Activation` no longer derives `Eq`,** because `LeakyReLU` and `ELU` carry an `f32` parameter. It still derives `PartialEq`. Only code that uses `Activation` as a `HashMap` key or in a `HashSet` needs an edit.
- **`Flatten` caches only the input shape, not the input tensor.** Its backward pass never read the values, since a flatten moves no data, so every forward pass copied the whole activation for nothing. `Flatten::forward` now makes 1 pass over the activation instead of 2. The public API and every result are unchanged.
- **All 4 depthwise and separable layers share 1 forward and 1 backward driver loop** in `conv_op_helpers`, instead of each carrying its own copy. The parallel gate, the task split, and the batch-order reduction are unchanged, so every value is the same as before.

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
