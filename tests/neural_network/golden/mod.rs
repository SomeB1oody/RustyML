//! The golden-fixture net: a bit-exact record of what every layer does today.
//!
//! # Why this net exists
//!
//! The layer abstraction is about to change in stages. The parameter store, the constructors,
//! the forward and backward signatures, and the model topology all move. The tests that exist
//! today get rewritten by the same people who do the rewrite. A rewritten test can therefore
//! encode the same mistake as the rewritten code, and still pass.
//!
//! This net is the 1 artifact captured before the change that cannot do that. It records what
//! the code produces today, exactly. Every later stage must reproduce the recorded numbers bit
//! for bit.
//!
//! # The contract
//!
//! **Keyed by layer type, not by constructor.** A fixture is identified by the string that
//! [`Layer::layer_type`] returns, plus a short configuration label. A data file holds no
//! constructor signature and no argument list. When a later stage changes a constructor, only
//! the small Rust builder function changes. The data file stays untouched. This property is
//! what makes the net survive the migration.
//!
//! **Weights are always set, never sampled.** Every fixture builds its layer and then calls
//! `set_weights` with values from a pure formula. No fixture uses a random state, a global
//! seed, or a random number generator for its weights. A later stage that changes an
//! initializer therefore cannot disturb the net.
//!
//! **Inputs come from a pure formula.** See [`golden_input`]. The formula gives both signs and
//! a spread of magnitudes, so a sign-branching activation and a max-selecting pool both
//! exercise each branch.
//!
//! **Comparison is bit-identical.** The harness compares [`f32::to_bits`] values, never an
//! epsilon. The purpose is to catch any behavior change, including one too small for a
//! tolerance to see. A mismatch reports the layer type, the case label, the tensor name, the
//! flat index, both bit patterns, and both values in decimal.
//!
//! **What each case records.** The input shape and data, the forward output, the input
//! gradient from `backward`, every parameter gradient, the parameter value that 1 optimizer
//! step leaves behind, and the output of a separate inference pass. Each case also records the
//! metadata that the layer reports after its forward pass: the `output_shape` string, both
//! counts of `param_count`, the name of every parameter, and the shape and the value
//! fingerprint of every weight that `LayerBase::weights` exposes, 1 time before the optimizer
//! step and 1 time after it. For a layer that does not depend on the training mode, the
//! harness also asserts that the inference pass returns exactly the forward output.
//!
//! The recorded tensor of the inference pass keeps the name `predict`, because that name is a
//! value in every data file. The call behind it is a forward pass with an inference context
//! now, and no `predict` method is left.
//!
//! **An inference pass parks nothing.** The harness asserts that the inference context holds
//! 0 caches after that pass. A layer that parks a cache in this mode leaks 1 value per pass,
//! and no backward pass ever takes it back. That invariant arrived with the context, and it is
//! an assertion and not a recorded value.
//!
//! **A parameter gradient carries its decay class.** `LayerBase::parameters_mut` gives each
//! tensor a `decays` flag, and every optimizer with a non-zero weight decay reads it. A change
//! from `ParamRef::no_decay` to `ParamRef::weight` therefore changes what training does to that
//! tensor, and no gradient value moves. The record holds the flag next to the gradient, and
//! compares it as strictly as a value.
//!
//! # 1 optimizer step binds each gradient to the parameter that it updates
//!
//! A recorded gradient alone does not say which tensor the gradient updates. The rewrite turned
//! the parameters into a named list, re-keyed the optimizer from a positional cursor to that
//! name, and then deleted the weight enum for a checkpoint addressed by name. Such a rewrite
//! can aim a gradient at the wrong parameter tensor and leave every recorded gradient value
//! unchanged.
//!
//! This is the defect class that the `step_param.<name>` tensor exists to catch: a gradient
//! that reaches the wrong parameter tensor, a parameter list that comes back in a new order,
//! and a parameter that the optimizer skips or updates 2 times. Every one of those keeps the
//! gradient values, and every one of those moves a recorded parameter value.
//!
//! The harness applies 1 step of the rule `param[i] -= OPTIMIZER_STEP * grad[i]` to the live
//! `value` slice of each parameter entry, and records the result next to the gradient. See
//! [`OPTIMIZER_STEP`]. The rule lives in this harness, and it comes from no optimizer of
//! `src/neural_network/optimizers`. A borrowed optimizer would make every recorded parameter
//! value change when that optimizer changes, and this net must record the layer behavior alone.
//!
//! The step writes into the layer through the mutable slice that `LayerBase::parameters_mut`
//! hands out.
//! The harness discards that layer instance directly after the step, and every pass builds its
//! own layer, so no other record reads a stepped value.
//!
//! # The second weight fingerprint proves that the step reached the layer
//!
//! The `step_param` tensors alone do not prove that the parameter store is the storage of the
//! layer. An implementation of `parameters` that gives back a copy of each value buffer passes
//! every one of them, because the harness steps that copy and then reads the same copy back.
//! Each recorded number stays where it was. Such a store is a store that an optimizer cannot
//! write through, so the whole training loop leaves the model where it started. Nothing else in
//! this net sees that, and it is the most likely defect of stage 1.
//!
//! Each case therefore takes a second value fingerprint of every weight that
//! `LayerBase::weights` exposes, directly after the optimizer step, and records it as
//! `step_weight.<name>` in a `step_weight` line. `LayerBase::weights` reads the arrays of the
//! layer itself, so a stepped copy no longer agrees with it.
//!
//! The 2 fingerprints of 1 name must differ when the step touched that array, and they must
//! agree when no parameter covers it. BatchNormalization shows both halves in 1 case: `gamma`
//! and `beta` move, and `moving_mean` and `moving_variance` do not.
//!
//! # Where a parameter name comes from
//!
//! The layer, and nothing else. `LayerBase::parameters_mut` gives each tensor a name,
//! `LayerBase::weights`
//! gives each array a name, and the 2 methods use 1 name set: the Keras 3 names `kernel`,
//! `recurrent_kernel`, `depthwise_kernel`, `pointwise_kernel`, `bias`, `embeddings`, `alpha`,
//! `gamma`, `beta`, `moving_mean`, and `moving_variance`. The checkpoint format addresses every
//! array by that name, so the name is a value that the file carries and not a label of this
//! harness.
//!
//! Each parameter of a case therefore has 1 `param_name` line, and the recorded name is the
//! name in the parameter entry. The line still ends with the word `layer`, and that word is now a
//! constant. It was a choice of 2 before: the layer named 1 set of tensors, the weight enum
//! named another through its field names, and the harness worked out which set a recorded name
//! belonged to. The change that deleted the enum merged the 2 sets, so 1 source is left.
//!
//! **The fixture asserts the name, and records none of it.** A fixture declares the name of
//! every parameter through [`GoldenCase::with_parameter_grads`], and the harness asserts that
//! the declared name equals the name in the parameter entry. A rename inside a layer therefore
//! fails the net at that assertion as well as moving the recorded line.
//!
//! **A parameter and the weight of the same name must be 1 storage.** Before the backward pass
//! the harness takes the address of the first element and the element count of every array that
//! `LayerBase::weights` exposes. It then compares each parameter against the array that carries
//! the
//! same name. See [`assert_same_storage`]. That comparison pins 3 things that a name alone
//! does not:
//!
//! 1. **The roster.** Every parameter has a weight of its name, so no parameter is a tensor
//!    that a checkpoint would not hold.
//! 2. **The storage identity of each parameter.** Parameter `kernel` starts at the first
//!    element of weight `kernel`, and holds the same element count. An optimizer that writes
//!    the parameter therefore writes what a saved model holds.
//! 3. **The 2 methods stay 1 list.** `parameters_mut` and `weights` are 2 separate methods of
//!    every
//!    layer, and nothing in the compiler binds them. A layer that renames an array in 1 of them
//!    alone fails here.
//!
//! The comparison is an assertion and not a recorded value, so it moves no line of any data
//! file.
//!
//! # What this net cannot see
//!
//! 3 gaps stay open. A green net says nothing about any one of them, and no gap closes the way
//! the gaps of the forced replay closed.
//!
//! **No constructor validation rule is in the net at all.** Every fixture builds a layer that
//! builds successfully, and every case records what that layer then does. The net holds no
//! rejected shape, no rejected kernel size, no rejected stride, and no rejected padding
//! combination. It records no error type and no error text. A stage that drops a validation
//! rule, that moves a rule from the constructor to the forward pass, or that accepts an input
//! that the rule must reject, keeps every recorded value of this net.
//!
//! Stage 0 shows the size of that gap. Stage 0 changed exactly such a rule: the rule that an
//! input must not be shorter than the effective kernel now applies under `Valid` alone, because
//! `Same` and `Causal` pad the geometry until it is legal, and the check therefore moved into
//! the forward pass. That change is invisible here. It rests on the rejection tests that pin
//! the accepted and the rejected geometry against Keras, in `conv_1d_2d.rs` and
//! `conv_3d_variants.rs`. Keep those tests. This net is no substitute for them.
//!
//! **Every blocked reduction is out of reach.** `DET_REDUCE_BLOCK` is 16384. The column folds
//! of `normalization::folds` take a block of `rows_per_block(c) * c` elements, which is never
//! below 16384, and the block fold of the global pools takes the same constant. A recorded
//! tensor holds 256 elements at most, so `par_col_sum`, `par_col_dot`, `merge_col_parts`, and
//! that pool fold always build exactly 1 block. A change that keeps the first partial of
//! `merge_col_parts` and drops every other one therefore passes this whole net. A task-size cap
//! cannot reach this. Each such task holds a partial of a floating-point sum, so a cap changes
//! the order of the additions, and the recorded bits then move. The gap needs a separate test
//! with an input larger than the block, and that test compares against a reference sum, not
//! against this net.
//!
//! **The net sees 1 layer at a time.** It builds 1 layer, calls the methods of the `Layer`
//! trait on it, and records what comes back. It builds no `Sequential`, calls no
//! `Optimizer::update`, computes no loss, and runs no `fit`. Every rule that holds a model
//! together is therefore out of scope: the order that layers run in, the way an optimizer walks
//! the parameters of a whole model, the pairing of a loss gradient with the last layer, and the
//! training loop itself. Stage 4 moves exactly that code, so stage 4 needs a guard of its own.
//!
//! # A recorded weight carries a value fingerprint
//!
//! 4 normalization layers expose 2 or more arrays of the same shape through
//! `LayerBase::weights`,
//! and BatchNormalization exposes 4. A recorded shape alone therefore lets a later stage
//! exchange 2 of those arrays with no failure. `LayerBase::weights` is the exact surface that the
//! checkpoint format reads, so such an exchange corrupts every saved model.
//!
//! Each recorded weight therefore carries a checksum of its values next to its shape. See
//! [`weight_checksum`] for the mixing function, which this harness owns. The harness takes the
//! fingerprint after the forward pass and before the optimizer step, so the 2 new fields stay
//! independent of each other.
//!
//! **`output_shape` is a display string today.** A later stage is expected to give it a typed
//! return value. The recorded field then changes with that stage. Such a change is expected,
//! and it is not a regression.
//!
//! **Size.** Every recorded tensor holds at most 256 elements. Batches stay at 2 and spatial
//! extents stay small, so the net stays readable in review.
//!
//! **Both modes for a mode-dependent layer.** Dropout, the SpatialDropout family,
//! GaussianNoise, GaussianDropout, and BatchNormalization each record 2 separate cases. The
//! training case pins the seed through the layer builder. The inference case records the
//! inference behavior.
//!
//! # Every case runs 2 times: gated, and then forced parallel
//!
//! About half of the layers hold a serial kernel and a rayon kernel behind a tunable gate. The
//! crate documents that the 2 kernels give the same result, bit for bit (see
//! `rustyml::tuning`). Every recorded tensor stays at 256 elements or fewer, and every gate
//! sits far above that, so a single pass reaches the serial kernel alone.
//!
//! [`run_family`] therefore replays the whole family a second time, with every gate that a
//! layer reads held at 0. Each gated kernel then takes its parallel branch. The second pass
//! compares against the same data file, records nothing, and adds no case.
//!
//! A difference between the 2 passes is a defect in the layer, and never a stale fixture. Do
//! not regenerate the data file for such a difference. The report names it a
//! serial-versus-parallel disagreement, and lists the gates it forced.
//!
//! The gates are process-global, and the test harness runs the tests of 1 binary at the same
//! time, so the 2 passes take the lock that `common::GateGuard` and `common::read_gates` share.
//! A forced pass holds the exclusive side, and a gated pass holds the shared side. No family
//! therefore records under gate values that another test installed, and no test that depends on
//! a gate value runs inside the forced window. The guard restores every gate when it drops,
//! which includes the path where a case panics.
//!
//! # The forced replay also splits the work into more than 1 task
//!
//! A gate alone is not enough. A gate selects the parallel branch, and the parallel branch then
//! asks a second, separate rule how large 1 task is. Each such rule carries a floor: 256 output
//! positions for a pooling forward pass, 16 channels for a pooling backward pass, 64 output
//! positions for a convolution forward pass, and a 16384-element budget for a resize pass and
//! for an embedding gather. Every fixture tensor holds 256 elements or fewer, so each rule gave
//! back the whole input, and the parallel branch built exactly 1 task.
//!
//! The arithmetic inside a parallel closure was therefore covered, and the arithmetic around it
//! was not: the chunk boundaries, the multi-index that a task decodes from its first flat
//! position, and the copy that puts each task result back in its own place. The coming stages
//! re-derive exactly that chunking.
//!
//! The forced replay therefore also installs a task-size cap of [`FORCED_SPLIT_CHUNK`] in every
//! driver of `common::NEURAL_NETWORK_SPLIT_CAPS`. Each capped driver then builds 1 task per 2
//! units of its own axis, so a fixture with 3 or more units gets several full tasks and a
//! partial last one. See `rustyml::bench_internals`, which owns the caps, documents what each
//! one counts, and lists the drivers that must never take one, because their tasks each hold a
//! partial of a floating-point reduction.
//!
//! A cap moves no value either. Each capped driver runs the same serial kernel over each task,
//! and joins the task results in task order. A difference under the cap is therefore the same
//! class of defect as a difference under a gate: a defect in the layer, and never a stale
//! fixture. A cap is process-global like a gate, and the same `common::GateGuard` saves it,
//! installs it, and restores it on the panic path.
//!
//! 1 driver takes no cap, because it does not hold that property today. See
//! [`UNCAPPED_DRIVER`], which names the driver, the defect, and the input that shows the defect
//! with no cap at all.
//!
//! # Regeneration is deliberate, and a moved value needs 2 acts
//!
//! A plain `cargo test` compares and fails. It never rewrites a data file. Regeneration needs
//! the environment variable [`REGEN_VARIABLE`], and the value of that variable is the reason
//! for the regeneration:
//!
//! ```text
//! RUSTYML_REGEN_GOLDEN="add the dilation cases of stage 0" cargo test --test neural_network golden
//! ```
//!
//! A harness that heals its own baseline is worthless, because the baseline then tracks the
//! bug instead of the behavior. To regenerate a data file is an explicit claim that the
//! behavior change was intended and reviewed. The guard below makes that claim visible in the
//! file. It also makes a claim over a value that the file already holds cost a second command.
//!
//! **The reason is a reason, and not a switch.** A value of [`MIN_REASON_LENGTH`] printable
//! ASCII characters or more is a reason. The value "1" is refused, and the refusal is a panic.
//! The harness writes the reason into the header of the data file as a `# reason` line, newest
//! first, up to [`MAX_RECORDED_REASONS`] entries. Each data file therefore carries the history
//! of its own changes, next to the values that those changes wrote.
//!
//! **The first capture of a family needs the reason alone.** The harness parses the file on
//! disk and works out exactly what the regeneration would do. A family whose data file does not
//! exist yet has no earlier value that a write can hide, so that write stays a 1-step
//! operation. [`check_family_roster`] refuses every other reason for an absent file, so this is
//! the 1 situation that reaches the path. A run that finds nothing to do writes nothing.
//!
//! **Every case of a data file that exists needs a second act.** An added case, a changed case,
//! and a removed case each need the acknowledgment token. See [`RegenerationPlan::affected`].
//! A case that the file lacks has 2 possible histories. No file ever held it, or it left the
//! file after the last write. No code of this harness can tell the 2 apart. The second history
//! turns a value that moves into a value that arrives, and that is the signature of the whole
//! laundering class.
//!
//! Such a run is refused 1 time. The refusal names every case that would change, the number of
//! differences in each one, and the first [`MAX_REPORTED_DIFFERENCES`] differences of each one.
//! It ends with 1 acknowledgment token. A second run writes the file, and only when
//! [`ACCEPT_VARIABLE`] carries that token:
//!
//! ```text
//! RUSTYML_REGEN_GOLDEN="the arg-max of an all-negative-infinity window now stays in its window" \
//!   RUSTYML_REGEN_GOLDEN_ACCEPT=spatial-1-3f6c1d0a2b4e5f78 \
//!   cargo test --test neural_network golden
//! ```
//!
//! The variable takes a list, because 1 run covers 5 families. A comma or any whitespace
//! separates 2 tokens.
//!
//! **The token covers the change, and not the intention to change.** See
//! [`acknowledgment_token`]. The token holds the family name, the number of affected cases, and
//! a fingerprint of the whole change set. The fingerprint reads every difference, and a
//! difference carries the new bits of the value that moves. It reads the values of every added
//! case as well. A second run that produces another change set therefore gives another token,
//! and the refusal repeats. An acknowledgment cannot carry over to a change that nobody read.
//!
//! **A byte layout that this harness did not write is repaired, and reported.** Every
//! comparison of this net reads a case by its key. A file that holds every recorded case with
//! every recorded value therefore agrees with the record, whatever order those cases stand in,
//! and whatever header stands above them. The regeneration compares the file against the text that
//! [`render`] and [`seal`] produce for the same record and the same reason history. See
//! [`canonical_text`]. A file that differs is rewritten in that layout, and the run then stops
//! with the report of the repair. The rewrite moves no recorded value, and the report exists
//! because a repair that nobody sees is a silent write.
//!
//! The 5 data files that ship today carry the header of an older version of [`render`]. The
//! first regeneration of each family therefore reports 1 such repair. The diff of that repair
//! holds header comment lines and the `content_digest` line alone.
//!
//! **The forced replay runs before the write.** A serial-versus-parallel disagreement fails the
//! run, and the data file keeps the values it had.
//!
//! # Cargo can hand a stale binary to a regeneration
//!
//! An audit of this net found the hazard by accident. The mutation driver of that audit put a
//! defect in a source file, ran the net, and then restored the file with a copy that kept the
//! mtime of the original. The fingerprint that cargo keeps reads the mtime, so cargo saw no
//! change, rebuilt nothing, and ran the binary that still held the defect. The regeneration of
//! that run wrote the defect into `misc.golden`, and the run reported 5 passed and 0 failed.
//! The earlier harness compared the forced replay against the fresh gated record alone, and
//! never against the file, so a regeneration could not fail. A silently wrong baseline is the
//! worst outcome this net can produce, because every later stage then reproduces the defect on
//! purpose.
//!
//! Take these steps for every regeneration:
//!
//! 1. Make sure that the working tree holds the sources that the new behavior needs.
//! 2. Force a rebuild of the test binary. Touch every source file that changed, or run
//!    `cargo clean -p rustyml`. A file that comes back with an mtime that it had before leaves
//!    the cargo fingerprint where it was, whatever the content of the file is.
//! 3. Run the regeneration, read the printed change set case by case, and only then repeat the
//!    run with the acknowledgment token.
//!
//! The guard cannot prove that a binary is fresh, and no code in a test binary can. It makes a
//! wrong regeneration loud instead of silent: a regeneration over a value that the file already
//! holds stops, prints what would move, and waits for a second command that names that exact
//! change.
//!
//! **The parts of the guard, and why each one is there.** A diff report alone informs an
//! operator who reads it, and warns nobody who does not. A count limit alone lets a narrow
//! wrong change through, and the audit defect was narrow. A reason string alone records what
//! the operator believed, and stops nothing. The self digest and the roster below keep the
//! other 3 honest, because a case that the file no longer holds is a case that no token can
//! protect. The 5 together give 1 property that none of them gives alone: no run of this net
//! writes a case into a data file that exists without a command that carries the fingerprint of
//! that exact case, and the file keeps the stated reason for it. The property covers the
//! commands that this harness reads. It does not cover a program that writes a data file
//! directly. See the section below.
//!
//! # A data file carries a digest of itself
//!
//! The header of every data file holds 1 `content_digest` line. That line records the number of
//! cases, the number of recorded values, and a digest over every other byte of the file. See
//! [`self_digest`]. The harness verifies all 3 before it compares, and before it plans a
//! regeneration. A file that fails the check took an edit that this harness did not write, and
//! the harness refuses it. The refusal names each of the 3 fields that disagrees, so a lost
//! case and a moved value read differently.
//!
//! The digest sees the edit that turns a changed case into an added case. An operator who
//! deletes a case from the data file, and who then regenerates, produces a plan that adds that
//! case back. A deleted case leaves a case count and a digest that the header contradicts, and
//! the harness stops before it plans anything. An addition over a file that exists needs the
//! acknowledgment token as well, so the same edit is loud on both sides.
//!
//! The digest covers the comments as well, which includes the reason history. A hand that
//! writes a regeneration the file never had into that provenance, and that leaves the digits as
//! they are, therefore stops every run of the net.
//!
//! **The digest is a fingerprint, and it is no signature. It does not close the data-file
//! route.** The mixing function is the FNV-1a round of [`mix_byte`], the constants stand at the
//! head of this file, and the code that reads the line ships next to the code that writes it.
//! An operator who can run this harness can therefore recompute the digits and reseal any file
//! at all. An audit of this net did exactly that. No secret and no key changes it, because a
//! test binary carries every byte that it reads.
//!
//! The digest exists to make an ACCIDENT loud: a truncated file, a half-applied patch, a merge
//! that dropped a case, a value that an editor changed by hand. It stops the 1-command bypass,
//! and it makes a hand edit a deliberate act that needs a program of its own. It stops no
//! determined rewrite.
//!
//! **Version control is what makes a deliberate rebaseline reviewable.** A resealed forgery is
//! valid to this harness. It still arrives in a diff as thousands of changed lines, next to the
//! source change that claims to explain them. The digest, the roster, and the token make
//! the harness loud. The review of that diff is what makes the new baseline true. Never take a
//! data file from outside version control, and read the diff of every regeneration case by
//! case.
//!
//! **A malformed data file blocks regeneration.** The guard needs the file to work out what
//! would change, so it parses the file first, and a malformed file is a panic. Restore such a
//! file from version control with `git checkout --`. Do not repair it by hand, and do not delete
//! it. Both are refused.
//!
//! # An absent data file needs the roster
//!
//! A data file that a deletion removed and a data file that never existed look the same on disk.
//! No code of this harness can tell the 2 apart from the disk alone. The harness therefore keeps
//! the roster [`ESTABLISHED_FAMILIES`], which names every family that owns a data file today.
//!
//! An absent data file of a family in that roster is a deletion, and the harness refuses it. An
//! absent data file of a family that the roster does not name is the first capture of a new
//! family, and the regeneration writes it. The safe default is the deletion, because a family
//! reaches the bootstrap path only when a source edit puts it there.
//!
//! The roster stays true, because the harness also refuses a data file that exists for a family
//! that the roster does not name. A first capture therefore writes the file, and the next run
//! demands the roster entry. [`guard_tests`] holds the same 2 checks over the whole data
//! directory, so a deleted data file also fails a run that no family test takes part in.
//!
//! An operator who deletes a data file and its roster entry in 1 change still reaches the
//! bootstrap path. That edit is in a source file, next to the deletion, and a reviewer reads
//! both in the same diff. A deletion is no longer a command that leaves nothing behind.
//!
//! **How to bootstrap a family.** Write the fixture file, register the family, and leave the
//! name out of [`ESTABLISHED_FAMILIES`]. Run the regeneration 1 time with a reason. Read the new
//! data file, case by case. Then add the name to the roster, and run the suite again.
//!
//! # The 1 recorded value that moved after the first capture
//!
//! The first capture of this net is not the file that ships. Exactly 1 recorded value moved
//! between the 2, in the window where the max-pool arg-max fix went in. The move is a
//! consequence of that fix, and the record itself shows it.
//!
//! The case is `MaxPooling1D handbuilt_all_negative_infinity`, and the tensor is `grad_input`.
//! The input is `[1.0, 2.0, -inf, -inf]`, the window is 2 wide, the stride is 2, and the
//! upstream gradient is `[-1.75, -0.3]`. Window 0 covers the positions 0 and 1, and the value
//! 2.0 at position 1 wins it. Window 1 covers the positions 2 and 3, and every element of that
//! window is negative infinity.
//!
//! The first capture recorded `[-0.3, -1.75, 0.0, 0.0]`. The file that ships records
//! `[0.0, -1.75, -0.3, 0.0]`. The gradient of window 1 therefore moved from position 0 to
//! position 2, and the gradient of window 0 stayed at position 1. A text diff shows the 2 pairs
//! as an exchange of neighbors, and the exchange is what a value that moves 2 places looks like
//! in a file of 1 value per line.
//!
//! The pooling engine seeded its arg-max buffer with `vec![0usize; ...]`, and it folds each
//! element with a greater-than test against a start value of negative infinity. The test
//! `-inf > -inf` is false, so a window of negative infinity alone assigned nothing, kept the
//! seed 0, and sent its gradient to the first element of the whole batch item. That element
//! belongs to another window. The fix seeds the arg-max of each window from the first in-bounds
//! element of that same window, and the first element of window 1 is position 2. The recorded
//! move is exactly the difference between the 2 seeds.
//!
//! No other value of any family moved in that window. The 5 other
//! `handbuilt_all_negative_infinity` cases, on the pooling layers of higher rank and on the
//! global pools, came in with the same change as new cases, and a new case moves nothing. No
//! other fixture builds a window in which every element loses to the start value, so no other
//! recorded value is in reach of that fix. A comparison of the whole first capture against the
//! file that ships confirms both statements, case by case and value by value.
//!
//! # The `output_shape` field of 39 cases disagrees with the file today
//!
//! The stage that moved the pass state into a context also made `Layer::output_shape` a pure
//! function of the build. The method runs `Layer::compute_output_shape_many` against
//! `LayerBase::known_input_shapes`, and that second method reports the shape that
//! `UnaryLayer::build` recorded. It no longer reports a shape that a forward pass wrote, and it
//! no longer formats a batch axis of its own.
//!
//! 39 cases therefore record an `output_shape` string that the layer no longer produces. 35 of
//! them hold a free batch axis that the layer now fixes: a fixture that leaves the build to
//! `UnaryLayer::forward_mut` builds for the whole shape of the tensor, batch extent included,
//! so `ZeroPadding1D symmetric_2` reports `"(2, 8, 3)"` where the file records `"(None, 8, 3)"`.
//! The 4 that are left hold a build shape that records less than the input: `Dense` records the
//! last axis alone, so a rank-3 case reports `"(None, 4)"`, and `Embedding` frees every axis, so
//! its cases report `"(None, None, 3)"`.
//!
//! No other recorded value moved. Every tensor, every weight fingerprint, every `step_weight`
//! fingerprint, every parameter name, and every `param_count` of all 5 families still agrees,
//! bit for bit, under the gated replay and under the forced replay alike. The harness leaves
//! the data files untouched and reports the disagreement, because a display string that moves
//! is a claim for a reviewer to read and not a value for a test to correct.
//!
//! # The data file grammar
//!
//! A data file is text. A blank line carries nothing, and so does a line whose first non-blank
//! character is "#". The regeneration path is the 1 reader that looks inside a comment: it
//! carries the `# reason` lines of the header over into the file that it writes. See
//! [`previous_reasons`]. Every other line starts with a keyword:
//!
//! ```text
//! format 6                     the format version, 1 time, at the head of the file
//! family <name>                the family name, 1 time, at the head of the file
//! content_digest <digits> cases <count> values <count>
//!                              the digest of the file over itself, 1 time, before the cases
//! case <layer-type> <label>    starts a case
//! mode training|inference      the mode that the case ran in
//! output_shape "<text>"        what output_shape returned after the forward pass
//! param_count trainable <count> non_trainable <count>
//!                              what param_count returned, both halves
//! param_name <name> layer      the name of 1 parameter, and the source of the name
//! weight <name> <extent>... checksum <digits>
//!                              the shape and the values of 1 exposed weight. See below
//! step_weight <name> <extent>... checksum <digits>
//!                              the same weight, read back after the optimizer step
//! tensor <name> <extent>...    starts a tensor. The value lines follow it
//! end                          ends a case
//! ```
//!
//! A case holds 1 `mode` line, 1 `output_shape` line, and 1 `param_count` line. It then holds 1
//! `param_name` line per parameter, 1 `weight` line and 1 `step_weight` line per exposed
//! weight, and 1 `tensor` block per recorded tensor. Each group keeps the order that the layer
//! gives it, and the `step_weight` group repeats the names of the `weight` group.
//!
//! The `content_digest` line takes no leading blank. It holds [`CHECKSUM_DIGITS`] lowercase
//! hexadecimal digits, then the word `cases` and the number of cases, then the word `values` and
//! the number of value lines. The digits are the digest of every byte of the file, with the
//! digits themselves read as zeros. See [`self_digest`].
//!
//! The `param_count` line holds both fields of `ParamCounts`, always, and either count may be
//! 0. A layer that holds no parameter at all records `trainable 0 non_trainable 0`.
//!
//! The last word of a `param_name` line is `layer`, always. The layer supplies every name. See
//! the section above.
//!
//! A `weight` line and a `step_weight` line each end with the word `checksum` and 16 lowercase
//! hexadecimal digits. The digits are the value fingerprint of the array, from
//! [`weight_checksum`]. The whole-number words between the name and the word `checksum` are the
//! shape.
//!
//! A `tensor` line for a parameter gradient carries 1 more word after its shape, `decays` or
//! `no_decay`. That word is the decay class of the parameter. No other tensor line
//! carries the word.
//!
//! A case records 2 tensors per parameter. The name `grad_param.<name>` holds the gradient, and
//! the name `step_param.<name>` holds the parameter value after 1 optimizer step. Both tensors
//! hold the flat slice that `LayerBase::parameters_mut` gives, so the shape is the element count
//! alone.
//!
//! A value line holds the 8 lowercase hexadecimal digits of `f32::to_bits`. The text after the
//! "#" on such a line is the same value in decimal, and it is a comment.
//!
//! The `output_shape` text is in double quotes, because the string holds spaces. A backslash, a
//! double quote, and a control character are written with a backslash.
//!
//! # How to add a family
//!
//! Each family owns 1 Rust file in this directory and 1 data file in `golden/data`. A family
//! agent writes 1 function per layer type, registers those functions in a list in the same
//! file, and adds 1 test that calls [`run_family`]. No family touches this file, and no family
//! touches another family file. See `misc.rs` for the reference family.
//!
//! The data file of a new family comes from 1 regeneration, and the name of that family then
//! goes into [`ESTABLISHED_FAMILIES`] of this file. That entry is the 1 step that a new family
//! adds here. See the roster section above for the order of the 2 steps.

#![allow(dead_code)]

use crate::common::{GateGuard, NEURAL_NETWORK_GATES, NEURAL_NETWORK_SPLIT_CAPS, read_gates};
use ndarray::{ArrayBase, ArrayD, ArrayViewD, Data, Dimension, IxDyn};
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::traits::{Layer, ParamId};
use rustyml::neural_network::{Ctx, Tensor};
use std::fmt::Write as _;
use std::path::PathBuf;

/// Fixtures for the convolution layers.
mod conv;
/// Fixtures for the reference family: Dense, the shape layers, and the activation layers.
mod misc;
/// Fixtures for the recurrent and embedding layers.
mod sequence;
/// Fixtures for the pooling, border, and resampling layers.
mod spatial;
/// Fixtures for the dropout, noise, and normalization layers.
mod stochastic;

/// The environment variable that turns regeneration on.
///
/// The value is the reason for the regeneration, and the data file records it. A value that is
/// no reason, such as "1", is refused. See [`regeneration_request`].
const REGEN_VARIABLE: &str = "RUSTYML_REGEN_GOLDEN";

/// The environment variable that acknowledges 1 exact change to a data file.
///
/// The value is the list of tokens that a refused regeneration printed. A comma or any
/// whitespace separates 2 tokens. See [`acknowledgment_token`].
const ACCEPT_VARIABLE: &str = "RUSTYML_REGEN_GOLDEN_ACCEPT";

/// The smallest number of characters that a regeneration reason may hold.
///
/// The bound keeps a bare switch, such as "1", "y", or "true", out of the reason field. A
/// reason of this length says something that a reviewer can read 1 year later.
const MIN_REASON_LENGTH: usize = 12;

/// The largest number of characters that a regeneration reason may hold. A reason is 1 comment
/// line of the data file, so it stays short.
const MAX_REASON_LENGTH: usize = 200;

/// The head of the comment line that carries 1 regeneration reason in a data file.
const REASON_LINE_PREFIX: &str = "# reason ";

/// The number of regeneration reasons that the header of a data file keeps, newest first.
const MAX_RECORDED_REASONS: usize = 8;

/// The largest number of removed cases, and of changed cases, that 1 refusal report prints.
const MAX_REPORTED_CHANGES: usize = 10;

/// The largest number of single differences that 1 refusal report prints per changed case.
const MAX_REPORTED_DIFFERENCES: usize = 3;

/// The version of the data-file format that this harness reads and writes.
///
/// Version 6 put both halves of `ParamCounts` in the `param_count` line, and it narrowed the
/// last word of a `param_name` line to `layer`. Version 5 added the `content_digest` header
/// line. Version 4 added the `param_name` line and the `step_weight` line. No version moved a
/// recorded value of the version before it.
const FORMAT_VERSION: u32 = 6;

/// The data-file format version that this harness also reads, and never writes.
///
/// A format bump makes every data file of the version before it unreadable, and an unreadable
/// file blocks the regeneration guard: the guard parses the file to work out what a
/// regeneration would change, and the acknowledgment token covers exactly that change set. With
/// no reader for the earlier version, the only way to bump the format is to delete the data
/// files, and a deleted data file is the 1 act this guard refuses outright.
///
/// This harness therefore reads 2 versions and writes 1. The parser decodes an earlier line
/// exactly as the earlier version wrote it, and invents nothing: a version 5 `param_count` line
/// holds 1 count, and it compares against the 2 counts of a version 6 record as the different
/// text that it is. Every case of such a file is therefore a changed case, and it needs the
/// token like any other.
///
/// The change that bumps [`FORMAT_VERSION`] again replaces the decode below with the decode of
/// its own predecessor.
const EARLIER_FORMAT_VERSION: u32 = FORMAT_VERSION - 1;

/// The largest number of elements that 1 recorded tensor may hold.
const MAX_TENSOR_ELEMENTS: usize = 256;

/// The largest number of problems that 1 failure report prints.
const MAX_REPORTED_PROBLEMS: usize = 20;

/// The name prefix of a parameter-gradient tensor. Such a tensor, and no other, carries a
/// decay class.
const PARAMETER_PREFIX: &str = "grad_param.";

/// The name prefix of a stepped-parameter tensor.
///
/// Such a tensor holds the parameter value that 1 step of the harness optimizer rule left
/// behind. It carries no decay class, because the matching gradient tensor already holds one.
const STEPPED_PREFIX: &str = "step_param.";

/// The last word of every `param_name` line, which says that the layer supplied the name.
///
/// The word is a constant now, and it was a choice of 2 before. The layer named 1 set of
/// tensors and the weight enum named another, and the harness worked out which one a recorded
/// name came from. The named checkpoint merged the 2 sets, so a name has 1 source and the word
/// records it.
const LAYER_NAME_SOURCE: &str = "layer";

// ---------------------------------------------------------------------------------------
// The 1 optimizer step
// ---------------------------------------------------------------------------------------

/// The learning rate of the 1 optimizer step that the harness applies to every parameter.
///
/// The whole rule is `param[i] -= OPTIMIZER_STEP * grad[i]`, and this harness owns it. No
/// optimizer of `src/neural_network/optimizers` takes part. A borrowed optimizer would make
/// every recorded parameter value change when that optimizer changes, and this net must record
/// the layer behavior alone.
///
/// The value is a power of 2, so the product is exact and the subtraction is the only step that
/// rounds. The magnitude keeps the parameter and the gradient at a comparable size, so both
/// reach the recorded result.
const OPTIMIZER_STEP: f32 = 0.125;

// ---------------------------------------------------------------------------------------
// The weight value fingerprint
// ---------------------------------------------------------------------------------------

/// The start value of [`weight_checksum`]. It is the 64-bit FNV-1a offset basis.
const CHECKSUM_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The multiplier of [`weight_checksum`]. It is the 64-bit FNV-1a prime.
const CHECKSUM_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The number of hexadecimal digits that a weight checksum holds in a data file.
const CHECKSUM_DIGITS: usize = 16;

// ---------------------------------------------------------------------------------------
// The self digest of a data file
// ---------------------------------------------------------------------------------------

/// The keyword of the header line that carries the self digest of a data file.
///
/// The whole line reads `content_digest <digits> cases <count> values <count>`. It stands
/// before the first case of the file. See [`self_digest`].
const DIGEST_KEYWORD: &str = "content_digest";

/// Every family that owns a data file in this repository today.
///
/// A data file that a deletion removed and a data file that never existed look the same on
/// disk. No code of this harness can tell the 2 apart from the disk alone, so this roster
/// carries the difference instead.
///
/// An absent data file of a family in this roster is a deletion, and the harness refuses it. An
/// absent data file of a family that this roster does not name is the first capture of a new
/// family, and the regeneration writes it. A data file that exists for a family that this
/// roster does not name is refused as well, so a first capture cannot leave the roster stale.
///
/// Keep the entries in alphabetical order.
const ESTABLISHED_FAMILIES: [&str; 5] = ["conv", "misc", "sequence", "spatial", "stochastic"];

// ---------------------------------------------------------------------------------------
// The tuning gates of the forced-parallel replay
// ---------------------------------------------------------------------------------------

/// The value that the forced-parallel replay puts in every gate.
///
/// Each gated kernel compares its work estimate against its gate with `>=`, so 0 selects the
/// parallel branch for every input, however small.
const FORCED_PARALLEL_GATE: usize = 0;

/// The value that the forced-parallel replay puts in every task-size cap.
///
/// A capped driver holds each task at this many units of its own axis or fewer. See
/// `common::GateGuard::with_split_cap` for the unit of each driver.
///
/// The value is 2, and not 1. A cap of 1 gives each task exactly 1 unit, which makes the first
/// unit of task `i` the number `i` itself. A defect that drops the task size out of that
/// product then leaves every boundary where it was. A cap of 2 keeps the product visible, and
/// it still splits an axis of 3 units. An axis with an odd number of units also gets a partial
/// last task, which reads the arithmetic that trims the last boundary.
const FORCED_SPLIT_CHUNK: usize = 2;

/// The 1 driver that the forced-parallel replay leaves at its calibrated task size.
///
/// # The convolution forward pass is not invariant to its row block
///
/// A task of this driver is 1 GEMM into a disjoint row block of the output. A row of that
/// product is 1 dot product per filter over the whole `k*Cin` axis, and the row block selects no
/// part of that axis, so the block should decide no value. That is not what the pass does. The
/// matrix-product backend picks its accumulation order from the row count of the block, so the
/// same rows give different result bits in a short block than in a long one.
///
/// This is not a property of the cap. The shipped gate `conv.parallel_min_flops` already selects
/// between 1 GEMM over the whole output plane and a block of `CONV_MIN_CHUNK_POSITIONS` rows,
/// with no cap installed. A `Conv2D` of input `[1, 20, 20, 3]`, kernel 3 by 3, 8 filters, and
/// `Valid` padding gives 23 of its 2592 output values with different bits on the 2 sides of that
/// gate. Every one of the 23 lies in the trailing partial block. The crate states that a gate
/// picks an execution strategy alone, so this is a defect, and it is older than this net.
///
/// The cap stays at 0 here until the pass is invariant. Capping it instead would hold the conv
/// family red for ever, because the data file records the values of the single-task pass. Do not
/// regenerate the data file to close this. Fix the pass, then delete this exception and the
/// `without_split_cap` call that reads it.
const UNCAPPED_DRIVER: &str = "conv.forced_chunk_positions";

// ---------------------------------------------------------------------------------------
// The pure value formulas
// ---------------------------------------------------------------------------------------

/// The input value at flat index `index`, from the formula `((index * 37) % 101 - 50) / 25`.
///
/// The result runs from -2.0 through 1.96 and takes 101 distinct values. It changes sign many
/// times, so a sign-branching activation such as ReLU, LeakyReLU, ELU, SELU, or PReLU takes
/// both branches. Neighboring elements differ a lot, so a max-selecting pool has a real choice
/// to make.
fn input_value(index: usize) -> f32 {
    (((index * 37) % 101) as f32 - 50.0) / 25.0
}

/// The weight value at flat index `index`, from the formula `((index * 53) % 79 - 39) / 40`.
///
/// The result runs from -0.975 through 0.975 and takes 79 distinct values. The spread keeps a
/// product from saturating an exponential activation.
fn weight_value(index: usize) -> f32 {
    (((index * 53) % 79) as f32 - 39.0) / 40.0
}

/// The upstream gradient value at flat index `index`, from `((index * 29) % 71 - 35) / 20`.
///
/// The result runs from -1.75 through 1.75. Both signs appear, so a gradient that flips a sign
/// shows up in the record.
fn gradient_value(index: usize) -> f32 {
    (((index * 29) % 71) as f32 - 35.0) / 20.0
}

/// Builds the fixture input tensor for a shape.
///
/// The harness calls this for every case. A family function does not need it, unless a layer
/// needs a hand-built input such as an index table.
///
/// # Parameters
///
/// - `shape` - Shape of the wanted tensor, batch axis first
///
/// # Returns
///
/// - `Tensor` - A tensor in C order, filled from the documented input formula
///
/// # Panics
///
/// - If `shape` holds more than 256 elements
pub fn golden_input(shape: &[usize]) -> Tensor {
    fill(shape, input_value)
}

/// Builds a deterministic weight tensor, starting the formula at flat index 0.
///
/// # Parameters
///
/// - `shape` - Shape of the wanted weight tensor
///
/// # Returns
///
/// - `ArrayD<f32>` - A tensor in C order, filled from the documented weight formula
///
/// # Panics
///
/// - If `shape` holds more than 256 elements
pub fn golden_weights(shape: &[usize]) -> ArrayD<f32> {
    golden_weights_from(shape, 0)
}

/// Builds a deterministic weight tensor, starting the formula at a chosen flat index.
///
/// A layer with more than 1 parameter tensor uses a different starting index for each one.
/// The tensors then hold different numbers, so a swap of 2 parameters cannot pass unseen.
///
/// # Parameters
///
/// - `shape` - Shape of the wanted weight tensor
/// - `first_index` - Flat index that the weight formula starts from
///
/// # Returns
///
/// - `ArrayD<f32>` - A tensor in C order, filled from the documented weight formula
///
/// # Panics
///
/// - If `shape` holds more than 256 elements
pub fn golden_weights_from(shape: &[usize], first_index: usize) -> ArrayD<f32> {
    fill(shape, |index| weight_value(index + first_index))
}

/// Builds the upstream gradient that the harness hands to `backward`.
fn golden_gradient(shape: &[usize]) -> Tensor {
    fill(shape, gradient_value)
}

/// Fills a tensor of `shape` from a flat-index formula, and checks the size cap.
fn fill(shape: &[usize], value: impl Fn(usize) -> f32) -> ArrayD<f32> {
    let count: usize = shape.iter().product();
    assert!(
        count <= MAX_TENSOR_ELEMENTS,
        "a golden tensor of shape {shape:?} holds {count} elements, and the cap is \
         {MAX_TENSOR_ELEMENTS}"
    );
    let data: Vec<f32> = (0..count).map(&value).collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("the data length is the shape product")
}

// ---------------------------------------------------------------------------------------
// The per-family entry point
// ---------------------------------------------------------------------------------------

/// 1 recorded case: 1 layer configuration, 1 input shape, and 1 training mode.
///
/// Build a case with [`GoldenCase::new`], and then add what the layer needs through the
/// remaining methods. The default case runs in training mode, declares no parameter gradient,
/// and asserts that an inference forward pass gives exactly the training forward output.
pub struct GoldenCase {
    /// Short configuration label. It carries no whitespace, and it is unique per layer type
    label: &'static str,
    /// Shape of the input tensor, batch axis first
    input_shape: Vec<usize>,
    /// Name of every parameter gradient, in the order that `LayerBase::parameters_mut` returns
    /// the tensors that hold one
    parameter_names: Vec<&'static str>,
    /// Mode of the context that the harness gives the forward and the backward pass
    training: bool,
    /// Whether the harness asserts that the inference pass equals the forward pass
    inference_matches_training: bool,
    /// Builds a fresh layer. The harness calls this 2 times, so the 2 passes never share state
    build: Box<dyn Fn() -> Box<dyn Layer>>,
}

impl GoldenCase {
    /// Starts a case for 1 layer configuration.
    ///
    /// # Parameters
    ///
    /// - `label` - Short configuration label without whitespace, such as `"alpha_0p5"`. It
    ///   must stay stable, because it is half of the key that the data file records
    /// - `input_shape` - Shape of the input tensor, batch axis first
    /// - `build` - Builds a fresh layer with its weights already set. The harness calls it 1
    ///   time for the inference pass and 1 more time for the forward and backward pass
    ///
    /// # Returns
    ///
    /// - `Self` - A case in training mode, with no parameter gradient declared
    pub fn new<F>(label: &'static str, input_shape: &[usize], build: F) -> Self
    where
        F: Fn() -> Box<dyn Layer> + 'static,
    {
        Self {
            label,
            input_shape: input_shape.to_vec(),
            parameter_names: Vec::new(),
            training: true,
            inference_matches_training: true,
            build: Box::new(build),
        }
    }

    /// Names every parameter gradient, in the order that `LayerBase::parameters_mut` returns
    /// the tensors that hold one.
    ///
    /// Use the name that the layer itself puts in the parameter entry, such as `"kernel"`,
    /// `"recurrent_kernel"`, `"depthwise_kernel"`, `"bias"`, `"embeddings"`, `"alpha"`,
    /// `"gamma"`, or `"beta"`. The harness fails when the count differs from what the layer
    /// gives a gradient, and it fails when any name differs from the name that the layer gives.
    ///
    /// That second check is what makes a rename inside a layer reach this net. It is an
    /// assertion and not a recorded value, so it costs the record nothing.
    pub fn with_parameter_grads(mut self, names: &[&'static str]) -> Self {
        self.parameter_names = names.to_vec();
        self
    }

    /// Runs the case in inference mode instead of training mode.
    ///
    /// The harness gives the forward and the backward pass a `Ctx::inference` context. A layer
    /// that does not depend on the mode reads the flag of that context and ignores it.
    pub fn in_inference_mode(mut self) -> Self {
        self.training = false;
        self
    }

    /// Drops the assertion that the inference pass equals the forward pass.
    ///
    /// Use this only for a mode-dependent layer in training mode, where the 2 paths differ by
    /// design. Every other case keeps the assertion.
    pub fn with_inference_that_differs(mut self) -> Self {
        self.inference_matches_training = false;
        self
    }
}

/// Every case for 1 layer type, together with the type name that keys them.
pub struct LayerFixture {
    /// The string that `Layer::layer_type` returns. The harness checks the built layer
    layer_type: &'static str,
    /// Produces the cases for this layer type, in the order the data file records them
    cases: fn() -> Vec<GoldenCase>,
}

impl LayerFixture {
    /// Registers the case function for 1 layer type.
    ///
    /// # Parameters
    ///
    /// - `layer_type` - The exact string that `Layer::layer_type` returns for this layer
    /// - `cases` - Function that produces every case for this layer type
    ///
    /// # Returns
    ///
    /// - `Self` - The registration entry for a family list
    pub fn new(layer_type: &'static str, cases: fn() -> Vec<GoldenCase>) -> Self {
        Self { layer_type, cases }
    }
}

/// Records 1 family and compares it against the family data file.
///
/// This is the only entry point that a family file calls. It runs every case, builds the
/// record, and then either compares it against `golden/data/<family>.golden` or, under a valid
/// regeneration request, writes that file. See [`regenerate`] for the guard that a request must
/// pass, and the module doc comment for the reason that the guard exists.
///
/// It also replays every case a second time with the tuning gates forced, and compares that
/// second record against the same data file. See the module doc comment. The second pass never
/// writes a data file, and it runs before the write, so a disagreement between the 2 passes
/// stops a regeneration.
///
/// # Parameters
///
/// - `family` - Family name. It is the data file stem, so it carries no whitespace
/// - `fixtures` - Every layer type of the family, in the order the data file records them
///
/// # Panics
///
/// - If a recorded value differs from the value in the data file
/// - If the forced-parallel replay differs from the gated replay
/// - If the data file is absent, unreadable, or malformed
/// - If the data file disagrees with its own `content_digest` header line, which says that an
///   edit that this harness did not write reached the file
/// - If the data file is absent and [`ESTABLISHED_FAMILIES`] names the family, or if the data
///   file exists and that roster does not name the family
/// - If a layer reports a type that differs from the registered one
/// - If the inference pass differs from the forward pass on a case that expects them to agree
/// - If a regeneration would add, change, or remove a case of a data file that exists, and the
///   environment carries no acknowledgment token for that exact change
/// - If a regeneration found a data file that holds every recorded value in another byte
///   layout. The run rewrites the file in the layout of [`render`], and it then stops with the
///   report of that repair
pub fn run_family(family: &str, fixtures: &[LayerFixture]) {
    let recorded = record_gated(fixtures);
    let path = data_path(family);
    let stored = read_stored(family, &path);
    check_family_roster(family, &path, stored.is_some());

    if let Some(request) = regeneration_request() {
        // The forced replay runs before the write, so no regeneration can put a
        // serial-versus-parallel disagreement in a data file
        compare(&record_forced(fixtures), &recorded, &path, Replay::Forced);
        if let Err(report) = regenerate(family, &path, &recorded, stored.as_ref(), &request) {
            panic!("{report}");
        }
        return;
    }

    let stored = stored.unwrap_or_else(|| {
        panic!(
            "the golden data file {} is absent\n\
             Record it with: {REGEN_VARIABLE}=\"<reason>\" cargo test --test neural_network \
             golden",
            path.display()
        )
    });
    compare(&recorded, &stored.cases, &path, Replay::Gated);
    compare(
        &record_forced(fixtures),
        &stored.cases,
        &path,
        Replay::Forced,
    );
}

/// Which of the 2 replays of a family produced a record.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Replay {
    /// Every tuning gate holds the value that the crate configured
    Gated,
    /// Every tuning gate holds 0, so each gated kernel takes its parallel branch
    Forced,
}

/// Records the family with the gates as the crate configured them.
///
/// The shared side of the gate lock keeps this pass out of the window where another test moves
/// a gate. See `common::read_gates`.
fn record_gated(fixtures: &[LayerFixture]) -> Vec<RecordedCase> {
    let _shared = read_gates();
    record_family(fixtures)
}

/// Records the family a second time with every gate at [`FORCED_PARALLEL_GATE`] and every
/// task-size cap at [`FORCED_SPLIT_CHUNK`].
///
/// The gate opens the parallel branch of each gated kernel, and the cap makes that branch build
/// more than 1 task. See the module doc comment for why both are needed.
///
/// The guard holds the exclusive side of the gate lock, and restores every gate and every cap
/// on drop, which includes the path where a case panics. See `common::GateGuard`.
fn record_forced(fixtures: &[LayerFixture]) -> Vec<RecordedCase> {
    let _gates = GateGuard::set_all(FORCED_PARALLEL_GATE)
        .with_split_cap(FORCED_SPLIT_CHUNK)
        .without_split_cap(UNCAPPED_DRIVER);
    record_family(fixtures)
}

/// The path of the data file for a family.
fn data_path(family: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/neural_network/golden/data");
    path.push(format!("{family}.golden"));
    path
}

// ---------------------------------------------------------------------------------------
// The regeneration guard
// ---------------------------------------------------------------------------------------

/// The text of a data file, together with the cases that the text holds.
///
/// The regeneration path needs both. The cases give the change that a regeneration would make,
/// and the text carries the reason lines of the regenerations before this one.
struct StoredFile {
    /// The whole file, as it is on disk
    text: String,
    /// Every case of the file, in file order
    cases: Vec<RecordedCase>,
}

/// Reads, parses, and verifies the data file of a family, and gives `None` when the file is
/// absent.
///
/// An absent file is the bootstrap state of a new family, and [`check_family_roster`] decides
/// whether the family may be in that state. Any other read error, and any malformed file, is a
/// panic. A file that disagrees with its own `content_digest` line is a panic as well, on the
/// comparison path and on the regeneration path alike. See [`verify_digest`].
fn read_stored(family: &str, path: &std::path::Path) -> Option<StoredFile> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!(
            "cannot read the golden data file {}: {error}",
            path.display()
        ),
    };
    let parsed = parse(&text, family, path);
    if let Err(report) = verify_digest(&text, &parsed.cases, parsed.claim, path) {
        panic!("{report}");
    }
    Some(StoredFile {
        text,
        cases: parsed.cases,
    })
}

/// Refuses a data file that the roster and the disk disagree about.
///
/// A deleted data file and a data file that never existed look the same on disk, so the harness
/// reads [`ESTABLISHED_FAMILIES`] to tell the 2 apart. The safe default is the deletion, and a
/// family reaches the bootstrap path only when the roster leaves it out.
///
/// # Parameters
///
/// - `family` - Family name, which is the data file stem
/// - `path` - The data file of the family, for the report alone
/// - `present` - Whether the data file exists
///
/// # Panics
///
/// - If the file is absent and the roster names the family
/// - If the file exists and the roster does not name the family
fn check_family_roster(family: &str, path: &std::path::Path, present: bool) {
    match (ESTABLISHED_FAMILIES.contains(&family), present) {
        (true, false) => panic!(
            "REFUSED: the golden data file {} is absent, and {family} is an established \
             family.\n\n\
             A data file that a deletion removed and a data file that never existed look the \
             same on disk. The harness therefore reads the roster ESTABLISHED_FAMILIES of \
             tests/neural_network/golden/mod.rs, and that roster names {family}.\n\n\
             A deletion turns every case of the family into the first capture of a new \
             family. A first capture needs the reason alone, so a regeneration over a deleted \
             file writes whatever the code produces, and it reports nothing. This refusal \
             exists to stop that state.\n\n\
             Restore the file:\n\n    \
             git checkout -- {}\n\n\
             A family that must give up its data file gives up its roster entry in the same \
             change. That edit is in a source file, and a reviewer reads it next to the \
             deletion.",
            path.display(),
            path.display()
        ),
        (false, true) => panic!(
            "the golden data file {} exists, and the roster ESTABLISHED_FAMILIES does not name \
             the family {family}.\n\n\
             The first capture of a family writes the data file, and the roster entry is the \
             second step. Add \"{family}\" to ESTABLISHED_FAMILIES in \
             tests/neural_network/golden/mod.rs. That roster is the 1 record that tells a \
             deleted data file from a data file that never existed.",
            path.display()
        ),
        _ => {}
    }
}

/// What the `content_digest` line of a data file claims about the body of that file.
#[derive(Clone, Copy)]
struct DigestClaim {
    /// The self digest of the whole file, from [`self_digest`]
    digest: u64,
    /// The number of cases that the file holds
    cases: usize,
    /// The number of recorded values that the file holds, over every tensor of every case
    values: usize,
}

/// The result of a parse: every case of a data file, and the claim of the header of that file.
struct ParsedFile {
    /// Every case of the file, in file order
    cases: Vec<RecordedCase>,
    /// What the `content_digest` line claims about those cases
    claim: DigestClaim,
}

/// The digest that a data file carries over itself.
///
/// The digest covers every byte of the file, with 1 exception that no digest can avoid: its own
/// digits read as zeros. The header comments, the reason history, the format line, the family
/// line, the case count, the value count, and every recorded value therefore take part.
///
/// The mixing function is the FNV-1a round of [`mix_byte`], which this harness owns. It gives
/// the same digits on every platform and on every release of the compiler. `DefaultHasher` does
/// not, so no part of this file uses it.
///
/// The function reads the text 1 line at a time and joins the lines with 1 line feed, so a file
/// with carriage returns and the same file without them give the same digits.
///
/// # Parameters
///
/// - `text` - The whole text of a data file
///
/// # Returns
///
/// - `u64` - The digest, which the header holds as [`CHECKSUM_DIGITS`] lowercase hexadecimal
///   digits
fn self_digest(text: &str) -> u64 {
    let canonical = with_digest_digits(text, &"0".repeat(CHECKSUM_DIGITS));
    let mut state = CHECKSUM_BASIS;
    for byte in canonical.bytes() {
        state = mix_byte(state, byte);
    }
    state
}

/// Puts the self digest of a rendered data file into the `content_digest` line of that file.
///
/// [`render`] writes that line with zeros in place of the digits, and this function fills them
/// in. The text that comes back is the text that the harness writes to disk.
fn seal(text: &str) -> String {
    let digest = self_digest(text);
    with_digest_digits(text, &format!("{digest:0width$x}", width = CHECKSUM_DIGITS))
}

/// Rewrites the text of a data file with `digits` in its `content_digest` line, and leaves
/// every other byte of every line as it is.
fn with_digest_digits(text: &str, digits: &str) -> String {
    let mut result = String::with_capacity(text.len() + digits.len());
    let mut written = false;
    for line in text.lines() {
        match tail_of_digest_line(line) {
            Some(tail) if !written => {
                written = true;
                result.push_str(DIGEST_KEYWORD);
                result.push(' ');
                result.push_str(digits);
                result.push_str(tail);
            }
            _ => result.push_str(line),
        }
        result.push('\n');
    }
    result
}

/// The part of a `content_digest` line that follows the digits, or `None` for any other line.
///
/// The match is strict. A line that carries the keyword with any leading blank, or with no
/// digits, is no digest line here, and it therefore goes into the digest as it stands. The
/// parser accepts only the strict form, so the writer and the reader always agree.
fn tail_of_digest_line(line: &str) -> Option<&str> {
    let rest = line.strip_prefix(DIGEST_KEYWORD)?.strip_prefix(' ')?;
    let end = rest.find(' ')?;
    Some(&rest[end..])
}

/// The number of cases, and the number of recorded values, that a body holds.
fn body_counts(cases: &[RecordedCase]) -> (usize, usize) {
    let values = cases
        .iter()
        .flat_map(|case| case.tensors.iter())
        .map(|tensor| tensor.bits.len())
        .sum();
    (cases.len(), values)
}

/// Compares the `content_digest` line of a data file against the body of that file.
///
/// The check is what makes an edit by hand visible. It refuses a file that lost a case, a file
/// that gained a case, a file that holds a changed value, and a file that a truncation cut
/// after a complete case. It runs before the harness compares, and before the harness plans a
/// regeneration, so no regeneration reads a body that the header contradicts.
///
/// # Parameters
///
/// - `text` - The whole text of the data file
/// - `cases` - Every case that the parser read out of that text
/// - `claim` - What the `content_digest` line of that text claims
/// - `path` - The data file, for the report
///
/// # Returns
///
/// - `Result<(), String>` - `Ok` when all 3 fields agree, and the refusal report otherwise
fn verify_digest(
    text: &str,
    cases: &[RecordedCase],
    claim: DigestClaim,
    path: &std::path::Path,
) -> Result<(), String> {
    let (case_count, value_count) = body_counts(cases);
    let digest = self_digest(text);
    let mut found: Vec<String> = Vec::new();
    if claim.cases != case_count {
        found.push(format!(
            "  cases: the header records {}, and the body holds {case_count}",
            claim.cases
        ));
    }
    if claim.values != value_count {
        found.push(format!(
            "  values: the header records {}, and the body holds {value_count}",
            claim.values
        ));
    }
    if claim.digest != digest {
        found.push(format!(
            "  content digest: the header records {:0width$x}, and the body gives \
             {digest:0width$x}",
            claim.digest,
            width = CHECKSUM_DIGITS
        ));
    }
    if found.is_empty() {
        return Ok(());
    }

    Err(format!(
        "REFUSED to read {}: the file disagrees with its own header.\n\n{}\n\n\
         The header of a data file holds 1 {DIGEST_KEYWORD} line. That line records the number \
         of cases, the number of recorded values, and a digest over every other byte of the \
         file. The harness writes all 3, and it verifies all 3 before it compares and before it \
         plans a regeneration. A file that fails the check took an edit that this harness did \
         not write.\n\n\
         A deleted case is the edit that this check exists to stop. A deleted case turns a \
         regeneration that would move a recorded value into a regeneration that adds a case \
         back. The check refuses the file whatever the regeneration would do, and it refuses it \
         before the plan exists.\n\n\
         Restore the file, and do not repair it by hand:\n\n    \
         git checkout -- {}\n\n\
         Do not delete the file either. The harness refuses an absent data file of a family \
         that ESTABLISHED_FAMILIES names.",
        path.display(),
        found.join("\n"),
        path.display()
    ))
}

/// What the environment asks of the regeneration path.
struct Regeneration {
    /// The reason that the operator gave. The data file records it as a `# reason` line
    reason: String,
    /// Every acknowledgment token that the operator supplied, in no particular order
    tokens: Vec<String>,
}

/// Reads the regeneration request out of the environment, and validates the reason.
///
/// An unset or empty [`REGEN_VARIABLE`] gives `None`, which is the comparison path. Any other
/// value must be a reason of [`MIN_REASON_LENGTH`] printable characters or more. The value "1"
/// is therefore refused, and the refusal is a panic and not a silent comparison run.
///
/// # Returns
///
/// - `Option<Regeneration>` - The request, or `None` when the environment asks for no
///   regeneration
///
/// # Panics
///
/// - If the value is too short, too long, or holds a character that a comment line cannot hold
fn regeneration_request() -> Option<Regeneration> {
    let reason = std::env::var(REGEN_VARIABLE).ok()?.trim().to_string();
    if reason.is_empty() {
        return None;
    }
    let printable = reason
        .chars()
        .all(|character| character == ' ' || character.is_ascii_graphic());
    assert!(
        printable && (MIN_REASON_LENGTH..=MAX_REASON_LENGTH).contains(&reason.len()),
        "{REGEN_VARIABLE}={reason:?} is not a reason. The value of the variable is the reason \
         for the regeneration, and the data file records it. Give {MIN_REASON_LENGTH} printable \
         ASCII characters or more, and {MAX_REASON_LENGTH} or fewer, such as:\n    \
         {REGEN_VARIABLE}=\"add the dilation cases of stage 0\" cargo test --test \
         neural_network golden"
    );
    let tokens = std::env::var(ACCEPT_VARIABLE)
        .unwrap_or_default()
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect();
    Some(Regeneration { reason, tokens })
}

/// 1 case that a regeneration would change, and every difference that the change holds.
struct ChangedCase {
    /// The layer type and the label of the case, as a report writes them
    key: String,
    /// 1 entry per difference, in the words that the comparison report uses
    differences: Vec<String>,
}

/// 1 case that a regeneration would add, and the values that the case brings.
///
/// A changed case carries the new bits of every value that moves, in its differences. An added
/// case has no difference of its own, because the file holds no earlier value to compare
/// against. The fingerprint takes that place in [`acknowledgment_token`], so an acknowledgment
/// of 1 added case cannot carry over to another content of the same case.
struct AddedCase {
    /// The layer type and the label of the case, as a report writes them
    key: String,
    /// A fold of the raw bits of every recorded value of the case, from [`case_fingerprint`]
    values: u64,
}

/// A fold of the raw bits of every recorded value of 1 case.
///
/// The fold reads the tensors in record order, and it reads the elements of each tensor in flat
/// C order. It uses the FNV-1a round of [`mix_byte`], like every other fingerprint of this
/// harness.
fn case_fingerprint(case: &RecordedCase) -> u64 {
    let mut state = CHECKSUM_BASIS;
    for tensor in &case.tensors {
        for bits in &tensor.bits {
            for byte in bits.to_le_bytes() {
                state = mix_byte(state, byte);
            }
        }
    }
    state
}

/// What 1 regeneration would do to the data file of a family.
///
/// The plan serves exactly 2 situations, and the rule differs between them. See [`affected`]
/// for the rule, and [`regenerate`] for the guard that applies it.
///
/// [`affected`]: RegenerationPlan::affected
struct RegenerationPlan {
    /// Every case that the record holds and the file does not
    added: Vec<AddedCase>,
    /// The key of every case that the file holds and the record does not
    removed: Vec<String>,
    /// Every case that both hold with a difference between them
    changed: Vec<ChangedCase>,
    /// Whether the data file is absent, which is the first capture of a family. Every case of
    /// such a plan is an addition, and there is no earlier value that an addition can hide
    bootstrap: bool,
}

impl RegenerationPlan {
    /// Whether the record and the file already agree, case for case and value for value.
    ///
    /// The answer says nothing about the byte layout of the file. A file that holds every case
    /// in another order gives an empty plan, because the comparison reads a case by its key.
    /// See [`canonical_text`].
    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    /// The number of cases that 1 acknowledgment token must cover.
    ///
    /// The count serves the 2 situations that a regeneration meets:
    ///
    /// 1. **The data file does not exist.** This is the first capture of a family, and
    ///    [`check_family_roster`] already refused every other reason for an absent file. Every
    ///    case is an addition, the file holds no earlier value, and no addition can therefore
    ///    hide one. The count is 0, and the reason alone writes the file.
    /// 2. **The data file exists.** An addition counts like a change and like a removal,
    ///    because the 3 look the same from the record alone. A case that the file lacks is
    ///    either a case that no file ever held, or a case that left the file after the last
    ///    write. The second one turns a value that moves into a value that arrives, and it is
    ///    the signature of the whole laundering class. The harness cannot tell the 2 apart, so
    ///    it demands the token for both.
    fn affected(&self) -> usize {
        let added = if self.bootstrap { 0 } else { self.added.len() };
        added + self.removed.len() + self.changed.len()
    }

    /// The number of single differences over every changed case.
    fn differences(&self) -> usize {
        self.changed.iter().map(|case| case.differences.len()).sum()
    }
}

/// Works out what a regeneration would do to the data file.
///
/// `stored` is `None` when the data file does not exist, and the plan carries that state as
/// [`RegenerationPlan::bootstrap`].
fn plan_regeneration(
    recorded: &[RecordedCase],
    stored: Option<&[RecordedCase]>,
) -> RegenerationPlan {
    let key_of = |case: &RecordedCase| format!("layer {}, case {}", case.layer_type, case.label);
    let mut plan = RegenerationPlan {
        added: Vec::new(),
        removed: Vec::new(),
        changed: Vec::new(),
        bootstrap: stored.is_none(),
    };
    let stored = stored.unwrap_or(&[]);

    for case in recorded {
        let Some(other) = stored
            .iter()
            .find(|candidate| candidate.key() == case.key())
        else {
            plan.added.push(AddedCase {
                key: key_of(case),
                values: case_fingerprint(case),
            });
            continue;
        };
        let mut differences = Vec::new();
        compare_case(case, other, &mut differences);
        if !differences.is_empty() {
            plan.changed.push(ChangedCase {
                key: key_of(case),
                differences,
            });
        }
    }

    for case in stored {
        if !recorded
            .iter()
            .any(|candidate| candidate.key() == case.key())
        {
            plan.removed.push(key_of(case));
        }
    }

    plan
}

/// The token that acknowledges exactly 1 change set.
///
/// The token holds the family name, the number of affected cases, and a fingerprint of the
/// whole plan. The fingerprint covers every difference, and a difference carries the new bits
/// of the value that moves. It covers the values of every added case as well, through
/// [`case_fingerprint`]. A run that produces another change set therefore gives another token,
/// and the acknowledgment of the operator cannot carry over to a change that the operator never
/// read.
fn acknowledgment_token(family: &str, plan: &RegenerationPlan) -> String {
    let mut text = String::new();
    writeln!(text, "family {family}").unwrap();
    for case in &plan.added {
        writeln!(text, "add {} values {:016x}", case.key, case.values).unwrap();
    }
    for key in &plan.removed {
        writeln!(text, "remove {key}").unwrap();
    }
    for case in &plan.changed {
        writeln!(text, "change {}", case.key).unwrap();
        for difference in &case.differences {
            writeln!(text, "  {difference}").unwrap();
        }
    }

    let mut state = CHECKSUM_BASIS;
    for byte in text.bytes() {
        state = mix_byte(state, byte);
    }
    format!("{family}-{}-{state:016x}", plan.affected())
}

/// The report that a refused regeneration panics with.
///
/// It names every case that would change, the number of differences in each one, and the
/// first few differences of each one. The operator reads that report, forces a rebuild, and
/// then repeats the run with the token.
fn refusal_report(
    family: &str,
    path: &std::path::Path,
    plan: &RegenerationPlan,
    token: &str,
) -> String {
    let mut text = String::new();
    writeln!(
        text,
        "REFUSED to regenerate {}\n\n\
         The {family} family record differs from the file in {} case(s): {} added, {} changed, \
         and {} removed. {} difference(s) in total.\n\n\
         The file exists, so an added case needs the token as well. A case that the file lacks \
         is either a case that no file ever held, or a case that left the file after the last \
         write. The second one turns a value that moves into a value that arrives. This harness \
         cannot tell the 2 apart, so it asks the operator.\n\n\
         Each difference below comes from the comparison, and it reads in the words of the \
         comparison. \"expected\" is the value that the file holds today, and \"found\" is the \
         value that this regeneration would write in place of it.\n",
        path.display(),
        plan.affected(),
        plan.added.len(),
        plan.changed.len(),
        plan.removed.len(),
        plan.differences()
    )
    .unwrap();

    for case in plan.added.iter().take(MAX_REPORTED_CHANGES) {
        writeln!(text, "  {}: added", case.key).unwrap();
    }
    for key in plan.removed.iter().take(MAX_REPORTED_CHANGES) {
        writeln!(text, "  {key}: removed").unwrap();
    }
    for case in plan.changed.iter().take(MAX_REPORTED_CHANGES) {
        writeln!(
            text,
            "  {}: {} difference(s)",
            case.key,
            case.differences.len()
        )
        .unwrap();
        for difference in case.differences.iter().take(MAX_REPORTED_DIFFERENCES) {
            writeln!(text, "      {difference}").unwrap();
        }
        let hidden = case
            .differences
            .len()
            .saturating_sub(MAX_REPORTED_DIFFERENCES);
        if hidden > 0 {
            writeln!(
                text,
                "      ... and {hidden} more difference(s) in this case"
            )
            .unwrap();
        }
    }
    let hidden = plan.added.len().saturating_sub(MAX_REPORTED_CHANGES)
        + plan.removed.len().saturating_sub(MAX_REPORTED_CHANGES)
        + plan.changed.len().saturating_sub(MAX_REPORTED_CHANGES);
    if hidden > 0 {
        writeln!(text, "  ... and {hidden} more affected case(s)").unwrap();
    }

    writeln!(
        text,
        "\n\
         A regeneration that moves a value that the file already holds makes the new value the \
         baseline. A defect in the code therefore becomes the record of correct behavior, which \
         is the worst outcome this net can produce. Take 2 steps before the second run:\n\
         \n    \
         1. Force a rebuild of the test binary. Cargo reuses the binary it has when a source \
         file comes back with an mtime that it had before, and a stale binary writes stale \
         values. Touch every source file that changed, or run `cargo clean -p rustyml`.\n    \
         2. Read every difference above, value by value, and decide that each one is intended.\n\
         \n\
         Then run the regeneration again with the token of this exact change:\n\
         \n    \
         {REGEN_VARIABLE}=\"<reason>\" \\\n      \
         {ACCEPT_VARIABLE}={token} \\\n      \
         cargo test --test neural_network golden\n\
         \n\
         The variable takes a list, because 1 run covers every family. Separate the tokens with \
         a comma or with whitespace. The token covers every difference above, which includes \
         the new bits of every value that moves. A second run that produces another change set \
         gives another token, and this refusal repeats."
    )
    .unwrap();

    text
}

/// The text that a data file of this record holds when its byte layout is the one that
/// [`render`] produces.
///
/// The check that reads this text is the 1 check that sees the layout of a file. Every other
/// check reads a case by its key, so a file that holds every case in another order passes all
/// of them, and [`RegenerationPlan::is_empty`] then reports that nothing would change.
///
/// The reasons come out of the file itself, and not from the request of this run. A run that
/// gives a new reason therefore leaves a canonical file canonical, and the check reports the
/// layout alone.
///
/// # Parameters
///
/// - `family` - Family name, which is the data file stem
/// - `recorded` - The gated record of this run
/// - `text` - The whole text of the file on disk
///
/// # Returns
///
/// - `String` - The sealed text that this record renders, with the reason history of the file
fn canonical_text(family: &str, recorded: &[RecordedCase], text: &str) -> String {
    seal(&render(family, recorded, &previous_reasons(Some(text))))
}

/// The report of a data file that the regeneration rewrote for its byte layout alone.
///
/// The write moved no recorded value, because the record and the file agree case for case and
/// value for value. The run stops nonetheless, because a layout that this harness did not write
/// is a state that an operator must see. A silent repair is the state that this whole guard
/// exists to remove.
fn layout_repair_report(family: &str, path: &std::path::Path) -> String {
    format!(
        "REPAIRED the byte layout of {}\n\n\
         The {family} family record and the file agree in every case and in every value. The \
         byte layout of the file was not the layout that this harness writes, so the \
         regeneration rewrote the file in that layout, and it moved no recorded value.\n\n\
         A file that holds its cases in another order reads the same to every other check of \
         this net, because a case is read by its key. Such a file also makes the next \
         regeneration diff unreadable, because every case then moves in the diff next to the \
         values that really changed.\n\n\
         Read the diff of the file now:\n\n    \
         git diff -- {}\n\n\
         Expect a reordering of whole cases, a header that this harness writes, or both. Expect \
         no value line to hold another number. Then run the suite again, which now compares \
         against the repaired file.",
        path.display(),
        path.display()
    )
}

/// Writes the data file of a family, when the guard permits the write.
///
/// A plan that moves nothing, over a file that already holds the byte layout of [`render`],
/// writes nothing. The first capture of a family writes on the reason alone, because the file
/// that would hold an earlier value does not exist. Over a file that exists, an added case, a
/// changed case, and a removed case each need the acknowledgment token of that exact change
/// set. See [`RegenerationPlan::affected`].
///
/// A file that holds every recorded value and another byte layout is rewritten on the reason
/// alone, and the run then stops with [`layout_repair_report`]. The rewrite moves no value, and
/// the report exists because a repair that nobody sees is a silent write.
///
/// # Parameters
///
/// - `family` - Family name, which is the data file stem
/// - `path` - The data file of the family
/// - `recorded` - The gated record of this run, which becomes the new file
/// - `stored` - The file on disk, or `None` when the file is absent
/// - `request` - The reason and the acknowledgment tokens from the environment
///
/// # Returns
///
/// - `Result<(), String>` - `Ok` when the guard permits the write and the write needs no
///   report, the refusal report when the plan needs a token that the environment does not
///   carry, and the repair report when the write corrected the byte layout alone
///
/// # Panics
///
/// - If the file cannot be written
fn regenerate(
    family: &str,
    path: &std::path::Path,
    recorded: &[RecordedCase],
    stored: Option<&StoredFile>,
    request: &Regeneration,
) -> Result<(), String> {
    let plan = plan_regeneration(recorded, stored.map(|file| file.cases.as_slice()));
    // A file that holds every recorded value in another byte layout gives an empty plan, and
    // the harness repairs the layout instead of leaving it
    let repair =
        stored.is_some_and(|file| canonical_text(family, recorded, &file.text) != file.text);
    if plan.is_empty() && !repair {
        return Ok(());
    }

    if plan.affected() > 0 {
        let token = acknowledgment_token(family, &plan);
        if !request.tokens.iter().any(|supplied| supplied == &token) {
            return Err(refusal_report(family, path, &plan, &token));
        }
    }

    let mut reasons = vec![request.reason.clone()];
    reasons.extend(previous_reasons(stored.map(|file| file.text.as_str())));
    reasons.truncate(MAX_RECORDED_REASONS);

    let directory = path.parent().expect("the data path has a parent");
    std::fs::create_dir_all(directory)
        .unwrap_or_else(|error| panic!("cannot create {}: {error}", directory.display()));
    // The rendered text carries zeros in place of its own digest digits, and `seal` fills them
    // in. The file that reaches the disk therefore always agrees with its own header
    std::fs::write(path, seal(&render(family, recorded, &reasons)))
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));

    if plan.is_empty() {
        return Err(layout_repair_report(family, path));
    }
    Ok(())
}

/// The reason of every regeneration that the header of a data file still carries, newest first.
fn previous_reasons(text: Option<&str>) -> Vec<String> {
    text.into_iter()
        .flat_map(str::lines)
        .filter_map(|line| line.strip_prefix(REASON_LINE_PREFIX))
        .map(|reason| reason.trim().to_string())
        .filter(|reason| !reason.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------------------

/// 1 recorded tensor: a name, a shape, and the raw f32 bits in flat C order.
#[derive(PartialEq, Eq)]
struct RecordedTensor {
    /// Tensor name, such as "input", "forward", "predict", or "grad_param.bias"
    name: String,
    /// Shape in C order
    shape: Vec<usize>,
    /// The decay class, for a parameter gradient alone. `None` for every other
    /// tensor. `Some(true)` is `ParamRef::weight`, and `Some(false)` is `ParamRef::no_decay`
    decays: Option<bool>,
    /// The value of every element, as the raw bits from `f32::to_bits`
    bits: Vec<u32>,
}

/// The name, the shape, and the value fingerprint of 1 weight that `LayerBase::weights` exposes.
///
/// A shape alone does not pin the identity of an array. 4 normalization layers expose 2 or more
/// arrays of the same shape, and BatchNormalization exposes 4. The fingerprint therefore rides
/// next to the shape, and a stage that exchanges 2 such arrays fails the comparison.
#[derive(Clone, PartialEq, Eq)]
struct RecordedWeight {
    /// The name that `LayerBase::weights` gives the array
    name: String,
    /// Shape in C order
    shape: Vec<usize>,
    /// The fingerprint of every value of the array, from [`weight_checksum`]
    checksum: u64,
}

/// 1 exposed weight, together with the storage that it borrows.
///
/// The record goes in the data file. The anchor never does. The harness compares the anchor
/// against the parameter of the same name, and a difference says that the 2 stopped being 1
/// storage. See [`assert_same_storage`].
struct ExposedWeight {
    /// What the data file holds about this weight
    record: RecordedWeight,
    /// The address of the first element and the element count, for an array that holds its
    /// elements in 1 contiguous run. `None` for every other array
    anchor: Option<(usize, usize)>,
}

/// 1 recorded case, keyed by the layer type and the configuration label.
struct RecordedCase {
    /// The string that `Layer::layer_type` returned
    layer_type: String,
    /// The configuration label of the case
    label: String,
    /// Whether the case ran in training mode
    training: bool,
    /// What `Layer::output_shape` returned after the forward pass. It is a display string
    /// today, and a later stage is expected to give it a typed return value
    output_shape: String,
    /// What `Layer::param_count` returned, as the words the data file holds
    param_count: String,
    /// The name of every parameter of `LayerBase::parameters_mut`, in the order that method
    /// returns
    /// them. The layer supplies every one of them
    param_names: Vec<String>,
    /// The shape and the value fingerprint of every weight that `LayerBase::weights` exposes, in
    /// the order that method returns them
    weights: Vec<RecordedWeight>,
    /// The same weights, fingerprinted a second time after the optimizer step. A parameter
    /// store that the step cannot write through leaves every one of these unchanged
    step_weights: Vec<RecordedWeight>,
    /// Every recorded tensor, in the order the harness produced them
    tensors: Vec<RecordedTensor>,
}

impl RecordedCase {
    /// The key that pairs a recorded case with a stored case.
    fn key(&self) -> (&str, &str) {
        (self.layer_type.as_str(), self.label.as_str())
    }
}

/// Runs every case of every fixture and builds the record.
fn record_family(fixtures: &[LayerFixture]) -> Vec<RecordedCase> {
    let mut recorded = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();

    for fixture in fixtures {
        for case in (fixture.cases)() {
            assert!(
                !case.label.is_empty() && !case.label.contains(char::is_whitespace),
                "the label of a {} case must be a single word without whitespace, got {:?}",
                fixture.layer_type,
                case.label
            );
            let key = (fixture.layer_type.to_string(), case.label.to_string());
            assert!(
                !seen.contains(&key),
                "layer type {} has 2 cases labeled {:?}",
                fixture.layer_type,
                case.label
            );
            seen.push(key);
            recorded.push(record_case(fixture.layer_type, &case));
        }
    }

    recorded
}

/// Runs 1 case and records every tensor it produces.
fn record_case(layer_type: &str, case: &GoldenCase) -> RecordedCase {
    let input = golden_input(&case.input_shape);

    // The 2 passes use separate layers, so the inference pass can never disturb the state that
    // the forward and backward pass builds. The mode is the context now, so the inference pass
    // takes a context of its own and reads no field of the layer
    let mut inference_layer = (case.build)();
    check_layer_type(layer_type, inference_layer.as_ref(), case.label);
    let mut inference_ctx = Ctx::inference();
    let predicted = inference_layer
        .forward_many_mut(&[&input], &mut inference_ctx)
        .unwrap_or_else(|error| {
            panic!(
                "{layer_type}/{}: the inference forward pass failed: {error}",
                case.label
            )
        });
    // An inference pass parks nothing at all, so the cache channel must stay empty. A layer
    // that parks a cache in this mode leaks 1 value per pass, and no backward pass takes it
    // back
    assert_eq!(
        inference_ctx.pending_caches(),
        0,
        "{layer_type}/{}: the inference forward pass parked {} caches, and it must park none",
        case.label,
        inference_ctx.pending_caches()
    );

    // 1 context serves the forward pass and the backward pass of the case
    let mut ctx = if case.training {
        Ctx::training()
    } else {
        Ctx::inference()
    };
    let mut layer = (case.build)();
    let forward = layer
        .forward_many_mut(&[&input], &mut ctx)
        .unwrap_or_else(|error| panic!("{layer_type}/{}: forward failed: {error}", case.label));
    // The forward pass takes `&self`, so a layer that changes non-trainable state proposes the
    // new value in the context. The layer takes it here, which completes the pass. The running
    // statistics of a normalization layer and the random stream of a dropout layer arrive this
    // way, and a model runs the same 2 steps in the same order. The position is 0, because the
    // harness drives 1 layer and sets no owner
    if ctx.has_state(0) {
        layer.apply_state(&mut ctx.state_slot(0));
    }

    if case.inference_matches_training {
        assert_eq!(
            forward.shape(),
            predicted.shape(),
            "{layer_type}/{}: the inference pass and the forward pass returned different shapes",
            case.label
        );
        for (index, (from_forward, from_predict)) in
            forward.iter().zip(predicted.iter()).enumerate()
        {
            assert_eq!(
                from_forward.to_bits(),
                from_predict.to_bits(),
                "{layer_type}/{}: the inference pass differs from the forward pass at flat \
                 index {index}: forward {from_forward:?}, inference {from_predict:?}",
                case.label
            );
        }
    }

    // The metadata comes from the layer after its forward pass. A layer that holds no output
    // shape until it has seen an input can only report one at this point
    let output_shape = layer.output_shape();
    let param_count = param_count_record(layer.param_count());
    // The fingerprints come from the weights as the forward pass left them, and before the
    // optimizer step below, so the 2 recorded fields stay independent of each other. The
    // anchors go no further than this function, and they prove that a parameter and the array
    // of the same name are 1 storage
    let exposed = exposed_weights(layer.as_ref());
    let weights: Vec<RecordedWeight> = exposed.iter().map(|item| item.record.clone()).collect();
    let anchors: Vec<(String, usize, usize)> = exposed
        .iter()
        .filter_map(|item| {
            item.anchor
                .map(|(address, length)| (item.record.name.clone(), address, length))
        })
        .collect();

    let upstream = golden_gradient(forward.shape());
    let mut grad_inputs = layer
        .backward_many(&upstream, &mut ctx)
        .unwrap_or_else(|error| panic!("{layer_type}/{}: backward failed: {error}", case.label));
    assert_eq!(
        grad_inputs.len(),
        1,
        "{layer_type}/{}: every fixture layer takes 1 input, and the backward pass gave {} \
         gradients",
        case.label,
        grad_inputs.len()
    );
    let grad_input = grad_inputs.remove(0);

    // The tensor keeps the name `predict`, because that name is in every data file. The call
    // behind it is an inference forward pass now
    let mut tensors = vec![
        tensor_record("input", &input),
        tensor_record("forward", &forward),
        tensor_record("predict", &predicted),
        tensor_record("grad_input", &grad_input),
    ];

    // `LayerBase::parameters_mut` yields flat slices in a stable order, which is exactly what
    // an optimizer consumes. The record keeps that flat form. The entry holds no gradient any
    // more, so the gradient comes from the store of the context, under the address that the
    // layer position and the parameter name build. The position is 0, because the harness
    // drives 1 layer and sets no owner
    //
    // The method yields every trainable tensor of the layer, whether a backward pass gave that
    // tensor a gradient or not. The recorded set is the tensors that HAVE a gradient, which is
    // exactly what the earlier entry list held. Each gradient goes into an owned vector here,
    // so the read of the context ends before the step loop writes through the layer
    let mut parameters: Vec<(&'static str, bool, &mut [f32], Vec<f32>)> = Vec::new();
    for parameter in layer.parameters_mut() {
        if let Some(grad) = ctx.grads().get(ParamId::new(0, parameter.name)) {
            let values: Vec<f32> = grad.iter().copied().collect();
            parameters.push((parameter.name, parameter.decays, parameter.value, values));
        }
    }
    assert_eq!(
        parameters.len(),
        case.parameter_names.len(),
        "{layer_type}/{}: the case names {} parameter gradients, and the layer returned {}",
        case.label,
        case.parameter_names.len(),
        parameters.len()
    );
    let mut param_names: Vec<String> = Vec::new();
    let mut layer_names: Vec<&'static str> = Vec::new();
    for ((parameter_name, decays, value, grad), fixture_name) in
        parameters.iter_mut().zip(case.parameter_names.iter())
    {
        // The optimizer keys its per-parameter state on the layer name, so 2 tensors of 1 layer
        // that share a name share their momentum. Nothing else in the crate would report it
        assert!(
            !layer_names.contains(parameter_name),
            "{layer_type}/{}: 2 parameters of this layer carry the name {parameter_name}",
            case.label
        );
        layer_names.push(parameter_name);
        // The layer names every parameter of its own. The fixture declares the same name, so a
        // rename inside a layer fails here instead of passing unseen. This is an assertion and
        // not a recorded value
        assert_eq!(
            parameter_name, fixture_name,
            "{layer_type}/{}: the layer names this parameter {parameter_name}, and the case \
             names it {fixture_name}",
            case.label
        );
        // The layer owns the name end to end now, so the recorded name is the name in the
        // parameter entry. The anchor comparison stayed, as an assertion: the array that
        // `LayerBase::weights` gives this name must be the storage that the parameter writes
        // through. See the section "Where a parameter name comes from"
        let name = parameter_name.to_string();
        assert_same_storage(&anchors, &name, value, layer_type, case.label);
        assert!(
            grad.len() <= MAX_TENSOR_ELEMENTS,
            "{layer_type}/{}: the gradient of {name} holds {} elements, and the cap is \
             {MAX_TENSOR_ELEMENTS}",
            case.label,
            grad.len()
        );
        assert_eq!(
            value.len(),
            grad.len(),
            "{layer_type}/{}: the parameter {name} holds {} values and {} gradient values, and \
             an optimizer needs the 2 slices to agree",
            case.label,
            value.len(),
            grad.len()
        );
        tensors.push(RecordedTensor {
            name: format!("{PARAMETER_PREFIX}{name}"),
            shape: vec![grad.len()],
            // The flag decides what an optimizer with a non-zero weight decay does to this
            // tensor, and it moves no gradient value, so the record must hold it
            decays: Some(*decays),
            bits: grad.iter().map(|value| value.to_bits()).collect(),
        });
        // 1 step of the harness rule binds this gradient to the tensor that it updates. A
        // stage that aims the gradient at another tensor keeps every gradient value, and moves
        // this one. The caller discards the layer instance directly after `record_case`
        // returns, so no later record reads a stepped value
        for (value, gradient) in value.iter_mut().zip(grad.iter()) {
            *value -= OPTIMIZER_STEP * gradient;
        }
        tensors.push(RecordedTensor {
            name: format!("{STEPPED_PREFIX}{name}"),
            shape: vec![value.len()],
            // The matching gradient tensor already holds the decay class of this parameter
            decays: None,
            bits: value.iter().map(|value| value.to_bits()).collect(),
        });
        param_names.push(name);
    }

    // The second fingerprint reads the arrays of the layer itself, and the step above wrote
    // through the slices that `parameters_mut` handed out. An implementation of that method
    // which gives back a copy of each value buffer therefore leaves every one of these
    // unchanged, while every `step_param` tensor above still holds the stepped numbers
    drop(parameters);
    let step_weights: Vec<RecordedWeight> = exposed_weights(layer.as_ref())
        .into_iter()
        .map(|item| item.record)
        .collect();

    RecordedCase {
        layer_type: layer_type.to_string(),
        label: case.label.to_string(),
        training: case.training,
        output_shape,
        param_count,
        param_names,
        weights,
        step_weights,
        tensors,
    }
}

/// Fails when a parameter and the exposed array of the same name are not 1 storage.
///
/// `LayerBase::parameters_mut` and `LayerBase::weights` name the same tensors, and the checkpoint
/// format
/// reads the second list. A parameter that an optimizer writes must therefore reach the array
/// that a saved model holds under that name. The 2 lists are 2 separate methods, and nothing in
/// the compiler binds them, so this comparison is what holds them together.
///
/// The check takes the address of the first element and the element count of the parameter, and
/// looks for the exposed array of the same name. That array must start at the same address and
/// hold the same number of elements.
///
/// # Parameters
///
/// - `anchors` - The name, the first-element address, and the element count of every exposed
///   array
/// - `name` - The name that the layer gave this parameter
/// - `value` - The parameter value slice that `LayerBase::parameters_mut` handed out
/// - `layer_type` - The layer type of the case, for the report
/// - `label` - The configuration label of the case, for the report
///
/// # Panics
///
/// - When no exposed array carries the name, or when the array of that name is another storage
fn assert_same_storage(
    anchors: &[(String, usize, usize)],
    name: &str,
    value: &[f32],
    layer_type: &str,
    label: &str,
) {
    let address = value.as_ptr() as usize;
    let Some((_, anchor_address, length)) = anchors.iter().find(|(other, _, _)| other == name)
    else {
        let held: Vec<&str> = anchors.iter().map(|(other, _, _)| other.as_str()).collect();
        panic!(
            "{layer_type}/{label}: the parameter {name} has no weight of that name; the layer \
             exposes {held:?}"
        );
    };
    assert!(
        *anchor_address == address && *length == value.len(),
        "{layer_type}/{label}: the parameter {name} and the weight {name} are not 1 storage. \
         The parameter holds {} elements at {address:#x}, and the weight holds {length} \
         elements at {anchor_address:#x}",
        value.len()
    );
}

/// Turns a `param_count` result into the words that the data file holds.
///
/// [`ParamCounts`] holds 2 independent counts, and the line holds both of them, always. The
/// earlier line held 1 number and a word that said which count it was. That form could not
/// record a layer that holds both kinds at once, so the 5 BatchNormalization cases recorded
/// their trainable count alone and left the running mean and the running variance out of the
/// record entirely. The 2 numbers here are the whole of what the method returns.
fn param_count_record(count: ParamCounts) -> String {
    format!(
        "trainable {} non_trainable {}",
        count.trainable, count.non_trainable
    )
}

/// Folds the values of 1 exposed array into a 64-bit fingerprint.
///
/// The rule is FNV-1a over the raw bytes, and this harness owns it. The state starts at
/// [`CHECKSUM_BASIS`]. For every element, in logical C order, the function takes
/// [`f32::to_bits`] and feeds the 4 bytes of that word, low byte first. It then feeds the 8
/// bytes of the element count, low byte first. Each byte is exclusive-ored into the state, and
/// the state is then multiplied by [`CHECKSUM_PRIME`] with wrapping arithmetic.
///
/// The count goes in as well, so an empty array gives a value that no start value repeats.
///
/// `DefaultHasher` is not used here. The standard library gives its output no stability
/// guarantee across Rust releases, and a data file must read the same in every year.
///
/// # Parameters
///
/// - `array` - 1 array that `LayerBase::weights` exposes. The harness also fingerprints owned
///   arrays of other ranks in its own tests, so the function takes every rank
///
/// # Returns
///
/// - `u64` - The fingerprint, which a data file holds as 16 lowercase hexadecimal digits
fn weight_checksum<S, D>(array: &ArrayBase<S, D>) -> u64
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    let mut state = CHECKSUM_BASIS;
    // `iter` walks logical C order, whatever strides the array carries
    for value in array.iter() {
        for byte in value.to_bits().to_le_bytes() {
            state = mix_byte(state, byte);
        }
    }
    for byte in (array.len() as u64).to_le_bytes() {
        state = mix_byte(state, byte);
    }
    state
}

/// 1 round of FNV-1a over 1 byte.
///
/// [`weight_checksum`] and [`acknowledgment_token`] both fold with this round.
#[inline]
fn mix_byte(state: u64, byte: u8) -> u64 {
    (state ^ u64::from(byte)).wrapping_mul(CHECKSUM_PRIME)
}

/// The address of the first element and the element count of 1 array that holds its elements in
/// 1 contiguous run.
///
/// An array with any other layout gives `None`. Every view here borrows the live layer, so the
/// address is the address of the storage of the layer itself.
///
/// # Parameters
///
/// - `array` - 1 view that `LayerBase::weights` borrowed from the layer
///
/// # Returns
///
/// - `Option<(usize, usize)>` - The first-element address and the element count, or `None`
fn contiguous_anchor(array: &ArrayViewD<'_, f32>) -> Option<(usize, usize)> {
    let slice = array.as_slice()?;
    Some((slice.as_ptr() as usize, slice.len()))
}

/// Turns `LayerBase::weights` into the name, the shape, the fingerprint, and the storage anchor of
/// every array that the layer exposes.
///
/// The order is the order the layer gives, which is the order a checkpoint records. A layer
/// with no array gives the empty list.
///
/// Every name here comes from the layer. The earlier version of this function held 1 name
/// literal per array of every weight-enum variant, and a rename inside a layer left those
/// literals where they were. The names now ride with the arrays, so this function invents
/// nothing at all.
///
/// # Parameters
///
/// - `layer` - The layer to read, after its forward pass
///
/// # Returns
///
/// - `Vec<ExposedWeight>` - 1 entry per array, in the order the layer gives
fn exposed_weights(layer: &dyn Layer) -> Vec<ExposedWeight> {
    layer
        .weights()
        .into_iter()
        .map(|entry| ExposedWeight {
            record: RecordedWeight {
                name: entry.name.to_string(),
                shape: entry.value.shape().to_vec(),
                checksum: weight_checksum(&entry.value),
            },
            anchor: contiguous_anchor(&entry.value),
        })
        .collect()
}

/// Fails when a built layer reports a type that differs from the registered one.
fn check_layer_type(expected: &str, layer: &dyn Layer, label: &str) {
    assert_eq!(
        layer.layer_type(),
        expected,
        "the fixture registers {expected}/{label}, and the built layer reports {}",
        layer.layer_type()
    );
}

/// Turns 1 tensor into its record, and checks the size cap.
fn tensor_record(name: &str, tensor: &Tensor) -> RecordedTensor {
    assert!(
        tensor.len() <= MAX_TENSOR_ELEMENTS,
        "the recorded tensor {name} holds {} elements, and the cap is {MAX_TENSOR_ELEMENTS}",
        tensor.len()
    );
    RecordedTensor {
        name: name.to_string(),
        shape: tensor.shape().to_vec(),
        // A decay class belongs to a parameter gradient alone
        decays: None,
        // `iter` walks logical C order, whatever strides the tensor carries
        bits: tensor.iter().map(|value| value.to_bits()).collect(),
    }
}

/// The word that the data file uses for a decay class.
fn decay_word(decays: bool) -> &'static str {
    if decays { "decays" } else { "no_decay" }
}

/// The name that a report gives a decay class, including the absent one.
fn decay_class_name(decays: Option<bool>) -> &'static str {
    match decays {
        Some(value) => decay_word(value),
        None => "absent",
    }
}

// ---------------------------------------------------------------------------------------
// The data file writer
// ---------------------------------------------------------------------------------------

/// Renders a family record as the text of its data file.
///
/// `reasons` holds the reason of every regeneration that the file keeps, newest first. The
/// header writes 1 `# reason` line per entry, and the next regeneration reads those lines back.
/// See [`previous_reasons`].
///
/// The text that comes back is not the text of the file yet. Its `content_digest` line carries
/// zeros in place of the digits, and [`seal`] fills them in.
fn render(family: &str, cases: &[RecordedCase], reasons: &[String]) -> String {
    let mut text = String::new();
    writeln!(text, "# Golden fixture data for the \"{family}\" family.").unwrap();
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "# The harness in tests/neural_network/golden wrote this file. Do not edit it by hand."
    )
    .unwrap();
    writeln!(text, "# Regenerate it with:").unwrap();
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "#     {REGEN_VARIABLE}=\"<reason>\" cargo test --test neural_network golden"
    )
    .unwrap();
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "# Regeneration is an explicit claim that the behavior change was intended and reviewed."
    )
    .unwrap();
    writeln!(
        text,
        "# The first capture of a family needs that claim alone. A run that adds, changes, or"
    )
    .unwrap();
    writeln!(
        text,
        "# removes a case of a file that exists is refused 1 time. The refusal prints every"
    )
    .unwrap();
    writeln!(
        text,
        "# difference and 1 acknowledgment token, and a second run must carry that token in"
    )
    .unwrap();
    writeln!(
        text,
        "# {ACCEPT_VARIABLE}. Force a rebuild of the test binary before every regeneration."
    )
    .unwrap();
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "# The {DIGEST_KEYWORD} line below records a digest of this whole file, the number of"
    )
    .unwrap();
    writeln!(
        text,
        "# cases, and the number of recorded values. The harness verifies all 3 before it reads"
    )
    .unwrap();
    writeln!(
        text,
        "# the file. An edit by hand therefore stops every run of the net. Restore such a file"
    )
    .unwrap();
    writeln!(
        text,
        "# from version control. A deletion of this file is refused as well."
    )
    .unwrap();
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "# Regeneration history, newest first. Each line holds the reason of 1 run that wrote"
    )
    .unwrap();
    writeln!(text, "# this file.").unwrap();
    writeln!(text, "#").unwrap();
    for reason in reasons {
        writeln!(text, "{REASON_LINE_PREFIX}{reason}").unwrap();
    }
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "# Each value line holds the raw f32 bits from to_bits, in 8 lowercase hexadecimal"
    )
    .unwrap();
    writeln!(
        text,
        "# digits. The text after the \"#\" is the same value in decimal, and it is a comment."
    )
    .unwrap();
    writeln!(text, "#").unwrap();
    writeln!(
        text,
        "# The module doc comment of tests/neural_network/golden/mod.rs holds the grammar."
    )
    .unwrap();
    writeln!(text).unwrap();
    writeln!(text, "format {FORMAT_VERSION}").unwrap();
    writeln!(text, "family {family}").unwrap();
    // The digits stand at zero here, because a digest cannot cover its own digits. `seal` puts
    // the digest of the finished text in their place
    let (case_count, value_count) = body_counts(cases);
    writeln!(
        text,
        "{DIGEST_KEYWORD} {} cases {case_count} values {value_count}",
        "0".repeat(CHECKSUM_DIGITS)
    )
    .unwrap();

    for case in cases {
        writeln!(text).unwrap();
        writeln!(text, "case {} {}", case.layer_type, case.label).unwrap();
        writeln!(
            text,
            "mode {}",
            if case.training {
                "training"
            } else {
                "inference"
            }
        )
        .unwrap();
        writeln!(text, "output_shape {}", quote(&case.output_shape)).unwrap();
        writeln!(text, "param_count {}", case.param_count).unwrap();
        for name in &case.param_names {
            writeln!(text, "param_name {name} {LAYER_NAME_SOURCE}").unwrap();
        }
        for (keyword, group) in [
            ("weight", &case.weights),
            ("step_weight", &case.step_weights),
        ] {
            for weight in group {
                write!(text, "{keyword} {}", weight.name).unwrap();
                for extent in &weight.shape {
                    write!(text, " {extent}").unwrap();
                }
                writeln!(
                    text,
                    " checksum {:0width$x}",
                    weight.checksum,
                    width = CHECKSUM_DIGITS
                )
                .unwrap();
            }
        }
        for tensor in &case.tensors {
            write!(text, "tensor {}", tensor.name).unwrap();
            for extent in &tensor.shape {
                write!(text, " {extent}").unwrap();
            }
            if let Some(decays) = tensor.decays {
                write!(text, " {}", decay_word(decays)).unwrap();
            }
            writeln!(text).unwrap();
            for &bits in &tensor.bits {
                writeln!(text, "  {:08x}  # {:?}", bits, f32::from_bits(bits)).unwrap();
            }
        }
        writeln!(text, "end").unwrap();
    }

    text
}

/// Writes a string as 1 quoted, escaped word, so that it survives a line of whitespace-split
/// text.
///
/// A backslash, a double quote, and the 3 control characters that a line cannot hold get a
/// backslash. [`unquote`] is the reverse.
fn quote(value: &str) -> String {
    let mut text = String::with_capacity(value.len() + 2);
    text.push('"');
    for character in value.chars() {
        match character {
            '\\' => text.push_str("\\\\"),
            '"' => text.push_str("\\\""),
            '\n' => text.push_str("\\n"),
            '\r' => text.push_str("\\r"),
            '\t' => text.push_str("\\t"),
            _ => text.push(character),
        }
    }
    text.push('"');
    text
}

/// Reads back what [`quote`] wrote, and gives `None` for text that it never writes.
fn unquote(text: &str) -> Option<String> {
    let inner = text.strip_prefix('"')?.strip_suffix('"')?;
    let mut value = String::with_capacity(inner.len());
    let mut characters = inner.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            // A bare double quote can only be the closing one, which the strip already took
            if character == '"' {
                return None;
            }
            value.push(character);
            continue;
        }
        match characters.next()? {
            '\\' => value.push('\\'),
            '"' => value.push('"'),
            'n' => value.push('\n'),
            'r' => value.push('\r'),
            't' => value.push('\t'),
            _ => return None,
        }
    }
    Some(value)
}

// ---------------------------------------------------------------------------------------
// The data file reader
// ---------------------------------------------------------------------------------------

/// Parses the text of a data file into the same record shape that the harness builds.
///
/// The parse reads the `content_digest` line as well, and it gives that claim back beside the
/// cases. It compares nothing. [`verify_digest`] holds the comparison, because a claim that
/// disagrees with the body is no parse error.
///
/// # Panics
///
/// - If the format version, the family name, or any line is not what the writer produces
/// - If the file holds no `content_digest` line, or holds more than 1
fn parse(text: &str, family: &str, path: &std::path::Path) -> ParsedFile {
    let fail = |line: usize, message: String| -> ! {
        panic!("{}:{line}: {message}", path.display());
    };

    let mut cases: Vec<RecordedCase> = Vec::new();
    let mut claim: Option<DigestClaim> = None;
    let mut format_seen = false;
    let mut family_seen = false;
    let mut open: Option<RecordedCase> = None;
    let mut tensor: Option<RecordedTensor> = None;
    let mut output_shape_seen = false;
    let mut param_count_seen = false;
    // Every line before the `format` line is a comment, so the version is known before the
    // first line that reads it
    let mut file_version = FORMAT_VERSION;

    for (offset, raw) in text.lines().enumerate() {
        let line = offset + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut words = trimmed.split_whitespace();
        let keyword = words.next().expect("a non-empty line holds a word");

        match keyword {
            "format" => {
                let version: u32 = words
                    .next()
                    .and_then(|word| word.parse().ok())
                    .unwrap_or_else(|| fail(line, "format needs a whole-number version".into()));
                if version != FORMAT_VERSION && version != EARLIER_FORMAT_VERSION {
                    fail(
                        line,
                        format!(
                            "format {version} is neither the format {FORMAT_VERSION} that this \
                             harness writes nor the format {EARLIER_FORMAT_VERSION} that it \
                             still reads"
                        ),
                    );
                }
                file_version = version;
                format_seen = true;
            }
            "family" => {
                let name = words
                    .next()
                    .unwrap_or_else(|| fail(line, "family needs a name".into()));
                if name != family {
                    fail(line, format!("family {name} is not the expected {family}"));
                }
                family_seen = true;
            }
            DIGEST_KEYWORD => {
                if claim.is_some() {
                    fail(line, format!("{DIGEST_KEYWORD} appears more than 1 time"));
                }
                if open.is_some() || !cases.is_empty() {
                    fail(
                        line,
                        format!("{DIGEST_KEYWORD} appears after the first case of the file"),
                    );
                }
                // The strict form that `with_digest_digits` reads back: the keyword, 1 blank,
                // the digits, and then the 2 counts
                if raw != trimmed || tail_of_digest_line(raw).is_none() {
                    fail(
                        line,
                        format!(
                            "{DIGEST_KEYWORD} takes no leading blank, and it takes \
                             {CHECKSUM_DIGITS} lowercase hexadecimal digits and then the counts"
                        ),
                    );
                }
                let digits = words
                    .next()
                    .unwrap_or_else(|| fail(line, format!("{DIGEST_KEYWORD} needs digits")));
                let readable = digits.len() == CHECKSUM_DIGITS
                    && digits
                        .bytes()
                        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
                if !readable {
                    fail(
                        line,
                        format!(
                            "the digest {digits} is not {CHECKSUM_DIGITS} lowercase \
                             hexadecimal digits"
                        ),
                    );
                }
                let counts: Vec<&str> = words.collect();
                let numbers = match counts.as_slice() {
                    ["cases", cases_word, "values", values_word] => cases_word
                        .parse::<usize>()
                        .ok()
                        .zip(values_word.parse::<usize>().ok()),
                    _ => None,
                };
                let Some((case_count, value_count)) = numbers else {
                    fail(
                        line,
                        format!(
                            "{DIGEST_KEYWORD} ends with the words: cases <count> values <count>"
                        ),
                    );
                };
                claim = Some(DigestClaim {
                    digest: u64::from_str_radix(digits, 16)
                        .expect("16 hexadecimal digits fit a 64-bit whole number"),
                    cases: case_count,
                    values: value_count,
                });
            }
            "case" => {
                if open.is_some() {
                    fail(line, "a case starts before the previous case ends".into());
                }
                let layer_type = words
                    .next()
                    .unwrap_or_else(|| fail(line, "case needs a layer type".into()));
                let label = words
                    .next()
                    .unwrap_or_else(|| fail(line, "case needs a label".into()));
                open = Some(RecordedCase {
                    layer_type: layer_type.to_string(),
                    label: label.to_string(),
                    training: true,
                    output_shape: String::new(),
                    param_count: String::new(),
                    param_names: Vec::new(),
                    weights: Vec::new(),
                    step_weights: Vec::new(),
                    tensors: Vec::new(),
                });
                output_shape_seen = false;
                param_count_seen = false;
            }
            "mode" => {
                let Some(case) = open.as_mut() else {
                    fail(line, "mode appears outside a case".into());
                };
                match words.next() {
                    Some("training") => case.training = true,
                    Some("inference") => case.training = false,
                    other => fail(
                        line,
                        format!("mode is {other:?}, not training or inference"),
                    ),
                }
            }
            "output_shape" => {
                let Some(case) = open.as_mut() else {
                    fail(line, "output_shape appears outside a case".into());
                };
                // The recorded string holds spaces, so it takes the whole rest of the line
                let quoted = trimmed["output_shape".len()..].trim();
                let Some(value) = unquote(quoted) else {
                    fail(
                        line,
                        format!("output_shape {quoted} is not a quoted, escaped string"),
                    );
                };
                case.output_shape = value;
                output_shape_seen = true;
            }
            "param_count" => {
                let Some(case) = open.as_mut() else {
                    fail(line, "param_count appears outside a case".into());
                };
                let words: Vec<&str> = words.collect();
                let current = match words.as_slice() {
                    ["trainable", trainable, "non_trainable", non_trainable] => {
                        trainable.parse::<usize>().is_ok() && non_trainable.parse::<usize>().is_ok()
                    }
                    _ => false,
                };
                // Version 5 held 1 count, and a word that said which of the 2 counts it was
                let earlier = file_version == EARLIER_FORMAT_VERSION
                    && match words.as_slice() {
                        ["none"] => true,
                        ["trainable" | "non_trainable", total] => total.parse::<usize>().is_ok(),
                        _ => false,
                    };
                if !current && !earlier {
                    fail(
                        line,
                        "param_count is trainable <count> non_trainable <count>".to_string(),
                    );
                }
                case.param_count = words.join(" ");
                param_count_seen = true;
            }
            "param_name" => {
                let Some(case) = open.as_mut() else {
                    fail(line, "param_name appears outside a case".into());
                };
                let name = words
                    .next()
                    .unwrap_or_else(|| fail(line, "param_name needs a name".into()));
                if words.next() != Some(LAYER_NAME_SOURCE) {
                    fail(
                        line,
                        format!("the name source of {name} is not {LAYER_NAME_SOURCE}"),
                    );
                }
                if words.next().is_some() {
                    fail(line, "param_name is a name and then a source".into());
                }
                case.param_names.push(name.to_string());
            }
            "weight" | "step_weight" => {
                let Some(case) = open.as_mut() else {
                    fail(line, format!("{keyword} appears outside a case"));
                };
                if tensor.is_some() || !case.tensors.is_empty() {
                    fail(
                        line,
                        format!("{keyword} appears after the first tensor of the case"),
                    );
                }
                let name = words
                    .next()
                    .unwrap_or_else(|| fail(line, format!("{keyword} needs a name")));
                // The line is the shape, then the word "checksum", then the digits
                let rest: Vec<&str> = words.collect();
                if rest.len() < 3 || rest[rest.len() - 2] != "checksum" {
                    fail(
                        line,
                        format!(
                            "{keyword} {name} needs a shape, and then the word checksum with \
                             {CHECKSUM_DIGITS} lowercase hexadecimal digits"
                        ),
                    );
                }
                let digits = rest[rest.len() - 1];
                let readable = digits.len() == CHECKSUM_DIGITS
                    && digits
                        .bytes()
                        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
                if !readable {
                    fail(
                        line,
                        format!(
                            "the checksum {digits} of {keyword} {name} is not \
                             {CHECKSUM_DIGITS} lowercase hexadecimal digits"
                        ),
                    );
                }
                let checksum = u64::from_str_radix(digits, 16)
                    .expect("16 hexadecimal digits fit a 64-bit whole number");
                let mut shape = Vec::new();
                for word in &rest[..rest.len() - 2] {
                    match word.parse::<usize>() {
                        Ok(extent) => shape.push(extent),
                        Err(_) => fail(line, format!("shape entry {word} is not a whole number")),
                    }
                }
                if shape.is_empty() {
                    fail(line, format!("{keyword} {name} has no shape"));
                }
                let group = if keyword == "weight" {
                    &mut case.weights
                } else {
                    &mut case.step_weights
                };
                group.push(RecordedWeight {
                    name: name.to_string(),
                    shape,
                    checksum,
                });
            }
            "tensor" => {
                let Some(case) = open.as_mut() else {
                    fail(line, "tensor appears outside a case".into());
                };
                if let Some(finished) = tensor.take() {
                    case.tensors.push(finished);
                }
                let name = words
                    .next()
                    .unwrap_or_else(|| fail(line, "tensor needs a name".into()));
                let mut shape = Vec::new();
                let mut decays = None;
                for word in words {
                    if decays.is_some() {
                        fail(
                            line,
                            "the decay class is the last word of a tensor line".into(),
                        );
                    }
                    match word {
                        "decays" => decays = Some(true),
                        "no_decay" => decays = Some(false),
                        _ => match word.parse::<usize>() {
                            Ok(extent) => shape.push(extent),
                            Err(_) => {
                                fail(line, format!("shape entry {word} is not a whole number"))
                            }
                        },
                    }
                }
                if shape.is_empty() {
                    fail(line, format!("tensor {name} has no shape"));
                }
                // Only a parameter gradient carries a decay class, and it always carries one
                if name.starts_with(PARAMETER_PREFIX) != decays.is_some() {
                    fail(
                        line,
                        format!(
                            "tensor {name} has the decay class {}, and a name that starts with \
                             {PARAMETER_PREFIX:?} takes a class while every other name takes none",
                            decay_class_name(decays)
                        ),
                    );
                }
                tensor = Some(RecordedTensor {
                    name: name.to_string(),
                    shape,
                    decays,
                    bits: Vec::new(),
                });
            }
            "end" => {
                let Some(mut case) = open.take() else {
                    fail(line, "end appears outside a case".into());
                };
                if let Some(finished) = tensor.take() {
                    case.tensors.push(finished);
                }
                if !output_shape_seen {
                    fail(
                        line,
                        format!("case {} has no output_shape line", case.label),
                    );
                }
                if !param_count_seen {
                    fail(line, format!("case {} has no param_count line", case.label));
                }
                cases.push(case);
            }
            _ => {
                let Some(active) = tensor.as_mut() else {
                    fail(line, format!("value {keyword} appears outside a tensor"));
                };
                match u32::from_str_radix(keyword, 16) {
                    Ok(bits) if keyword.len() == 8 => active.bits.push(bits),
                    _ => fail(
                        line,
                        format!("{keyword} is not 8 hexadecimal digits of f32 bits"),
                    ),
                }
            }
        }
    }

    if !format_seen {
        fail(
            1,
            "the file has no format line. An empty file and a file that a truncation cut \
             before its header both read like this"
                .into(),
        );
    }
    if !family_seen {
        fail(1, "the file has no family line".into());
    }
    if open.is_some() {
        fail(
            text.lines().count(),
            "the last case has no end line. A file that a truncation cut inside a case reads \
             like this"
                .into(),
        );
    }
    let Some(claim) = claim else {
        fail(
            1,
            format!(
                "the file has no {DIGEST_KEYWORD} line. Every data file that this harness \
                 writes carries one, so restore the file from version control"
            ),
        );
    };

    ParsedFile { cases, claim }
}

// ---------------------------------------------------------------------------------------
// The comparison and its failure report
// ---------------------------------------------------------------------------------------

/// Compares the fresh record against the stored one, and panics with a full report.
///
/// `replay` names the pass that produced `recorded`, and it selects the advice that the report
/// ends with. A [`Replay::Forced`] difference is a defect in a layer, never a stale fixture.
fn compare(
    recorded: &[RecordedCase],
    stored: &[RecordedCase],
    path: &std::path::Path,
    replay: Replay,
) {
    let mut problems: Vec<String> = Vec::new();

    for case in recorded {
        let Some(other) = stored
            .iter()
            .find(|candidate| candidate.key() == case.key())
        else {
            problems.push(format!(
                "layer {}, case {}: the data file holds no record for this case. Record it \
                 with {REGEN_VARIABLE}=\"<reason>\"",
                case.layer_type, case.label
            ));
            continue;
        };
        compare_case(case, other, &mut problems);
    }

    for case in stored {
        if !recorded
            .iter()
            .any(|candidate| candidate.key() == case.key())
        {
            problems.push(format!(
                "layer {}, case {}: the data file holds this case, and no fixture produces it \
                 any more. Remove it with {REGEN_VARIABLE}=\"<reason>\"",
                case.layer_type, case.label
            ));
        }
    }

    if problems.is_empty() {
        return;
    }

    let total = problems.len();
    problems.truncate(MAX_REPORTED_PROBLEMS);
    let shown = problems.len();
    let (headline, advice) = match replay {
        Replay::Gated => (
            "the recorded behavior differs from",
            format!(
                "The layer behavior changed. Fix the layer, or, when the change was intended \
                 and reviewed, record the new behavior with:\n    \
                 {REGEN_VARIABLE}=\"<reason>\" cargo test --test neural_network golden"
            ),
        ),
        Replay::Forced => (
            "the forced-parallel replay differs from",
            format!(
                "This is the forced-parallel replay. Every gate below held \
                 {FORCED_PARALLEL_GATE}, so each gated kernel took its parallel branch, and \
                 every task-size cap below held {FORCED_SPLIT_CHUNK}, so each capped driver \
                 split its work into more than 1 task. The gated replay of the same cases \
                 matched the data file. A serial kernel and its parallel kernel therefore \
                 disagree, or a multi-task split gives another result than a single task. Both \
                 are a defect in the layer. Do NOT record this with {REGEN_VARIABLE}. The \
                 forced gates were:\n    {}\nThe forced task-size caps were:\n    {}",
                NEURAL_NETWORK_GATES
                    .iter()
                    .map(|(name, _, _)| *name)
                    .collect::<Vec<&str>>()
                    .join("\n    "),
                NEURAL_NETWORK_SPLIT_CAPS
                    .iter()
                    .map(|(name, _, _)| *name)
                    .collect::<Vec<&str>>()
                    .join("\n    ")
            ),
        ),
    };
    panic!(
        "{headline} {}\n\n{}\n\n{total} problem(s) in total, {shown} shown.\n{advice}",
        path.display(),
        problems.join("\n"),
    );
}

/// Compares 1 case, tensor by tensor, and appends every problem it finds.
fn compare_case(recorded: &RecordedCase, stored: &RecordedCase, problems: &mut Vec<String>) {
    let head = format!("layer {}, case {}", recorded.layer_type, recorded.label);

    if recorded.training != stored.training {
        problems.push(format!(
            "{head}: mode is {} now, and the data file records {}",
            mode_name(recorded.training),
            mode_name(stored.training)
        ));
    }

    if recorded.output_shape != stored.output_shape {
        problems.push(format!(
            "{head}: output_shape is {:?} now, and the data file records {:?}",
            recorded.output_shape, stored.output_shape
        ));
    }

    if recorded.param_count != stored.param_count {
        problems.push(format!(
            "{head}: param_count is {:?} now, and the data file records {:?}",
            recorded.param_count, stored.param_count
        ));
    }

    compare_param_names(&head, recorded, stored, problems);
    compare_weights(
        &head,
        "weight",
        &recorded.weights,
        &stored.weights,
        problems,
    );
    compare_weights(
        &head,
        "step_weight",
        &recorded.step_weights,
        &stored.step_weights,
        problems,
    );

    for tensor in &recorded.tensors {
        let Some(other) = stored
            .tensors
            .iter()
            .find(|candidate| candidate.name == tensor.name)
        else {
            problems.push(format!(
                "{head}, tensor {}: the data file holds no such tensor",
                tensor.name
            ));
            continue;
        };
        compare_tensor(&head, tensor, other, problems);
    }

    for tensor in &stored.tensors {
        if !recorded
            .tensors
            .iter()
            .any(|candidate| candidate.name == tensor.name)
        {
            problems.push(format!(
                "{head}, tensor {}: the data file holds this tensor, and the layer no longer \
                 produces it",
                tensor.name
            ));
        }
    }
}

/// Compares the parameter roster, and appends every problem it finds.
///
/// A rename in the parameter store of a later stage lands here first, and so does a parameter
/// that the layer stops yielding or starts yielding.
fn compare_param_names(
    head: &str,
    recorded: &RecordedCase,
    stored: &RecordedCase,
    problems: &mut Vec<String>,
) {
    if recorded.param_names != stored.param_names {
        problems.push(format!(
            "{head}: the parameters are {:?} now, and the data file records {:?}",
            recorded.param_names, stored.param_names
        ));
    }
}

/// Compares the shapes and the value fingerprints of 1 group of exposed weights, and appends
/// every problem it finds.
///
/// `keyword` is the data-file keyword of the group, so the report names the group that differs.
fn compare_weights(
    head: &str,
    keyword: &str,
    recorded: &[RecordedWeight],
    stored: &[RecordedWeight],
    problems: &mut Vec<String>,
) {
    let before = problems.len();
    for weight in recorded {
        let Some(other) = stored
            .iter()
            .find(|candidate| candidate.name == weight.name)
        else {
            problems.push(format!(
                "{head}, {keyword} {}: the layer exposes this weight, and the data file holds \
                 no such weight",
                weight.name
            ));
            continue;
        };
        if weight.shape != other.shape {
            problems.push(format!(
                "{head}, {keyword} {}: the shape is {:?} now, and the data file records {:?}",
                weight.name, weight.shape, other.shape
            ));
        }
        if weight.checksum != other.checksum {
            problems.push(format!(
                "{head}, {keyword} {}: the value checksum is {:0width$x} now, and the data file \
                 records {:0width$x}. Either the values of the array changed, or 2 exposed \
                 arrays of the same shape changed places",
                weight.name,
                weight.checksum,
                other.checksum,
                width = CHECKSUM_DIGITS
            ));
        }
    }

    for weight in stored {
        if !recorded
            .iter()
            .any(|candidate| candidate.name == weight.name)
        {
            problems.push(format!(
                "{head}, {keyword} {}: the data file holds this weight, and the layer no \
                 longer exposes it",
                weight.name
            ));
        }
    }

    let found: Vec<&str> = recorded.iter().map(|weight| weight.name.as_str()).collect();
    let expected: Vec<&str> = stored.iter().map(|weight| weight.name.as_str()).collect();
    if found != expected {
        problems.push(format!(
            "{head}: the {keyword} group holds {found:?} now, and the data file records \
             {expected:?}"
        ));
    }

    // A step_weight fingerprint reads the arrays of the layer after the optimizer step. The
    // step wrote through the slices of `LayerBase::parameters_mut`, so a store that gives back a
    // copy of each value buffer keeps every one of these at its value before the step
    if problems.len() > before && keyword == "step_weight" {
        problems.push(format!(
            "{head}: a {keyword} fingerprint holds the weight after 1 step of \
             param -= {OPTIMIZER_STEP} * grad, and LayerBase::weights reads the arrays of the \
             layer itself. Check whether LayerBase::parameters_mut hands out the storage of the \
             layer or a copy of it. A copy makes every step_param tensor agree and leaves the \
             layer unchanged, which is training that does nothing"
        ));
    }
}

/// Compares 1 tensor bit by bit, and appends every problem it finds.
fn compare_tensor(
    head: &str,
    recorded: &RecordedTensor,
    stored: &RecordedTensor,
    problems: &mut Vec<String>,
) {
    if recorded.decays != stored.decays {
        problems.push(format!(
            "{head}, tensor {}: the decay class is {} now, and the data file records {}. The \
             class decides what an optimizer with a non-zero weight decay does to the parameter",
            recorded.name,
            decay_class_name(recorded.decays),
            decay_class_name(stored.decays)
        ));
    }
    if recorded.shape != stored.shape {
        problems.push(format!(
            "{head}, tensor {}: shape is {:?} now, and the data file records {:?}",
            recorded.name, recorded.shape, stored.shape
        ));
        return;
    }
    if recorded.bits.len() != stored.bits.len() {
        problems.push(format!(
            "{head}, tensor {}: holds {} values now, and the data file records {}",
            recorded.name,
            recorded.bits.len(),
            stored.bits.len()
        ));
        return;
    }

    let before = problems.len();
    for (index, (&found, &expected)) in recorded.bits.iter().zip(stored.bits.iter()).enumerate() {
        if found != expected {
            problems.push(format!(
                "{head}, tensor {}, flat index {index}: expected bits 0x{expected:08x} \
                 ({:?}), found bits 0x{found:08x} ({:?})",
                recorded.name,
                f32::from_bits(expected),
                f32::from_bits(found)
            ));
        }
    }

    // A stepped parameter holds `param - OPTIMIZER_STEP * grad`. When the matching gradient
    // tensor still agrees, the gradient reached another parameter than the recorded one
    if problems.len() > before && recorded.name.starts_with(STEPPED_PREFIX) {
        problems.push(format!(
            "{head}, tensor {}: this tensor holds the parameter after 1 step of \
             param -= {OPTIMIZER_STEP} * grad. Check whether the gradient reached the right \
             parameter tensor, whether the parameter list came back in a new order, and \
             whether the step ran 1 time for each parameter",
            recorded.name
        ));
    }
}

/// The word that the data file uses for a training mode.
fn mode_name(training: bool) -> &'static str {
    if training { "training" } else { "inference" }
}

// ---------------------------------------------------------------------------------------
// The tests of the regeneration guard
// ---------------------------------------------------------------------------------------

/// Tests of the guard, over a scratch data file that no family reads.
///
/// Every test builds its own file out of the cases of the misc data file, and then drives
/// [`regenerate`] and [`verify_digest`] against that scratch file. No test of this module
/// writes a data file of the net.
///
/// The tests cover the 2 legitimate paths that an attack cannot reach from the command line:
/// the first capture of a family whose data file does not exist, and an acknowledged change.
/// They cover the 2 refused paths beside them: an addition over a data file that exists, and a
/// file whose byte layout this harness did not write.
mod guard_tests {
    use super::*;

    /// The reason that every test of this module gives to the guard.
    const TEST_REASON: &str = "a test of the regeneration guard";

    /// The path of 1 scratch data file under the build directory of the crate.
    fn scratch(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("target/golden_guard_tests");
        std::fs::create_dir_all(&path).expect("the scratch directory is writable");
        path.push(format!("{name}.golden"));
        path
    }

    /// A record of `count` cases, in the shape that a family produces.
    ///
    /// The cases are of this module, and no test reads a data file of the net. A regeneration
    /// run rewrites those files while these tests run, and a test that read one would take a
    /// half-written file.
    ///
    /// The record holds every field that the writer and the reader carry: both modes, a
    /// parameter name, an exposed weight before and after the optimizer step, a plain tensor, a
    /// parameter gradient with its decay class, and the parameter that 1 step left behind.
    fn sample_cases(count: usize) -> Vec<RecordedCase> {
        (0..count)
            .map(|index| {
                let seed = index as u32;
                let bits = |offset: u32| -> Vec<u32> {
                    (0..6)
                        .map(|element| 0x3f80_0000 + seed * 16 + offset + element)
                        .collect()
                };
                RecordedCase {
                    layer_type: "Sample".to_string(),
                    label: format!("case_{index}"),
                    training: index % 2 == 0,
                    output_shape: "(None, 3)".to_string(),
                    param_count: "trainable 6 non_trainable 0".to_string(),
                    param_names: vec!["kernel".to_string()],
                    weights: vec![RecordedWeight {
                        name: "kernel".to_string(),
                        shape: vec![2, 3],
                        checksum: 0x0123_4567_89ab_cdef + u64::from(seed),
                    }],
                    step_weights: vec![RecordedWeight {
                        name: "kernel".to_string(),
                        shape: vec![2, 3],
                        checksum: 0xfedc_ba98_7654_3210 - u64::from(seed),
                    }],
                    tensors: vec![
                        RecordedTensor {
                            name: "input".to_string(),
                            shape: vec![2, 3],
                            decays: None,
                            bits: bits(0),
                        },
                        RecordedTensor {
                            name: "grad_param.kernel".to_string(),
                            shape: vec![6],
                            decays: Some(true),
                            bits: bits(6),
                        },
                        RecordedTensor {
                            name: "step_param.kernel".to_string(),
                            shape: vec![6],
                            decays: None,
                            bits: bits(12),
                        },
                    ],
                }
            })
            .collect()
    }

    /// Writes `cases` to `path` as a sealed data file, and reads that file back.
    fn store(path: &std::path::Path, cases: &[RecordedCase]) -> StoredFile {
        let text = seal(&render("sample", cases, &[]));
        std::fs::write(path, text).expect("the scratch file is writable");
        read_stored("sample", path).expect("the scratch file exists")
    }

    /// Joins lines with 1 line feed each, which is the form that [`self_digest`] reads.
    fn join(lines: &[String]) -> String {
        let mut text = String::new();
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
        text
    }

    /// The first capture of a family writes the data file, and it needs the reason alone.
    ///
    /// This is the 1 situation in which an addition needs no token. The data file does not
    /// exist, so it holds no value that the addition can hide.
    #[test]
    fn a_first_capture_writes_on_the_reason_alone() {
        let all = sample_cases(6);
        let path = scratch("bootstrap");
        // The bootstrap state is an absent file, and a run of this test may find the file of
        // the run before it
        let _ = std::fs::remove_file(&path);

        let plan = plan_regeneration(&all, None);
        assert_eq!(plan.added.len(), all.len(), "every case is an addition");
        assert!(plan.bootstrap, "the data file does not exist");
        assert_eq!(
            plan.affected(),
            0,
            "a first capture hides no value of any file"
        );

        let request = Regeneration {
            reason: TEST_REASON.to_string(),
            tokens: Vec::new(),
        };
        regenerate("sample", &path, &all, None, &request)
            .expect("a first capture needs the reason alone");

        // `read_stored` verifies the header of the file that the write produced
        let written = read_stored("sample", &path).expect("the write produced a file");
        assert_eq!(written.cases.len(), all.len());
        assert!(plan_regeneration(&all, Some(&written.cases)).is_empty());
    }

    /// A case that the record holds and a file that exists lacks needs the token of that
    /// change.
    ///
    /// A deleted case is what makes this route matter. A deletion turns a value that moves into
    /// a value that arrives, and an addition that needs no token writes the new value with no
    /// report at all.
    #[test]
    fn an_addition_to_an_existing_file_needs_the_token_of_that_change() {
        let all = sample_cases(6);
        let path = scratch("additive");
        // The file holds every case except the last one, which is the state that a deletion
        // leaves behind
        let stored = store(&path, &all[..all.len() - 1]);
        let before = std::fs::read_to_string(&path).expect("the scratch file is readable");

        let plan = plan_regeneration(&all, Some(&stored.cases));
        assert_eq!(plan.added.len(), 1, "the file lacks exactly 1 case");
        assert!(!plan.bootstrap, "the data file exists");
        assert_eq!(
            plan.affected(),
            1,
            "an addition to a file that exists counts"
        );
        let token = acknowledgment_token("sample", &plan);

        let request = Regeneration {
            reason: TEST_REASON.to_string(),
            tokens: Vec::new(),
        };
        let report = regenerate("sample", &path, &all, Some(&stored), &request)
            .expect_err("the reason alone is no acknowledgment of an addition");
        assert!(report.contains("REFUSED to regenerate"));
        assert!(
            report.contains(": added"),
            "the refusal names the added case"
        );
        assert!(
            report.contains(&token),
            "the refusal names the token of this change"
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("the scratch file is readable"),
            before,
            "a refused regeneration writes nothing"
        );

        let accepted = Regeneration {
            reason: TEST_REASON.to_string(),
            tokens: vec![token],
        };
        regenerate("sample", &path, &all, Some(&stored), &accepted)
            .expect("the token of this change permits the write");
        let written = read_stored("sample", &path).expect("the write produced a file");
        assert_eq!(written.cases.len(), all.len());
        assert!(plan_regeneration(&all, Some(&written.cases)).is_empty());
    }

    /// A file that holds every recorded value in another byte layout is repaired, and the run
    /// then reports the repair.
    ///
    /// Every other check of this net reads a case by its key, so a reordered file passes all of
    /// them, and the plan of such a file is empty.
    #[test]
    fn a_reordered_file_is_repaired_and_reported() {
        let all = sample_cases(6);
        let mut reversed = sample_cases(6);
        reversed.reverse();

        let path = scratch("reordered");
        let stored = store(&path, &reversed);
        let before = std::fs::read_to_string(&path).expect("the scratch file is readable");
        assert!(
            plan_regeneration(&all, Some(&stored.cases)).is_empty(),
            "a reordered file holds every case and every value"
        );
        assert_ne!(
            canonical_text("sample", &all, &before),
            before,
            "the byte layout of the file is not the layout that render writes"
        );

        let request = Regeneration {
            reason: TEST_REASON.to_string(),
            tokens: Vec::new(),
        };
        let report = regenerate("sample", &path, &all, Some(&stored), &request)
            .expect_err("a repaired layout is reported");
        assert!(report.contains("REPAIRED the byte layout"));

        let written = std::fs::read_to_string(&path).expect("the scratch file is readable");
        assert_ne!(written, before, "the repair rewrote the file");
        assert_eq!(
            canonical_text("sample", &all, &written),
            written,
            "the repaired file holds the layout that render writes"
        );
        let read_back = read_stored("sample", &path).expect("the write produced a file");
        assert!(plan_regeneration(&all, Some(&read_back.cases)).is_empty());
        // A second regeneration over the repaired file writes nothing and reports nothing
        regenerate("sample", &path, &all, Some(&read_back), &request)
            .expect("a canonical file that holds every value needs no write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("the scratch file is readable"),
            written,
            "a run that finds nothing to do leaves the file as it is"
        );
    }

    /// A regeneration that moves a recorded value is refused, and then the token of that exact
    /// change permits it.
    #[test]
    fn a_changed_case_needs_the_token_of_that_change() {
        let all = sample_cases(6);
        let mut stale = sample_cases(6);
        let target = stale
            .iter_mut()
            .find(|case| case.tensors.iter().any(|tensor| !tensor.bits.is_empty()))
            .expect("a sample case holds a recorded value");
        target.tensors[0].bits[0] ^= 1;

        let path = scratch("changed");
        let stored = store(&path, &stale);
        let before = std::fs::read_to_string(&path).expect("the scratch file is readable");

        let plan = plan_regeneration(&all, Some(&stored.cases));
        assert_eq!(plan.affected(), 1, "exactly 1 case of the file moves");
        let token = acknowledgment_token("sample", &plan);

        let wrong = Regeneration {
            reason: TEST_REASON.to_string(),
            tokens: vec!["sample-1-0000000000000000".to_string()],
        };
        let report = regenerate("sample", &path, &all, Some(&stored), &wrong)
            .expect_err("a token of another change set is no acknowledgment");
        assert!(report.contains("REFUSED to regenerate"));
        assert!(
            report.contains(&token),
            "the refusal names the token of this change"
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("the scratch file is readable"),
            before,
            "a refused regeneration writes nothing"
        );

        let accepted = Regeneration {
            reason: TEST_REASON.to_string(),
            tokens: vec![token],
        };
        regenerate("sample", &path, &all, Some(&stored), &accepted)
            .expect("the token of this change permits the write");
        let written = read_stored("sample", &path).expect("the write produced a file");
        assert!(plan_regeneration(&all, Some(&written.cases)).is_empty());
    }

    /// A case that a hand takes out of the body leaves a header that the body contradicts.
    #[test]
    fn a_deleted_case_fails_the_self_digest() {
        let all = sample_cases(6);
        let sealed = seal(&render("sample", &all, &[]));
        let claim = sealed
            .lines()
            .find(|line| line.starts_with(DIGEST_KEYWORD))
            .expect("a sealed file holds its digest line")
            .to_string();
        // The body loses its last case, and the header of the whole file stays
        let short: Vec<String> = render("sample", &all[..all.len() - 1], &[])
            .lines()
            .map(|line| {
                if line.starts_with(DIGEST_KEYWORD) {
                    claim.clone()
                } else {
                    line.to_string()
                }
            })
            .collect();
        let text = join(&short);

        let path = scratch("deleted");
        let parsed = parse(&text, "sample", &path);
        let report = verify_digest(&text, &parsed.cases, parsed.claim, &path)
            .expect_err("a deleted case is an edit that this harness did not write");
        assert!(report.contains("cases: the header records"));
        assert!(report.contains("values: the header records"));
        assert!(report.contains("content digest: the header records"));
    }

    /// A value that a hand moves leaves both counts as they are, and moves the digest alone.
    #[test]
    fn a_changed_value_fails_the_self_digest() {
        let all = sample_cases(6);
        let mut lines: Vec<String> = seal(&render("sample", &all, &[]))
            .lines()
            .map(str::to_string)
            .collect();
        // A value line is 2 blanks, the 8 digits of the bits, and then the decimal comment
        let index = lines
            .iter()
            .position(|line| line.starts_with("  ") && line.contains(" # "))
            .expect("the file holds a value line");
        let mut bytes = lines[index].clone().into_bytes();
        bytes[9] = if bytes[9] == b'0' { b'1' } else { b'0' };
        lines[index] = String::from_utf8(bytes).expect("a value line is ASCII");
        let text = join(&lines);

        let path = scratch("moved");
        let parsed = parse(&text, "sample", &path);
        let report = verify_digest(&text, &parsed.cases, parsed.claim, &path)
            .expect_err("a moved value is an edit that this harness did not write");
        assert!(report.contains("content digest: the header records"));
        assert!(!report.contains("cases: the header records"));
        assert!(!report.contains("values: the header records"));
    }

    /// The digest reads every byte of the file except its own digits, and it holds a value that
    /// this harness fixes.
    #[test]
    fn the_self_digest_reads_every_byte_except_its_own_digits() {
        // The text is bytes for the digest alone, and no reader of a data file parses it
        let head = "format 5\nfamily misc\n";
        let text = format!("{head}{DIGEST_KEYWORD} ffffffffffffffff cases 0 values 0\n");
        let other = format!("{head}{DIGEST_KEYWORD} 0123456789abcdef cases 0 values 0\n");
        assert_eq!(
            self_digest(&text),
            self_digest(&other),
            "the digits of the line take no part in the digest"
        );

        let counted = format!("{head}{DIGEST_KEYWORD} ffffffffffffffff cases 1 values 0\n");
        assert_ne!(
            self_digest(&text),
            self_digest(&counted),
            "the counts of the line take part in the digest"
        );
        let commented = format!("# a comment\n{text}");
        assert_ne!(
            self_digest(&text),
            self_digest(&commented),
            "a comment line takes part in the digest"
        );

        // The value is a property of the FNV-1a round of this harness, and of nothing else. A
        // release of the compiler cannot move it
        assert_eq!(format!("{:016x}", self_digest(&text)), "d4b111782b0df812");
    }

    /// Every family that owns a data file is in the roster, and every family of the roster owns
    /// a data file.
    ///
    /// The test sees a deleted data file even when nobody runs the family test that reads it.
    #[test]
    fn the_roster_names_every_data_file_on_disk() {
        assert!(
            ESTABLISHED_FAMILIES
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "keep ESTABLISHED_FAMILIES in alphabetical order"
        );
        for family in ESTABLISHED_FAMILIES {
            let path = data_path(family);
            assert!(
                path.exists(),
                "the roster names {family}, and {} is absent",
                path.display()
            );
        }

        let directory = data_path(ESTABLISHED_FAMILIES[0]);
        let directory = directory.parent().expect("the data path has a parent");
        for entry in std::fs::read_dir(directory).expect("the data directory is readable") {
            let path = entry.expect("the directory entry is readable").path();
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("a data file has a name");
            assert!(
                ESTABLISHED_FAMILIES.contains(&stem),
                "{} exists, and the roster does not name {stem}",
                path.display()
            );
        }
    }
}
