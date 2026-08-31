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
//! step leaves behind, and the `predict` output. Each case also records the metadata that the
//! layer reports after its forward pass: the `output_shape` string, the `param_count`
//! classification, the name and the source of the name of every parameter, and the shape and
//! the value fingerprint of every weight that `get_weights` exposes, 1 time before the
//! optimizer step and 1 time after it. For a layer that does not depend on the training mode,
//! the harness also asserts that `predict` returns exactly the `forward` output. That assertion
//! pins a property that a later stage makes structural.
//!
//! **A parameter gradient carries its decay class.** `Layer::parameters` gives each tensor a
//! `decays` flag, and every optimizer with a non-zero weight decay reads it. A change from
//! `ParamGrad::no_decay` to `ParamGrad::weight` therefore changes what training does to that
//! tensor, and no gradient value moves. The record holds the flag next to the gradient, and
//! compares it as strictly as a value.
//!
//! # 1 optimizer step binds each gradient to the parameter that it updates
//!
//! A recorded gradient alone does not say which tensor the gradient updates. Stage 1 of the
//! rewrite deletes `LayerWeight`, turns the parameters into a named list, and re-keys the
//! optimizer from a positional cursor to a path. Such a rewrite can aim a gradient at the wrong
//! parameter tensor and leave every recorded gradient value unchanged.
//!
//! This is the defect class that the `step_param.<name>` tensor exists to catch: a gradient
//! that reaches the wrong parameter tensor, a parameter list that comes back in a new order,
//! and a parameter that the optimizer skips or updates 2 times. Every one of those keeps the
//! gradient values, and every one of those moves a recorded parameter value.
//!
//! The harness applies 1 step of the rule `param[i] -= OPTIMIZER_STEP * grad[i]` to the live
//! `value` slice of each `ParamGrad`, and records the result next to the gradient. See
//! [`OPTIMIZER_STEP`]. The rule lives in this harness, and it comes from no optimizer of
//! `src/neural_network/optimizers`. A borrowed optimizer would make every recorded parameter
//! value change when that optimizer changes, and this net must record the layer behavior alone.
//!
//! The step writes into the layer through the mutable slice that `Layer::parameters` hands out.
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
//! Each case therefore takes a second value fingerprint of every weight that `get_weights`
//! exposes, directly after the optimizer step, and records it as `step_weight.<name>` in a
//! `step_weight` line. `get_weights` reads the arrays of the layer itself, so a stepped copy no
//! longer agrees with it.
//!
//! The 2 fingerprints of 1 name must differ when the step touched that array, and they must
//! agree when no parameter covers it. BatchNormalization shows both halves in 1 case: `gamma`
//! and `beta` move, and `running_mean` and `running_var` do not.
//!
//! # Where a parameter name comes from
//!
//! Each parameter of a case has 1 `param_name` line, and that line ends with the source of the
//! name: `layer` or `fixture`.
//!
//! `ParamGrad` carries no name today, and no method of `Layer` returns one, so the layer cannot
//! name a parameter directly. A fixture names its parameters through
//! [`GoldenCase::with_parameter_grads`]. A name from that list alone pins nothing about the
//! layer, and a stage that renames a parameter therefore keeps every recorded value and every
//! recorded name.
//!
//! The harness closes as much of that hole as the code of today permits. Before the backward
//! pass it takes the address of the first element and the element count of every array that
//! `get_weights` exposes and that borrows the live layer. It then compares the address and the
//! count of each parameter value against that list. A parameter that starts at the address of
//! exactly 1 exposed array is the same storage as that array, so the harness takes the name of
//! that array and marks the line `layer`. See [`resolve_parameter_name`].
//!
//! **What the `layer` mark pins, and what it does not.** Every name in the net is a literal
//! that this harness authors. The name of an exposed array is a literal of the [`exposed_weights`]
//! macro, next to the field access that reads the array, and a fixture supplies the same list
//! to [`GoldenCase::with_parameter_grads`]. The `layer` mark therefore pins 3 things, and the
//! name of the layer is none of them:
//!
//! 1. **The order of the parameter list.** The names ride in the order that
//!    `Layer::parameters` returns, and the comparison is order-sensitive.
//! 2. **The storage identity of each parameter.** Parameter `i` starts at the first element of
//!    the exposed array that carries this name, and it holds the same element count.
//! 3. **The roster.** No parameter appears 2 times, none is missing, and none is a tensor that
//!    `get_weights` hides.
//!
//! The string itself carries no provenance. A rename in the layer changes no literal here. The
//! compiler catches only the field access next to the literal: a renamed field breaks
//! [`exposed_weights`] where it reads the field, and the mechanical repair of that access
//! leaves the old literal in place. The record then keeps a name that the layer no longer uses,
//! and the whole net stays green. Stage 1 re-authors all 21 arms of [`exposed_weights`],
//! because it deletes `LayerWeight`, so this is not a remote hazard.
//!
//! Every parameter of every layer type of this net resolves to `layer` today. No layer keeps a
//! trainable tensor that `get_weights` hides, and no layer exposes a copy of one. A line marked
//! `fixture` therefore says that a parameter and its exposed weight stopped being the same
//! storage, which is the defect of the section above, seen from the other side.
//!
//! **What stage 1 must change here.** Stage 1 gives `ParamGrad` a name of its own. The harness
//! must then read that name from the layer, delete the address comparison and
//! [`resolve_parameter_name`] with it, delete the literals of [`exposed_weights`], and mark
//! every line `layer`. [`GoldenCase::with_parameter_grads`] then keeps the count assertion
//! alone, and the recorded names come from the layer end to end.
//!
//! That change makes the name a recorded value for the first time, so the old record cannot
//! stay. Regenerate every data file exactly 1 time in the same change, review the new name of
//! every parameter of every case against the layer that gives it, and expect no other field of
//! any case to move. The regeneration guard makes that review a separate act: a data file that
//! changes needs the acknowledgment token of its own change set. See the regeneration section
//! below. Until stage 1 does that, no rename in a layer can reach this net.
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
//! 4 normalization layers expose 2 or more arrays of the same shape through `get_weights`, and
//! BatchNormalization exposes 4. A recorded shape alone therefore lets a later stage exchange 2
//! of those arrays with no failure. `get_weights` is the exact surface that the checkpoint
//! format reads, so such an exchange corrupts every saved model.
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
//! # The data file grammar
//!
//! A data file is text. A blank line carries nothing, and so does a line whose first non-blank
//! character is "#". The regeneration path is the 1 reader that looks inside a comment: it
//! carries the `# reason` lines of the header over into the file that it writes. See
//! [`previous_reasons`]. Every other line starts with a keyword:
//!
//! ```text
//! format 5                     the format version, 1 time, at the head of the file
//! family <name>                the family name, 1 time, at the head of the file
//! content_digest <digits> cases <count> values <count>
//!                              the digest of the file over itself, 1 time, before the cases
//! case <layer-type> <label>    starts a case
//! mode training|inference      the mode that the case ran in
//! output_shape "<text>"        what output_shape returned after the forward pass
//! param_count <class>          what param_count returned. See below
//! param_name <name> layer|fixture
//!                              the name of 1 parameter, and where the name came from
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
//! The `param_count` class is `trainable <count>`, `non_trainable <count>`, or `none`. The 3
//! forms are the 3 variants of `TrainingParameters`.
//!
//! The last word of a `param_name` line is `layer` when the layer supplied the name, and
//! `fixture` when the fixture supplied it. See the section above.
//!
//! A `weight` line and a `step_weight` line each end with the word `checksum` and 16 lowercase
//! hexadecimal digits. The digits are the value fingerprint of the array, from
//! [`weight_checksum`]. The whole-number words between the name and the word `checksum` are the
//! shape.
//!
//! A `tensor` line for a parameter gradient carries 1 more word after its shape, `decays` or
//! `no_decay`. That word is the `ParamGrad` decay class of the parameter. No other tensor line
//! carries the word.
//!
//! A case records 2 tensors per parameter. The name `grad_param.<name>` holds the gradient, and
//! the name `step_param.<name>` holds the parameter value after 1 optimizer step. Both tensors
//! hold the flat slice that `Layer::parameters` gives, so the shape is the element count alone.
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
use ndarray::{ArrayBase, ArrayD, Data, Dimension, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::traits::Layer;
use std::borrow::Cow;
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
/// Version 5 added the `content_digest` header line. Version 4 added the `param_name` line and
/// the `step_weight` line. Neither version moved a recorded value of the version before it.
const FORMAT_VERSION: u32 = 5;

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

/// The word that a `param_name` line uses when the layer supplied the name.
const LAYER_NAME_SOURCE: &str = "layer";

/// The word that a `param_name` line uses when the fixture supplied the name.
const FIXTURE_NAME_SOURCE: &str = "fixture";

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
/// and asserts that `predict` returns exactly the `forward` output.
pub struct GoldenCase {
    /// Short configuration label. It carries no whitespace, and it is unique per layer type
    label: &'static str,
    /// Shape of the input tensor, batch axis first
    input_shape: Vec<usize>,
    /// Name of every parameter gradient, in the order that `Layer::parameters` returns them
    parameter_names: Vec<&'static str>,
    /// Mode that the harness selects before it runs the layer
    training: bool,
    /// Whether the harness asserts that `predict` returns exactly the `forward` output
    predict_matches_forward: bool,
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
    ///   time for the `predict` pass and 1 more time for the forward and backward pass
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
            predict_matches_forward: true,
            build: Box::new(build),
        }
    }

    /// Names every parameter gradient, in the order that `Layer::parameters` returns them.
    ///
    /// Use the name that the layer gives the tensor in its `LayerWeight` variant, such as
    /// `"weight"`, `"bias"`, `"kernel"`, `"recurrent_kernel"`, `"gamma"`, `"beta"`, or
    /// `"alpha"`. The harness fails when the count differs from what `parameters` returns.
    pub fn with_parameter_grads(mut self, names: &[&'static str]) -> Self {
        self.parameter_names = names.to_vec();
        self
    }

    /// Runs the case in inference mode instead of training mode.
    ///
    /// The harness calls `set_training_if_mode_dependent(false)` before the forward pass. A
    /// layer that does not depend on the mode ignores the call.
    pub fn in_inference_mode(mut self) -> Self {
        self.training = false;
        self
    }

    /// Drops the assertion that `predict` returns exactly the `forward` output.
    ///
    /// Use this only for a mode-dependent layer in training mode, where the 2 paths differ by
    /// design. Every other case keeps the assertion.
    pub fn with_predict_that_differs(mut self) -> Self {
        self.predict_matches_forward = false;
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
/// - If `predict` differs from `forward` on a case that expects them to agree
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
    /// The `ParamGrad` decay class, for a parameter gradient alone. `None` for every other
    /// tensor. `Some(true)` is `ParamGrad::weight`, and `Some(false)` is `ParamGrad::no_decay`
    decays: Option<bool>,
    /// The value of every element, as the raw bits from `f32::to_bits`
    bits: Vec<u32>,
}

/// The name, the shape, and the value fingerprint of 1 weight that `Layer::get_weights` exposes.
///
/// A shape alone does not pin the identity of an array. 4 normalization layers expose 2 or more
/// arrays of the same shape, and BatchNormalization exposes 4. The fingerprint therefore rides
/// next to the shape, and a stage that exchanges 2 such arrays fails the comparison.
#[derive(Clone, PartialEq, Eq)]
struct RecordedWeight {
    /// The field name that the `LayerWeight` variant gives the array
    name: String,
    /// Shape in C order
    shape: Vec<usize>,
    /// The fingerprint of every value of the array, from [`weight_checksum`]
    checksum: u64,
}

/// 1 exposed weight, together with the storage that it borrows.
///
/// The record goes in the data file. The anchor never does. The harness uses the anchor to find
/// out which exposed array a parameter of `Layer::parameters` is, and it therefore uses the
/// anchor to take the name of that parameter from the layer. See [`resolve_parameter_name`].
struct ExposedWeight {
    /// What the data file holds about this weight
    record: RecordedWeight,
    /// The address of the first element and the element count, for an array that borrows the
    /// live layer and holds its elements in 1 contiguous run. `None` for every other array,
    /// which includes an array that the weight container owns
    anchor: Option<(usize, usize)>,
}

/// The name of 1 parameter of `Layer::parameters`, together with the source of that name.
#[derive(PartialEq, Eq)]
struct RecordedParameterName {
    /// The name that the tensors `grad_param.<name>` and `step_param.<name>` carry
    name: String,
    /// `true` when the layer supplied the name, and `false` when the fixture supplied it. See
    /// the module doc comment
    from_layer: bool,
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
    /// The name of every parameter of `Layer::parameters`, in the order that method returns
    /// them, and the source of each name
    param_names: Vec<RecordedParameterName>,
    /// The shape and the value fingerprint of every weight that `Layer::get_weights` exposes,
    /// in the order of the `LayerWeight` variant
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

    // The 2 passes use separate layers, so the `predict` pass can never disturb the state that
    // the forward and backward pass builds
    let mut inference_layer = (case.build)();
    inference_layer.set_training_if_mode_dependent(case.training);
    check_layer_type(layer_type, inference_layer.as_ref(), case.label);
    let predicted = inference_layer
        .predict(&input)
        .unwrap_or_else(|error| panic!("{layer_type}/{}: predict failed: {error}", case.label));

    let mut layer = (case.build)();
    layer.set_training_if_mode_dependent(case.training);
    let forward = layer
        .forward(&input)
        .unwrap_or_else(|error| panic!("{layer_type}/{}: forward failed: {error}", case.label));

    if case.predict_matches_forward {
        assert_eq!(
            forward.shape(),
            predicted.shape(),
            "{layer_type}/{}: predict and forward returned different shapes",
            case.label
        );
        for (index, (from_forward, from_predict)) in
            forward.iter().zip(predicted.iter()).enumerate()
        {
            assert_eq!(
                from_forward.to_bits(),
                from_predict.to_bits(),
                "{layer_type}/{}: predict differs from forward at flat index {index}: \
                 forward {from_forward:?}, predict {from_predict:?}",
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
    // anchors go no further than this function, and they give each parameter the name that the
    // layer holds for the same storage
    let exposed = exposed_weights(&layer.get_weights());
    let weights: Vec<RecordedWeight> = exposed.iter().map(|item| item.record.clone()).collect();
    let anchors: Vec<(String, usize, usize)> = exposed
        .iter()
        .filter_map(|item| {
            item.anchor
                .map(|(address, length)| (item.record.name.clone(), address, length))
        })
        .collect();

    let upstream = golden_gradient(forward.shape());
    let grad_input = layer
        .backward(&upstream)
        .unwrap_or_else(|error| panic!("{layer_type}/{}: backward failed: {error}", case.label));

    let mut tensors = vec![
        tensor_record("input", &input),
        tensor_record("forward", &forward),
        tensor_record("predict", &predicted),
        tensor_record("grad_input", &grad_input),
    ];

    // `Layer::parameters` yields flat slices in a stable order, which is exactly what an
    // optimizer consumes. The record keeps that flat form
    let mut parameters = layer.parameters();
    assert_eq!(
        parameters.len(),
        case.parameter_names.len(),
        "{layer_type}/{}: the case names {} parameter gradients, and the layer returned {}",
        case.label,
        case.parameter_names.len(),
        parameters.len()
    );
    let mut param_names: Vec<RecordedParameterName> = Vec::new();
    for (parameter, fixture_name) in parameters.iter_mut().zip(case.parameter_names.iter()) {
        // The layer owns the name whenever the parameter is 1 of the arrays that
        // `get_weights` exposes. The fixture name is the fallback, and the record says which
        let (name, from_layer) = resolve_parameter_name(&anchors, parameter.value, fixture_name);
        assert!(
            !param_names.iter().any(|earlier| earlier.name == name),
            "{layer_type}/{}: 2 parameters resolve to the name {name}",
            case.label
        );
        assert!(
            parameter.grad.len() <= MAX_TENSOR_ELEMENTS,
            "{layer_type}/{}: the gradient of {name} holds {} elements, and the cap is \
             {MAX_TENSOR_ELEMENTS}",
            case.label,
            parameter.grad.len()
        );
        assert_eq!(
            parameter.value.len(),
            parameter.grad.len(),
            "{layer_type}/{}: the parameter {name} holds {} values and {} gradient values, and \
             an optimizer needs the 2 slices to agree",
            case.label,
            parameter.value.len(),
            parameter.grad.len()
        );
        tensors.push(RecordedTensor {
            name: format!("{PARAMETER_PREFIX}{name}"),
            shape: vec![parameter.grad.len()],
            // The flag decides what an optimizer with a non-zero weight decay does to this
            // tensor, and it moves no gradient value, so the record must hold it
            decays: Some(parameter.decays),
            bits: parameter.grad.iter().map(|value| value.to_bits()).collect(),
        });
        // 1 step of the harness rule binds this gradient to the tensor that it updates. A
        // stage that aims the gradient at another tensor keeps every gradient value, and moves
        // this one. The caller discards the layer instance directly after `record_case`
        // returns, so no later record reads a stepped value
        for (value, gradient) in parameter.value.iter_mut().zip(parameter.grad.iter()) {
            *value -= OPTIMIZER_STEP * gradient;
        }
        tensors.push(RecordedTensor {
            name: format!("{STEPPED_PREFIX}{name}"),
            shape: vec![parameter.value.len()],
            // The matching gradient tensor already holds the decay class of this parameter
            decays: None,
            bits: parameter
                .value
                .iter()
                .map(|value| value.to_bits())
                .collect(),
        });
        param_names.push(RecordedParameterName { name, from_layer });
    }

    // The second fingerprint reads the arrays of the layer itself, and the step above wrote
    // through the slices that `parameters` handed out. An implementation of `parameters` that
    // gives back a copy of each value buffer therefore leaves every one of these unchanged,
    // while every `step_param` tensor above still holds the stepped numbers
    drop(parameters);
    let step_weights: Vec<RecordedWeight> = exposed_weights(&layer.get_weights())
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

/// Gives 1 parameter of `Layer::parameters` its name, and says where the name came from.
///
/// `ParamGrad` carries no name today, so the layer cannot name a parameter directly. The layer
/// does name every array that `get_weights` exposes. This function therefore takes the address
/// and the length of the parameter, and looks for the 1 exposed array that starts at the same
/// address and holds the same number of elements. Such an array is the same storage as the
/// parameter, so its name is the name of the parameter, and the layer supplied it.
///
/// A parameter that matches no exposed array, or that matches more than 1, keeps the name that
/// the fixture gave it.
///
/// # Parameters
///
/// - `anchors` - The name, the first-element address, and the element count of every exposed
///   array that borrows the live layer
/// - `value` - The parameter value slice that `Layer::parameters` handed out
/// - `fixture` - The name that the fixture gave this parameter
///
/// # Returns
///
/// - `(String, bool)` - The name, and `true` when the layer supplied it
fn resolve_parameter_name(
    anchors: &[(String, usize, usize)],
    value: &[f32],
    fixture: &str,
) -> (String, bool) {
    let address = value.as_ptr() as usize;
    let mut found: Option<&str> = None;
    for (name, anchor_address, length) in anchors {
        if *anchor_address == address && *length == value.len() {
            if found.is_some() {
                // 2 exposed arrays cannot be the same storage, so this says the anchor list is
                // not trustworthy. Fall back to the fixture name
                return (fixture.to_string(), false);
            }
            found = Some(name);
        }
    }
    match found {
        Some(name) => (name.to_string(), true),
        None => (fixture.to_string(), false),
    }
}

/// Turns a `param_count` result into the words that the data file holds.
fn param_count_record(count: TrainingParameters) -> String {
    match count {
        TrainingParameters::Trainable(total) => format!("trainable {total}"),
        TrainingParameters::NonTrainable(total) => format!("non_trainable {total}"),
        TrainingParameters::NoTrainable => "none".to_string(),
    }
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
/// - `array` - 1 array that `Layer::get_weights` exposes. A weight container holds arrays of
///   more than 1 dimension type, so the function takes every rank
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
/// An array with any other layout gives `None`. The caller passes only an array that borrows the
/// live layer, because the address of an owned array says nothing about the storage of the layer
/// and dies with the weight container.
///
/// # Parameters
///
/// - `array` - 1 array that a `LayerWeight` variant borrows from the layer
///
/// # Returns
///
/// - `Option<(usize, usize)>` - The first-element address and the element count, or `None`
fn contiguous_anchor<S, D>(array: &ArrayBase<S, D>) -> Option<(usize, usize)>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    let slice = array.as_slice()?;
    Some((slice.as_ptr() as usize, slice.len()))
}

/// Turns a `get_weights` result into the name, the shape, the fingerprint, and the storage
/// anchor of every array it exposes.
///
/// The order follows the fields of the `LayerWeight` variant, which is the order that a reader
/// of the enum sees. A layer with no trainable parameter gives the empty list.
///
/// The name of each array is a literal here, and the field access next to it is not. The
/// compiler binds `w.gamma` to the field `gamma` of the weight container of the layer, so a
/// stage that renames that field breaks this function instead of passing unseen.
fn exposed_weights(weights: &LayerWeight<'_>) -> Vec<ExposedWeight> {
    /// Builds the list from a name and array pair per exposed weight.
    macro_rules! shapes {
        ($($name:literal => $array:expr),+ $(,)?) => {
            vec![$(ExposedWeight {
                record: RecordedWeight {
                    name: $name.to_string(),
                    shape: $array.shape().to_vec(),
                    checksum: weight_checksum(&$array),
                },
                // An owned array is a copy that dies with this container, so it anchors nothing
                anchor: match &$array {
                    Cow::Borrowed(live) => contiguous_anchor(*live),
                    Cow::Owned(_) => None,
                },
            }),+]
        };
    }

    match weights {
        LayerWeight::Dense(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::SimpleRNN(w) => shapes!(
            "kernel" => w.kernel,
            "recurrent_kernel" => w.recurrent_kernel,
            "bias" => w.bias,
        ),
        LayerWeight::LSTM(w) => shapes!(
            "kernel" => w.kernel,
            "recurrent_kernel" => w.recurrent_kernel,
            "bias" => w.bias,
        ),
        LayerWeight::GRU(w) => shapes!(
            "kernel" => w.kernel,
            "recurrent_kernel" => w.recurrent_kernel,
            "bias" => w.bias,
        ),
        LayerWeight::Conv1D(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::Conv2D(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::Conv3D(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::Conv1DTranspose(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::Conv2DTranspose(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::Conv3DTranspose(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::DepthwiseConv1D(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::DepthwiseConv2D(w) => shapes!("weight" => w.weight, "bias" => w.bias),
        LayerWeight::SeparableConv1D(w) => shapes!(
            "depthwise_weight" => w.depthwise_weight,
            "pointwise_weight" => w.pointwise_weight,
            "bias" => w.bias,
        ),
        LayerWeight::SeparableConv2D(w) => shapes!(
            "depthwise_weight" => w.depthwise_weight,
            "pointwise_weight" => w.pointwise_weight,
            "bias" => w.bias,
        ),
        LayerWeight::BatchNormalization(w) => shapes!(
            "gamma" => w.gamma,
            "beta" => w.beta,
            "running_mean" => w.running_mean,
            "running_var" => w.running_var,
        ),
        LayerWeight::LayerNormalization(w) => shapes!("gamma" => w.gamma, "beta" => w.beta),
        LayerWeight::InstanceNormalization(w) => shapes!("gamma" => w.gamma, "beta" => w.beta),
        LayerWeight::GroupNormalization(w) => shapes!("gamma" => w.gamma, "beta" => w.beta),
        LayerWeight::Embedding(w) => shapes!("embeddings" => w.embeddings),
        LayerWeight::PReLU(w) => shapes!("alpha" => w.alpha),
        LayerWeight::Empty => Vec::new(),
    }
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

/// The word that a `param_name` line uses for the source of a parameter name.
fn name_source_word(from_layer: bool) -> &'static str {
    if from_layer {
        LAYER_NAME_SOURCE
    } else {
        FIXTURE_NAME_SOURCE
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
        for parameter in &case.param_names {
            writeln!(
                text,
                "param_name {} {}",
                parameter.name,
                name_source_word(parameter.from_layer)
            )
            .unwrap();
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
                if version != FORMAT_VERSION {
                    fail(
                        line,
                        format!("format {version} is not the supported format {FORMAT_VERSION}"),
                    );
                }
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
                let class = words.next().unwrap_or("");
                let count = words.next();
                let valid = match (class, count) {
                    ("none", None) => true,
                    ("trainable" | "non_trainable", Some(total)) => total.parse::<usize>().is_ok(),
                    _ => false,
                };
                if !valid || words.next().is_some() {
                    fail(
                        line,
                        "param_count is trainable <count>, non_trainable <count>, or none"
                            .to_string(),
                    );
                }
                case.param_count = match count {
                    Some(total) => format!("{class} {total}"),
                    None => class.to_string(),
                };
                param_count_seen = true;
            }
            "param_name" => {
                let Some(case) = open.as_mut() else {
                    fail(line, "param_name appears outside a case".into());
                };
                let name = words
                    .next()
                    .unwrap_or_else(|| fail(line, "param_name needs a name".into()));
                let from_layer = match words.next() {
                    Some(LAYER_NAME_SOURCE) => true,
                    Some(FIXTURE_NAME_SOURCE) => false,
                    other => fail(
                        line,
                        format!(
                            "the name source of {name} is {other:?}, not \
                             {LAYER_NAME_SOURCE} or {FIXTURE_NAME_SOURCE}"
                        ),
                    ),
                };
                if words.next().is_some() {
                    fail(line, "param_name is a name and then a source".into());
                }
                case.param_names.push(RecordedParameterName {
                    name: name.to_string(),
                    from_layer,
                });
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

/// Compares the parameter roster and the source of every parameter name, and appends every
/// problem it finds.
///
/// A rename in the parameter store of a later stage lands here first. So does a parameter name
/// that stops coming from the layer, which says that the parameter is no longer the storage
/// that `get_weights` exposes.
fn compare_param_names(
    head: &str,
    recorded: &RecordedCase,
    stored: &RecordedCase,
    problems: &mut Vec<String>,
) {
    let describe = |names: &[RecordedParameterName]| -> Vec<String> {
        names
            .iter()
            .map(|parameter| {
                format!(
                    "{} ({})",
                    parameter.name,
                    name_source_word(parameter.from_layer)
                )
            })
            .collect()
    };
    if recorded.param_names != stored.param_names {
        problems.push(format!(
            "{head}: the parameters are {:?} now, and the data file records {:?}. A name marked \
             {LAYER_NAME_SOURCE:?} comes from the layer, and a name marked \
             {FIXTURE_NAME_SOURCE:?} comes from the fixture alone",
            describe(&recorded.param_names),
            describe(&stored.param_names)
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
                "{head}, {keyword} {}: get_weights exposes this weight, and the data file holds \
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
                 records {:0width$x}. Either the values behind get_weights changed, or 2 \
                 exposed arrays of the same shape changed places",
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
                "{head}, {keyword} {}: the data file holds this weight, and get_weights no \
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
    // step wrote through the slices of `Layer::parameters`, so a store that gives back a copy
    // of each value buffer keeps every one of these at its value before the step
    if problems.len() > before && keyword == "step_weight" {
        problems.push(format!(
            "{head}: a {keyword} fingerprint holds the weight after 1 step of \
             param -= {OPTIMIZER_STEP} * grad, and get_weights reads the arrays of the layer \
             itself. Check whether Layer::parameters hands out the storage of the layer or a \
             copy of it. A copy makes every step_param tensor agree and leaves the layer \
             unchanged, which is training that does nothing"
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
                    param_count: "trainable 6".to_string(),
                    param_names: vec![RecordedParameterName {
                        name: "weight".to_string(),
                        from_layer: true,
                    }],
                    weights: vec![RecordedWeight {
                        name: "weight".to_string(),
                        shape: vec![2, 3],
                        checksum: 0x0123_4567_89ab_cdef + u64::from(seed),
                    }],
                    step_weights: vec![RecordedWeight {
                        name: "weight".to_string(),
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
                            name: "grad_param.weight".to_string(),
                            shape: vec![6],
                            decays: Some(true),
                            bits: bits(6),
                        },
                        RecordedTensor {
                            name: "step_param.weight".to_string(),
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
