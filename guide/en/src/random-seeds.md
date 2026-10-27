# Random seeds

A seed controls random choices in an operation.
Use explicit local seeds when an API accepts them.
For example, `train_test_split` accepts `Some(42)` as its last argument.

## Use the global seed

`set_global_seed` sets a seed stream for the current thread.
Operations without local seeds can use this stream.
Reset the seed before you repeat the same sequence of randomized operations.

```rust
use ndarray::array;
use rustyml::error::RustymlResult;
use rustyml::utils::train_test_split::train_test_split;
use rustyml::{clear_global_seed, set_global_seed};

fn main() -> RustymlResult<()> {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    set_global_seed(42);
    let first = train_test_split(x.clone(), y.clone(), Some(0.33), None)?;
    set_global_seed(42);
    let second = train_test_split(x, y, Some(0.33), None)?;
    println!("same test rows: {}", first.1 == second.1);
    clear_global_seed();
    Ok(())
}
```

Output:

```text
same test rows: true
```

An explicit local seed takes priority over the global seed.
It does not advance the global seed stream.
Without either seed, operations that need randomness can use system entropy.

The global seed does not affect generators that already exist.
Other threads have separate seed streams.
A different operation order can assign different seeds to unseeded operations.
Call `clear_global_seed` when subsequent operations must not use the stream.

## Neural networks and solvers

Use `SequentialBuilder::new_with_seed(42)` before you build a neural network.
It controls the batch shuffle seed.
Set randomized layer seeds with `with_random_state` before building.
Use `model.set_seed(42)` to control later randomized training operations.
It does not reset existing weights.

Some solvers expose their own seed settings.
For example, `SVDSolver::Randomized` accepts an explicit seed.
The global seed does not replace this setting.

Seeds help repeat experiments with the same data, settings, and operation order.
Do not expect identical floating-point results across all platforms or library versions.
