# Guide maintenance

The guide contains two mdBook projects.
`en` contains the English source.
`zh-Hans` contains its Simplified Chinese translation.
Both books use the same chapter filenames, code examples, and output blocks.

## Build the books

1. Run `mdbook build guide/en` from the repository root.
2. Run `mdbook build guide/zh-Hans` from the repository root.
3. Copy `guide/index.html` to `guide/book/index.html` for the language selector.

The deployment workflow uses mdBook 0.5.4.
The local book build also works with mdBook 0.5.2.
The build writes HTML under `guide/book`.

## Run the examples

From the repository root, run one source example with this command:

```sh
cargo run --manifest-path guide/examples/Cargo.toml --bin first_model
```

The example package uses the local RustyML source and ndarray 0.17.2.
Its other binary names match the Rust filenames in `guide/examples`.

To verify every printed output, run this command:

```sh
python3 guide/check_examples.py
```

If Cargo has cached all dependencies, add `--offline`.
The checker requires Python 3 and Cargo.
It compiles every Rust block from the English book against the local crate.
It compares program output with the next `text` block.
Text, integers, shapes, labels, and boolean values require exact matches.
Selected continuous numeric lines use explicit tolerances in `FLOAT_OUTPUT_LINES`.
Other lines require exact matches, including labels printed as floats.
It also checks translated blocks and the separate source examples for exact matches.
The checker stores build files under `target/guide-check` unless `CARGO_TARGET_DIR` specifies another directory.

## Float tolerances and CI

The main CI workflow checks examples on Linux and macOS for pushes and pull requests targeting `master`.
It also tests the output comparison rules.

The checker uses a relative tolerance of `1e-6` for selected float output.
Each selected line also has an absolute tolerance.
Three-decimal scores allow `1e-3`, one unit in the last printed decimal place.
This allowance handles small differences across a rounding boundary.
The six-decimal error output allows `1e-6`.
Simple exact regression examples also use `1e-6`, despite their shorter printed output.

Two finite values match when their absolute difference does not exceed either tolerance limit.
The relative limit equals `1e-6` times the larger absolute value.
NaN, infinity, changed text, and changed integer tokens fail comparison.
Translated code and documented output must still match exactly.

For new numeric output, select the required precision before adding a line rule.
Use more decimal places when a smaller tolerance matters.
Do not widen tolerances to hide training failures or changed model behavior.
Example assertions still check properties such as lower loss and unchanged predictions after loading.

Run the comparison tests with this command:

```sh
python3 -m unittest discover -s guide -p 'test_check_examples.py'
```

## Update a chapter

1. Check the current public API and its input requirements.
2. Write the English chapter with the STE rules below.
3. Run each complete example against the local crate.
4. Copy its actual output into the chapter.
5. Translate the English chapter into Simplified Chinese.
6. Preserve code and output blocks exactly.
7. Run the example checker.
8. Build both books.

Keep the guide about public API use.
Explain input shapes, method order, returned values, and common errors.
Link to the API reference for details outside the guide scope.

## English writing rules

Use active voice and short, common words.
Use one name for each concept.
Keep each sentence within 20 words.
Keep each paragraph about one topic, with no more than six sentences.
Give one instruction per sentence.
Use numbered steps for procedures.
Do not use contractions, semicolons, em dashes, or marketing adjectives in prose.
Use simple verbs for actions.
These rules do not change code syntax, identifiers, commands, or actual output.
