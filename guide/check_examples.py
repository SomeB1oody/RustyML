#!/usr/bin/env python3
"""Run book examples and compare their output with both translations."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


GUIDE = Path(__file__).resolve().parent
FENCE = re.compile(r"^([ \t]*)```([^\n]*)\n(.*?)^\1```[ \t]*$", re.M | re.S)


# Set tolerances only for continuous numeric output. Labels remain exact.
FLOAT_OUTPUT_LINES = {
    "installation.md": {"prediction:": 1e-6},
    "errors.md": {"prediction:": 1e-6},
    "first-model.md": {"test MAE:": 1e-6, "prediction for 6:": 1e-6},
    "model-files.md": {"restored prediction:": 1e-6},
    "supervised.md": {"regression:": 1e-6, "positive probabilities:": 1e-3},
    "evaluation.md": {
        "MSE:": 1e-3,
        "RMSE:": 1e-3,
        "MAE:": 1e-3,
        "R2:": 1e-3,
        "accuracy:": 1e-3,
        "precision:": 1e-3,
        "ROC AUC:": 1e-3,
        "macro F1:": 1e-3,
        "ARI:": 1e-3,
        "NMI:": 1e-3,
        "silhouette:": 1e-3,
    },
}
REL_TOL = 1e-6
NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)


def numeric_line_matches(actual, expected, abs_tol):
    """Compare finite floats with tolerance and all other characters exactly."""
    actual_numbers = list(NUMBER.finditer(actual))
    expected_numbers = list(NUMBER.finditer(expected))
    if len(actual_numbers) != len(expected_numbers):
        return False
    actual_end = expected_end = 0
    has_float = False
    for found, wanted in zip(actual_numbers, expected_numbers):
        if actual[actual_end:found.start()] != expected[expected_end:wanted.start()]:
            return False
        actual_token, expected_token = found.group(), wanted.group()
        if any(character in expected_token.lower() for character in ".e"):
            has_float = True
            # Do not turn an integer count into a floating-point result.
            if not any(character in actual_token.lower() for character in ".e"):
                return False
            actual_value, expected_value = Decimal(actual_token), Decimal(expected_token)
            if not (actual_value.is_finite() and expected_value.is_finite()):
                return False
            limit = max(
                Decimal(str(abs_tol)),
                Decimal(str(REL_TOL)) * max(abs(actual_value), abs(expected_value)),
            )
            if abs(actual_value - expected_value) > limit:
                return False
        elif actual_token != expected_token:
            return False
        actual_end, expected_end = found.end(), wanted.end()
    return has_float and actual[actual_end:] == expected[expected_end:]


def output_matches(actual, expected, chapter):
    """Use chapter-specific float tolerances and exact comparison elsewhere."""
    actual_lines = actual.splitlines(keepends=True)
    expected_lines = expected.splitlines(keepends=True)
    if len(actual_lines) != len(expected_lines):
        return False
    rules = FLOAT_OUTPUT_LINES.get(chapter, {})
    for found, wanted in zip(actual_lines, expected_lines):
        tolerance = next((value for prefix, value in rules.items() if wanted.startswith(prefix)), None)
        if tolerance is None:
            if found != wanted:
                return False
        elif not numeric_line_matches(found, wanted, tolerance):
            return False
    return True


def blocks(path):
    return [(match[2].strip(), match[3]) for match in FENCE.finditer(path.read_text())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached Cargo dependencies only.")
    args = parser.parse_args()
    snippets = []
    examples = {path.read_text(): path.name for path in (GUIDE / "examples").glob("*.rs")}
    matched_sources = set()
    english_files = sorted((GUIDE / "en/src").glob("*.md"))
    chinese_files = sorted((GUIDE / "zh-Hans/src").glob("*.md"))
    if {path.name for path in english_files} != {path.name for path in chinese_files}:
        raise SystemExit("The books have different chapter files.")
    for english in english_files:
        content = blocks(english)
        chinese = GUIDE / "zh-Hans/src" / english.name
        if content != blocks(chinese):
            raise SystemExit(f"Code or output differs between translations: {english.name}")
        for index, (language, source) in enumerate(content):
            if not language.startswith("rust"):
                continue
            if language != "rust" or "fn main(" not in source:
                raise SystemExit(f"Expected a complete Rust program: {english.name}, block {index}")
            if index + 1 >= len(content) or content[index + 1][0] != "text":
                raise SystemExit(f"Missing example output: {english.name}, block {index}")
            if source in examples:
                matched_sources.add(examples[source])
            snippets.append((english.name, source, content[index + 1][1]))
    if matched_sources != set(examples.values()):
        raise SystemExit(f"Example sources differ from the books: {set(examples.values()) - matched_sources}")
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(
        Path(environment.get("CARGO_TARGET_DIR", GUIDE.parent / "target/guide-check")).resolve()
    )
    with tempfile.TemporaryDirectory(prefix="rustyml-book-") as temporary:
        project = Path(temporary)
        binaries = project / "src/bin"
        binaries.mkdir(parents=True)
        manifest = (
            '[package]\nname = "rustyml-book-check"\nversion = "0.1.0"\nedition = "2024"\n'
            '[dependencies]\n'
            f'rustyml = {{ path = {json.dumps(str(GUIDE.parent))} }}\n'
            'ndarray = "=0.17.2"\n'
        )
        (project / "Cargo.toml").write_text(manifest)
        for index, (_, source, _) in enumerate(snippets):
            (binaries / f"example_{index}.rs").write_text(source)
        command = ["cargo", "build", "--quiet", "--bins", "--manifest-path", str(project / "Cargo.toml")]
        if args.offline:
            command.append("--offline")
        subprocess.run(command, check=True, env=environment)
        for index, (chapter, _, expected) in enumerate(snippets):
            suffix = ".exe" if os.name == "nt" else ""
            executable = Path(environment["CARGO_TARGET_DIR"]) / "debug" / f"example_{index}{suffix}"
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, cwd=project)
            if not output_matches(result.stdout, expected, chapter):
                raise SystemExit(f"Output differs: {chapter}\nExpected:\n{expected}\nActual:\n{result.stdout}")
            print(f"PASS {chapter} (example {index + 1})")
    print(f"Verified {len(snippets)} programs and matching translations.")


if __name__ == "__main__":
    main()
