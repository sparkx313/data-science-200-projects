"""Analyze Titanic survival patterns from a passenger CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


class InputError(ValueError):
    """Raised when the supplied passenger data cannot be analyzed."""


def parse_outcome(value: str | None) -> int | None:
    normalized = (value or "").strip().casefold()
    if normalized in {"1", "1.0", "yes", "true", "survived"}:
        return 1
    if normalized in {"0", "0.0", "no", "false", "died", "dead"}:
        return 0
    return None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze_csv(input_path: Path, target_column: str = "Survived") -> dict:
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise InputError(f"Input file does not exist: {source}")

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(8192)
            if not sample:
                raise InputError("Input CSV is empty.")
            file.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error as exc:
                raise InputError("Could not detect the CSV separator; use a standard comma, semicolon, tab, or pipe-delimited file.") from exc

            reader = csv.DictReader(file, dialect=dialect, restkey="__extra_fields__")
            original_headers = reader.fieldnames
            if not original_headers:
                raise InputError("CSV has no header row.")
            canonical_to_original = {}
            for header in original_headers:
                if header is None:
                    continue
                canonical = header.strip().casefold()
                if canonical in canonical_to_original:
                    raise InputError(f"Duplicate column name after trimming and case-folding: {header}")
                canonical_to_original[canonical] = header

            target_key = target_column.strip().casefold()
            if target_key not in canonical_to_original:
                raise InputError(
                    f"Required outcome column {target_column!r} not found. "
                    f"Available columns: {', '.join(original_headers)}"
                )
            target_header = canonical_to_original[target_key]

            rows_read = 0
            valid_rows = 0
            invalid_outcomes = 0
            survived = 0
            groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
            age_by_outcome: dict[str, list[float]] = defaultdict(list)
            fare_by_outcome: dict[str, list[float]] = defaultdict(list)
            missing_counts = {header: 0 for header in original_headers}
            numeric_invalid = {"Age": 0, "Fare": 0}
            numeric_all: dict[str, list[float]] = {"Age": [], "Fare": []}

            for row_number, row in enumerate(reader, start=2):
                rows_read += 1
                if row.get("__extra_fields__") is not None:
                    raise InputError(f"Row {row_number} contains more values than the header.")
                for header in original_headers:
                    if row.get(header) is None or not str(row.get(header)).strip():
                        missing_counts[header] += 1

                outcome = parse_outcome(row.get(target_header))
                if outcome is None:
                    invalid_outcomes += 1
                    continue
                valid_rows += 1
                survived += outcome
                outcome_name = "survived" if outcome else "did_not_survive"

                for dimension in ("Sex", "Pclass", "Embarked"):
                    header = canonical_to_original.get(dimension.casefold())
                    if header:
                        value = (row.get(header) or "").strip() or "Missing"
                        groups[dimension][value].append(outcome)

                age_header = canonical_to_original.get("age")
                age = _number(row.get(age_header)) if age_header else None
                age_group = "Missing"
                if age is not None and age >= 0:
                    if age < 18:
                        age_group = "Under 18"
                    elif age < 65:
                        age_group = "18 to 64"
                    else:
                        age_group = "65 and over"
                if age_header:
                    groups["Age group"][age_group].append(outcome)

                for field, bucket in (("Age", age_by_outcome), ("Fare", fare_by_outcome)):
                    header = canonical_to_original.get(field.casefold())
                    if header:
                        value = age if field == "Age" else _number(row.get(header))
                        if value is None:
                            numeric_invalid[field] += 1
                        else:
                            numeric_all[field].append(value)
                            bucket[outcome_name].append(value)

    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(f"Could not read CSV: {exc}") from exc

    if rows_read == 0:
        raise InputError("CSV contains a header but no passenger rows.")
    if valid_rows == 0:
        raise InputError(f"No usable values found in outcome column {target_column!r}.")

    group_report = {}
    for dimension, values in groups.items():
        group_report[dimension] = {}
        for label, outcomes in sorted(values.items(), key=lambda item: item[0].casefold()):
            group_report[dimension][label] = {
                "passengers": len(outcomes),
                "survived": sum(outcomes),
                "survival_rate": sum(outcomes) / len(outcomes),
                "confidence_interval_95": wilson_interval(sum(outcomes), len(outcomes)),
            }

    numeric_report = {}
    for field in ("Age", "Fare"):
        header = canonical_to_original.get(field.casefold())
        if not header:
            continue
        numeric_report[field] = {
            "all_valid_outcomes": numeric_summary(numeric_all[field]),
            "by_outcome": {
                label: numeric_summary(values)
                for label, values in sorted(age_by_outcome.items()) if field == "Age"
            } if field == "Age" else {
                label: numeric_summary(values)
                for label, values in sorted(fare_by_outcome.items())
            },
            "unparseable_or_missing": numeric_invalid[field],
        }

    return {
        "project": "Titanic Survival Data Exploration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source),
        "rows_read": rows_read,
        "rows_with_valid_outcome": valid_rows,
        "rows_with_unrecognized_outcome": invalid_outcomes,
        "survival": {
            "survived": survived,
            "did_not_survive": valid_rows - survived,
            "rate": survived / valid_rows,
            "confidence_interval_95": wilson_interval(survived, valid_rows),
        },
        "group_survival": group_report,
        "numeric_features": numeric_report,
        "missing_values": missing_counts,
        "notes": [
            "Descriptive associations do not establish that a feature caused survival.",
            "Rows with missing or unrecognized outcome values are excluded from survival rates.",
            "Confidence intervals use the Wilson method and describe sampling uncertainty.",
        ],
    }


def _number(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None and value.strip() else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def write_report(report: dict, output_path: Path, input_path: Path) -> None:
    destination = output_path.expanduser().resolve()
    source = input_path.expanduser().resolve()
    if destination == source:
        raise InputError("The JSON report path must be different from the input CSV.")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise InputError(f"Could not write report: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure survival rates and passenger patterns from a Titanic CSV.")
    parser.add_argument("--input", required=True, type=Path, help="passenger CSV file")
    parser.add_argument("--target-column", default="Survived", help="binary outcome column (default: Survived)")
    parser.add_argument("--output", type=Path, default=Path("titanic_survival_report.json"))
    args = parser.parse_args()
    try:
        report = analyze_csv(args.input, args.target_column)
        write_report(report, args.output, args.input)
    except InputError as exc:
        parser.exit(2, f"Error: {exc}\n")
    survival = report["survival"]
    print(f"Rows analyzed: {report['rows_with_valid_outcome']} of {report['rows_read']}")
    print(f"Survival rate: {survival['rate']:.1%}")
    print(f"95% Wilson interval: {survival['confidence_interval_95'][0]:.1%}–{survival['confidence_interval_95'][1]:.1%}")
    print(f"Report saved to {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
