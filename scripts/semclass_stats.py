#!/usr/bin/env python3
"""Create SEMCLASS frequency statistics for the CoBaLD corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cobaldclusterisation.semclass_stats import (
    semclass_table_to_markdown,
    semclass_threshold_table_from_corpus,
)


def _parse_splits(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cumulative table showing how many SEMCLASS labels are "
            "represented by at least N non-punctuation surface tokens."
        )
    )
    parser.add_argument("--data-dir", default="CobaldRus")
    parser.add_argument(
        "--hierarchy",
        default="semantic-hierarchy/hyperonims_hierarchy.csv",
        help=(
            "Path to hyperonims_hierarchy.csv or to a cloned semantic-hierarchy "
            "repository directory."
        ),
    )
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--max-occurrences", type=int, default=15)
    parser.add_argument(
        "--output",
        help="Optional output path. Supported suffixes: .md, .csv, .xlsx.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    table = semclass_threshold_table_from_corpus(
        args.data_dir,
        splits=_parse_splits(args.splits),
        max_occurrences=args.max_occurrences,
        hierarchy=args.hierarchy,
    )

    if args.output is None:
        print(semclass_table_to_markdown(table))
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".md":
        output_path.write_text(
            semclass_table_to_markdown(table) + "\n",
            encoding="utf-8",
        )
    elif suffix == ".csv":
        table.to_csv(output_path, index=False)
    elif suffix == ".xlsx":
        table.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported output suffix. Use .md, .csv, or .xlsx.")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
