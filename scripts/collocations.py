#!/usr/bin/env python3
"""Create CoBaLD collocation score tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cobaldclusterisation.collocations import (
    CollocationConfig,
    run_collocation_experiment,
)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create bigram and trigram collocation score tables for CoBaLD."
    )
    parser.add_argument("--data-dir", default="CobaldRus")
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument(
        "--token-bases",
        default="lemma,form",
        help="Comma-separated token bases: lemma, form, or both.",
    )
    parser.add_argument(
        "--ngram-sizes",
        default="2,3",
        help="Comma-separated n-gram sizes. Supported values: 2,3.",
    )
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--sort-by", default="pmi")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="Preserve source capitalization in token text.",
    )
    parser.add_argument(
        "--output",
        default="outputs/collocations/cobald_collocations.xlsx",
        help="Output .xlsx workbook path.",
    )
    parser.set_defaults(lowercase=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = CollocationConfig(
        data_dir=args.data_dir,
        splits=_parse_csv(args.splits),
        token_bases=_parse_csv(args.token_bases),
        ngram_sizes=_parse_int_csv(args.ngram_sizes),
        min_freq=args.min_freq,
        sort_by=args.sort_by,
        max_rows=args.max_rows,
        lowercase=args.lowercase,
        output_path=Path(args.output),
    )
    result = run_collocation_experiment(config)
    print(f"sentences={len(result['sentences'])}")
    for name, table in result["tables"].items():
        print(f"{name}_rows={len(table)}")
    print(f"wrote={result['output_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
