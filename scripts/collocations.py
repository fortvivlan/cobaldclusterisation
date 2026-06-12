#!/usr/bin/env python3
"""Create CoBaLD collocation score tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cobaldclusterisation.collocations import (
    CollocationClusterConfig,
    CollocationConfig,
    run_collocation_cluster_experiment,
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
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help=(
            "Minimum n-gram count for the collocation score workbook. The "
            "cluster workbook enforces at least 5."
        ),
    )
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
    parser.add_argument(
        "--artifact",
        help=(
            "Optional clustering *_artifacts.pkl path. When provided, also "
            "writes a collocation cluster-overlap workbook."
        ),
    )
    parser.add_argument(
        "--cluster-output",
        help=(
            "Optional output path for the collocation cluster-overlap workbook. "
            "Defaults next to the artifact in the results folder."
        ),
    )
    parser.add_argument(
        "--max-cluster-examples",
        type=int,
        default=300,
        help=(
            "Compatibility option. The simplified cluster workbook no longer "
            "writes exact occurrence example sheets."
        ),
    )
    parser.add_argument(
        "--no-cluster-plots",
        dest="cluster_plots",
        action="store_false",
        help="Do not write optional collocation cluster-overlap plot PNGs.",
    )
    parser.set_defaults(cluster_plots=True)
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
    if args.artifact:
        cluster_config = CollocationClusterConfig(
            data_dir=args.data_dir,
            splits=_parse_csv(args.splits),
            token_bases=_parse_csv(args.token_bases),
            ngram_sizes=_parse_int_csv(args.ngram_sizes),
            min_freq=args.min_freq,
            sort_by=args.sort_by,
            max_rows=args.max_rows,
            lowercase=args.lowercase,
            artifact_path=args.artifact,
            output_path=args.cluster_output,
            max_examples_per_run=args.max_cluster_examples,
            create_plots=args.cluster_plots,
        )
        cluster_result = run_collocation_cluster_experiment(cluster_config)
        print(f"cluster_overlap_wrote={cluster_result['output_path']}")
        for plot_path in cluster_result["plot_paths"]:
            print(f"cluster_overlap_plot={plot_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
