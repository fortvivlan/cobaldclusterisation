#!/usr/bin/env python3
"""Analyze identical CoBaLD forms with different SEMCLASS labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cobaldclusterisation.form_semclass_similarity import (
    DEFAULT_ARTIFACT_PATH,
    FormSemclassSimilarityConfig,
    run_form_semclass_similarity_experiment,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare embeddings and cluster assignments for case-insensitive "
            "identical CoBaLD forms with different SEMCLASS labels."
        )
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--embeddings",
        help=(
            "Aligned embedding payload pickle. If omitted, the script first checks "
            "artifact metadata and then the Drive root for rubert_tiny2_cobald.pkl."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output .xlsx workbook path. Defaults next to the artifact pickle.",
    )
    parser.add_argument("--max-examples-per-run", type=int, default=200)
    parser.add_argument("--examples-per-form", type=int, default=4)
    parser.add_argument(
        "--no-plots",
        dest="create_plots",
        action="store_false",
        help="Skip optional matplotlib PNG scatter plots.",
    )
    parser.set_defaults(create_plots=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = FormSemclassSimilarityConfig(
        artifact_path=Path(args.artifact),
        embeddings_path=Path(args.embeddings) if args.embeddings else None,
        output_path=Path(args.output) if args.output else None,
        max_examples_per_run=args.max_examples_per_run,
        examples_per_form=args.examples_per_form,
        create_plots=args.create_plots,
    )
    result = run_form_semclass_similarity_experiment(config)
    print(f"artifact={result['artifact_path']}")
    print(f"embeddings={result['embeddings_path']}")
    print(f"wrote={result['output_path']}")
    for plot_path in result["plot_paths"]:
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
