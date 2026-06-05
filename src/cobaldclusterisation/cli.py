"""Command-line entry points for repeatable CoBaLD experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .clustering import (
    run_clustering_suite,
    save_clustering_results,
    scores_to_dataframe,
)
from .data import corpus_to_dataframe, load_corpus
from .embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
    load_embeddings_pickle,
)
from .rubert_baseline import build_cluster_configs


def _parse_splits(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _cmd_parse(args: argparse.Namespace) -> None:
    sentences = load_corpus(args.data_dir, splits=_parse_splits(args.splits))
    token_table = corpus_to_dataframe(sentences)
    target_table = corpus_to_dataframe(
        sentences,
        include_punctuation=False,
        include_empty=False,
    )
    print(f"sentences={len(sentences)}")
    print(f"token_rows={len(token_table)}")
    print(f"embedding_targets={len(target_table)}")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        token_table.to_csv(output_path, index=False)
        print(f"wrote={output_path}")


def _cmd_embed(args: argparse.Namespace) -> None:
    config = EmbeddingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layer=args.layer,
        device=args.device,
        seed=args.seed,
        normalize=args.normalize,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        include_punctuation_context=args.include_punctuation_context,
    )
    payload = generate_embeddings_from_corpus(
        data_dir=args.data_dir,
        output_path=args.output,
        drive_dir=args.drive_dir,
        splits=_parse_splits(args.splits),
        config=config,
        show_progress=not args.no_progress,
    )
    print(f"embeddings_shape={payload['embeddings'].shape}")
    print(f"tokens={len(payload['tokens'])}")
    print(f"saved={payload['saved_path']}")


def _parse_n_clusters(values: list[str]) -> int | str | list[int]:
    if len(values) == 1:
        value = values[0]
        if value == "data_semclass":
            return value
        return int(value)
    return [int(value) for value in values]


def _cluster_configs(args: argparse.Namespace, tokens: pd.DataFrame) -> list[object]:
    return build_cluster_configs(
        algorithms=args.algorithms,
        n_clusters=_parse_n_clusters(args.n_clusters),
        tokens=tokens,
        random_state=args.seed,
        normalize=args.normalize,
        batch_size=args.batch_size,
        n_init=args.n_init,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        hdbscan_n_jobs=args.n_jobs,
        metric=args.metric,
        birch_threshold=args.birch_threshold,
        birch_branching_factor=args.birch_branching_factor,
        bisecting_strategy=args.bisecting_strategy,
        graph_n_neighbors=args.graph_n_neighbors,
        graph_resolution=args.graph_resolution,
        graph_metric=args.graph_metric,
        graph_n_jobs=args.n_jobs,
    )


def _cmd_cluster(args: argparse.Namespace) -> None:
    hierarchy = args.hierarchy if args.hierarchy else None
    payload = load_embeddings_pickle(args.embeddings)
    results = run_clustering_suite(
        payload,
        configs=_cluster_configs(args, payload["tokens"]),
        hierarchy=hierarchy,
        hierarchy_depths=args.hierarchy_depths,
        show_progress=not args.no_progress,
    )
    output_path = save_clustering_results(results, args.output)
    scores_path = output_path.with_suffix(".scores.csv")
    scores_to_dataframe(results).to_csv(scores_path, index=False)
    for index, result in enumerate(results):
        summary_path = output_path.with_suffix(f".summary_{index}.csv")
        summary = result["summary"]
        if isinstance(summary, pd.DataFrame):
            summary.to_csv(summary_path, index=False)
        alignment_path = output_path.with_suffix(f".alignment_{index}.csv")
        alignment = result.get("hierarchy_alignment")
        if isinstance(alignment, pd.DataFrame) and not alignment.empty:
            alignment.to_csv(alignment_path, index=False)
    print(f"runs={len(results)}")
    print(f"saved={output_path}")
    print(f"scores={scores_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cobald")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse corpus and print counts.")
    parse_parser.add_argument("--data-dir", default="CobaldRus")
    parse_parser.add_argument("--splits", default="train,dev")
    parse_parser.add_argument("--output")
    parse_parser.set_defaults(func=_cmd_parse)

    embed_parser = subparsers.add_parser("embed", help="Generate contextual embeddings.")
    embed_parser.add_argument("--data-dir", default="CobaldRus")
    embed_parser.add_argument("--splits", default="train,dev")
    embed_parser.add_argument("--output", default="outputs/embeddings/rubert_tiny2.pkl")
    embed_parser.add_argument("--drive-dir")
    embed_parser.add_argument("--model-name", default="cointegrated/rubert-tiny2")
    embed_parser.add_argument("--batch-size", type=int, default=8)
    embed_parser.add_argument("--max-length", type=int, default=512)
    embed_parser.add_argument("--layer", type=int, default=-1)
    embed_parser.add_argument("--device")
    embed_parser.add_argument("--seed", type=int, default=42)
    embed_parser.add_argument("--normalize", action="store_true")
    embed_parser.add_argument(
        "--no-punctuation-context",
        dest="include_punctuation_context",
        action="store_false",
        help="Remove punctuation tokens from transformer context.",
    )
    embed_parser.set_defaults(include_punctuation_context=True)
    embed_parser.add_argument("--trust-remote-code", action="store_true")
    embed_parser.add_argument("--torch-dtype")
    embed_parser.add_argument("--no-progress", action="store_true")
    embed_parser.set_defaults(func=_cmd_embed)

    cluster_parser = subparsers.add_parser("cluster", help="Cluster saved embeddings.")
    cluster_parser.add_argument("embeddings")
    cluster_parser.add_argument("--output", default="outputs/clusters/results.pkl")
    cluster_parser.add_argument("--hierarchy")
    cluster_parser.add_argument("--hierarchy-depths", type=int, nargs="+", default=[1, 2, 3])
    cluster_parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["bisecting_kmeans", "birch", "knn_leiden"],
        choices=[
            "kmeans",
            "minibatch_kmeans",
            "bisecting_kmeans",
            "birch",
            "knn_leiden",
            "knn_louvain",
            "agglomerative",
            "hdbscan",
        ],
    )
    cluster_parser.add_argument("--n-clusters", nargs="+", default=["data_semclass"])
    cluster_parser.add_argument("--batch-size", type=int, default=4096)
    cluster_parser.add_argument("--n-init", type=int, default=1)
    cluster_parser.add_argument("--min-samples", type=int, default=10)
    cluster_parser.add_argument("--min-cluster-size", type=int, default=10)
    cluster_parser.add_argument("--metric", default="euclidean")
    cluster_parser.add_argument("--birch-threshold", type=float, default=0.75)
    cluster_parser.add_argument("--birch-branching-factor", type=int, default=100)
    cluster_parser.add_argument(
        "--bisecting-strategy",
        default="biggest_inertia",
        choices=["biggest_inertia", "largest_cluster"],
    )
    cluster_parser.add_argument("--graph-n-neighbors", type=int, default=15)
    cluster_parser.add_argument("--graph-resolution", type=float, default=1.0)
    cluster_parser.add_argument("--graph-metric", default="cosine")
    cluster_parser.add_argument("--n-jobs", type=int)
    cluster_parser.add_argument("--seed", type=int, default=42)
    cluster_parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    cluster_parser.add_argument("--no-progress", action="store_true")
    cluster_parser.set_defaults(normalize=True)
    cluster_parser.set_defaults(func=_cmd_cluster)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
