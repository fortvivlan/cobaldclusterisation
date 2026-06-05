from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobaldclusterisation.clustering import (
    ClusterConfig,
    evaluate_clusters,
    fit_predict_clusters,
    hierarchy_alignment_table,
    hierarchy_ancestor_labels,
    infer_semclass_cluster_count,
    run_clustering_suite,
    summarize_clusters,
)
from cobaldclusterisation.data import corpus_to_dataframe, load_corpus
from cobaldclusterisation.rubert_baseline import (
    build_cluster_configs,
    format_score_table,
)
from cobaldclusterisation.scores import scores_to_metric_dataframe, write_scores


def _tokens() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FORM": ["кошка", "кот", "банк", "деньги"],
            "LEMMA": ["кошка", "кот", "банк", "деньги"],
            "SEMCLASS": ["ANIMAL", "ANIMAL", "MONEY", "MONEY"],
            "context_text": [
                "кошка спит",
                "кот спит",
                "банк открыт",
                "деньги лежат",
            ],
        }
    )


def test_fit_evaluate_and_summarize_clusters() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ],
        dtype=np.float32,
    )
    labels, _ = fit_predict_clusters(
        embeddings,
        config=ClusterConfig(
            algorithm="kmeans",
            n_clusters=2,
            normalize=False,
            random_state=0,
            n_init=1,
        ),
    )

    scores = evaluate_clusters(embeddings, labels, tokens=_tokens())
    summary = summarize_clusters(embeddings, labels, _tokens())

    assert scores["semclass_exact_purity"] == 1.0
    assert scores["semclass_exact_n_labeled"] == 4.0
    assert set(summary["token_count"]) == {2}
    assert set(summary["lemma_count"]) == {2}
    assert set(summary["semclasses"]) == {"ANIMAL: 2", "MONEY: 2"}
    assert "\n" in "\n".join(summary["lemmas"])


def test_fit_predict_supports_scalable_sklearn_algorithms() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ],
        dtype=np.float32,
    )

    for algorithm in ("bisecting_kmeans", "birch"):
        labels, _ = fit_predict_clusters(
            embeddings,
            config=ClusterConfig(
                algorithm=algorithm,
                n_clusters=2,
                normalize=False,
                random_state=0,
                n_init=1,
            ),
        )

        assert labels.shape == (4,)
        assert len(set(labels.tolist())) == 2


def test_fit_predict_knn_leiden_optional_graph_backend() -> None:
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    pytest.importorskip("pynndescent")
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ],
        dtype=np.float32,
    )

    labels, _ = fit_predict_clusters(
        embeddings,
        config=ClusterConfig(
            algorithm="knn_leiden",
            normalize=False,
            metric="euclidean",
            graph_n_neighbors=2,
            random_state=0,
        ),
    )

    assert labels.shape == (4,)


def test_summarize_clusters_lists_all_lemmas_but_limits_top_forms() -> None:
    tokens = pd.DataFrame(
        {
            "FORM": ["форма1", "форма2", "форма3"],
            "LEMMA": ["лемма1", "лемма2", "лемма3"],
            "SEMCLASS": ["CLASS_A", "CLASS_A", "CLASS_B"],
            "context_text": ["one", "two", "three"],
        }
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_clusters(embeddings, [0, 0, 0], tokens, top_n=1)
    row = summary.iloc[0]

    assert row["lemmas"] == "лемма1: 1\nлемма2: 1\nлемма3: 1"
    assert row["top_forms"] == "форма1: 1"
    assert row["semclasses"] == "CLASS_A: 2\nCLASS_B: 1"


def test_hierarchy_ancestor_labels() -> None:
    hierarchy = pd.DataFrame(
        [
            {"class_id": "1", "parent_id": None, "depth": 0, "class_name": "ROOT"},
            {"class_id": "2", "parent_id": "1", "depth": 1, "class_name": "ENTITY"},
            {"class_id": "3", "parent_id": "2", "depth": 2, "class_name": "ANIMAL"},
            {"class_id": "4", "parent_id": "2", "depth": 2, "class_name": "MONEY"},
        ]
    )

    labels = hierarchy_ancestor_labels(["ANIMAL", "MONEY", "_"], hierarchy, depth=1)

    assert labels.tolist() == ["ENTITY", "ENTITY", "_"]


def test_hierarchy_alignment_table_reports_best_label_per_depth() -> None:
    hierarchy = pd.DataFrame(
        [
            {"class_id": "1", "parent_id": None, "depth": 0, "class_name": "ROOT"},
            {"class_id": "2", "parent_id": "1", "depth": 1, "class_name": "ENTITY"},
            {"class_id": "3", "parent_id": "2", "depth": 2, "class_name": "ANIMAL"},
            {"class_id": "4", "parent_id": "2", "depth": 2, "class_name": "MONEY"},
        ]
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    alignment = hierarchy_alignment_table(
        labels,
        _tokens(),
        hierarchy,
        hierarchy_depths=(1, 2),
    )

    animal_row = alignment[
        (alignment["depth"] == 2) & (alignment["cluster"] == 0)
    ].iloc[0]
    assert animal_row["best_label"] == "ANIMAL"
    assert animal_row["best_label_count"] == 2
    assert animal_row["cluster_purity"] == 1.0
    assert animal_row["gold_label_coverage"] == 1.0


def test_infer_semclass_cluster_count_from_bundled_corpus() -> None:
    sentences = load_corpus("CobaldRus", splits=("train", "dev"))
    tokens = corpus_to_dataframe(
        sentences,
        include_punctuation=False,
        include_empty=False,
    )

    assert infer_semclass_cluster_count(tokens) == 565


def test_run_clustering_suite_accepts_progress_flag() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ],
        dtype=np.float32,
    )
    payload = {"embeddings": embeddings, "tokens": _tokens()}

    results = run_clustering_suite(
        payload,
        configs=[
            ClusterConfig(
                algorithm="kmeans",
                n_clusters=2,
                normalize=False,
                random_state=0,
                n_init=1,
            )
        ],
        show_progress=True,
    )

    assert len(results) == 1
    assert results[0]["labels"].shape == (4,)


def test_rubert_baseline_builds_colab_friendly_configs() -> None:
    configs = build_cluster_configs(
        algorithms=["KMeans", "MiniBatchKMeans", "HDBSCAN"],
        n_clusters=[50, 100],
        min_cluster_size=25,
        hdbscan_n_jobs=-1,
    )

    assert [config.algorithm for config in configs] == [
        "kmeans",
        "kmeans",
        "minibatch_kmeans",
        "minibatch_kmeans",
        "hdbscan",
    ]
    assert [config.n_clusters for config in configs[:4]] == [50, 100, 50, 100]
    assert configs[-1].min_cluster_size == 25
    assert configs[-1].n_jobs == -1


def test_rubert_baseline_builds_new_default_configs() -> None:
    configs = build_cluster_configs(
        tokens=_tokens(),
        n_clusters="data_semclass",
        graph_n_neighbors=12,
    )

    assert [config.algorithm for config in configs] == [
        "bisecting_kmeans",
        "birch",
        "knn_leiden",
    ]
    assert configs[0].n_clusters == 2
    assert configs[1].n_clusters == 2
    assert configs[2].graph_n_neighbors == 12
    assert configs[2].metric == "cosine"


def test_rubert_baseline_formats_score_table() -> None:
    table = format_score_table(
        "ruBERT baseline",
        {"silhouette": 0.12345, "n_noise": 2},
    )

    assert "metric - result - reference note" in table
    assert "silhouette - 0.1235 - -1 to 1; higher is better" in table
    assert "n_noise - 2 - count; HDBSCAN noise points" in table


def test_scores_to_metric_dataframe_uses_metric_rows() -> None:
    results = [
        {
            "scores": {"silhouette": 0.5, "n_noise": 0},
            "config": {"algorithm": "kmeans", "n_clusters": 2},
        },
        {
            "scores": {"silhouette": 0.25, "n_noise": 3},
            "config": {"algorithm": "hdbscan", "min_cluster_size": 10},
        },
    ]

    scores = scores_to_metric_dataframe(results)

    assert scores["metric"].tolist() == ["silhouette", "n_noise"]
    assert scores.loc[0, "run_1_kmeans_k2"] == 0.5
    assert scores.loc[1, "run_2_hdbscan_min10"] == 3


def test_write_scores_saves_metric_csv_and_xlsx(tmp_path) -> None:
    results = [
        {
            "scores": {"silhouette": 0.5, "n_noise": 0},
            "config": {"algorithm": "kmeans", "n_clusters": 2, "normalize": True},
        }
    ]

    scores_csv, scores_xlsx, scores_txt = write_scores(
        results,
        output_dir=tmp_path,
        label="score test",
    )

    scores = pd.read_csv(scores_csv)

    assert Path(scores_xlsx).exists()
    assert len(scores_txt) == 1
    assert scores["metric"].tolist() == ["silhouette", "n_noise"]
    assert scores.loc[0, "run_1_kmeans_k2"] == 0.5
