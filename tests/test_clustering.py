import numpy as np
import pandas as pd

from cobaldclusterisation.clustering import (
    ClusterConfig,
    evaluate_clusters,
    fit_predict_clusters,
    hierarchy_ancestor_labels,
    run_clustering_suite,
    summarize_clusters,
)


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
    assert set(summary["semclasses"]) == {"ANIMAL", "MONEY"}
    assert "\n" in "\n".join(summary["top_lemmas"])


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
