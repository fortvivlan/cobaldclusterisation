import numpy as np
import pandas as pd

from cobaldclusterisation.clustering import (
    ClusterConfig,
    evaluate_clusters,
    fit_predict_clusters,
    hierarchy_ancestor_labels,
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
    assert set(summary["size"]) == {2}
    assert set(summary["representative_semclass"]) == {"ANIMAL", "MONEY"}


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
