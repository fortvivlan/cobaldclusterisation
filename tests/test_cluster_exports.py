from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from cobaldclusterisation.cluster_exports import (
    annotate_tokens,
    output_prefix,
    write_clustering_outputs,
)


def _tokens() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["1", "2", "1", "2"],
            "FORM": ["кошка", "кот", "банк", "деньги"],
            "LEMMA": ["кошка", "кот", "банк", "деньги"],
            "UPOS": ["NOUN"] * 4,
            "XPOS": ["_"] * 4,
            "FEATS": ["_"] * 4,
            "HEAD": ["0"] * 4,
            "DEPREL": ["root"] * 4,
            "DEPS": ["_"] * 4,
            "MISC": ["_"] * 4,
            "DEEPSLOT": ["_"] * 4,
            "SEMCLASS": ["ANIMAL", "ANIMAL", "MONEY", "MONEY"],
            "split": ["train", "train", "dev", "dev"],
            "sentence_index": [0, 0, 1, 1],
            "sentence_id": ["s1", "s1", "s2", "s2"],
        }
    )


def _result() -> dict[str, object]:
    return {
        "labels": np.array([0, 0, 1, 1], dtype=np.int64),
        "scores": {"semclass_exact_purity": 1.0},
        "summary": pd.DataFrame(
            {
                "cluster": [0, 1],
                "cluster_name": ["кошка", "банк"],
                "token_count": [2, 2],
            }
        ),
        "hierarchy_alignment": pd.DataFrame(),
        "config": {"algorithm": "bisecting_kmeans", "n_clusters": 100},
    }


def test_output_prefix_groups_cluster_counts_and_family() -> None:
    results = [
        {"config": {"algorithm": "bisecting_kmeans", "n_clusters": 100}},
        {"config": {"algorithm": "birch", "n_clusters": 200}},
        {"config": {"algorithm": "knn_leiden", "graph_n_neighbors": 15}},
    ]

    assert (
        output_prefix(results, label="rubert tiny2")
        == "100,200cl_rubert_tiny2_Hierarchical"
    )


def test_annotate_tokens_uses_readable_auto_class_names() -> None:
    annotated = annotate_tokens(_tokens(), [0, 0, 1, 1], result=_result())

    assert annotated["AUTO_SEMCLASS"].tolist() == [
        "cluster_0:кошка",
        "cluster_0:кошка",
        "cluster_1:банк",
        "cluster_1:банк",
    ]


def test_write_clustering_outputs_saves_pickle_and_conllu_plus(
    tmp_path: Path,
) -> None:
    paths = write_clustering_outputs(
        [_result()],
        _tokens(),
        output_dir=tmp_path,
        label="rubert_tiny2",
    )

    assert Path(paths.excel).name == "100cl_rubert_tiny2_Hierarchical_summary.xlsx"
    assert Path(paths.scores_xlsx).name == "100cl_rubert_tiny2_Hierarchical_scores.xlsx"
    assert Path(paths.artifacts_pickle).exists()
    assert paths.annotated_conllu_plus is not None

    with Path(paths.artifacts_pickle).open("rb") as handle:
        payload = pickle.load(handle)

    assert payload["prefix"] == "100cl_rubert_tiny2_Hierarchical"
    assert payload["runs"][0]["labels"].tolist() == [0, 0, 1, 1]
    assert "AUTO_SEMCLASS" in payload["runs"][0]["annotated_tokens"].columns

    conllu = Path(paths.annotated_conllu_plus).read_text(encoding="utf-8")
    assert "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC DEEPSLOT SEMCLASS AUTO_SEMCLASS" in conllu
    assert "# sent_id = s1" in conllu
    assert "кошка\tкошка\tNOUN" in conllu
    assert "cluster_1:банк" in conllu
