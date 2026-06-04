from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobaldclusterisation import rubert_baseline
from cobaldclusterisation.embeddings import save_embeddings_pickle


def _tokens(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FORM": [f"токен{i}" for i in range(n_rows)],
            "LEMMA": [f"лемма{i}" for i in range(n_rows)],
            "SEMCLASS": ["CLASS"] * n_rows,
            "context_text": [f"контекст {i}" for i in range(n_rows)],
        }
    )


def _fake_write_cluster_summary_excel(
    results: list[dict[str, object]],
    output_path: Path,
) -> str:
    output_path.write_text("excel", encoding="utf-8")
    return str(output_path)


def _fake_write_scores(
    results: list[dict[str, object]],
    *,
    output_dir: Path,
    label: str,
) -> tuple[str, str, list[str]]:
    scores_csv = output_dir / f"{label}_scores.csv"
    scores_xlsx = output_dir / f"{label}_scores.xlsx"
    scores_txt = output_dir / f"{label}_scores.txt"
    scores_csv.write_text("metric,run_1\nsilhouette,0.5\n", encoding="utf-8")
    scores_xlsx.write_text("excel", encoding="utf-8")
    scores_txt.write_text("scores", encoding="utf-8")
    return str(scores_csv), str(scores_xlsx), [str(scores_txt)]


def test_rubert_run_can_use_saved_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved_path = save_embeddings_pickle(
        {
            "embeddings": np.arange(48, dtype=np.float32).reshape(8, 6),
            "tokens": _tokens(8),
            "config": {"model_name": "saved"},
        },
        tmp_path / "saved_embeddings.pkl",
    )
    seen_shapes: list[tuple[int, int]] = []

    def fake_run_clustering(
        clustering_embeddings: np.ndarray,
        clustering_tokens: pd.DataFrame,
        **kwargs: object,
    ) -> dict[str, object]:
        seen_shapes.append(clustering_embeddings.shape)
        assert len(clustering_tokens) == 8
        return {
            "labels": np.zeros(8, dtype=np.int64),
            "scores": {"silhouette": 0.5},
            "summary": pd.DataFrame({"cluster": [0]}),
            "model": object(),
            "config": {"algorithm": "kmeans", "n_clusters": 2, "normalize": False},
        }

    monkeypatch.setattr(
        rubert_baseline,
        "generate_embeddings_from_corpus",
        lambda **kwargs: pytest.fail("embedding generation should be skipped"),
    )
    monkeypatch.setattr(rubert_baseline, "run_clustering", fake_run_clustering)
    monkeypatch.setattr(
        rubert_baseline,
        "write_cluster_summary_excel",
        _fake_write_cluster_summary_excel,
    )
    monkeypatch.setattr(rubert_baseline, "_write_scores", _fake_write_scores)

    result = rubert_baseline.run(
        embeddings_path=saved_path,
        algorithms=["KMeans"],
        n_clusters=2,
        output_dir=tmp_path,
        show_progress=False,
    )

    assert seen_shapes == [(8, 6)]
    assert result["paths"]["embeddings"] == str(saved_path)
    assert Path(result["paths"]["excel"]).exists()
    assert Path(result["paths"]["scores_xlsx"]).exists()
