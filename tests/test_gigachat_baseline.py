from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobaldclusterisation import gigachat_baseline
from cobaldclusterisation.embeddings import EmbeddingConfig


def _tokens(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FORM": [f"токен{i}" for i in range(n_rows)],
            "LEMMA": [f"лемма{i}" for i in range(n_rows)],
            "SEMCLASS": ["CLASS"] * n_rows,
            "context_text": [f"контекст {i}" for i in range(n_rows)],
        }
    )


def test_gigachat_run_uses_model_specific_embedding_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokens = _tokens(8)
    embeddings = np.arange(48, dtype=np.float32).reshape(8, 6)
    seen_configs: list[EmbeddingConfig] = []

    def fake_generate_embeddings_from_corpus(**kwargs: object) -> dict[str, object]:
        config = kwargs["config"]
        assert isinstance(config, EmbeddingConfig)
        seen_configs.append(config)
        return {
            "embeddings": embeddings,
            "tokens": tokens,
            "saved_path": str(tmp_path / "full.pkl"),
            "config": {"model_name": config.model_name},
        }

    def fake_run_clustering(
        clustering_embeddings: np.ndarray,
        clustering_tokens: pd.DataFrame,
        **kwargs: object,
    ) -> dict[str, object]:
        assert clustering_embeddings.shape == (8, 2)
        assert len(clustering_tokens) == 8
        return {
            "labels": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
            "scores": {"silhouette": 0.5, "n_noise": 0},
            "summary": pd.DataFrame(
                {
                    "cluster": [0, 1],
                    "cluster_name": ["a", "b"],
                    "token_count": [4, 4],
                }
            ),
            "model": object(),
            "config": {"algorithm": "kmeans", "n_clusters": 2, "normalize": False},
        }

    def fake_write_cluster_summary_excel(
        results: list[dict[str, object]],
        output_path: Path,
    ) -> str:
        output_path.write_text("excel", encoding="utf-8")
        return str(output_path)

    def fake_write_scores(
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

    monkeypatch.setattr(
        gigachat_baseline,
        "generate_embeddings_from_corpus",
        fake_generate_embeddings_from_corpus,
    )
    monkeypatch.setattr(gigachat_baseline, "run_clustering", fake_run_clustering)
    monkeypatch.setattr(
        gigachat_baseline,
        "write_cluster_summary_excel",
        fake_write_cluster_summary_excel,
    )
    monkeypatch.setattr(gigachat_baseline, "_write_scores", fake_write_scores)

    result = gigachat_baseline.run(
        algorithms=["KMeans"],
        n_clusters=2,
        output_dir=tmp_path,
        projection_n_components=2,
        projection_batch_size=4,
        show_progress=False,
    )

    assert len(seen_configs) == 1
    config = seen_configs[0]
    assert config.model_name == gigachat_baseline.MODEL_NAME
    assert config.batch_size == 1
    assert config.torch_dtype == "bfloat16"
    assert config.trust_remote_code is False
    assert Path(result["paths"]["clustering_features"]).exists()
    assert Path(result["paths"]["excel"]).exists()
    assert Path(result["paths"]["scores_csv"]).exists()
    assert Path(result["paths"]["scores_xlsx"]).exists()
    assert all(Path(path).exists() for path in result["paths"]["scores_txt"])


def test_gigachat_run_can_override_trust_remote_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_configs: list[EmbeddingConfig] = []

    def fake_generate_embeddings_from_corpus(**kwargs: object) -> dict[str, object]:
        config = kwargs["config"]
        assert isinstance(config, EmbeddingConfig)
        seen_configs.append(config)
        return {
            "embeddings": np.arange(48, dtype=np.float32).reshape(8, 6),
            "tokens": _tokens(8),
            "saved_path": str(tmp_path / "full.pkl"),
            "config": {"model_name": config.model_name},
        }

    monkeypatch.setattr(
        gigachat_baseline,
        "generate_embeddings_from_corpus",
        fake_generate_embeddings_from_corpus,
    )
    monkeypatch.setattr(
        gigachat_baseline,
        "run_clustering",
        lambda *args, **kwargs: {
            "labels": np.zeros(8, dtype=np.int64),
            "scores": {"n_clusters_found": 1},
            "summary": pd.DataFrame({"cluster": [0]}),
            "model": object(),
            "config": {"algorithm": "kmeans", "n_clusters": 1, "normalize": False},
        },
    )
    monkeypatch.setattr(
        gigachat_baseline,
        "write_cluster_summary_excel",
        lambda results, output_path: str(output_path),
    )
    monkeypatch.setattr(
        gigachat_baseline,
        "_write_scores",
        lambda results, output_dir, label: (
            str(output_dir / f"{label}_scores.csv"),
            str(output_dir / f"{label}_scores.xlsx"),
            [],
        ),
    )

    gigachat_baseline.run(
        algorithms=["KMeans"],
        n_clusters=1,
        output_dir=tmp_path,
        projection_n_components=2,
        projection_batch_size=4,
        trust_remote_code=True,
        show_progress=False,
    )

    assert seen_configs[0].trust_remote_code is True
