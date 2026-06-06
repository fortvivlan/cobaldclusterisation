from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobaldclusterisation import rubert_baseline
from cobaldclusterisation.embeddings import EmbeddingConfig, save_embeddings_pickle


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


def test_rubert_run_can_train_on_syntagrus_payload_and_apply_to_cobald(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cobald_path = save_embeddings_pickle(
        {
            "embeddings": np.arange(16, dtype=np.float32).reshape(4, 4),
            "tokens": _tokens(4),
            "config": {"model_name": "saved-cobald"},
        },
        tmp_path / "cobald.pkl",
    )
    training_path = save_embeddings_pickle(
        {
            "embeddings": np.arange(24, dtype=np.float32).reshape(6, 4),
            "tokens": _tokens(6).assign(SEMCLASS="_", split="syntagrus"),
            "config": {"model_name": "saved-syntagrus"},
        },
        tmp_path / "syntagrus.pkl",
    )
    seen_shapes: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    def fake_run_clustering_with_training_data(
        training_embeddings: np.ndarray,
        evaluation_embeddings: np.ndarray,
        evaluation_tokens: pd.DataFrame,
        **kwargs: object,
    ) -> dict[str, object]:
        seen_shapes.append(
            (
                training_embeddings.shape,
                evaluation_embeddings.shape,
                len(evaluation_tokens),
            )
        )
        return {
            "labels": np.zeros(len(evaluation_tokens), dtype=np.int64),
            "scores": {"silhouette": 0.5, "training_rows": 6.0},
            "summary": pd.DataFrame({"cluster": [0]}),
            "model": object(),
            "config": {
                "algorithm": "kmeans",
                "n_clusters": 2,
                "normalize": False,
                "prediction_method": "model_predict",
            },
        }

    monkeypatch.setattr(
        rubert_baseline,
        "generate_embeddings_from_corpus",
        lambda **kwargs: pytest.fail("embedding generation should be skipped"),
    )
    monkeypatch.setattr(
        rubert_baseline,
        "run_clustering",
        lambda *args, **kwargs: pytest.fail("CoBaLD-only clustering should be skipped"),
    )
    monkeypatch.setattr(
        rubert_baseline,
        "run_clustering_with_training_data",
        fake_run_clustering_with_training_data,
    )
    monkeypatch.setattr(
        rubert_baseline,
        "write_cluster_summary_excel",
        _fake_write_cluster_summary_excel,
    )
    monkeypatch.setattr(rubert_baseline, "_write_scores", _fake_write_scores)

    result = rubert_baseline.run(
        embeddings_path=cobald_path,
        training_embeddings_path=training_path,
        training_source="syntagrus",
        algorithms=["KMeans"],
        n_clusters=2,
        output_dir=tmp_path,
        show_progress=False,
    )

    assert seen_shapes == [((6, 4), (4, 4), 4)]
    assert result["training_source"] == "syntagrus"
    assert result["paths"]["training_embeddings"] == str(training_path)
    assert result["paths"]["embeddings"] == str(cobald_path)


def test_prepare_training_payloads_can_merge_syntagrus_and_cobald(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    syntagrus_dir = tmp_path / "UD_Russian-SynTagRus"
    syntagrus_dir.mkdir()
    syntagrus_file = syntagrus_dir / "ru_syntagrus-ud-train-a.conllu"
    syntagrus_file.write_text(
        "\n".join(
            [
                "# sent_id = s1",
                "# text = Кот спит.",
                "1\tКот\tкот\tNOUN\t_\t_\t2\tnsubj\t_\t_",
                "2\tспит\tспать\tVERB\t_\t_\t0\troot\t_\t_",
                "3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cobald_payload = {
        "embeddings": np.ones((2, 2), dtype=np.float32),
        "tokens": _tokens(2),
        "saved_path": str(tmp_path / "cobald.pkl"),
        "config": {"model_name": "dummy"},
    }

    def fake_generate_cobald_embeddings(**kwargs: object) -> dict[str, object]:
        return cobald_payload

    def fake_generate_token_embeddings(
        sentences: object,
        **kwargs: object,
    ) -> dict[str, object]:
        sentence_list = list(sentences)
        assert [token.form for token in sentence_list[0].embedding_targets()] == [
            "Кот",
            "спит",
        ]
        return {
            "embeddings": np.zeros((2, 2), dtype=np.float32),
            "tokens": pd.DataFrame(
                {
                    "FORM": ["Кот", "спит"],
                    "LEMMA": ["кот", "спать"],
                    "SEMCLASS": ["_", "_"],
                    "split": ["syntagrus", "syntagrus"],
                }
            ),
            "config": {"model_name": "dummy"},
        }

    monkeypatch.setattr(
        rubert_baseline,
        "generate_token_embeddings",
        fake_generate_token_embeddings,
    )

    _, training_payload, source, data_paths = (
        rubert_baseline.prepare_training_and_cobald_payloads(
            data_dir=tmp_path / "CobaldRus",
            splits=("train",),
            training_source="syntagrus_cobald",
            output_dir=tmp_path,
            embedding_output_path=tmp_path / "cobald.pkl",
            training_embedding_filename="merged.pkl",
            embeddings_path=None,
            training_embeddings_path=None,
            drive_output_dir=None,
            embedding_config=EmbeddingConfig(model_name="dummy"),
            syntagrus_dir=syntagrus_dir,
            syntagrus_files=("ru_syntagrus-ud-train-a.conllu",),
            syntagrus_repo_url="https://example.invalid/repo.git",
            clone_syntagrus_if_missing=False,
            show_progress=False,
            generate_cobald_embeddings=fake_generate_cobald_embeddings,
        )
    )

    assert source == "syntagrus_cobald"
    assert training_payload["embeddings"].shape == (4, 2)
    assert len(training_payload["tokens"]) == 4
    assert training_payload["saved_path"] == str(tmp_path / "merged.pkl")
    assert data_paths == [str(syntagrus_file)]
