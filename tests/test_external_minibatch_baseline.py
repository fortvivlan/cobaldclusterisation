from pathlib import Path

import numpy as np
import pandas as pd

from cobaldclusterisation import external_minibatch_baseline as external


def test_tokenize_raw_text_keeps_words_and_punctuation() -> None:
    tokens = external.tokenize_raw_text("Кошка спит, банк-онлайн открыт.")

    assert tokens == ["Кошка", "спит", ",", "банк-онлайн", "открыт", "."]
    assert [external.is_word_token(token) for token in tokens] == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]


def test_collect_external_contexts_respects_token_cap() -> None:
    dataset = [
        {"text": "кошка спит. банк открыт."},
        {"text": "кот играет. деньги лежат."},
    ]

    chunks = external.collect_external_contexts(
        dataset,
        max_train_tokens=5,
        max_context_word_tokens=2,
        show_progress=False,
    )

    assert sum(chunk.target_count for chunk in chunks) == 5
    assert chunks[-1].target_count == 1
    assert chunks[-1].tokens == ["кот"]


class DummyEmbedder:
    hidden_size = 2

    def __init__(self, config: object) -> None:
        self.config = config

    def embed_pretokenized(
        self,
        words_by_context: list[list[str]],
        target_indices_by_context: list[list[int]],
    ) -> np.ndarray:
        vectors: list[list[float]] = []
        for words, target_indices in zip(
            words_by_context,
            target_indices_by_context,
            strict=True,
        ):
            for index in target_indices:
                token = words[index].lower()
                if token in {"кошка", "кот"}:
                    vectors.append([1.0, 0.0])
                elif token in {"банк", "деньги"}:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)


def _write_conllu(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# sent_id = 1",
                "# text = кошка банк .",
                "1\tкошка\tкошка\tNOUN\t_\t_\t0\troot\t_\t_\t_\tANIMAL",
                "2\tбанк\tбанк\tNOUN\t_\t_\t0\troot\t_\t_\t_\tMONEY",
                "3\t.\t.\tPUNCT\t_\t_\t0\tpunct\t_\t_\t_\t_",
                "",
                "# sent_id = 2",
                "# text = кот деньги .",
                "1\tкот\tкот\tNOUN\t_\t_\t0\troot\t_\t_\t_\tANIMAL",
                "2\tденьги\tденьги\tNOUN\t_\t_\t0\troot\t_\t_\t_\tMONEY",
                "3\t.\t.\tPUNCT\t_\t_\t0\tpunct\t_\t_\t_\t_",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_run_trains_external_kmeans_and_scores_cobald(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "CobaldRus"
    data_dir.mkdir()
    _write_conllu(data_dir / "train.conllu")
    _write_conllu(data_dir / "dev.conllu")

    monkeypatch.setattr(external, "_TransformerEmbedder", DummyEmbedder)
    monkeypatch.setattr(
        external,
        "load_external_dataset_stream",
        lambda **kwargs: [
            {"text": "кошка кот кошка кот"},
            {"text": "банк деньги банк деньги"},
        ],
    )

    result = external.run(
        data_dir=data_dir,
        hierarchy=None,
        max_train_tokens=8,
        output_dir=tmp_path,
        model_name="dummy",
        model_label="dummy_external",
        embedding_batch_size=2,
        kmeans_batch_size=4,
        n_clusters="data_semclass",
        device=None,
        show_progress=False,
    )

    paths = result["paths"]
    assert Path(paths["training_features"]).exists()
    assert Path(paths["cobald_features"]).exists()
    assert Path(paths["kmeans_model"]).exists()
    assert Path(paths["clustered_tokens_xlsx"]).exists()
    assert Path(paths["scores_xlsx"]).exists()
    assert result["results"][0]["scores"]["semclass_exact_n_labeled"] == 8.0
    assert result["results"][0]["scores"]["semclass_exact_purity"] == 1.0

    clustered = pd.read_csv(paths["clustered_tokens_csv"])
    assert "cluster" in clustered.columns
    assert len(clustered) == 8
