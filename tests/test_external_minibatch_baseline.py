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


class FakeEncoding:
    def __init__(self, word_ids: list[int]) -> None:
        self._word_ids = word_ids

    def word_ids(self) -> list[int]:
        return self._word_ids


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(
        self,
        tokens: list[str],
        *,
        is_split_into_words: bool,
        add_special_tokens: bool,
        truncation: bool,
    ) -> FakeEncoding:
        assert is_split_into_words
        assert not add_special_tokens
        assert not truncation
        word_ids: list[int] = []
        for index, token in enumerate(tokens):
            piece_count = 4 if token.startswith("длинный") else 1
            word_ids.extend([index] * piece_count)
        return FakeEncoding(word_ids)


def test_make_tokenizer_safe_external_chunks_preserves_targets() -> None:
    chunk = external.RawTextChunk(
        tokens=["кошка", "длинный", "банк", "кот"],
        target_indices=[0, 1, 2, 3],
        document_index=0,
        chunk_index=0,
    )

    safe_chunks = external.make_tokenizer_safe_external_chunks(
        [chunk],
        tokenizer=FakeTokenizer(),
        max_length=6,
        show_progress=False,
    )

    assert [safe_chunk.tokens for safe_chunk in safe_chunks] == [
        ["кошка"],
        ["длинный"],
        ["банк", "кот"],
    ]
    assert sum(safe_chunk.target_count for safe_chunk in safe_chunks) == 4


def _write_ud_conllu(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# sent_id = syntagrus-1",
                "# text = Кошка спит, банк открыт.",
                "1\tКошка\tкошка\tNOUN\t_\tAnimacy=Anim\t2\tnsubj\t_\t_",
                "2\tспит\tспать\tVERB\t_\t_\t0\troot\t_\t_",
                "3\t,\t,\tPUNCT\t_\t_\t2\tpunct\t_\t_",
                "4\tбанк\tбанк\tNOUN\t_\t_\t5\tnsubj\t_\t_",
                "5\tоткрыт\tоткрыть\tVERB\t_\t_\t2\tconj\t_\t_",
                "6\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_",
                "",
                "# sent_id = syntagrus-2",
                "# text = Кот и деньги.",
                "1-2\tКот-и\t_\t_\t_\t_\t_\t_\t_\t_",
                "1\tКот\tкот\tNOUN\t_\t_\t0\troot\t_\t_",
                "2\tи\tи\tCCONJ\t_\t_\t3\tcc\t_\t_",
                "3\tденьги\tденьги\tNOUN\t_\t_\t1\tconj\t_\t_",
                "4\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_iter_ud_conllu_chunks_keeps_forms_lemmas_and_word_targets(
    tmp_path: Path,
) -> None:
    ud_path = tmp_path / "ru_syntagrus-ud-train-a.conllu"
    _write_ud_conllu(ud_path)

    chunks = list(
        external.iter_ud_conllu_chunks(
            [ud_path],
            max_context_tokens=4,
            max_context_word_tokens=3,
        )
    )

    assert [chunk.tokens for chunk in chunks] == [
        ["Кошка", "спит", ",", "банк"],
        ["открыт", "."],
        ["Кот", "и", "деньги", "."],
    ]
    assert chunks[0].lemmas == ["кошка", "спать", ",", "банк"]
    assert chunks[0].target_indices == [0, 1, 3]
    assert chunks[2].tokens[0] == "Кот"
    assert "Кот-и" not in chunks[2].tokens


class DummyEmbedder:
    hidden_size = 2

    def __init__(self, config: object) -> None:
        self.config = config
        self.tokenizer = FakeTokenizer()

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


def test_run_uses_ud_syntagrus_for_external_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "CobaldRus"
    data_dir.mkdir()
    _write_conllu(data_dir / "train.conllu")
    _write_conllu(data_dir / "dev.conllu")

    syntagrus_dir = tmp_path / "UD_Russian-SynTagRus"
    syntagrus_dir.mkdir()
    _write_ud_conllu(syntagrus_dir / "ru_syntagrus-ud-train-a.conllu")

    monkeypatch.setattr(external, "_TransformerEmbedder", DummyEmbedder)

    result = external.run(
        data_dir=data_dir,
        hierarchy=None,
        external_source=external.SYNTAGRUS_EXTERNAL_SOURCE,
        syntagrus_dir=syntagrus_dir,
        syntagrus_files=("ru_syntagrus-ud-train-a.conllu",),
        max_train_tokens=4,
        output_dir=tmp_path,
        model_name="dummy",
        model_label="dummy_syntagrus_external",
        embedding_batch_size=2,
        kmeans_batch_size=4,
        n_clusters="data_semclass",
        device=None,
        show_progress=False,
    )

    assert result["config"]["external_source"] == external.SYNTAGRUS_EXTERNAL_SOURCE
    assert result["config"]["external_training_tokens"] == 4
    assert result["config"]["external_data_paths"] == [
        str(syntagrus_dir / "ru_syntagrus-ud-train-a.conllu")
    ]
    assert Path(result["paths"]["training_features"]).exists()
    clustered = pd.read_csv(result["paths"]["clustered_tokens_csv"])
    assert "cluster" in clustered.columns
    assert len(clustered) == 8


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
