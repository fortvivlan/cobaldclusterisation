"""External-corpus MiniBatch K-Means pipeline for CoBaLD evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import re
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize as l2_normalize
from tqdm.auto import tqdm

from .clustering import (
    ClusterConfig,
    evaluate_clusters,
    hierarchy_alignment_table,
    infer_semclass_cluster_count,
    summarize_clusters,
)
from .data import Sentence, Token, corpus_to_dataframe, load_corpus
from .embeddings import (
    EmbeddingConfig,
    _ensure_padding_token,
    _require_transformers,
    _resolve_device,
    _select_hidden_state,
    _sentence_context_tokens,
    _set_seed,
    _target_word_indices,
    _torch_dtype_from_name,
)
from .rubert_baseline import DEFAULT_COLAB_DIR, DEFAULT_DRIVE_DIR
from .scores import write_scores as _write_scores
from .summary_excel import write_cluster_summary_excel, write_hierarchy_alignment_excel


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-2"
DEFAULT_DATASET_SUBSET = "rus_Cyrl"
DEFAULT_RUBERT_MODEL = "DeepPavlov/rubert-base-cased"
DEFAULT_SAMBALINGO_MODEL = "sambanovasystems/SambaLingo-Russian-Base"
DEFAULT_GIGACHAT_MODEL = "ai-sage/GigaChat3-10B-A1.8B-base"

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]", re.UNICODE)


@dataclass(slots=True)
class RawTextChunk:
    """One tokenized raw-text context and its embedding target positions."""

    tokens: list[str]
    target_indices: list[int]
    document_index: int
    chunk_index: int

    @property
    def target_count(self) -> int:
        return len(self.target_indices)

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass(slots=True)
class ExternalMiniBatchPaths:
    """Output paths produced by :func:`run`."""

    training_features: str
    cobald_features: str
    kmeans_model: str
    projection_model: str | None
    clustered_tokens_csv: str
    clustered_tokens_xlsx: str
    excel: str
    hierarchy_alignment_excel: str
    scores_csv: str
    scores_xlsx: str
    scores_txt: list[str]
    config_json: str


class _TransformerEmbedder:
    """Reusable hidden-state extractor for raw text and CoBaLD sentences."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")

        torch, AutoModel, AutoTokenizer = _require_transformers()
        _set_seed(config.seed)
        self.torch = torch
        self.config = config
        self.device = _resolve_device(torch, config.device)
        dtype = _torch_dtype_from_name(torch, config.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            trust_remote_code=config.trust_remote_code,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "A fast tokenizer is required because token-to-subword alignment "
                "uses word_ids()."
            )
        _ensure_padding_token(self.tokenizer)

        model_kwargs: dict[str, object] = {"trust_remote_code": config.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(config.model_name, **model_kwargs)
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))

    def embed_pretokenized(
        self,
        words_by_context: Sequence[Sequence[str]],
        target_indices_by_context: Sequence[Sequence[int]],
    ) -> np.ndarray:
        """Embed selected token positions from pretokenized contexts."""

        if len(words_by_context) != len(target_indices_by_context):
            raise ValueError("Context and target batches must have the same length")
        if not words_by_context:
            return np.empty((0, self.hidden_size), dtype=np.float32)

        encoded = self.tokenizer(
            list(words_by_context),
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        model_inputs = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }
        with self.torch.no_grad():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=self.config.layer != -1,
                return_dict=True,
            )
            hidden = _select_hidden_state(outputs, layer=self.config.layer)

            vectors: list[np.ndarray] = []
            for batch_index, target_indices in enumerate(target_indices_by_context):
                word_ids = encoded.word_ids(batch_index=batch_index)
                for word_index in target_indices:
                    piece_positions = [
                        position
                        for position, aligned_word_index in enumerate(word_ids)
                        if aligned_word_index == word_index
                    ]
                    if not piece_positions:
                        continue
                    token_hidden = hidden[batch_index, piece_positions, :].mean(dim=0)
                    vector = token_hidden.detach().cpu().numpy().astype(np.float32)
                    if self.config.normalize:
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm
                    vectors.append(vector)

        return (
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, self.hidden_size), dtype=np.float32)
        )


def _require_datasets() -> object:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "External corpus streaming requires the optional dependency "
            "'datasets'. Install in Colab with: pip install -e '.[embeddings,external]'"
        ) from exc
    return load_dataset


def is_word_token(token: str) -> bool:
    """Return whether a token should be embedded as a word-like target."""

    return bool(_WORD_RE.search(token))


def tokenize_raw_text(text: str) -> list[str]:
    """Tokenize raw text into Unicode word-like and punctuation tokens."""

    return [match.group(0) for match in _TOKEN_RE.finditer(text)]


def iter_raw_text_chunks(
    text: str,
    *,
    document_index: int = 0,
    start_chunk_index: int = 0,
    max_context_tokens: int = 192,
    max_context_word_tokens: int = 128,
) -> Iterator[RawTextChunk]:
    """Yield bounded pretokenized contexts from one raw-text document."""

    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    if max_context_word_tokens < 1:
        raise ValueError("max_context_word_tokens must be positive")

    chunk_tokens: list[str] = []
    chunk_word_count = 0
    chunk_index = start_chunk_index

    def flush() -> RawTextChunk | None:
        nonlocal chunk_tokens, chunk_word_count, chunk_index
        if not chunk_tokens:
            return None
        target_indices = [
            index for index, token in enumerate(chunk_tokens) if is_word_token(token)
        ]
        emitted = RawTextChunk(
            tokens=chunk_tokens,
            target_indices=target_indices,
            document_index=document_index,
            chunk_index=chunk_index,
        )
        chunk_tokens = []
        chunk_word_count = 0
        chunk_index += 1
        return emitted if emitted.target_indices else None

    for token in tokenize_raw_text(text):
        token_is_word = is_word_token(token)
        if chunk_tokens and (
            len(chunk_tokens) >= max_context_tokens
            or (token_is_word and chunk_word_count >= max_context_word_tokens)
        ):
            emitted = flush()
            if emitted is not None:
                yield emitted
        chunk_tokens.append(token)
        if token_is_word:
            chunk_word_count += 1

    emitted = flush()
    if emitted is not None:
        yield emitted


def _trim_chunk_to_target_count(
    chunk: RawTextChunk,
    *,
    target_count: int,
) -> RawTextChunk:
    if target_count >= chunk.target_count:
        return chunk
    if target_count < 1:
        raise ValueError("target_count must be positive")
    final_target_index = chunk.target_indices[target_count - 1]
    tokens = chunk.tokens[: final_target_index + 1]
    target_indices = [
        index for index, token in enumerate(tokens) if is_word_token(token)
    ]
    return RawTextChunk(
        tokens=tokens,
        target_indices=target_indices,
        document_index=chunk.document_index,
        chunk_index=chunk.chunk_index,
    )


def collect_external_contexts(
    dataset: Iterable[dict[str, object]],
    *,
    max_train_tokens: int,
    text_column: str = "text",
    max_documents: int | None = None,
    max_context_tokens: int = 192,
    max_context_word_tokens: int = 128,
    show_progress: bool = True,
) -> list[RawTextChunk]:
    """Collect enough streamed raw-text contexts to reach the token budget."""

    if max_train_tokens < 1:
        raise ValueError("max_train_tokens must be positive")

    chunks: list[RawTextChunk] = []
    target_total = 0
    progress = tqdm(
        total=max_train_tokens,
        desc="Collecting raw-text targets",
        disable=not show_progress,
    )
    try:
        for document_index, example in enumerate(dataset):
            if max_documents is not None and document_index >= max_documents:
                break
            text = str(example.get(text_column, "") or "")
            if not text.strip():
                continue
            for chunk in iter_raw_text_chunks(
                text,
                document_index=document_index,
                max_context_tokens=max_context_tokens,
                max_context_word_tokens=max_context_word_tokens,
            ):
                remaining = max_train_tokens - target_total
                if remaining <= 0:
                    break
                if chunk.target_count > remaining:
                    chunk = _trim_chunk_to_target_count(
                        chunk,
                        target_count=remaining,
                    )
                chunks.append(chunk)
                target_total += chunk.target_count
                progress.update(chunk.target_count)
                if target_total >= max_train_tokens:
                    break
            if target_total >= max_train_tokens:
                break
    finally:
        progress.close()

    if target_total == 0:
        raise ValueError("No word-like targets were collected from the external corpus")
    return chunks


def load_external_dataset_stream(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_subset: str = DEFAULT_DATASET_SUBSET,
    split: str = "train",
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
) -> Iterable[dict[str, object]]:
    """Load and optionally shuffle a Hugging Face streaming dataset."""

    load_dataset = _require_datasets()
    dataset = load_dataset(
        dataset_name,
        name=dataset_subset,
        split=split,
        streaming=True,
    )
    if shuffle_buffer_size > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return dataset


def _batch_slices(items: Sequence[object], batch_size: int) -> Iterator[Sequence[object]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _iter_external_embedding_batches(
    embedder: _TransformerEmbedder,
    chunks: Sequence[RawTextChunk],
    *,
    batch_size: int,
    show_progress: bool,
    desc: str,
) -> Iterator[np.ndarray]:
    iterator = _batch_slices(chunks, batch_size)
    total = (len(chunks) + batch_size - 1) // batch_size
    progress = tqdm(iterator, total=total, desc=desc, disable=not show_progress)
    for batch in progress:
        yield embedder.embed_pretokenized(
            [chunk.tokens for chunk in batch],
            [chunk.target_indices for chunk in batch],
        )


def _normalize_if_requested(matrix: np.ndarray, *, normalize: bool) -> np.ndarray:
    if not normalize:
        return np.asarray(matrix, dtype=np.float32)
    return l2_normalize(np.asarray(matrix, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )


def _fit_incremental_pca_from_external(
    embedder: _TransformerEmbedder,
    chunks: Sequence[RawTextChunk],
    *,
    n_components: int,
    embedding_batch_size: int,
    pca_batch_size: int,
    normalize_input: bool,
    show_progress: bool,
) -> IncrementalPCA:
    if n_components < 1:
        raise ValueError("projection_n_components must be positive")
    if pca_batch_size < n_components:
        raise ValueError("projection_batch_size must be >= projection_n_components")

    reducer = IncrementalPCA(n_components=n_components, batch_size=pca_batch_size)
    buffer: list[np.ndarray] = []
    buffered_rows = 0

    for vectors in _iter_external_embedding_batches(
        embedder,
        chunks,
        batch_size=embedding_batch_size,
        show_progress=show_progress,
        desc="Embedding external corpus for PCA",
    ):
        if len(vectors) == 0:
            continue
        vectors = _normalize_if_requested(vectors, normalize=normalize_input)
        buffer.append(vectors)
        buffered_rows += len(vectors)
        while buffered_rows >= pca_batch_size:
            stacked = np.vstack(buffer).astype(np.float32, copy=False)
            fit_batch = stacked[:pca_batch_size]
            reducer.partial_fit(fit_batch)
            remainder = stacked[pca_batch_size:]
            buffer = [remainder] if len(remainder) else []
            buffered_rows = len(remainder)

    if buffered_rows >= n_components:
        reducer.partial_fit(np.vstack(buffer).astype(np.float32, copy=False))
    if not hasattr(reducer, "components_"):
        raise ValueError(
            "Not enough external embedding rows to fit IncrementalPCA. "
            "Increase max_train_tokens or lower projection_n_components."
        )
    return reducer


def _write_external_features(
    embedder: _TransformerEmbedder,
    chunks: Sequence[RawTextChunk],
    output_path: Path,
    *,
    projection: IncrementalPCA | None,
    feature_dim: int,
    embedding_batch_size: int,
    normalize_projection_input: bool,
    normalize_projection_output: bool,
    show_progress: bool,
) -> np.memmap:
    n_rows = sum(chunk.target_count for chunk in chunks)
    features = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_rows, feature_dim),
    )
    cursor = 0
    for vectors in _iter_external_embedding_batches(
        embedder,
        chunks,
        batch_size=embedding_batch_size,
        show_progress=show_progress,
        desc="Writing external features",
    ):
        if len(vectors) == 0:
            continue
        if projection is not None:
            vectors = _normalize_if_requested(
                vectors,
                normalize=normalize_projection_input,
            )
            vectors = projection.transform(vectors).astype(np.float32, copy=False)
            vectors = _normalize_if_requested(
                vectors,
                normalize=normalize_projection_output,
            )
        end = cursor + len(vectors)
        features[cursor:end] = vectors
        cursor = end
    features.flush()
    if cursor != n_rows:
        raise ValueError(
            f"Expected to write {n_rows} external feature rows, wrote {cursor}. "
            "This usually means tokenizer truncation removed target tokens."
        )
    return features


def _iter_feature_batches(
    features: np.ndarray,
    *,
    batch_size: int,
) -> Iterator[np.ndarray]:
    for start in range(0, len(features), batch_size):
        yield np.asarray(features[start : start + batch_size], dtype=np.float32)


def fit_minibatch_kmeans_incremental(
    features: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 4096,
    random_state: int = 42,
    n_init: int = 1,
    normalize: bool = True,
    show_progress: bool = True,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans over a feature matrix or memmap incrementally."""

    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if len(features) < n_clusters:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds feature rows={len(features)}"
        )
    if batch_size < n_clusters:
        batch_size = n_clusters

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )
    total = (len(features) + batch_size - 1) // batch_size
    progress = tqdm(
        _iter_feature_batches(features, batch_size=batch_size),
        total=total,
        desc="Fitting MiniBatchKMeans",
        disable=not show_progress,
    )
    for batch in progress:
        batch = _normalize_if_requested(batch, normalize=normalize)
        model.partial_fit(batch)
    return model


def _cobald_context_tokens(
    sentence: Sentence,
    *,
    include_punctuation_context: bool,
) -> list[Token]:
    return _sentence_context_tokens(
        sentence,
        include_punctuation=include_punctuation_context,
    )


def _iter_cobald_embedding_batches(
    embedder: _TransformerEmbedder,
    sentences: Sequence[Sentence],
    *,
    batch_size: int,
    include_punctuation_context: bool,
    show_progress: bool,
) -> Iterator[tuple[np.ndarray, list[dict[str, object]]]]:
    embedding_sentences = [
        sentence
        for sentence in sentences
        if _cobald_context_tokens(
            sentence,
            include_punctuation_context=include_punctuation_context,
        )
    ]
    iterator = _batch_slices(embedding_sentences, batch_size)
    total = (len(embedding_sentences) + batch_size - 1) // batch_size
    progress = tqdm(
        iterator,
        total=total,
        desc="Embedding CoBaLD",
        disable=not show_progress,
    )

    for batch in progress:
        context_tokens_by_sentence = [
            _cobald_context_tokens(
                sentence,
                include_punctuation_context=include_punctuation_context,
            )
            for sentence in batch
        ]
        target_indices_by_sentence = [
            _target_word_indices(context_tokens)
            for context_tokens in context_tokens_by_sentence
        ]
        words_by_sentence = [
            [token.form for token in context_tokens]
            for context_tokens in context_tokens_by_sentence
        ]
        vectors = embedder.embed_pretokenized(
            words_by_sentence,
            target_indices_by_sentence,
        )
        rows: list[dict[str, object]] = []
        for sentence, context_tokens, target_indices in zip(
            batch,
            context_tokens_by_sentence,
            target_indices_by_sentence,
            strict=True,
        ):
            embedding_context_text = sentence.context_text(
                include_punctuation=include_punctuation_context
            )
            for word_index in target_indices:
                token = context_tokens[word_index]
                rows.append(
                    {
                        **token.to_dict(context_text=sentence.text),
                        "embedding_context_text": embedding_context_text,
                        "embedding_index": len(rows),
                    }
                )
        if len(vectors) != len(rows):
            raise ValueError(
                "CoBaLD tokenizer truncation removed target tokens. "
                "Increase max_length or lower context length."
            )
        yield vectors, rows


def _write_cobald_features(
    embedder: _TransformerEmbedder,
    sentences: Sequence[Sentence],
    output_path: Path,
    *,
    projection: IncrementalPCA | None,
    feature_dim: int,
    embedding_batch_size: int,
    include_punctuation_context: bool,
    normalize_projection_input: bool,
    normalize_projection_output: bool,
    show_progress: bool,
) -> tuple[np.memmap, pd.DataFrame]:
    token_count = sum(
        len(sentence.embedding_targets())
        for sentence in sentences
    )
    features = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(token_count, feature_dim),
    )
    rows: list[dict[str, object]] = []
    cursor = 0
    for vectors, batch_rows in _iter_cobald_embedding_batches(
        embedder,
        sentences,
        batch_size=embedding_batch_size,
        include_punctuation_context=include_punctuation_context,
        show_progress=show_progress,
    ):
        if projection is not None:
            vectors = _normalize_if_requested(
                vectors,
                normalize=normalize_projection_input,
            )
            vectors = projection.transform(vectors).astype(np.float32, copy=False)
            vectors = _normalize_if_requested(
                vectors,
                normalize=normalize_projection_output,
            )
        end = cursor + len(vectors)
        features[cursor:end] = vectors
        for row in batch_rows:
            row["embedding_index"] = len(rows)
            rows.append(row)
        cursor = end
    features.flush()
    if cursor != token_count:
        raise ValueError(f"Expected {token_count} CoBaLD rows, wrote {cursor}")
    return features, pd.DataFrame(rows)


def _predict_labels_in_batches(
    model: MiniBatchKMeans,
    features: np.ndarray,
    *,
    batch_size: int,
    normalize: bool,
    show_progress: bool,
) -> np.ndarray:
    labels = np.empty(len(features), dtype=np.int64)
    progress = tqdm(
        range(0, len(features), batch_size),
        desc="Predicting CoBaLD clusters",
        disable=not show_progress,
    )
    for start in progress:
        end = min(start + batch_size, len(features))
        batch = _normalize_if_requested(features[start:end], normalize=normalize)
        labels[start:end] = model.predict(batch)
    return labels


def _write_clustered_tokens(
    tokens: pd.DataFrame,
    labels: Sequence[int],
    *,
    csv_path: Path,
    xlsx_path: Path,
) -> tuple[str, str]:
    clustered = tokens.copy()
    clustered.insert(0, "cluster", np.asarray(labels, dtype=np.int64))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(csv_path, index=False)
    try:
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it in Colab with: "
            "pip install openpyxl"
        ) from exc
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        clustered.to_excel(writer, sheet_name="tokens", index=False)
        worksheet = writer.sheets["tokens"]
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        worksheet.freeze_panes = "A2"
    return str(csv_path), str(xlsx_path)


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "external_minibatch"


def _resolve_cluster_count(
    n_clusters: int | str,
    *,
    tokens: pd.DataFrame,
) -> int:
    if isinstance(n_clusters, str):
        if n_clusters == "data_semclass":
            return infer_semclass_cluster_count(tokens)
        return int(n_clusters)
    return int(n_clusters)


def _save_pickle(value: object, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


def _save_config(config: dict[str, object], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path)


def _build_result(
    *,
    cobald_features: np.ndarray,
    cobald_tokens: pd.DataFrame,
    labels: np.ndarray,
    cluster_config: ClusterConfig,
    hierarchy: str | Path | pd.DataFrame | None,
    hierarchy_depths: Sequence[int],
) -> dict[str, object]:
    matrix = _normalize_if_requested(
        cobald_features,
        normalize=cluster_config.normalize,
    )
    scores = evaluate_clusters(
        matrix,
        labels,
        tokens=cobald_tokens,
        hierarchy=hierarchy,
        hierarchy_depths=hierarchy_depths,
        random_state=cluster_config.random_state,
    )
    summary = summarize_clusters(matrix, labels, cobald_tokens)
    alignment = (
        hierarchy_alignment_table(
            labels,
            cobald_tokens,
            hierarchy,
            hierarchy_depths=hierarchy_depths,
        )
        if hierarchy is not None
        else pd.DataFrame()
    )
    return {
        "labels": labels,
        "scores": scores,
        "summary": summary,
        "hierarchy_alignment": alignment,
        "model": None,
        "config": asdict(cluster_config),
    }


def run(
    *,
    data_dir: str | Path = "CobaldRus",
    hierarchy: str | Path | pd.DataFrame | None = "hyperonims_hierarchy.csv",
    splits: Sequence[str] = ("train", "dev"),
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_subset: str = DEFAULT_DATASET_SUBSET,
    split: str = "train",
    text_column: str = "text",
    max_train_tokens: int = 3_000_000,
    max_documents: int | None = None,
    shuffle_buffer_size: int = 10_000,
    output_dir: str | Path = DEFAULT_COLAB_DIR,
    save_models_to_drive: bool = False,
    drive_dir: str | Path = DEFAULT_DRIVE_DIR,
    model_name: str = DEFAULT_RUBERT_MODEL,
    model_label: str = "rubert_base_external_fineweb2",
    n_clusters: int | str = "data_semclass",
    embedding_batch_size: int = 8,
    kmeans_batch_size: int = 4096,
    max_length: int = 512,
    device: str | None = "cuda",
    torch_dtype: str | None = None,
    trust_remote_code: bool = False,
    seed: int = 42,
    include_punctuation_context: bool = True,
    normalize_embeddings: bool = False,
    normalize_for_clustering: bool = True,
    projection_n_components: int | None = None,
    projection_batch_size: int = 8192,
    normalize_projection_input: bool = True,
    normalize_projection_output: bool = True,
    max_context_tokens: int = 192,
    max_context_word_tokens: int = 128,
    n_init: int = 1,
    hierarchy_depths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    show_progress: bool = True,
) -> dict[str, object]:
    """Train external MiniBatch K-Means and evaluate predicted CoBaLD clusters."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(drive_dir) if save_models_to_drive else output_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_label(model_label)

    sentences = load_corpus(data_dir, splits=splits)
    cobald_reference_tokens = corpus_to_dataframe(
        sentences,
        include_punctuation=False,
        include_empty=False,
    )
    resolved_n_clusters = _resolve_cluster_count(
        n_clusters,
        tokens=cobald_reference_tokens,
    )

    dataset = load_external_dataset_stream(
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        split=split,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    external_chunks = collect_external_contexts(
        dataset,
        max_train_tokens=max_train_tokens,
        text_column=text_column,
        max_documents=max_documents,
        max_context_tokens=max_context_tokens,
        max_context_word_tokens=max_context_word_tokens,
        show_progress=show_progress,
    )
    external_token_count = sum(chunk.target_count for chunk in external_chunks)
    print(f"external_training_tokens={external_token_count}")
    print(f"external_contexts={len(external_chunks)}")
    print(f"cobald_tokens={len(cobald_reference_tokens)}")
    print(f"n_clusters={resolved_n_clusters}")

    embedding_config = EmbeddingConfig(
        model_name=model_name,
        batch_size=embedding_batch_size,
        max_length=max_length,
        device=device,
        seed=seed,
        normalize=normalize_embeddings,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        include_punctuation_context=include_punctuation_context,
    )
    embedder = _TransformerEmbedder(embedding_config)

    projection: IncrementalPCA | None = None
    if projection_n_components is not None:
        projection = _fit_incremental_pca_from_external(
            embedder,
            external_chunks,
            n_components=projection_n_components,
            embedding_batch_size=embedding_batch_size,
            pca_batch_size=projection_batch_size,
            normalize_input=normalize_projection_input,
            show_progress=show_progress,
        )
        feature_dim = projection_n_components
    else:
        feature_dim = embedder.hidden_size

    training_features_path = artifact_dir / f"{safe_label}_external_features.dat"
    training_features = _write_external_features(
        embedder,
        external_chunks,
        training_features_path,
        projection=projection,
        feature_dim=feature_dim,
        embedding_batch_size=embedding_batch_size,
        normalize_projection_input=normalize_projection_input,
        normalize_projection_output=normalize_projection_output,
        show_progress=show_progress,
    )

    kmeans = fit_minibatch_kmeans_incremental(
        training_features,
        n_clusters=resolved_n_clusters,
        batch_size=kmeans_batch_size,
        random_state=seed,
        n_init=n_init,
        normalize=normalize_for_clustering,
        show_progress=show_progress,
    )

    cobald_features_path = artifact_dir / f"{safe_label}_cobald_features.dat"
    cobald_features, cobald_tokens = _write_cobald_features(
        embedder,
        sentences,
        cobald_features_path,
        projection=projection,
        feature_dim=feature_dim,
        embedding_batch_size=embedding_batch_size,
        include_punctuation_context=include_punctuation_context,
        normalize_projection_input=normalize_projection_input,
        normalize_projection_output=normalize_projection_output,
        show_progress=show_progress,
    )

    labels = _predict_labels_in_batches(
        kmeans,
        cobald_features,
        batch_size=kmeans_batch_size,
        normalize=normalize_for_clustering,
        show_progress=show_progress,
    )

    cluster_config = ClusterConfig(
        algorithm="minibatch_kmeans",
        n_clusters=resolved_n_clusters,
        random_state=seed,
        normalize=normalize_for_clustering,
        batch_size=kmeans_batch_size,
        n_init=n_init,
    )
    result = _build_result(
        cobald_features=cobald_features,
        cobald_tokens=cobald_tokens,
        labels=labels,
        cluster_config=cluster_config,
        hierarchy=hierarchy,
        hierarchy_depths=hierarchy_depths,
    )
    result["model"] = kmeans
    results = [result]

    excel_path = write_cluster_summary_excel(
        results,
        output_dir / f"{safe_label}_cluster_summaries.xlsx",
    )
    hierarchy_alignment_path = write_hierarchy_alignment_excel(
        results,
        output_dir / f"{safe_label}_hierarchy_alignment.xlsx",
    )
    scores_csv, scores_xlsx, scores_txt = _write_scores(
        results,
        output_dir=output_dir,
        label=safe_label,
    )
    clustered_tokens_csv, clustered_tokens_xlsx = _write_clustered_tokens(
        cobald_tokens,
        labels,
        csv_path=output_dir / f"{safe_label}_cobald_clustered_tokens.csv",
        xlsx_path=output_dir / f"{safe_label}_cobald_clustered_tokens.xlsx",
    )

    kmeans_path = _save_pickle(kmeans, artifact_dir / f"{safe_label}_kmeans.pkl")
    projection_path = (
        _save_pickle(projection, artifact_dir / f"{safe_label}_pca.pkl")
        if projection is not None
        else None
    )
    config_path = _save_config(
        {
            "dataset_name": dataset_name,
            "dataset_subset": dataset_subset,
            "split": split,
            "text_column": text_column,
            "max_train_tokens": max_train_tokens,
            "external_training_tokens": external_token_count,
            "external_contexts": len(external_chunks),
            "model_name": model_name,
            "model_label": model_label,
            "embedding_config": asdict(embedding_config),
            "n_clusters": resolved_n_clusters,
            "cluster_config": asdict(cluster_config),
            "projection_n_components": projection_n_components,
            "projection_batch_size": projection_batch_size,
            "normalize_projection_input": normalize_projection_input,
            "normalize_projection_output": normalize_projection_output,
            "max_context_tokens": max_context_tokens,
            "max_context_word_tokens": max_context_word_tokens,
            "hierarchy_depths": list(hierarchy_depths),
        },
        output_dir / f"{safe_label}_config.json",
    )

    paths = ExternalMiniBatchPaths(
        training_features=str(training_features_path),
        cobald_features=str(cobald_features_path),
        kmeans_model=kmeans_path,
        projection_model=projection_path,
        clustered_tokens_csv=clustered_tokens_csv,
        clustered_tokens_xlsx=clustered_tokens_xlsx,
        excel=excel_path,
        hierarchy_alignment_excel=hierarchy_alignment_path,
        scores_csv=scores_csv,
        scores_xlsx=scores_xlsx,
        scores_txt=scores_txt,
        config_json=config_path,
    )
    return {
        "results": results,
        "paths": asdict(paths),
        "config": json.loads(Path(config_path).read_text(encoding="utf-8")),
    }


def run_rubert_external(**kwargs: object) -> dict[str, object]:
    """Run the external pipeline with the default RuBERT base model."""

    defaults = {
        "model_name": DEFAULT_RUBERT_MODEL,
        "model_label": "rubert_base_external_fineweb2",
        "embedding_batch_size": 16,
        "torch_dtype": None,
        "projection_n_components": None,
    }
    defaults.update(kwargs)
    return run(**defaults)


def run_sambalingo_external(**kwargs: object) -> dict[str, object]:
    """Run the external pipeline with SambaLingo Russian Base."""

    defaults = {
        "model_name": DEFAULT_SAMBALINGO_MODEL,
        "model_label": "sambalingo_russian_base_external_fineweb2",
        "embedding_batch_size": 1,
        "torch_dtype": "float16",
        "projection_n_components": 256,
        "projection_batch_size": 8192,
        "normalize_for_clustering": True,
    }
    defaults.update(kwargs)
    return run(**defaults)


def run_gigachat_external(**kwargs: object) -> dict[str, object]:
    """Run the external pipeline with GigaChat3 10B A1.8B base."""

    defaults = {
        "model_name": DEFAULT_GIGACHAT_MODEL,
        "model_label": "gigachat3_10b_a1_8b_base_external_fineweb2",
        "embedding_batch_size": 1,
        "torch_dtype": "bfloat16",
        "trust_remote_code": False,
        "projection_n_components": 256,
        "projection_batch_size": 8192,
        "normalize_for_clustering": True,
    }
    defaults.update(kwargs)
    return run(**defaults)
