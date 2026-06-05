"""Contextual embedding generation for parsed CoBaLD sentences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import logging
from pathlib import Path
import pickle
import random
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data import Sentence, Token, load_corpus


DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"


@dataclass(slots=True)
class EmbeddingConfig:
    """Parameters for contextual token embedding generation."""

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 8
    max_length: int = 512
    layer: int = -1
    device: str | None = None
    seed: int = 42
    normalize: bool = False
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    include_punctuation_context: bool = True


def _require_transformers() -> tuple[object, object, object]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Embedding generation requires the optional dependencies "
            "'torch' and 'transformers'. Install with: "
            "pip install -e '.[embeddings]'"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _torch_dtype_from_name(torch: object, name: str | None) -> object | None:
    if name is None:
        return None
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc


def _model_load_kwargs(
    *,
    trust_remote_code: bool,
    dtype: object | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def _coerce_rope_parameter_floats(value: object) -> None:
    """Coerce known RoPE config integers that recent Transformers wants as floats."""

    if isinstance(value, dict):
        for key in ("beta_fast", "beta_slow"):
            item = value.get(key)
            if isinstance(item, int) and not isinstance(item, bool):
                value[key] = float(item)
        for item in value.values():
            _coerce_rope_parameter_floats(item)
        return

    for attr in ("rope_parameters", "rope_scaling"):
        nested = getattr(value, attr, None)
        if nested is not None:
            _coerce_rope_parameter_floats(nested)


class _RopeParameterTypeWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            "`rope_parameters`'s beta_fast field must be a float" in message
            or "`rope_parameters`'s beta_slow field must be a float" in message
        )


@contextmanager
def _suppress_rope_parameter_type_warnings() -> Iterator[None]:
    logger = logging.getLogger("transformers.modeling_rope_utils")
    warning_filter = _RopeParameterTypeWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def _load_auto_model(
    model_name: str,
    *,
    trust_remote_code: bool,
    dtype: object | None,
) -> object:
    from transformers import AutoConfig, AutoModel

    with _suppress_rope_parameter_type_warnings():
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    _coerce_rope_parameter_floats(config)
    return AutoModel.from_pretrained(
        model_name,
        config=config,
        **_model_load_kwargs(trust_remote_code=trust_remote_code, dtype=dtype),
    )


def _resolve_device(torch: object, device: str | None) -> object:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sentence_context_tokens(
    sentence: Sentence,
    *,
    include_punctuation: bool,
) -> list[Token]:
    return sentence.surface_tokens(include_punctuation=include_punctuation)


def _target_word_indices(context_tokens: Sequence[Token]) -> list[int]:
    return [
        index
        for index, token in enumerate(context_tokens)
        if not token.is_punctuation
    ]


def _batch_slices(items: Sequence[Sentence], batch_size: int) -> Iterable[Sequence[Sentence]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _select_hidden_state(outputs: object, *, layer: int) -> object:
    if layer == -1:
        return outputs.last_hidden_state
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise ValueError("Model did not return hidden states for layer selection.")
    return outputs.hidden_states[layer]


def _ensure_padding_token(tokenizer: object) -> None:
    """Allow batched embedding extraction with decoder-only tokenizers."""

    if getattr(tokenizer, "pad_token", None) is not None:
        return
    fallback_token = getattr(tokenizer, "eos_token", None) or getattr(
        tokenizer,
        "unk_token",
        None,
    )
    if fallback_token is None:
        raise ValueError(
            "Tokenizer has no pad_token, eos_token, or unk_token. Set a padding "
            "token before generating batched embeddings."
        )
    tokenizer.pad_token = fallback_token


def generate_token_embeddings(
    sentences: Sequence[Sentence],
    *,
    config: EmbeddingConfig | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Generate contextual embeddings for surface non-punctuation tokens.

    Punctuation can be kept or removed from the contextual input. Empty/ellipsis
    rows such as ``#NULL`` are not sent to the tokenizer and are not embedding
    targets.
    """

    config = config or EmbeddingConfig()
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    torch, _AutoModel, AutoTokenizer = _require_transformers()
    _set_seed(config.seed)
    device = _resolve_device(torch, config.device)
    dtype = _torch_dtype_from_name(torch, config.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "A fast tokenizer is required because token-to-subword alignment "
            "uses word_ids()."
        )
    _ensure_padding_token(tokenizer)

    model = _load_auto_model(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        dtype=dtype,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    embedding_sentences = [
        sentence
        for sentence in sentences
        if _sentence_context_tokens(
            sentence,
            include_punctuation=config.include_punctuation_context,
        )
    ]

    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    iterator = _batch_slices(embedding_sentences, config.batch_size)
    if show_progress:
        total = (len(embedding_sentences) + config.batch_size - 1) // config.batch_size
        iterator = tqdm(iterator, total=total, desc="Embedding sentences")

    with torch.no_grad():
        for batch in iterator:
            context_tokens_by_sentence = [
                _sentence_context_tokens(
                    sentence,
                    include_punctuation=config.include_punctuation_context,
                )
                for sentence in batch
            ]
            words_by_sentence = [
                [token.form for token in context_tokens]
                for context_tokens in context_tokens_by_sentence
            ]
            encoded = tokenizer(
                words_by_sentence,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = model(
                **model_inputs,
                output_hidden_states=config.layer != -1,
                return_dict=True,
            )
            hidden = _select_hidden_state(outputs, layer=config.layer)

            for batch_index, (sentence, context_tokens) in enumerate(
                zip(batch, context_tokens_by_sentence, strict=True)
            ):
                word_ids = encoded.word_ids(batch_index=batch_index)
                embedding_context_text = sentence.context_text(
                    include_punctuation=config.include_punctuation_context
                )
                for word_index in _target_word_indices(context_tokens):
                    token = context_tokens[word_index]
                    piece_positions = [
                        position
                        for position, aligned_word_index in enumerate(word_ids)
                        if aligned_word_index == word_index
                    ]
                    if not piece_positions:
                        raise ValueError(
                            "Tokenizer alignment produced no subword pieces for "
                            f"{token.form!r} in sentence {sentence.sent_id!r}. "
                            "Increase max_length or inspect tokenization."
                        )
                    token_hidden = hidden[batch_index, piece_positions, :].mean(dim=0)
                    vector = (
                        token_hidden.detach()
                        .to(dtype=torch.float32)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    if config.normalize:
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm
                    rows.append(
                        {
                            **token.to_dict(context_text=sentence.text),
                            "embedding_context_text": embedding_context_text,
                            "embedding_index": len(rows),
                        }
                    )
                    vectors.append(vector)

    embeddings = (
        np.vstack(vectors).astype(np.float32)
        if vectors
        else np.empty((0, int(getattr(model.config, "hidden_size", 0))), dtype=np.float32)
    )
    token_table = pd.DataFrame(rows)
    return {
        "embeddings": embeddings,
        "tokens": token_table,
        "config": {
            **asdict(config),
            "target_policy": "surface_non_punctuation_tokens",
            "context_policy": (
                "surface_tokens_with_punctuation"
                if config.include_punctuation_context
                else "surface_tokens_without_punctuation"
            ),
        },
    }


def save_embeddings_pickle(
    payload: dict[str, object],
    output_path: str | Path,
    *,
    drive_dir: str | Path | None = None,
) -> Path:
    """Save an embedding payload to a pickle file, optionally under Google Drive."""

    output_path = Path(output_path)
    if drive_dir is not None:
        output_path = Path(drive_dir) / output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def load_embeddings_pickle(path: str | Path) -> dict[str, object]:
    """Load an embedding payload saved by :func:`save_embeddings_pickle`."""

    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def generate_embeddings_from_corpus(
    *,
    data_dir: str | Path = "CobaldRus",
    output_path: str | Path = "outputs/embeddings/rubert_tiny2.pkl",
    drive_dir: str | Path | None = None,
    splits: Sequence[str] = ("train", "dev"),
    config: EmbeddingConfig | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Load train/dev, generate embeddings, save them, and return the payload."""

    sentences = load_corpus(data_dir, splits=splits)
    payload = generate_token_embeddings(
        sentences,
        config=config,
        show_progress=show_progress,
    )
    saved_path = save_embeddings_pickle(payload, output_path, drive_dir=drive_dir)
    payload["saved_path"] = str(saved_path)
    return payload
