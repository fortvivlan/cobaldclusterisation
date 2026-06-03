"""Parsing helpers for the CoBaLD-Russian CoNLL-U-like corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


CONLLU_COLUMNS: tuple[str, ...] = (
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
    "DEEPSLOT",
    "SEMCLASS",
)


@dataclass(slots=True)
class Token:
    """A single CoBaLD token row with all 12 source columns as attributes."""

    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str
    deepslot: str
    semclass: str
    sentence_id: str | None = None
    split: str | None = None
    sentence_index: int | None = None
    token_index: int | None = None

    @property
    def ID(self) -> str:
        return self.id

    @property
    def FORM(self) -> str:
        return self.form

    @property
    def LEMMA(self) -> str:
        return self.lemma

    @property
    def UPOS(self) -> str:
        return self.upos

    @property
    def XPOS(self) -> str:
        return self.xpos

    @property
    def FEATS(self) -> str:
        return self.feats

    @property
    def HEAD(self) -> str:
        return self.head

    @property
    def DEPREL(self) -> str:
        return self.deprel

    @property
    def DEPS(self) -> str:
        return self.deps

    @property
    def MISC(self) -> str:
        return self.misc

    @property
    def DEEPSLOT(self) -> str:
        return self.deepslot

    @property
    def SEMCLASS(self) -> str:
        return self.semclass

    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        *,
        sentence_id: str | None = None,
        split: str | None = None,
        sentence_index: int | None = None,
        token_index: int | None = None,
    ) -> "Token":
        """Create a token from the 12 tab-separated corpus columns."""

        if len(columns) != len(CONLLU_COLUMNS):
            raise ValueError(
                f"Expected {len(CONLLU_COLUMNS)} columns, got {len(columns)}: {columns!r}"
            )
        return cls(
            id=columns[0],
            form=columns[1],
            lemma=columns[2],
            upos=columns[3],
            xpos=columns[4],
            feats=columns[5],
            head=columns[6],
            deprel=columns[7],
            deps=columns[8],
            misc=columns[9],
            deepslot=columns[10],
            semclass=columns[11],
            sentence_id=sentence_id,
            split=split,
            sentence_index=sentence_index,
            token_index=token_index,
        )

    @property
    def is_punctuation(self) -> bool:
        """Whether the token is punctuation according to UPOS."""

        return self.upos.upper() == "PUNCT"

    @property
    def is_empty_node(self) -> bool:
        """Whether the token is an empty/ellipsis row such as ``2.1 #NULL``."""

        return "." in self.id or self.form == "#NULL"

    @property
    def is_multiword_token(self) -> bool:
        """Whether the row is a CoNLL-U multiword-token range."""

        return "-" in self.id

    @property
    def has_semclass(self) -> bool:
        """Whether a manual semantic class is present."""

        return self.semclass not in {"", "_"}

    @property
    def is_surface_token(self) -> bool:
        """Whether this row corresponds to an explicit surface token."""

        return not self.is_empty_node and not self.is_multiword_token

    def to_dict(self, *, context_text: str | None = None) -> dict[str, object]:
        """Return a DataFrame-friendly representation with original column names."""

        row: dict[str, object] = {
            "ID": self.id,
            "FORM": self.form,
            "LEMMA": self.lemma,
            "UPOS": self.upos,
            "XPOS": self.xpos,
            "FEATS": self.feats,
            "HEAD": self.head,
            "DEPREL": self.deprel,
            "DEPS": self.deps,
            "MISC": self.misc,
            "DEEPSLOT": self.deepslot,
            "SEMCLASS": self.semclass,
            "sentence_id": self.sentence_id,
            "split": self.split,
            "sentence_index": self.sentence_index,
            "token_index": self.token_index,
            "is_punctuation": self.is_punctuation,
            "is_empty_node": self.is_empty_node,
        }
        if context_text is not None:
            row["context_text"] = context_text
        return row


@dataclass(slots=True)
class Sentence:
    """A parsed sentence with metadata comments and ordered token rows."""

    tokens: list[Token]
    metadata: dict[str, str] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    split: str | None = None
    index: int | None = None

    @property
    def sent_id(self) -> str | None:
        """Sentence id from the CoNLL-U comment metadata, if present."""

        return self.metadata.get("sent_id")

    @property
    def text(self) -> str:
        """Sentence text from comments, or a simple reconstruction as fallback."""

        if "text" in self.metadata:
            return self.metadata["text"]
        parts: list[str] = []
        for token in self.surface_tokens(include_punctuation=True):
            parts.append(token.form)
        return " ".join(parts)

    def surface_tokens(self, *, include_punctuation: bool = True) -> list[Token]:
        """Return explicit surface tokens, optionally excluding punctuation."""

        return [
            token
            for token in self.tokens
            if token.is_surface_token
            and (include_punctuation or not token.is_punctuation)
        ]

    def context_text(self, *, include_punctuation: bool = True) -> str:
        """Return a tokenized context string for embedding experiments."""

        if include_punctuation and "text" in self.metadata:
            return self.metadata["text"]
        return " ".join(
            token.form
            for token in self.surface_tokens(include_punctuation=include_punctuation)
        )

    def embedding_targets(self) -> list[Token]:
        """Return default embedding targets: surface non-punctuation tokens."""

        return self.surface_tokens(include_punctuation=False)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert the sentence tokens to a pandas DataFrame."""

        return tokens_to_dataframe(self.tokens, context_text=self.text)


def parse_conllu(path: str | Path, *, split: str | None = None) -> list[Sentence]:
    """Parse a CoBaLD CoNLL-U-like file into sentences.

    The corpus has exactly 12 tab-separated token columns. Comment lines are
    preserved and comments of the form ``# key = value`` are exposed as metadata.
    """

    path = Path(path)
    sentences: list[Sentence] = []
    comments: list[str] = []
    metadata: dict[str, str] = {}
    token_rows: list[list[str]] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal comments, metadata, token_rows, start_line
        if not comments and not token_rows:
            start_line = end_line + 1
            return
        sentence_index = len(sentences)
        sent_id = metadata.get("sent_id")
        tokens = [
            Token.from_columns(
                row,
                sentence_id=sent_id,
                split=split,
                sentence_index=sentence_index,
                token_index=token_index,
            )
            for token_index, row in enumerate(token_rows)
        ]
        sentences.append(
            Sentence(
                tokens=tokens,
                metadata=dict(metadata),
                comments=list(comments),
                split=split,
                index=sentence_index,
            )
        )
        comments = []
        metadata = {}
        token_rows = []
        start_line = end_line + 1

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                flush(line_number)
                continue
            if line.startswith("#"):
                comments.append(line)
                body = line[1:].strip()
                if " = " in body:
                    key, value = body.split(" = ", 1)
                    metadata[key.strip()] = value.strip()
                continue
            columns = line.split("\t")
            if len(columns) != len(CONLLU_COLUMNS):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(CONLLU_COLUMNS)} "
                    f"columns, got {len(columns)}. Sentence started at line {start_line}."
                )
            token_rows.append(columns)

    flush(line_number if "line_number" in locals() else 0)
    return sentences


def load_corpus(
    data_dir: str | Path = "CobaldRus",
    *,
    splits: Sequence[str] = ("train", "dev"),
) -> list[Sentence]:
    """Load and merge the requested corpus splits from ``data_dir``."""

    data_dir = Path(data_dir)
    sentences: list[Sentence] = []
    for split in splits:
        sentences.extend(parse_conllu(data_dir / f"{split}.conllu", split=split))
    return sentences


def iter_tokens(
    sentences: Iterable[Sentence],
    *,
    include_punctuation: bool = False,
    include_empty: bool = False,
) -> Iterator[Token]:
    """Iterate tokens using the default project filtering policy."""

    for sentence in sentences:
        for token in sentence.tokens:
            if not include_empty and not token.is_surface_token:
                continue
            if not include_punctuation and token.is_punctuation:
                continue
            yield token


def tokens_to_dataframe(
    tokens: Iterable[Token],
    *,
    context_text: str | None = None,
) -> "pd.DataFrame":
    """Convert tokens to a pandas DataFrame."""

    import pandas as pd

    return pd.DataFrame([token.to_dict(context_text=context_text) for token in tokens])


def corpus_to_dataframe(
    sentences: Iterable[Sentence],
    *,
    include_punctuation: bool = True,
    include_empty: bool = True,
) -> "pd.DataFrame":
    """Convert a sentence collection to one token table."""

    import pandas as pd

    rows: list[dict[str, object]] = []
    for sentence in sentences:
        for token in sentence.tokens:
            if not include_empty and not token.is_surface_token:
                continue
            if not include_punctuation and token.is_punctuation:
                continue
            rows.append(token.to_dict(context_text=sentence.text))
    return pd.DataFrame(rows)


def load_semclass_hierarchy(path: str | Path) -> "pd.DataFrame":
    """Load the first four columns of ``hyperonims_hierarchy.csv``.

    The source file has no header. The returned columns are ``class_id``,
    ``parent_id``, ``depth``, and ``class_name``.
    """

    import pandas as pd

    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, columns in enumerate(reader, start=1):
            if not columns:
                continue
            if len(columns) < 4:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 4 CSV columns, "
                    f"got {len(columns)}"
                )
            parent_id = columns[1] if columns[1] not in {"", "NULL"} else None
            rows.append(
                {
                    "class_id": columns[0],
                    "parent_id": parent_id,
                    "depth": int(columns[2]),
                    "class_name": columns[3],
                }
            )
    hierarchy = pd.DataFrame(rows, dtype=object)
    if not hierarchy.empty:
        hierarchy["depth"] = hierarchy["depth"].astype(int)
    return hierarchy
