from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

pytest.importorskip("nltk")

from cobaldclusterisation.collocations import (
    build_collocation_tables,
    collocation_table_from_documents,
    sentence_token_documents,
    write_collocation_excel,
)
from cobaldclusterisation.data import Sentence, Token


def _token(
    form: str,
    lemma: str,
    *,
    token_id: str = "1",
    upos: str = "NOUN",
) -> Token:
    return Token.from_columns(
        [
            token_id,
            form,
            lemma,
            upos,
            "_",
            "_",
            "0",
            "root",
            "_",
            "_",
            "_",
            "_",
        ]
    )


def test_sentence_token_documents_do_not_cross_sentence_boundaries() -> None:
    sentences = [
        Sentence(
            tokens=[
                _token("Нью", "нью", token_id="1"),
                _token("Йорк", "йорк", token_id="2"),
            ]
        ),
        Sentence(
            tokens=[
                _token("Сити", "сити", token_id="1"),
                _token("Холл", "холл", token_id="2"),
            ]
        ),
    ]

    documents = sentence_token_documents(sentences, basis="lemma")
    table = collocation_table_from_documents(
        documents,
        ngram_size=2,
        min_freq=1,
    )

    assert "йорк сити" not in set(table["ngram"])
    assert {"нью йорк", "сити холл"}.issubset(set(table["ngram"]))


def test_build_collocation_tables_separates_lemmas_and_surface_forms() -> None:
    sentences = [
        Sentence(
            tokens=[
                _token("Белые", "белый", token_id="1", upos="ADJ"),
                _token("дома", "дом", token_id="2"),
            ]
        ),
        Sentence(
            tokens=[
                _token("белого", "белый", token_id="1", upos="ADJ"),
                _token("дома", "дом", token_id="2"),
            ]
        ),
    ]

    tables = build_collocation_tables(
        sentences,
        token_bases=("lemma", "form"),
        ngram_sizes=(2,),
        min_freq=2,
    )

    assert tables["lemma_bigrams"].loc[0, "ngram"] == "белый дом"
    assert tables["lemma_bigrams"].loc[0, "frequency"] == 2
    assert tables["form_bigrams"].empty


def test_collocation_table_applies_min_freq_and_sorts_by_pmi() -> None:
    documents = [
        ["a", "b"],
        ["a", "b"],
        ["a", "c"],
        ["x", "y"],
        ["x", "y"],
        ["x", "y"],
    ]

    table = collocation_table_from_documents(
        documents,
        ngram_size=2,
        min_freq=2,
        sort_by="pmi",
    )

    assert set(table["ngram"]) == {"a b", "x y"}
    assert "a c" not in set(table["ngram"])
    assert table["pmi"].is_monotonic_decreasing


def test_write_collocation_excel_creates_expected_sheets(tmp_path: Path) -> None:
    output_path = tmp_path / "collocations.xlsx"
    tables = {
        "lemma_bigrams": pd.DataFrame(
            {
                "token_basis": ["lemma"],
                "ngram_size": [2],
                "ngram": ["белый дом"],
                "frequency": [2],
                "pmi": [1.0],
            }
        )
    }
    metadata = pd.DataFrame({"key": ["min_freq"], "value": [2]})

    write_collocation_excel(tables, output_path, metadata=metadata)

    workbook = load_workbook(output_path)

    assert set(workbook.sheetnames) == {"lemma_bigrams", "metadata"}
    assert workbook["lemma_bigrams"].freeze_panes == "A2"
    assert workbook["metadata"].freeze_panes == "A2"
