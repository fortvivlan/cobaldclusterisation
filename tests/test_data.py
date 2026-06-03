from pathlib import Path

import pytest

from cobaldclusterisation.data import (
    corpus_to_dataframe,
    iter_tokens,
    load_semclass_hierarchy,
    parse_conllu,
)
from cobaldclusterisation.resources import resolve_hierarchy_path


CONLLU_FIXTURE = """# sent_id = s1
# text = А, тест.
1\tА\tа\tNOUN\tNoun\t_\t0\troot\t0:root\t_\tObject\tLETTER
2\t,\t,\tPUNCT\tPUNCT\t_\t1\tpunct\t1:punct\t_\t_\t_
2.1\t#NULL\tбыть\tVERB\tVerb\t_\t_\t_\t0:root\tellipsis\tPredicate\tBE
3\tтест\tтест\tNOUN\tNoun\t_\t1\tconj\t1:conj\tSpaceAfter=No\tObject\tEVENT
4\t.\t.\tPUNCT\tPUNCT\t_\t1\tpunct\t1:punct\t_\t_\t_

"""


def test_parse_conllu_preserves_columns_and_filters_targets(tmp_path: Path) -> None:
    path = tmp_path / "sample.conllu"
    path.write_text(CONLLU_FIXTURE, encoding="utf-8")

    sentences = parse_conllu(path, split="train")

    assert len(sentences) == 1
    sentence = sentences[0]
    assert sentence.sent_id == "s1"
    assert sentence.text == "А, тест."
    assert len(sentence.tokens) == 5
    assert sentence.tokens[0].form == "А"
    assert sentence.tokens[0].FORM == "А"
    assert sentence.tokens[0].semclass == "LETTER"
    assert sentence.tokens[0].SEMCLASS == "LETTER"
    assert sentence.tokens[2].is_empty_node

    assert [token.form for token in sentence.surface_tokens(include_punctuation=True)] == [
        "А",
        ",",
        "тест",
        ".",
    ]
    assert [token.form for token in sentence.embedding_targets()] == ["А", "тест"]
    assert [token.form for token in iter_tokens(sentences)] == ["А", "тест"]
    assert sentence.context_text(include_punctuation=True) == "А, тест."
    assert sentence.context_text(include_punctuation=False) == "А тест"

    frame = corpus_to_dataframe(sentences)
    assert list(frame["FORM"]) == ["А", ",", "#NULL", "тест", "."]
    assert frame.loc[0, "context_text"] == "А, тест."


def test_parse_conllu_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.conllu"
    path.write_text("1\ttoo\tfew\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 12 columns"):
        parse_conllu(path)


def test_load_semclass_hierarchy_reads_first_four_columns(tmp_path: Path) -> None:
    path = tmp_path / "hierarchy.csv"
    path.write_text(
        "1,NULL,0,ROOT,0,1\n2,1,1,CHILD,0,1\n",
        encoding="utf-8",
    )

    hierarchy = load_semclass_hierarchy(path)

    assert list(hierarchy.columns) == ["class_id", "parent_id", "depth", "class_name"]
    assert hierarchy.loc[0, "parent_id"] is None
    assert hierarchy.loc[1, "depth"] == 1
    assert hierarchy.loc[1, "class_name"] == "CHILD"


def test_resolve_hierarchy_path_accepts_repo_directory(tmp_path: Path) -> None:
    repo_dir = tmp_path / "semantic-hierarchy"
    repo_dir.mkdir()
    hierarchy_path = repo_dir / "hyperonims_hierarchy.csv"
    hierarchy_path.write_text("1,NULL,0,ROOT,0,1\n", encoding="utf-8")

    assert resolve_hierarchy_path(repo_dir) == hierarchy_path
