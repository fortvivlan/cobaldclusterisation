import pandas as pd

from cobaldclusterisation.semclass_stats import (
    build_semclass_threshold_table,
    hierarchy_semclass_count,
    semclass_counts,
)


def test_semclass_counts_ignores_missing_labels() -> None:
    tokens = pd.DataFrame(
        {
            "SEMCLASS": [
                "ANIMAL",
                "ANIMAL",
                "MONEY",
                "_",
                "",
                None,
            ]
        }
    )

    counts = semclass_counts(tokens)

    assert counts.to_dict() == {"ANIMAL": 2, "MONEY": 1}


def test_build_semclass_threshold_table_is_cumulative() -> None:
    counts = pd.Series({"ANIMAL": 3, "MONEY": 2, "PLACE": 1})

    table = build_semclass_threshold_table(counts, max_occurrences=3)

    assert table["minimum_occurrences"].tolist() == [1, 2, 3]
    assert table["semclass_count"].tolist() == [3, 2, 1]
    assert table["labeled_token_count"].tolist() == [6, 5, 3]


def test_build_semclass_threshold_table_can_include_existing_count() -> None:
    counts = pd.Series({"ANIMAL": 3, "MONEY": 2, "PLACE": 1})

    table = build_semclass_threshold_table(
        counts,
        max_occurrences=1,
        existing_semclass_count=10,
    )

    assert table["existing_semclass_count"].tolist() == [10]


def test_hierarchy_semclass_count_accepts_repo_directory(tmp_path) -> None:
    hierarchy_dir = tmp_path / "semantic-hierarchy"
    hierarchy_dir.mkdir()
    hierarchy_path = hierarchy_dir / "hyperonims_hierarchy.csv"
    hierarchy_path.write_text(
        "1,NULL,0,ROOT,0,1\n2,1,1,ANIMAL,0,1\n3,1,1,MONEY,0,1\n",
        encoding="utf-8",
    )

    assert hierarchy_semclass_count(hierarchy_dir) == 3
