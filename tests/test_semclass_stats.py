import pandas as pd

from cobaldclusterisation.semclass_stats import (
    build_semclass_threshold_table,
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
