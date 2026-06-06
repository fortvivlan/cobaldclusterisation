from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from cobaldclusterisation.summary_excel import write_cluster_summary_excel


def test_write_cluster_summary_excel_adds_semclass_lemma_examples_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.xlsx"
    write_cluster_summary_excel(
        [
            {
                "config": {"algorithm": "kmeans", "n_clusters": 2},
                "summary": pd.DataFrame(
                    {
                        "cluster": [0],
                        "semclasses": ["CLASS: 1"],
                    }
                ),
            }
        ],
        path,
    )

    workbook = load_workbook(path)
    worksheet = workbook.active
    headers = [cell.value for cell in worksheet[1]]

    assert headers == ["cluster", "semclasses", "semclass_lemma_examples"]
