"""Excel export helpers for cluster summary tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def write_cluster_summary_excel(
    results: Sequence[dict[str, object]],
    output_path: str | Path,
) -> str:
    """Write all cluster summaries to one Excel workbook."""

    try:
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it in Colab with: "
            "pip install openpyxl"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for index, result in enumerate(results):
            config = result["config"]
            algorithm = str(config["algorithm"])
            cluster_part = (
                f"min{config.get('min_cluster_size')}"
                if algorithm == "hdbscan"
                else f"k{config.get('n_clusters')}"
            )
            sheet_name = f"{index}_{algorithm}_{cluster_part}"[:31]
            result["summary"].to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
    return str(output_path)
