"""Excel export helpers for cluster summary tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .scores import _run_part


SEMCLASS_LEMMA_EXAMPLES_COLUMN = "semclass_lemma_examples"


def _summary_for_excel(summary: pd.DataFrame) -> pd.DataFrame:
    if SEMCLASS_LEMMA_EXAMPLES_COLUMN in summary.columns:
        return summary

    export_summary = summary.copy()
    insert_at = (
        export_summary.columns.get_loc("semclasses") + 1
        if "semclasses" in export_summary.columns
        else len(export_summary.columns)
    )
    export_summary.insert(insert_at, SEMCLASS_LEMMA_EXAMPLES_COLUMN, "")
    return export_summary


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
            cluster_part = _run_part(config)
            sheet_name = f"{index}_{algorithm}_{cluster_part}"[:31]
            _summary_for_excel(result["summary"]).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
            worksheet = writer.sheets[sheet_name]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
    return str(output_path)


def write_hierarchy_alignment_excel(
    results: Sequence[dict[str, object]],
    output_path: str | Path,
) -> str:
    """Write hierarchy-alignment tables for all clustering runs."""

    try:
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it in Colab with: "
            "pip install openpyxl"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_sheet = False
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for index, result in enumerate(results):
            alignment = result.get("hierarchy_alignment")
            if not isinstance(alignment, pd.DataFrame) or alignment.empty:
                continue
            config = result["config"]
            algorithm = str(config["algorithm"])
            sheet_name = f"{index}_{algorithm}_{_run_part(config)}"[:31]
            alignment.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
            wrote_sheet = True
        if not wrote_sheet:
            pd.DataFrame().to_excel(writer, sheet_name="alignment", index=False)
    return str(output_path)
