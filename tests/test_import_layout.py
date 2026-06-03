import subprocess
import sys
from pathlib import Path


def test_import_from_parent_directory_resolves_src_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent = repo_root.parent
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(parent)!r}); "
        "from cobaldclusterisation import corpus_to_dataframe, ensure_external_data, load_corpus; "
        "print(corpus_to_dataframe.__name__, ensure_external_data.__name__, load_corpus.__name__)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "corpus_to_dataframe ensure_external_data load_corpus"
