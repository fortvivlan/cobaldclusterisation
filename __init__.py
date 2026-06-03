"""Import shim for running notebooks from the parent of this repository.

The project uses a ``src/`` layout. In Colab, importing from the parent
directory of the clone can otherwise resolve this repository directory as an
empty namespace package. This shim forwards that import to the real package.
"""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent / "src" / "cobaldclusterisation"
_SRC_INIT = _SRC_PACKAGE / "__init__.py"

if not _SRC_INIT.is_file():
    raise ImportError(f"Cannot locate source package at {_SRC_INIT}")

__path__.insert(0, str(_SRC_PACKAGE))
exec(compile(_SRC_INIT.read_text(encoding="utf-8"), str(_SRC_INIT), "exec"), globals())
