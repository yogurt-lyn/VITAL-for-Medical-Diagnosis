"""Load visual-merger distill helpers from compiled bytecode (source was removed)."""
from __future__ import annotations

import marshal
from pathlib import Path

_pyc = Path(__file__).resolve().parent / "__pycache__" / "visual_merger_distill.cpython-312.pyc"
with open(_pyc, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _pyc, _f, _data, marshal, Path
exec(_code, globals())
