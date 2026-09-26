"""Load qwen3_vl from compiled bytecode (source was removed)."""
from __future__ import annotations

import marshal
from pathlib import Path

_here = Path(__file__).resolve()
_pyc = next(
    (_root / "_bytecode_backup" / "qwen3_vl.pyc")
    for _root in _here.parents
    if (_root / "_bytecode_backup" / "qwen3_vl.pyc").is_file()
)
with open(_pyc, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _here, _pyc, _f, _data, marshal, Path
exec(_code, globals())
