"""Shim: load recovered module bytecode from verl/_bytecode_backup/losses.pyc."""
from __future__ import annotations

import marshal
from pathlib import Path

_backup = str(Path(__file__).resolve().parents[2] / "_bytecode_backup" / "losses.pyc")
with open(_backup, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _backup, _f, _data, marshal, Path
exec(_code, globals())
