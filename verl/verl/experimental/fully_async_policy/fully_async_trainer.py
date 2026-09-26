"""Load fully_async_trainer from compiled bytecode (source was removed)."""
from __future__ import annotations

import marshal
from pathlib import Path

_here = Path(__file__).resolve()
_pyc = next(
    (_root / "_bytecode_backup" / "fully_async_trainer.pyc")
    for _root in _here.parents
    if (_root / "_bytecode_backup" / "fully_async_trainer.pyc").is_file()
)
with open(_pyc, "rb") as _f:
    _data = _f.read()
_code = marshal.loads(_data[16:])
del _here, _pyc, _f, _data, marshal, Path
exec(_code, globals())
