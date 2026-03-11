from . import _liballo as _C

_mod = _C._load_submodule("utils")
globals().update(_mod.__dict__)
