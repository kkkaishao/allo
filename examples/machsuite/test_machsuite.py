# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def _iter_tests():
    package_path = Path(__file__).resolve().parent
    prefix = f"{__package__}."
    for module_info in pkgutil.walk_packages([str(package_path)], prefix):
        if module_info.name == __name__:
            continue
        module = importlib.import_module(module_info.name)
        module_label = module_info.name.removeprefix(prefix).replace(".", "_")
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            test = getattr(module, name)
            if callable(test):
                yield f"test_{module_label}_{name[5:]}", test


for _name, _test in _iter_tests():
    _test.__name__ = _name
    _test.__qualname__ = _name
    globals()[_name] = _test
