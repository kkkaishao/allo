# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=c-extension-no-member

import ctypes

from . import _liballo

_backend = _liballo._load_submodule("execution_engine")


class ExecutionEngine:
    def __init__(
        self,
        module,
        opt_level=2,
        shared_libs=(),
        enable_object_dump=False,
        enable_pic=False,
    ):
        self._engine = _backend.ExecutionEngine(
            module,
            opt_level,
            list(shared_libs),
            enable_object_dump,
            enable_pic,
        )
        self._callbacks = []

    def raw_lookup(self, name: str) -> int:
        return self._engine.raw_lookup(name)

    def lookup(self, name: str):
        func = self.raw_lookup("_mlir_ciface_" + name)
        if not func:
            raise RuntimeError(f"Unknown function {name}")
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        return prototype(func)

    def invoke(self, name: str, *ctypes_args):
        func = self.lookup(name)
        packed_args = (ctypes.c_void_p * len(ctypes_args))()
        for i, arg in enumerate(ctypes_args):
            packed_args[i] = ctypes.cast(arg, ctypes.c_void_p)
        func(packed_args)

    def raw_register_runtime(self, name: str, addr: int):
        self._engine.raw_register_runtime(name, addr)

    def register_runtime(self, name: str, ctypes_callback):
        callback = ctypes.cast(ctypes_callback, ctypes.c_void_p)
        self._callbacks.append(ctypes_callback)
        self.raw_register_runtime("_mlir_ciface_" + name, callback.value)

    def initialize(self):
        self._engine.initialize()

    def dump_to_object_file(self, file_name: str):
        self._engine.dump_to_object_file(file_name)


__all__ = ["ExecutionEngine"]
