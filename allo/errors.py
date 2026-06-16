# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class AlloError(Exception):
    """Base class for all Allo exceptions."""


class AlloFatalError(AlloError):
    """A fatal, user-facing error.

    Terminates the process via ``SystemExit`` in CLI contexts, but is raised
    as a regular exception under notebook mode so it does not abruptly stop a
    Jupyter kernel. See :mod:`allo.logging`.
    """

    def __init__(self, message: str, *, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code
