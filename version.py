# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools_scm import get_version


def dynamic_metadata(field, settings=None):
    assert field == "version"
    ver = get_version(local_scheme="no-local-version")
    if os.environ.get("ALLO_ENABLE_ORTOOLS", None) is not None:
        ver = ver + "+ortools"
    return ver


def get_requires_for_dynamic_metadata(settings=None):
    return ["setuptools_scm"]
