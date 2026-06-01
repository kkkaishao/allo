/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Declarations for the Allo type/attribute binding populators, implemented in
 * AlloTypes.cpp / AlloAttrs.cpp and invoked from AlloModule.cpp's NB_MODULE.
 */

#ifndef ALLO_BINDINGS_ALLOBINDINGS_H
#define ALLO_BINDINGS_ALLOBINDINGS_H

#include "nanobind/nanobind.h"

namespace allo {
void populateAlloTypes(nanobind::module_ &m);
void populateAlloAttrs(nanobind::module_ &m);
} // namespace allo

#endif // ALLO_BINDINGS_ALLOBINDINGS_H
