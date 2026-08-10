# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether the RTL emitter still reads the source IR.

The backend's founding rule is that the L2 `Datapath` is a CLOSED, immutable
artifact and the emitter is a pure function of it. Purity is already a compiler
check: the emitter takes `const Datapath &`. Closure is not, because holding an
`Operation *` for provenance is legitimate while asking it a question is not.

So this bans the READ set: an emitter translation unit may not pull a fact out
of the IR it is lowering. Every such fact belongs on the model, frozen when the
model was built, or the same number has two homes and they can disagree. Two
real costs paid for that before the rule existed: an operator latency resolved
by symbol lookup at emit time, which pinned `cleanupDcpOps` to the very end and
made block order load-bearing; and a schedule cycle re-stamped mid-build, which
switched off the emitter's own model-against-hardware check.

Writing an attribute is NOT banned: the emitter stamps `seq.hlmem` and
`seq.read`/`seq.write` ops it has just built, which is hardware construction
rather than a read of the source.

    python3 scripts/lint/check_emitter_closure.py
"""

import os
import re
import sys

# The emit half (E10). Everything else in `Microarch` is above the seal and may
# read the IR: the builder derives the model FROM it, and verification reports
# on it.
FILES = (
    "mlir/include/allo/Microarch/HWEmitter.h",
    "mlir/include/allo/Microarch/Primitives.h",
    "mlir/lib/allo/Microarch/ControlEmitter.cpp",
    "mlir/lib/allo/Microarch/DatapathEmitter.cpp",
    "mlir/lib/allo/Microarch/MemoryEmitter.cpp",
    "mlir/lib/allo/Microarch/HWEmitter.cpp",
    "mlir/lib/allo/Microarch/Primitives.cpp",
)

BANNED = (
    (r"\bgetAttr\s*\(", "reads an attribute off the IR; put the fact on the model"),
    (
        r"\bgetAttrOfType\s*<",
        "reads an attribute off the IR; put the fact on the model",
    ),
    (r"\bhasAttr\s*\(", "asks the IR a question; put the fact on the model"),
    (r"\bSymbolTable::lookup", "resolves a symbol at emit time; copy what it says"),
    (r"\bdcpStart\s*\(", "reads the schedule off the IR; use the cell's `stage`"),
    (r"\bdcpLatency\s*\(", "reads the schedule off the IR; use the cell's latency"),
    (r"\breadyCycleOf\s*\(", "reads the schedule off the IR; use `readyCycle`"),
)

# A `//` or `///` line, so prose naming a banned symbol does not fail the check.
COMMENT = re.compile(r"^\s*(//|\*|/\*)")


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    found = []
    for rel in FILES:
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if COMMENT.match(line):
                    continue
                for pattern, why in BANNED:
                    if re.search(pattern, line):
                        found.append((rel, n, line.strip(), why))
    for rel, n, text, why in found:
        print(f"{rel}:{n}: {why}\n    {text}", file=sys.stderr)
    if found:
        print(
            f"\n{len(found)} emitter closure violation(s): the emitter is not a "
            f"pure function of the sealed model.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Emitter closure: {len(FILES)} files clean")


if __name__ == "__main__":
    main()
