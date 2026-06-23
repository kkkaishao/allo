# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-harness generation for the 3-CU LLaMA decoder-layer demo."""

import os
import shutil

from hardware import cu_meta, LINKS, DEFAULT_S, L_LANES

_HERE = os.path.dirname(os.path.abspath(__file__))

# phys_opt + post-route phys_opt
_VIVADO = [
    "[vivado]",
    "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=1",
    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=1",
    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore",
]


_ELEM_BYTES = {"f32": 4, "i8": 1}


def _bytes(shape, dtype="f32"):
    n = 1
    for s in shape:
        n *= s
    return (n + 1) // 2 if dtype == "i4" else n * _ELEM_BYTES[dtype]  # i4 = 2/byte


def write_system_cfg(meta, out_dir, lanes):
    lines = [
        "# Copyright Allo authors. All Rights Reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# v++ LINK configuration for the 3-CU LLaMA decoder-layer.",
        "",
        "[connectivity]",
    ]
    for key in ("cu0", "cu1", "cu2"):
        lines.append(f"nk={meta[key]['kernel']}:1:{key}")
    lines.append("")
    lines.append("# SLR floorplan")
    for key in ("cu0", "cu1", "cu2"):
        lines.append(f"slr={key}:{meta[key]['slr']}")
    lines.append("")
    lines.append("# On-chip activation handoffs.")
    for prod, cons, names in LINKS:
        for name in names:
            lines.append(f"#   {prod} -> {cons} : {name} (x{lanes} lanes)")
            for lane in range(lanes):
                p = name + "_r" * lane
                lines.append(f"stream_connect={prod}.{p}:{cons}.{p}")
    lines.append("")
    lines.append("# Weights / IO -> HBM.")
    for key in ("cu0", "cu1", "cu2"):
        for i, (name, shape, direction, hbm, dtype) in enumerate(meta[key]["maxi"]):
            lines.append(
                f"sp={key}.v{i}:HBM[{hbm}]"
                f"  # v{i}={name} {direction:3s} {dtype} {tuple(shape)}"
                f" = {_bytes(shape, dtype)} B"
            )
    lines.append("")
    lines += _VIVADO
    with open(os.path.join(out_dir, "system.cfg"), "w") as f:
        f.write("\n".join(lines) + "\n")


def host_body(meta):
    """Per-CU buffer setup that fills host.cpp.in's @HOST_BODY@ marker: allocate
    each m_axi buffer, upload read-only inputs from <name>.data, set the leading
    (m_axi) args, launch all CUs, wait, then download the `out` buffer."""
    setup, starts, waits, syncs = [], [], [], []
    for key in ("cu0", "cu1", "cu2"):
        m = meta[key]
        setup.append(f'  auto k_{key} = xrt::kernel(device, uuid, "{m["kernel"]}");')
        setup.append(f"  auto r_{key} = xrt::run(k_{key});")
        for i, (name, shape, direction, _hbm, dtype) in enumerate(m["maxi"]):
            nb = _bytes(shape, dtype)
            setup.append(
                f"  auto bo_{name} = xrt::bo(device, {nb}u, " f"k_{key}.group_id({i}));"
            )
            if direction == "in":
                setup.append(
                    f'  {{ auto v = read_data("{name}.data", {nb}u); '
                    f"bo_{name}.write(v.data()); "
                    f"bo_{name}.sync(XCL_BO_SYNC_BO_TO_DEVICE); }}"
                )
            setup.append(f"  r_{key}.set_arg({i}, bo_{name});")
            if direction == "out":
                syncs.append(f"  bo_{name}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);")
                syncs.append(
                    f"  {{ std::vector<char> v({nb}u); bo_{name}.read("
                    f'v.data()); write_data("{name}.data", v.data(), {nb}u); }}'
                )
        starts.append(f"  r_{key}.start();")
        waits.append(f"  r_{key}.wait();")
    return (
        "\n".join(setup)
        + "\n\n  // launch all CUs; they pipeline CU0->CU1->CU2 over AXIS\n"
        + "\n".join(starts)
        + "\n"
        + "\n".join(waits)
        + "\n\n"
        + "\n".join(syncs)
    )


def write_host(meta, out_dir):
    with open(os.path.join(_HERE, "host.cpp.in")) as f:
        template = f.read()
    assert "  // @HOST_BODY@" in template, "host.cpp.in missing @HOST_BODY@ marker"
    host = template.replace("  // @HOST_BODY@", host_body(meta))
    with open(os.path.join(out_dir, "host.cpp"), "w") as f:
        f.write(host)


def write_sample_inputs(meta, out_dir):
    """Sample data for each distinct read-only port."""
    import numpy as np

    rng = np.random.default_rng(0)
    seen = set()
    for key in ("cu0", "cu1", "cu2"):
        for name, shape, direction, _hbm, dtype in meta[key]["maxi"]:
            if direction != "in" or name in seen:
                continue
            seen.add(name)
            n = 1
            for s in shape:
                n *= s
            path = os.path.join(out_dir, f"{name}.data")
            if dtype == "f32":
                rng.standard_normal(n, dtype=np.float32).tofile(path)
            elif dtype == "i8":
                rng.integers(-8, 8, n, dtype=np.int8).tofile(path)
            else:  # i4: 2 nibbles per byte
                rng.integers(0, 256, (n + 1) // 2, dtype=np.uint8).tofile(path)


def generate_harness(
    config,
    out_dir,
    *,
    S=DEFAULT_S,
    lanes=L_LANES,
    write_inputs=True,
    variant="f32",
    group_size=None,
):
    """Write the build harness (``system.cfg``, ``host.cpp``, ``Makefile`` and, if
    ``write_inputs``, sample ``*.data`` emulation inputs) into ``out_dir``."""
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    meta = cu_meta(config, S, variant, group_size)
    write_system_cfg(meta, out_dir, lanes)
    write_host(meta, out_dir)
    shutil.copyfile(os.path.join(_HERE, "Makefile"), os.path.join(out_dir, "Makefile"))
    if write_inputs:
        write_sample_inputs(meta, out_dir)
    return meta
