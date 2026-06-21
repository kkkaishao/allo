#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a command inside the Vitis container, with the Xilinx
# toolchain + XRT + PLATFORM already sourced. The host Xilinx install, your $HOME,
# and the current directory are bind-mounted, and the container's working dir
# mirrors the host $PWD -- so relative paths and `make` "just work".
#
# Override via env vars: IMAGE, XILINX_VITIS, XILINX_XRT, PLATFORM.

set -euo pipefail

IMAGE=${IMAGE:-vitis_runtime:2023.2}
XILINX_VITIS=${VITIS:-/tools/Xilinx/Vitis/2023.2}
XILINX_XRT=${XRT:-/opt/xilinx/xrt}
PLATFORM=${PLATFORM:-/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm}

HOST_PWD=$PWD

# Attach a TTY only when we actually have one (keeps the script usable in pipes,
# CI, and background runs).
TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

# Command to run inside the container; default to an interactive shell.
if [ "$#" -eq 0 ]; then
  INNER='exec bash'
else
  INNER="exec $(printf '%q ' "$@")"
fi

echo "run-vitis: image=$IMAGE  vitis=$XILINX_VITIS  platform=$(basename "$PLATFORM")  cwd=$HOST_PWD" >&2

# $USER, $HOME, and $LOGNAME must be set for hw_emu to work properly
exec docker run --rm "${TTY[@]}" \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v /tools/Xilinx/:/tools/Xilinx:ro \
  -v "$HOME:$HOME" \
  -v "$HOST_PWD:$HOST_PWD" \
  -v /opt/xilinx/platforms:/opt/xilinx/platforms \
  -e USER="$USER" -e HOME="$HOME" -e LOGNAME="$LOGNAME" \
  "$IMAGE" bash -lc "
    source '$XILINX_VITIS/settings64.sh'
    source '$XILINX_XRT/setup.sh'
    export PLATFORM='$PLATFORM'
    cd '$HOST_PWD'
    $INNER
  "
