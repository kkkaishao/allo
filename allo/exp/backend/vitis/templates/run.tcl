# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Auto-generated run.tcl for Vitis HLS

open_project -reset hls_prj
set_top {top}
add_files kernel.cpp

open_solution -reset solution1 -flow_target {flow_target}
set_part {{{part}}}
create_clock -period {clock_period:.4f} -name default

csynth_design
exit
