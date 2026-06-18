# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set cosim [expr {{[info exists env(COSIM)] ? $env(COSIM) : "0"}}]

open_project hls_prj
set_top {top}
open_solution hls -flow_target {flow_target}

add_files kernel.cpp
add_files -tb host.cpp -cflags "-O2 -pthread"

set_part {part}
create_clock -period {period:.3f} -name default

csynth_design
if {{ $cosim eq "1" }} {{
  cosim_design
}}
close_project
exit
