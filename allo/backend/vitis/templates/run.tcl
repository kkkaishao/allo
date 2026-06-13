# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set run_cosim $::env(RUN_COSIM)

open_project hls_prj -reset
set_top {top}
open_solution -reset hls -flow_target {flow_target}

add_files kernel.cpp
add_files -tb host.cpp

set_part {part}
create_clock -period {period:.3f} -name default

csynth_solution
if { $run_cosim eq "true" } {
  cosim_solution
}
close_project
