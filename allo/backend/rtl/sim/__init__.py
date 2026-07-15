# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RTL-backend simulation, built on cocotb.

``shell`` builds the DUT (emitted Verilog + extern-IP behavioral models) and runs
it through ``cocotb_tools.runner`` on the chosen simulator; ``cocotb_tb`` is the
generic, config-driven testbench that services the memory ports from numpy.
"""
