# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Auto-generated Makefile for legacy (pre-2025.2) Vitis C simulation.
#
# Vitis HLS used gcc as the C-simulation compiler through 2024.2

TOP ?=
KERNEL_CPP ?= kernel.cpp
KERNEL_H ?= kernel.h
OUT ?= {csim_shared_library}

VITIS_ROOT ?= {vitis_root}
HLS_ROOT ?= $(VITIS_ROOT)
CXX ?= $(lastword $(sort $(wildcard $(HLS_ROOT)/tps/lnx64/gcc-*/bin/g++)))
MATHHLS_LIB ?= $(HLS_ROOT)/lnx64/lib/csim
FPO_LIB ?= $(firstword $(wildcard $(HLS_ROOT)/lnx64/tools/fpo_*))
CRT_DIR ?=

OPT_FLAGS ?= -O3 -march=native

HLS_INCLUDES ?= \
  -I$(HLS_ROOT)/include \
  -I$(HLS_ROOT)/include/ap_sysc \
  -I$(HLS_ROOT)/common/technology/generic/SystemC \
  -I$(HLS_ROOT)/common/technology/generic/SystemC/AESL_FP_comp \
  -I$(HLS_ROOT)/common/technology/generic/SystemC/AESL_comp \
  -I$(HLS_ROOT)/lnx64/tools/auto_cc/include \
  -I/usr/include/x86_64-linux-gnu
HLS_DEFINES ?= -D__HLS_COSIM__ -D__HLS_CSIM__ -D__VITIS_HLS__ -D__SIM_FPO__ -D__DSP48E2__
CXXFLAGS ?= -std=gnu++17 -shared -fPIC -fpermissive \
  -Wno-unknown-pragmas -Wno-abi

LDFLAGS ?= \
  -Wl,-rpath,$(MATHHLS_LIB) -L$(MATHHLS_LIB) -lhlsmc++-GCC46 -lhlsm-GCC46 \
  -Wl,-rpath,$(FPO_LIB) -L$(FPO_LIB) -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel
EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS ?=

CRT_FLAG := $(if $(CRT_DIR),-B$(CRT_DIR),)

.PHONY: all clean

all: $(OUT)

$(OUT): $(KERNEL_CPP) $(KERNEL_H)
	$(CXX) $(OPT_FLAGS) $(CXXFLAGS) $(CRT_FLAG) $(HLS_INCLUDES) $(HLS_DEFINES) \
	  $(EXTRA_CXXFLAGS) $(KERNEL_CPP) -o $(OUT) $(LDFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f $(OUT)
