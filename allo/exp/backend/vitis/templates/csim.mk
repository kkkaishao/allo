# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Auto-generated Makefile for Python-native Vitis C simulation.
#
# This is the native flow for Vitis 2025.2+, where the csim compiler is an AMD
# clang fork driven by `-fhls-csim`

TOP ?=
KERNEL_CPP ?= kernel.cpp
KERNEL_H ?= kernel.h
OUT ?= {csim_shared_library}

VITIS_ROOT ?= {vitis_root}
CXX ?= $(firstword $(wildcard $(VITIS_ROOT)/lnx64/tools/clang-1*/bin/clang++))
GCC_TOOLCHAIN ?= $(lastword $(sort $(wildcard $(VITIS_ROOT)/tps/lnx64/gcc-*)))
VITIS_HOST_LIB ?= $(VITIS_ROOT)/lib/lnx64.o
MATHHLS_LIB ?= $(VITIS_ROOT)/lnx64/lib/csim
FPO_LIB ?= $(firstword $(wildcard $(VITIS_ROOT)/lnx64/tools/fpo_*))

HLS_INCLUDES ?= \
  -I$(VITIS_ROOT)/include \
  -I$(VITIS_ROOT)/include/ap_sysc \
  -I$(VITIS_ROOT)/common/technology/generic/SystemC \
  -I$(VITIS_ROOT)/common/technology/generic/SystemC/AESL_FP_comp \
  -I$(VITIS_ROOT)/common/technology/generic/SystemC/AESL_comp \
  -I$(VITIS_ROOT)/lnx64/tools/auto_cc/include \
  -I/usr/include/x86_64-linux-gnu

HLS_DEFINES ?= -D__HLS_COSIM__ -D__HLS_CSIM__ -D__VITIS_HLS__ -D__SIM_FPO__ -D__DSP48E2__

OPT_FLAGS ?= -O3 -march=native
HLS_CXXFLAGS ?= -std=gnu++17 -shared -fPIC -fpermissive \
  -Wno-unknown-pragmas -Wno-abi -Wno-c++11-narrowing \
  -fhls-csim -fhlstoplevel=$(TOP) \
  --gcc-toolchain=$(GCC_TOOLCHAIN)
HLS_LDFLAGS ?= \
  -Wl,-rpath,$(MATHHLS_LIB) -L$(MATHHLS_LIB) -lhlsmc++-GCC46 -lhlsm-GCC46 \
  -Wl,-rpath,$(FPO_LIB) -L$(FPO_LIB) -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel

EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS ?=

.PHONY: all clean

all: $(OUT)

$(OUT): $(KERNEL_CPP) $(KERNEL_H)
	@LD_LIBRARY_PATH=$(VITIS_HOST_LIB):$$LD_LIBRARY_PATH \
	$(CXX) $(OPT_FLAGS) $(HLS_CXXFLAGS) $(HLS_INCLUDES) $(HLS_DEFINES) \
	  $(EXTRA_CXXFLAGS) $(KERNEL_CPP) -o $(OUT) $(HLS_LDFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f $(OUT)
