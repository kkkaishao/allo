# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Auto-generated Makefile for Vitis emulation / hardware builds with an
# XRT-native host. PLATFORM is intentionally left blank: `make` inherits it from
# the environment (export PLATFORM=/path/to/<shell>.xpfm), so no .xpfm is baked
# into the generated project.

TOP := {top}
TARGET ?= hw_emu
PLATFORM ?=
FREQ_MHZ ?= {freq_mhz}

XILINX_XRT ?= /opt/xilinx/xrt
VITIS_ROOT ?= {vitis_root}

VPP := v++
EMCONFIGUTIL := emconfigutil
CXX := g++

XSA := $(strip $(patsubst %.xpfm,%,$(notdir $(PLATFORM))))
BUILD_DIR := build_dir.$(TARGET).$(XSA)
TEMP_DIR := _x.$(TARGET).$(XSA)

KERNEL_XO := $(BUILD_DIR)/$(TOP).xo
XCLBIN := $(BUILD_DIR)/$(TOP).xclbin
HOST_EXE := host.exe
EMCONFIG := emconfig.json
HLS_PRJ := hls_prj

VPP_FLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps --temp_dir $(TEMP_DIR)
ifneq ($(TARGET),hw)
VPP_FLAGS += -g
endif
VPP_LDFLAGS += --kernel_frequency $(FREQ_MHZ) --optimize 2

CXXFLAGS += -std=c++17 -Wall -O2 -I$(XILINX_XRT)/include
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lxrt_coreutil -luuid

.PHONY: all csynth xo xclbin host emconfig run precheck check-platform check-xrt clean

all: xclbin host emconfig

# Standalone C synthesis (QoR report) via the v++ HLS flow. Part-based (read from
# hls.cfg), so it needs no PLATFORM. Report: $(HLS_PRJ)/hls/syn/report/csynth.xml.
csynth: kernel.cpp kernel.h hls.cfg
	$(VPP) -c --mode hls --config hls.cfg --work_dir $(HLS_PRJ)

check-platform:
ifeq ($(PLATFORM),)
	$(error PLATFORM is not set. export PLATFORM=/path/to/<shell>.xpfm and retry)
endif

check-xrt:
ifeq ($(wildcard $(XILINX_XRT)/include/xrt),)
	$(error XILINX_XRT not found at $(XILINX_XRT). source /opt/xilinx/xrt/setup.sh and retry)
endif

# Kernel C/C++ -> .xo (HLS). The fast, frontend-validating compile step.
xo: $(KERNEL_XO)
$(KERNEL_XO): kernel.cpp kernel.h | check-platform
	@mkdir -p $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) -k $(TOP) -o $@ kernel.cpp

# .xo -> .xclbin (link; emulation SystemC models / hw synth+impl run here).
xclbin: $(XCLBIN)
$(XCLBIN): $(KERNEL_XO) | check-platform
	$(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) -o $@ $(KERNEL_XO)

# XRT-native host. Independent of the kernel build (loads the xclbin at runtime).
host: $(HOST_EXE)
$(HOST_EXE): host.cpp kernel.h | check-xrt
	$(CXX) $(CXXFLAGS) -o $@ host.cpp $(LDFLAGS)

emconfig: $(EMCONFIG)
$(EMCONFIG): | check-platform
	$(EMCONFIGUTIL) --platform $(PLATFORM) --od .

# Fast pre-check: emit .xo + host (+ emconfig for emulation), skipping the
# multi-hour / platform-locked link step. Validates the generated project.
precheck: xo host
ifneq ($(TARGET),hw)
precheck: emconfig
endif

run: all
ifeq ($(TARGET),hw)
	./$(HOST_EXE) $(XCLBIN)
else
	XCL_EMULATION_MODE=$(TARGET) ./$(HOST_EXE) $(XCLBIN)
endif

clean:
	rm -rf build_dir.* _x.* $(HOST_EXE) $(EMCONFIG) *.log *.jou .Xil .run
