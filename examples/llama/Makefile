# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

TARGET ?= hw_emu
PLATFORM ?=
FREQ_MHZ ?= 275
XILINX_XRT ?=
VPP := v++
EMCONFIGUTIL := emconfigutil
CXX := g++
KERNELS := cu0 cu1 cu2
XSA := $(strip $(patsubst %.xpfm,%,$(notdir $(PLATFORM))))
BUILD_DIR := build_dir.$(TARGET).$(XSA)
TEMP_DIR := _x.$(TARGET).$(XSA)
XCLBIN := $(BUILD_DIR)/top.xclbin
HOST_EXE := host.exe
EMCONFIG := emconfig.json
VIVADO_IMPL_JOBS ?= 16
VPP_FLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps --temp_dir $(TEMP_DIR)

ifneq ($(TARGET),hw)
VPP_FLAGS += -g
endif

VPP_LDFLAGS += --kernel_frequency $(FREQ_MHZ) --optimize 2 --vivado.impl.jobs $(VIVADO_IMPL_JOBS) --config system.cfg
CXXFLAGS += -std=c++17 -Wall -O2 -I$(XILINX_XRT)/include
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lxrt_coreutil -luuid

XOS := $(BUILD_DIR)/cu0.xo $(BUILD_DIR)/cu1.xo $(BUILD_DIR)/cu2.xo

.PHONY: all xo xclbin host emconfig run clean check-platform

all: xclbin host emconfig

check-platform:
ifeq ($(PLATFORM),)
	$(error PLATFORM is not set. export PLATFORM=/path/to/<shell>.xpfm and retry)
endif

check-xrt:
ifeq ($(XILINX_XRT),)
	$(error XILINX_XRT is not set. source XRT setup script and retry)
endif

$(BUILD_DIR)/cu0.xo: cu0.cpp kernel.h | check-platform
	@mkdir -p $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) -k cu0 -o $@ cu0.cpp | tee hls_cu0.log
$(BUILD_DIR)/cu1.xo: cu1.cpp kernel.h | check-platform
	@mkdir -p $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) -k cu1 -o $@ cu1.cpp | tee hls_cu1.log
$(BUILD_DIR)/cu2.xo: cu2.cpp kernel.h | check-platform
	@mkdir -p $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) -k cu2 -o $@ cu2.cpp | tee hls_cu2.log

xo: $(XOS)

xclbin: $(XCLBIN)
$(XCLBIN): $(XOS) system.cfg | check-platform
	$(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) -o $@ $(XOS) | tee sys_link.log

host: $(HOST_EXE) | check-xrt
$(HOST_EXE): host.cpp kernel.h
	$(CXX) $(CXXFLAGS) -o $@ host.cpp $(LDFLAGS)

emconfig: $(EMCONFIG)
$(EMCONFIG): | check-platform
	$(EMCONFIGUTIL) --platform $(PLATFORM) --od .

run: all
ifeq ($(TARGET),hw)
	./$(HOST_EXE) $(XCLBIN)
else
	XCL_EMULATION_MODE=$(TARGET) ./$(HOST_EXE) $(XCLBIN) | tee emu_run.log
endif

clean:
	rm -rf build_dir.* _x.* $(HOST_EXE) $(EMCONFIG) *.log *.jou .Xil .run
