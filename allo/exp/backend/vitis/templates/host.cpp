// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Auto-generated XRT-native host for Vitis emulation / hardware execution.
// Buffer arguments are marshalled through input<i>.data / output<i>.data; every
// buffer is synced back after the run (the frontend has no load/store analysis,
// matching the C-simulation writeback).

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {{

std::vector<char> read_data(const std::string &path, size_t bytes) {{
  std::vector<char> buf(bytes);
  std::ifstream f(path, std::ios::binary);
  if (!f) {{
    std::cerr << "Failed to open input file: " << path << "\n";
    std::exit(EXIT_FAILURE);
  }}
  f.read(buf.data(), static_cast<std::streamsize>(bytes));
  return buf;
}}

void write_data(const std::string &path, const char *data, size_t bytes) {{
  std::ofstream f(path, std::ios::binary);
  if (!f) {{
    std::cerr << "Failed to open output file: " << path << "\n";
    std::exit(EXIT_FAILURE);
  }}
  f.write(data, static_cast<std::streamsize>(bytes));
}}

}} // namespace

int main(int argc, char **argv) {{
  if (argc != 2) {{
    std::cerr << "Usage: " << argv[0] << " <xclbin>\n";
    return EXIT_FAILURE;
  }}

  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(argv[1]);
  auto kernel = xrt::kernel(device, uuid, "{top}");

{body}
  std::cout << "Finished execution!\n";
  return EXIT_SUCCESS;
}}
