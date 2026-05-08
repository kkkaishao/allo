#include "ir.h"

#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace mlir;

namespace {

MlirStringRef toStringRef(const std::string &str) {
  return mlirStringRefCreate(str.data(), str.size());
}

class PyExecutionEngine {
public:
  PyExecutionEngine(ModuleOp module, int optLevel,
                    const std::vector<std::string> &sharedLibs,
                    bool enableObjectDump, bool enablePIC) {
    std::vector<MlirStringRef> libPaths;
    libPaths.reserve(sharedLibs.size());
    for (const std::string &path : sharedLibs)
      libPaths.push_back(toStringRef(path));

    engine = mlirExecutionEngineCreate(
        wrap(module), optLevel, static_cast<int>(libPaths.size()),
        libPaths.data(), enableObjectDump, enablePIC);
    if (mlirExecutionEngineIsNull(engine))
      throw std::runtime_error("Failed to create MLIR ExecutionEngine");
  }

  PyExecutionEngine(PyExecutionEngine &&other) noexcept : engine(other.engine) {
    other.engine = MlirExecutionEngine{nullptr};
  }

  PyExecutionEngine(const PyExecutionEngine &) = delete;
  PyExecutionEngine &operator=(const PyExecutionEngine &) = delete;

  ~PyExecutionEngine() {
    if (!mlirExecutionEngineIsNull(engine))
      mlirExecutionEngineDestroy(engine);
  }

  uintptr_t rawLookup(const std::string &name) {
    auto *func = mlirExecutionEngineLookupPacked(engine, toStringRef(name));
    return reinterpret_cast<uintptr_t>(func);
  }

  void rawRegisterRuntime(const std::string &name, uintptr_t addr) {
    mlirExecutionEngineRegisterSymbol(engine, toStringRef(name),
                                      reinterpret_cast<void *>(addr));
  }

  void initialize() { mlirExecutionEngineInitialize(engine); }

  void dumpToObjectFile(const std::string &fileName) {
    mlirExecutionEngineDumpToObjectFile(engine, toStringRef(fileName));
  }

private:
  MlirExecutionEngine engine{nullptr};
};

} // namespace

void bindExecutionEngine(nb::module_ &m) {
  nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
      .def(
          "__init__",
          [](PyExecutionEngine &self, ModuleOp module, int optLevel,
             const std::vector<std::string> &sharedLibs, bool enableObjectDump,
             bool enablePIC) {
            new (&self) PyExecutionEngine(module, optLevel, sharedLibs,
                                          enableObjectDump, enablePIC);
          },
          nb::arg("module"), nb::arg("opt_level") = 2,
          nb::arg("shared_libs") = std::vector<std::string>{},
          nb::arg("enable_object_dump") = false, nb::arg("enable_pic") = false)
      .def("raw_lookup", &PyExecutionEngine::rawLookup, nb::arg("name"))
      .def("raw_register_runtime", &PyExecutionEngine::rawRegisterRuntime,
           nb::arg("name"), nb::arg("addr"))
      .def("initialize", &PyExecutionEngine::initialize)
      .def("dump_to_object_file", &PyExecutionEngine::dumpToObjectFile,
           nb::arg("file_name"));
}
