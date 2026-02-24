#include "ir.h"

using namespace mlir;
using namespace mlir::allo;

void init_allo_ir(nb::module_ &m) {

  nb::enum_<allo::PartitionKindEnum>(m, "PartitionKind")
      .value("Complete", allo::PartitionKindEnum::Complete)
      .value("Block", allo::PartitionKindEnum::Block)
      .value("Cyclic", allo::PartitionKindEnum::Cyclic);

  nb::class_<allo::PartitionAttr, Attribute>(m, "PartitionAttr")
      .def_static(
          "get",
          [](MLIRContext &context, const std::vector<int64_t> &dims,
             const std::vector<uint32_t> &kinds,
             const std::vector<int64_t> &factors) {
            SmallVector<PartitionKindEnum, 4> kindAttrs;
            for (auto k : kinds) {
              kindAttrs.push_back(static_cast<PartitionKindEnum>(k));
            }
            return allo::PartitionAttr::get(&context, kindAttrs, factors, dims);
          },
          nb::arg("context"), nb::arg("dims"), nb::arg("kinds"),
          nb::arg("factors"));
  PyAttributeRegistry::registerAttr<allo::PartitionAttr>();

  nb::class_<allo::ChannelType, Type>(m, "ChannelType")
      .def_static(
          "get",
          [](MLIRContext &context, Type type, unsigned depth = 2) {
            return allo::ChannelType::get(&context, type, depth);
          },
          nb::arg("context"), nb::arg("data_type"), nb::arg("capacity") = 2);
  PyTypeRegistry::registerType(allo::ChannelType::getTypeID(), [](Type t) {
    return nb::cast(mlir::cast<allo::ChannelType>(t));
  });

  nb::class_<allo::ChanCreateOp, OpState>(m, "ChanCreateOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::string &name,
             ChannelType &chanType, const std::vector<int64_t> &shape = {}) {
            return ChanCreateOp::create(builder, builder.get_loc(), name,
                                        chanType, shape);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("chan_type"),
          nb::arg("shape") = std::vector<int64_t>{});

  nb::class_<allo::ChanAcquireOp, OpState>(m, "ChanAcquireOp")
      .def_static("create", [](AlloOpBuilder &builder, const std::string &chan,
                               const std::vector<Value> &indices,
                               Type &dataType, int size = 1) {
        return allo::ChanAcquireOp::create(builder, builder.get_loc(), chan,
                                           indices, dataType, size);
      });

  nb::class_<allo::ChanReleaseOp, OpState>(m, "ChanReleaseOp")
      .def_static("create", [](AlloOpBuilder &builder, const std::string &chan,
                               const std::vector<Value> &indices,
                               const std::vector<Value> &buffers) {
        return allo::ChanReleaseOp::create(builder, builder.get_loc(), chan,
                                           indices, buffers);
      });

  nb::class_<allo::ChanGetOp, OpState>(m, "ChanGetOp")
      .def_static("create",
                  [](AlloOpBuilder &builder, Type &dataTy,
                     const std::string &chan, const std::vector<Value> &indices,
                     bool blocking = false) -> Value {
                    return allo::ChanGetOp::create(builder, builder.get_loc(),
                                                   dataTy, chan, indices,
                                                   blocking);
                  });

  nb::class_<allo::ChanPutOp, OpState>(m, "ChanPutOp")
      .def_static("create", [](AlloOpBuilder &builder, const std::string &chan,
                               const std::vector<Value> &indices, Value &val,
                               bool blocking = false) {
        return allo::ChanPutOp::create(builder, builder.get_loc(), chan,
                                       indices, val, blocking);
      });

  nb::class_<allo::KernelOp, OpState>(m, "KernelOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::string &name,
             FunctionType &type,
             const std::vector<DictionaryAttr> &argAttrs = {},
             const std::vector<int64_t> &vtMap = {}) {
            return allo::KernelOp::create(builder, builder.get_loc(), name,
                                          type, {}, argAttrs, vtMap);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("type"),
          nb::arg("argAttrs") = std::vector<DictionaryAttr>{},
          nb::arg("vtMap") = std::vector<int64_t>{})
      .def(
          "get_arg_at",
          [](allo::KernelOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def_prop_ro("num_args", &allo::KernelOp::getNumArguments)
      .def(
          "add_entry_block",
          [](allo::KernelOp &self) -> Block * { return self.addEntryBlock(); },
          nb::rv_policy::reference)
      .def(
          "set_arg_attr",
          [](allo::KernelOp &self, unsigned arg_no, const std::string &name,
             Attribute &attr) {
            if (arg_no >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            // set arg attributes "name" to Value &"val"
            self.setArgAttr(arg_no, name, attr);
          },
          nb::arg("arg_no"), nb::arg("name"), nb::arg("attr"))
      .def_prop_ro("func_type", &allo::KernelOp::getFunctionType)
      .def("set_type", &allo::KernelOp::setType, nb::arg("type"))
      .def_prop_ro("func_name",
                   [](allo::KernelOp &self) { return self.getName().str(); })
      .def_prop_rw(
          "virtual_mapping",
          [](allo::KernelOp &self) {
            return std::vector<int64_t>(self.getVirtualMappingVec());
          },
          [](allo::KernelOp &self, const std::vector<int64_t> &vmap) {
            self.setVirtualMappingAttr(
                VirtMapAttr::get(self.getContext(), vmap));
          });

  nb::class_<allo::ReturnOp, OpState>(m, "ReturnOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &operands) {
            return allo::ReturnOp::create(builder, builder.get_loc(), operands);
          },
          nb::arg("builder"), nb::arg("operands"));

  nb::class_<allo::CallOp, OpState>(m, "CallOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, allo::KernelOp &kernel,
             const std::vector<Value> &args) {
            return allo::CallOp::create(builder, builder.get_loc(), kernel,
                                        args);
          },
          nb::arg("builder"), nb::arg("kernel"), nb::arg("args"));

  nb::class_<allo::GetProgramIdOp, OpState>(m, "GetProgramIdOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, unsigned dim) -> Value {
            return allo::GetProgramIdOp::create(builder, builder.get_loc(),
                                                dim);
          },
          nb::arg("builder"), nb::arg("dim"));

  nb::class_<allo::GetNumProgramsOp, OpState>(m, "GetNumProgramsOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, unsigned dim) -> Value {
            return allo::GetNumProgramsOp::create(builder, builder.get_loc(),
                                                  dim);
          },
          nb::arg("builder"), nb::arg("dim"));

  nb::class_<allo::BitExtractOp, OpState>(m, "BitExtractOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &start,
             int64_t width) -> Value {
            return allo::BitExtractOp::create(builder, builder.get_loc(), input,
                                              start, width);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("start"),
          nb::arg("width"));

  nb::class_<allo::BitInsertOp, OpState>(m, "BitInsertOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &dest, Value &src, Value &start,
             int64_t width) -> Value {
            return allo::BitInsertOp::create(builder, builder.get_loc(), dest,
                                             src, start, width);
          },
          nb::arg("builder"), nb::arg("dest"), nb::arg("src"), nb::arg("start"),
          nb::arg("width"));

  nb::class_<allo::BitConcatOp, OpState>(m, "BitConcatOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder,
             const std::vector<Value> &inputs) -> Value {
            return allo::BitConcatOp::create(builder, builder.get_loc(),
                                             inputs);
          },
          nb::arg("builder"), nb::arg("inputs"));
}