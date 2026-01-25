#include "ir.h"

using namespace mlir;
using namespace mlir::allo;

void init_allo_ir(nb::module_ &m) {
  nb::class_<allo::ChannelType, Type>(m, "ChannelType")
      .def_static(
          "get",
          [](MLIRContext &context, Type type, unsigned capacity = 2) {
            return allo::ChannelType::get(&context, type, capacity);
          },
          nb::arg("context"), nb::arg("data_type"), nb::arg("capacity") = 2);
  PyTypeRegistry::registerType(allo::ChannelType::getTypeID(), [](Type t) {
    return nb::cast(mlir::cast<allo::ChannelType>(t));
  });

  nb::class_<allo::StreamType, Type>(m, "StreamType")
      .def_static(
          "get",
          [](MLIRContext &context, Type type, unsigned depth = 2) {
            return allo::StreamType::get(&context, type, depth);
          },
          nb::arg("context"), nb::arg("data_type"), nb::arg("depth") = 2);
  PyTypeRegistry::registerType(allo::StreamType::getTypeID(), [](Type t) {
    return nb::cast(mlir::cast<allo::StreamType>(t));
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

  nb::class_<allo::ChanToStreamOp, OpState>(m, "ChanToStreamOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::string &chan,
             const std::vector<Value> &indices, StreamType &type,
             std::optional<AffineMap> map) -> Value {
            return allo::ChanToStreamOp::create(builder, builder.get_loc(),
                                                chan, indices, type,
                                                map.value_or(AffineMap{}));
          },
          nb::arg("builder"), nb::arg("chan"), nb::arg("indices"),
          nb::arg("type"), nb::arg("map").none() = std::nullopt)
      .def_prop_ro("chan",
                   [](ChanToStreamOp &self) { return self.getChan().str(); })
      .def_prop_ro("map", [](ChanToStreamOp &self) { return self.getMap(); });

  nb::class_<allo::ChanAcquireBufferOp, OpState>(m, "ChanAcquireBufferOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::string &chan,
             const std::vector<Value> &indices, Type &dataType, int size = 1) {
            return allo::ChanAcquireBufferOp::create(
                builder, builder.get_loc(), chan, indices, dataType, size);
          })
      .def_prop_ro("chan", [](ChanAcquireBufferOp &self) {
        return self.getChan().str();
      });

  nb::class_<allo::ChanReleaseBufferOp, OpState>(m, "ChanReleaseBufferOp")
      .def_static("create",
                  [](AlloOpBuilder &builder, const std::string &chan,
                     const std::vector<Value> &indices,
                     const std::vector<Value> &buffers) {
                    return allo::ChanReleaseBufferOp::create(
                        builder, builder.get_loc(), chan, indices, buffers);
                  })
      .def_prop_ro("chan", [](ChanReleaseBufferOp &self) {
        return self.getChan().str();
      });

  nb::class_<allo::StreamGetOp, OpState>(m, "StreamGetOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &stream) -> Value {
            if (!isa<StreamType>(stream.getType())) {
              throw nb::value_error(
                  "StreamGetOp must be created with a StreamType value!");
            }
            return allo::StreamGetOp::create(builder, builder.get_loc(),
                                             stream);
          },
          nb::arg("builder"), nb::arg("stream"));

  nb::class_<allo::StreamPutOp, OpState>(m, "StreamPutOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val, Value &stream) {
            return allo::StreamPutOp::create(builder, builder.get_loc(), stream,
                                             val);
          },
          nb::arg("builder"), nb::arg("val"), nb::arg("stream"));

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
            auto attrs = self.getVirtualMapping().getAsRange<IntegerAttr>();
            std::vector<int64_t> vmap;
            for (auto attr : attrs) {
              vmap.push_back(attr.getInt());
            }
            return vmap;
          },
          [](allo::KernelOp &self, const std::vector<int64_t> &vmap) {
            SmallVector<Attribute, 4> newMap;
            auto i64 = IntegerType::get(self->getContext(), 64);
            for (auto v : vmap) {
              newMap.push_back(IntegerAttr::get(i64, v));
            }
            if (!newMap.empty()) {
              newMap.push_back(IntegerAttr::get(i64, 1));
            }
            self.setVirtualMappingAttr(
                ArrayAttr::get(self->getContext(), newMap));
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