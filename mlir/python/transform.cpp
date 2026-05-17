#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"

#include <utility>

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

#include "llvm/Support/SourceMgr.h"

#include "allo/TransformOps/AlloTransformOps.h"

using namespace mlir;

void bindTransform(nb::module_ &m) {
  m.def(
      "apply_transforms",
      [](Operation &payload, Operation &transformRoot,
         ModuleOp &transformModule) -> std::pair<bool, std::string> {
        // capture the error message from the diagnostic handler
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        llvm::SourceMgr sourceMgr;
        mlir::SourceMgrDiagnosticHandler handler(
            sourceMgr, transformModule->getContext(), os);

        transform::TransformOptions options;
        options.enableEnforceSingleToplevelTransformOp();
        auto ret = transform::applyTransformNamedSequence(
            &payload, &transformRoot, transformModule, options);
        os.flush();
        return {failed(ret), errMsg};
      },
      nb::arg("payload"), nb::arg("transform_root"),
      nb::arg("transform_module"));

  nb::class_<transform::OperationType, Type>(m, "OperationType")
      .def_static(
          "get",
          [](MLIRContext &context, std::string_view opName) {
            return transform::OperationType::get(
                &context, StringAttr::get(&context, opName));
          },
          nb::arg("context"), nb::arg("op_name"));

  nb::class_<transform::ParamType, Type>(m, "ParamType")
      .def_static(
          "get",
          [](MLIRContext &context, Type &type) {
            return transform::ParamType::get(&context, type);
          },
          nb::arg("context"), nb::arg("type"));

  nb::class_<transform::AnyOpType, Type>(m, "AnyOpType")
      .def_static(
          "get",
          [](MLIRContext &context) {
            return transform::AnyOpType::get(&context);
          },
          nb::arg("context"));

  nb::class_<transform::AnyParamType, Type>(m, "AnyParamType")
      .def_static(
          "get",
          [](MLIRContext &context) {
            return transform::AnyParamType::get(&context);
          },
          nb::arg("context"));

  nb::class_<transform::AnnotateOp, OpState>(m, "AnnotateOp")
      .def(
          "__init__",
          [](transform::AnnotateOp &self, AlloOpBuilder &builder, Value &target,
             std::string_view name, Attribute value) {
            auto anyParam = transform::AnyParamType::get(builder.getContext());
            auto param = transform::ParamConstantOp::create(
                builder, builder.getLocation(), anyParam, value);
            self = transform::AnnotateOp::create(builder, builder.getLocation(),
                                                 target, name, param);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("name"),
          nb::arg("value"));

  nb::class_<transform::GetDefiningOp, OpState>(m, "GetDefiningOp")
      .def(
          "__init__",
          [](transform::GetDefiningOp &self, AlloOpBuilder &builder,
             Value &target) {
            auto anyOpType = transform::AnyOpType::get(builder.getContext());
            self = transform::GetDefiningOp::create(
                builder, builder.getLocation(), anyOpType, target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::NamedSequenceOp, OpState>(m, "NamedSequenceOp")
      .def(
          "__init__",
          [](transform::NamedSequenceOp &self, AlloOpBuilder &builder,
             std::string_view name, Type &rootType,
             const std::vector<Type> &resTypes) {
            auto bodyBuilder = [](OpBuilder &, Location, BlockArgument) {
              return;
            };
            self = transform::NamedSequenceOp::create(
                builder, builder.getLocation(), name, rootType, resTypes,
                bodyBuilder);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("root_type"),
          nb::arg("res_types"))
      .def(
          "get_entry_block",
          [](transform::NamedSequenceOp &self) {
            return &self->getRegion(0).front();
          },
          nb::rv_policy::reference)
      .def(
          "get_arg_at",
          [](transform::NamedSequenceOp &self, unsigned idx) -> BlockArgument {
            return self.getArgument(idx);
          },
          nb::arg("idx"));

  nb::class_<transform::YieldOp, OpState>(m, "YieldOp")
      .def(
          "__init__",
          [](transform::YieldOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &operands) {
            self = transform::YieldOp::create(builder, builder.getLocation(),
                                              operands);
          },
          nb::arg("builder"), nb::arg("operands"));

  // common transformations
  nb::class_<transform::ApplyCommonSubexpressionEliminationOp, OpState>(
      m, "ApplyCSEOp")
      .def(
          "__init__",
          [](transform::ApplyCommonSubexpressionEliminationOp &self,
             AlloOpBuilder &builder, Value &target) {
            self = transform::ApplyCommonSubexpressionEliminationOp::create(
                builder, builder.getLocation(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ApplyDeadCodeEliminationOp, OpState>(m, "ApplyDCEOp")
      .def(
          "__init__",
          [](transform::ApplyDeadCodeEliminationOp &self,
             AlloOpBuilder &builder, Value &target) {
            self = transform::ApplyDeadCodeEliminationOp::create(
                builder, builder.getLocation(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ApplyCanonicalizationPatternsOp, OpState>(
      m, "ApplyCanonicalizationOp")
      .def(
          "__init__",
          [](transform::ApplyCanonicalizationPatternsOp &self,
             AlloOpBuilder &builder) {
            self = transform::ApplyCanonicalizationPatternsOp::create(
                builder, builder.getLocation());
          },
          nb::arg("builder"));

  nb::class_<transform::ApplyLoopInvariantCodeMotionOp, OpState>(m,
                                                                 "ApplyLICMOp")
      .def(
          "__init__",
          [](transform::ApplyLoopInvariantCodeMotionOp &self,
             AlloOpBuilder &builder, Value &target) {
            self = transform::ApplyLoopInvariantCodeMotionOp::create(
                builder, builder.getLocation(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ApplyPatternsOp, OpState>(m, "ApplyPatternsOp")
      .def(
          "__init__",
          [](transform::ApplyPatternsOp &self, AlloOpBuilder &builder,
             Value &target) {
            self = transform::ApplyPatternsOp::create(
                builder, builder.getLocation(), target);
          },
          nb::arg("builder"), nb::arg("target"))
      .def(
          "get_body",
          [](transform::ApplyPatternsOp &self) { return self.getBody(); },
          nb::rv_policy::reference);

  nb::class_<transform::ApplyRegisteredPassOp, OpState>(m,
                                                        "ApplyRegisteredPassOp")
      .def(
          "__init__",
          [](transform::ApplyRegisteredPassOp &self, AlloOpBuilder &builder,
             Value &target, std::string_view passName,
             DictionaryAttr passOptions, const std::vector<Value> &dynArgs) {
            auto anyOpType = transform::AnyOpType::get(builder.getContext());
            self = transform::ApplyRegisteredPassOp::create(
                builder, builder.getLocation(), anyOpType, target, passName,
                passOptions, dynArgs);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("pass_name"),
          nb::arg("pass_options"), nb::arg("dynamic_args"));

  // operation matching
  nb::class_<transform::MatchOp, OpState>(m, "MatchOp")
      .def(
          "__init__",
          [](transform::MatchOp &self, AlloOpBuilder &builder, Value &target,
             Type &resType, const std::vector<std::string> &opNames = {},
             DictionaryAttr opAttrs = {}) {
            auto match = transform::MatchOp::create(
                builder, builder.getLocation(), resType, target);
            if (!opNames.empty()) {
              llvm::SmallVector<llvm::StringRef, 2> opNamesRef;
              for (const auto &name : opNames)
                opNamesRef.push_back(name);
              auto opNamesAttr = builder.getStrArrayAttr(opNamesRef);
              match->setAttr(match.getOpsAttrName(), opNamesAttr);
            }
            match->setAttr(match.getOpAttrsAttrName(), opAttrs);
            self = match;
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("res_type"),
          nb::arg("op_names"), nb::arg("op_attrs") = DictionaryAttr());

  nb::class_<transform::MergeHandlesOp, OpState>(m, "MergeHandlesOp")
      .def(
          "__init__",
          [](transform::MergeHandlesOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &handles, bool deduplicate) {
            self = transform::MergeHandlesOp::create(
                builder, builder.getLocation(), handles, deduplicate);
          },
          nb::arg("builder"), nb::arg("handles"),
          nb::arg("deduplicate") = true);

  nb::class_<transform::SplitHandleOp, OpState>(m, "SplitHandleOp")
      .def(
          "__init__",
          [](transform::SplitHandleOp &self, AlloOpBuilder &builder,
             Value &handle, unsigned numSplits) {
            self = transform::SplitHandleOp::create(
                builder, builder.getLocation(), handle, numSplits);
          },
          nb::arg("builder"), nb::arg("handle"), nb::arg("num_splits"));

  nb::class_<transform::LoopUnrollOp, OpState>(m, "LoopUnrollOp")
      .def(
          "__init__",
          [](transform::LoopUnrollOp &self, AlloOpBuilder &builder,
             Value &target, int factor) {
            self = transform::LoopUnrollOp::create(
                builder, builder.getLocation(), target,
                static_cast<uint64_t>(factor));
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factor"));

  nb::enum_<allo::PartitionKindEnum>(m, "PartitionKind")
      .value("Complete", allo::PartitionKindEnum::CompletePartition)
      .value("Block", allo::PartitionKindEnum::BlockPartition)
      .value("Cyclic", allo::PartitionKindEnum::CyclicPartition)
      .export_values();

  nb::class_<allo::PartitionAttr, Attribute>(m, "PartitionAttr")
      .def_static(
          "get",
          [](MLIRContext &context, nb::list &subPartitions) {
            SmallVector<allo::PartitionAxisAttr> partitionAxes;
            for (nb::handle item : subPartitions) {
              auto triple = nb::cast<nb::tuple>(item);
              if (triple.size() != 3) {
                throw nb::value_error(
                    "Each sub-partition must be a tuple/list of size 3: (dim, "
                    "kind, factor).");
              }
              int64_t dim = nb::cast<int64_t>(triple[0]);
              auto kind = nb::cast<allo::PartitionKindEnum>(triple[1]);
              int64_t factor = nb::cast<int64_t>(triple[2]);
              partitionAxes.push_back(
                  allo::PartitionAxisAttr::get(&context, kind, factor, dim));
            }
            return allo::PartitionAttr::get(&context, partitionAxes);
          },
          nb::arg("context"), nb::arg("sub_partitions"));

  nb::class_<transform::RaiseToAffineOp, OpState>(m, "RaiseToAffineOp")
      .def(
          "__init__",
          [](transform::RaiseToAffineOp &self, AlloOpBuilder &builder,
             Value &target) {
            self = transform::RaiseToAffineOp::create(
                builder, builder.getLocation(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::OutlineOp, OpState>(m, "OutlineOp")
      .def(
          "__init__",
          [](transform::OutlineOp &self, AlloOpBuilder &builder, Value &target,
             std::string_view kernelName) {
            self = transform::OutlineOp::create(builder, builder.getLocation(),
                                                target, kernelName);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("kernel_name"))
      .def(
          "__init__",
          [](transform::OutlineOp &self, AlloOpBuilder &builder, Value &target,
             std::string_view kernelName, const std::vector<int32_t> &mapping) {
            self = transform::OutlineOp::create(builder, builder.getLocation(),
                                                target, kernelName, mapping);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("kernel_name"),
          nb::arg("mapping"));

  nb::class_<transform::TagPipelineOp, OpState>(m, "TagPipelineOp")
      .def(
          "__init__",
          [](transform::TagPipelineOp &self, AlloOpBuilder &builder,
             Value &target, int ii) {
            self = transform::TagPipelineOp::create(
                builder, builder.getLocation(), target,
                static_cast<uint64_t>(ii));
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("ii"));

  nb::class_<transform::AlloLoopUnrollOp, OpState>(m, "AlloLoopUnrollOp")
      .def(
          "__init__",
          [](transform::AlloLoopUnrollOp &self, AlloOpBuilder &builder,
             Value &target, int64_t factor, bool tagOnly) {
            UnitAttr tagOnlyAttr = tagOnly ? builder.getUnitAttr() : UnitAttr();
            self = transform::AlloLoopUnrollOp::create(
                builder, builder.getLocation(), target,
                builder.getI64IntegerAttr(factor), tagOnlyAttr);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factor"),
          nb::arg("tag_only") = false);

  nb::class_<transform::LoopReorderOp, OpState>(m, "LoopReorderOp")
      .def(
          "__init__",
          [](transform::LoopReorderOp &self, AlloOpBuilder &builder,
             Value &target, const std::vector<int32_t> &order) {
            self = transform::LoopReorderOp::create(
                builder, builder.getLocation(), target, order);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("order"));

  nb::class_<transform::LoopSplitOp, OpState>(m, "LoopSplitOp")
      .def(
          "__init__",
          [](transform::LoopSplitOp &self, AlloOpBuilder &builder,
             Value &target, int factor) {
            self = transform::LoopSplitOp::create(
                builder, builder.getLocation(), target, factor);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factor"));

  nb::class_<transform::LoopTileOp, OpState>(m, "LoopTileOp")
      .def(
          "__init__",
          [](transform::LoopTileOp &self, AlloOpBuilder &builder, Value &target,
             const std::vector<int64_t> &factors) {
            self = transform::LoopTileOp::create(builder, builder.getLocation(),
                                                 target, factors);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("factors"));

  nb::class_<transform::LoopFlattenOp, OpState>(m, "LoopFlattenOp")
      .def(
          "__init__",
          [](transform::LoopFlattenOp &self, AlloOpBuilder &builder,
             Value &target) {
            self = transform::LoopFlattenOp::create(
                builder, builder.getLocation(), target);
          },
          nb::arg("builder"), nb::arg("target"));

  nb::class_<transform::ReuseAtOp, OpState>(m, "ReuseAtOp")
      .def(
          "__init__",
          [](transform::ReuseAtOp &self, AlloOpBuilder &builder, Value &target,
             Value &axis, bool ring) {
            self = transform::ReuseAtOp::create(builder, builder.getLocation(),
                                                target, axis, ring);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("axis"),
          nb::arg("ring") = false);

  nb::class_<transform::ComputeAtOp, OpState>(m, "ComputeAtOp")
      .def(
          "__init__",
          [](transform::ComputeAtOp &self, AlloOpBuilder &builder,
             Value &producer, Value &consumer) {
            self = transform::ComputeAtOp::create(
                builder, builder.getLocation(), producer, consumer);
          },
          nb::arg("builder"), nb::arg("producer"), nb::arg("consumer_loop"));

  nb::class_<transform::BufferAtOp, OpState>(m, "BufferAtOp")
      .def(
          "__init__",
          [](transform::BufferAtOp &self, AlloOpBuilder &builder, Value &target,
             Value &axis) {
            self = transform::BufferAtOp::create(builder, builder.getLocation(),
                                                 target, axis);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("axis"));

  nb::class_<transform::MatchValueOp, OpState>(m, "MatchValueOp")
      .def(
          "__init__",
          [](transform::MatchValueOp &self, AlloOpBuilder &builder,
             Value &target, int64_t index, int64_t sourceKind) {
            self = transform::MatchValueOp::create(
                builder, builder.getLocation(), target, index, sourceKind);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("index"),
          nb::arg("source_kind") = 0);

  nb::class_<transform::PartitionOp, OpState>(m, "PartitionOp")
      .def(
          "__init__",
          [](transform::PartitionOp &self, AlloOpBuilder &builder,
             Value &target, allo::PartitionAttr &partition) {
            self = transform::PartitionOp::create(
                builder, builder.getLocation(), target, partition);
          },
          nb::arg("builder"), nb::arg("target"), nb::arg("partition"));
}
