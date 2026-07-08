/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// YAML traits
//===----------------------------------------------------------------------===//

LLVM_YAML_IS_SEQUENCE_VECTOR(OperatorEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(MemPrimitive)

namespace llvm::yaml {

template <> struct ScalarEnumerationTraits<OpKind> {
  static void enumeration(IO &io, OpKind &k) {
    io.enumCase(k, "add", OpKind::Add);
    io.enumCase(k, "sub", OpKind::Sub);
    io.enumCase(k, "mul", OpKind::Mul);
    io.enumCase(k, "div", OpKind::Div);
    io.enumCase(k, "rem", OpKind::Rem);
    io.enumCase(k, "neg", OpKind::Neg);
    io.enumCase(k, "cmp", OpKind::Cmp);
    io.enumCase(k, "and", OpKind::And);
    io.enumCase(k, "or", OpKind::Or);
    io.enumCase(k, "xor", OpKind::Xor);
    io.enumCase(k, "shl", OpKind::Shl);
    io.enumCase(k, "shr", OpKind::Shr);
    io.enumCase(k, "select", OpKind::Select);
    io.enumCase(k, "int_cast", OpKind::ICastI);
    io.enumCase(k, "int_float_cast", OpKind::FCastI);
    io.enumCase(k, "float_cast", OpKind::FCastF);
    io.enumCase(k, "mem_read", OpKind::MemRead);
    io.enumCase(k, "mem_write", OpKind::MemWrite);
    io.enumCase(k, "stream_read", OpKind::StreamRead);
    io.enumCase(k, "stream_write", OpKind::StreamWrite);
  }
};

template <> struct ScalarEnumerationTraits<OpDType> {
  static void enumeration(IO &io, OpDType &d) {
    io.enumCase(d, "int", OpDType::Int);
    io.enumCase(d, "uint", OpDType::UInt);
    io.enumCase(d, "half", OpDType::Half);
    io.enumCase(d, "bfloat16", OpDType::BFloat16);
    io.enumCase(d, "float", OpDType::Float);
    io.enumCase(d, "double", OpDType::Double);
    io.enumCase(d, "any", OpDType::None);
  }
};

// `units:` is a dynamic map of pool-name -> unit count.
template <> struct CustomMappingTraits<std::map<std::string, uint32_t>> {
  static void inputOne(IO &io, StringRef key,
                       std::map<std::string, uint32_t> &v) {
    io.mapRequired(key.str().c_str(), v[key.str()]);
  }
  static void output(IO &io, std::map<std::string, uint32_t> &v) {
    for (auto &kv : v)
      io.mapRequired(kv.first.c_str(), kv.second);
  }
};

// `width: {min, max}` -- either bound optional.
template <> struct MappingTraits<WidthRange> {
  static void mapping(IO &io, WidthRange &w) {
    io.mapOptional("min", w.min);
    io.mapOptional("max", w.max);
  }
};

// `delay_ns:` is polymorphic: a scalar (symmetric, in == out) or a `{in, out}`
// map. `DelaySpec` carries ONLY PolymorphicTraits; the scalar view is a plain
// `double` and the map view a distinct `DelayMap` (distinct traits avoid the
// yamlize ambiguity). On input, the YAML node kind selects which optional is
// engaged; on output, `getKind` picks scalar iff the scalar form was used.
template <> struct MappingTraits<DelayMap> {
  static void mapping(IO &io, DelayMap &d) {
    io.mapOptional("in", d.in);
    io.mapOptional("out", d.out);
  }
};
template <> struct PolymorphicTraits<DelaySpec> {
  static NodeKind getKind(const DelaySpec &d) {
    return d.scalar ? NodeKind::Scalar : NodeKind::Map;
  }
  static double &getAsScalar(DelaySpec &d) {
    if (!d.scalar)
      d.scalar = 0.0;
    return *d.scalar;
  }
  static DelayMap &getAsMap(DelaySpec &d) {
    if (!d.map)
      d.map.emplace();
    return *d.map;
  }
  static DelayMap &getAsSequence(DelaySpec &d) { return getAsMap(d); } // unused
};

template <> struct MappingTraits<OperatorEntry> {
  static void mapping(IO &io, OperatorEntry &e) {
    io.mapOptional("op", e.kind);
    io.mapOptional("dtype", e.dtype);
    io.mapOptional("mlir_op", e.mlirOp);
    io.mapOptional("width", e.width);
    io.mapOptional("latency", e.latency);
    io.mapOptional("delay_ns", e.delay);
    io.mapOptional("pipelined", e.pipelined);
    io.mapOptional("unit", e.unit);
    io.mapOptional("impl", e.impl);
  }
};

// `memory:` section -- the storage-timing library. `default:` is the storage
// implementation unbound arrays use; `primitives:` characterizes each
// implementation's read/write timing; `fifo:` is stream get/put timing. Latency
// and delay are each split by direction (`read`/`write`).
template <> struct ScalarEnumerationTraits<MemoryImplEnum> {
  static void enumeration(IO &io, MemoryImplEnum &v) {
    io.enumCase(v, "auto", MemoryImplEnum::Auto);
    io.enumCase(v, "register", MemoryImplEnum::Register);
    io.enumCase(v, "lutram", MemoryImplEnum::LUTRAM);
    io.enumCase(v, "bram", MemoryImplEnum::BRAM);
    io.enumCase(v, "uram", MemoryImplEnum::URAM);
  }
};
template <> struct MappingTraits<RWLatency> {
  static void mapping(IO &io, RWLatency &v) {
    io.mapOptional("read", v.read);
    io.mapOptional("write", v.write);
  }
};
template <> struct MappingTraits<RWDelay> {
  static void mapping(IO &io, RWDelay &v) {
    io.mapOptional("read", v.read);
    io.mapOptional("write", v.write);
  }
};
template <> struct MappingTraits<MemKindTiming> {
  static void mapping(IO &io, MemKindTiming &m) {
    io.mapOptional("latency", m.latency);
    io.mapOptional("delay_ns", m.delay);
  }
};
template <> struct MappingTraits<MemPrimitive> {
  static void mapping(IO &io, MemPrimitive &p) {
    io.mapRequired("name", p.impl);
    io.mapOptional("latency", p.timing.latency);
    io.mapOptional("delay_ns", p.timing.delay);
  }
};
template <> struct MappingTraits<MemoryLibrary> {
  static void mapping(IO &io, MemoryLibrary &m) {
    io.mapOptional("default", m.defaultImpl);
    io.mapOptional("primitives", m.primitives);
    io.mapOptional("fifo", m.fifo);
  }
};

template <> struct MappingTraits<OperatorLibrary> {
  static void mapping(IO &io, OperatorLibrary &lib) {
    io.mapOptional("device", lib.device);
    io.mapOptional("frequency_mhz", lib.frequencyMhz);
    io.mapOptional("cycle_time_ns", lib.cycleTimeNs);
    io.mapOptional("units", lib.units);
    io.mapOptional("operators", lib.entries);
    io.mapOptional("advanced_operators", lib.advancedEntries);
    io.mapOptional("default", lib.defaultEntry);
    io.mapOptional("memory", lib.memory);
  }
};

} // namespace llvm::yaml

//===----------------------------------------------------------------------===//
// Default library
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::allo::defaultLibraryYAML() {
  // The complete built-in library -- MUST stay in sync with the Python mirror
  // allo/backend/rtl/oplib/builtin.yaml (see its header). Every abstract kind
  // carries a timing and a realization (`impl`: `comb` or an IP module name
  // `<f|d|i><op>_l<lat>`). Single-precision float latencies are measured on
  // U55C at 300 MHz; double is ~2x; integer mul/div/rem and delays are
  // placeholders. Integer add/sub stay combinational (a real gate delay);
  // compare/select/ shift/bitwise/int-resize are combinational; float/double
  // cores and int<-> float conversions are multi-cycle IP.
  return R"yaml(
device: builtin
frequency_mhz: 300.0
operators:
  - op: add
    dtype: int
    latency: 0
    delay_ns: 1.2
    impl: comb
  - op: sub
    dtype: int
    latency: 0
    delay_ns: 1.2
    impl: comb
  - op: mul
    dtype: int
    latency: 3
    impl: imul_l3
  - op: div
    dtype: int
    latency: 20
    impl: idiv_l20
  - op: rem
    dtype: int
    latency: 20
    impl: irem_l20
  - op: neg
    dtype: int
    latency: 0
    delay_ns: 1.0
    impl: comb
  - op: cmp
    dtype: int
    latency: 0
    delay_ns: 1.0
    impl: comb
  - op: and
    dtype: int
    latency: 0
    delay_ns: 0.4
    impl: comb
  - op: or
    dtype: int
    latency: 0
    delay_ns: 0.4
    impl: comb
  - op: xor
    dtype: int
    latency: 0
    delay_ns: 0.4
    impl: comb
  - op: shl
    dtype: int
    latency: 0
    delay_ns: 0.5
    impl: comb
  - op: shr
    dtype: int
    latency: 0
    delay_ns: 0.5
    impl: comb
  - op: select
    dtype: int
    latency: 0
    delay_ns: 0.5
    impl: comb
  - op: int_cast
    dtype: int
    latency: 0
    delay_ns: 0.3
    impl: comb
  - op: add
    dtype: float
    latency: 7
    delay_ns: 0.5
    impl: fadd_l7
  - op: sub
    dtype: float
    latency: 7
    delay_ns: 0.5
    impl: fsub_l7
  - op: mul
    dtype: float
    latency: 4
    delay_ns: 0.5
    impl: fmul_l4
  - op: div
    dtype: float
    latency: 12
    delay_ns: 0.5
    impl: fdiv_l12
  - op: cmp
    dtype: float
    latency: 1
    delay_ns: 0.5
    impl: fcmp_l1
  - op: select
    dtype: float
    latency: 0
    delay_ns: 0.5
    impl: comb
  - op: neg
    dtype: float
    latency: 0
    delay_ns: 0.3
    impl: comb
  - op: add
    dtype: double
    latency: 14
    delay_ns: 0.5
    impl: dadd_l14
  - op: sub
    dtype: double
    latency: 14
    delay_ns: 0.5
    impl: dsub_l14
  - op: mul
    dtype: double
    latency: 9
    delay_ns: 0.5
    impl: dmul_l9
  - op: div
    dtype: double
    latency: 24
    delay_ns: 0.5
    impl: ddiv_l24
  - op: cmp
    dtype: double
    latency: 1
    delay_ns: 0.5
    impl: dcmp_l1
  - op: select
    dtype: double
    latency: 0
    delay_ns: 0.5
    impl: comb
  - op: neg
    dtype: double
    latency: 0
    delay_ns: 0.3
    impl: comb
  - op: add
    dtype: bfloat16
    latency: 4
    delay_ns: 0.5
    impl: bfadd_l4
  - op: sub
    dtype: bfloat16
    latency: 4
    delay_ns: 0.5
    impl: bfsub_l4
  - op: mul
    dtype: bfloat16
    latency: 2
    delay_ns: 0.5
    impl: bfmul_l2
  - op: int_float_cast
    latency: 3
    delay_ns: 0.5
    impl: i2f_l3
  - op: float_cast
    latency: 2
    delay_ns: 0.5
    impl: fcvt_l2
default:
  latency: 0
  delay_ns: 0.1
  impl: builtin
memory:
  default: lutram
  primitives:
    - name: register
      latency: {read: 0, write: 1}
      delay_ns: {read: 0.1, write: 0.1}
    - name: lutram
      latency: {read: 1, write: 1}
      delay_ns: {read: 0.5, write: 0.5}
    - name: bram
      latency: {read: 1, write: 1}
      delay_ns: {read: 0.7, write: 0.7}
    - name: uram
      latency: {read: 2, write: 1}
      delay_ns: {read: 0.9, write: 0.9}
  fifo:
    latency: {read: 1, write: 1}
    delay_ns: {read: 0.5, write: 0.5}
)yaml";
}

const OperatorLibrary &OperatorLibrary::defaultLibrary() {
  static OperatorLibrary lib = llvm::cantFail(parse(defaultLibraryYAML()));
  return lib;
}

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

llvm::Expected<OperatorLibrary> OperatorLibrary::parse(llvm::StringRef yaml) {
  OperatorLibrary lib;
  llvm::yaml::Input in(yaml);
  in >> lib;
  if (std::error_code ec = in.error())
    return llvm::createStringError(ec, "failed to parse operator library YAML");

  // Validate allocation-pool references: a `unit` must be declared in `units`
  // and sit on a non-zero-latency op (a combinational unit is not limited).
  auto validate = [&](const std::vector<OperatorEntry> &es) -> llvm::Error {
    for (const OperatorEntry &e : es) {
      if (!e.unit)
        continue;
      if (!lib.units.count(*e.unit))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "unit '%s' is not declared in `units`",
                                       e.unit->c_str());
      if (e.latency == 0)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "unit '%s' on a zero-latency (combinational) operator",
            e.unit->c_str());
    }
    return llvm::Error::success();
  };
  if (llvm::Error err = validate(lib.entries))
    return std::move(err);
  if (llvm::Error err = validate(lib.advancedEntries))
    return std::move(err);

  // Validate realization (`impl`) for internal consistency. `impl` is optional
  // here: it is only consumed on the HW-emission path (where a compute op
  // lacking a realization errors by name), so a scheduling-only library may
  // omit it. What we do reject: `impl` on a mem/stream row (it does not apply),
  // and a native keyword on a multi-cycle operator (native is combinational; a
  // multi-cycle op must name an IP).
  auto isComputeKind = [](OpKind k) {
    return k != OpKind::MemRead && k != OpKind::MemWrite &&
           k != OpKind::StreamRead && k != OpKind::StreamWrite;
  };
  auto checkImpl = [](const OperatorEntry &e, const char *what,
                      bool compute) -> llvm::Error {
    if (e.impl.empty())
      return llvm::Error::success();
    if (!compute)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: memory/stream operators take no 'impl'", what);
    if (isNativeImpl(e.impl) && e.latency != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s: native 'impl: %s' requires latency 0; a multi-cycle operator "
          "needs an IP name",
          what, e.impl.c_str());
    return llvm::Error::success();
  };
  for (const OperatorEntry &e : lib.entries)
    if (e.kind)
      if (llvm::Error err =
              checkImpl(e, "operators row", isComputeKind(*e.kind)))
        return std::move(err);
  for (const OperatorEntry &e : lib.advancedEntries)
    if (llvm::Error err = checkImpl(e, "advanced_operators row", true))
      return std::move(err);
  if (llvm::Error err = checkImpl(lib.defaultEntry, "default row", true))
    return std::move(err);
  return lib;
}

llvm::Expected<OperatorLibrary>
OperatorLibrary::loadFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf =
      llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = buf.getError())
    return llvm::createStringError(ec, "cannot open operator library '%s'",
                                   path.str().c_str());
  return parse((*buf)->getBuffer());
}

std::optional<double> OperatorLibrary::cycleTime() const {
  if (cycleTimeNs)
    return cycleTimeNs;
  if (frequencyMhz && *frequencyMhz > 0.0)
    return 1000.0 / *frequencyMhz;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Classification: concrete IR op -> abstract signature
//===----------------------------------------------------------------------===//

namespace {

// The numeric (int/float) type characterizing `op`: result(0)'s type (element
// type if shaped), else operand(0)'s; null if neither is numeric.
Type numericType(Operation *op) {
  auto asNumeric = [](Type t) -> Type {
    if (auto shaped = dyn_cast<ShapedType>(t))
      t = shaped.getElementType();
    return t.isIntOrFloat() ? t : Type();
  };
  if (op->getNumResults() > 0)
    if (Type t = asNumeric(op->getResult(0).getType()))
      return t;
  if (op->getNumOperands() > 0)
    if (Type t = asNumeric(op->getOperand(0).getType()))
      return t;
  return Type();
}

// The type that characterizes `op` of kind `kind`. For int<->float conversions
// the float format drives converter timing, so we key on the float side (making
// both directions -- e.g. sitofp and fptosi -- match one row); otherwise the
// generic numeric type is used.
Type characteristicType(Operation *op, OpKind kind) {
  if (kind == OpKind::FCastI) {
    for (Type t : op->getResultTypes())
      if (isa<FloatType>(t))
        return t;
    for (Value v : op->getOperands())
      if (isa<FloatType>(v.getType()))
        return v.getType();
  }
  // A compare's result is i1; its realization (an integer vs floating-point
  // comparator) is set by the OPERANDS it compares, so key on the operand type,
  // not the i1 result -- else arith.cmpf would match a `cmp dtype: int` row.
  if (kind == OpKind::Cmp && op->getNumOperands() > 0) {
    Type t = op->getOperand(0).getType();
    if (auto shaped = dyn_cast<ShapedType>(t))
      t = shaped.getElementType();
    if (t.isIntOrFloat())
      return t;
  }
  return numericType(op);
}

OpDType floatDType(Type t) {
  if (t.isF16())
    return OpDType::Half;
  if (t.isBF16())
    return OpDType::BFloat16;
  if (t.isF64())
    return OpDType::Double;
  return OpDType::Float; // f32 and other float widths -> generic float family.
}

// Ops the IR marks as operating on unsigned integers.
bool isUnsignedOp(llvm::StringRef name) {
  return name == "arith.divui" || name == "arith.remui" ||
         name == "arith.shrui" || name == "arith.extui" ||
         name == "arith.uitofp" || name == "arith.fptoui";
}

} // namespace

OpSignature mlir::allo::classify(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  OpKind kind =
      llvm::StringSwitch<OpKind>(name)
          .Cases({"arith.addi", "arith.addf"}, OpKind::Add)
          .Cases({"arith.subi", "arith.subf"}, OpKind::Sub)
          .Cases({"arith.muli", "arith.mulf"}, OpKind::Mul)
          .Cases({"arith.divsi", "arith.divui", "arith.divf"}, OpKind::Div)
          .Cases({"arith.remsi", "arith.remui", "arith.remf"}, OpKind::Rem)
          .Case("arith.negf", OpKind::Neg)
          .Cases({"arith.cmpi", "arith.cmpf"}, OpKind::Cmp)
          .Case("arith.andi", OpKind::And)
          .Case("arith.ori", OpKind::Or)
          .Case("arith.xori", OpKind::Xor)
          .Case("arith.shli", OpKind::Shl)
          .Cases({"arith.shrsi", "arith.shrui"}, OpKind::Shr)
          .Case("arith.select", OpKind::Select)
          .Cases({"arith.extsi", "arith.extui", "arith.trunci"}, OpKind::ICastI)
          .Cases({"arith.index_cast", "arith.index_castui"}, OpKind::ICastI)
          .Cases({"arith.sitofp", "arith.uitofp"}, OpKind::FCastI)
          .Cases({"arith.fptosi", "arith.fptoui"}, OpKind::FCastI)
          .Cases({"arith.extf", "arith.truncf"}, OpKind::FCastF)
          .Cases({"affine.load", "memref.load"}, OpKind::MemRead)
          .Cases({"affine.store", "memref.store"}, OpKind::MemWrite)
          .Case("allo.stream.get", OpKind::StreamRead)
          .Case("allo.stream.put", OpKind::StreamWrite)
          .Default(OpKind::Unknown);

  OpDType dtype = OpDType::None;
  unsigned width = 0;
  if (Type t = characteristicType(op, kind)) {
    width = t.getIntOrFloatBitWidth();
    if (isa<FloatType>(t))
      dtype = floatDType(t);
    else
      dtype = isUnsignedOp(name) ? OpDType::UInt : OpDType::Int;
  }
  return {kind, dtype, width};
}

//===----------------------------------------------------------------------===//
// Realization predicate (shared by the parser and EmitHW). Whether an op's
// native EmitHW lowering exists is EmitHW's own concern (see `combEmitted`).
//===----------------------------------------------------------------------===//

bool mlir::allo::isNativeImpl(llvm::StringRef impl) {
  return impl == "comb" || impl == "hwarith" || impl == "builtin";
}

StallContract mlir::allo::stallContract(llvm::StringRef impl) {
  assert(!isNativeImpl(impl) &&
         "native impl is stateless -- no stall contract");
  // Every IP in the library today is a fixed-latency pipeline exposing a
  // clock-enable. The `stall:` YAML override (FreeRunning / Elastic) plugs in
  // here when a specific operator needs it.
  return StallContract::ClockEnable;
}

//===----------------------------------------------------------------------===//
// Lookup
//===----------------------------------------------------------------------===//

namespace {

// `int` is the umbrella datatype: a row wanting `int` also matches unsigned
// ops.
bool dtypeMatches(OpDType want, OpDType actual) {
  return want == actual || (want == OpDType::Int && actual == OpDType::UInt);
}

// A row's width predicate holds. An op with no numeric width (0) can never
// satisfy a width predicate, but matches a row that declares none.
bool widthMatches(const OperatorEntry &e, unsigned width) {
  if (!e.width)
    return true;
  if (width == 0)
    return false;
  if (e.width->min && width < *e.width->min)
    return false;
  if (e.width->max && width > *e.width->max)
    return false;
  return true;
}

} // namespace

// `typeName` must be stable and unique per entry so that ops matching the same
// entry share one operator type.
OperatorChar OperatorLibrary::resolveEntry(const OperatorEntry &e,
                                           llvm::StringRef typeName) const {
  OperatorChar c;
  c.typeName = typeName.str();
  c.latency = e.latency;
  c.pipelined = e.pipelined;
  c.impl = e.impl;
  if (e.delay) {
    c.inDelay = e.delay->inNs();
    c.outDelay = e.delay->outNs();
    // A multi-cycle operator registers its output (a pipelined IP / hard
    // block): nothing combinational follows its final stage, so its outgoing
    // delay is 0 and it terminates a chain. The scalar `delay_ns` shorthand is
    // then the setup delay into the first stage. An explicit `{in, out}` map
    // overrides this -- authored for a unit whose last cycle is combinational
    // (a nonzero tail that chains into a same-cycle successor).
    if (e.latency > 0 && e.delay->scalar)
      c.outDelay = 0.0;
  }
  // A zero-latency (combinational) operator must have equal in/out delays.
  if (e.latency == 0 && c.inDelay != c.outDelay)
    c.inDelay = c.outDelay = std::max(c.inDelay, c.outDelay);
  if (e.unit) {
    c.unit = *e.unit;
    auto it = units.find(*e.unit);
    c.unitLimit = it != units.end() ? it->second : 0;
  }
  return c;
}

OperatorChar OperatorLibrary::lookup(Operation *op) const {
  // Advanced (raw MLIR op name) rows match first.
  llvm::StringRef name = op->getName().getStringRef();
  OpSignature sig = classify(op);

  // Memory / stream accesses are the storage dimension: characterized by the
  // memory library (`memory:`), not the compute operator table. All accesses of
  // one direction share a stable operator type.
  switch (sig.kind) {
  case OpKind::MemRead:
  case OpKind::MemWrite:
  case OpKind::StreamRead:
  case OpKind::StreamWrite: {
    MemoryLibrary::Timing t = memory.timing(op);
    OperatorChar c;
    // Array accesses key the operator type on their storage implementation:
    // loads of a 0-cycle register and a 2-cycle URAM must be *distinct*
    // operator types, or they collapse onto one shared latency.
    c.typeName = (sig.kind == OpKind::MemRead      ? "mem.read."
                  : sig.kind == OpKind::MemWrite   ? "mem.write."
                  : sig.kind == OpKind::StreamRead ? "stream.read"
                                                   : "stream.write");
    if (sig.kind == OpKind::MemRead || sig.kind == OpKind::MemWrite)
      c.typeName += stringifyMemoryImplEnum(t.impl).str();
    c.latency = t.latency;
    c.inDelay = c.outDelay = t.delay;
    c.pipelined = t.pipelined;
    return c;
  }
  default:
    break;
  }

  for (const auto &[idx, e] : llvm::enumerate(advancedEntries)) {
    if (e.mlirOp != name)
      continue;
    if (e.dtype && !dtypeMatches(*e.dtype, sig.dtype))
      continue;
    if (!widthMatches(e, sig.width))
      continue;
    return resolveEntry(e, ("adv#" + llvm::Twine(idx)).str());
  }

  for (const auto &[idx, e] : llvm::enumerate(entries)) {
    if (!e.kind || *e.kind != sig.kind)
      continue;
    if (e.dtype && !dtypeMatches(*e.dtype, sig.dtype))
      continue;
    if (!widthMatches(e, sig.width))
      continue;
    return resolveEntry(e, ("op#" + llvm::Twine(idx)).str());
  }

  return resolveEntry(defaultEntry, "default");
}

std::vector<std::string>
OperatorLibrary::unregisteredAdvancedOps(MLIRContext &ctx) const {
  std::vector<std::string> unknown;
  for (const OperatorEntry &e : advancedEntries)
    if (!e.mlirOp.empty() && !ctx.isOperationRegistered(e.mlirOp))
      unknown.push_back(e.mlirOp);
  return unknown;
}

// The per-memref memory-port/banking model moved to MemoryModel.{h,cpp} (the
// storage dimension). `populateOperatorTypesImpl` still drives it via
// `detail::MemoryBankModel`.
