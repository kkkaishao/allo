/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// What a memory access's ADDRESS is, and what it costs as hardware.
//
// Two things live here because they have to agree. `addressExprsOf` states
// WHICH expressions an access's address hardware computes, and the rest of the
// file prices them. Three layers read the first (the scheduler, the
// strength-reduction planner, the emitter) and two read the second, so a
// disagreement between them is a schedule that was proved against arithmetic
// nobody built, or worse, one that never paid for arithmetic that is there.
//
// The cost is charged to the storage port the cone feeds
// (`OperatorLibrary::lookup`), because an address is folded into the access's
// affine map rather than standing as an operation of its own, so no dependence
// carries its delay. `simplifiedForHardware` additionally decides which of
// several equivalent forms of one address is the one to build.
//
// The cost is STRUCTURAL over the affine expression rather than a lookup in the
// device's `comb` delay table, because an affine map guarantees what a generic
// expression does not: a `Mul` is always by a constant and a `FloorDiv` / `Mod`
// always by a constant divisor. A constant multiply is a shift-add network, not
// a multiplier; a power-of-two divisor is wiring, not a divider. Pricing those
// through the `comb` mul/div rows would overstate the common case by an order
// of magnitude, which is the whole reason this is not a table lookup.
//===----------------------------------------------------------------------===//

#ifndef ALLO_SCHEDULING_ADDRESSCOST_H
#define ALLO_SCHEDULING_ADDRESSCOST_H

#include "allo/Scheduling/OperatorLibrary.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::allo {

struct BankLayout;

/// The expressions an access's address hardware computes, and the width they
/// are carried at. THE definition of "what address does this access build",
/// shared by the three layers that would otherwise each answer it: the
/// scheduler prices these, `planAddressGenerators` splits `offset` into the
/// registers that can carry it and what is left, and the emitter builds that
/// split beside the digit it evaluates here.
///
/// The flat row-major index is not among them, and that is the point. A banked
/// access addresses ONE bank at the in-bank offset, which is a different and
/// usually cheaper expression: `A[2*i]` under cyclic-2 has flat address `2*i`,
/// a shift-add, and offset `i`, no hardware at all. An access that also
/// computes its bank at runtime builds a second cone for the digit, which is a
/// real divider whenever the factor is not a power of two.
struct AddressExprs {
  AffineExpr offset;  // the element index WITHIN the bank this access reaches
  AffineExpr bank;    // which bank, or null when it is decided at compile time
  unsigned width = 0; // bits `offset` is carried at (one bank's, not the whole)
};

/// \p map's address expressions over a memref of \p shape banked as \p layout,
/// given the bank `assign-banks` assigned the access (nullopt when it roams).
///
/// Uniform over banked and unbanked: an unpartitioned memref is a one-bank
/// layout whose `offset` IS the flat row-major index and whose `bank` is the
/// constant 0 nothing builds, so there is one path rather than a special case.
AddressExprs addressExprsOf(const BankLayout &layout, AffineMap map,
                            llvm::ArrayRef<int64_t> shape,
                            std::optional<unsigned> assignedBank);

/// What an address expression costs as hardware: the critical path through it,
/// plus the operators it instantiates as an area proxy.
struct AddressCost {
  double delay = 0.0;  // ns, longest path from an operand to the address
  unsigned adders = 0; // carry chains, including a coefficient's shift-adds
  unsigned multipliers = 0; // generic multipliers (a NON-constant coefficient)
  unsigned dividers = 0;    // dividers / remainder units

  /// Nothing is instantiated: the address is wiring off an existing value.
  bool trivial() const {
    return adders == 0 && multipliers == 0 && dividers == 0;
  }
};

/// \p e simplified, unless simplifying made it WORSE to build.
///
/// `simplifyAffineExpr` is a canonicalizer, not a cost function, and the two
/// disagree in one direction that matters here: it flattens `x mod k` into
/// `x - (x floordiv k) * k`, three chained operators where the residue was a
/// mask. It also does real work no rewrite can replace, merging `(x mod 6) mod
/// 3` into one residue. Neither form wins in general, so several are built and
/// the shallower one is kept, ranked on a device-INDEPENDENT weighting so that
/// every layer asking for this expression is handed the same one. It has to be
/// device-independent: only `addressCostOf` holds an `OperatorLibrary`, while
/// the split is derived at three layers that do not.
///
/// THE WEIGHTS ARE INERT. Every candidate is a rewrite of one expression, and a
/// rewrite that helps removes a LEVEL rather than trading a divider for adders,
/// so the ranking degenerates to depth. Across the banking space (cyclic and
/// block, factors 2/3/4, one and two dimensional, coalesced and not) flattening
/// them to 1/1/1 selects the same form in all 120 cases, as does substituting
/// the real device's delays. What would make them load-bearing is a candidate
/// trading one operator kind for another, such as pricing a constant division
/// as the reciprocal multiply synthesis builds, and those numbers would have to
/// come from the device.
///
/// That candidate needs one more thing this signature does not carry: a BOUND
/// on the dividend. `(x*m) floordiv 2^s` equals `x floordiv k` only over a
/// range, and unbounded `m` is as wide as the datapath, so the form is not
/// representable at one evaluation width, let alone cheap. Bounded by a loop's
/// trip count it is a few shift-adds against a divider. The same bound is what
/// narrowing a `mod` whose dividend is smaller than its modulus asks for, so
/// the two arrive together or not at all.
AffineExpr simplifiedForHardware(AffineExpr e, unsigned numDims,
                                 unsigned numSymbols);

/// The device delays an address cone is priced against, read from the operator
/// library's combinational rows.
struct AddressDelays {
  double add = 0.0; // one carry-chain adder / subtractor
  double mul = 0.0; // a generic multiplier
  double div = 0.0; // a divider / remainder unit

  /// The width those numbers are characterized at. The device states one delay
  /// per kind with no width axis, and the emitter carries every address in i32
  /// (`uarch::hwWidth` of an `index`), so i32 is what the numbers already mean.
  /// A narrower cone is scaled linearly off this, which is the carry-chain
  /// approximation: an FPGA adder's delay tracks its width, while a shift or a
  /// mask costs no logic at any width. The real upgrade is a per-width
  /// characterization table, which is what Vitis queries per core instance.
  static constexpr unsigned refWidth = 32;
};

/// Read the comb rows an address cone can be built from.
AddressDelays addressDelaysOf(const OperatorLibrary &lib);

/// The cost of \p e when its arithmetic is carried at \p width bits.
///
/// It prices the hardware SYNTHESIS BUILDS, not the ops we happen to emit,
/// because the estimate exists to answer "does this fit in a clock period" and
/// the answer is a property of the netlist. Two places that matters:
///
/// * A constant coefficient is a signed-digit shift-add network, not a
///   multiplier, whether or not the emitter writes it as one. `x * 15` is
///   `(x << 4) - x`, one adder, and every tool recodes it that way.
/// * `width` is not free to choose. Truncation is reduction modulo `2^width`,
///   and `+`, `-` and `*` commute with it, so carrying them narrow is exact
///   given the address itself fits. `floordiv` and `mod` do NOT commute with
///   it, so a divider and everything feeding it stay at `refWidth` however
///   narrow the address is; only the divider's result may be truncated.
AddressCost addressCost(AffineExpr e, const AddressDelays &delays,
                        unsigned width);

/// The cost of \p map composed with \p shape's row-major strides, the FLAT
/// element index. Not what a banked access builds (`addressExprsOf` is), so
/// this is for asking about the flat form specifically, as
/// `loop-canonicalization` does when it checks whether coalescing would leave a
/// divider behind. A null \p map prices as zero, which is the stream and
/// non-access case: an ARRAY access always carries a map, the identity one when
/// its subscript is not affine.
AddressCost addressCost(AffineMap map, llvm::ArrayRef<int64_t> shape,
                        const AddressDelays &delays, unsigned width);

/// Whether a register can follow an operand, and its per-iteration step when
/// one can. The step is not decoration: a DIGIT of a counter is maintained by
/// wrapping a register once per iteration, so a step that could carry it past
/// two multiples of the divisor is not maintainable, and both layers have to
/// refuse the same ones.
using CarriedFn = llvm::function_ref<std::optional<int64_t>(unsigned)>;

/// An address as `base + sum(coeff * digit-of-operand) + residual`, where
/// `operand` indexes the access map's operands the way an affine access carries
/// them (dims, then symbols).
///
/// The terms are exactly the part a REGISTER can carry, which is more than the
/// part that advances by a constant. A scaled counter does: the difference
/// between consecutive values is constant, so it is advanced rather than
/// rebuilt. A DIGIT of one does not, and is a register all the same: `(x
/// floordiv D) mod K` advances on the iterations where `x` crosses a multiple
/// of `D` and wraps at `K`, which is a comparator off the address path where
/// the `floordiv` and `mod` were arithmetic on it. The residual is everything
/// else, in the operands' own numbering, and is null when there is nothing left
/// to evaluate.
///
/// Two terms may follow ONE counter, and each layer combines them by the
/// identity it holds (the scheduler by operand value, the builder by region):
/// map composition leaves `d0*6 + d0` where its folder did not reassociate, and
/// leaves two operand positions bound to one value. Uncombined, that is a
/// second register tracking a counter already tracked, plus the adder joining
/// them.
///
/// The split is PARTIAL, which is the whole point of the shape. An address is
/// not one decision: `A[i,j]` with `i` a counter and `j` data-dependent has a
/// row stride a register can follow and a column a register cannot, and taking
/// them together would cost the reduction to the weaker half, rebuilding `i*N`
/// every cycle to add `j` to it. A `floordiv` or `mod` over anything but a
/// counter lands in the residual for the same reason.
struct SplitAddress {
  /// One term a register can carry: `coeff * digit(scale * operand + offset)`,
  /// where `digit(x)` is `(x floordiv divisor) mod modulus`.
  ///
  /// With `divisor == 1` and no modulus that is the plain scaled counter, whose
  /// register advances by a constant. A DIGIT of one is periodic instead: it
  /// advances by nothing most iterations and wraps or carries on the ones where
  /// its argument crosses a multiple of `divisor`, which a register maintains
  /// just as cheaply (`RegionBlock::AddrStride`) and which a `floordiv` or
  /// `mod` on the address path pays for every cycle.
  struct Term {
    unsigned operand;
    int64_t coeff;
    int64_t scale = 1;   // the counter's own coefficient, inside the digit
    int64_t offset = 0;  // the counter's own constant, inside the digit
    int64_t divisor = 1; // 1: no `floordiv`
    int64_t modulus = 0; // 0: no `mod`
    bool isDigit() const { return divisor != 1 || modulus != 0; }
  };
  llvm::SmallVector<Term, 4> terms;
  int64_t base = 0;
  AffineExpr residual;
  /// Digits the residual READS rather than the address sums: a register costs
  /// nothing to read, so an operator that is cheap on a register but expensive
  /// on a counter belongs on top of one. `(x mod 5) floordiv 2` is the shape:
  /// the residue is a register and the `floordiv 2` over it is a shift, where
  /// the pair evaluated together is two real dividers.
  ///
  /// The residual names them as SYMBOLS numbered from the map's own
  /// `numSymbols`, so no existing leaf is renumbered and the emitter appends
  /// their values to the operand list it already passes.
  llvm::SmallVector<Term, 2> reads;
};

/// Split \p e, an address expression over \p numDims dims then symbols, with
/// \p carried naming the operands a register can follow.
///
/// Two layers decide the reduction independently (the scheduler, to price the
/// address it will get, and the emitter, to build it), so they share this one
/// definition of what the shape is, as `staticBankOf` is shared for banks. Both
/// pass it `addressExprsOf(...).offset`, so a banked access is split on the
/// expression its bank is actually addressed through.
///
/// A subtree holding nothing carried is residual WHOLE, never redistributed, so
/// an address that reduces nothing comes back out as it went in, priced and
/// built as the one expression it is.
SplitAddress splitAddress(AffineExpr e, unsigned numDims, unsigned numSymbols,
                          CarriedFn carried);

/// What \p addr costs once every term arrives from a register that advances
/// with its operand: the coefficients are gone and what is left is the network
/// summing the terms with the residual.
///
/// Priced in the order `buildAddr` writes it, one input per term and the
/// residual last, so the count is the emitter's chain rather than the adder
/// tree an optimal one would be. Entering last is not cosmetic: the residual's
/// own cone then runs BESIDE the registers' adders instead of under them. The
/// base costs nothing either way, absorbed into the first register's start
/// value, or, with no register to absorb it, the whole address.
AddressCost splitAddressCost(const SplitAddress &addr,
                             const AddressDelays &delays, unsigned width);

/// The width an address over \p shape is carried at: enough bits to index it,
/// which is what `DatapathEmitter::addrWidth` narrows to. Stated once here
/// because the pricing and the emitted datapath have to agree on it, and they
/// are decided in different passes. `addressExprsOf` applies it to the PER-BANK
/// shape, which is what one bank's address port is wide.
unsigned addressWidthOf(llvm::ArrayRef<int64_t> shape);

/// What the address of \p op costs AS THE EMITTER WILL BUILD IT. Zero for a
/// stream or a non-access; every array access is priced, its subscript affine
/// or not, since the row-major linearization over it and the bank digit off it
/// are address arithmetic either way.
///
/// BOTH cones are reduced and charged: the in-bank offset and, when the access
/// roams, the bank digit, which is itself a digit of a counter.
///
/// This is where the strength-reduction decision is made, once, for both
/// layers: every term that follows an enclosing counter with constant bounds is
/// carried in a register that advances with it
/// (`DatapathBuilder::planAddressGenerators`), so only the summing network and
/// whatever did not reduce are charged. Deciding it here rather than in each
/// layer is what stops the schedule from paying for arithmetic that is not
/// built, and, the dangerous direction, from not paying for arithmetic that
/// is.
///
/// The two layers ask the same question of different IR, and the builder asks
/// two more of the schedule that this cannot: whether the counter's bounds
/// resolved to constants, and whether the term is wanted in the same cycle as
/// the others. Both can only send a term back to the residual, so this is the
/// optimistic side of a pre-existing gap, not a new one.
///
/// A banked access is priced on its `AddressExprs`, not on the flat index. Both
/// halves matter: the offset is narrower and often free where the flat address
/// is not, and a runtime bank digit is a second cone the emitter builds either
/// way. The two run off the same operands, so the delay is their MAX while the
/// operator counts add.
AddressCost addressCostOf(Operation *op, const OperatorLibrary &lib);

/// `addressCostOf`'s delay, QUANTIZED to a hundredth of a nanosecond because
/// the caller names an operator type after this number: two sites whose names
/// agree must carry the same delay, or whichever registered last would silently
/// redefine the other.
double addressDelayOf(Operation *op, const OperatorLibrary &lib);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_ADDRESSCOST_H
