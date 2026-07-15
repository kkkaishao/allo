/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// BindingPolicy: the seam where resource sharing decisions are made. Following
// "store decisions, derive structure", a policy is a PURE decision -- given the
// trivially-allocated datapath (one FuncUnit per op) plus the MRT, it returns
// which units to fold onto one physical unit. The builder applies the decision
// (updating the binding maps) and re-derives the interconnect (the sharing
// muxes); the emitter never sees which policy ran.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_BINDINGPOLICY_H
#define ALLO_MICROARCH_BINDINGPOLICY_H

#include "allo/Microarch/Datapath.h"

#include <memory>
#include <vector>

namespace mlir::allo::uarch {

/// A resource-binding policy. `plan` inspects the trivially-bound datapath and
/// returns unit groups to merge; each group's units fold onto its first, units
/// not named keep their own unit. An empty result is the trivial binding (no
/// sharing). The policy only decides -- it must not mutate `dp`.
struct BindingPolicy {
  virtual ~BindingPolicy() = default;
  virtual std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp) const = 0;
};

/// Every op keeps its own unit: `plan` returns no groups.
struct TrivialBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp) const override;
};

/// Greedy within-region sharing: fold same-operator-type units whose MRT
/// reservations are disjoint onto one unit (left-edge over the reservation
/// table). Interconnect-agnostic: it shares every compatible op regardless of
/// operator area. A cost-driven policy can replace it.
struct GreedyShareBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp) const override;
};

/// The policy named by a pass option ("trivial" / "greedy-share"); null on an
/// unknown name.
std::unique_ptr<BindingPolicy> bindingPolicyFor(llvm::StringRef name);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_BINDINGPOLICY_H
