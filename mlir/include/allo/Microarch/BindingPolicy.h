/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_BINDINGPOLICY_H
#define ALLO_MICROARCH_BINDINGPOLICY_H

#include "allo/Microarch/Datapath.h"

#include <memory>
#include <vector>

namespace mlir::allo::uarch {

/// A resource-binding policy. `plan` inspects the trivially-bound datapath and
/// returns unit groups to merge; each group's units fold onto its first, units
/// not named keep their own unit. An empty result is the trivial binding (no
/// sharing). A policy only decides. It must not mutate `dp`.
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
