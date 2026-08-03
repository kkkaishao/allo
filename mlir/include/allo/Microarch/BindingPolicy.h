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

/// Timing data a policy needs beyond the model: the clock the schedule was cut
/// against, and the device rows that price a unit's inputs and a mux level.
struct BindingContext {
  float cycleTime;
  const OperatorLibrary &lib;
};

/// A resource-binding policy. `plan` inspects the trivially-bound datapath and
/// returns unit groups to merge; each group's units fold onto its first, units
/// not named keep their own unit. An empty result is the trivial binding (no
/// sharing). A policy only decides. It must not mutate `dp`.
struct BindingPolicy {
  virtual ~BindingPolicy() = default;
  virtual std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const = 0;
};

/// Every op keeps its own unit: `plan` returns no groups.
struct TrivialBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const override;
};

/// Greedy within-region sharing: fold same-operator-type units whose MRT
/// reservations are disjoint onto one unit (left-edge over the reservation
/// table), while the multiplexer that fold grows still fits the clock.
/// Interconnect-agnostic: it shares every compatible op whose timing allows,
/// regardless of operator area. A cost-driven policy can replace it.
struct GreedyShareBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const override;
};

/// Build the allocation the scheduler decided: fold together every unit whose
/// bound op names the same `dcp.compute` `unit` symbol, and leave every other
/// unit alone. An op the scheduler left unallocated (a combinational operator,
/// or the only one of its identity in its region) has nothing to share with.
struct PlannedBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const override;
};

/// The policy named by a pass option ("trivial" / "greedy-share" / "planned");
/// null on an unknown name.
std::unique_ptr<BindingPolicy> bindingPolicyFor(llvm::StringRef name);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_BINDINGPOLICY_H
