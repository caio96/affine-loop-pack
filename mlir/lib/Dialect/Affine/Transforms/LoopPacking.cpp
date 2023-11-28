//===- LoopPacking.cpp --- Loop packing pass ----------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that applies packing to loop nests if profitable
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPPACKING
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using llvm::dbgs;
using llvm::SmallMapVector;

#define DEBUG_TYPE "affine-loop-pack"

namespace {

/// A pass to perform loop packing on suitable packing candidates of a loop
/// nest.
struct LoopPacking : public affine::impl::AffineLoopPackingBase<LoopPacking> {
  LoopPacking() = default;

  void runOnOperation() override;
  void runOnOuterForOp(AffineForOp outerForOp,
                       DenseSet<Operation *> &copyNests);
};

} // end anonymous namespace

/// Creates a pass to perform loop packing.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createLoopPackingPass() {
  return std::make_unique<LoopPacking>();
}

/// Returns the mininum trip count of the loop if at least one map
/// element is a constant, std::nullopt otherwise.
/// Loops should be normalized before using this function.
static std::optional<uint64_t> getApproxTripCount(AffineForOp forOp) {
  SmallVector<Value> operands;
  AffineMap map;
  getTripCountMapAndOperands(forOp, &map, &operands);

  if (!map)
    return std::nullopt;

  // Loop nests with higher trip counts generally benefit more
  // from packing. To be conservative, this identifies the
  // minimum constant trip count in a loop.
  std::optional<uint64_t> tripCount;
  for (auto resultExpr : map.getResults()) {
    if (auto constExpr = dyn_cast<AffineConstantExpr>(resultExpr)) {
      if (tripCount.has_value())
        tripCount = std::min(tripCount.value(),
                             static_cast<uint64_t>(constExpr.getValue()));
      else
        tripCount = constExpr.getValue();
    }
  }
  return tripCount;
}

/// Given a forOp, this function will collect the memrefs of loads or stores.
/// Only collects memRefs with rank greater or equal to minRank.
static void getMemRefsInForOp(AffineForOp forOp, SetVector<Value> &memRefs,
                              unsigned minRank) {
  memRefs.clear();
  forOp.walk([&](Operation *op) {
    Value memRef;
    if (auto load = dyn_cast<AffineLoadOp>(op))
      memRef = load.getMemRef();
    else if (auto store = dyn_cast<AffineStoreOp>(op))
      memRef = store.getMemRef();
    else
      return WalkResult::advance();

    unsigned rank = memRef.getType().cast<MemRefType>().getRank();
    if (rank >= minRank)
      memRefs.insert(memRef);

    return WalkResult::advance();
  });
}

/// Stores information of an AffineForOp.
class PackingLoopInfo {
public:
  /// For loop
  AffineForOp forOp;
  /// Depth of forOp
  uint32_t depth;
  /// Constant trip count of forOp
  std::optional<uint64_t> tripCount;

  PackingLoopInfo() = default;

  PackingLoopInfo(AffineForOp forOp) : forOp(forOp) {
    // +1 otherwise a single forOp would have depth 0
    this->depth = getNestingDepth(this->forOp) + 1;
    this->tripCount = getApproxTripCount(this->forOp);
  };
};

/// Analyse and apply packing to a loop and its nestings.
void LoopPacking::runOnOuterForOp(AffineForOp outerForOp,
                                  DenseSet<Operation *> &copyNests) {
  // Skip copy forOps.
  if (copyNests.count(outerForOp) != 0)
    return;

  // Holds all memrefs with rank>=2 used in this for op and guarantees order.
  // The analysis requires a minimum rank of 2 because it is interested in
  // packing non contiguous portions of data such as a tile.
  SetVector<Value> memRefs;
  const int32_t minRank = 2;
  getMemRefsInForOp(outerForOp, memRefs, minRank);
  if (memRefs.empty()) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, no memRefs with rank >= 2 found.\n");
    return;
  }

  // Structure to information about the loops and their memrefs.
  SmallMapVector<AffineForOp, PackingLoopInfo, 4> loopInfoMap;
  uint32_t maxLoopDepth = 0;
  outerForOp.walk([&](AffineForOp forOp) {
    loopInfoMap[forOp] = PackingLoopInfo(forOp);
    maxLoopDepth = std::max(maxLoopDepth, loopInfoMap[forOp].depth);
  });

  // At least loop one loop of depth 3 is required for an invariant loop
  // to appear in memrefs with rank >= 2.
  const int32_t requiredLoopDepth = 3;
  if (maxLoopDepth < requiredLoopDepth) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, loops have a depth smaller than 3.\n");
    return;
  }
}

void LoopPacking::runOnOperation() {
  func::FuncOp f = getOperation();

  // Store created copies to skip them.
  DenseSet<Operation *> copyNests;
  copyNests.clear();

  for (auto &block : f) {
    // Every outer forOp.
    for (AffineForOp outerForOp : block.getOps<AffineForOp>())
      runOnOuterForOp(outerForOp, copyNests);
  }
}
