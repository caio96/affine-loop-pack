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

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
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

/// Verifies if a memRefRegion is contiguous within a tile.
/// Given a tile shape, after dropping consecutive outermost dimensions of size
/// '1', and consecutive innermost dimensions of full size (equal to the memRef
/// shape), if the resulting shape is >= 2D, i.e. matrix or above, the memref
/// accesses are not all contiguous within the tile.
/// For example:
///         Let memRefShape = [ 128 128 128 ]
///         Let   tileShape = [   1  32  32 ]
/// Following the criteria, this function 'trims' the tile shape to [ 32 32 ],
/// indicating that the tile accesses non-contiguous elements in memory.
/// i.e. there is a stride between each mult(32) and mult(32)+1 memory access
/// where mult() represents a multiple of 32.
static bool isTileShapeContiguous(ArrayRef<int64_t> memRefShape,
                                  ArrayRef<int64_t> tileShape) {
  assert(memRefShape.size() == tileShape.size());
  int32_t begin = 0;
  int32_t end = memRefShape.size() - 1;
  int32_t innerDims = 0;

  // 'Drop' the 1's from the outermost dimensions until a non-1 is encountered.
  while (begin <= end && tileShape[begin] == 1)
    begin++;

  // If the entire tile is ones then it is a single element access in the
  // structure and thus contiguous.
  if (begin == static_cast<int32_t>(memRefShape.size()))
    return true;

  // 'Drop' the full size dimensions from the innermost dimensions until a
  // non-full is encountered.
  // TODO: Handle symbolic values.
  while (0 <= end && tileShape[end] == memRefShape[end])
    end--;

  // If the tile shape completely matches the memref region shape then the
  // accesses are continuous.
  if (end == -1)
    return true;

  // After 'dropping' according to the conditions above, if the resulting shape
  // is not a vector (1D) then the memref accesses are not contiguous.
  innerDims = end - begin + 1;
  return innerDims < 2;
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

/// Check if a memRef is loaded in a forOp.
static bool isMemRefLoadedInForOp(AffineForOp forOp, Value memRef) {
  auto walkRes = forOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      if (memRef == loadOp.getMemRef())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return walkRes.wasInterrupted();
}

/// Returns true if all accesses to memRef inside forOp
/// are invariant to forOp.
static bool memRefAccessesInvariantToLoop(AffineForOp forOp, Value memRef) {

  SmallVector<AffineLoadOp> loads;
  SmallVector<AffineStoreOp> stores;

  // Gather loads and stores.
  forOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      if (memRef == loadOp.getMemRef())
        loads.push_back(loadOp);
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (memRef == storeOp.getMemRef())
        stores.push_back(storeOp);
    }
  });

  if (loads.empty() && stores.empty())
    return true;

  // Collects indices used in all loads and stores.
  DenseSet<Value> accessIndices;
  for (AffineLoadOp load : loads)
    for (Value operand : load.getIndices())
      accessIndices.insert(operand);
  for (AffineStoreOp store : stores)
    for (Value operand : store.getIndices())
      accessIndices.insert(operand);

  // Collect the IV thisLoopIV of forOp and add dependent loop IVs:
  // If thisLoopIV is used in any other inner loops as an upper or lower bound
  // operand, add the IV of these inner loops as a dependent loop IVs.
  // Do this iteratively for the new IVs as well.
  DenseSet<Value> loopIVs;
  DenseSet<Value> checkDepsList;
  auto thisLoopIV = forOp.getInductionVar();
  checkDepsList.insert(thisLoopIV);

  // Check dependencies and then move loop IV to loopIVs.
  while (!checkDepsList.empty()) {
    // Pop IV from deps list
    Value currLoopIV = *checkDepsList.begin();
    checkDepsList.erase(currLoopIV);

    // Check if it is was not visited already
    if (!loopIVs.insert(currLoopIV).second)
      continue;

    AffineForOp currForOp = getForInductionVarOwner(currLoopIV);

    currForOp.walk([&](AffineForOp walkForOp) {
      if (walkForOp == currForOp)
        return WalkResult::advance();

      for (auto upOperand : walkForOp.getUpperBoundOperands()) {
        if (currLoopIV == upOperand) {
          checkDepsList.insert(walkForOp.getInductionVar());
          return WalkResult::advance();
        }
      }
      for (auto lbOperand : walkForOp.getLowerBoundOperands()) {
        if (currLoopIV == lbOperand) {
          checkDepsList.insert(walkForOp.getInductionVar());
          return WalkResult::advance();
        }
      }
      return WalkResult::advance();
    });
  }

  // Check if all indices are invariant to all IVs
  for (auto accessVal : accessIndices) {
    for (auto loopVal : loopIVs)
      if (!isAccessIndexInvariant(loopVal, accessVal))
        return false;
  }

  // If all acesses are invariant, the loop is invariant to the memref.
  return true;
}

/// Computes memRefRegions for a memRef within a forOp.
static void createMemRefRegions(AffineForOp forOp, Value memRef,
                                std::unique_ptr<MemRefRegion> &readRegion,
                                std::unique_ptr<MemRefRegion> &writeRegion) {
  // To check for errors when walking the block.
  bool error = false;
  unsigned depth = getNestingDepth(forOp);

  // Walk this range of operations to gather all memory regions.
  forOp.walk([&](Operation *op) {
    // Skip ops that are not loads and stores of memRef
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      if (memRef != loadOp.getMemRef())
        return WalkResult::advance();
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (memRef != storeOp.getMemRef())
        return WalkResult::advance();
    } else {
      return WalkResult::advance();
    }

    // Compute the MemRefRegion accessed.
    auto region = std::make_unique<MemRefRegion>(op->getLoc());
    if (failed(region->compute(op, depth, /*sliceState=*/nullptr,
                               /*addMemRefDimBounds=*/false))) {
      LLVM_DEBUG(dbgs() << "[DEBUG] Error obtaining memory region of " << memRef
                        << " : semi-affine maps?\n");
      error = true;
      return WalkResult::interrupt();
    }

    // Attempts to update regions
    auto updateRegion = [&](const std::unique_ptr<MemRefRegion> &targetRegion) {
      if (!targetRegion) {
        return;
      }

      // Perform a union with the existing region.
      if (failed(targetRegion->unionBoundingBox(*region))) {
        LLVM_DEBUG(dbgs() << "[DEBUG] Error obtaining memory region of "
                          << memRef << " : semi-affine maps?\n");
        error = true;
        return;
      }
      // Union was computed and stored in 'targetRegion': copy to 'region'.
      region->getConstraints()->clearAndCopyFrom(
          *targetRegion->getConstraints());
    };

    // Update region if region already exists.
    updateRegion(readRegion);
    if (error)
      return WalkResult::interrupt();
    updateRegion(writeRegion);
    if (error)
      return WalkResult::interrupt();

    // Add region if region is empty.
    if (region->isWrite() && !writeRegion) {
      writeRegion = std::move(region);
    } else if (!region->isWrite() && !readRegion) {
      readRegion = std::move(region);
    }

    return WalkResult::advance();
  });

  if (error) {
    Block::iterator begin = Block::iterator(forOp);
    begin->emitWarning("Creating and updating regions failed in this block\n");
    // clean regions
    readRegion = nullptr;
    writeRegion = nullptr;
  }
}

/// Get tile shape of memRef given a memRefRegion.
/// Result is returned in tileShape.
static void computeTileShape(Value memRef,
                             std::unique_ptr<MemRefRegion> &region,
                             llvm::SmallVectorImpl<int64_t> &tileShape) {
  auto memRefType = memRef.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  tileShape.clear();

  // Compute the extents of the buffer.
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  std::optional<int64_t> numElements =
      region->getConstantBoundingSizeAndShape(&tileShape, &lbs, &lbDivisors);
  if (!numElements.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-constant region size not supported\n");
  }
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
  /// Map from memref to a it's tile shape in this loop
  DenseMap<Value, SmallVector<int64_t>> memRefTileShapeMap;
  /// Map from memref to it's read region
  DenseMap<Value, std::unique_ptr<MemRefRegion>> memRefReadRegion;
  /// Map from memref to it's write region
  DenseMap<Value, std::unique_ptr<MemRefRegion>> memRefWriteRegion;

  PackingLoopInfo() = default;

  PackingLoopInfo(AffineForOp forOp) : forOp(forOp) {
    // +1 otherwise a single forOp would have depth 0
    this->depth = getNestingDepth(this->forOp) + 1;
    this->tripCount = getApproxTripCount(this->forOp);
  };

  /// Returns the read region of a memref in this loop, if known.
  /// Otherwise, return a nullptr.
  /// The result is computed only when requested and then stored.
  std::unique_ptr<MemRefRegion> &getMemRefReadRegion(Value memRef) {
    if (this->memRefReadRegion.count(memRef) == 0)
      createMemRefRegions(this->forOp, memRef, this->memRefReadRegion[memRef],
                          this->memRefWriteRegion[memRef]);

    return this->memRefReadRegion[memRef];
  }

  /// Returns the write region of a memref in this loop, if known.
  /// Otherwise, return a nullptr.
  /// The result is computed only when requested and then stored.
  std::unique_ptr<MemRefRegion> &getMemRefWriteRegion(Value memRef) {
    if (this->memRefWriteRegion.count(memRef) == 0)
      createMemRefRegions(this->forOp, memRef, this->memRefReadRegion[memRef],
                          this->memRefWriteRegion[memRef]);

    return this->memRefWriteRegion[memRef];
  }

  /// Returns the tile shape of a memref in this loop if known.
  /// Otherwise returns an empty array.
  /// The result is computed only when requested and then stored.
  ArrayRef<int64_t> getMemRefTileShape(Value memRef) {
    if (this->memRefTileShapeMap.count(memRef) == 0) {
      // Using only the read region as the tile shape should be equivalent
      // for write the region (and we don't have to pack a write only region)
      auto &region = this->getMemRefReadRegion(memRef);
      if (region)
        computeTileShape(memRef, region, this->memRefTileShapeMap[memRef]);
      else
        this->memRefTileShapeMap[memRef] = SmallVector<int64_t>();
    }
    return this->memRefTileShapeMap[memRef];
  }
};

/// Information about a packing candidate.
class PackingAttributes {
public:
  /// ID, for printing purposes and for enabling manual selection of candidates.
  int64_t id = -1;
  /// Target memRef
  Value memRef;
  /// LoopInfo of target forOp.
  PackingLoopInfo *loop;
  /// Permutation vector, if it has a value, the packed buffer can be rearranged
  /// so its data layout correlates better with the current loop order.
  std::optional<SmallVector<size_t>> permutationOrder;

  PackingAttributes(Value memRef, PackingLoopInfo &loop)
      : memRef(memRef), loop(&loop) {}

  ArrayRef<int64_t> getTileShape() const {
    return this->loop->getMemRefTileShape(this->memRef);
  }

  std::optional<bool> isContiguous() const {
    auto tileShape = this->getTileShape();
    if (tileShape.empty())
      return std::nullopt;

    auto memRefShape = this->memRef.getType().cast<MemRefType>().getShape();
    return isTileShapeContiguous(memRefShape, tileShape);
  }

  std::unique_ptr<MemRefRegion> &getReadRegion() const {
    return this->loop->getMemRefReadRegion(this->memRef);
  }

  std::unique_ptr<MemRefRegion> &getWriteRegion() const {
    return this->loop->getMemRefWriteRegion(this->memRef);
  }

  void setPermutationOrder() {
    auto memRefType = this->memRef.getType().cast<MemRefType>();
    auto rank = memRefType.getRank();

    // Give up on non-trivial layout map.
    if (!memRefType.getLayout().isIdentity()) {
      this->permutationOrder = std::nullopt;
      return;
    }

    this->loop->forOp.walk([&](Operation *op) {
      // Get stores and loads related affected by this packing.
      if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        if (this->memRef != loadOp.getMemRef())
          return WalkResult::advance();
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        if (this->memRef != storeOp.getMemRef())
          return WalkResult::advance();
      } else {
        return WalkResult::advance();
      }

      // Get access map of load/store.
      MemRefAccess access(op);
      AffineValueMap map;
      access.getAccessMap(&map);

      // Stores the depth of the forOp that owns an IV used as an operand
      // for each result of the access map. If there are multiple
      // IVs used as operands to a result, save the depth of the forOp
      // that is the deepest
      SmallVector<uint32_t> indexLoopIVDepth;
      indexLoopIVDepth.reserve(map.getNumResults());

      // For each result of a map expression.
      for (unsigned int resultIdx = 0; resultIdx < map.getNumResults();
           resultIdx++) {
        uint32_t maxNestingDepth = 0;
        // For every operand of the map (load/store indices).
        for (auto operand : map.getOperands()) {
          // If resultIdx^th result is a function of a loop IV,
          // check the depth of the IV owner and save the max depth
          if (isAffineForInductionVar(operand) &&
              map.isFunctionOf(resultIdx, operand)) {
            AffineForOp ownerForOp = getForInductionVarOwner(operand);
            maxNestingDepth = std::max(maxNestingDepth, getNestingDepth(ownerForOp));
          }
        }
        // Push back in the order of the results
        indexLoopIVDepth.push_back(maxNestingDepth);
      }

      // Create identity permutation
      SmallVector<size_t> idx;
      idx.resize(rank);
      std::iota(idx.begin(), idx.end(), 0);

      // Sort vector of indices idx based on the elements of indexLoopIVDepth.
      llvm::stable_sort(idx, [&indexLoopIVDepth](size_t i1, size_t i2) {
        return indexLoopIVDepth[i1] < indexLoopIVDepth[i2];
      });

      // Initialize permutation order attribute.
      if (!this->permutationOrder.has_value()) {
        this->permutationOrder = idx;
      } else {
        // If two loads/stores have conflicting permutation orders within this
        // packing, do not permute.
        if (this->permutationOrder != idx) {
          this->permutationOrder = std::nullopt;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // Create identity permutation
    SmallVector<size_t> identity;
    identity.resize(rank);
    std::iota(identity.begin(), identity.end(), 0);

    // Set permutationOrder to std::nullopt if no permutation is needed.
    if (this->permutationOrder == identity)
      this->permutationOrder = std::nullopt;
  }
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

  // Structure storing packing candidates.
  SmallVector<PackingAttributes> packingCandidates;

  // Reuse filter
  // Possible packing candidates: all loop and memref combinations.
  // Only add candidates that show reuse, that is, all accesses to the memRef
  // are invariant to the loop.
  for (auto &loopInfo : loopInfoMap) {
    for (const auto &memRef : memRefs) {
      AffineForOp forOp = loopInfo.first;
      PackingLoopInfo &forOpInfo = loopInfo.second;
      // Skip outermost loop (depth > 1): would pack the entire memRef.
      if (forOpInfo.depth > 1 &&
          isMemRefLoadedInForOp(forOp, memRef) &&
          memRefAccessesInvariantToLoop(forOp, memRef))
        packingCandidates.push_back(PackingAttributes(memRef, forOpInfo));
    }
  }
  if (packingCandidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, no invariant loops found.\n");
    return;
  }

  // Check if packing should be permuted.
  for (auto &packing : packingCandidates)
    packing.setPermutationOrder();

  // Remove packing options that have no read region (only write)
  // or that could not compute memRef read regions.
  // Only write regions are not considered for packing.
  packingCandidates.erase(
      std::remove_if(
          packingCandidates.begin(), packingCandidates.end(),
          [&](PackingAttributes &attr) {
            if (!attr.getReadRegion()) {
              LLVM_DEBUG(dbgs() << "[DEBUG] Candidate removed: Could not compute memRef "
                                   "read regions or has only write region.\n");
              return true;
            }
            return false;
          }),
      packingCandidates.end());
  if (packingCandidates.empty()) {
    LLVM_DEBUG(
        dbgs() << "[DEBUG] Not packing, could not compute memRef regions.\n");
    return;
  }

  // Remove packing candidates that are contiguous and have no permutation.
  packingCandidates.erase(
      std::remove_if(
          packingCandidates.begin(), packingCandidates.end(),
          [&](PackingAttributes &attr) {
            if (!attr.isContiguous().has_value() ||
                (attr.isContiguous().value() &&
                 attr.permutationOrder == std::nullopt)) {
              LLVM_DEBUG(dbgs() << "[DEBUG] Candidate removed: region is contiguous "
                                   "and has no permutation.\n");
              return true;
            }
            return false;
          }),
      packingCandidates.end());

  if (packingCandidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, regions are contiguous and have "
                         "no permutation.\n");
    return;
  }

  // Get total footprint for the outerloop
  std::optional<int64_t> totalFootprint =
      getMemoryFootprintBytes(outerForOp, /*memorySpace=*/0);

  // Set cache threshold.
  uint64_t cacheThresholdSizeInKiB = this->l3CacheSizeInKiB;
  // If the computation fits in one cache, consider only upper levels.
  if (totalFootprint.has_value()) {
    uint64_t totalFootprintValue =
        static_cast<uint64_t>(totalFootprint.value());

    // No need for packing if everything already fits in l1 cache.
    if (totalFootprintValue < this->l1CacheSizeInKiB * 1024) {
      LLVM_DEBUG(
          dbgs() << "[DEBUG] Not packing, everything already fits in L1.\n");
      return;
    }

    // Computation fits in L2.
    if (totalFootprintValue < this->l2CacheSizeInKiB * 1024) {
      cacheThresholdSizeInKiB = this->l1CacheSizeInKiB;
      LLVM_DEBUG(dbgs() << "[DEBUG] Cache threshold set to L1.\n");
      // Computation fits in L3.
    } else if (totalFootprintValue < this->l3CacheSizeInKiB * 1024) {
      cacheThresholdSizeInKiB = this->l2CacheSizeInKiB;
      LLVM_DEBUG(dbgs() << "[DEBUG] Cache threshold set to L2.\n");
      // Computation does not fit in L3.
    } else {
      LLVM_DEBUG(dbgs() << "[DEBUG] Cache threshold set to L3.\n");
    }
  }

  AffineCopyOptions copyOptions = {
      /*generateDMA=*/false, /*slowMemorySpace=*/0,
      /*fastMemorySpace=*/0, /*tagMemorySpace=*/0,
      /*fastMemCapacityBytes=*/cacheThresholdSizeInKiB * 1024};
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
