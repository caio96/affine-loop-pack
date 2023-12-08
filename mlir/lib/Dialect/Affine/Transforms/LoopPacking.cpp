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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

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
using llvm::SmallSet;

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

/// True if a loop is a parent of an operation.
static bool isLoopParentOfOp(AffineForOp forOp, Operation *op) {
  Operation *currOp = op;

  while ((currOp = currOp->getParentOp())) {
    if (auto currFor = dyn_cast<AffineForOp>(currOp)) {
      if (currFor == forOp)
        return true;
    }
  }
  return false;
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

/// Get number of entries needed (in a TLB for example) to fit a tile of \p
/// tileShape with elements of type \p memRefType in a memRef of shape \p
/// memRefShape, given that each entry can fit \p entrySizeInB Bytes (for a TLB,
/// each entry can address \p entrySizeInB Bytes).
static std::optional<uint64_t> getEntriesNeeded(MemRefType memRefType,
                                                ArrayRef<int64_t> memRefShape,
                                                ArrayRef<int64_t> tileShape,
                                                uint64_t entrySizeInB) {
  assert(memRefShape.size() == tileShape.size());

  auto typeSizeBytes = getMemRefIntOrFloatEltSizeInBytes(memRefType);
  if (!typeSizeBytes)
    return std::nullopt;

  // Number of elements addressed in one (TLB) entry.
  uint64_t elementsInEntry = floorDiv(entrySizeInB, typeSizeBytes.value());

  // Find contiguous portion of the tile,
  // the innermost dimension is always included.
  int64_t contiguousDim = memRefShape.size() - 1;
  while (contiguousDim >= 1 &&
         tileShape[contiguousDim - 1] == memRefShape[contiguousDim - 1]) {
    --contiguousDim;
  }

  // Count elements in contiguous portion.
  uint64_t contiguousElems = 1;
  for (size_t idx = contiguousDim; idx < memRefShape.size(); ++idx)
    contiguousElems *= tileShape[idx];

  // Get number of pages needed to address contiguous portion of the tile.
  uint64_t entriesNeeded = ceilDiv(contiguousElems, elementsInEntry);

  // Now for the dimension where the tile in not contiguous
  int64_t remainingDims = contiguousDim - 1;

  // Offset between the two elements obtained by incrementing an index
  // at dimension remainingDim of the tile. For example, in a tensor
  // A[4][4][4] with a tile A'[1][2][4], only the innermost dimension is
  // contiguous and at this point remainingDims equals 1 and offset is equal to
  // 4, which is the offset between A'[0][0][0] and A'[0][1][0].
  uint64_t offset = 1;
  for (size_t idx = remainingDims + 1; idx < memRefShape.size(); idx++)
    offset *= memRefShape[idx];

  while (remainingDims >= 0) {
    // Check if a single page can address multiple contiguous portions (first
    // while iteration) of the tile at once, e.g. if 512 elements fit in an
    // entry, memref has dimensions 256x256, and tile 6x5, 3 entries are needed
    // to fit the tile as each entry can fit 2 full rows of the memref.
    uint64_t entriesOverflow =
        ceilDiv(tileShape[remainingDims] * offset, elementsInEntry);

    // Otherwise multiply the number entries needed for the contiguous portion
    // (first while iteration) and multiply by the size of the tile
    // dimension. 6 in the previous example.
    uint64_t entriesNoOverflow = entriesNeeded * tileShape[remainingDims];

    // Get the smallest one
    entriesNeeded = std::min(entriesOverflow, entriesNoOverflow);

    // Update offset and go to the next dim
    offset *= memRefShape[remainingDims];
    remainingDims--;
  }

  return entriesNeeded;
}

/// Applies the packing transformation.
/// An inital copy loop is created for the readRegion. If a copy loop is
/// inserted for a tensor that is written to, i.e., has a writeRegion,
/// then there is a subsequent 'uncopying' loop to update the original
/// values in memory. The uses of the original memref is the target
/// forOp are substituted by the packed memref. Also applies the packing
/// permutation if there is one.
static LogicalResult generatePackings(
    Value memRef, Operation *forOp, std::unique_ptr<MemRefRegion> &readRegion,
    std::unique_ptr<MemRefRegion> &writeRegion,
    DenseSet<Operation *> &copyNests, AffineCopyOptions &copyOptions,
    ArrayRef<size_t> permutationOrder) {

  DenseMap<Value, Value> fastBufferMap;

  Block::iterator begin = Block::iterator(forOp);
  Block::iterator end = std::next(Block::iterator(forOp));
  Block *block = begin->getBlock();

  LogicalResult ret = success();

  auto processRegions = [&](const std::unique_ptr<MemRefRegion> &region) {
    if (!region)
      return;

    Block::iterator copyInPlacementStart, copyOutPlacementStart;
    Block *copyPlacementBlock;

    copyInPlacementStart = begin;
    copyOutPlacementStart = end;
    copyPlacementBlock = block;

    uint64_t sizeInBytes;
    Block::iterator nBegin, nEnd;
    LogicalResult iRet = generateCopy(
        *region, block, begin, end, copyPlacementBlock, copyInPlacementStart,
        copyOutPlacementStart, copyOptions, fastBufferMap, copyNests,
        &sizeInBytes, &nBegin, &nEnd, permutationOrder);
    if (succeeded(iRet)) {
      // begin/end could have been invalidated, and need update.
      begin = nBegin;
      end = nEnd;
    } else {
      ret = failure();
    }
  };

  processRegions(readRegion);
  processRegions(writeRegion);

  return ret;
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
  /// Map from memref to its footprint in this forOp
  DenseMap<Value, std::optional<int64_t>> memRefFootprintMap;
  /// Map from memref to its footprint in one iteration of this forOp
  DenseMap<Value, std::optional<int64_t>> memRefSingleIterationFootprintMap;

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

  /// Returns footprint of a memref in this loop if available.
  /// The result is computed only when requested and then stored.
  std::optional<int64_t> getMemRefFootprint(Value memRef) {
    if (this->memRefFootprintMap.count(memRef) == 0)
      this->memRefFootprintMap[memRef] =
          getMemoryFootprintBytes(this->forOp, /*memorySpace*/ 0, memRef);

    return this->memRefFootprintMap[memRef];
  }

  /// Returns footprint of a memref in one iteration of this loop if available.
  /// The result is computed only when requested and then stored.
  std::optional<int64_t> getMemRefSingleIterationFootprint(Value memRef) {
    if (this->memRefSingleIterationFootprintMap.count(memRef) == 0) {
      // Get footprint of only the body of the loop
      auto *forBodyBlock = this->forOp.getBody();

      this->memRefSingleIterationFootprintMap[memRef] = getMemoryFootprintBytes(
          *forBodyBlock, forBodyBlock->begin(), forBodyBlock->end(),
          /*memorySpace*/ 0, memRef);
    }

    return this->memRefSingleIterationFootprintMap[memRef];
  }

  void dump() {
    LLVM_DEBUG(dbgs() << "    Affine ForOp Information:\n");
    LLVM_DEBUG(dbgs() << "        LoopIV location: "
                      << this->forOp.getInductionVar().getLoc() << "\n");
    LLVM_DEBUG(dbgs() << "        Upper map: " << this->forOp.getUpperBoundMap()
                      << "\n");
    LLVM_DEBUG(dbgs() << "        Lower map: " << this->forOp.getLowerBoundMap()
                      << "\n");
    LLVM_DEBUG(dbgs() << "        Depth: " << this->depth << "\n");
    if (this->tripCount.has_value()) {
      LLVM_DEBUG(dbgs() << "        TripCount: " << this->tripCount.value()
                        << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "        TripCount: unknown\n");
    }
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
  /// Footprint required so that the packed buffer is not evicted while reused.
  std::optional<uint64_t> residencyFootprint;
  /// Number of TLB entries not required anymore after packing this candidate.
  uint64_t tlbImprovement = 0;
  /// True if permutation improves the indexing of an innermost loop IV.
  bool innermostLoopIVPermutation = false;
  /// Ratio that approximates the benefit/cost of packing this candidate.
  std::optional<double> gainCostRatio;

  PackingAttributes(Value memRef, PackingLoopInfo &loop)
      : memRef(memRef), loop(&loop) {}

  ArrayRef<int64_t> getTileShape() const {
    return this->loop->getMemRefTileShape(this->memRef);
  }

  // Return the tile shape and applies the packing permutation
  // to it if there is any
  SmallVector<int64_t> getPackedShape() const {
    auto tileShape = this->loop->getMemRefTileShape(this->memRef);
    SmallVector<int64_t> packedShape{tileShape.begin(), tileShape.end()};
    if (this->permutationOrder.has_value())
      permuteWithIndexVector(packedShape, this->permutationOrder.value());
    return packedShape;
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

  std::optional<int64_t> getFootprint() const {
    return this->loop->getMemRefFootprint(this->memRef);
  }

  // Order is based on gain/cost ratio of packings and loop depth.
  bool operator<(const PackingAttributes &other) const {
    if (this->gainCostRatio.has_value() && other.gainCostRatio.has_value()) {
      if (this->gainCostRatio.value() != other.gainCostRatio.value())
        return this->gainCostRatio.value() < other.gainCostRatio.value();
    } else if (this->gainCostRatio.has_value() &&
               !other.gainCostRatio.has_value()) {
      return false;
    } else if (!this->gainCostRatio.has_value() &&
               other.gainCostRatio.has_value()) {
      return true;
    }
    // If the gain/cost order fails or ties, order packing based
    // on loop depth. Lower depth is preferred because it means the
    // packing has a larger target region.
    return this->loop->depth > other.loop->depth;
  }

  bool operator>(const PackingAttributes &other) const { return other < *this; }

  // Returns true if this packing is redundant with another
  // packing, meaning that they both pack the same memRef
  // the their target loops overlap.
  bool isRedundantWith(PackingAttributes &other) {
    if (this->memRef == other.memRef) {
      if (isLoopParentOfOp(other.loop->forOp, this->loop->forOp) ||
          isLoopParentOfOp(this->loop->forOp, other.loop->forOp)) {
        return true;
      }
    }
    return false;
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

  // This function checks if the permutation improves the indexing
  // of an innermost loop IV. However, the elements of the innermost
  // dimension must be bigger than two lines of cache. It stores the result
  // in innermostLoopIVPermutation and also returns the result.
  bool setInnermostLoopIVPermutation(uint64_t cacheLineSizeInB) {
    this->innermostLoopIVPermutation = false;

    // Check if there is permutation.
    if (!this->permutationOrder.has_value())
      return this->innermostLoopIVPermutation;

    auto tileShape = this->getTileShape();
    if (tileShape.empty()) {
      return this->innermostLoopIVPermutation;
    }

    this->loop->forOp.walk([&](Operation *op) {
      // Get stores and loads related affected by this packing.
      if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        if (this->memRef != loadOp.getMemRef())
          return WalkResult::advance();
        // Give up on non-trivial layout map.
        if (!loadOp.getMemRefType().getLayout().isIdentity()) {
          this->permutationOrder = std::nullopt;
          return WalkResult::interrupt();
        }
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        if (this->memRef != storeOp.getMemRef())
          return WalkResult::advance();
        // Give up on non-trivial layout map.
        if (!storeOp.getMemRefType().getLayout().isIdentity()) {
          this->permutationOrder = std::nullopt;
          return WalkResult::interrupt();
        }
      } else {
        return WalkResult::advance();
      }

      // Get access map of load/store.
      MemRefAccess access(op);
      AffineValueMap map;
      access.getAccessMap(&map);

      // Check if an operand that uses the IV of an innermost loop
      // is permuted to an to an inner dimension
      for (unsigned int resultIdx = 0; resultIdx < map.getNumResults();
           resultIdx++) {
        for (auto operand : map.getOperands()) {
          if (isAffineForInductionVar(operand) &&
              map.isFunctionOf(resultIdx, operand)) {
            AffineForOp ownerForOp = getForInductionVarOwner(operand);
            if (isInnermostAffineForOp(ownerForOp) &&
                this->permutationOrder.value()[resultIdx] != resultIdx) {
              this->innermostLoopIVPermutation = true;
              return WalkResult::interrupt();
            }
          }
        }
      }

      return WalkResult::advance();
    });

    // Verify that the innermost dimension is at least two cache lines long.
    // Otherwise, ignore innermost loop permutation.
    if (this->innermostLoopIVPermutation) {
      auto packedShape = this->getPackedShape();
      auto innermostElements = packedShape.back();
      auto memRefType = this->memRef.getType().cast<MemRefType>();

      auto typeSizeBytes = getMemRefIntOrFloatEltSizeInBytes(memRefType);
      if (!typeSizeBytes.has_value()){
        this->innermostLoopIVPermutation = false;
        return this->innermostLoopIVPermutation;
      }

      if (innermostElements * typeSizeBytes.value() <
          2 * static_cast<int64_t>(cacheLineSizeInB))
        this->innermostLoopIVPermutation = false;
    }

    return this->innermostLoopIVPermutation;
  }

  /// Residency footprint is an approximation of space in cache necessary so the
  /// packing remains in cache. The packing is reused at every iteration of the
  /// invariant forOp. It will remain in cache if there is enough space for the
  /// packing itself and for the other memrefs. Therefore residency footprint is
  /// calculated with:
  ///   - footprint of the packing
  ///   - 2 * footprint of other memrefs used in one iteration of the invariant
  ///   loop
  /// Twice the footprint of other memrefs so that the packing is not evicted in
  /// an LRU policy. If there are more children loops or if operations, this is
  /// an over approximation. This function computes the residencyFootprint and
  /// returns true, but if the residencyFootprint is not computable, it returns
  /// false
  bool setResidencyFootprint() {
    if (!this->getFootprint().has_value() ||
        this->getFootprint().value() == 0) {
      this->residencyFootprint = std::nullopt;
      return false;
    }

    // Get all memrefs in the target loop
    SetVector<Value> memRefs;
    getMemRefsInForOp(this->loop->forOp, memRefs, /*minRank=*/1);

    uint64_t otherMemrefsFootprint = 0;

    for (const auto &memRef : memRefs) {
      if (this->memRef == memRef)
        continue;

      auto footprint = this->loop->getMemRefSingleIterationFootprint(memRef);
      if (!footprint.has_value()) {
        this->residencyFootprint = std::nullopt;
        return false;
      }

      otherMemrefsFootprint += 2 * footprint.value();
    }

    this->residencyFootprint =
        this->getFootprint().value() + otherMemrefsFootprint;

    return true;
  }

  /// Approximates TLB usage with and without packing this tile in forOp.
  /// If this computation would not fit L1D TLB before packing
  /// and it fits after packing, return the number of tlb entries packing saves.
  /// Returns 0 otherwise.
  /// PackedTiles defines a list of candidates that should be considered packed.
  /// For them, the packed shape is considered instead of the original memRef
  /// shape.
  uint64_t improvesTLBUsage(
      AffineForOp forOp, uint64_t tlbPageSizeInKiB, uint64_t tlbEntries,
      SmallMapVector<AffineForOp, PackingLoopInfo, 4> &loopInfoMap,
      ArrayRef<PackingAttributes *> packedCandidates = std::nullopt) {
    // forOp must be either the target loop or a child loop of it
    assert(isLoopParentOfOp(this->loop->forOp, forOp) ||
           this->loop->forOp == forOp);

    // Get all memRefs in forOp.
    SetVector<Value> memRefs;
    getMemRefsInForOp(forOp, memRefs, /*minRank=*/1);

    // The target memRef may not be used in forOp
    if (!memRefs.contains(this->memRef))
      return 0;

    // For every memRef used in this forOp, other than the target memRef,
    // packing this candidate does not change the number of TLB entries
    // that they use. Still, they are computed to check the L1 dTLB threshold
    uint64_t entries = 0;
    for (const auto &memRef : memRefs) {
      if (memRef == this->memRef)
        continue;

      // Get memRef type
      auto memRefType = memRef.getType().cast<MemRefType>();
      // Get tile shape of memRef in forOp.
      ArrayRef<int64_t> tileShape =
          loopInfoMap[forOp].getMemRefTileShape(memRef);
      if (tileShape.empty())
        return 0;

      // Check if the memRef is already packed
      PackingAttributes *packed = nullptr;
      for (auto *packing : packedCandidates) {
        if (packing->memRef == memRef &&
            (packing->loop->forOp == forOp ||
             isLoopParentOfOp(packing->loop->forOp, forOp))) {
          packed = packing;
          break;
        }
      }

      // If the memRef is not packed, add the entries needed
      // for its tile in this loop, given the original memRefShape
      if (!packed) {
        ArrayRef<int64_t> memRefShape = memRefType.getShape();
        auto memRefEntries = getEntriesNeeded(
            memRefType, memRefShape, tileShape, 1024 * tlbPageSizeInKiB);
        if (!memRefEntries.has_value())
          return 0;
        entries += memRefEntries.value();

        // If packed, add the entries needed
        // for its tile in this loop, given the packing shape
      } else {
        // Get packed shape (already permuted if there is a permutation)
        auto packedShape = packed->getPackedShape();
        if (packedShape.empty())
          return 0;

        // Get tile shape and permuted it if there is a permutation
        SmallVector<int64_t> updatedTileShape{tileShape.begin(),
                                              tileShape.end()};
        if (packed->permutationOrder.has_value())
          permuteWithIndexVector(updatedTileShape,
                                 packed->permutationOrder.value());

        auto memRefEntries = getEntriesNeeded(
            memRefType, packedShape, updatedTileShape, 1024 * tlbPageSizeInKiB);
        if (!memRefEntries)
          return 0;
        entries += memRefEntries.value();
      }
    }

    uint64_t entriesPacking = entries;
    uint64_t entriesNoPacking = entries;

    // Now consider the memref of this candidate,
    // which will use a different number of entries
    // if it is packed of not

    // Get memRef type and shape
    auto memRefType = this->memRef.getType().cast<MemRefType>();
    ArrayRef<int64_t> memRefShape = memRefType.getShape();
    // Get tile shape of memRef in forOp.
    ArrayRef<int64_t> tileShape =
        loopInfoMap[forOp].getMemRefTileShape(this->memRef);
    if (tileShape.empty())
      return 0;
    // Packed shape is the tile shape of the candidate
    auto packedShape = this->getPackedShape();
    if (packedShape.empty())
      return 0;
    // Get tile shape and permuted it if there is a permutation
    SmallVector<int64_t> updatedTileShape{tileShape.begin(), tileShape.end()};
    if (this->permutationOrder.has_value())
      permuteWithIndexVector(updatedTileShape, this->permutationOrder.value());

    // Compute entries packing this candidate
    std::optional<uint64_t> memRefEntries = getEntriesNeeded(
        memRefType, packedShape, updatedTileShape, 1024 * tlbPageSizeInKiB);
    if (!memRefEntries)
      return 0;
    entriesPacking += memRefEntries.value();

    // Compute entries not packing this candidate
    memRefEntries = getEntriesNeeded(memRefType, memRefShape, tileShape,
                                     1024 * tlbPageSizeInKiB);
    if (!memRefEntries)
      return 0;
    entriesNoPacking += memRefEntries.value();

    // If before packing this loop did not fit the tlb, and
    // after packing it fits, return improvement
    if (entriesNoPacking > tlbEntries && entriesPacking <= tlbEntries)
      return entriesNoPacking - entriesPacking;

    // Otherwise return 0
    return 0;
  }

  // This function multiplies the tlb improvement on a loop by the number of
  // times the loop is run inside the target loop. This is done for the target
  // loop and its inner loops. It returns the tlb improvement.
  uint64_t setTLBImprovement(
      uint64_t l1dtlbPageSizeInKiB, uint64_t l1dtlbEntries,
      SmallMapVector<AffineForOp, PackingLoopInfo, 4> &loopInfoMap,
      ArrayRef<PackingAttributes *> packedTiles = std::nullopt) {

    this->loop->forOp.walk([&](AffineForOp forOp) {
      uint64_t improvement = this->improvesTLBUsage(
          forOp, l1dtlbPageSizeInKiB, l1dtlbEntries, loopInfoMap, packedTiles);

      // Multiply improvement by the number of times this loop is run.
      Operation *currOp = forOp;
      while ((currOp = currOp->getParentOp())) {
        // The target loop does not multiply the improvement
        if (currOp == this->loop->forOp->getParentOp())
          break;

        // If the trip count has no value, the improvement
        // will not be multiplied by the trip count
        if (auto currFor = dyn_cast<AffineForOp>(currOp))
          if (loopInfoMap[currFor].tripCount.has_value())
            improvement *= loopInfoMap[currFor].tripCount.value();
      }

      this->tlbImprovement += improvement;
    });

    return this->tlbImprovement;
  }

  /// Calculates how beneficial it is to apply this packing candidate.
  void setGainCostRatio(uint64_t cacheLineSizeInB) {
    // Get memRef and tile shape.
    auto memRefType = this->memRef.getType().cast<MemRefType>();
    auto tileShape = this->getTileShape();
    if (tileShape.empty())
      return;
    ArrayRef<int64_t> memRefShape = memRefType.getShape();

    // Get size of elements in the tile
    auto typeSizeBytes = getMemRefIntOrFloatEltSizeInBytes(memRefType);
    if (!typeSizeBytes)
      return;

    // Number of elements that fit in one cache line.
    uint64_t elementsInCacheLine =
        floorDiv(cacheLineSizeInB, typeSizeBytes.value());
    // How many cache lines are needed for the tile
    auto cacheLines = getEntriesNeeded(memRefType, memRefShape,
                                       tileShape, cacheLineSizeInB);
    if (!cacheLines)
      return;

    // Cost: number of cache lines needed for packing the tile
    //       (multiplied by two if it needs unpacking)
    uint64_t cost = cacheLines.value() * elementsInCacheLine;
    if (this->getWriteRegion()) {
      cost *= 2;
    }

    // Gain: reuse factor (trip count) * TLB entries saved
    uint64_t gain = this->tlbImprovement;

    assert(cost != 0 && "Cost should not be zero.");
    this->gainCostRatio = static_cast<double>(gain) / static_cast<double>(cost);
  }

  void dump() const {
    LLVM_DEBUG(dbgs() << "[Packing Candidate]\n");

    if (this->id != -1)
      LLVM_DEBUG(dbgs() << "    Id: " << this->id << "\n");

    LLVM_DEBUG(dbgs() << "    MemRef: " << this->memRef << "\n");

    this->loop->dump();

    LLVM_DEBUG(dbgs() << "    Tile shape: ");
    if (this->getTileShape().empty()) {
      LLVM_DEBUG(dbgs() << " unknown ");
    } else {
      for (auto i : this->getTileShape())
        LLVM_DEBUG(dbgs() << i << " ");
    }
    LLVM_DEBUG(dbgs() << "\n");

    if (this->permutationOrder.has_value()) {
      LLVM_DEBUG(dbgs() << "    Permutation: ");
      for (auto i : this->permutationOrder.value())
        LLVM_DEBUG(dbgs() << i << " ");
      LLVM_DEBUG(dbgs() << "\n");

      LLVM_DEBUG(dbgs() << "    Packed shape: ");
      if (this->getPackedShape().empty()) {
        LLVM_DEBUG(dbgs() << " unknown ");
      } else {
        for (auto i : this->getPackedShape())
          LLVM_DEBUG(dbgs() << i << " ");
      }
      LLVM_DEBUG(dbgs() << "\n");

      LLVM_DEBUG(dbgs() << "    Permutes innermost loop IV: "
                        << this->innermostLoopIVPermutation << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "    Permutation: None");
    }

    if (this->isContiguous().has_value()) {
      LLVM_DEBUG(dbgs() << "    Contiguous: " << this->isContiguous().value()
                        << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "    Contiguous: unknown\n");
    }

    if (this->getFootprint().has_value()) {
      LLVM_DEBUG(dbgs() << "    Footprint: " << this->getFootprint().value()
                        << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "    Footprint: unknown\n");
    }

    if (this->residencyFootprint.has_value()) {
      LLVM_DEBUG(dbgs() << "    Residency footprint: "
                        << this->residencyFootprint.value() << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "    Residency footprint: unknown\n");
    }

    LLVM_DEBUG(dbgs() << "    TLB improvement: " << this->tlbImprovement
                      << "\n");

    if (this->gainCostRatio.has_value()) {
      LLVM_DEBUG(dbgs() << "    GainCostRatio: "
                        << llvm::format("%.8f", this->gainCostRatio.value())
                        << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "    GainCostRatio: unknown\n");
    }
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
              LLVM_DEBUG(dbgs()
                         << "[DEBUG] Candidate removed: region is contiguous "
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

  // Remove packing options that could not compute residency footprints
  // or that have a residency footprints bigger than cache threshold
  packingCandidates.erase(
      std::remove_if(
          packingCandidates.begin(), packingCandidates.end(),
          [&](PackingAttributes &attr) {
            if (!attr.setResidencyFootprint()) {
              LLVM_DEBUG(dbgs()
                         << "[DEBUG] Candidate removed: could not compute "
                            "footprint or residency footprint.\n");
              return true;
            }
            if (attr.residencyFootprint.value() >
                cacheThresholdSizeInKiB * 1024) {
              LLVM_DEBUG(dbgs()
                         << "[DEBUG] Candidate removed: residency footprint "
                            "bigger than target level of cache.\n");
              return true;
            }
            return false;
          }),
      packingCandidates.end());
  if (packingCandidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, could not compute residency "
                         "footprints, or residency footprint is too big.\n");
    return;
  }

  // Remove packing options that do not improve TLB use.
  packingCandidates.erase(
      std::remove_if(
          packingCandidates.begin(), packingCandidates.end(),
          [&](PackingAttributes &attr) {
            if (attr.setTLBImprovement(this->l1dtlbPageSizeInKiB,
                                       this->l1dtlbEntries, loopInfoMap) == 0 &&
                !attr.setInnermostLoopIVPermutation(this->cacheLineSizeInB)) {
              LLVM_DEBUG(dbgs()
                         << "[DEBUG] Candidate removed: does not improve TLB "
                            "usage and has no innermost loop permutation.\n");
              return true;
            }
            return false;
          }),
      packingCandidates.end());
  if (packingCandidates.empty()) {
    LLVM_DEBUG(dbgs() << "[DEBUG] Not packing, does not improve tlb usage "
                         "and have no innermost permutation\n");
    return;
  }

  // Set gain/cost ratio of each candidate.
  for (auto &packing : packingCandidates) {
    packing.setGainCostRatio(this->cacheLineSizeInB);
  }

  // Sort the packing options from best to worst options
  // according to their gain/cost and loop depth.
  llvm::stable_sort(packingCandidates, std::greater<PackingAttributes>());
  // Give each candidates an ID
  int counter = 1;
  for (auto &packing : packingCandidates) {
    packing.id = counter++;
  }

  // Store final packing selection.
  SmallVector<PackingAttributes *> packingSelection;

  // User selected packings
  if (!this->packingOptions.empty()) {

    // Print candidates if the option is -1
    if (this->packingOptions.size() == 1 && this->packingOptions[0] == -1) {
      LLVM_DEBUG(dbgs() << "---------------------------------------------"
                        << "\n");
      for (auto &packing : packingCandidates)
        packing.dump();
      LLVM_DEBUG(dbgs() << "---------------------------------------------"
                        << "\n");
      return;
    }

    // Verify that candidates selected exist
    for (auto opt : this->packingOptions) {
      if (opt <= 0 || opt >= counter) {
        LLVM_DEBUG(
            dbgs()
            << "[ERROR] Invalid packing candidates was manually selected: "
            << opt << "\n");
        return signalPassFailure();
      }
    }

    // Put user-selected packing in a set to avoid duplicates.
    SmallSet<int32_t, 4> options;
    for (auto opt : this->packingOptions)
      options.insert(opt);

    // Add user-selected packings
    for (auto &packing : packingCandidates) {
      if (options.contains(packing.id)) {
        // Check if selecting this packing would be redundant.
        for (auto *selected : packingSelection) {
          if (selected->isRedundantWith(packing)) {
            LLVM_DEBUG(
                dbgs()
                << "[ERROR] The same MemRef was selected to be packed twice "
                   "in overlapping loops (options "
                << packing.id << " and " << selected->id << ")" << "\n");
            return signalPassFailure();
          }
        }
        packingSelection.push_back(&packing);
      }
    }
  // Greedy approach to select packings:
  // Tries to select all remaining candidates starting from
  // best to worst. Does not pack if a candidate is redundant
  // to another that was already selected.
  } else {
    for (auto &packing : packingCandidates) {
      // Recalculate TLB improvement considering packings
      // that were already selected
      if (!packingSelection.empty()) {
        packing.setTLBImprovement(this->l1dtlbPageSizeInKiB, this->l1dtlbEntries,
                                  loopInfoMap, packingSelection);
        // Do not pack if no improvement and no permutation on innermost loop IV.
        if (packing.tlbImprovement == 0 && !packing.innermostLoopIVPermutation)
          continue;
      }

      // Should only pack if selected packings are not redundant.
      bool shouldPack = true;
      for (auto *selected : packingSelection) {
        if (selected->isRedundantWith(packing)) {
          shouldPack = false;
          break;
        }
      }
      if (shouldPack)
        packingSelection.push_back(&packing);
    }
  }

  // Print selected packings
  LLVM_DEBUG(dbgs() << "[Packing Selection]--------------------------\n");
  for (const auto *packing : packingSelection)
    packing->dump();
  LLVM_DEBUG(dbgs() << "---------------------------------------------\n");

  // For each memref, the packs are generated at the found hoisting locations.
  for (const auto *packing : packingSelection) {
    ArrayRef<size_t> permutation;
    if (packing->permutationOrder.has_value())
      permutation = packing->permutationOrder.value();

    if (failed(generatePackings(packing->memRef, packing->loop->forOp,
                                packing->getReadRegion(),
                                packing->getWriteRegion(), copyNests,
                                copyOptions, permutation))) {
      LLVM_DEBUG(dbgs() << "[DEBUG] Failed to generate packing"
                        << packing->id
                        << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "[DEBUG] Succeeded generating packing "
                        << packing->id
                        << "\n");
    }
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
