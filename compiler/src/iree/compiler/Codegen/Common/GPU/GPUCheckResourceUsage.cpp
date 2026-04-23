// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCHECKRESOURCEUSAGEPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
static unsigned getDatalayoutIndexBitwidth(mlir::FunctionOpInterface func) {
  auto mod = func->getParentOfType<ModuleOp>();
  LowerToLLVMOptions options(mod.getContext(), DataLayout(mod));
  return options.getIndexBitwidth();
}

/// Returns the static allocation size in bits for `shapedType`. Dynamic dims
/// are skipped (the caller has already verified the alloc has no dynamic
/// sizes). Returns `failure()` on 64-bit signed overflow.
static FailureOr<int64_t> shapedTypeStaticSizeInBits(
    memref::AllocOp allocOp, ShapedType shapedType,
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
  int64_t allocSize = 1;
  for (int64_t dimSize : shapedType.getShape()) {
    if (ShapedType::isDynamic(dimSize)) {
      continue;
    }
    if (llvm::MulOverflow(allocSize, dimSize, allocSize)) {
      return failure();
    }
  }

  int64_t elementSizeInBits;
  if (auto elementType = dyn_cast<ShapedType>(shapedType.getElementType())) {
    FailureOr<int64_t> nestedSize =
        shapedTypeStaticSizeInBits(allocOp, elementType, getIndexBitwidth);
    if (failed(nestedSize)) {
      return failure();
    }
    elementSizeInBits = *nestedSize;
  } else {
    auto eltTy = shapedType.getElementType();
    if (eltTy.isIndex()) {
      auto func = allocOp->getParentOfType<mlir::FunctionOpInterface>();
      assert(getIndexBitwidth &&
             "getIndexBitwidth should have been set earlier");
      elementSizeInBits = getIndexBitwidth(func);
    } else {
      elementSizeInBits = IREE::Util::getTypeBitWidth(shapedType.getElementType());
    }
  }
  if (llvm::MulOverflow(allocSize, elementSizeInBits, allocSize)) {
    return failure();
  }
  return allocSize;
}

/// Returns success if the total shared memory allocation size is less than the
/// limit.
static LogicalResult checkGPUAllocationSize(
    mlir::FunctionOpInterface funcOp, unsigned limit,
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
  if (funcOp.getFunctionBody().empty()) {
    return success();
  }

  SmallVector<memref::AllocOp> allocOps;
  funcOp.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });
  if (allocOps.empty()) {
    return success();
  }

  int64_t cumSize = 0;
  for (auto allocOp : allocOps) {
    auto allocType = cast<MemRefType>(allocOp.getType());
    if (!hasSharedMemoryAddressSpace(allocType)) {
      continue;
    }

    if (!allocOp.getDynamicSizes().empty()) {
      return allocOp.emitOpError(
          "has unsupported dynamic shared memory allocations");
    }

    FailureOr<int64_t> allocSizeOr =
        shapedTypeStaticSizeInBits(allocOp, allocType, getIndexBitwidth);
    if (failed(allocSizeOr)) {
      return allocOp.emitOpError("shared memory allocation of type ")
             << allocType
             << " has a size that does not fit in a signed 64-bit integer";
    }
    int64_t allocSize = *allocSizeOr;
    if (allocOp.getAlignment()) {
      int64_t alignmentInBits = *allocOp.getAlignment() * 8;
      int64_t numChunks = llvm::divideCeil(allocSize, alignmentInBits);
      if (llvm::MulOverflow(numChunks, alignmentInBits, allocSize)) {
        return allocOp.emitOpError("shared memory allocation of type ")
               << allocType
               << " has a size that does not fit in a signed 64-bit integer "
                  "after alignment padding";
      }
    }
    if (llvm::AddOverflow(cumSize, allocSize / 8, cumSize)) {
      return allocOp.emitOpError(
          "cumulative shared memory allocation size overflows a signed 64-bit "
          "integer");
    }
  }
  if (cumSize > limit) {
    return emitError(funcOp->getLoc())
           << "function '" << funcOp.getName() << "' uses " << cumSize
           << " bytes of shared memory; exceeded the limit of " << limit
           << " bytes";
  }
  return success();
}

class GPUCheckResourceUsagePass final
    : public impl::GPUCheckResourceUsagePassBase<GPUCheckResourceUsagePass> {
public:
  explicit GPUCheckResourceUsagePass(
      std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth)
      : getIndexBitwidth(getIndexBitwidth) {}

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    unsigned limit =
        target ? target.getWgp().getMaxWorkgroupMemoryBytes() : 64 * 1024;
    if (failed(checkGPUAllocationSize(funcOp, limit,
                                      getIndexBitwidth
                                          ? getIndexBitwidth
                                          : getDatalayoutIndexBitwidth))) {
      return signalPassFailure();
    }
  }

private:
  std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGPUCheckResourceUsagePass(
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
  return std::make_unique<GPUCheckResourceUsagePass>(getIndexBitwidth);
}

} // namespace mlir::iree_compiler
