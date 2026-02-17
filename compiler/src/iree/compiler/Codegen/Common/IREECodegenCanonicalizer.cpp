// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-canonicalizer"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREECODEGENCANONICALIZERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Helper method to check if a `subview` operation is trivially a no-op. This
/// is the case if the all offsets are zero, all strides are 1, and the source
/// shape is same as the size of the subview. In such cases, the subview can
/// be folded into its source.
static bool isTrivialSubViewOp(memref::SubViewOp subviewOp) {
  if (subviewOp.getSourceType().getRank() != subviewOp.getType().getRank()) {
    return false;
  }

  if (!areAllConstantIntValue(subviewOp.getMixedOffsets(), 0) ||
      !areAllConstantIntValue(subviewOp.getMixedStrides(), 1)) {
    return false;
  }

  // Check all size values match the source sizes.
  ArrayRef<int64_t> sourceShape = subviewOp.getSourceType().getShape();
  if (areConstantIntValues(subviewOp.getMixedSizes(), sourceShape)) {
    return true;
  }

  // If the sizes are dynamic, traverse the IR to find a ShapeAwareOpInterface
  // to get the dynamic sizes of the source.
  auto opResult = dyn_cast<OpResult>(subviewOp.getSource());
  while (opResult) {
    Operation *owner = opResult.getOwner();
    if (isa<IREE::Util::ShapeAwareOpInterface>(owner)) {
      break;
    }
    // Only continue traversing operands that don't affect the type
    if (!owner->hasTrait<OpTrait::SameOperandsAndResultType>()) {
      return false;
    }
    // SameOperandsAndResultType guarantees 1 operand.
    opResult = dyn_cast<OpResult>(owner->getOperand(0));
  }
  if (!opResult) {
    return false;
  }
  auto shapeAwareProducer =
      cast<IREE::Util::ShapeAwareOpInterface>(opResult.getOwner());
  if (!shapeAwareProducer) {
    return false;
  }
  SmallVector<Value> sourceDynamicSizes =
      shapeAwareProducer.getResultDynamicDimsFromValue(opResult);
  SmallVector<OpFoldResult> sourceMixedSizes =
      getMixedValues(sourceShape, sourceDynamicSizes, subviewOp.getContext());
  return llvm::equal(sourceMixedSizes, subviewOp.getMixedSizes());
}

/// Returns true if linalgOp is a broadcast. Handles both the named
/// linalg.broadcast op and broadcast-like linalg.generic.
static bool isBroadcastLinalgOp(linalg::LinalgOp linalgOp) {
  if (isa<linalg::BroadcastOp>(linalgOp.getOperation())) {
    return true;
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  return genericOp && linalg::isaBroadcastOpInterface(genericOp).has_value();
}

/// For a broadcast LinalgOp (named or generic), returns the single input and
/// init values. Call only when isBroadcastLinalgOp returned true.
static std::pair<Value, Value> getBroadcastInputAndInit(linalg::LinalgOp op) {
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op.getOperation())) {
    return {broadcastOp.getInput(), broadcastOp.getInit()};
  }
  auto genericOp = cast<linalg::GenericOp>(op.getOperation());
  return {genericOp.getDpsInputOperand(0)->get(),
          genericOp.getDpsInitOperand(0)->get()};
}

/// Fold broadcast(tensor.empty()) -> tensor.empty() with the broadcast result
/// shape. Handles both linalg.broadcast and broadcast-like linalg.generic.
/// Placed in codegen canonicalizer so it runs on dispatch IR where the pattern
/// is applied (Flow canonicalize runs at module level and may not see nested
/// executables).
class FoldBroadcastWithEmptyTensor
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::Base::Base;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isBroadcastLinalgOp(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "not a broadcast op");
    }
    auto [input, init] = getBroadcastInputAndInit(linalgOp);
    if (!input.getDefiningOp<tensor::EmptyOp>()) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "input not defined by tensor.empty");
    }
    SmallVector<OpFoldResult> resultSizes =
        tensor::getMixedSizes(rewriter, linalgOp.getLoc(), init);
    auto resultType = cast<RankedTensorType>(init.getType());
    Value newEmpty = tensor::EmptyOp::create(
        rewriter, linalgOp.getLoc(), resultSizes, resultType.getElementType());
    rewriter.replaceOp(linalgOp.getOperation(), newEmpty);
    return success();
  }
};

/// Canonicalize subview ops that are no-ops, using Util::ShapeAwareOpInterface
/// to check dynamic subviews.
/// TODO(Max191): Fold this pattern into TrivialSubViewOpFolder in llvm-project
/// once there is an appropriate interface for dynamic shape retrieval. The
/// existing interface (ReifyRankedShapedTypeOpInterface) can mutate the IR,
/// so it is not viable for canonicalizers.
class DynamicTrivialSubViewOpFolder final
    : public OpRewritePattern<memref::SubViewOp> {
public:
  using Base::Base;

  LogicalResult matchAndRewrite(memref::SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    if (!isTrivialSubViewOp(subViewOp)) {
      return failure();
    }
    if (subViewOp.getSourceType() == subViewOp.getType()) {
      rewriter.replaceOp(subViewOp, subViewOp.getSource());
      return success();
    }
    rewriter.replaceOpWithNewOp<memref::CastOp>(subViewOp, subViewOp.getType(),
                                                subViewOp.getSource());
    return success();
  }
};

struct IREECodegenCanonicalizerPass final
    : impl::IREECodegenCanonicalizerPassBase<IREECodegenCanonicalizerPass> {
public:
  using Base::Base;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Normal);

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(owningPatterns);
    }
    for (RegisteredOperationName op : context->getRegisteredOperations()) {
      if (op.getStringRef() == memref::CopyOp::getOperationName()) {
        owningPatterns.add<DynamicTrivialSubViewOpFolder>(context);
      }
      op.getCanonicalizationPatterns(owningPatterns, context);
    }
    owningPatterns.add<FoldBroadcastWithEmptyTensor>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace
} // namespace mlir::iree_compiler
