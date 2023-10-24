// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_INTERFACES
#define IREE_DIALECT_LINALGEXT_INTERFACES

include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtBase.td"

// The interface is a subset of LinalgStructuredInterface.
def LinalgExtInterface : OpInterface<"LinalgExtOp"> {
  let methods = [];

  let extraClassDeclaration = [{
    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible.
    LogicalResult reifyResultShapes(OpBuilder &b,
        mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = (*this)->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyLinalgExtOpInterface($_op); }];
}

def AggregatedOpInterface : OpInterface<"AggregatedOpInterface"> {
  let description = [{
    Interface for decomposing aggregated operations into a sequence of simpler
    ops.
  }];
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Method to decompose the operation into simpler operations.

          On success, this method returns one `Value` per result in the
          original operation.
          The order of the returned values must match the order of the
          original values.
          In other words, the returned vector can be used directly with
          `RewriterBase::replaceOp(this, returnedValues)`.
        }],
        /*retType=*/"FailureOr<SmallVector<Value>>",
        /*methodName=*/"decomposeOperation",
        /*args=*/(ins
            "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return {};
        }]
      >
  ];
}
#endif  // IREE_DIALECT_LINALGEXT_INTERFACES
