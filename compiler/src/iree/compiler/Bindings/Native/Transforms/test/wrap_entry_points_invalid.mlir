// RUN: iree-opt --pass-pipeline='builtin.module(iree-abi-wrap-entry-points{invocation-model=sync})' --split-input-file --verify-diagnostics %s

// expected-error @+1 {{iree.abi.output on argument 1 refers to result 5 but the function has only 1 results}}
util.func public @outputIndexOutOfRange(
    %arg0: tensor<4xf32>,
    %ret0: !hal.buffer {iree.abi.output = 5 : index}
) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

// expected-error @+1 {{result 0 has multiple iree.abi.output storage arguments; only one is permitted}}
util.func public @outputDuplicateStorage(
    %arg0: tensor<4xf32>,
    %ret0a: !hal.buffer {iree.abi.output = 0 : index},
    %ret0b: !hal.buffer {iree.abi.output = 0 : index}
) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.return %0 : tensor<4xf32>
}
