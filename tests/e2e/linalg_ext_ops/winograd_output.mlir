
func.func @winograd_output_nchw() -> tensor<1x1x6x6xf32> {
  %input = util.unfoldable_constant dense<1.0> : tensor<8x8x1x1x1x1xf32>

  %init = tensor.empty() : tensor<1x1x6x6xf32>
  %1 = iree_linalg_ext.winograd.output_transform
       output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
       ins(%input : tensor<8x8x1x1x1x1xf32>)
       outs(%init : tensor<1x1x6x6xf32>) -> tensor<1x1x6x6xf32>
  return %1 : tensor<1x1x6x6xf32>
}
