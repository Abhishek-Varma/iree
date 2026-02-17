// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-canonicalize))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_dynamic_trivial_subview(%size: index) -> memref<?xf32, #hal.descriptor_type<storage_buffer>> {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%size}
  %assume_align = memref.assume_alignment %0, 64 : memref<?xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %assume_align[0] [%size] [1] : memref<?xf32, #hal.descriptor_type<storage_buffer>> to memref<?xf32, #hal.descriptor_type<storage_buffer>>
  return %subview : memref<?xf32, #hal.descriptor_type<storage_buffer>>
}
// CHECK-LABEL: @fold_dynamic_trivial_subview
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
//       CHECK:   %[[ASSUME_ALIGN:.+]] = memref.assume_alignment %[[SUBSPAN]]
//   CHECK-NOT:   memref.subview
//       CHECK:   return %[[ASSUME_ALIGN]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @no_fold_dynamic_real_subview(%size: index, %slice_size: index) -> memref<?xf32, #hal.descriptor_type<storage_buffer>> {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%size}
  %assume_align = memref.assume_alignment %0, 64 : memref<?xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %assume_align[0] [%slice_size] [1] : memref<?xf32, #hal.descriptor_type<storage_buffer>> to memref<?xf32, #hal.descriptor_type<storage_buffer>>
  return %subview : memref<?xf32, #hal.descriptor_type<storage_buffer>>
}
// CHECK-LABEL: @no_fold_dynamic_real_subview
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:   return %[[SUBVIEW]]

// -----

// Test that broadcast(tensor.empty()) is folded to tensor.empty() with the
// broadcast result shape (named linalg.broadcast).
func.func @fold_broadcast_with_empty_tensor() -> tensor<16x64xf32> {
  %input_empty = tensor.empty() : tensor<16xf32>
  %init = tensor.empty() : tensor<16x64xf32>
  %bcast = linalg.broadcast ins(%input_empty : tensor<16xf32>) outs(%init : tensor<16x64xf32>) dimensions = [1]
  return %bcast : tensor<16x64xf32>
}
// CHECK-LABEL: func.func @fold_broadcast_with_empty_tensor
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<16x64xf32>
//       CHECK:   return %[[EMPTY]]

// -----

// Test that broadcast-like linalg.generic(tensor.empty()) is folded to
// tensor.empty() with the broadcast result shape.
func.func @fold_broadcast_generic_with_empty_tensor() -> tensor<16x64xf32> {
  %input_empty = tensor.empty() : tensor<16xf32>
  %init = tensor.empty() : tensor<16x64xf32>
  %bcast = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input_empty : tensor<16xf32>) outs(%init : tensor<16x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<16x64xf32>
  return %bcast : tensor<16x64xf32>
}
// CHECK-LABEL: func.func @fold_broadcast_generic_with_empty_tensor
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<16x64xf32>
//       CHECK:   return %[[EMPTY]]
