// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-check-resource-usage))" %s --verify-diagnostics -split-input-file | FileCheck %s

module {
  // expected-error @+1 {{uses 274432 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc() {
    memref.alloc() : memref<274432xi8, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check that we don't choke on memrefs of index.
// CHECK-LABEL: func.func @shared_mem_alloc_index()
module {
  func.func @shared_mem_alloc_index() {
    memref.alloc() : memref<64xindex, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check that memrefs of index return a valid size.
module {
  // expected-error @+1 {{uses 144984 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc_index_too_big() {
    memref.alloc() : memref<18123xindex, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Regression test for https://github.com/iree-org/iree/issues/24149. Before the
// byte accounting was widened to int64_t, an allocation whose bit count
// exceeded int32 range would silently wrap to a small (often negative) number
// and either pass the check spuriously or produce a nonsensical diagnostic
// like "uses -8192 bytes of shared memory". The pass must now emit a clear
// error on the offending alloc instead.
module {
  func.func @shared_mem_alloc_overflows_int64() {
    // expected-error @+1 {{'memref.alloc' op shared memory allocation of type 'memref<32x9007199254740991x2x64xf16, #gpu.address_space<workgroup>>' has a size that does not fit in a signed 64-bit integer}}
    memref.alloc() : memref<32x9007199254740991x2x64xf16, #gpu.address_space<workgroup>>
    return
  }
}
