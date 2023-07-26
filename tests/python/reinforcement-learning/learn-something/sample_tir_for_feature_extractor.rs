@main = primfn(var_p0: handle, var_p1: handle, var_T_dense: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {p0: Buffer(p0_1: Pointer(global float16), float16, [384i64, 768i64], []),
             p1: Buffer(p1_1: Pointer(global float16), float16, [768i64, 768i64], []),
             T_dense: Buffer(T_dense_1: Pointer(global float16), float16, [384i64, 768i64], [])}
  buffer_map = {var_p0: p0, var_p1: p1, var_T_dense: T_dense} {
  attr [blockIdx.y: int64] "pragma_auto_unroll_max_step" = 1024i64;
  attr [blockIdx.y] "pragma_unroll_explicit" = 1i64;
  attr [IterVar(blockIdx.y, [0i64:1i64], "ThreadIndex", "blockIdx.y")] "thread_extent" = 1i64;
  attr [IterVar(blockIdx.x: int64, [0i64:72i64], "ThreadIndex", "blockIdx.x")] "thread_extent" = 72i64;
  attr [IterVar(threadIdx.y: int64, [0i64:8i64], "ThreadIndex", "threadIdx.y")] "thread_extent" = 8i64;
  attr [IterVar(threadIdx.x: int64, [0i64:32i64], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32i64;
  allocate(T_dense_reindex_shared: Pointer(shared float16), float16, [64i64, 64i64]), storage_scope = shared;
  T_dense_reindex_shared_1 = decl_buffer(T_dense_reindex_shared, float16, [64i64, 64i64])
   {
    allocate(T_dense_reindex_shared_wmma.accumulator: Pointer(wmma.accumulator float16), float16, [16i64, 32i64]), storage_scope = wmma.accumulator;
    T_dense_reindex_shared_wmma.accumulator_1 = decl_buffer(T_dense_reindex_shared_wmma.accumulator, float16, [16i64, 32i64])
     {
      for (ax1_0_4_init: int64, 0i64, 2i64) {
        @tir.tvm_fill_fragment(T_dense_reindex_shared_wmma.accumulator, 16, 16, 16, ax1_0_4_init, 0f32, dtype=handle)
      }
      for (ax2_0_0: int64, 0i64, 6i64) {
        allocate(p0_reindex_shared: Pointer(shared float16), float16, [64i64, 136i64]), storage_scope = shared;
        p0_reindex_shared_1 = decl_buffer(p0_reindex_shared, float16, [64i64, 128i64])
        ;
        allocate(p1_reindex_shared: Pointer(shared float16), float16, [64i64, 136i64]), storage_scope = shared;
        p1_reindex_shared_1 = decl_buffer(p1_reindex_shared, float16, [64i64, 128i64])
         {
          for (ax0_ax1_fused_0: int64, 0i64, 4i64) {
            for (ax0_ax1_fused_3: int64, 0i64, 8i64) "vectorized" {
              p0_reindex_shared_1[(((ax0_ax1_fused_0*16i64) + (threadIdx.y*2i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3), 128i64)), floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3), 128i64)] 
              = p0[
                ((((floordiv(blockIdx.x, 12i64)*64i64) + (ax0_ax1_fused_0*16i64)) + (threadIdx.y*2i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3), 128i64)), 
              ((ax2_0_0*128i64) + floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3), 128i64))]
            }
          }
          for (ax0_ax1_fused_0_1: int64, 0i64, 4i64) {
            for (ax0_ax1_fused_3_1: int64, 0i64, 8i64) "vectorized" {
              p1_reindex_shared_1[(((ax0_ax1_fused_0_1*16i64) + (threadIdx.y*2i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3_1), 128i64)), floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3_1), 128i64)] = p1[((((floormod(blockIdx.x, 12i64)*64i64) + (ax0_ax1_fused_0_1*16i64)) + (threadIdx.y*2i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3_1), 128i64)), ((ax2_0_0*128i64) + floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3_1), 128i64))]
            }
          }
          for (ax2_0_1: int64, 0i64, 2i64) {
            allocate(p0_reindex_shared_wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [16i64, 64i64]), storage_scope = wmma.matrix_a;
            p0_reindex_shared_wmma.matrix_a_1 = decl_buffer(p0_reindex_shared_wmma.matrix_a, float16, [16i64, 64i64])
            ;
            allocate(p1_reindex_shared_wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [32i64, 64i64]), storage_scope = wmma.matrix_b;
            p1_reindex_shared_wmma.matrix_b_1 = decl_buffer(p1_reindex_shared_wmma.matrix_b, float16, [32i64, 64i64])
             {
              for (ax1_0: int64, 0i64, 4i64) {
                @tir.tvm_load_matrix_sync(p0_reindex_shared_wmma.matrix_a, 16, 16, 16, ax1_0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), p0_reindex_shared, (((floordiv(threadIdx.y, 2i64)*2176i64) + (ax2_0_1*64i64)) + (ax1_0*16i64)), 2176i64, 1, dtype=handle), 136i64, "row_major", dtype=handle)
              }
              for (ax0_0: int64, 0i64, 2i64) {
                for (ax1_0_1: int64, 0i64, 4i64) {
                  @tir.tvm_load_matrix_sync(p1_reindex_shared_wmma.matrix_b, 16, 16, 16, ((ax0_0*4i64) + ax1_0_1), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), p1_reindex_shared, ((((floormod(threadIdx.y, 2i64)*4352i64) + (ax0_0*2176i64)) + (ax2_0_1*64i64)) + (ax1_0_1*16i64)), 2176i64, 1, dtype=handle), 136i64, "col_major", dtype=handle)
                }
              }
              for (ax2_0_2: int64, 0i64, 4i64) {
                for (ax1_0_4: int64, 0i64, 2i64) {
                  @tir.tvm_mma_sync(T_dense_reindex_shared_wmma.accumulator, ax1_0_4, p0_reindex_shared_wmma.matrix_a, ax2_0_2, p1_reindex_shared_wmma.matrix_b, ((ax1_0_4*4i64) + ax2_0_2), T_dense_reindex_shared_wmma.accumulator, ax1_0_4, dtype=handle)
                }
              }
            }
          }
        }
      }
      for (ax1_0_2: int64, 0i64, 2i64) {
        @tir.tvm_store_matrix_sync(T_dense_reindex_shared_wmma.accumulator, 16, 16, 16, ax1_0_2, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), T_dense_reindex_shared, (((floordiv(threadIdx.y, 2i64)*1024i64) + (floormod(threadIdx.y, 2i64)*32i64)) + (ax1_0_2*16i64)), 1024i64, 2, dtype=handle), 64i64, "row_major", dtype=handle)
      }
    }
    for (ax0_ax1_fused_0_2: int64, 0i64, 8i64) {
      for (ax0_ax1_fused_3_2: int64, 0i64, 2i64) "vectorized" {
        T_dense[(((floordiv(blockIdx.x, 12i64)*64i64) + (ax0_ax1_fused_0_2*8i64)) + threadIdx.y), ((floormod(blockIdx.x, 12i64)*64i64) + ((threadIdx.x*2i64) + ax0_ax1_fused_3_2))] = T_dense_reindex_shared_1[((ax0_ax1_fused_0_2*8i64) + threadIdx.y), ((threadIdx.x*2i64) + ax0_ax1_fused_3_2)]
      }
    }
  }
}
