
@main = primfn(var_p0: handle, var_p1: handle, var_T_dense: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {p0: Buffer(p0_1: Pointer(global float16), float16, [384i64, 768i64], []),
             p1: Buffer(p1_1: Pointer(global float16), float16, [768i64, 768i64], []),
             T_dense: Buffer(T_dense_1: Pointer(global float16), float16, [384i64, 768i64], [])}
  buffer_map = {var_p0: p0, var_p1: p1, var_T_dense: T_dense} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (blockIdx.y: int64, 0i64, 1i64) "thread_binding" {
      for (blockIdx.x: int64, 0i64, 144i64) "thread_binding" {
        for (threadIdx.y: int64, 0i64, 2i64) "thread_binding" {
          for (threadIdx.x: int64, 0i64, 32i64) "thread_binding" {
            block([], "") {
              tir.reads([p0[(floordiv(blockIdx.x, 12i64)*32i64):((floordiv(blockIdx.x, 12i64)*32i64) + 32i64), 0i64:768i64], p1[(floormod(blockIdx.x, 12i64)*64i64):((floormod(blockIdx.x, 12i64)*64i64) + 64i64), 0i64:768i64]])
              tir.writes([T_dense[(floordiv(blockIdx.x, 12i64)*32i64):((floordiv(blockIdx.x, 12i64)*32i64) + 32i64), (floormod(blockIdx.x, 12i64)*64i64):((floormod(blockIdx.x, 12i64)*64i64) + 64i64)]])
              // Allocate shared memory
              T_dense_reindex_shared = alloc_buffer(float16[32i64, 64i64])
               {
                block([], "") {
                  tir.reads([p0[(floordiv(blockIdx.x, 12i64)*32i64):((floordiv(blockIdx.x, 12i64)*32i64) + 32i64), 0i64:768i64], p1[(floormod(blockIdx.x, 12i64)*64i64):((floormod(blockIdx.x, 12i64)*64i64) + 64i64), 0i64:768i64]])
                  tir.writes([T_dense_reindex_shared[(threadIdx.y*16i64):((threadIdx.y*16i64) + 16i64), 0i64:64i64]])
                  T_dense_reindex_shared_wmma.accumulator = alloc_buffer(float16[16i64, 64i64])
                   {
                    for (ax1_0_3_init: int64, 0i64, 4i64) {
                      block([], "T_dense_o_init") {
                        tir.reads([])
                        tir.writes([T_dense_reindex_shared_wmma.accumulator[0i64:16i64, (ax1_0_3_init*16i64):((ax1_0_3_init*16i64) + 16i64)]])
                        tir.attrs({"meta_schedule.thread_extent_low_inclusive": 32i64, "meta_schedule.thread_extent_high_inclusive": 1024i64, "warp_execution": 1i64})
                        @tir.tvm_fill_fragment(T_dense_reindex_shared_wmma.accumulator_1: Pointer(wmma.accumulator float16), 16, 16, 16, ax1_0_3_init, 0f32, dtype=handle)
                    }
                    for (ax2_0_0: int64, 0i64, 16i64) {
                      block([], "") {
                        tir.reads([p0[(floordiv(blockIdx.x, 12i64)*32i64):((floordiv(blockIdx.x, 12i64)*32i64) + 32i64), (ax2_0_0*48i64):((ax2_0_0*48i64) + 48i64)], p1[(floormod(blockIdx.x, 12i64)*64i64):((floormod(blockIdx.x, 12i64)*64i64) + 64i64), (ax2_0_0*48i64):((ax2_0_0*48i64) + 48i64)], T_dense_reindex_shared_wmma.accumulator[0i64:16i64, 0i64:64i64]])
                        tir.writes([T_dense_reindex_shared_wmma.accumulator[0i64:16i64, 0i64:64i64]])
                        p0_reindex_shared = alloc_buffer(float16[32i64, 48i64])
                        p1_reindex_shared = alloc_buffer(float16[64i64, 48i64])
                        p0_reindex_shared_wmma.matrix_a = alloc_buffer(float16[16i64, 48i64])
                        p1_reindex_shared_wmma.matrix_b = alloc_buffer(float16[64i64, 48i64])
                         {
                          for (ax0_ax1_fused_0: int64, 0i64, 6i64) {
                            for (ax0_ax1_fused_3: int64, 0i64, 4i64) "vectorized" {
                              block([], "p0_reindex_shared") {
                                tir.reads([p0[((floordiv(blockIdx.x, 12i64)*32i64) + floordiv(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64)), ((ax2_0_0*48i64) + floormod(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64))]])
                                tir.writes([p0_reindex_shared[floordiv(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64), floormod(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64)]])
                                tir.attrs({"buffer_dim_align": [[0, 0, 32, 8]]})
                                p0_reindex_shared[floordiv(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64), floormod(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64)] = p0[((floordiv(blockIdx.x, 12i64)*32i64) + floordiv(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64)), ((ax2_0_0*48i64) + floormod(((((ax0_ax1_fused_0*256i64) + (threadIdx.y*128i64)) + (threadIdx.x*4i64)) + ax0_ax1_fused_3), 48i64))]
                            }
                          }
                          for (ax0_ax1_fused_0_1: int64, 0i64, 24i64) {
                            for (ax0_ax1_fused_3_1: int64, 0i64, 2i64) "vectorized" {
                              block([], "p1_reindex_shared") {
                                tir.reads([p1[((floormod(blockIdx.x, 12i64)*64i64) + floordiv(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64)), ((ax2_0_0*48i64) + floormod(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64))]])
                                tir.writes([p1_reindex_shared[floordiv(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64), floormod(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64)]])
                                tir.attrs({"buffer_dim_align": [[0, 0, 32, 8]]})
                                p1_reindex_shared[floordiv(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64), floormod(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64)] = p1[((floormod(blockIdx.x, 12i64)*64i64) + floordiv(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64)), ((ax2_0_0*48i64) + floormod(((((ax0_ax1_fused_0_1*128i64) + (threadIdx.y*64i64)) + (threadIdx.x*2i64)) + ax0_ax1_fused_3_1), 48i64))]
                            }
                          }
                           {
                            for (ax1_0: int64, 0i64, 3i64) {
                              block([], "p0_reindex_shared_wmma.matrix_a_o") {
                                tir.reads([p0_reindex_shared[(threadIdx.y*16i64):((threadIdx.y*16i64) + 16i64), (ax1_0*16i64):((ax1_0*16i64) + 16i64)]])
                                tir.writes([p0_reindex_shared_wmma.matrix_a[0i64:16i64, (ax1_0*16i64):((ax1_0*16i64) + 16i64)]])
                                @tir.tvm_load_matrix_sync(p0_reindex_shared_wmma.matrix_a_1: Pointer(wmma.matrix_a float16), 16, 16, 16, ax1_0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), p0_reindex_shared_1: Pointer(shared float16), ((threadIdx.y*1152i64) + (ax1_0*16i64)), 1152i64, 1, dtype=handle), 72i64, "row_major", dtype=handle)
                            }
                            for (ax0_0: int64, 0i64, 4i64) {
                              for (ax1_0_1: int64, 0i64, 3i64) {
                                block([], "p1_reindex_shared_wmma.matrix_b_o") {
                                  tir.reads([p1_reindex_shared[(ax0_0*16i64):((ax0_0*16i64) + 16i64), (ax1_0_1*16i64):((ax1_0_1*16i64) + 16i64)]])
                                  tir.writes([p1_reindex_shared_wmma.matrix_b[(ax0_0*16i64):((ax0_0*16i64) + 16i64), (ax1_0_1*16i64):((ax1_0_1*16i64) + 16i64)]])
                                  @tir.tvm_load_matrix_sync(p1_reindex_shared_wmma.matrix_b_1: Pointer(wmma.matrix_b float16), 16, 16, 16, ((ax0_0*3i64) + ax1_0_1), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), p1_reindex_shared_1: Pointer(shared float16), ((ax0_0*1152i64) + (ax1_0_1*16i64)), 1152i64, 1, dtype=handle), 72i64, "col_major", dtype=handle)
                              }
                            }
                            for (ax1_0_3: int64, 0i64, 4i64) {
                              for (ax2_0_2: int64, 0i64, 3i64) {
                                block([], "T_dense_o_update") {
                                  tir.reads([T_dense_reindex_shared_wmma.accumulator[0i64:16i64, (ax1_0_3*16i64):((ax1_0_3*16i64) + 16i64)], p0_reindex_shared_wmma.matrix_a[0i64:16i64, (ax2_0_2*16i64):((ax2_0_2*16i64) + 16i64)], p1_reindex_shared_wmma.matrix_b[(ax1_0_3*16i64):((ax1_0_3*16i64) + 16i64), (ax2_0_2*16i64):((ax2_0_2*16i64) + 16i64)]])
                                  tir.writes([T_dense_reindex_shared_wmma.accumulator[0i64:16i64, (ax1_0_3*16i64):((ax1_0_3*16i64) + 16i64)]])
                                  tir.attrs({"meta_schedule.thread_extent_low_inclusive": 32i64, "meta_schedule.thread_extent_high_inclusive": 1024i64, "warp_execution": 1i64})
                                  @tir.tvm_mma_sync(T_dense_reindex_shared_wmma.accumulator_1, ax1_0_3, p0_reindex_shared_wmma.matrix_a_1, ax2_0_2, p1_reindex_shared_wmma.matrix_b_1, ((ax1_0_3*3i64) + ax2_0_2), T_dense_reindex_shared_wmma.accumulator_1, ax1_0_3, dtype=handle)
                              }
                            }
                          }
                        }
                    }
                    for (ax1_0_2: int64, 0i64, 4i64) {
                      block([], "T_dense_reindex_shared_wmma.accumulator_o") {
                        tir.reads([T_dense_reindex_shared_wmma.accumulator[0i64:16i64, (ax1_0_2*16i64):((ax1_0_2*16i64) + 16i64)]])
                        tir.writes([T_dense_reindex_shared[(threadIdx.y*16i64):((threadIdx.y*16i64) + 16i64), (ax1_0_2*16i64):((ax1_0_2*16i64) + 16i64)]])
                        @tir.tvm_store_matrix_sync(T_dense_reindex_shared_wmma.accumulator_1, 16, 16, 16, ax1_0_2, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), T_dense_reindex_shared_1: Pointer(shared float16), ((threadIdx.y*1024i64) + (ax1_0_2*16i64)), 1024i64, 2, dtype=handle), 64i64, "row_major", dtype=handle)
                    }
                  }
                for (ax0_ax1_fused_0_2: int64, 0i64, 4i64) {
                  for (ax0_ax1_fused_3_2: int64, 0i64, 8i64) "vectorized" {
                    block([], "T_dense_reindex_shared") {
                      tir.reads([T_dense_reindex_shared[(((ax0_ax1_fused_0_2*8i64) + (threadIdx.y*4i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64)), floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64)]])
                      tir.writes([T_dense[((((floordiv(blockIdx.x, 12i64)*32i64) + (ax0_ax1_fused_0_2*8i64)) + (threadIdx.y*4i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64)), ((floormod(blockIdx.x, 12i64)*64i64) + floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64))]])
                      T_dense[((((floordiv(blockIdx.x, 12i64)*32i64) + (ax0_ax1_fused_0_2*8i64)) + (threadIdx.y*4i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64)), ((floormod(blockIdx.x, 12i64)*64i64) + floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64))] = T_dense_reindex_shared[(((ax0_ax1_fused_0_2*8i64) + (threadIdx.y*4i64)) + floordiv(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64)), floormod(((threadIdx.x*8i64) + ax0_ax1_fused_3_2), 64i64)]
                  }
                }
              }
          }
        }
      }
    }