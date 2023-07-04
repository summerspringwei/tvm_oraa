
# Arranged according to ncu official document https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

"""To simulate feature extractor from code,
we only extract tour kinds of features:
    (1) Hardware features like number of SM's, L2 cache size;
    (2) Kernel launch configurations, like block dim and grid dim and ocupancy
    (3) Memory related info, like number of bytes read/write, cache miss ratio
    (4) Compute related info, like number of FMA instructions
"""

metrics = [
    "derived__memory_l1_wavefronts_shared_excessive [byte]",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]

device_attrs = [
    "device__attribute_fb_bus_width",
    "device__attribute_fbp_count",
    "device__attribute_l2s_count",
    "device__attribute_clock_rate",
    "device__attribute_global_memory_bus_width",
    "device__attribute_l2_cache_size",
    "device__attribute_max_gpu_frequency_khz",
    "device__attribute_concurrent_kernels",
    "device__attribute_max_warps_per_multiprocessor",
    "device__attribute_max_blocks_per_multiprocessor",
    "device__attribute_max_ipc_per_multiprocessor", # Use this to normalize
    "device__attribute_max_ipc_per_scheduler", # Use this to normalize
    "device__attribute_num_l2s_per_fbp",
    "device__attribute_max_warps_per_scheduler",
    "device__attribute_max_mem_frequency_khz",
    "device__attribute_max_registers_per_block",
    "device__attribute_max_registers_per_multiprocessor",
    "device__attribute_max_registers_per_thread",
    "device__attribute_max_shared_memory_per_block",
    "device__attribute_max_shared_memory_per_block_optin",
    "device__attribute_max_shared_memory_per_multiprocessor",
    "device__attribute_max_threads_per_block",
    "device__attribute_max_threads_per_multiprocessor",
    "device__attribute_memory_clock_rate",
    "device__attribute_num_schedulers_per_multiprocessor",
    "device__attribute_limits_max_cta_per_sm",
    "device__attribute_multiprocessor_count",
]


launch_metrics = [
    "Grid Size",
    "Block Size",
    "launch__occupancy_limit_blocks [block]",
    "launch__occupancy_limit_registers [block]",
    "launch__occupancy_limit_shared_mem [block]",
    "launch__occupancy_limit_warps [block]",
    "launch__registers_per_thread [register/thread]",
    "launch__shared_mem_config_size [Kbyte]",
    "launch__shared_mem_per_block_static [Kbyte/block]",
    "launch__waves_per_multiprocessor",
]

dram_attrs = [
    "dram__bytes.sum.peak_sustained [byte/cycle]",
    "dram__bytes.sum.per_second [Gbyte/second]",
    "dram__bytes_read.sum [Mbyte]",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]",
    "dram__bytes_write.sum [byte]",
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed [%]",
    "dram__cycles_elapsed.avg.per_second [cycle/nsecond]",
    "dram__sectors_read.sum [sector]",
    "dram__sectors_write.sum [sector]",
]

memory_section_attrs = [
    # Shared memory
    "smsp__sass_inst_executed_op_shared_ld",                        # Shared memory load
    "smsp__sass_inst_executed_op_shared_st",                        # Shared memory store
    "smsp__inst_executed_op_ldsm",                                  # of warp instructions executed: LDSM
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld",         # bank conflicts of load shared memory
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st",         # bank conflicts of store shared memory
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld",         # bank conflicts of load global memory
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_red",        # bank conflicts of reduce global memory
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st",         # bank conflicts of store global memory
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_atom",       # bank conflicts of atomic operation global memory
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum [request]",
    "sass__inst_executed_global_loads [inst]",                      # Global memory load
    "sass__inst_executed_global_stores [inst]",                     # Global memory store
    "sm__sass_inst_executed_op_ldgsts_cache_bypass.sum [inst]",     # Load from global store to share bypass
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum [sector]",   # L2 cache hit
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum [sector]",  # L2 cache miss
    "smsp__sass_inst_executed_op_memory_128b.sum [inst]",
    "smsp__sass_inst_executed_op_memory_16b.sum [inst]",
    "smsp__sass_inst_executed_op_memory_32b.sum [inst]",
    "smsp__sass_inst_executed_op_memory_64b.sum [inst]"
]

# For now we do not consider computation ops
compute_inst_attrs = [
    "smsp__inst_executed_op_branch.sum [inst]",
    "sm__inst_executed_pipe_cbu_pred_on_any.avg.pct_of_peak_sustained_elapsed [%]",
    "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active [%]",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active [%]",
    "sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed [%]",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active [%]",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed [%]"
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active [%]",
]

fbpa_attrs = [
    "inst_executed [inst]",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_bank_reads.avg.pct_of_peak_sustained_elapsed [%]",
    "l1tex__data_bank_writes.avg.pct_of_peak_sustained_elapsed [%]",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__m_l1tex2xbar_write_bytes.sum.per_second [Gbyte/second]",
    "sass__inst_executed_global_loads [inst]",
    "sass__inst_executed_global_stores [inst]",
    "sass__inst_executed_shared_loads [inst]",
    "sass__inst_executed_shared_stores [inst]",
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active [%]",
    "sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed [%]",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active [%]",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed [%]",
    "",
]

source_metrics = [
    "smsp__branch_targets_threads_divergent",
]

questions = [
    "derived__avg_thread_executed [thread]",
    "derived__memory_l2_theoretical_sectors_global_excessive [byte]",
    "derived__smsp__inst_executed_op_branch_pct [%]",
    "device__attribute_async_engine_count",
    "device__attribute_memory_pools_supported",
    "device__attribute_multiprocessor_count",
    "device__attribute_num_tex_per_multiprocessor",
    "device__attribute_total_constant_memory",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed [%]",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed [%]",
    "l1tex__m_l1tex2xbar_write_bytes.sum.per_second [Gbyte/second]",
    "lts__gcomp_input_sectors.sum [sector]",
    "memory_l1_tag_requests_global [sectors]",
    "memory_l1_wavefronts_shared [sectors]",
    "memory_l1_wavefronts_shared_ideal [sectors]",
    "memory_l2_theoretical_sectors_global [sectors]",
    "sass__thread_inst_executed_true_per_opcode",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed [%]",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active [%]",
    "sm__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_active [%]",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed [%]",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained [inst/cycle]",
    "smsp__cycles_elapsed.sum [cycle]",
    "smsp__inst_executed.avg [inst]", # important
    "smsp__pcsamp_warps_issue_stalled_tex_throttle [warp]", # Important
    "smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_st.sum.pct_of_peak_sustained_elapsed [%]",
    "thread_inst_executed [inst]",
    "",
    "",
]


y_labels = [
    "gpc__cycles_elapsed.max [cycle]",
    "gpu__time_duration.sum [usecond]",
]