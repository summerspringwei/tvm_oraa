# import tvm

# from tvm import te, tir


# def expand_func(w_bit: tir.Var, off_tir:tir.Var, w_bit_tir: tir.Var,\
#                  weight: te.Tensor, BITMASK: te.Tensor):
#     shape1 = (32, )
#     def smaller(off):
#         const_32 = tvm.tir.const(32,"int32")
#         return tvm.tir.LE(off_tir + w_bit_tir, const_32)
#     def expand(i):
#         off = tir.indexmod(i * w_bit, 32)
#         small = smaller(off)
#         return te.if_then_else(small, 
#                                 (
#                                     tir.shift_right(weight[tir.indexdiv(tir.multiply(i, w_bit), 32)], off) &
#                                     BITMASK[tir.min(32 - off, w_bit)]
#                                 ),
#                                 (
#                                     tir.shift_right(weight[tir.indexdiv(tir.multiply(i, w_bit), 32)], off) &
#                                     BITMASK[tir.min(32 - off, w_bit)]
#                                 )
#                 )
    
#     return te.compute(shape=shape1, fcompute=expand, name="expand")


# # 1. Pass tir.Var rather than IntImm
# # 2. Bind Variables to tvm's function's args
# # 3. Use tir.indexmod or tir.indexdiv rather than '/' or '%'
# # 4. Use tir.if_then_else rather than python's if else

# def test_expand_func():
#     # w_bit = te.placeholder((1,), dtype="int32", name='w_bit')
#     w_bit = tir.Var("w_bit", dtype="int32")
#     off_tir = tvm.tir.Var("off","int32")
#     w_bit_tir = tvm.tir.Var("w bit","int32")
#     weight = te.placeholder((1024,), dtype='int32', name='weight')
#     BITMASK = te.placeholder((1024,), dtype='int32', name='bitmask')
#     output = expand_func(w_bit, off_tir, w_bit_tir, weight, BITMASK)
#     s = te.create_schedule(output.op)
#     print(tvm.lower(s, [w_bit, off_tir, w_bit_tir, weight, BITMASK], simple_mode=True))
#     # func = tvm.build(s, [w_bit, weight, BITMASK], target='llvm', name="out", )
#     # print(func)


# if __name__=="__main__":
#     test_expand_func()


import os
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import tvm
from tvm.ir import IRModule
from tvm import relay, tir
from tvm.target import Target
from tvm import meta_schedule as ms
from tvm.meta_schedule.database import json_database
from tvm.tir.tensor_intrin import cuda


def my_matmul(m, n, k):
  A = relay.var('A', shape=(m, k), dtype='float16')
  B = relay.var('B', shape=(n, k), dtype='float16')
  my_func = relay.Function([A, B], relay.nn.dense(A, B))

  return my_func


class MetaSchedulerRunnerAndAnalyzer:
  def __init__(self, task_name, relay_func, relay_args, cost_model="xgboost", work_dir="ms_work_dir", gpu_name="nvidia-a100") -> None:
    self.task_name = task_name
    self.work_dir = work_dir
    self.relay_func = relay_func
    self.relay_func_args = relay_args
    self.gpu_name = gpu_name
    self.cost_model=cost_model
    self.id_str = self.task_name + "_" + self.cost_model + "_" + self.gpu_name
    self.path_to_save_model = os.path.join(
      "saved_cost_model", "cost_model_saved_" + self.id_str)
    self.path_to_save_figure = "figures"
    mod = IRModule({})
    mod['main'] = self.relay_func(*(self.relay_func_args))
    executor = relay.backend.Executor("graph", {"link-params": True})
    mod = mod.with_attr("executor", executor)
    self.mod = mod


  def run_ms_tune(self, cost_model, 
                  max_trials_global, max_trials_per_task, 
                    path_to_trained_model=None, save=False):
    target = Target("nvidia/"+self.gpu_name)
    database = ms.relay_integration.tune_relay(
        mod=self.mod,
        params={},
        target=target,
        work_dir=self.work_dir,
        cost_model=cost_model,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=8,
        num_tuning_cores="logical",
        path_to_trained_cost_model=path_to_trained_model,
        path_to_save_cost_model= self.path_to_save_model if save else None,
    )

if __name__=="__main__":
  tool = MetaSchedulerRunnerAndAnalyzer(
     "matmul_m384k768n768_from_scratch", my_matmul, (384, 768, 768), 
     gpu_name="geforce-rtx-3090",
    #  work_dir="saved_work_dir/matmul_m384k768n768_from_bert_trained_mlp_with_rank_error_count"
     work_dir="saved_work_dir/tmp_feature_extractor_test_64"
     )
  # tool.run_ms_tune("xgb", 2000, 2000, path_to_trained_model="saved_cost_model/cost_model_saved_matmul_m384k768n768_xgboost")
  tool.run_ms_tune("xgb", 16, 16)
  # tool.run_ms_tune("mlp", 128, 128, path_to_trained_model="saved_work_dir/((bert_large,[(1,64)]),cuda)_ms_workdir/mlp_model_state")
  # baseline = tool.load_npy_latency_us("np_data/ms_work_dir_matmul_m384k768n768_run_secs_xgboost.npy")
  # pretrain = tool.load_npy_latency_us("np_data/ms_work_dir_matmul_m384k768n768_run_secs_with_pretrain_model_xgboost.npy")
  # tool.draw_data_arr_compares([baseline, pretrain], offset=200)
  # load_tuning_records_and_apply("saved_work_dir/ms_work_dir_matmul_m384k768n768_with_validation_pretrain_xgboost/")
