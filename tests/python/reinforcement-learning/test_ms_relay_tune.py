import tvm
from tvm.ir import IRModule
from tvm import relay
from tvm.target import Target
from tvm import meta_schedule as ms

import numpy as np


# def my_matmul():
#   np_a = np.ones((1024, 2048), dtype=np.float16)
#   np_b = np.ones((2048, 512), dtype=np.float16)
#   A = relay.Contant(tvm.nd.array(np_a))
#   B = relay.Contant(tvm.nd.array(np_b))
#   # A = relay.var('A', shape=(1024, 2048), dtype='float16')
#   # B = relay.var('B', shape=(2048, 512), dtype='float16')
#   C = relay.nn.dense(A, B)
#   return relay.Function([A, B], C)


def run_ms_tune(m, n, k, work_dir="ms_work_dir", path_to_trained_model=None):
  mod = IRModule({})
  A = relay.var('A', shape=(m, k), dtype='float16')
  B = relay.var('B', shape=(n, k), dtype='float16')
  my_func = relay.Function([A, B], relay.nn.dense(A, B))
  mod['main'] = my_func
  executor = relay.backend.Executor("graph", {"link-params": True})
  mod = mod.with_attr("executor", executor)
  target = Target("nvidia/nvidia-a100")
  database = ms.relay_integration.tune_relay(
      mod=mod,
      params={},
      target=target,
      work_dir=work_dir,
      max_trials_global=2000,
      max_trials_per_task=2000,
      path_to_trained_cost_model=path_to_trained_model,
      path_to_save_cost_model=None
  )

  


if __name__=="__main__":
  trained_path = "/home/xiachunwei/Software/tvm_oraa/tests/python/reinforcement-learning/cost_model_saved_matmul_m384k768n768"
  # run_ms_tune("/home/xiachunwei/Software/fusion/frozen_pbs/BERT-base/BERT-base.pb")
  run_ms_tune(384, 768, 768, path_to_trained_model=trained_path)
