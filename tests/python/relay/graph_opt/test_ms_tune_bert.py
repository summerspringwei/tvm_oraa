import onnx
import tvm
from tvm.ir import IRModule
from tvm import relay
from tvm.target import Target
from tvm import meta_schedule as ms
import tensorflow as tf
import numpy as np


def load_tf_model(model_file):
  with tf.io.gfile.GFile(model_file, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def my_matmul():
  np_a = np.ones((1024, 2048), dtype=np.float16)
  np_b = np.ones((2048, 512), dtype=np.float16)
  A = relay.Contant(tvm.nd.array(np_a))
  B = relay.Contant(tvm.nd.array(np_b))
  # A = relay.var('A', shape=(1024, 2048), dtype='float16')
  # B = relay.var('B', shape=(2048, 512), dtype='float16')
  C = relay.nn.dense(A, B)
  return relay.Function([A, B], C)


def run_ms_tune(work_dir="ms_work_dir"):
  mod = IRModule({})
  A = relay.var('A', shape=(1024, 2048), dtype='float16')
  B = relay.var('A', shape=(512, 2048), dtype='float16')
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
      max_trials_global=1,
      max_trials_per_task=1
  )


def run_ms_tune(model_file, work_dir="ms_work_dir"):
  graph_def = load_tf_model(model_file)
  mod, params = relay.frontend.from_tensorflow(graph_def, )
  print(graph_def)
  executor = relay.backend.Executor("graph", {"link-params": True})
  mod = mod.with_attr("executor", executor)
  print(mod)
  target = Target("nvidia/nvidia-a100")
  
  database = ms.relay_integration.tune_relay(
      mod=mod,
      params={},
      target=target,
      work_dir=work_dir,
      max_trials_global=100000,
      max_trials_per_task=1000
  )


if __name__=="__main__":
  # run_ms_tune("/home/xiachunwei/Software/fusion/frozen_pbs/BERT-base/BERT-base.pb")
  run_ms_tune()
