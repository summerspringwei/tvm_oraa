import pickle

import tvm
from tvm import relay
from tvm import meta_schedule as ms

import onnx


def export_onnx_to_relay_tasks(model_file, tvm_task_file):
  graph_def = onnx.load(model_file)
  mod, params = relay.frontend.from_onnx(graph_def)
  tasks = ms.relay_integration.extract_tasks(mod, 
        target=tvm.target.Target("nvidia/nvidia-a100") , params=params)
  # pickle.dump((mod, params), open("xxx", 'wb')) # Failed for mod contains C pointers
  pickle.dump(tasks, open(tvm_task_file, 'wb'))


def export_all_efficientnet_tasks():
  for i in range(1, 8):
    model_file = f"/home2/xiachunwei/Software/fusion/frozen_pbs/efficientnet-b0/efficientnet-b{i}.onnx"
    tvm_task_file = f"efficientnet-b{i}-tvm-tasks"
    export_onnx_to_relay_tasks(model_file, tvm_task_file)


