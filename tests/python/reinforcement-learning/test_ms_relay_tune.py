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

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

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
        num_tuning_cores="logical",
        path_to_trained_cost_model=path_to_trained_model,
        path_to_save_cost_model= self.path_to_save_model if save else None,
    )
  

  def load_tuning_records_latency_to_npy(self):
    my_database = json_database.JSONDatabase(work_dir=self.work_dir)
    all_tuning_records = my_database.get_all_tuning_records()
    all_run_secs = np.array([record.run_secs \
                             for record in all_tuning_records if float(record.run_secs[0]) < 0.1], \
                              dtype=np.float32)
    np.save(os.path.join("np_data", self.id_str+"_run_secs.npy"), all_run_secs)


  def draw_multiple_bar_points(self, data_arrs: List[np.ndarray], offset=0):
    # Draw comparation
    fig = plt.figure()  # an empty figure with no Axes
    fig, axs = plt.subplots()  # a figure with a single Axes
    min_length = data_arrs[0].size
    for arr in data_arrs:
        if arr.size < min_length:
            min_length = arr.size
    new_data_arrs = []
    logging.info(data_arrs)
    for arr in data_arrs:
        logging.info(arr)
        new_data_arrs.append(arr[0:min_length])
    
    t = range(offset, offset+min_length)
    colors = ['r', 'g', 'b']
    idx = 0
    for arr in data_arrs:
        axs.plot(t, arr, colors[idx])
        idx += 1
    axs.set_xlabel("Iterations")
    axs.set_ylabel("Latency (us)")
    plt.savefig(os.path.join(self.path_to_save_figure, self.id_str + "_" + str(offset) + "_latency_compares.pdf"))
    plt.savefig(os.path.join(self.path_to_save_figure, self.id_str + "_" + str(offset) + "_latency_compares.tiff"))


  def draw_data_arr_compares(self, data_arr, offset=0):
    data_arr = [arr.reshape(-1) for arr in data_arr]
    min_length = min([arr.size for arr in data_arr])
    for arr in data_arr:
       if arr.size != min_length:
          logging.warning("{} not equal to min_length: {}".format(arr.size, min_length))
    # Reverse 
    data_arr = [np.flip(arr)[-min_length:] for arr in data_arr]
    # Draw comparation
    self.draw_multiple_bar_points(data_arr)
    self.draw_multiple_bar_points([arr[offset:] for arr in data_arr], offset=offset)

  
  def load_npy_latency_us(self, file_path):
      all_run_secs = np.load(file_path).astype(np.float32)
      all_run_micro_seconds = all_run_secs * 1e6
      min_value = np.min(all_run_micro_seconds)
      print(min_value)
      lines = []
      tmp_file_path = os.path.join("/tmp", file_path)
      if not os.path.exists(os.path.dirname(tmp_file_path)):
         os.mkdir(os.path.dirname(tmp_file_path))
      with open(tmp_file_path, 'w') as f:
          for i in range(len(all_run_micro_seconds)):
              lines.append(str(all_run_micro_seconds[i]) + "\n")
          f.writelines(lines)
          f.flush()
      return all_run_micro_seconds
  
def load_tuning_records_and_apply(work_dir_path):
    database = ms.database.create('json', 
      os.path.join(work_dir_path, "database_workload.json"),
      os.path.join(work_dir_path, "database_tuning_record.json"))
    all_tuning_record = database.get_all_tuning_records()
    
    target = Target("nvidia/nvidia-a100")
    tune_context = ms.TuneContext(mod=all_tuning_record[0].workload.mod, target=target, 
                    space_generator="post-order-apply", 
                    search_strategy="evolutionary")
    extractor = ms.FeatureExtractor.create("per-store-feature")
    candidates = []
    idx = 0
    for record in all_tuning_record:
      # print(record.workload.mod.script())
      sch = tir.Schedule(record.workload.mod)
      record.trace.apply_to_schedule(sch, False)
      # print(sch.mod.script())
      candidates.append(ms.MeasureCandidate(sch, record.args_info))
      idx += 1
      print(idx)
      if idx > 10:
        break
    features = extractor.extract_from(tune_context, candidates)
    for f in features:
      print(f.numpy())


if __name__=="__main__":
  # tool = MetaSchedulerRunnerAndAnalyzer("matmul_m384k768n768_from_scratch", my_matmul, (384, 768, 768), gpu_name="geforce-rtx-3090")
  # tool.run_ms_tune("xgb", 2000, 2000, path_to_trained_model="saved_cost_model/cost_model_saved_matmul_m384k768n768_xgboost")
  # tool.run_ms_tune("xgb", 2000, 2000)
  # baseline = tool.load_npy_latency_us("np_data/ms_work_dir_matmul_m384k768n768_run_secs_xgboost.npy")
  # pretrain = tool.load_npy_latency_us("np_data/ms_work_dir_matmul_m384k768n768_run_secs_with_pretrain_model_xgboost.npy")
  # tool.draw_data_arr_compares([baseline, pretrain], offset=200)
  load_tuning_records_and_apply("saved_work_dir/ms_work_dir_matmul_m384k768n768_with_validation_pretrain_xgboost/")
