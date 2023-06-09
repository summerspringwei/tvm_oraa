import os

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.target import Target
from tvm.meta_schedule.database import json_database
# Must import cuda to register the intrinsics in the workload file
from tvm.tir.tensor_intrin import cuda

import numpy as np
import matplotlib.pyplot as plt


def load_tuning_records(dir_path):
    mod = IRModule({})
    A = relay.var('A', shape=(384, 768), dtype='float16')
    B = relay.var('B', shape=(768, 768), dtype='float16')
    my_func = relay.Function([A, B], relay.nn.dense(A, B))
    mod['main'] = my_func
    executor = relay.backend.Executor("graph", {"link-params": True})
    mod = mod.with_attr("executor", executor)
    target = Target("nvidia/nvidia-a100")
    my_database = json_database.JSONDatabase(work_dir=dir_path)
    all_tuning_records = my_database.get_all_tuning_records()
    all_run_secs = np.array([record.run_secs for record in all_tuning_records if float(record.run_secs[0]) < 0.1], dtype=np.float32)
    np.save(dir_path+"_run_secs.npy", all_run_secs)


def load_run_secs(file_path):
    all_run_secs = np.load(file_path).astype(np.float32)
    all_run_micro_seconds = all_run_secs * 1e6
    min_value = np.min(all_run_micro_seconds)
    print(min_value)
    lines = []
    with open("tmp.txt", 'w') as f:
        for i in range(len(all_run_micro_seconds)):
            lines.append(str(all_run_micro_seconds[i]) + "\n")
        f.writelines(lines)
        f.flush()
    return all_run_micro_seconds    


def draw_run_secs(all_run_secs, dir_path):
    fig = plt.figure()  # an empty figure with no Axes
    fig, axs = plt.subplots()  # a figure with a single Axes
    t = range(len(all_run_secs))
    axs.plot(t, all_run_secs)
    axs.set_ylabel("latency (us)")
    plt.savefig(dir_path+"_latency.pdf")
    plt.savefig(dir_path+"_latency.jpg")


def draw_multiple_bar_points(data_arrs, figure_name, offset=0):
    # Draw comparation
    fig = plt.figure()  # an empty figure with no Axes
    fig, axs = plt.subplots()  # a figure with a single Axes
    min_length = data_arrs[0].size
    for arr in data_arrs:
        if arr.size < min_length:
            min_length = arr.size
    new_data_arrs = []
    for arr in data_arrs:
        new_data_arrs.append(arr[0:min_length])
    
    t = range(offset, offset+min_length)
    colors = ['r', 'g', 'b']
    idx = 0
    for arr in data_arrs:
        axs.plot(t, arr, colors[idx])
        idx += 1
    axs.set_xlabel("Iterations")
    axs.set_ylabel("Latency (us)")
    plt.savefig(figure_name+"_latency_compares.pdf")
    plt.savefig(figure_name+"_latency_compares.tiff")


def draw_two_run_compares(baseline_run_secs, pretrained_run_secs, figure_name):
    baseline_run_secs = baseline_run_secs.reshape((baseline_run_secs.size))
    pretrained_run_secs = pretrained_run_secs.reshape((pretrained_run_secs.size))
    min_length = min(baseline_run_secs.size, pretrained_run_secs.size)
    # Reverse 
    baseline_run_secs = np.flip(baseline_run_secs)
    pretrained_run_secs = np.flip(pretrained_run_secs)
    baseline_run_secs = baseline_run_secs[-min_length:]
    pretrained_run_secs = pretrained_run_secs[-min_length:]
    # Draw comparation
    draw_multiple_bar_points([baseline_run_secs, pretrained_run_secs], figure_name+"-2000")
    draw_multiple_bar_points([baseline_run_secs[200:], pretrained_run_secs[200:]], figure_name+"-1800", offset=200)


if __name__=="__main__":
    saved_work_dirs = "saved_work_dir"
    saved_np_data = "np_data"
    benchmark_dir_path = "ms_work_dir_matmul_m384k768n768"
    # benchmark_dir_path = "ms_work_dir"
    # benchmark_dir_path = "ms_work_dir_matmul_m1024k1024n1024"
    # load_tuning_records(benchmark_dir_path)
    baseline_run_secs = load_run_secs(os.path.join(saved_np_data, "ms_work_dir_matmul_m384k768n768"+"_run_secs_xgboost.npy"))
    pretrained_run_secs = load_run_secs(os.path.join(saved_np_data, "ms_work_dir_matmul_m384k768n768_run_secs_with_pretrain_model_xgboost.npy"))
    draw_two_run_compares(baseline_run_secs, pretrained_run_secs, "figures/matmul_m384k768n768")
    # draw_run_secs(all_run_secs, benchmark_dir_path)
