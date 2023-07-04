import os
import numpy as np
from matplotlib import pyplot as plt
from typing import List
import logging

def draw_multiple_bar_points(data_arrs: List[np.ndarray], offset=0, x_label="", y_label="", file_name="figure"):
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
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    plt.savefig(os.path.join("figures", f"{file_name}.pdf"))
    plt.savefig(os.path.join("figures", f"{file_name}.tiff"))


def read_validation_from_log(file_path):
    data_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find("validation p-rmse") >= 0:
                value = float(line.split(" ")[-1])
                data_arr.append(value)
    return data_arr



def read_pairwaise_rank_error_from_log(file_path):
    data_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find("pairwise_rank_error_count") >= 0:
                com = line.split(" ")
                error_count = float(com[-1])
                batch = com[-2].split("@")[-1]
                if batch.find(':') > 0:
                    batch = batch[:-1]
                instance_count = int(batch)
                data_arr.append((error_count, instance_count))
    return data_arr


def read_topk_intersection_count(file_path):
    data_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find("top_k_intersection_count") >= 0:
                com = line.split(" ")
                intersection_count = float(com[-1])
                batch = com[-2].split("@")[-1]
                if batch.find('/') > 0:
                    batch = batch.split("/")[-1][:-1]
                instance_count = int(batch)
                data_arr.append((intersection_count, instance_count))
    return data_arr


if __name__=="__main__":
    # folder_name = "matmul_m384k768n768_from_scratch_mlp_with_rank_error_count"
    # folder_name = "matmul_m384k768n768_from_bert_trained_mlp_with_rank_error_count"
    folder_name = "matmul_m384k768n768_from_scratch_xgb_with_rank_error_count"
    file_path = f"saved_work_dir/{folder_name}/logs/tvm.meta_schedule.logging.task_scheduler.log"
    data_arr = (read_pairwaise_rank_error_from_log(file_path))
    values = np.array([v[0] for v in data_arr if v[0]>0], dtype=np.float32)
    instance_count = np.array([v[1]*(v[1]-1)/2 for v in data_arr if v[0]>0], dtype=np.float32)
    draw_multiple_bar_points([values / instance_count], x_label="Iterations", 
                             y_label="pairwise_error_ratio", file_name=folder_name+"_pairwise_error_count")
    print(values / instance_count)
    print(np.average(values / instance_count))
    data_arr = read_topk_intersection_count(file_path)
    values = np.array([v[0] for v in data_arr if v[0]>2], dtype=np.float32)
    print(values)
    print(np.average(values))
    instance_count = np.array([v[1] for v in data_arr], dtype=np.float32)
    draw_multiple_bar_points([values], x_label="Iterations", 
                             y_label="top_32 from 64", file_name=folder_name+"_top_32_count")
    print(np.average(values))
