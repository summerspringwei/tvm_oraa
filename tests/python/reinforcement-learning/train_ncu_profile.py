import sys
import importlib
import logging
import os
from typing import List, Tuple

import numpy as np

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

import ncu_process_profile
from ncu_process_profile import normalize_profile_data, split_train_test, \
    load_and_preprocess_profile_data, xgb_train_and_validate, get_data_frame_from_file, \
    load_all_data, preprocess_all_data_frame, split_all_train_test, xgb_train_and_validate, \
    analyze_validation_result


def test_all_prediction():
    workload_id_list = [3, 4, 6, 7]
    workload_str_list = [f"bert_base_1_128_workload_{i}-ncu.xlsx" for i in workload_id_list]
    feature_mask = [1, 1, 1, 0]
    for train_workload in workload_str_list:
        for test_workload in workload_str_list:
            if train_workload == test_workload:
                continue
            logging.info(f"train_workload: {train_workload}, test_workload: {test_workload}")
            xlsx_file_path = os.path.join("saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir", train_workload)
            train_features_ndarray_list, train_latency = load_and_preprocess_profile_data(
                xlsx_file_path, num_average=2, feature_mask=feature_mask)
            train_train_dataset, train_test_dataset = split_train_test(
                train_features_ndarray_list, train_latency, split_ratio=10)
            xlsx_file_path = os.path.join("saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir", test_workload)
            train_features_ndarray_list, train_latency = load_and_preprocess_profile_data(
                xlsx_file_path, num_average=2, feature_mask=feature_mask)
            test_train_dataset, test_test_dataset = split_train_test(
                train_features_ndarray_list, train_latency, split_ratio=0)
            trainer = xgb_train_and_validate(train_train_dataset, test_test_dataset, None, None)
            logging.info(f"{train_workload} -> {test_workload}:")
            analyze_validation_result(trainer)


def main():
    # xlsx_file_path = "saved_work_dir/feature_extractor_test/feature_extractor_test-ncu.xlsx"
    # features_ndarray_list_1, latency_1 = preprocess_profile_data(
    #     xlsx_file_path, num_average = 2)
    # train_dataset_1, test_dataset_1 = split_train_test(
    #     features_ndarray_list_1, latency_1, split_ratio=10)
    # xlsx_file_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_3-ncu.xlsx"
    # features_ndarray_list_3, latency_3 = load_and_preprocess_profile_data(
    #     xlsx_file_path, num_average=2)
    # train_dataset_3, test_dataset_3 = split_train_test(
    #     features_ndarray_list_3, latency_3, split_ratio=10)
    xlsx_file_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_4-ncu.xlsx"
    feature_mask = [1, 0, 1, 1]
    features_ndarray_list_6, latency_6 = load_and_preprocess_profile_data(
        xlsx_file_path, num_average=2, feature_mask=feature_mask)
    train_dataset_6, test_dataset_6 = split_train_test(
        features_ndarray_list_6, latency_6, split_ratio=10)

    xlsx_file_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_5-ncu.xlsx"
    features_ndarray_list_7, latency_7 = load_and_preprocess_profile_data(
        xlsx_file_path, num_average=2, feature_mask=feature_mask)
    train_dataset_7, test_dataset_7 = split_train_test(
        features_ndarray_list_7, latency_7, split_ratio=0)
    # train_mlp(features_ndarray_list, latency, "saved_work_dir/feature_extractor_test/", None)
    xgb_train_and_validate(train_dataset_6, test_dataset_7,
                           None, None)


def main_loop(files_list: List[str], num_average_list: List[int], split_ratio_list: List[int]):
    data_cache = False
    data_frame_list = []
    while True:
        if not data_cache:
            data_frame_list = load_all_data(files_list)
            data_cache = True
        data_frame_list = preprocess_all_data_frame(data_frame_list, num_average_list)
        train_test_list = split_all_train_test(data_frame_list, split_ratio_list)
        xgb_train_and_validate(train_test_list[0][0], train_test_list[1][1], None, None)
        sys.stdin.readline()
        importlib.reload(ncu_process_profile)


def train_and_validate_tvm_features():
    def get_batched(features, costs, batch_size = 128):
        features = [f.astype(np.float32) for f in features]
        costs = [[c] for c in costs]
        train_dataset = []
        num_records = len(features)
        start, end = 0, batch_size
        while start < num_records:
            train_dataset.append((features[start: min(end, num_records)], costs[start: min(end, num_records)]))
            start += batch_size
            end += batch_size
        return train_dataset

    workload_id_list = [3, 4, 6, 7]
    workload_str_list = [f"workload_{i}_features_run_secs.npy" for i in workload_id_list]
    for train_workload in workload_str_list:
        for test_workload in workload_str_list:
            if train_workload == test_workload:
                continue
            logging.info(f"train_workload: {train_workload}, test_workload: {test_workload}")
            file_path = os.path.join("saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir", train_workload)
            features, costs = np.load(file_path, allow_pickle=True)
            train_dataset = get_batched(features, costs)
            file_path = os.path.join("saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir", test_workload)
            features, costs = np.load(file_path, allow_pickle=True)
            test_dataset = get_batched(features, costs)
            trainer = xgb_train_and_validate(train_dataset, test_dataset, None, None)
            logging.info(f"{train_workload} -> {test_workload}:")
            analyze_validation_result(trainer)


if __name__ == "__main__":
    # main()
    # test_all_prediction()
    train_and_validate_tvm_features()

    # main_loop([
    #     # "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_3-ncu.xlsx", 
    #            "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_6-ncu.xlsx",
    #            "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_7-ncu.xlsx"
    #     ], [2, 2], [10, 0])
