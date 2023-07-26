import sys
import importlib
import logging
from typing import List, Tuple

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

import process_ncu_profile
from process_ncu_profile import prepare_profile_data, split_train_test, \
    load_and_preprocess_profile_data, xgb_train_and_validate, get_data_frame_from_file, \
    load_all_data, preprocess_all_data_frame, split_all_train_test, xgb_train_and_validate


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

    xlsx_file_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_6-ncu.xlsx"
    features_ndarray_list_6, latency_6 = load_and_preprocess_profile_data(
        xlsx_file_path, num_average=2)
    train_dataset_6, test_dataset_6 = split_train_test(
        features_ndarray_list_6, latency_6, split_ratio=10)

    xlsx_file_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_7-ncu.xlsx"
    features_ndarray_list_7, latency_7 = load_and_preprocess_profile_data(
        xlsx_file_path, num_average=2)
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
        importlib.reload(process_ncu_profile)


if __name__ == "__main__":
    main()
    # main_loop([
    #     # "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_3-ncu.xlsx", 
    #            "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_6-ncu.xlsx",
    #            "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/bert_base_1_128_workload_7-ncu.xlsx"
    #     ], [2, 2], [10, 0])
