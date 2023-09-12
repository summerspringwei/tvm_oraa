import os
import logging

from itertools import chain as itertools_chain
from builtins import len
from typing import List, Tuple

import pandas as pd
import numpy as np

import ncu_metrics_utils

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def get_data_frame_from_file(file_path: str):
    fmt = file_path.split(".")[-1]
    if fmt != "xlsx":
        logging.error(f"Only support xlsx file, but give {fmt}")
    read_data_frame = pd.read_excel(file_path, engine='openpyxl')
    logging.info(read_data_frame.info())
    return read_data_frame


def metric_to_col_idx(read_data_frame: pd.DataFrame, metrics_list: List[str]) -> List[Tuple[str, int]]:
    """Convert metric to corresponding column idx in the excel

      Parameters
      ----------
      file_path: str
          The path to the excel file
      metrics_list: List[str],
          The list of metrics to find it's column index

      Returns
      ----------
      extracted_data_frame: pandas.DataFrame
    """
    new_metrics_list = []
    for metric in metrics_list:
        if metric in read_data_frame.keys():
            new_metrics_list.append(metric)
        else:
            find = False
            for k in read_data_frame.keys():
                if k.find(metric) >= 0:
                    new_metrics_list.append(k)
                    logging.info(f"Match metric: {metric} with {k}")
                    find = True
            if not find:
                logging.warning(f"Can not find metric: {metric}")
    logging.info(new_metrics_list)
    logging.info(len(new_metrics_list))
    # Note, we omit the unit at the second line
    extracted_data_frame = read_data_frame.loc[1:, new_metrics_list]
    # Filter latency being nan
    extracted_data_frame = extracted_data_frame.dropna(how='any')
    logging.info(extracted_data_frame)
    return extracted_data_frame


def gather_metrics():
    metrics_list = []
    all_sections = [ncu_metrics_utils.device_attrs, ncu_metrics_utils.launch_metrics,
                    ncu_metrics_utils.dram_attrs, ncu_metrics_utils.memory_section_attrs,
                    ncu_metrics_utils.compute_inst_attrs, ncu_metrics_utils.source_metrics,
                    ncu_metrics_utils.y_labels
                    ]
    for sec in all_sections:
        metrics_list.extend(sec)

    return metrics_list


def normalize_profile_data(extracted_data_frame: pd.DataFrame, feature_mask: List[int] = None):
    limited_blocks_all = extracted_data_frame.loc[:, [
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    ]].to_numpy()

    limited_blocks = np.min(limited_blocks_all, axis=1, keepdims=True)

    processed_ndarray = [limited_blocks]
    logging.info(limited_blocks.shape)
    headers = set(extracted_data_frame.columns)
    for (metric, func_args_tuple) in ncu_metrics_utils.preprocessing_func_mapping(feature_mask):
        if metric not in headers:
            logging.warning(f"{metric} not in preprocessing function mappings")
            continue
        metric_ndarray = extracted_data_frame.loc[:, [metric]].to_numpy()
        if len(func_args_tuple) == 2:
            func, args = func_args_tuple
        else:
            func, args = func_args_tuple[0], None
        if args is None:
            new_ndarray = func(metric_ndarray)
        elif isinstance(args[0], str):
            mother_metrics = extracted_data_frame.loc[:, [args[0]]]
            new_ndarray = func(metric_ndarray, mother_metrics.to_numpy())
        elif isinstance(args[0], int):
            new_ndarray = func(metric_ndarray, np.array(args[0]))
        processed_ndarray.append(new_ndarray)

    result = np.stack(processed_ndarray)

    return result


def shuffle_train_data(features: np.ndarray, labels: np.ndarray):
    num_elements, = labels.shape
    labels = labels.reshape((num_elements, 1))
    features_labels = np.concatenate([features, labels], axis=1)
    logging.info(features_labels.shape)
    np.random.shuffle(features_labels)
    return features_labels[:, 0:-1], features_labels[:, -1]


def average_record(data_array: np.ndarray, num_repeat: int):
    """Get the average value of data_array 
    as every num_repat raw refers to the same kernel's results"""
    assert(len(data_array.shape) >= 2)
    if num_repeat == 1:
        return data_array
    mod = data_array.shape[0] % num_repeat
    if mod != 0:
        data_array = data_array[:-mod]
    com = [data_array[i::num_repeat] for i in range(num_repeat)]
    return np.average(com, axis=0)


def preprocess_profile_data(data_frame: pd.DataFrame, num_average: int = 1, feature_mask: List[int] = None):
    # 3. Normalize x-data
    features_ndarray = normalize_profile_data(
        data_frame, feature_mask).squeeze().transpose()
    # 4. Concatenate x and y
    logging.info(
        f"x_shape: {features_ndarray.shape}")
    features_label = average_record(features_ndarray, num_average)
    logging.info(features_label.shape)
    # 6. Shuffle data
    np.random.shuffle(features_label)
    features, latency = features_label[:, 0:-1], features_label[:, -1]
    logging.info(
        f"features shape: {features.shape}, labels shape {latency.shape}")
    # 7. Convert to list of numpy array
    features_ndarray_list = [np.expand_dims(np.array(feature), 0).astype(
        np.float32) for feature in features.tolist()]

    return features_ndarray_list, latency


def extract_and_save_data_frame(xlsx_file_path: str):
    data_frame = get_data_frame_from_file(xlsx_file_path)
    extracted_data_frame = metric_to_col_idx(data_frame, gather_metrics())
    dirname = os.path.dirname(xlsx_file_path)
    basename = os.path.basename(xlsx_file_path)
    new_name = os.path.join(dirname, "extracted_"+basename)
    extracted_data_frame.to_excel(new_name)
    return extracted_data_frame


def load_and_preprocess_profile_data(xlsx_file_path: str, num_average: int = 1, feature_mask: List[int] = None):
    dirname = os.path.dirname(xlsx_file_path)
    basename = os.path.basename(xlsx_file_path)
    new_name = os.path.join(dirname, "extracted_"+basename)
    if os.path.exists(new_name):
        logging.info(f"Load from {new_name}")
        data_frame = get_data_frame_from_file(new_name)
    else:
        logging.info(f"Extract from {xlsx_file_path}")
        data_frame = extract_and_save_data_frame(xlsx_file_path)
    
    return preprocess_profile_data(data_frame, num_average=num_average, feature_mask=feature_mask)


def train_mlp(features: List[np.ndarray], labels: np.ndarray, folder_path: str, state_to_load: str = None):
    from tvm.meta_schedule.cost_model.mlp_model import State, SegmentSumMLPTrainer, SegmentSumMLPConfig, TrainerConfig
    state = State(model_config=SegmentSumMLPConfig(
        input_dim=40, use_norm=True, use_sigmoid=True))
    idx, batch_size = 0, 128
    num_sample = len(features)
    # state.add_to_group(features, labels, "hashhash")
    while idx < num_sample:
        state.add_to_group((features[idx: min(idx+batch_size, num_sample)]),
                           labels[idx: min(idx+batch_size, num_sample)], str(idx))
        idx += batch_size
    # return
    logging.info("Load all workload features")

    trainer = SegmentSumMLPTrainer(train_config=TrainerConfig(
        num_epoch_full=500, learning_rate=1e-4), state=state)
    trainer.train_full()
    # if state_to_load is None:
    #     trainer.train_full()
    # else:
    #     trainer.train_incremental()
    trainer.state.save(os.path.join(
        folder_path, "mlp_model_nsight_profile_state"))


def split_train_test(features, labels, batch_size=128, split_ratio=7):
    idx = 0
    num_sample = len(features)
    train_dataset, test_dataset = [], []
    from random import randrange
    while idx < num_sample:
        batched_feature = np.expand_dims(
            features[idx: min(idx+batch_size, num_sample)], axis=1)
        batched_labels = labels[idx: min(idx+batch_size, num_sample)]
        xs = list(itertools_chain.from_iterable(batched_feature))
        ys = np.min(batched_labels) / batched_labels
        if randrange(10) <= split_ratio:
            train_dataset.append((xs, ys))
        else:
            test_dataset.append((xs, ys))
        idx += batch_size

    return train_dataset, test_dataset


def xgb_train_and_validate(train_dataset: List[Tuple[List[np.ndarray], List[float]]],
                           test_dataset: List[Tuple[List[np.ndarray], List[float]]],
                           state_to_load: str = None, state_to_save: str = None):
    logging.info(f"Train xgb with {len(train_dataset)} training batches and {len(test_dataset)} batches")
    from tvm.meta_schedule.cost_model.xgb_model import XGBConfig, XGBModel
    trainer = XGBModel()
    if state_to_load is not None:
        trainer.load(state_to_load)
    for xs, ys in train_dataset:
        # try:
            # logging.info("Get {} real candidate latency".format(ys.size))
            trainer._train(
                xs=xs,
                ys=ys,
            )
        # except Exception as e:
        #     logging.warning(e)
        #     logging.warning(f"Has nan value {xs}")

    for xs, ys in test_dataset:
        # try:
            for key, score in trainer._validate(
                xs=xs,
                ys=ys,
            ):
                logging.info(f"XGB validation {key}: {score:.6f}")
        # except Exception:
        #     # logging.warning(f"Has nan value {xs}")

    if state_to_save is not None:
        trainer.save(state_to_save)
    logging.info(trainer.validation_map)
    return trainer


def load_all_data(files_list: List[str]):
    data_frame_list = [get_data_frame_from_file(
        file_path) for file_path in files_list]
    return data_frame_list


def preprocess_all_data_frame(data_frame_list: List[pd.DataFrame], num_average_list: List[int]):
    return [preprocess_profile_data(df, num_average) for df, num_average in zip(data_frame_list, num_average_list)]


def split_all_train_test(data_frame_list: List[pd.DataFrame], split_ratio_list: List[int]):
    return [split_train_test(feature, label, split_ratio=split_ratio) for (feature, label), split_ratio in zip(data_frame_list, split_ratio_list)]


def analyze_validation_result(trainer, top_one_threshold=0.05):
    a = "top_k_intersection_count@32/128"
    top_k_average = np.average(trainer.validation_map[a])
    logging.info(f"Top k average: {top_k_average}")
    count = 0
    num_batch = len(trainer.validation_map["top_one_performance_gap@128"])
    for degration in trainer.validation_map["top_one_performance_gap@128"]:
        count += 1 if degration < top_one_threshold else 0
    logging.info(f"Top one performance gap less than {top_one_threshold}: {count}/{num_batch}")
