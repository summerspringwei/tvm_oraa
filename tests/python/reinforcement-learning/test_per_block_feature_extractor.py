import os
import re
from multiprocessing import Pool
from typing import Mapping, List

import numpy as np

import tvm
from tvm import tir
from tvm import meta_schedule as ms
from tvm.meta_schedule import database
from tvm.target import Target
from tvm.ir.module import IRModule
from tvm.meta_schedule.logging import get_logger
from tvm.tir.tensor_intrin import cuda

logger = get_logger(__name__)
import logging
logger.setLevel(logging.DEBUG)


def record2schedule(record: database.TuningRecord):
    sch = tir.Schedule(record.workload.mod)
    record.trace.apply_to_schedule(sch, False)
    candidate = ms.MeasureCandidate(sch, record.args_info)
    return candidate


def load_tuning_records_and_save_features(work_dir_path):
    # Load database from file
    database = ms.database.create('json', 
      os.path.join(work_dir_path, "database_workload.json"),
      os.path.join(work_dir_path, "database_tuning_record.json"))
    all_tuning_record = database.get_all_tuning_records()
    print(f"Load {len(all_tuning_record)} records")
    # Split tuning records according to workload
    workload_record_map: dict[database.WorkLoad, List[database.TuningRecord]] = {}
    for record in all_tuning_record:
        if record.workload in workload_record_map.keys():
            workload_record_map[record.workload].append(record)
        else:
            workload_record_map[record.workload] = [record]
    
    extractor = ms.FeatureExtractor.create("per-block-feature", extract_workload=True)

    idx = 0
    for workload, record_list in workload_record_map.items():
        logger.info(workload.mod.script(), len(record_list))
        
        if len(record_list) == 0:
            continue
        
        tune_context = ms.TuneContext(mod=workload.mod, target=record_list[0].target, 
                        space_generator="post-order-apply", 
                        search_strategy="evolutionary")

        # Build schedule according to tuning records in serial
        schedule_arr = [tir.Schedule(workload.mod) for _ in record_list]
        trace_arr = [record.trace for record in record_list]
        tir.schedule.apply_trace_to_schedule_in_parallel(schedule_arr, trace_arr, False)
        candidate_arr = [ms.MeasureCandidate(sch, record.args_info) for sch, record in zip(schedule_arr, record_list)]
        # Convert run_secs from tvm.ir.container.Array to list
        run_secs_list = []
        for record in record_list:
            run_secs = np.array([float(sec) for sec in record.run_secs])
            mead_sec = np.mean(run_secs)
            run_secs_list.append(mead_sec)
        # run_secs_list = [record.run_secs for record in record_list]
        features = extractor.extract_from(tune_context, candidate_arr)
        exit(0)
        features = [f.numpy() for f in features]
        np.save(os.path.join(work_dir_path, f"workload_{idx}_features_run_secs"), 
                (features, np.array(run_secs_list)), allow_pickle=True)
        with open(os.path.join(work_dir_path, f"workload_{idx}_script"), 'w') as f:
            f.write(workload.mod.script())
            f.flush()
        logger.info(f"Finished workload {idx}")
        idx += 1


def test_single():
    # extractor = ms.FeatureExtractor.create("per-block-feature", extract_workload=True)
    # tune_context = ms.TuneContext(mod=mod, target=tvm.target.Target("nvidia/nvidia-a100"), 
    #                     space_generator="post-order-apply", 
    #                     search_strategy="evolutionary")
    
    # features = extractor.extract_from(tune_context, )
    pass

if __name__ == '__main__':
    # folder_path = "saved_work_dir/matmul_m384k768n768_from_scratch_mlp_with_rank_error_count"
    folder_path = "saved_work_dir/feature_extractor_test"
    load_tuning_records_and_save_features(folder_path)
