import os
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

logger = get_logger("load_dataset")
import logging
logger.setLevel(logging.DEBUG)


def record2schedule(record: database.TuningRecord):
    sch = tir.Schedule(record.workload.mod)
    record.trace.apply_to_schedule(sch, False)
    candidate = ms.MeasureCandidate(sch, record.args_info)
    return candidate


# def trace2schedule(mod, IRModule, trace: tir.Trace):
#     sch = tir.Schedule(record.workload.mod)
#     record.trace.apply_to_schedule(sch, False)
#     candidate = ms.MeasureCandidate(sch, record.args_info)
#     return candidate


def load_tuning_records_and_save_features(work_dir_path):
    # Load database from file
    database = ms.database.create('json', 
      os.path.join(work_dir_path, "database_workload.json"),
      os.path.join(work_dir_path, "database_tuning_record.json"))
    all_tuning_record = database.get_all_tuning_records()
    
    # Split tuning records according to workload
    workload_record_map: dict[database.WorkLoad, List[database.TuningRecord]] = {}
    for record in all_tuning_record:
        if record.workload in workload_record_map.keys():
            workload_record_map[record.workload].append(record)
        else:
            workload_record_map[record.workload] = [record]
    
    extractor = ms.FeatureExtractor.create("per-store-feature")

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
        run_secs_list = [record.run_secs for record in record_list]
        features = extractor.extract_from(tune_context, candidate_arr)
        features = [f.numpy() for f in features]
        np.save(os.path.join(work_dir_path, f"workload_{idx}_features_run_secs"), 
                (features, run_secs_list))
        with open(os.path.join(work_dir_path, f"workload_{idx}_script"), 'w') as f:
            f.write(workload.mod.script())
            f.flush()
        logger.info(f"Finished workload {idx}")
        idx += 1


def load_tuning_records_and_train(work_dir_path):
    from tvm.meta_schedule.cost_model.mlp_model import State, SegmentSumMLPTrainer, MLPModel
    from tvm.meta_schedule.runner import RunnerResult
    # Load database from file
    database = ms.database.create('json', 
      os.path.join(work_dir_path, "database_workload.json"),
      os.path.join(work_dir_path, "database_tuning_record.json"))
    all_tuning_record = database.get_all_tuning_records()
    
    # Split tuning records according to workload
    workload_record_map: dict[database.WorkLoad, List[database.TuningRecord]] = {}
    for record in all_tuning_record:
        if record.workload in workload_record_map.keys():
            workload_record_map[record.workload].append(record)
        else:
            workload_record_map[record.workload] = [record]

    idx = 0
    model = MLPModel()
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
        run_secs_list = [record.run_secs for record in record_list]
        runner_result_list = [RunnerResult(secs, "ok") for secs in run_secs_list]
        model.update(tune_context, candidate_arr, runner_result_list)
        print(f"Trained workload {idx}")
        idx += 1
    model.save(os.path.join(work_dir_path, "mlp_model_state"))


def load_feature_to_train(folder_path):
    from tvm.meta_schedule.cost_model.mlp_model import State, SegmentSumMLPTrainer
    state = State()
    for i in range(10):
        np_file_path = os.path.join(folder_path, f"workload_{i}_features_run_secs.npy")
        features, costs = np.load(np_file_path, allow_pickle=True)
        # logger.info(f"Load workload {i} with {len(features)} records")
        print(f"Load workload {i} with {len(features)} records")
        # print(features)
        # print(costs.shape)
        # print(isinstance(costs, ))
        costs = np.array([c[0] for c in costs])
        state.add_to_group(features, costs, f"workload_{i}_features_run_secs")
        
    trainer = SegmentSumMLPTrainer(state=state)
    trainer.train_full()
    trainer.state.save(os.path.join(folder_path, "mlp_model_state"))


if __name__ == '__main__':
    folder_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/"
    # load_tuning_records_and_save_features(folder_path)
    # load_feature_to_train(folder_path)
    load_tuning_records_and_train(folder_path)
    # logger.setLevel(logging.DEBUG)
    # logger.warn("aaa")
