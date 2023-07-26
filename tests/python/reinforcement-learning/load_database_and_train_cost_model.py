import os
import re
import pickle
import subprocess
import logging
from typing import List, Tuple

import numpy as np

import tvm
from tvm import tir
from tvm import meta_schedule as ms
from tvm.meta_schedule import database, FeatureExtractor
from tvm.target import Target
from tvm.tir.tensor_intrin import cuda
from tvm.ir.module import IRModule
from tvm.meta_schedule.logging import get_logger


logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


def record2schedule(record: database.TuningRecord):
    """Convert TuningRecord to tir.Schedule

    Parameters
    ----------
    record: database.TuningRecord
        The record to convert
    
    Returns
    -------
    candidate: The MesureCandidate which include the transformed schedule and args
    """
    sch = tir.Schedule(record.workload.mod)
    record.trace.apply_to_schedule(sch, False)
    candidate = ms.MeasureCandidate(sch, record.args_info)
    return candidate


def load_tuning_records(work_dir_path: str):
    """Load database from workdir

    Parameters
    ----------
    work_dir_path: str
        The path to work dir
    
    Returns
    -------
    workload_record_map: dict[database.WorkLoad, List[database.TuningRecord]]
        A dict with key being the tuned workload, and value being a list of tuning record
    """
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
    
    return workload_record_map


def extract_candidate_and_features(extractor: FeatureExtractor, mod: IRModule, \
                     record_list: List[database.TuningRecord]):
    """Extract tuning record to features

    Parameters
    ----------
    extractor: FeatureExtractor
        The extractor to extract features from scheduled mod
    mod: IRModule,
        The scheduled IRModule
    record_list: List[database.TuningRecord]
        A List of TuningRecord

    Returns
    ----------
    (candidate_arr, features, run_secs_list): Tuple[List[MeasureCandidate], List[np.ndarray], List[float]]
    """
    if len(record_list) == 0:
        return []
    tune_context = ms.TuneContext(mod=mod, target=record_list[0].target, 
                        space_generator="post-order-apply", 
                        search_strategy="evolutionary")
    # Build schedule according to tuning records in serial
    schedule_arr = [tir.Schedule(mod) for _ in record_list]
    trace_arr = [record.trace for record in record_list]
    tir.schedule.apply_trace_to_schedule_in_parallel(schedule_arr, trace_arr, False)
    candidate_arr = [ms.MeasureCandidate(sch, record.args_info) for sch, record in zip(schedule_arr, record_list)]
    # Convert run_secs from tvm.ir.container.Array to list
    run_secs_list = []
    for record in record_list:
        run_secs = np.array([float(sec) for sec in record.run_secs])
        mead_sec = np.mean(run_secs)
        run_secs_list.append(mead_sec)
    features = extractor.extract_from(tune_context, candidate_arr)
    features = [f.numpy() for f in features]

    return (candidate_arr, features, run_secs_list)


def save_features_to_file(mod: IRModule, features: List[np.ndarray], 
                          run_secs_list: List[float], work_dir_path: str, idx: int):
    """Save features and mod's script to file
    """
    np.save(os.path.join(work_dir_path, f"workload_{idx}_features_run_secs"), 
                (features, np.array(run_secs_list)), allow_pickle=True)
    with open(os.path.join(work_dir_path, f"workload_{idx}_script"), 'w') as f:
        f.write(mod.script())
        f.flush()
    logger.info(f"Finished workload {idx}")


def load_tuning_records_and_save_features(work_dir_path):
    """Extract features from work dir and save
    """
    workload_record_map = load_tuning_records(work_dir_path)

    extractor = ms.FeatureExtractor.create("per-store-feature", extract_workload=True)
    idx = 0
    for workload, record_list in workload_record_map.items():
        logger.info(workload.mod.script(), len(record_list))
        (_, features, run_secs_list) = extract_candidate_and_features(extractor, workload.mod, record_list)
        save_features_to_file(workload.mod, features, run_secs_list, work_dir_path, idx)
        logger.info(f"Finished workload {idx}")
        idx += 1


def load_tuning_records_and_train(work_dir_path):
    from tvm.meta_schedule.cost_model.mlp_model import State, SegmentSumMLPTrainer, MLPModel
    from tvm.meta_schedule.runner import RunnerResult
    workload_record_map = load_tuning_records(work_dir_path)

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


def load_feature_to_train(folder_path: str, state_to_load: str = None, target: str = "nvidia/nvidia-a100"):
    from tvm.meta_schedule.cost_model.mlp_model import State, SegmentSumMLPTrainer
    state = State()
    if state_to_load is not None:
        state.load(state_to_load, target=target)
    workload_feature_files = [f for f in os.listdir(folder_path) \
             if os.path.isfile(os.path.join(folder_path, f)) and \
             re.search("workload_[0-9]+_features_run_secs", f)]
    for f in workload_feature_files:
        np_file_path = os.path.join(folder_path, f)
        features, costs = np.load(np_file_path, allow_pickle=True)
        features = [f.astype(np.float32) for f in features]
        costs =  costs.astype(np.float32)
        # features, costs = np.vstack(features).astype(np.float), np.vstack(costs).astype(np.float)
        # logger.info(f"Load workload {i} with {len(features)} records")
        print(f"Load {f} with {len(features)} records")
        # print(features)
        # print(costs.shape)
        # print(isinstance(costs, ))
        # costs = np.array([c for c in costs])
        state.add_to_group(features, costs, f)
    # return
    logger.info("Load all workload features")
    trainer = SegmentSumMLPTrainer(state=state)
    trainer.train_full()
    # if state_to_load is None:
    #     trainer.train_full()
    # else:
    #     trainer.train_incremental()
    trainer.state.save(os.path.join(folder_path, "mlp_model_state"))


def build_tuning_records_to_runer_inputs(work_dir_path: str, 
                                         target: str = "nvidia/nvidia-a100"):
    """Load tuning records and build to libraries
    Parameters
    ----------
    work_dir_path: str, 
        The path to work dir
    target: str
        The build target
    """
    from tvm.meta_schedule.builder import builder
    from tvm.meta_schedule.runner import runner
    # 1. Load tuning records
    workload_record_map = load_tuning_records(work_dir_path)
    extractor = ms.FeatureExtractor.create("per-store-feature", extract_workload=True)
    idx = 0
    for workload, record_list in workload_record_map.items():
        logger.info(workload.mod.script(), len(record_list))
        # 2. Load measure candidates
        (measure_candidate, _, _) = extract_candidate_and_features(extractor, workload.mod, record_list)
        schedule_arr = [mc.sch for mc in measure_candidate]
        args_info_arr = [mc.args_info for mc in measure_candidate]
        build_input = [builder.BuilderInput(sch.mod, Target(target)) for sch in schedule_arr]

        # 3. Create builder to compile IRModule to executable library
        local_builder = builder.create(kind="local", max_workers=80, timeout_sec=30)
        build_result = local_builder.build(build_input)
        runner_input = []
        # 4. Copy build library to lib folder for further run
        lib_dir = os.path.join(work_dir_path, "lib")
        if not os.path.exists(lib_dir):
            os.mkdir(lib_dir)
        for br, bi, args_info in zip(build_result, build_input, args_info_arr):
            module_hash = str(bi.mod.__hash__())
            dst_file_folder = os.path.join(lib_dir, module_hash)
            if not os.path.exists(dst_file_folder):
                os.mkdir(dst_file_folder)
            dst_file_path = os.path.join(dst_file_folder, "tvm_tmp_mod.tar")
            if os.path.exists(dst_file_path):
                continue
            try:
                subprocess.run(["cp", br.artifact_path, dst_file_path])
                runner_input.append((runner.RunnerInput(dst_file_path, "cuda", []), [arg.as_json() for arg in args_info]))
            except Exception:
                print(f"Copy error: artifact_path: {br.artifact_path}, dst_file_path: {dst_file_path}")
        # 5. Dump builder results (artifact_path, device_info, args_info) to picke file for further load
        with open(os.path.join(work_dir_path, f"workload_{idx}_build_results.pkl"), 'wb') as f:
            pickle.dump(runner_input, f)
        idx += 1


def load_runner_input_and_run(work_dir_path, device_id = 0):
    """Load tuning records and build to libraries
    Parameters
    ----------
    work_dir_path: str, 
        The path to work dir
    device_id: int
        The device to run on
    """
    from tvm.meta_schedule.runner import runner
    from tvm.meta_schedule import arg_info
    from tvm.meta_schedule.runner import EvaluatorConfig
    workload_build_files = [f for f in os.listdir(folder_path) \
            if os.path.isfile(os.path.join(folder_path, f)) and \
            re.search("workload_[0-9]+_build_results.pkl", f)]
    build_file = os.path.join(work_dir_path, "build_results.pkl")
    if os.path.exists(build_file):
        workload_build_files.append("build_results.pkl")
    for build_file in workload_build_files:
        if build_file != "workload_7_build_results.pkl":
            continue
        with open(os.path.join(work_dir_path, build_file), 'rb') as f:
            runner_inputs_arr = pickle.load(f)
            print(f.name)
        runner_inputs_arr = [runner.RunnerInput(ri.artifact_path, ri.device_type, 
                                [arg_info.ArgInfo.from_json(args) for args in args_info_json])
                            for (ri, args_info_json) in runner_inputs_arr]
        evaluator_config = EvaluatorConfig(number = 1, repeat=1)
        local_runner = runner.create("local", evaluator_config=evaluator_config, device_id = device_id)
        runner_feature = local_runner.run(runner_inputs_arr)
        runner_result = [r.result().run_secs for r in runner_feature]
        print(runner_result)
        print(len(runner_result))


if __name__ == '__main__':
    # folder_path = "saved_work_dir/((bert_large,[(1,64)]),cuda)_ms_workdir/"
    # folder_path = "saved_work_dir/((bert_base,[(1,128)]),cuda)_ms_workdir/"
    folder_path = "saved_work_dir/((bert_large,[(1,256)]),cuda)_ms_workdir/"
    # folder_path = "saved_work_dir/feature_extractor_test"
    # load_tuning_records_and_save_features(folder_path)
    load_feature_to_train(folder_path, 
                          state_to_load="saved_work_dir/((bert_large,[(1,256)]),cuda)_ms_workdir/mlp_model_state",
                          target="nvidia/geforce-rtx-3090")
    # load_tuning_records_and_train(folder_path)
    # logger.setLevel(logging.DEBUG)
    # logger.warn("aaa")
    # load_tuning_records_to_runer_inputs("saved_work_dir/feature_extractor_test")
    # build_tuning_records_to_runer_inputs(folder_path, target="nvidia/geforce-rtx-3090")
    # load_runner_input_and_run(folder_path, device_id=1)
