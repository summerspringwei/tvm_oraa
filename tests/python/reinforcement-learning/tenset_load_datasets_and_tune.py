import pickle
import logging
import os

import tvm
from tvm import auto_scheduler, te, relay, IRModule
from tvm.target import Target
from tvm import meta_schedule as ms


FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

NETWORK_INFO_FOLDER="/home/xiachunwei/Dataset/dataset_tvm/dataset_gpu/network_info/"


def load_and_register_tasks():
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


def load_ansor_tasks_and_tune_with_meta_scheduler(tasks_file, num_tasks_to_tune=-1, workdir_name="ms_workdir"):
    (ansor_tasks, _) = pickle.load(open(tasks_file, 'rb'))
    logging.info(ansor_tasks)
    logging.info(f"Load {len(ansor_tasks)} tasks")
    tune_functions, task_weights = [], []
    # target = Target("nvidia/nvidia-a100")
    target = Target("nvidia/geforce-rtx-3090")
    tasks_to_tune = ansor_tasks[0:] if num_tasks_to_tune < 0 else ansor_tasks[:num_tasks_to_tune]
    for t in tasks_to_tune:
        prim_func = te.create_prim_func(t.compute_dag.tensors)
        logging.info(prim_func.script())
        mod = IRModule(functions={"main": te.create_prim_func(t.compute_dag.tensors)})
        tune_functions.append(ms.TuneContext(mod=mod, target=target, 
                                             space_generator="post-order-apply", search_strategy="evolutionary",task_name=t.workload_key))
        task_weights.append(1)
    logging.info(f"Create tune_tasks with {len(tune_functions)} functions")
    # return
    if not os.path.exists(workdir_name):
        os.mkdir(workdir_name)
    database = ms.tune_tasks(
        tasks=tune_functions,
        task_weights=task_weights,
        work_dir=workdir_name,
        max_trials_global=20000,
        max_trials_per_task=2000,
        num_trials_per_iter=2000,
        builder="local",
        runner="local",
        database="json",
        cost_model="mlp",
        measure_callbacks="default",
        task_scheduler="gradient",
        module_equality="structural",
    )


if __name__=="__main__":
    # load_ansor_tasks_and_tune_with_meta_scheduler(NETWORK_INFO_FOLDER+"all_tasks.pkl", 2)
    load_ansor_tasks_and_tune_with_meta_scheduler(NETWORK_INFO_FOLDER+"((bert_base,[(1,128)]),cuda).task.pkl",
                                                  num_tasks_to_tune=2,
                                                   workdir_name="((bert_base,[(1,128)]),cuda)_ms_workdir")
