import pickle
import logging

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


def load_ansor_tasks_and_tune_with_meta_scheduler(tasks_file, num_tasks_to_tune=-1):
    ansor_tasks = pickle.load(open(tasks_file, 'rb'))
    logging.info(f"Load {len(ansor_tasks)} tasks")
    tune_functions, task_weights = [], []
    target = Target("nvidia/nvidia-a100")
    for t in ansor_tasks[:num_tasks_to_tune]:
        prim_func = te.create_prim_func(t.compute_dag.tensors)
        print(prim_func.script())
        mod = IRModule(functions={"main": te.create_prim_func(t.compute_dag.tensors)})
        tune_functions.append(ms.TuneContext(mod=mod, target=target, 
                                             space_generator="post-order-apply", search_strategy="evolutionary",task_name=t.workload_key))
        task_weights.append(1)
    logging.info(f"Create tune_tasks with {len(tune_functions)} functions")

    database = ms.tune_tasks(
        tasks=tune_functions,
        task_weights=task_weights,
        work_dir="tmp-ms-workdir",
        max_trials_global=200,
        max_trials_per_task=200,
        num_trials_per_iter=200,
        builder="local",
        runner="local",
        database="json",
        cost_model="xgb",
        measure_callbacks="default",
        task_scheduler="gradient",
        module_equality="structural",
    )


if __name__=="__main__":
    load_ansor_tasks_and_tune_with_meta_scheduler(NETWORK_INFO_FOLDER+"all_tasks.pkl", 2)
