import pickle
import logging
import os
import argparse

import tvm
from tvm import auto_scheduler, te, relay, IRModule
from tvm.target import Target
from tvm import meta_schedule as ms

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

NETWORK_INFO_FOLDER = "/home/xiachunwei/Dataset/dataset_tvm/dataset_gpu/network_info/"

parser = argparse.ArgumentParser(prog='',
                                 description='Load from tenset and tune',
                                 epilog='')
parser.add_argument(
    '--task',
    type=str,
    default="/home/xiachunwei/Dataset/dataset_tvm/dataset_gpu/network_info/((bert_large,[(1,128)]),cuda).task.pkl",
    help='path to the task with pkl format')
parser.add_argument('--workdir_name', type=str, default="saved_ms_workdir")
parser.add_argument('--num_tasks_to_tune',
                    type=int,
                    default=-1,
                    help='number of tasks to tune')
parser.add_argument('--max_trials_global',
                    type=int,
                    default=200,
                    help='argument for meta scheduler')
parser.add_argument('--max_trials_per_task',
                    type=int,
                    default=20,
                    help='argument for meta scheduler')
parser.add_argument('--num_trials_per_iter',
                    type=int,
                    default=20,
                    help='argument for meta scheduler')
parser.add_argument('--cost_model', type=str, default="mlp")


def load_and_register_tasks():
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


def load_ansor_tasks_and_tune_with_meta_scheduler(tasks_file,
                                                  num_tasks_to_tune=-1,
                                                  workdir_name="saved_ms_workdir",
                                                  max_trials_global=200,
                                                  max_trials_per_task=20,
                                                  num_trials_per_iter=20,
                                                  cost_model="mlp"):
    (ansor_tasks, _) = pickle.load(open(tasks_file, 'rb'))
    logging.info(ansor_tasks)
    logging.info(f"Load {len(ansor_tasks)} tasks")
    tune_functions, task_weights = [], []
    # target = Target("nvidia/nvidia-a100")
    target = Target("nvidia/geforce-rtx-3090")
    tasks_to_tune = ansor_tasks[
        2:] if num_tasks_to_tune < 0 else ansor_tasks[:num_tasks_to_tune]
    for t in tasks_to_tune:
        prim_func = te.create_prim_func(t.compute_dag.tensors)
        logging.info(prim_func.script())
        mod = IRModule(
            functions={"main": te.create_prim_func(t.compute_dag.tensors)})
        tune_functions.append(
            ms.TuneContext(mod=mod,
                           target=target,
                           space_generator="post-order-apply",
                           search_strategy="evolutionary",
                           task_name=t.workload_key))
        task_weights.append(1)
    logging.info(f"Create tune_tasks with {len(tune_functions)} functions")
    # return
    if not os.path.exists(workdir_name):
        os.mkdir(workdir_name)
    database = ms.tune_tasks(
        tasks=tune_functions,
        task_weights=task_weights,
        work_dir=workdir_name,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder="local",
        runner="local",
        database="json",
        cost_model=cost_model,
        measure_callbacks="default",
        task_scheduler="gradient",
        module_equality="structural",
    )


if __name__ == "__main__":
    # load_ansor_tasks_and_tune_with_meta_scheduler(NETWORK_INFO_FOLDER+"all_tasks.pkl", 2)
    args = parser.parse_args()
    load_ansor_tasks_and_tune_with_meta_scheduler(
        args.task,
        num_tasks_to_tune=args.num_tasks_to_tune,
        workdir_name=os.path.join(args.work_dir, os.path.splitext(os.path.basename(args.task))[0]),
        max_trials_global=args.max_trials_global,
        max_trials_per_task=args.max_trials_per_task,
        num_trials_per_iter=args.num_trials_per_iter,
        cost_model=args.cost_model)
    