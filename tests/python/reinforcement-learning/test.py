import argparse

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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.task)
    print(args.workdir_name)
    print(args.num_tasks_to_tune)
    print(args.max_trials_global)
    print(args.max_trials_per_task)
    print(args.num_trials_per_iter)
    print(args.cost_model)
