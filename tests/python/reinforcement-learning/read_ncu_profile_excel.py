import logging
from builtins import len
from typing import List, Tuple

import pandas as pd
import numpy as np

import ncu_metrics_to_train

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def metric_to_col_idx(file_path: str, metrics_list: List[str]) -> List[Tuple[str, int]]:
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
  fmt = file_path.split(".")[-1]
  if fmt != "xlsx":
    logging.error(f"Only support xlsx file, but give {fmt}")
  read_data_frame = pd.read_excel(file_path, engine='openpyxl')
  print(read_data_frame.info())
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
  print(new_metrics_list)
  print(len(new_metrics_list))
  extracted_data_frame = read_data_frame.loc[:, new_metrics_list]
  print(extracted_data_frame)
  return extracted_data_frame


def gather_metrics():
  metrics_list = []
  all_sections = [ncu_metrics_to_train.device_attrs, ncu_metrics_to_train.launch_metrics,
                  ncu_metrics_to_train.dram_attrs, ncu_metrics_to_train.memory_section_attrs,
                  ncu_metrics_to_train.compute_inst_attrs, ncu_metrics_to_train.source_metrics]
  for sec in all_sections:
    metrics_list.extend(sec)
  
  return metrics_list


def self_normalize(data: np.ndarray):
  if (data.max() - data.min()) == 0:
    return np.zeros(data.shape, data.dtype)
  return (data-data.min()) / (data.max() - data.min())


def normalize_to(data: np.ndarray, mother_data: np.ndarray):
  return data / mother_data


def minimize(concated_ndarry_list):
  return np.min(concated_ndarry_list, axis=1)


def identity(data: np.ndarray):
  return data


def prepare_profile_data(extracted_data_frame: pd.DataFrame):
  preprocessing_mapping = {
    "launch__grid_size": (normalize_to, ("device__attribute_multiprocessor_count", )),
    "launch__block_size": (normalize_to, (256, )),
    # limit blocks
    "launch__waves_per_multiprocessor": (self_normalize, ),
    "dram__bytes.sum.peak_sustained [byte/cycle]": (self_normalize, ),
    "dram__bytes.sum.per_second [Gbyte/second]": (self_normalize, ),
    "dram__bytes_read.sum [Mbyte]": (self_normalize, ), 
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]": (normalize_to, (100, )),
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed [%]": (normalize_to, (100, )),
    "dram__cycles_elapsed.avg.per_second [cycle/nsecond]": (self_normalize, ),
    "dram__sectors_read.sum [sector]": (self_normalize, ),
    "dram__sectors_write.sum [sector]": (self_normalize, ),

    "sass__inst_executed_shared_loads [inst]": (self_normalize, ),
    "sass__inst_executed_shared_stores [inst]": (self_normalize, ),
    "smsp__inst_executed_op_ldsm.sum [inst]": (self_normalize, ),
    "smsp__inst_executed_op_shared_atom.sum [inst]": (self_normalize, ),
    "smsp__inst_executed_op_global_red.sum [inst]": (self_normalize, ),
    "smsp__inst_executed_op_ldsm.sum.pct_of_peak_sustained_elapsed [%]": (normalize_to, (100, )),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": (self_normalize, ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum": (self_normalize, ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": (self_normalize, ),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": (self_normalize, ),

    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum [request]": (self_normalize, ),
    "sass__inst_executed_global_loads [inst]": (self_normalize, ),
    "sass__inst_executed_global_stores [inst]": (self_normalize, ),
    "sm__sass_inst_executed_op_ldgsts_cache_bypass.sum [inst]": (self_normalize, ),
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum [sector]": (self_normalize, ),
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum [sector]": (self_normalize, ),
    "smsp__sass_inst_executed_op_memory_128b.sum [inst]": (self_normalize, ), 
    "smsp__sass_inst_executed_op_memory_16b.sum [inst]": (self_normalize, ), 
    "smsp__sass_inst_executed_op_memory_32b.sum [inst]": (self_normalize, ), 
    "smsp__sass_inst_executed_op_memory_64b.sum [inst]": (self_normalize, ), 

    "smsp__inst_executed_op_branch.sum [inst]": (self_normalize, ),
    "sm__inst_executed_pipe_cbu_pred_on_any.avg.pct_of_peak_sustained_elapsed [%]": (normalize_to, (100, )),
    "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active [%]": (normalize_to, (100, )),
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active [%]": (normalize_to, (100, )),
    "sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed [%]": (normalize_to, (100, )),
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active [%]": (normalize_to, (100, )),
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed [%]": (normalize_to, (100, )),
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active [%]": (normalize_to, (100, )),
  }
  
  limited_blocks_all = extracted_data_frame.loc[:, [
    "launch__occupancy_limit_blocks [block]",
    "launch__occupancy_limit_registers [block]",
    "launch__occupancy_limit_shared_mem [block]",
    "launch__occupancy_limit_warps [block]",
  ]].to_numpy()
  
  limited_blocks = np.min(limited_blocks_all, axis=1)
  
  processed_ndarray = [limited_blocks]
  for metric, func_args_tuple in preprocessing_mapping.items():
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
  
  processed_ndarray = np.concatenate(processed_ndarray, axis=1)
  print(processed_ndarray)

  return processed_ndarray


if __name__=="__main__":
  xlsx_file_path = "saved_work_dir/feature_extractor_test/feature_extractor_test-ncu.xlsx"
  extracted_data_frame = metric_to_col_idx(xlsx_file_path, gather_metrics())
  prepare_profile_data(extracted_data_frame)
