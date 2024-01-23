## 张量程序的成本模型



### 如何从源代码构建
本项目基于TVM构建。
首先，下载这个仓库.
```shell
git clone https://github.com/summerspringwei/tvm_oraa.git
git checkout RL-scheduler
```
之后，请参阅[从源代码安装](https://tvm.apache.org/docs/install/from_source.html)从源码编译安装。

### 简介

我们基于TVM实现了一个改进的成本模型。
以下图显示了TVM自动调整的工作流程。

![元调度器](readme_figures/tvm-meta-scheduler-workflow.png)

### 动机
我们发现当前成本模型的预测准确性较低。
首先，我们测试了XGBoost模型的根均方误差（RMSE）。
![元调度器](readme_figures/tvm-rmse-error-xgboost.png)
横轴表示训练迭代次数，纵轴表示RMSE。
蓝线表示在预训练模型上的预测，而黄线表示从头开始训练的模型。
我们可以看到预训练模型和从头训练模型之间的差异很小。

我们还测试了MLP模型的预测准确性。
纵轴显示了对64个测试样本的`top-32`准确性。
左图显示了MLP从头开始模型的准确性，而右图显示了预训练模型的准确性。
准确性差异也很小（$0.71$ vs $0.74$）。
![元调度器](readme_figures/tvm-mlp-top32-acc.png)


### 特征

我们选择以下特征：
* 设备属性
    * SM的数量
    * 内存带宽
    * L2缓存大小
* 内存访问
    * 加载/存储的数量
    * 延迟
    * 带宽
* On-chip缓冲区访问
    * 共享内存加载/存储的数量
    * 缓存未命中比例
* 启动维度
    * 启动块和线程的数量

详细说明请参考`mcu_metrics_utils.py:preprocessing_func_mapping`。


### 整体工作流程

首先，我们从调整记录（调整记录是一系列调度）中收集特征。
调整记录将被降级到`tir`，然后编译为内核。
其次，我们在硬件上运行内核，并从配置文件中收集特征。
最后，我们根据配置文件上的训练数据训练成本模型并验证准确性。

### 1. 准备数据集

下载张量数据集：
```shell
pip3 install gdown
gdown https://drive.google.com/uc?id=1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK
unzip dataset_gpu_v3.3.zip
```

请参考`get_started_with_cost_model_experiments`获取详细说明。

### 2. 获取调优记录
运行以下命令：

```shell
python3 tenset_load_datasets_and_tune.py \
    --task "path/to/dataset_gpu/network_info/((bert_base,[(1,128)]),cuda).task.pkl"
```
如果要更改调整试验的次数或成本模型，请运行```python3 tenset_load_datasets_and_tune.py --help```
查看详细参数。

### 3. 运行调整记录并收集配置文件
运行

```shell
ncu --target-processes all \
    --clock-control none --set full \
    --csv -o bert_base_1_128_workload_0-ncu \
    python3 load_database_and_train_cost_model.py
```
这将运行CUDA内核并收集特征。

### 4. 比较成本模型的准确性
运行

```shell
python3 train_ncu_profile.py
```
最后，我们可以得到输出：
元调度器
![meta-scheduler](readme_figures/tvm-meta-scheduler-ncu-accuracy.png)