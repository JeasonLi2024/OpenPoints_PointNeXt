# ModelNet40 PointNeXt 点云分类

本项目使用 PointNeXt 官方仓库及其 OpenPoints 子模块，实现 ModelNet40
三维点云分类。数据中每个点为：

```text
x, y, z, nx, ny, nz
```

- `x, y, z`：三维坐标；
- `nx, ny, nz`：对应点的法向量。

默认实验同时使用坐标和法向量，也提供仅使用坐标的对照配置。Windows
目录用于代码准备和数据检查，正式训练应在 Linux NVIDIA GPU 服务器上完成。

## 代码来源

- PointNeXt 主仓库提交：`8a0ec91de5329b3b3e346422fa43a51ba24fd450`
- OpenPoints 提交：`baeca5e319aa2e756d179e494469eb7f5ffd29f0`
- 上游原始说明：[README_UPSTREAM.md](README_UPSTREAM.md)

模型主干、分类头、FPS、ball query、局部聚合和 CUDA 算子均使用官方实现。
本项目只增加本地数据适配、任务配置、运行脚本和文档，并修复官方单 GPU
训练结束时无条件销毁进程组的问题。

## 项目结构

```text
OpenPoints_PointNeXt/
├── cfgs/modelnet40_normals/
│   ├── default.yaml
│   ├── pointnext-s_c64.yaml
│   └── pointnext-s_c64_xyz.yaml
├── examples/classification/          # 官方分类训练入口
├── openpoints/                       # 官方 OpenPoints 源码
├── modelnet40_train_data/
│   └── modelnet40_normal_resampled/
├── modelnet40_test_data/
│   └── modelnet40_normal_resampled/test/
├── scripts/
├── tools/check_dataset.py
├── requirements-modelnet40.txt
└── README.md
```

数据、日志、权重、编译产物和两个临时官方仓库均已加入 `.gitignore`。
`.gitattributes` 会强制 Linux 脚本和源码使用 LF 行尾，避免 Windows 开发后上传
服务器出现 `/usr/bin/env: bash\r` 错误。

## 数据划分

当前数据检查结果应为：

| 划分 | 样本数 | 用途 |
| --- | ---: | --- |
| 原始训练集 | 9843 | 分层划分 train/val |
| train | 8365 | 参数训练 |
| val | 1478 | 选择最佳 checkpoint |
| test | 2468 | 最终测试 |

验证集按类别从原训练集固定划分，比例为 15%，`split_seed=42`。测试集不参与
模型选择，避免把测试集同时作为验证集使用。

快速检查目录、样本数量和每类代表文件：

```bash
python tools/check_dataset.py
```

首次上传服务器后建议做一次全量格式检查：

```bash
python tools/check_dataset.py --full
```

每个点文件必须是逗号分隔的六列浮点数，至少包含 1024 个点，且不能包含
`NaN` 或无穷值。

## 预处理与增强

数据适配器位于
`openpoints/dataset/modelnet/modelnet40_separate_normals.py`，处理流程如下：

1. xyz 减去质心；
2. xyz 按最大半径缩放到单位球；
3. normals 逐点重新归一化；
4. 训练集随机打乱点顺序；
5. 官方分类训练入口先 FPS 到 1200 点，再随机选择 1024 点；
6. 验证和测试固定使用 1024 点；
7. 训练时执行随机缩放和平移。

默认使用各向同性缩放，使法向量方向与几何变换保持一致。ModelNet40
通常是方向对齐数据，官方配置注明旋转增强无收益，因此基准配置不启用随机旋转。
如需加入旋转，必须对 xyz 和 normals 使用同一个旋转矩阵。

## 模型

默认配置 `pointnext-s_c64.yaml` 使用官方 `PointNextEncoder`：

- PointNeXt-S，width 64；
- blocks：`[1, 1, 1, 1, 1, 1]`；
- strides：`[1, 2, 2, 2, 2, 1]`；
- ball query，`nsample=32`；
- `dp_fj` 局部特征，max 聚合；
- 输入通道 6：`[x, y, z, nx, ny, nz]`；
- 输出类别数 40。

`pointnext-s_c64_xyz.yaml` 使用相同网络，但输入通道为 3，仅使用 xyz，可用于
比较法向量带来的收益。

训练超参数沿用官方 ModelNet40 recipe：

- 1024 点；
- AdamW，学习率 `0.001`；
- weight decay `0.05`；
- cosine scheduler；
- 600 epochs；
- label smoothing `0.2`；
- gradient clipping `1`；
- 默认随机种子 `42`。

## Linux GPU 环境

本项目默认安装方案面向当前服务器环境：

- Linux x86_64；
- Ubuntu 24.04；
- Python 3.11；
- PyTorch 2.8.0；
- CUDA 12.9；
- RTX 4090，计算能力 8.9；
- 与 CUDA 对应的 GCC/G++。

创建独立环境并安装 PyTorch 官方 CUDA 12.9 wheel：

```bash
conda create -n pointnext-cu129 python=3.11 -y
conda activate pointnext-cu129

python -m pip install --upgrade pip
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

验证 PyTorch、CUDA Toolkit 与 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
nvcc --version
```

两处 CUDA 版本都应为 `12.9`。然后在项目根目录执行：

```bash
bash scripts/setup_linux_modelnet40.sh
```

该脚本会检查 CUDA PyTorch、安装本任务依赖、编译
`openpoints/cpp/pointnet2_batch` CUDA 扩展，并执行数据快速检查。脚本默认设置
`TORCH_CUDA_ARCH_LIST=8.9`，只为 RTX 4090 编译目标架构。

旧版官方复现环境的依赖保存在 `requirements-modelnet40-legacy.txt`，但
PyTorch 1.10.1/CUDA 11.3 不能为 RTX 4090 原生生成 `sm_89` 代码，不建议在当前
服务器上使用。

先确认以下版本彼此兼容：

```bash
nvidia-smi
nvcc --version
gcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 训练

默认坐标加法向量，单 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_modelnet40_normals.sh
```

仅坐标基线：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_modelnet40_xyz.sh
```

命令行可覆盖 YAML 参数：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_modelnet40_normals.sh \
  batch_size=16 \
  val_batch_size=32 \
  dataloader.num_workers=4 \
  seed=42
```

默认 `batch_size=32`。显存不足时先降到 16 或 8。OpenPoints 会把所有
`CUDA_VISIBLE_DEVICES` 中可见的 GPU 用于多进程训练；多 GPU 时
`batch_size` 是每个进程的数据批大小。

运行目录、完整合并配置、TensorBoard 日志和 checkpoint 会写入 `log/`
下的实验目录。实际路径会在启动日志中打印。最佳模型按验证集 OA 保存。

## 测试

测试坐标加法向量模型：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_modelnet40_normals.sh \
  /path/to/best_checkpoint.pth
```

测试仅坐标模型：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_modelnet40_xyz.sh \
  /path/to/best_checkpoint.pth
```

日志中的 `OA` 是整体实例准确率，`mAcc` 是 40 类准确率的宏平均。最终报告建议
同时记录：

- OA、mAcc 和每类准确率；
- GPU 型号；
- Python、PyTorch、CUDA、GCC 版本；
- seed、batch size 和实际合并后的配置；
- 使用的 checkpoint 路径。

## 常见问题

### CUDA 扩展无法导入

确认当前环境中执行过 `bash scripts/setup_linux_modelnet40.sh`，并检查编译时
使用的 PyTorch/CUDA 与运行时环境一致。切换 PyTorch 或 CUDA 后必须重新编译。

如果出现以下错误：

```text
The detected CUDA version (...) mismatches the version that was used to compile
PyTorch (...)
```

其中前者是 `CUDA_HOME` 下 `nvcc` 的版本，后者是 `torch.version.cuda`。两者的
主版本和次版本必须一致。例如 PyTorch 为 CUDA 11.3 时：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
which nvcc
nvcc --version
echo "${CUDA_HOME}"
```

服务器同时安装多个 CUDA Toolkit 时，应切换到 11.3 后重新编译：

```bash
export CUDA_HOME=/usr/local/cuda-11.3
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

rm -rf openpoints/cpp/pointnet2_batch/build
python -m pip uninstall -y pointnet2-cuda
bash scripts/setup_linux_modelnet40.sh
```

如果服务器没有 `/usr/local/cuda-11.3`，需要由管理员安装对应 Toolkit，或改装
一个与服务器现有 Toolkit 匹配的 CUDA 版 PyTorch。仅安装 Conda
`cudatoolkit` 运行库通常不会提供编译扩展所需的 `nvcc`。

注意：`nvidia-smi` 显示的是驱动最高支持的 CUDA 版本，不等同于当前用于编译的
CUDA Toolkit 版本。判断编译版本应以 `nvcc --version` 和 `CUDA_HOME` 为准。

### 显存不足

降低 `batch_size` 和 `val_batch_size`。源文件包含的点数可能明显大于 1024，
训练入口会在 GPU 上完成 FPS，因此数据批在采样前仍占用一定显存。

### 修改了数据路径

覆盖配置即可，不需要改源码：

```bash
bash scripts/train_modelnet40_normals.sh \
  dataset.common.train_data_dir=/data/modelnet40_train_data/modelnet40_normal_resampled \
  dataset.common.test_data_dir=/data/modelnet40_test_data/modelnet40_normal_resampled/test
```

### 继续训练

使用官方 `mode=resume` 和 checkpoint：

```bash
bash scripts/train_modelnet40_normals.sh \
  mode=resume \
  pretrained_path=/path/to/checkpoint.pth
```

## 开发约束

- 数据格式适配保持在 dataset 类中，不修改 PointNeXt 主干网络；
- 新实验新增 YAML，不覆盖基准配置；
- 测试集只用于最终评估；
- 不提交数据、日志、checkpoint、虚拟环境和 CUDA 编译产物；
- Linux 上运行脚本时始终从项目根目录启动。

## 参考

- [PointNeXt 论文](https://arxiv.org/abs/2206.04670)
- [PointNeXt 官方仓库](https://github.com/guochengqian/PointNeXt)
- [OpenPoints](https://github.com/guochengqian/openpoints)
