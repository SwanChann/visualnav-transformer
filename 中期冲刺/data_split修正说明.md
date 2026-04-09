# data_split.py 命令修正说明

## 问题发现

在 NoMaD_复现指南.md 的 2.2 节"运行数据分割脚本"中，命令参数有误。

## 错误内容

**原来的错误命令**：
```bash
python data_split.py \
  --data-folder ~/codespace/visualnav-transformer/nomad_dataset/go_stanford \
  --train-split 0.8 \
  --plot-dataset
```

**问题**：
1. ❌ `--data-folder` → 实际参数名是 `--data-dir`
2. ❌ `--train-split` → 实际参数名是 `--split`
3. ❌ `--plot-dataset` → 此参数不存在
4. ❌ 缺少必需参数 `--dataset-name`

## 正确的参数

根据 `train/data_split.py` 源代码，正确的参数应该是：

```python
parser.add_argument("--data-dir", "-i", help="Directory containing the data", required=True)
parser.add_argument("--dataset-name", "-d", help="Name of the dataset", required=True)
parser.add_argument("--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)")
parser.add_argument("--data-splits-dir", "-o", default="vint_train/data/data_splits", help="Data splits directory")
```

### 参数说明

| 参数 | 简写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data-dir` | `-i` | ✅ | 无 | 包含轨迹文件夹的数据目录 |
| `--dataset-name` | `-d` | ✅ | 无 | 数据集名称（用于创建子目录） |
| `--split` | `-s` | ❌ | 0.8 | 训练集比例（0-1之间的浮点数） |
| `--data-splits-dir` | `-o` | ❌ | `vint_train/data/data_splits` | 分割文件输出目录 |

## 修正后的正确命令

### GoStanford 数据集示例

```bash
# 进入训练目录
cd ~/visualnav-transformer/train

# 激活环境
conda activate nomad_train

# 运行数据分割（80%训练，20%测试）
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.8

# 或使用默认分割比例（0.8）
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford
```

### 调整分割比例

```bash
# 90%训练, 10%测试
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.9

# 70%训练, 30%测试
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.7
```

### 其他数据集示例

```bash
# TartanDrive
python data_split.py \
    --data-dir ~/nomad_dataset/tartan_drive \
    --dataset-name tartan_drive \
    --split 0.9

# RECON
python data_split.py \
    --data-dir ~/nomad_dataset/recon \
    --dataset-name recon \
    --split 0.9

# SCAND
python data_split.py \
    --data-dir ~/nomad_dataset/scand \
    --dataset-name scand \
    --split 0.9
```

## 输出文件位置

运行成功后，分割文件会保存在：

```
train/vint_train/data/data_splits/
└── go_stanford/              # --dataset-name 指定的名称
    ├── train/
    │   └── traj_names.txt   # 训练集轨迹列表
    └── test/
        └── traj_names.txt   # 测试集轨迹列表
```

## 验证命令

```bash
# 查看生成的目录结构
ls ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/

# 查看训练集轨迹数量
wc -l ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/traj_names.txt

# 查看测试集轨迹数量
wc -l ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/test/traj_names.txt

# 查看训练集内容（前10条）
head -n 10 ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/traj_names.txt
```

## 已修正的文档位置

以下章节已在 `NoMaD_复现指南.md` 中修正：

1. ✅ **第二步：创建数据分割 → 2.2 运行数据分割脚本**
2. ✅ **第二步：创建数据分割 → 2.3 验证分割结果**
3. ✅ **第二步：创建数据分割 → 2.4 调整分割比例**
4. ✅ **第三步：创建数据分割 → 3.2 执行数据分割**
5. ✅ **第三步：创建数据分割 → 3.5 对所有数据集创建分割**
6. ✅ **第三步：创建数据分割 → 3.7 调整分割比例**
7. ✅ **完整工作流 → 步骤4: 创建数据分割**
8. ✅ **进阶使用 → 自定义数据集训练**

## 常见错误和解决方案

### 错误1：参数名错误

```bash
# ❌ 错误
python data_split.py --data-folder ~/data --train-split 0.8

# 报错：error: unrecognized arguments: --data-folder --train-split

# ✅ 正确
python data_split.py --data-dir ~/data --dataset-name my_dataset --split 0.8
```

### 错误2：缺少必需参数

```bash
# ❌ 错误
python data_split.py --data-dir ~/data --split 0.8

# 报错：error: the following arguments are required: --dataset-name

# ✅ 正确
python data_split.py --data-dir ~/data --dataset-name my_dataset --split 0.8
```

### 错误3：路径错误

```bash
# ❌ 错误（路径不存在）
python data_split.py \
  --data-dir ~/nomad_dataset/go_stanford \
  --dataset-name go_stanford

# 报错：[Errno 2] No such file or directory

# ✅ 正确（确保路径存在）
ls ~/nomad_dataset/go_stanford  # 先检查路径
python data_split.py \
  --data-dir ~/nomad_dataset/go_stanford \
  --dataset-name go_stanford
```

## 完整示例：从下载到分割

```bash
# 1. 下载数据集（假设已处理好）
cd ~/visualnav-transformer/nomad_dataset

# 2. 验证数据存在
ls go_stanford/
# 应该看到轨迹文件夹：no10vcF_10_0/ no10vcF_11_0/ ...

# 3. 进入训练目录
cd ~/visualnav-transformer/train

# 4. 运行数据分割
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.8

# 5. 验证输出
ls vint_train/data/data_splits/go_stanford/train/
ls vint_train/data/data_splits/go_stanford/test/

# 6. 查看分割结果
echo "训练集轨迹数："
wc -l vint_train/data/data_splits/go_stanford/train/traj_names.txt

echo "测试集轨迹数："
wc -l vint_train/data/data_splits/go_stanford/test/traj_names.txt
```

## 总结

所有相关的 `data_split.py` 命令已在指南中修正为正确的参数：

- `--data-dir`（而非 `--data-folder`）
- `--dataset-name`（必需参数）
- `--split`（而非 `--train-split`）
- 移除了不存在的 `--plot-dataset` 参数

现在可以正确执行数据分割步骤了！
