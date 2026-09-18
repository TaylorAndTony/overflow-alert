## 基于时间序列和 LightGBM 的溢流预警模型

注：

- `WELL_000006` 内文件名错误，打成了 `WELL_000009`, 需要手动更正为 `WELL_000006`；
- `WELL_000007` 内 `WELL_00007时序数据` 少了一个0, 需要更正为 `WELL_000007时序数据`
- `WELL_000007` 内 `WELL_00007溢流基本信息.xlsx` 少了一个0, 需要更正为 `WELL_000007溢流基本信息.xlsx`

## 方法简介


### 模型架构

模型基于时间序列和 LightGBM 构建。将所有时序数据读入后，滑动窗口切分正负样本。

### 特征工程


滑动窗口切分正负样本后，在此基础上，计算出关键列的最大、最小、标准差等统计特征。随后将所有特征合并为一行，再次计算斜率等统计特征。完成一口井到多个 csv，每个 csv 到一行特征的转换。

### 训练策略

模型训练时，使用 LightGBM 训练。一行表示一个时间片样本，多行为一口井的多个样本，送入模型训练。

### 预测逻辑

将test内的csv读入，对 test 文件夹内的 csv 进行预测，并保存为 `result.csv` 文件。

## 运行环境

https://pytorch.org/get-started/locally/

Python 3.14

依赖使用 [uv](https://uv.oaix.tech/getting-started/installation/) 管理。如果使用 uv，可一键开始运行：

`uv run predict.py --data_dir ./data --output ./result.csv`

如果不使用 uv，请手动安装依赖：

`pip install -r requirements.txt`


## 运行命令

直接运行，利用已经训练好的模型（位于 ./model/ ）进行预测。 **请把 ./data 替换为实际数据集路径, data 文件夹内为 csv 文件 **

`python predict.py --data_dir ./data --output ./result.csv`

或者使用 uv

`uv run predict.py --data_dir ./data --output ./result.csv`


随后会生成 `result.csv` 文件，即为竞赛提交的预测结果。



## 文件说明

标注核心脚本、模型权重、特征处理模块用途

- `predict.py`: 预测脚本
- `model/`: 模型权重
- `src`: 如需自行训练，需要利用 src 目录下的文件代码进行训练。

其中 src 目录下有 3 个重要文件：

### `B0_wash.py`

数据处理模块

1. 读取所有时序 xls 文件，按每一口井，合并成一井一个的 csv 时序文件供后续处理。
2. 读取溢流基本信息.xlsx，解析里面的数据
3. 滑动窗口构建正负样本。
4. 构建特征集，形成此项目使用的 train_v3.csv + test_v3.csv

```
原始 xls/csv → 拼接
 → 按溢流时间切正负样本窗口
 → 保存为独立 CSV
            ↓
逐文件 build_feature_dataset（时序特征）
            ↓
逐文件 build_one_row_features（压缩为单行）
            ↓
拼接所有样本 → train_v3.csv + test_v3.csv
```

### `B1_train.py`

训练模块，使用 LightGBM 训练。模型保存到 `models/v3.pkl`，列名保存到 `models/v3_features.pkl`，并打印分类报告

### `B2_predict.py`

预测模块，使用训练好的模型进行预测，并保存为 csv 文件。


## 训练方法

### 修改 src/B0_wash.py 里面的：

```python
TRAIN_DIR = r"D:\.datasets\oil_data\train"
```

到数据集所在位置，例如：

```python
TRAIN_DIR = r"D:\数据集\train"
```

其中 train 文件夹与竞赛下发数据集内的 train 大部分相同，除了做出如下修改：

- `WELL_000006` 内文件名错误，打成了 `WELL_000009`, 需要手动更正为 `WELL_000006`；
- `WELL_000007` 内 `WELL_00007时序数据` 少了一个0, 需要更正为 `WELL_000007时序数据`
- `WELL_000007` 内 `WELL_00007溢流基本信息.xlsx` 少了一个0, 需要更正为 `WELL_000007溢流基本信息.xlsx`


### 修改 src/B2_predict.py 里面的：

```python
TEST_DIR = Path(r"D:\.datasets\oil_data\test\data")
```

到数据集所在位置，例如：

```python
TEST_DIR = Path(r"D:\数据集\test\data")
```

data 文件夹与竞赛下发数据集内的 test\data 完全一致，是 csv 文件。

### 随后在此项目目录下，依次运行：

1. `python src/B0_wash.py`
2. 脚本提示 `是否生成正负样本 csv 文件? 初次运行请输入 y (y/n) ` 请输入 y
3. `python src/B1_train.py`
4. `python src/B2_predict.py`

即可完成训练+预测。


## others

`uv export --format requirements-txt -o requirements.txt`

`uv pip install torch torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu130`