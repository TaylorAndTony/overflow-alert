"""
从所有井数据生成机器学习样本

正样本:
    溢流事件附近滑动窗口扩增

负样本:
    正常时间随机窗口

输出:
dataset/
 ├── samples/
 │      WELL_xxx_pos_001.csv
 │      WELL_xxx_neg_001.csv
 │
 └── metadata.csv

"""

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# =========================
# path
# =========================

TRAIN_DIR = Path(r"D:\.datasets\oil_data\train")


SAVE_DIR = Path("dataset/samples")


# =========================
# 参数
# =========================

WINDOW_SECONDS = 1800  # 30分钟窗口

# 正样本滑动步长
POS_STRIDE_SECONDS = 300  # 5分钟


NEG_NUM = 10  # 每口井负样本数量


# =========================
# 合并xls
# =========================


def concat_excel(paths):

    dfs = []

    for p in paths:
        print("reading", p.name)

        dfs.append(pd.read_excel(p))

    return pd.concat(dfs, ignore_index=True)


def concat_dataset(well_folder):

    well_folder = Path(well_folder)

    cache = well_folder / "data.csv"

    if cache.exists():
        print("use cache", well_folder.name)

        return pd.read_csv(cache)

    data_dir = well_folder / f"{well_folder.name}时序数据"

    files = sorted(data_dir.glob("*.xls"))

    df = concat_excel(files)

    df.to_csv(cache, index=False)

    return df


# =========================
# 读取事件信息
# =========================


def read_metrics(well_folder):

    well_folder = Path(well_folder)

    file = well_folder / f"{well_folder.name}溢流基本信息.xlsx"

    return pd.read_excel(file)


# =========================
# 时间处理
# =========================


def prepare_time(df):

    # 第一列时间是真实时间

    df["timestamp"] = pd.to_datetime(df["时间"])

    df = df.sort_values("timestamp")

    df = df.reset_index(drop=True)

    return df


# =========================
# 正样本扩增
# =========================


def generate_positive_samples(df, overflow_time):

    samples = []

    window = pd.Timedelta(seconds=WINDOW_SECONDS)

    stride = pd.Timedelta(seconds=POS_STRIDE_SECONDS)

    # 生成范围
    #
    # Te-60min
    #       ...
    # Te
    #

    start_time = overflow_time - window - pd.Timedelta(minutes=60)

    end_time = overflow_time

    t = start_time

    idx = 0

    while t <= end_time:
        end = t + window

        # 必须包含溢流附近

        if end >= overflow_time:
            sample = df[(df.timestamp >= t) & (df.timestamp <= end)].copy()

            if len(sample) > 100:
                sample["label"] = 1

                sample["sample_id"] = f"pos_{idx}"

                samples.append(sample)

                idx += 1

        t += stride

    return samples


# =========================
# 负样本
# =========================


def generate_negative_samples(df, overflow_time):

    samples = []

    window = pd.Timedelta(seconds=WINDOW_SECONDS)

    # 事件前1小时以前
    candidates = df[df.timestamp < overflow_time - pd.Timedelta(hours=1)]["timestamp"]

    used = set()

    idx = 0

    while len(samples) < NEG_NUM:
        end = random.choice(candidates)

        if end in used:
            continue

        used.add(end)

        start = end - window

        sample = df[(df.timestamp >= start) & (df.timestamp <= end)].copy()

        if len(sample) < 100:
            continue

        sample["label"] = 0

        sample["sample_id"] = f"neg_{idx}"

        samples.append(sample)

        idx += 1

    return samples


# =========================
# 单井
# =========================


def build_samples(well_folder):

    df = concat_dataset(well_folder)

    info = read_metrics(well_folder)

    df = prepare_time(df)

    overflow_time = pd.to_datetime(info["溢流发生时间"].iloc[0])

    positive = generate_positive_samples(df, overflow_time)

    negative = generate_negative_samples(df, overflow_time)

    return positive + negative


# =========================
# 全部井
# =========================


def build_all_wells():

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []

    for well_folder in tqdm(TRAIN_DIR.glob("WELL_*")):
        print("\nprocess", well_folder.name)

        try:
            samples = build_samples(well_folder)

        except Exception as e:
            print("skip", e)

            continue

        pos_id = 0
        neg_id = 0

        for sample in samples:
            label = sample["label"].iloc[0]

            if label == 1:
                name = f"{well_folder.name}_pos_{pos_id}"

                pos_id += 1

            else:
                name = f"{well_folder.name}_neg_{neg_id}"

                neg_id += 1

            path = SAVE_DIR / f"{name}.csv"

            sample.to_csv(path, index=False)

            metadata.append(
                {
                    "sample_id": name,
                    "well_id": well_folder.name,
                    "label": label,
                    "rows": len(sample),
                    "start": sample.timestamp.min(),
                    "end": sample.timestamp.max(),
                }
            )

    meta = pd.DataFrame(metadata)

    meta.to_csv("dataset/metadata.csv", index=False)

    print(meta.head())

    print(meta.label.value_counts())


if __name__ == "__main__":
    build_all_wells()
