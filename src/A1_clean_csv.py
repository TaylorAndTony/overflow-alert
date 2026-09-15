import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================
# Path
# ===============================

DATA_DIR = Path(
    r".\dataset\samples_v2"
)

OUTPUT = Path(
    r".\dataset\features_v2.csv"
)


# ===============================
# invalid values
# ===============================

INVALID_VALUES = [
    -999.25,
    -999,
    -9999,
    -9999.0
]


# ===============================
# parse filename
# ===============================

def parse_filename(path):

    name = path.stem

    # example:
    # WELL_000001_pos
    # WELL_000001_neg_1

    if "pos" in name:
        label = 1
    else:
        label = 0

    well = re.search(
        r"WELL_\d+",
        name
    ).group()

    return well, label


# ===============================
# load csv
# ===============================

def load_sample(path):

    df = pd.read_csv(
        path
    )

    well, label = parse_filename(path)

    return df, well, label


# ===============================
# clean
# ===============================

def clean_dataframe(df):

    df = df.copy()

    # invalid -> nan

    df.replace(
        INVALID_VALUES,
        np.nan,
        inplace=True
    )

    # 删除时间字段

    drop_cols = []

    for c in df.columns:

        if (
            "时间" in c
            or "日期" in c
            or c == "timestamp"
        ):
            drop_cols.append(c)

    df.drop(
        columns=drop_cols,
        inplace=True,
        errors="ignore"
    )

    # 全部转数字

    for c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    return df


# ===============================
# slope
# ===============================

def calc_slope(series):

    y = series.values

    mask = np.isfinite(y)

    if mask.sum() < 5:
        return np.nan

    x = np.arange(
        len(y)
    )

    k = np.polyfit(
        x[mask],
        y[mask],
        1
    )[0]

    return k


# ===============================
# feature extraction
# ===============================

def extract_features(
        df,
        prefix=""
):

    features = {}

    n = len(df)

    # 最后5分钟
    # 1Hz约300点
    # 3Hz约100点

    last_n = min(
        300,
        n//5
    )

    last_part = df.iloc[-last_n:]

    for col in df.columns:


        x = df[col]

        if x.notna().sum() == 0:
            continue

        # 全窗口统计

        if 'label' in col:
            continue
        
        features[f"{prefix}{col}_mean"] = x.mean()

        features[f"{prefix}{col}_std"] = x.std()

        features[f"{prefix}{col}_max"] = x.max()

        features[f"{prefix}{col}_min"] = x.min()

        features[f"{prefix}{col}_median"] = x.median()

        # 首尾变化

        features[f"{prefix}{col}_first"] = x.iloc[0]

        features[f"{prefix}{col}_last"] = x.iloc[-1]

        features[f"{prefix}{col}_delta"] = (
            x.iloc[-1]
            -
            x.iloc[0]
        )

        # 趋势

        features[f"{prefix}{col}_slope"] = (
            calc_slope(x)
        )

        # 最后5分钟变化

        features[f"{prefix}{col}_last5_delta"] = (
            last_part.iloc[-1][col]
            -
            last_part.iloc[0][col]
        )

    return features


# ===============================
# process all
# ===============================

def build_feature_dataset():

    records = []

    files = sorted(
        DATA_DIR.glob("*.csv")
    )

    print(
        "samples:",
        len(files)
    )

    for path in tqdm(files):

        df, well, label = load_sample(path)

        df = clean_dataframe(df)

        # interpolation

        df.interpolate(
            method="linear",
            limit_direction="both",
            inplace=True
        )

        # remaining nan

        df.fillna(
            df.median(),
            inplace=True
        )

        feat = extract_features(df)

        feat["sample_id"] = path.stem

        feat["well_id"] = well

        feat["label"] = label

        records.append(
            feat
        )

    feature_df = pd.DataFrame(
        records
    )

    # sample信息放前面

    cols = [
        "sample_id",
        "well_id",
        "label"
    ]

    feature_df = feature_df[
        cols +
        [
            c for c in feature_df.columns
            if c not in cols
        ]
    ]

    feature_df.to_csv(
        OUTPUT,
        index=False
    )

    print(
        "saved:",
        OUTPUT
    )

    print(
        feature_df.shape
    )

    return feature_df


if __name__ == "__main__":

    build_feature_dataset()
