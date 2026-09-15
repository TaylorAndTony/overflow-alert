from pathlib import Path

import numpy as np
import pandas as pd

import joblib


# =====================================================
# 路径
# =====================================================

TEST_DIR = Path(
    r"D:\.datasets\oil_data\test\data"
)


MODEL_PATH = Path(
    r"./models/overflow_lgbm_2.pkl"
)


FEATURE_PATH = Path(
    r"./feature_names.pkl"
)


OUTPUT = Path(
    "result.csv"
)


# =====================================================
# 异常值
# =====================================================

INVALID_VALUES = [
    -999.25,
    -999,
    -9999,
    -9999.0
]


# =====================================================
# 数据清洗
# 必须和训练一致
# =====================================================

def clean_dataframe(df):

    df = df.copy()

    # 删除时间相关字段

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

    # 删除标签
    # 防止泄漏

    df.drop(
        columns=[
            "label",
            "sample_id",
            "well_id"
        ],
        inplace=True,
        errors="ignore"
    )

    # 异常值

    df.replace(
        INVALID_VALUES,
        np.nan,
        inplace=True
    )

    # 转数字

    for c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    # 插值

    df.interpolate(
        method="linear",
        limit_direction="both",
        inplace=True
    )

    # 剩余nan

    df.fillna(
        df.median(),
        inplace=True
    )

    return df


# =====================================================
# slope
# =====================================================

def calc_slope(x):

    y = x.values

    if len(y) < 5:
        return 0.0

    xx = np.arange(
        len(y)
    )

    return np.polyfit(
        xx,
        y,
        1
    )[0]


# =====================================================
# 特征提取
# 和feature_extract.py保持一致
# =====================================================

def extract_features(df):

    features = {}

    n = len(df)

    last_n = min(
        300,
        max(1, n//5)
    )

    last = df.iloc[-last_n:]

    for col in df.columns:

        x = df[col]

        features[f"{col}_mean"] = x.mean()

        features[f"{col}_std"] = x.std()

        features[f"{col}_max"] = x.max()

        features[f"{col}_min"] = x.min()

        features[f"{col}_median"] = x.median()

        features[f"{col}_first"] = x.iloc[0]

        features[f"{col}_last"] = x.iloc[-1]

        features[f"{col}_delta"] = (
            x.iloc[-1]
            -
            x.iloc[0]
        )

        features[f"{col}_slope"] = (
            calc_slope(x)
        )

        features[f"{col}_last5_delta"] = (
            last.iloc[-1][col]
            -
            last.iloc[0][col]
        )

    return pd.DataFrame(
        [features]
    )


# =====================================================
# 单个文件预测
# =====================================================

def predict_one(
        csv_path,
        model,
        feature_names
):

    print(
        "\nprocessing:",
        csv_path.name
    )

    df = pd.read_csv(
        csv_path
    )

    print(
        "raw shape:",
        df.shape
    )

    df = clean_dataframe(
        df
    )

    print(
        "clean shape:",
        df.shape
    )

    X = extract_features(
        df
    )

    print(
        "before align:",
        X.shape
    )

    # ==========================
    # 特征严格对齐
    # ==========================

    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    print(
        "after align:",
        X.shape
    )

    prob = model.predict_proba(
        X
    )[0, 1]

    pred = int(
        prob >= 0.5
    )


    return {
        '切片ID': csv_path.stem,
        '溢流判断': pred
    }


# =====================================================
# main
# =====================================================

if __name__ == "__main__":

    print("loading model...")

    model = joblib.load(
        MODEL_PATH
    )

    feature_names = joblib.load(
        FEATURE_PATH
    )

    print(
        "model loaded"
    )

    print(
        "feature number:",
        len(feature_names)
    )

    results = []

    files = sorted(
        TEST_DIR.glob("*.csv")
    )

    print(
        "test files:",
        len(files)
    )

    for f in files:

        result = predict_one(
            f,
            model,
            feature_names
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by="切片ID",
        ascending=True
    )

    result_df.to_csv(
        OUTPUT,
        index=False
    )

    print("\n========== RESULT ==========")

    print(
        result_df
    )

    print(
        "\nsaved:",
        OUTPUT
    )
