from pathlib import Path

import numpy as np
import pandas as pd

import joblib

try:
    from src import B0_wash
except:
    import B0_wash


try:
    from src import B1_train
except:
    import B1_train

# =====================================================
# 路径
# =====================================================

TEST_DIR = Path(r"D:\.datasets\oil_data\test\data")


MODEL_PATH = Path(r"./models/v3.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load("models/v3_features.pkl")

OUTPUT = Path("result.csv")


if __name__ == "__main__":
    string = "切片ID,溢流判断\n"
    for csv in TEST_DIR.glob("*.csv"):
        print(f"读取文件：{csv.name}")
        df = pd.read_csv(csv)


        df = B0_wash.build_feature_dataset(df, False)
        df = B0_wash.build_one_row_features(df, False)

        df = B1_train.clean_feature_names(df)

        df = df.reindex(columns=feature_names, fill_value=0)

        train_cols = list(model.feature_names_in_)
        pred_cols = list(df.columns)

        # missing = [c for c in train_cols if c not in pred_cols]
        # extra = [c for c in pred_cols if c not in train_cols]

        # print("测试集缺少列:", missing)
        # print("测试集多出列:", extra)

        # 现在的 df 就是测试集内的所有数据，一行一个样本，送入 lightgbm 模型进行预测
        # print(df.nunique().sort_values().head(20))

        # prob=model.predict_proba(df)[:,1]

        # print(prob)

        y_pred = model.predict(df)

        print(type(y_pred))  # <class 'numpy.ndarray'>

        print("y_pred:", y_pred)
        string += f"{csv.stem},{y_pred[0]}\n"
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(string)

    print("完成")