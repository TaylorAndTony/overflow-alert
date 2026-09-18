import re

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

COLUMN_MAPPING = {}

# def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
#     """清洗列名，将特殊字符替换为下划线，并确保无重复"""
#     df = df.rename(columns=lambda x: re.sub(r"[^A-Za-z0-9_]", "_", str(x)))

#     # 处理清洗后可能出现的重复列名
#     seen = {}
#     new_cols = []
#     for col in df.columns:
#         if col in seen:
#             seen[col] += 1
#             new_cols.append(f"{col}_{seen[col]}")
#         else:
#             seen[col] = 0
#             new_cols.append(col)
#     df.columns = new_cols
#     return df

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_col = []
    seen={}
    for col in df.columns:
        new_name = re.sub(r"[^a-zA-Z0-9_]+", "_", col)
        if new_name in seen:
            seen[new_name] += 1
            new_name = f"{new_name}_{seen[new_name]}"
        else:
            seen[new_name] = 0
        new_col.append(new_name)
        COLUMN_MAPPING[new_name] = col
    df.columns = new_col
    return df


# 1. 读取数据
df2 = pd.read_csv(r"./test_v3.csv")

y_test = df2["label"]

x_test = df2.drop(columns=["label"])


x_test = clean_feature_names(x_test)

# joblib.dump(model, "models/v3.pkl")
# joblib.dump(list(x_train.columns), "models/v3_features.pkl")

model = joblib.load("models/v3.pkl")


# 7. 预测与评估
prob = model.predict_proba(x_test)[:, 1]

threshold = 0.5

y_pred = (prob > threshold).astype(int)


# 在模型训练/预测完成后，恢复原始列名
x_test = x_test.rename(columns=COLUMN_MAPPING)

# 特征重要性表也可以用原始列名
importance = pd.DataFrame(
    {"feature": x_test.columns, "importance": model.feature_importances_}
)

head = importance.sort_values("importance", ascending=False).head(30)
print(head)


print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
