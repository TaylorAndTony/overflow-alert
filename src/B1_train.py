import re

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """清洗列名，将特殊字符替换为下划线，并确保无重复"""
    df = df.rename(columns=lambda x: re.sub(r"[^A-Za-z0-9_]", "_", str(x)))

    # 处理清洗后可能出现的重复列名
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


# 1. 读取数据
df = pd.read_csv(r"./train_v3.csv")
df2 = pd.read_csv(r"./test_v3.csv")

# 2. 提取标签 y
y_train = df["label"]

# 3. 提取特征 X
# 直接排除 id 列和 label 列，避免重复删除报错，同时确保 X 中不包含非数值特征
x_train = df.drop(columns=["label"])

y_test = df2["label"]

x_test = df2.drop(columns=["label"])


x_train = clean_feature_names(x_train)
x_test = clean_feature_names(x_test)

# 5. 初始化模型
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=8,
    max_depth=3,
    class_weight={0: 1, 1: 10},
    random_state=42,
)

# 6. 训练模型
model.fit(x_train, y_train)


joblib.dump(model, "models/v3.pkl")
joblib.dump(list(x_train.columns), "models/v3_features.pkl")


# 7. 预测与评估
y_pred = model.predict(x_test)

print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
