from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import joblib
import pickle

# 1. 读取数据
df = pd.read_csv(r'D:\Study5\超深油气井钻井过程溢流实时预警\overflow\dataset\features.csv')

# 2. 提取标签 y
y = df["label"]

# 3. 提取特征 X
# 直接排除 id 列和 label 列，避免重复删除报错，同时确保 X 中不包含非数值特征
X = df.drop(columns=["sample_id", "well_id", "label"])

# 打印X的所有列明


print(list(X.columns))


# 4. 划分训练集和测试集（推荐做法，避免用训练集直接评估导致过拟合）
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. 初始化模型
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=8,
    max_depth=3,
    class_weight={
        0: 1,
        1: 10
    },
    random_state=42
)

# 6. 训练模型
model.fit(X_train, y_train)


joblib.dump(
    model,
    "models/overflow_lgbm_2.pkl"
)


joblib.dump(
    list(X.columns),
    "feature_names.pkl"
)


# 7. 预测与评估
y_pred = model.predict(X_test)

print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

