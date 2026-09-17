from pathlib import Path
import random

import numpy as np
import pandas as pd

from tqdm import tqdm


# =========================
# 参数
# =========================

"""
井深(8004) m
钻头位置(8005) m
迟到井深(8006) m
迟到时间(8007) min
大钩高度(8008) m
大钩速度(8009) m/s
大钩负荷(8010) kN
钻压(8011) kN
转盘转速(8012) RPM
扭矩(8013) kN.m
累计泵冲次1(8014) stk
累计泵冲次2(8015) stk
累计泵冲次3(8016) stk
累计泵冲总和(8017) stk
立管压力(8018) MPa
套管压力(8019) MPa
钻时(8020) min/m
钻头进尺(8021) m
纯钻进时间(8022) min
DC指数(8023) 
SIGMA指数(8024) 
单位进尺成本(8025) $/M
套管鞋处有效循环密度(8026) kg/m3
垂直井深(8027) m
钻井状态参数(8028) 
迟到泵冲数(8029) stk
大钳扭矩(8030) kN.m
录井钻头垂深(8031) m
瞬时钻速(8032) m/hr
累积流量(8033) m3
起下钻罐(8034) m3
钻头直径(8035) m
钻井总时间(8036) hr
深度成本(8037) $/M
DCS(8038) 
当量密度(8039) g/cm3
上覆地压(8040) MPa
DC压力梯度(8041) 
N指数(8042) 
F指数(8043) 
参考SIGMA(8044) 
钻头转速(8045) RPM
钻速(8046) m/hr
总冲数(8047) spm
实际SIGMA(8048) 
方入(8049) m
井底上空(8050) m
理论悬重(8051) kN
顶驱扭矩(8052) A
瞬时钻时(8053) min/m
提钻超拉(8054) kN
井内立柱数(8055) 
井外立柱数(8056) 
钻头号(8057) 
钻井天数(8058) 
泵冲次1(8059) spm
泵冲次2(8060) spm
泵冲次3(8061) spm
总泵冲(8062) spm
取样日期时间(8101) 
日期(8102) 
时间(8103) 
入口密度(8104) g/cm3
出口密度(8105) g/cm3
入口温度(8106) degC
出口温度(8107) degC
入口电导(8108) s/m
出口电导(8109) s/m
入口流量(8110) L/s
出口流量(8111) %
总池体积(8112) m3
池体积1(8113) m3
池体积2(8114) m3
池体积3(8115) m3
池体积4(8116) m3
池体积5(8117) m3
池体积6(8118) m3
池体积7(8119) m3
池体积8(8120) m3
池体积9(8121) m3
池体积10(8122) m3
池体积11(8123) m3
池体积12(8124) m3
溢漏体积(8125) m3
活动池变化量(8126) m3
进口压力(8127) MPa
出口压力(8128) MPa
出口密度1(8129) g/cm3
出口电导1(8130) s/m
出口温度1(8131) degC
池体积13(8132) m3
池体积14(8133) m3
总池体积1(8134) m3
总池体积2(8135) m3
泥浆溢流(8136) m3
出口流量(百分)(8137) %
取样日期时间(8201) 
日期(8202) 
时间(8203) 
二氧化碳(8204) %
氢(8205) %
氦(8206) %
硫化氢1(8207) ppm
硫化氢2(8208) ppm
硫化氢3(8209) ppm
硫化氢4(8210) ppm
硫化氢5(8211) ppm
甲烷(8212) %
乙烷(8213) %
丙烷(8214) %
正丁烷(8215) %
异丁烷(8216) %
正戊烷(8217) %
异戊烷(8218) %
全烃(8219) %
计算全氢(8220) %
硫化氢6(8221) ppm
硫化氢7(8222) ppm
硫化氢8(8223) ppm
可燃气体4(8224) ppm
可燃气体5(8225) ppm
气体流量(8226) L/s
计算全烃(8228) ppm
可燃气体1(8229) ppm
可燃气体2(8230) ppm
可燃气体3(8231) ppm
"""

INVALID_VALUES = [-999.25, -999, -9999, -9999.0]

COLUMNS = [
    "井深(8004) m",
    "钻头位置(8005) m",
    "钻压(8011) kN",
    "扭矩(8013) kN.m",
    "大钳扭矩(8030) kN.m",
    "出口温度(8107) degC",
    "出口电导(8109) s/m",
    "总池体积(8112) m3",
    "池体积1(8113) m3",
    "池体积2(8114) m3",
    "池体积3(8115) m3",
    "全烃(8219) %",
    "甲烷(8212) %",
    "乙烷(8213) %",
    "丙烷(8214) %",
    "正丁烷(8215) %",
    "异丁烷(8216) %",
    "正戊烷(8217) %",
    "异戊烷(8218) %",
]
EPSILON = 1e-12

# 预处理单井超大 csv 时，滑动窗口大小
WELL_WINDOW_SECONDS = 1800  # 30分钟窗口

# 正样本滑动步长
POS_STRIDE_SECONDS = 300

NEG_NUM = 10  # 每口井负样本数量

# 处理正负样本时，设定滑动窗口大小，例如 60 秒（假设 1s 采样频率）
SAMPLE_WINDOW_SIZE = 60

# =========================
# concat
# =========================


def concat_excel(paths: list[Path]):
    dfs = []
    for p in paths:
        print("  合并表格，读取", p.name)
        dfs.append(pd.read_excel(p))
    return pd.concat(dfs, ignore_index=True)


def concat_dataset(well_folder):

    well_folder = Path(well_folder)

    cache = well_folder / "data.csv"

    if cache.exists():
        print("合并表格存在，使用", well_folder.name)

        return pd.read_csv(cache)

    print("缓存不存在，合并表格", well_folder.name)
    data_dir = well_folder / f"{well_folder.name}时序数据"

    files = sorted(data_dir.glob("*.xls"))

    df = concat_excel(files)

    df.to_csv(cache, index=False)

    return df


# =========================
# 事件
# =========================


def read_metrics(well_folder):
    well_folder = Path(well_folder)
    file = well_folder / f"{well_folder.name}溢流基本信息.xlsx"
    return pd.read_excel(file)


def remove_invalid_values(df):

    df = df.copy()

    invalid = [-999.25, -9999]

    for col in df.columns:
        if col in COLUMNS:
            df[col + "_missing"] = df[col].isin(invalid).astype(int)

    df.replace(invalid, np.nan, inplace=True)

    return df


def generate_pos_neg_samples(
    well_name: str, df_data: pd.DataFrame, df_metric: pd.DataFrame
) -> None:
    t_溢流发生时间: str = df_metric["溢流发生时间"][0]
    t_关井时间: str = df_metric["关井时间"][0]
    d_溢流时井深: str = df_metric["溢流时井深 (m)"][0]

    t_overflow: pd.Timestamp = pd.to_datetime(t_溢流发生时间)
    t_well_shut: pd.Timestamp = pd.to_datetime(t_关井时间)

    write_to = Path("./dataset/samples_v3")
    write_to.mkdir(parents=True, exist_ok=True)

    samples = []

    window = pd.Timedelta(seconds=WELL_WINDOW_SECONDS)
    stride = pd.Timedelta(seconds=POS_STRIDE_SECONDS)

    # 给 df_data 添加 pd 时间列
    df_data["timestamp"] = pd.to_datetime(df_data["时间"])

    # 正样本：从 t_overflow 开始，向时间更旧的方向滑动窗口
    # 窗口右边界从 t_overflow 逐步向前滑动，步长为 stride
    t_right = t_overflow  # 窗口的右边界（初始为溢流发生时刻）
    t_left_bound = t_overflow - window  # 窗口左边界不能早于这个时间

    while t_right > t_left_bound:
        t_left = t_right - window  # 当前窗口的左边界

        # 截取当前窗口内的数据
        window_data = df_data[
            (df_data["timestamp"] >= t_left) & (df_data["timestamp"] < t_right)
        ]

        window_data = window_data.copy()

        window_data["label"] = 1

        if len(window_data) > 0:
            samples.append(
                {
                    "t_left": t_left,
                    "t_right": t_right,
                    "data": window_data,
                    "label": 1,  # 正样本
                }
            )

        # 窗口整体向前滑动一个步长
        t_right -= stride

    for i, sample in enumerate(samples):
        sample["data"].to_csv(write_to / f"{well_name}_pos_{i}.csv", index=False)
        print("保存正样本", i)

    # 负样本：从 t_overflow - window 往前滑动窗口，均匀滑动到最左端，总共滑动 NEG_NUM 个窗口

    # 首先计算出最左端的时间
    the_most_left = df_data["timestamp"].min()

    # 窗口左边界不能早于这个时间
    # 计算这个窗口滑动时使用的 stride，基于给定的数量计算步幅
    neg_window_slide_stride = (t_overflow - the_most_left) / NEG_NUM

    for i in range(NEG_NUM):
        t_left = the_most_left + i * neg_window_slide_stride
        t_right = t_left + window
        t_right = min(t_right, t_overflow)

        # 截取当前窗口内的数据
        window_data = df_data[
            (df_data["timestamp"] >= t_left) & (df_data["timestamp"] < t_right)
        ]
        window_data = window_data.copy()

        window_data["label"] = 0

        if len(window_data) > 0:
            samples.append(
                {
                    "t_left": t_left,
                    "t_right": t_right,
                    "data": window_data,
                    "label": 0,  # 负样本
                }
            )
            window_data.to_csv(write_to / f"{well_name}_neg_{i}.csv", index=False)
            print("保存负样本", i)


def build_feature_dataset(df, has_label=True) -> pd.DataFrame:

    df = remove_invalid_values(df)

    # 对于 df，只保留其包含在 COLUMNS 中的列
    col = COLUMNS[:]
    if has_label:  # 训练集
        col.append("label")
    df = df[col]

    print("构建特征数据集")

    # 缓存所有新增特征列
    feature_dict = {}

    for col in COLUMNS:
        if col == "label":
            continue

        # 计算窗口内的均值（平滑高频噪声）
        feature_dict[f"{col}_ma_{SAMPLE_WINDOW_SIZE}"] = (
            df[col].rolling(window=SAMPLE_WINDOW_SIZE, min_periods=1).mean()
        )

        # 计算窗口内的标准差（捕捉溢流前的剧烈波动）
        feature_dict[f"{col}_std_{SAMPLE_WINDOW_SIZE}"] = (
            df[col].rolling(window=SAMPLE_WINDOW_SIZE, min_periods=1).std()
        )

        # 计算窗口内的最大值（捕捉压力突升极值）
        feature_dict[f"{col}_max_{SAMPLE_WINDOW_SIZE}"] = (
            df[col].rolling(window=SAMPLE_WINDOW_SIZE, min_periods=1).max()
        )

        # 一阶差分：当前值与上一秒的差值，捕捉瞬间突变
        feature_dict[f"{col}_diff_1"] = df[col].diff(periods=1)

        # 二阶差分：当前值与前两秒的差值，捕捉瞬间突变
        feature_dict[f"{col}_diff_2"] = df[col].diff(periods=2)

        # 环比增长率：当前值相对于上一秒的变化率
        feature_dict[f"{col}_growth_rate"] = (df[col] - df[col].shift(1)) / (
            df[col].shift(1) + EPSILON
        )

        # 结合 rolling 和 shift：计算当前窗口均值与上一窗口均值的差值
        ma_col = f"{col}_ma_{SAMPLE_WINDOW_SIZE}"
        feature_dict[f"{col}_ma_trend"] = feature_dict[ma_col] - feature_dict[
            ma_col
        ].shift(1)

    # 一次性将所有特征列拼接到 df 上
    df = pd.concat([df, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    # 处理生成的缺失值 (NaN)
    df.fillna(0, inplace=True)

    return df


def calc_slope(series):

    y = series.values

    x = np.arange(len(y))

    if len(y) < 2:
        return 0

    return np.polyfit(x, y, 1)[0]


def build_one_row_features(df, has_label=True) -> pd.DataFrame:
    """
    把一个多行经过 build_feature_dataset 处理的 DataFrame 转换为单行特征
    """
    features = {}
    if has_label:
        features["label"] = df["label"].iloc[0]
    for col in df.columns:
        if col == "label":
            continue
        x = df[col].astype(float)

        features[f"{col}_mean"] = x.mean()

        features[f"{col}_std"] = x.std()

        features[f"{col}_max"] = x.max()

        features[f"{col}_min"] = x.min()

        features[f"{col}_median"] = x.median()

        # 初始状态

        features[f"{col}_first"] = x.iloc[0]

        # 当前状态

        features[f"{col}_last"] = x.iloc[-1]

        # 变化量

        features[f"{col}_delta"] = x.iloc[-1] - x.iloc[0]

        # 最近60秒

        last60 = x.tail(60)

        features[f"{col}_last60_mean"] = last60.mean()

        features[f"{col}_last60_max"] = last60.max()

        # 趋势

        features[f"{col}_slope"] = calc_slope(x)

    return pd.DataFrame([features])


if __name__ == "__main__":
    if input("是否根据数据集重新生成正负样本csv文件? (y/n)") == "y":
        for folder in Path(r"D:\.datasets\oil_data\train").iterdir():
            if not folder.is_dir():
                continue

            print(folder.name)
            data = concat_dataset(folder)
            metrics = read_metrics(folder)
            generate_pos_neg_samples(folder.stem, data, metrics)

    df_list = []
    for f in Path("./dataset/samples_v3").glob("*.csv"):
        print(f"为 {f.name} 生成特征")
        df = pd.read_csv(f)
        df = build_feature_dataset(df)
        df = build_one_row_features(df)
        df_list.append(df)

    # merge all samples to one big dataset
    df = pd.concat(df_list)
    # df.to_csv("./train_v3.csv")
    # 取出df的末尾10行
    df_last = df.tail(10)
    # 取出开头到末尾第10行
    df_middle = df.iloc[:-10]
    # 分别写入csv文件
    df_last.to_csv("./test_v3.csv")
    df_middle.to_csv("./train_v3.csv")
