import numpy as np
from pathlib import Path
import pandas as pd


def concat_excel(paths: list[Path]) -> pd.DataFrame:
    """
    concat excel files
    """
    lst = []
    for path in paths:
        print(f'  reading {path.name}...')
        df = pd.read_excel(path)
        lst.append(df)
    return pd.concat(lst)

def concat_dataset(well_folder: Path | str) -> pd.DataFrame:
    """ 
    合并所有的 xls 时序数据到 pd.DataFrame

    well_folder:  D:/.datasets/oil_data/train/WELL_000001  里面必须有 `WELL_000001时序数据` 文件夹
    """
    if isinstance(well_folder, str):
        well_folder = Path(well_folder)

    target = well_folder / 'data.csv'

    if target.exists():
        print(f'{target.name} exists, skip concat')
        return pd.read_csv(target)

    sub_dir = well_folder / f'{well_folder.name}时序数据'

    print(f'Concating {well_folder.name}...')
    files = []
    files = sorted(
        sub_dir.glob("*.xls")
    )
    df = concat_excel(files)
    df.to_csv(target, index=False)
    return df


df = concat_dataset(
r'D:\.datasets\oil_data\train\WELL_0000010'
)


df["timestamp"] = pd.to_datetime(
    df["时间"]
)


target = pd.Timestamp(
    "2023-07-24 07:38:00"
)


print(
    df[
        (df.timestamp >= target-pd.Timedelta(minutes=5))
        &
        (df.timestamp <= target+pd.Timedelta(minutes=5))
    ][
        ["时间","井深(8004) m"]
    ]
)