from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from src import B2_predict

# python predict.py --data_dir ./data --output ./result.csv

# 脚本开头
np.seterr(invalid="ignore")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=r"D:\.datasets\oil_data\test\data"
    )
    parser.add_argument("--output", type=str, default="./result.csv")

    args = parser.parse_args()
    print(args.data_dir)
    print(args.output)
    B2_predict.TEST_DIR = Path(args.data_dir)
    B2_predict.OUTPUT = Path(args.output)
    B2_predict.main()


if __name__ == "__main__":
    main()
