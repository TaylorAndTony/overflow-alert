from argparse import ArgumentParser
from pathlib import Path
import random
# python predict.py --data_dir ./data --output ./result.csv


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'D:\.datasets\oil_data\test\data')
    parser.add_argument('--output', type=str, default='./result.csv')

    args = parser.parse_args()
    print(args.data_dir)
    print(args.output)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('序号,切片ID,溢流判断\n')
        idx=1
        for file in Path(args.data_dir).glob('*.csv'):
            print(file.name)
            # r =random.randint(0,1)
            r = 0
            f.write(f'{idx},{file.stem},{r}\n')
            idx+=1



if __name__ == '__main__':
    main()
