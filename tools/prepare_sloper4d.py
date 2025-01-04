from datasets.sloper4d_dataset import SLOPER4D_Dataset
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw-path', type=str, default='/mnt/datasets/SLOPER4D')
    parser.add_argument('--buffer-path', type=str, default='./data/sloper4d')
    args = parser.parse_args()
    print(args)
    for split in ['train', 'test']:
        dataset = SLOPER4D_Dataset(
            raw_path=args.raw_path,
            buffer_path=args.buffer_path,
            split=split,
        )