from datasets.lidarh26m_dataset import LiDARH26MPoseDataset
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw-path', type=str, default='/mnt/datasets/lidarhuman26M')
    parser.add_argument('--buffer-path', type=str, default='./data/lidarh26m')
    args = parser.parse_args()
    print(args)
    for split in ['train', 'test']:
        dataset = LiDARH26MPoseDataset(
            raw_path=args.raw_path,
            buffer_path=args.buffer_path,
            split=split,
        )