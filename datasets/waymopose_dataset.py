import os
import numpy as np
import pickle

from pathlib import Path
from torch.utils.data import Dataset

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose

@DATASETS.register_module()
class WaymoPoseDataset(Dataset):
    def __init__(self,
                 raw_path='/mnt/datasets/waymo_open_dataset_v_2_0_0/',
                 buffer_path='./data/waymo_v2',
                 split='training',
                 modality=['lidar',],
                 keypoint_range=[*range(1, 14), 15],
                 transforms=None,
                 ) -> None:
        super().__init__()
        self.raw_path = Path(raw_path) / split
        self.buffer_path = Path(buffer_path)
        if not self.buffer_path.exists():
            os.makedirs(self.buffer_path)
        
        self.split = split
        self.keypoint_range = keypoint_range
        self.modality = modality
        self._db = self._read_infos()
        self._lidar_buffers = dict()
        self._lidar_buffers_fresh = dict()
        
        self.transforms = Compose(transforms)

    def _read_infos(self):
        db_buffer_file = self.buffer_path / f"{self.split}.pkl"
        if db_buffer_file.exists():
            return pickle.load(open(db_buffer_file, 'rb'))
        else:
            raise Exception("Please generate database first.")

    def __getitem__(self, index):
        instance_info = self._db[index]
        sample = dict(**instance_info)
        sample['id'] = index
        sample['coord'], sample['color'] = sample['coord'][:, :3], sample['coord'][:, 3:]
        sample['keypoints_3d'] = sample['keypoints_3d'].copy()[self.keypoint_range, :]
        sample['keypoints_3d'][..., 3] = (sample['keypoints_3d'][..., 3] > 0).astype(np.float32)
        
        return self.transforms(sample)
    
    def __len__(self):
        return len(self._db)