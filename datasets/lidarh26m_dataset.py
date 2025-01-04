import os
import numpy as np
import pickle
import shutil
import json
from collections import defaultdict
from pathlib import Path
from plyfile import PlyData
from typing import Callable

from torch import Tensor
from functools import lru_cache
import torch
from torch.utils.data import Dataset

import tqdm
import smplx

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose

@DATASETS.register_module()
class LiDARH26MPoseDataset(Dataset):
    _getitem_call: Callable[['LiDARH26MPoseDataset', int], str] = None
    def __init__(self,
                 raw_path='/mnt/datasets/lidarhuman26M',
                 buffer_path='./data/lidarh26m',
                 smpl_model_path='./smpl_models',
                 split='train',
                 modality=['lidar'],
                 keypoint_range=[0,1,2,4,5,7,8,12,15,16,17,18,19,20,21],
                 transforms=None,
                 interval=1,
                 seq_len=None,
                 seq_interval=1,
                 ) -> None:
        super().__init__()
        self.raw_path = Path(raw_path)
        self.buffer_path = Path(buffer_path)
        self.smpl_model_path = Path(smpl_model_path)
        if not self.buffer_path.exists():
            self.buffer_path.mkdir(parents=True)
        self.split = split
        self.keypoint_range = keypoint_range
        self.modality = modality
        self._db = self._read_infos()
        
        self.interval = interval
        self.seq_len = seq_len
        self.seq_interval = seq_interval

        if self.seq_len is not None:
            self._db = self._prepare_seq_db(self._db)
            self._getitem_call = self.getitem_mf
        else:
            self._getitem_call = self.getitem
        
        self.transforms = Compose(transforms)
        
        human_model = smplx.create(
            str(self.smpl_model_path), 
            model_type='smpl',
            gender='neutral', 
            use_face_contour=False,
            ext="npz"
        )
        
        part_mapping = json.load(open(self.smpl_model_path / 'smpl_body_parts_2_faces.json'))
        faces = human_model.faces
        faces_label = np.zeros(faces.shape[0], dtype=np.int32)
        for i, val in enumerate(part_mapping.values()):
            faces_label[val] = i + 1
            
        self.smpl_mixin = dict(
            face=faces,
            face_label=faces_label
        )
    
    def _prepare_seq_db(self, db):
        seq_dict = defaultdict(list)
        seq_list = []
        for frame in db:
            seq_dict[frame['sequence_id'].split('/')[0]].append(frame)
        for v in seq_dict.values():
            for fs in range(0, len(v) - self.seq_len, self.seq_interval):
                seq_list.append(v[fs:fs+self.seq_len])
        return seq_list
    
    def _read_infos(self):
        seq_list_file = self.raw_path / f"{self.split}.txt"
        assert (self.raw_path / f"{self.split}.txt").exists()
        dir_list = open(seq_list_file, 'r').read().split('\n')
        buffer_file = self.buffer_path / f"{self.split}.pkl"
        if buffer_file.exists():
            return pickle.load(open(buffer_file, 'rb'))
        
        db = []
        print(f"Extracting {self.split} data from raw files...")
        for dir_name in tqdm.tqdm(dir_list):
            for f in sorted((self.raw_path / 'labels' / '3d' / 'pose' / dir_name).glob('*.json')):
                anno = json.load(open(f, 'r'))
                anno['sequence_id'] = f"{dir_name}/{f.name[:-5]}"
                anno['beta'] = np.array(anno['beta'])
                anno['pose'] = np.array(anno['pose'])
                anno['trans'] = np.array(anno['trans'])
                db.append(anno)

        print(f"Generating {self.split} gt keypoints...")
        results = self._smpl_extract_joint(db)
        
        for i, s in enumerate(db):
            s['keypoints_3d'] = results['joints'][i]
            s['vertices'] = results['vertices'][i]
        
        pickle.dump(db, open(buffer_file, 'wb'))
        return db
        
    def _smpl_extract_joint(self, 
                            db,
                            batch_size=32,
                            out_keys=['joints', 'vertices']
                            ):
        human_model = smplx.create(str(self.smpl_model_path), 
                                   model_type = 'smpl',
                                    gender='neutral', 
                                    use_face_contour=False,
                                    ext="npz").cuda()
        def sample_gen():
            for i in range(0, len(db), batch_size):
                batch = db[i:i+batch_size]
                pose = np.stack([b['pose'] for b in batch])
                beta = np.stack([b['beta'] for b in batch])
                trans = np.stack([b['trans'] for b in batch])
                yield pose, beta, trans
        
        results = {k: [] for k in out_keys}
        with torch.no_grad():
            for b_pose, b_beta, b_trans in tqdm.tqdm(sample_gen(), 
                                                    total=len(db) // batch_size + 1):
                b_pose = Tensor(b_pose).cuda().float()
                b_beta = Tensor(b_beta).cuda().float()
                b_trans = Tensor(b_trans).cuda().float()
                
                smpl_out = human_model(
                    betas=b_beta,
                    body_pose=b_pose[..., 3:],
                    global_orient=b_pose[..., :3],
                    transl=b_trans
                )
                for k in out_keys:
                    results[k].append(smpl_out[k].cpu().numpy())
        
        return {k: np.concatenate(results[k], axis=0) for k in out_keys}
    
    
    @lru_cache(maxsize=32)
    def get_sensor_data(self, sequence_id):
        ply_data = PlyData.read(self.raw_path / "labels/3d/segment" / f"{sequence_id}.ply")
        points = np.array([[x, y, z] for x, y, z in ply_data['vertex'].data])
        return dict(point_cloud=points)
    
    def getitem_mf(self, index):
        index = index * self.interval
        seq_info = self._db[index]
        
        sample = []
        for frame_info in seq_info:
            frame = dict(**frame_info, **self.smpl_mixin)
            sensor_data = self.get_sensor_data(frame_info['sequence_id'])
            frame['coord'] = sensor_data['point_cloud'][:, :3]
            frame['num_points'] = frame['coord'].shape[0]
            frame['bbox'] = frame['trans']
            frame['vertex'] = frame['vertices']

            frame['keypoints_3d'] = frame['keypoints_3d'][self.keypoint_range].copy()
            vis = np.ones(frame['keypoints_3d'].shape[0])
            frame['keypoints_3d'] = np.c_[frame['keypoints_3d'], vis]

            sample.append(self.transforms(frame))
        
        return sample
        
    
    def getitem(self, index):
        index = index * self.interval
        instance_info = self._db[index]
        sensor_data = self.get_sensor_data(instance_info['sequence_id'])
        inst_point_cloud = sensor_data['point_cloud']

        sample = dict(**instance_info, **self.smpl_mixin)
        sample['id'] = index
        sample['coord'] = inst_point_cloud[:, :3]
        sample['bbox'] = sample['trans']
        sample['vertex'] = sample['vertices']
        
        sample['keypoints_3d'] = sample['keypoints_3d'][self.keypoint_range].copy()
        vis = np.ones(sample['keypoints_3d'].shape[0])
        sample['keypoints_3d'] = np.c_[sample['keypoints_3d'], vis]
        
        return self.transforms(sample)
    
    def __getitem__(self, index):
        return self._getitem_call(index)
    
    def __len__(self):
        return len(self._db) // self.interval