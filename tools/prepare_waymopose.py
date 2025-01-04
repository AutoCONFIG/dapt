import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import duckdb
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.keypoints import KeypointType
import tensorflow as tf
from functools import partial, reduce
from argparse import ArgumentParser
from utils import box_np_ops
import numpy as np
import pickle as pkl
from collections import defaultdict, OrderedDict
from typing import Optional
from pathlib import Path
from mmengine import track_parallel_progress


def convert_range_image_to_point_cloud(
    range_image: v2._lidar.RangeImage,
    calibration: v2._context.LiDARCalibrationComponent,
    pixel_pose: Optional[v2._lidar.PoseRangeImage] = None,
    frame_pose: Optional[v2._pose.VehiclePoseComponent] = None,
    extra_tensor: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Convert lidar range image to point cloud
    
    Returns:
        tf.Tensor: A 6-D tensor, (x, y, z, intensity, elongation, nlz_mask) 
    """
    range_image_cartesian = v2._lidar_utils.convert_range_image_to_cartesian(
        range_image=range_image,
        calibration=calibration,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        keep_polar_features=False,
    )
    range_image_tensor = range_image.tensor
    range_image_mask = range_image_tensor[..., 0] > 0
    
    if extra_tensor is not None:
        range_image_cartesian = tf.concat(
            [
                range_image_cartesian,
                extra_tensor
            ],
            axis=-1,
        )
    
    points_tensor = tf.gather_nd(
        range_image_cartesian,
        tf.compat.v1.where(range_image_mask)
    )
    
    return points_tensor


def process_sequence_pointcloud(sequence_name: str, raw_path: Path, buffer_path: Path, extract_instance=False):
    sql = f"""
    SELECT 
        *
    FROM (
        SELECT DISTINCT lidar_hkp.index as index
        FROM '{str(raw_path / 'lidar_hkp' / sequence_name)}' lidar_hkp
        LEFT JOIN '{str(raw_path / 'lidar_box' / sequence_name)}' lidar_box
        ON lidar_hkp.index = lidar_box.index and lidar_hkp."key.laser_object_id" = lidar_box."key.laser_object_id"
    ) frames

    LEFT JOIN '{str(raw_path / 'vehicle_pose' / sequence_name)}' vehicle_pose
    ON frames.index = vehicle_pose.index

    JOIN '{str(raw_path / 'lidar' / sequence_name)}' lidar
    ON frames.index = lidar.index

    LEFT JOIN '{str(raw_path / 'lidar_pose' / sequence_name)}' lidar_pose
    ON lidar.index = lidar_pose.index and lidar."key.laser_name" = lidar_pose."key.laser_name"

    LEFT JOIN '{str(raw_path / 'lidar_calibration' / sequence_name)}' lidar_calibration
    ON lidar."key.laser_name" = lidar_calibration."key.laser_name"
    """
    seq_df = duckdb.sql(sql).to_df()

    points_all = defaultdict(list)
    for i, row in seq_df.iterrows():
        row[row.isna()] = None
        lidar = v2.LiDARComponent.from_dict(row)
        lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)
        lidar_pose = v2.LiDARPoseComponent.from_dict(row)
        vehicle_pose = v2.VehiclePoseComponent.from_dict(row)

        for range_image, pixel_pose, frame_pose in zip(
            [lidar.range_image_return1, lidar.range_image_return2],
            [lidar_pose.range_image_return1, None],
            [vehicle_pose, None]
        ):
            if pixel_pose and pixel_pose.shape is None:
                pixel_pose = None

            laser_extra = range_image.tensor[..., 1:4]
            
            points_tensor: tf.Tensor = convert_range_image_to_point_cloud(
                range_image=range_image,
                calibration=lidar_calibration,
                pixel_pose=pixel_pose,
                frame_pose=frame_pose,
                extra_tensor=laser_extra
            )
            points_all[lidar.key.frame_timestamp_micros].append(points_tensor.numpy().astype(np.float32))
            
    points_all = {k: np.concatenate(points_all[k], axis=0) for k in points_all}
    
    if extract_instance:
        return extract_instance_pointcloud(sequence_name, buffer_path, points_all)
    else:
        if not (buffer_path / 'waymo_pkls').exists():
            os.makedirs(buffer_path / 'waymo_pkls')
        pkl.dump(points_all, open(buffer_path / 'waymo_pkls' / sequence_name.replace('.parquet', '.pkl'), 'wb'))
        return None


def extract_instance_pointcloud(sequence_name, buffer_path, points_all=None):
    sql = f"""
    SELECT *
    FROM '{str(raw_path / 'lidar_hkp' / sequence_name)}' lidar_hkp
    LEFT JOIN '{str(raw_path / 'lidar_box' / sequence_name)}' lidar_box
    ON lidar_hkp.index = lidar_box.index and lidar_hkp."key.laser_object_id" = lidar_box."key.laser_object_id"
    ORDER BY lidar_hkp.index
    """
    seq_df = duckdb.sql(sql).to_df()
    points_all = (points_all if points_all is not None 
                  else pkl.load(open(Path(buffer_path) / sequence_name.replace('.parquet', '.pkl'), 'rb')))
    db = []
    
    for i, row in seq_df.iterrows():
        lidar_box = v2.LiDARBoxComponent.from_dict(row)
        lidar_hkp = v2.LiDARHumanKeypointsComponent.from_dict(row)
        keypoints_3d = OrderedDict({k.value: np.zeros(4) for k in KeypointType})
        for t, x, y, z, vis, in zip(
            lidar_hkp.lidar_keypoints.type,
            lidar_hkp.lidar_keypoints.keypoint_3d.location_m.x,
            lidar_hkp.lidar_keypoints.keypoint_3d.location_m.y,
            lidar_hkp.lidar_keypoints.keypoint_3d.location_m.z,
            lidar_hkp.lidar_keypoints.keypoint_3d.visibility.is_occluded,
        ):
            keypoints_3d[t] = np.array([x, y, z, 1 if vis else 2])
        keypoints_3d = np.stack([*keypoints_3d.values()])
        bbox = np.array([
            lidar_box.box.center.x,
            lidar_box.box.center.y,
            lidar_box.box.center.z,
            lidar_box.box.size.x,
            lidar_box.box.size.y,
            lidar_box.box.size.z,
            lidar_box.box.heading,
        ])
        
        point_indices = box_np_ops.points_in_rbbox(points_all[lidar_box.key.frame_timestamp_micros], 
                                                   bbox[None, ...])[:, 0]
        coord = points_all[lidar_box.key.frame_timestamp_micros][point_indices, :].copy()
        instance_info = dict(
            segment_context_name=lidar_box.key.segment_context_name,
            frame_timestamp_micros=lidar_box.key.frame_timestamp_micros,
            object_id=lidar_hkp.key.laser_object_id,
            type=lidar_box.type,
            num_lidar_points_in_box=lidar_box.num_lidar_points_in_box,
            keypoints_3d=keypoints_3d,
            coord=coord,
            bbox=bbox,
        )
        db.append(instance_info)
    return db


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw-path', type=str, default='/mnt/datasets/waymo_open_dataset_v_2_0_0')
    parser.add_argument('--buffer-path', type=str, default='./data/waymo_v2')
    parser.add_argument('--extract-instance', action='store_true', default=True)
    parser.add_argument('--nproc', type=int, default=2)
    args = parser.parse_args()
    print(args)
    for split in ['training', 'validation']:
        raw_path = Path(args.raw_path) / split
        assert raw_path.exists(), f"Raw data not found in {raw_path}"
        buffer_path = Path(args.buffer_path)
        sequence_names = [*map(lambda x: x.name, (raw_path / 'lidar').glob('*.parquet'))]
        fn = partial(process_sequence_pointcloud, 
                     raw_path=raw_path, 
                     buffer_path=buffer_path, 
                     extract_instance=args.extract_instance)
        results = track_parallel_progress(fn, sequence_names, nproc=args.nproc)
        pkl.dump(reduce(list.__add__, results), open(buffer_path / f'{split}.pkl', 'wb'))