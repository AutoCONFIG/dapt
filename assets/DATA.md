## SMPL Models (for pretraining)
1. Download SMPL Model (10 shape PCs) from [SMPL official website](https://smpl.is.tue.mpg.de/).
2. Ensure the `smpl_models` folder has the following structure:
```
smpl_models
|-- smpl
|   |-- SMPL_FEMALE.npz
|   |-- SMPL_FEMALE.pkl
|   |-- SMPL_MALE.npz
|   |-- SMPL_MALE.pkl
|   |-- SMPL_NEUTRAL.npz
|   `-- SMPL_NEUTRAL.pkl
|-- smpl_body_parts_2_faces.json
`-- smpl_vert_segmentation.json
```
## LiDARHuman2.6M
1. Download from [LiDARCap official website](http://www.lidarhumanmotion.net/lidarcap/).
2. Use the provided script for LiDARHuman2.6M preparation:
```bash
conda activate dapt
python -m tools.prepare_lidarh26m \
  --raw-path /path/to/raw \
  --buffer-path ./data/lidarh26m
```
## Waymo Open Dataset v2.0
1. Install [gcloud](https://cloud.google.com/storage/docs/discover-object-storage-gcloud).
2. Grant access of Waymo Open Dataset v2.0 from [Waymo official website](https://waymo.com/open/download/), note we only use the following components:
```bash
gcloud auth login

gcloud storage cp -r \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_box" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_calibration" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_hkp" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_pose" \
  "gs://waymo_open_dataset_v_2_0_0/training/vehicle_pose" \
  /path/to/raw/training

gsutil -m cp -r \
  "gs://waymo_open_dataset_v_2_0_0/validation/lidar" \
  "gs://waymo_open_dataset_v_2_0_0/validation/lidar_box" \
  "gs://waymo_open_dataset_v_2_0_0/validation/lidar_calibration" \
  "gs://waymo_open_dataset_v_2_0_0/validation/lidar_hkp" \
  "gs://waymo_open_dataset_v_2_0_0/validation/lidar_pose" \
  "gs://waymo_open_dataset_v_2_0_0/validation/vehicle_pose" \
  /path/to/raw/validation
```

3. Create a temporary environment and use the provided script for WOD preparation, this operation may **take hours to finish**:
```bash
conda create -f wod_environment.yml
conda activate wod
python -m tools.prepare_waymopose \
  --raw-path /path/to/raw \
  --buffer-path ./data/waymo_v2 \
  --nproc 2
```
If you want to use the processed data, download from [**here**](https://1drv.ms/f/s!As3g2ozBLMgFkJFfEzkp5jvmV7Vp3A?e=pRgQFW).
## SLOPER4D
1. Download from [SLOPER4D official website](http://www.lidarhumanmotion.net/data-sloper4d/).
2. Note that SLOPER4D has no train-test split, the generation process follows [Neighborhood Enhanced LiDARCap](https://ojs.aaai.org/index.php/AAAI/article/view/28545):
```bash
conda activate dapt
python -m tools.prepare_sloper4d \
  --raw-path /path/to/raw \
  --buffer-path ./data/sloper4d
```
## HumanM3
1. Download & run data extraction following [Human-M3 official repository](https://github.com/soullessrobot/Human-M3-Dataset), ensure the integrity of db files `db/collection/train.pkl` and `db/collection/test.pkl`.
2. Copy / soft link `train.pkl` and `test.pkl` to `data/humanm3`