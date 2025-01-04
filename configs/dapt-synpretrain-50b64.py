_base_ = ["base.py", "base_smpl_subset.py"]

num_epochs = 50
lr = 3e-4
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0
)

model = dict(
    type="MultiTaskPoseNet",
    backbone=dict(
        type='PT-v3m1-dapt',
        num_keypoints={{ _base_.keypoint_num }},
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(128, 128, 128, 256),
        dec_kp_channels=(128, 128, 128, 256, 256),
        dec_kp_mixer='attn',
        dec_num_head=(8, 8, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    heads=dict(
        query=dict(
            type='Coord3dHead',
            in_channel=128, 
            loss_weight=0.5
        ),
        feat=dict(
            type='SegmentationHead',
            in_channel=128,
            num_classes=25,
            ignore_class=0,
            loss_weight=1.0
        )
    )
)

train_pipeline = [
    dict(
        type="GenerateRayCastingScene",
        alpha_range=(-1, 1, 2650),
        gamma_range=(-1 / 18, 1 / 18, 64),
        center=(0, 0, 0.5),
        center_dist_range=(5, 20),
        ground_plane_normal_angle=1/36,
        range_noise=0.02,
        keep_rate=0.6),
    dict(
        type="GenerateNoisePoints",
        center_range=((-.4, .4), (-.4, .4), (-.8, .8)),
        radius=0.3,
        num_area=2,
        num_points_max=150,
    ),
    dict(type="BodyPartRemoval"),
    dict(type="Centering"),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        keys=("coord", "segment"),
        return_grid_coord=True,
    ),
    dict(type="ShufflePoint",),
    dict(type="ToTensor",),
    dict(type="Collect",
        keys=("coord", 
                "grid_coord", 
                "keypoints_3d",
                "segment"
            ),
        feat_keys=("coord",),
    )
]

test_pipeline = [
    dict(
        type="GenerateRayCastingScene",
        alpha_range=(-1, 1, 2650),
        gamma_range=(-1 / 18, 1 / 18, 64),
        center=(0, 0, 0.5),
        center_dist_range=(5, 20),
        ground_plane_normal_angle=1/36,
        range_noise=0.02),
    dict(type="Centering"),
    dict(type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        keys=("coord", "segment"),
        return_grid_coord=True,
    ),
    dict(type="ToTensor",),
    dict(type="Collect",
         keys=("coord", "grid_coord", "keypoints_3d", "segment"),
         feat_keys=("coord",),
    )
]

data = dict(
    num_workers=8,
    train_batch_size = 64,
    val_batch_size = 128,
    train_dataset=dict(
        type='LiDARH26MPoseDataset',
        modality=['lidar'],
        split='train',
        transforms=train_pipeline,
        keypoint_range={{ _base_.keypoint_range }},
        interval=2,
    ),
    val_dataset=dict(
        type='LiDARH26MPoseDataset',
        modality=['lidar'],
        split='test',
        transforms=test_pipeline,
        keypoint_range={{ _base_.keypoint_range }},
        interval=5,
    )
)

exp_name ="{{fileBasenameNoExtension}}"
work_dir = f"./work_dir/{exp_name}"
tracker = dict(
    project_name='pc_pose', 
    init_kwargs=dict(
        wandb=dict(name=exp_name)
    )
)
