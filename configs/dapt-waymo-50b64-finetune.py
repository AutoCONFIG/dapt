_base_ = ["base.py", "base_waymo.py"]

num_epochs = 50
lr = 5e-4
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=5e-4,
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0
)

cc_grid_params = dict(
    grid_size=(72, 72, 120),
    coord_range=[[-0.9, 0.9], 
                [-0.9, 0.9], 
                [-1.5, 1.5]],
    sigma=2.0,
)

model = dict(
    type="MultiTaskPoseNet",
    pretrained=None,
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
            type='Coord3dClsHead',
            num_keypoints={{ _base_.keypoint_num }},
            in_channel=128, 
            beta=1.0,
            use_dark=True,
            use_joint_wise_weight=False,
            loss_weight=1.0,
            **cc_grid_params
        ),
        inter_feat=dict(
            type='InterCoord3dHead',
            in_channel=512,
            num_keypoints={{ _base_.keypoint_num }},
            loss_weight=0.2
        )
    )
)

train_pipeline = [
    dict(type="Centering"),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomJitter",sigma=0.005, clip=0.02),
    dict(type="RandomFlip", p=0.3, keypoint_flip_index={{ _base_.keypoint_flip_index }}),
    dict(type="RandomDropout",),
    dict(type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color"),
        return_grid_coord=True,
    ),
    dict(type="ShufflePoint",),
    dict(type="GenerateTargetCoordCls", **cc_grid_params),
    dict(type="ToTensor",),
    dict(type="Collect",
        keys=(
            "coord", 
            "grid_coord",
            "keypoints_3d",
            "coord_label_x",
            "coord_label_y",
            "coord_label_z",
        ),
        feat_keys=("coord",),
    )
]

test_pipeline = [
    dict(type="Centering"),
    dict(type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color"),
        return_grid_coord=True,
    ),
    dict(type="ToTensor",),
    dict(type="Collect",
         keys=("coord", "grid_coord", "keypoints_3d"),
         feat_keys=("coord",),
    )
]

data = dict(
    num_workers=8,
    train_batch_size = 64,
    val_batch_size = 128,
    train_dataset=dict(
        type='WaymoPoseDataset',
        modality=['lidar',],
        split='training',
        keypoint_range={{ _base_.keypoint_range }},
        transforms=train_pipeline
    ),
    val_dataset=dict(
        type='WaymoPoseDataset',
        modality=['lidar',],
        split='validation',
        keypoint_range={{ _base_.keypoint_range }},
        transforms=test_pipeline
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
