skeleton = [
    "center",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

keypoint_range = [*range(15)]
keypoint_num = len(keypoint_range)
keypoint_flip_index = [keypoint_range.index(i) for i in [0, 2, 1, 4, 3, 5, 6, 7, 8, 10, 9, 12, 11, 14, 13, ] if i in keypoint_range]

subset=dict(
    HEAD=[7, 8],
    SHOULDERS=[9, 10],
    ELBOWS=[11, 12],
    WRISTS=[13, 14],
    HIP=[1, 2],
    KNEE=[3, 4],
    ANKLES=[5, 6],
    ALL=[*range(1, 15)]
)

metrics = [
    dict(type='MPJPEMetric', subsets=subset, pa=False, keypoint_range=keypoint_range),
    dict(type='MPJPEMetric', subsets=subset, pa=True, keypoint_range=keypoint_range),
    dict(type='PCKMetric', thres=0.3),
    dict(type='PCKMetric', thres=0.5),
]