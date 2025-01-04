skeleton = [
    "unspecified",
    "nose",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "forehead",
    "head",
]

link = [
    (), (), ()
]

keypoint_range = [*range(1, 14), 15]
keypoint_num = len(keypoint_range)
keypoint_flip_index = [keypoint_range.index(i) for i in [0, 1, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 14, 15] if i in keypoint_range]

subset = dict(
    HEAD=[1, 15],
    SHOULDERS=[2, 8],
    ELBOWS=[3, 9],
    WRISTS=[4, 10],
    HIP=[5, 11],
    KNEE=[6, 12],
    ANKLES=[7, 13],
    ALL=[*range(1, 14), 15]
)

metrics = [
    dict(type='MPJPEMetric', subsets=subset, pa=False, keypoint_range=keypoint_range),
    dict(type='PCKMetric', thres=0.3),
    dict(type='PCKMetric', thres=0.5),
]