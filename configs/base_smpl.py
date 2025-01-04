skeleton = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand"
]

keypoint_range = [*range(24)]
keypoint_num = len(keypoint_range)
keypoint_link = [
    (1, 0), (2, 0), (3, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 5), (9, 6), (10, 7), (11, 8), 
    (12, 9), (13, 9), (14, 9), (15, 12), (16, 13), (17, 14), (18, 16), (19, 17), (20, 18), (21, 19), (22, 20), (23, 21)
]
keypoint_flip_index = [keypoint_range.index(i) 
                       for i in [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22] 
                       if i in keypoint_range]
subset = dict(
    HEAD=[12, 15],
    SHOULDERS=[16, 17],
    ELBOWS=[18, 19],
    WRISTS=[20, 21],
    HIP=[1, 2],
    KNEE=[4, 5],
    ANKLES=[7, 8],
    ALL=[*range(24)]
)

metrics = [
    dict(type='MPJPEMetric', subsets=subset, pa=False, keypoint_range=keypoint_range),
    dict(type='MPJPEMetric', subsets=subset, pa=True, keypoint_range=keypoint_range),
    dict(type='PCKMetric', thres=0.3),
    dict(type='PCKMetric', thres=0.5),
    # dict(type='Accel')
]