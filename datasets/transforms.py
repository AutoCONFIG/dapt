from math import ceil
import random
import numpy as np
import open3d as o3d
from pathlib import Path
import cv2 

from pointcept.datasets.transform import TRANSFORMS, GridSample

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def resized_center_crop(img: np.ndarray, 
                        tgt_size):
    h, w = img.shape[:2]
    tw, th = tgt_size

    if tw > th:
        m = np.array([[tw / w, 0, 0],
                      [0, tw / w, (th - tw) / 2]])
    else:
        m = np.array([[th / h, 0, (tw - th) / 2],
                      [0, th / h, 0]])

    img = cv2.warpAffine(img, m, tgt_size)
    return img

def orthogonal2radial(points: np.ndarray, center: np.ndarray):
    points = points - center
    r = np.linalg.norm(points, axis=1)
    a = np.arctan2(points[:, 1], points[:, 0])
    g = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))
    return np.stack([r, a, g], axis=1)


@TRANSFORMS.register_module()
class ReSample:
    def __init__(self, 
                 num_points=1024,
                 pad_points=False,
                 keys=['color', 'segment']):
        self.num_points = num_points
        self.pad_points = pad_points
        self.keys = keys
    
    def __call__(self, sample):
        now_pt_num = int(sample['coord'].shape[0])

        if now_pt_num > self.num_points:
            choice_indx = np.random.randint(0, now_pt_num, size = [self.num_points])
            sample['coord'] = sample['coord'][choice_indx,:]
            for k in self.keys:
                if k in sample:
                    sample[k] = sample[k][choice_indx]
        elif self.pad_points:
            choice_indx = np.random.randint(0, now_pt_num, size = [self.num_points - now_pt_num])
            sample['coord'] = np.concatenate([sample['coord'], sample['coord'][choice_indx]], axis=0)
            for k in self.keys:
                if k in sample:
                    sample[k] = np.concatenate([sample[k], sample[k][choice_indx]], axis=0)
       
        return sample


@TRANSFORMS.register_module()
class Centering:
    def __init__(self, center_key='center'):
        self.center_key = center_key

    def __call__(self, sample):
        if self.center_key in sample:
            center = sample[self.center_key]
        else:
            center = sample['bbox'][None, :3]
        sample['coord'] = sample['coord'] - center
        sample['keypoints_3d'][..., :3] -= center
        return sample


@TRANSFORMS.register_module()
class GeneratePointLabels:
    def __init__(self, dist_threshold=0.1, overwrite=False) -> None:
        self.dist_threshold = dist_threshold
        self.overwrite = overwrite
    
    def __call__(self, sample):
        if 'segment' in sample and not self.overwrite:
            return sample
        dist = np.linalg.norm(sample['coord'][:, None, :3] - sample['keypoints_3d'][None, :, :3], axis=-1)
        label, vmin = dist.argmin(axis=-1), dist.min(axis=-1)
        mask = (vmin > self.dist_threshold) | (sample['keypoints_3d'][label, 3] < 0.5)
        label += 1
        label[mask] = 0
        sample['segment'] = label
        return sample


@TRANSFORMS.register_module()
class GenerateTargetCoordCls:
    def __init__(self, 
                coord_range=[[-0.9, 0.9], 
                            [-0.9, 0.9], 
                            [-1.5, 1.5]],
                grid_size=[72, 72, 120],
                sigma=5.0) -> None:
        self.coord_range = np.array(coord_range)
        self.grid_size = np.array(grid_size)
        self.sigma = np.array([sigma]).reshape(-1)
    
    def __call__(self, sample):
        K, C = sample['keypoints_3d'].shape[-2:]
        kpts = sample['keypoints_3d'].reshape(K, C)
        x, y, z = [*map(np.arange, self.grid_size)]

        grid_coord = (kpts[..., :3] - self.coord_range[None, :, 0]) \
            * self.grid_size / (self.coord_range[:, 1] - self.coord_range[:, 0])
            
        vis = sample['keypoints_3d_vis'].reshape(K, 1) if 'keypoints_3d_vis' in sample else kpts[..., 3]

        sample['coord_label_x'] = np.exp(-((x[None, :] - grid_coord[:, 0:1]) ** 2) / 2 / self.sigma ** 2) * vis[:, None]
        sample['coord_label_y'] = np.exp(-((y[None, :] - grid_coord[:, 1:2]) ** 2) / 2 / self.sigma ** 2) * vis[:, None]
        sample['coord_label_z'] = np.exp(-((z[None, :] - grid_coord[:, 2:3]) ** 2) / 2 / self.sigma ** 2) * vis[:, None]
        return sample

def mesh_by_normals(
    center: np.ndarray, 
    normal: np.ndarray, 
    plane_size=1.0
):
    u = np.cross(normal, np.array([0, 1, 0]))
    if np.linalg.norm(u) == 0:
        u = np.cross(normal, np.array([1, 0, 0]))
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    vertices = np.array([
        center + plane_size * (-u - v),
        center + plane_size * (u - v),
        center + plane_size * (u + v),
        center + plane_size * (-u + v)
    ])

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_triangle_normals()
    return o3d.t.geometry.TriangleMesh.from_legacy(mesh)


@TRANSFORMS.register_module()
class GenerateRayCastingScene:
    def __init__(
        self,
        alpha_range=(-1, 1, 2650),
        gamma_range=(-1 / 18, 1 / 18, 64),
        center=(0, 0, 0.0),
        center_dist_range=(5, 20),
        ground_plane=True,
        ground_plane_normal_angle=1/18,
        range_noise=0.02,
        keep_rate=0.8,
        sample_location=True,
    ):
        self.alpha_range = np.linspace(*alpha_range) * np.pi
        self.gamma_range = np.linspace(*gamma_range) * np.pi
        self.center = np.array(center)
        
        self.center_dist_range = center_dist_range
        self.ground_plage = ground_plane
        self.ground_plane_normal_angle = ground_plane_normal_angle
        self.range_noise = range_noise
        
        alpha_range = self.alpha_range[:, None]
        gamma_range = self.gamma_range[None, :]
        direction = np.stack([
            (np.cos(alpha_range) * np.cos(gamma_range)),
            (np.sin(alpha_range) * np.cos(gamma_range)),
            (np.ones_like(alpha_range) * np.sin(gamma_range))
        ], axis=-1)

        rays = np.concatenate([self.center[None, None, :].repeat(direction.shape[0], 0).repeat(direction.shape[1], 1), 
                               direction], 
                               axis=-1, dtype=np.float32)
        self.rays = o3d.core.Tensor(np_array=rays)
        self.keep_rate = keep_rate
        self.sample_location = sample_location

    def __call__(self, sample):
        """
        use:
            face, vertex
        modify:
            
        """
        assert 'face' in sample
        assert 'vertex' in sample
        
        vertex = sample['vertex']
        face = sample['face']
        
        # centering human
        if 'bbox' in sample:
            offset = -sample['bbox'][None, :3]
        else:
            offset = 0

        if self.sample_location:        
            # random rotation & transition on xy
            near, far = self.center_dist_range
            rot_z, glob_alpha, glob_range = (
                np.random.rand(3) 
                * np.array([2 * np.pi, 2 * np.pi, far - near]) 
                + np.array([0., 0., near]))
            r_mat = np.array([
                [np.cos(rot_z), -np.sin(rot_z), 0],
                [np.sin(rot_z), np.cos(rot_z), 0],
                [0., 0., 1.],
            ])
            t_mat = np.array([glob_range * np.cos(glob_alpha), glob_range * np.sin(glob_alpha), -offset[0, 2] - vertex[:, 2].min()])
        else:
            r_mat = np.eye(3)
            t_mat = sample['bbox'][None, :3]
        vertex = (vertex + offset) @ r_mat.T + t_mat

        ## point cloud generation
        # laser filtering
        points_radial = orthogonal2radial(vertex, self.center)
        points_range_min, points_range_max = points_radial.min(axis=0), points_radial.max(axis=0)

        if points_range_max[1] - points_range_min[1] > 1.8 * np.pi:
            a = points_radial[:, 1]
            a[a < 0] += 2 * np.pi
            alpha_mask = (self.alpha_range > a.min() - 0.01 * np.pi) | (self.alpha_range < a.max() - 1.99 * np.pi)
        else:
            alpha_mask = (self.alpha_range > points_range_min[1] - 0.01 * np.pi) & (self.alpha_range < points_range_max[1] + 0.01 * np.pi)
        gamma_mask = (self.gamma_range > points_range_min[2] - 0.01 * np.pi) & (self.gamma_range < points_range_max[2] + 0.01 * np.pi)
        
        rays = self.rays[alpha_mask, :][:, gamma_mask]
        
        # area ray masking
        mask_size = min(rays.shape[:2]) // 8
        mH, mW = ceil(rays.shape[0] / mask_size), ceil(rays.shape[1] / mask_size)
        mask = np.zeros((mH * mW, mask_size * mask_size), dtype=bool)
        mask[np.random.permutation(mH * mW)[:int(mH * mW * self.keep_rate)], :] = True
        mask = mask \
            .reshape(mH, mW, mask_size, mask_size) \
            .transpose((0, 2, 1, 3)) \
            .reshape(mH * mask_size, mW * mask_size) \
            [:rays.shape[0], :rays.shape[1]].flatten()

        rays = rays.reshape((-1, 6))[mask, :]

        # random ground plane
        ground_plane_center = vertex.mean(axis=0) * np.array([1, 1, 0]) + np.array([0, 0, vertex[:, 2].min()])
        normal_gamma = (0.5 - np.random.randn() * self.ground_plane_normal_angle) * np.pi
        normal_alpha = (np.random.rand() - 0.5) * 2 * np.pi
        ground_plane_normal = np.array([
            np.cos(normal_alpha) * np.cos(normal_gamma),
            np.sin(normal_alpha) * np.cos(normal_gamma),
            np.sin(normal_gamma)
        ])
        ground_plane = mesh_by_normals(ground_plane_center, ground_plane_normal, 1.0 if self.ground_plage else 0.001)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertex)
        mesh.triangles = o3d.utility.Vector3iVector(face)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        scene.add_triangles(ground_plane) 
        
        # casting rays
        result = scene.cast_rays(rays)
        hit = result['t_hit'].isfinite()
        t_hit = result['t_hit'][hit]
        i_hit = result['geometry_ids'][hit]
        l_hit = result['primitive_ids'][hit].to(o3d.core.Dtype.Int32)
        l_hit[(i_hit==0).logical_not_()] = -1
        t_hit = t_hit + o3d.core.Tensor(np_array=np.random.random((t_hit.shape[0])).astype(np.float32) * self.range_noise)
        
        sample['vertex'] = vertex
        sample['dist'] = glob_range
        sample['coord'] = (rays[hit][:,:3] + rays[hit][:,3:] * t_hit.reshape((-1,1))).cpu().numpy()
        if 'face_label' in sample:
            label = np.append(sample['face_label'], 0)
            sample['segment'] = label[l_hit.cpu().numpy()]
        if 'keypoints_3d' in sample:
            sample['keypoints_3d'][:, :3] = ((sample['keypoints_3d'][:, :3] + offset) @ r_mat.T) + t_mat
        if 'bbox' in sample:
            sample['bbox'][:3] = t_mat
        return sample

@TRANSFORMS.register_module()
class GenerateNoisePoints:
    def __init__(
        self,
        center_range=((-.5, .5), (-.5, .5), (-.8, .8)),
        radius=0.3,
        num_area=2,
        num_points_max=100,
        p=0.5
    ):
        self.center_range =np.array(center_range)
        self.radius = radius
        self.num_area = num_area
        self.p = p
        self.num_points_max = num_points_max

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample
        center = np.random.uniform(self.center_range[:, 0], self.center_range[:, 1], (self.num_area, 3))
        if 'bbox' in sample:
            center += sample['bbox'][:3]
        n_points = np.random.randint(0, self.num_points_max, self.num_area)
        points = []
        for i in range(self.num_area):
            r = np.random.uniform(0.0, self.radius, n_points[i])
            theta = np.random.uniform(0.0, np.pi, n_points[i])
            phi = np.random.uniform(0.0, 2*np.pi, n_points[i])
            points.append(center[i] + np.stack([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ], axis=-1))

        sample['coord'] = np.concatenate([sample['coord'], *points], axis=0)
        sample['segment'] = np.concatenate([sample['segment'], np.zeros(n_points.sum(), dtype=np.int32)], axis=0)
        return sample


@TRANSFORMS.register_module()
class BodyPartRemoval:
    def __init__(
        self,
        p=0.5,
        num_removes=1,
    ):
        self.p = p
        self.num_removes = num_removes

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample
        labels = np.unique(sample['segment'])
        if labels.shape[0] == 1:
            return sample
        remove_labels = np.random.choice(labels[labels != 0], self.num_removes, replace=False)
        mask = ~np.isin(sample['segment'], remove_labels)
        sample['coord'] = sample['coord'][mask]
        sample['segment'] = sample['segment'][mask]
        return sample


@TRANSFORMS.register_module()
class GetBodyDirection:
    def __init__(self):
        pass

    def __call__(self, sample):
        # Face direction
        body_dir = sample['keypoints_3d'][[1, 9], :2] - sample['keypoints_3d'][[2, 10], :2]
        body_dir = body_dir.mean(axis=0) @ np.array([[0, -1], [1, 0]])
        sample['directions'] = body_dir
        return sample
    

@TRANSFORMS.register_module(force=True)
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "keypoints_3d" in data_dict.keys():
            data_dict["keypoints_3d"][..., :3] = np.dot(data_dict["keypoints_3d"][..., :3], np.transpose(rot_t))
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module(force=True)
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "keypoints_3d" in data_dict.keys():
            data_dict["keypoints_3d"][..., :3] = np.dot(data_dict["keypoints_3d"][..., :3], np.transpose(rot_t))
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module(force=True)
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        scale = np.random.uniform(
            self.scale[0], self.scale[1], 3 if self.anisotropic else 1
        )
        if "coord" in data_dict.keys():
            data_dict["coord"] *= scale
        if "keypoints_3d" in data_dict.keys():
            data_dict["keypoints_3d"][..., :3] *= scale
        return data_dict


@TRANSFORMS.register_module(force=True)
class RandomFlip(object):
    def __init__(self, p=0.5, keypoint_flip_index=None):
        self.p = p
        self.keypoint_flip_index = keypoint_flip_index or [0, *range(7, 13), *range(1, 7), 13]

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "keypoints_3d" in data_dict.keys():
                data_dict["keypoints_3d"][:, 0] = -data_dict["keypoints_3d"][:, 0]
                data_dict["keypoints_3d"] = data_dict["keypoints_3d"][self.keypoint_flip_index, :]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "keypoints_3d" in data_dict.keys():
                data_dict["keypoints_3d"][:, 1] = -data_dict["keypoints_3d"][:, 1]
                data_dict["keypoints_3d"] = data_dict["keypoints_3d"][self.keypoint_flip_index, :]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict