import torch
import abc
from torch import Tensor
from typing import List, Dict, Any, Optional

from pointcept.utils.registry import Registry

METRICS = Registry("metrics")


def build_metric(cfg):
    """Build datasets."""
    return METRICS.build(cfg)

@METRICS.register_module()
class MPJPEMetric:
    def __init__(self, subsets=None, pa=False, keypoint_range=None) -> None:
        self.weights = []
        self.sum = []
        subsets = subsets or dict(
            HEAD=[0, 13],
            SHOULDERS=[1, 7],
            ELBOWS=[2, 8],
            WRISTS=[3, 9],
            HIP=[4, 10],
            KNEE=[5, 11],
            ANKLES=[6, 12],
            ALL=[*range(14)]
        )
        if keypoint_range is not None:
            subsets = {k: [keypoint_range.index(v) for v in vs if v in keypoint_range] for k, vs in subsets.items()}
        self.subsets = subsets
        self.pa = pa
    
    def _similarity_trasform_batch(self, S1, S2, vis):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0, 2, 1)

        return S1_hat
    
    def update(self, 
               batch: Dict[str, Tensor]):
        pred = batch['pred_keypoints_3d'][..., :3]
        gt = batch['keypoints_3d'][..., :3]
        weight = batch['keypoints_3d'][..., 3]
        if self.pa:
            pred = self._similarity_trasform_batch(pred, gt, weight)
        error = (pred - gt).pow(2.0).sum(dim=-1).sqrt() * weight
        self.sum.append(error)
        self.weights.append(weight)
    
    def results(self):
        sum_all = torch.cat(self.sum)
        weights_all = torch.cat(self.weights)
        mask = weights_all > 0
        weights_all[~mask] = 1e-7
        result_dict = {
            name if not self.pa else f"PA-{name}": 
                (sum_all[..., indices].sum(-1) / 
                 weights_all[..., indices].sum(-1)).mean().item()
            for name, indices in self.subsets.items()
        }
        return result_dict
    
    def __repr__(self) -> str:
        return '\n' + '\n'.join(f"{k}: {v:.5f}" for k, v in self.results().items())


@METRICS.register_module()
class PCKMetric:
    def __init__(self, thres, subsets=None, torsal_length=0.5127067):
        self.thres = thres
        self.subsets = subsets
        self.sum = []
        self.weights = []
        self.torsal_length = torsal_length
        
    def update(self, 
               batch: Dict[str, Tensor]):
        pred = batch['pred_keypoints_3d'][..., :3]
        gt = batch['keypoints_3d'][..., :3]
        weight = batch['keypoints_3d'][..., 3]
        error = pred.sub(gt).norm(dim=-1)
        self.sum.append(error)
        self.weights.append(weight)
    
    def results(self):
        sum_all = torch.cat(self.sum)
        weights_all = torch.cat(self.weights)
        result_dict = {
            f"PCK-{self.thres * 100:.0f}": 
            ((sum_all < self.thres * self.torsal_length).mul(weights_all).sum(-1) / weights_all.sum(-1)).mean().item()
        }
        return result_dict
    
    def __repr__(self) -> str:
        return '\n' + '\n'.join(f"{k}: {v * 100:.2f}" for k, v in self.results().items())
    

@METRICS.register_module()
class AccelError:
    def __init__(self, subsets=None):
        self.subsets = subsets
        self.preds = []
        self.gts = []
        self.weights = []
        
    def update(self, 
               pred: torch.Tensor, 
               gt: torch.Tensor, 
               weight: torch.Tensor):
        """
        pred: B K 3
        gt: B K 3
        weight: B K
        """
        self.preds.append(pred)
        self.gts.append(gt)
        self.weights.append(weight)
    
    def results(self):
        preds_all = torch.cat(self.preds)
        gts_all = torch.cat(self.gts)
        weights_all = torch.cat(self.weights)
        
        accel_gt = gts_all[:-2] - 2 * gts_all[1:-1] + gts_all[2:]
        accel_pred = preds_all[:-2] - 2 * preds_all[1:-1] + preds_all[2:]

        normed = (accel_pred - accel_gt).norm(dim=-1)
        
        invis = weights_all < 0.5
        invis1 = torch.roll(invis, -1)
        invis2 = torch.roll(invis, -2)
        new_vis = ~(invis | invis1 | invis2)[:-2]
        
        result_dict = {
            f"AE": 
            normed[new_vis].mean().item()
        }
        return result_dict
    
    def __repr__(self) -> str:
        return '\n' + '\n'.join(f"{k}: {v * 1000:.2f}" for k, v in self.results().items())