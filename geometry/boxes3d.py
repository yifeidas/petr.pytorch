# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np
import torch
from pyquaternion import Quaternion
from pytorch3d.transforms import transform3d as t3d
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from torch.cuda import amp

# yapf: disable
BOX3D_CORNER_MAPPING = [
    [1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1]
]
# yapf: enable


def _to_tensor(x, dim):
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32)
    elif isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, tuple):
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, Quaternion):
        x = torch.tensor(x.elements, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported type: {type(x).__name__}")

    if x.ndim == 1:
        x = x.reshape(-1, dim)
    elif x.ndim > 2:
        raise ValueError(f"Invalid shape of input: {x.shape.__str__()}")
    return x


class GenericBoxes3D():
    def __init__(self, quat, tvec, size):
        self.quat = _to_tensor(quat, dim=4)
        self.tvec = _to_tensor(tvec, dim=3)
        self.size = _to_tensor(size, dim=3)

    # @property
    # def tvec(self):
    #     return self._tvec

    @property
    @amp.autocast(enabled=False)
    def corners(self):
        translation = t3d.Translate(self.tvec, device=self.device)

        R = quaternion_to_matrix(self.quat)
        rotation = t3d.Rotate(R=R.transpose(1, 2), device=self.device)  # Need to transpose to make it work.

        tfm = rotation.compose(translation)

        _corners = 0.5 * self.quat.new_tensor(BOX3D_CORNER_MAPPING).T
        # corners_in_obj_frame = self.size.unsqueeze(1) * _corners.unsqueeze(0)
        lwh = self.size[:, [1, 0, 2]]  # wlh -> lwh
        corners_in_obj_frame = lwh.unsqueeze(1) * _corners.unsqueeze(0)

        corners3d = tfm.transform_points(corners_in_obj_frame)

        return corners3d

    @classmethod
    def from_vectors(cls, vecs, device="cpu"):
        """
        Parameters
        ----------
        vecs: Iterable[np.ndarray]
            Iterable of 10D pose representation.

        intrinsics: np.ndarray
            (3, 3) intrinsics matrix.
        """
        quats, tvecs, sizes = [], [], []
        for vec in vecs:
            quat = vec[:4]
            tvec = vec[4:7]
            size = vec[7:]

            quats.append(quat)
            tvecs.append(tvec)
            sizes.append(size)

        quats = torch.as_tensor(quats, dtype=torch.float32, device=device)
        tvecs = torch.as_tensor(tvecs, dtype=torch.float32, device=device)
        sizes = torch.as_tensor(sizes, device=device)

        return cls(quats, tvecs, sizes)

    @classmethod
    def cat(cls, boxes_list, dim=0):

        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0), torch.empty(0), torch.empty(0))
        assert all([isinstance(box, GenericBoxes3D) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        quat = torch.cat([b.quat for b in boxes_list], dim=dim)
        tvec = torch.cat([b.tvec for b in boxes_list], dim=dim)
        size = torch.cat([b.size for b in boxes_list], dim=dim)

        cat_boxes = cls(quat, tvec, size)
        return cat_boxes

    def split(self, split_sizes, dim=0):
        assert sum(split_sizes) == len(self)
        quat_list = torch.split(self.quat, split_sizes, dim=dim)
        tvec_list = torch.split(self.tvec, split_sizes, dim=dim)
        size_list = torch.split(self.size, split_sizes, dim=dim)

        return [GenericBoxes3D(*x) for x in zip(quat_list, tvec_list, size_list)]

    def __getitem__(self, item):
        """
        """
        if isinstance(item, int):
            return GenericBoxes3D(self.quat[item].view(1, -1), self.tvec[item].view(1, -1), self.size[item].view(1, -1))

        quat = self.quat[item]
        tvec = self.tvec[item]
        size = self.size[item]

        assert quat.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert tvec.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert size.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)

        return GenericBoxes3D(quat, tvec, size)

    def __len__(self):
        assert len(self.quat) == len(self.tvec) == len(self.size)
        return self.quat.shape[0]

    def clone(self):
        """
        """
        return GenericBoxes3D(self.quat.clone(), self.tvec.clone(), self.size.clone())

    def vectorize(self):
        xyz = self.tvec
        return torch.cat([self.quat, xyz, self.size], dim=1)

    @property
    def device(self):
        return self.quat.device

    def to(self, *args, **kwargs):
        quat = self.quat.to(*args, **kwargs)
        tvec = self.tvec.to(*args, **kwargs)
        size = self.size.to(*args, **kwargs)
        return GenericBoxes3D(quat, tvec, size)


def box3d_ang_to_box3d_quat(box3d):
    # input: Nx10 [[cls, prob, x,y,z,w,l,h,sin,cos]]
    # output: Nx12 [[cls, prob, q0,q1,q2,q3,x,y,z,w,l,h]]
    if box3d.size(1) == 10:
        pc = box3d[:, 0:2]
        xyzwlh = box3d[:, 2:8]
        ang = box3d[:, 8:10]
    else:
        pc = box3d[:, 0:1]
        xyzwlh = box3d[:, 1:7]
        ang = box3d[:, 7:9]
    N = box3d.size(0)
    if N == 0:
        return torch.empty(0, 12)
    rot = torch.atan2(ang[:, 0], ang[:, 1])
    rot_mat = torch.zeros(N, 3, 3)
    rot_mat[:, 2, 2] = 1
    rot_mat[:, 0, 0] = rot.cos()
    rot_mat[:, 1, 1] = rot.cos()
    rot_mat[:, 0, 1] = -rot.sin()
    rot_mat[:, 1, 0] = rot.sin()
    quat_list = []
    for i in range(N):
        rot = rot_mat[i, :, :].numpy()
        q = Quaternion._from_matrix(rot, rtol=1e-3, atol=1e-5)
        quat = torch.from_numpy(q.q).float()
        quat_list.append(quat)
    quat_v = torch.stack(quat_list, dim=0)
    box_out = torch.cat([pc, quat_v, xyzwlh], dim=1)
    return box_out.clone()