import torch
import numpy as np


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def pointcloud_to_voxel(pointcloud, xbound, ybound, zbound):
    # x front, y left, z up, in vehicle coordirate
    dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
    voxel = torch.zeros((nx[2], nx[0], nx[1]), dtype=torch.float32)
    idx_x = ((pointcloud[:, 1] - bx[1]) / dx[1]).long()
    idx_y = ((pointcloud[:, 0] - bx[0]) / dx[0]).long()
    idx_z = ((pointcloud[:, 2] - bx[2]) / dx[2]).long()
    mask = (idx_x >= 0) * (idx_x < nx[1]) * (idx_y >= 0) * (idx_y < nx[0]) * (idx_z >= 0) * (idx_z < nx[2])
    voxel[idx_z[mask], idx_y[mask], idx_x[mask]] = 1
    return voxel
