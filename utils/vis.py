import torch
import numpy as np
import cv2
from geometry.voxelize import pointcloud_to_voxel

def draw_box_in_bev(box_corners, bev_img, xbound, ybound):
    assert box_corners.shape[1] == 4
    assert box_corners.shape[0] == 3
    corner_list = []
    for i in range(4):
        xyz = box_corners[:, i]
        y_idx = (xyz[0] - xbound[0]) / xbound[2]
        x_idx = (xyz[1] - ybound[0]) / ybound[2]
        corner_list.append(np.array([x_idx, y_idx], dtype=np.float32))
    pts = np.stack(corner_list, axis=0).astype(np.int32)
    # print(pts)
    cv2.polylines(bev_img, [pts], True, (0,255,0), 1)
    return bev_img

def draw_voxel_heatmap(voxel, heatmap):
    # voxel = voxel.squeeze(0).cpu().sum(dim=0)
    # bev_pts = (voxel / 4).clamp(0, 1)
    bev_pts = voxel.squeeze().cpu()
    bev_pts = bev_pts.numpy()
    heatmap = heatmap.squeeze(0).cpu()
    heat, _ = heatmap.max(dim=0)
    heat = heat.numpy()
    H, W = heat.shape
    b = np.zeros((H, W), dtype=np.float32)
    img = np.stack((b, bev_pts, heat), axis=2)
    return img

def draw_voxel(voxel, scale=1):
    voxel = voxel.squeeze(0).cpu().sum(dim=0)
    bev_pts = (voxel / scale)#.clamp(0, 1)
    return bev_pts.numpy()

def pointcloud_to_bevimg(pointcloud, xbound, ybound, zbound):
    # x front, y left, z up, in vehicle coordirate
    voxel = pointcloud_to_voxel(pointcloud, xbound, ybound, zbound)
    bev_pts = voxel.sum(dim=0).squeeze().cpu()
    bev_pts = (bev_pts / 4).clamp(0, 1)
    bev_pts = (bev_pts.numpy() * 255).astype(np.uint8)
    bev_img = cv2.cvtColor(bev_pts, cv2.COLOR_GRAY2RGB)
    return bev_img

def draw_box_in_bev(box_corners, bev_img, xbound, ybound, color=(0,255,0)):
    assert box_corners.shape[1] == 4
    assert box_corners.shape[0] == 3
    corner_list = []
    for i in range(4):
        xyz = box_corners[:, i]
        y_idx = (xyz[0] - xbound[0]) / xbound[2]
        x_idx = (xyz[1] - ybound[0]) / ybound[2]
        corner_list.append(np.array([x_idx, y_idx], dtype=np.float32))
    pts = np.stack(corner_list, axis=0).astype(np.int32)
    cv2.polylines(bev_img, [pts], True, color, 1)
    return bev_img