import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms as T 
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
from utils.tools import get_rot_matrix
from geometry.voxelize import pointcloud_to_voxel
from pyquaternion import Quaternion
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
import random
import math
import numpy as np
import os

CAMERA_NAMES = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')
CATEGORY_IDS = {
    'barrier': 0,
    'bicycle': 1,
    'bus': 2,
    'car': 3,
    'construction_vehicle': 4,
    'motorcycle': 5,
    'pedestrian': 6,
    'traffic_cone': 7,
    'trailer': 8,
    'truck': 9,
}

DS_SAMPLE_DICT = {
        'car':2,
        'truck':3,
        'construction_vehicle':7,
        'bus':4,
        'trailer':6,
        'barrier':2,
        'motorcycle':6,
        'bicycle':6,
        'pedestrian':2,
        'traffic_cone':2
}

def category_to_detection_name(category_name: str):
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

def nusc_box3d_to_array(box):
    assert type(box) == Box
    cate = np.array(box.label, dtype=np.float32, ndmin=1)
    yaw, pitch, roll = box.orientation.yaw_pitch_roll
    tvec = box.center.copy()
    size = box.wlh.copy()
    tang = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    return np.concatenate([cate, tvec, size, tang], axis=0)

class NuscenesLoader(Dataset):
    def __init__(self, root_dir, nusc=None, datum_names=CAMERA_NAMES, min_num_lidar_points=5, min_box_visibility=0.2, mode='train'):
        self.root_dir = root_dir
        assert mode in ['train', 'val']
        self.mode = mode
        self.name = 'nuscenes'
        version = "v1.0-trainval"
        # version = "v1.0-mini"
        self.ds_sampling = True
        if nusc is None:
            self.nusc = NuScenes(version=version, dataroot=root_dir, verbose=True)
        else:
            self.nusc = nusc
        self.category_list = list(CATEGORY_IDS.keys())

        self.datum_names = datum_names
        self.min_num_lidar_points = min_num_lidar_points
        self.min_box_visibility = min_box_visibility
        scenes_in_splits = create_splits_scenes()
        self.scenes = scenes_in_splits[mode]
        self.cls_num = 10

        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
        self.samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        # sort by scene, timestamp (only to make chronological viz easier)
        self.samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        print("raw dateset length: ", len(self.samples))
        if self.mode == 'train' and self.ds_sampling:
            class_sample_idxs = {cat_id: [] for cat_id in range(self.cls_num)}
            for i in range(len(self.samples)):
                sample = self.samples[i]
                instance_num_i = np.zeros(10)
                fimage, intrin, rot_CV, trans_CV, box_list = self.get_image_data(i, 'CAM_FRONT')
                cat_list = []
                for box in box_list:
                    cate = box.label
                    cat_list.append(cate)
                    instance_num_i[cate] += 1
                cat_list_unique = np.unique(cat_list).tolist()
                for cate in cat_list_unique:
                    class_sample_idxs[cate].append(i)
            duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
            class_distribution = {
                k: len(v) / duplicated_samples
                for k, v in class_sample_idxs.items()
            }
            sample_indices = []
            frac = 1.0 / self.cls_num
            ratios = [frac / v for v in class_distribution.values()]
            for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
                sample_indices += np.random.choice(cls_inds, int(len(cls_inds) * ratio)).tolist()
            self.sample_indices = sample_indices
            print(self.category_list)
            print("class_distribution: ", class_distribution)
            print("repeat ratios: ", ratios)
            print("CBGS length: ", len(self.sample_indices))


    def __len__(self):
        return len(self.samples)

    def get_image_data(self, index, cam):
        sample = self.samples[index]
        samp = self.nusc.get('sample_data', sample['data'][cam])
        W = samp['width']
        H = samp['height']
        fimage = os.path.join(self.nusc.dataroot, samp['filename'])
        sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
        intrin = np.array(sens['camera_intrinsic'])
        extrin = transform_matrix(sens['translation'], Quaternion(sens['rotation']), inverse=False)
        rot_CV = Quaternion(sens['rotation'])
        trans_CV = np.array(sens['translation'])
        # world to vehicle transform
        egopose_cam = self.nusc.get('ego_pose', samp['ego_pose_token'])
        trans_WV = -np.array(egopose_cam['translation'])
        rot_WV = Quaternion(egopose_cam['rotation']).inverse
        box_list = []
        for tok in sample['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            if inst['num_lidar_pts'] + inst['num_radar_pts'] < self.min_num_lidar_points:
                continue
            category = category_to_detection_name(inst['category_name'])
            if category is None:
                continue
            cat_id = CATEGORY_IDS[category]
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']), label=cat_id)
            box.translate(trans_WV)
            box.rotate(rot_WV)
            box_list.append(box)
        return fimage, intrin, rot_CV, trans_CV, box_list

    def get_lidar_points(self, index, min_distance=1.0):
        ## return point cloud in 3xN array
        sample = self.samples[index]
        sample_data_token = sample['data']['LIDAR_TOP']
        current_sd_rec = self.nusc.get('sample_data', sample_data_token)
        current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
        # current_pc.remove_close(min_distance)
        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False)
        current_pc.transform(car_from_current)
        points = current_pc.points[:3, :].copy()
        return points

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

class BEVNuscenesLoader(NuscenesLoader):
    def __init__(self, root_dir, nusc, width, height, data_aug_conf, grid_conf, bev_aug, use_lidar=True, mode='train'):
        super(BEVNuscenesLoader, self).__init__(root_dir, nusc=nusc, datum_names=CAMERA_NAMES, mode=mode)
        self.cams = CAMERA_NAMES
        self.width = width
        self.height = height
        self.bev_aug = bev_aug
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.use_lidar = use_lidar
        if self.mode == 'train':
            self.transform = T.Compose([T.ColorJitter(0.3, 0.3, 0.3, 0.1),
                                        T.ToTensor()])
        else:
            self.transform = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in self.cams:
            fimage, intrin, rot_CV, trans_CV, box_list = self.get_image_data(index, cam)
            img = Image.open(fimage)
            raw_W, raw_H = img.size
            img = img.resize((self.width, self.height), resample=Image.NEAREST)
            intrin[0, :] *= ( self.width / raw_W)
            intrin[1, :] *= ( self.height / raw_H)
            imgs.append(self.transform(img))
            rots.append(torch.Tensor(rot_CV.rotation_matrix))
            trans.append(torch.Tensor(trans_CV))
            intrins.append(torch.Tensor(intrin))
        if (self.mode == 'train') and self.bev_aug:
            bev_rot, box_list = self.bev_augmentation(box_list)
        else:
            bev_rot = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])

        pts = torch.zeros((36000, 3))
        if self.mode == 'val' and self.use_lidar:
            lidar_pts = torch.from_numpy(self.get_lidar_points(index).T)
            N = min(lidar_pts.shape[0], 36000)
            pts[:N, :] =  lidar_pts[:N, :]
        box_list_out = []
        for box in box_list:
            mask = (box.center[0] > self.grid_conf['xbound'][0]) * (box.center[0] < self.grid_conf['xbound'][1]) * \
                    (box.center[1] > self.grid_conf['ybound'][0]) * (box.center[1] < self.grid_conf['ybound'][1]) * \
                    (box.center[2] > self.grid_conf['zbound'][0]) * (box.center[2] < self.grid_conf['zbound'][1]) 
            if mask:
                box_list_out.append(box)
        box3d_b = torch.zeros(128, 9)
        if len(box_list_out) > 0:
            box3d_list = [nusc_box3d_to_array(box) for box in box_list_out]
            box3d = torch.Tensor(np.stack(box3d_list, axis=0))
            N = min(128, box3d.size(0))
            box3d_b[:N, :] = box3d[:N, :]
        data_dict = {
            'image': torch.stack(imgs),
            'rots': torch.stack(rots), 
            'trans': torch.stack(trans),
            'intrins': torch.stack(intrins),
            'bev_rot': bev_rot.unsqueeze(0).repeat(len(imgs), 1, 1),
            'pointcloud': pts,
            'label': box3d_b
        }
        return data_dict

    def close_bev_aug(self):
        self.bev_aug = False

    def bev_augmentation(self, box_list):
        angle = np.random.uniform(*self.data_aug_conf['bev_rot'])
        rot = get_rot_matrix(angle)
        rot_mat = torch.eye(3)
        rot_mat[:2,:2] = rot
        for box_rot in box_list:
            xyz = torch.from_numpy(box_rot.center).float()
            xyz = torch.matmul(rot_mat, xyz.unsqueeze(1))
            rot_q = Quaternion._from_matrix(rot_mat.double().numpy())
            box_rot.orientation = box_rot.orientation * rot_q
            box_rot.center = xyz.squeeze(1).numpy()
        return rot_mat, box_list

    def image_augmentation(self, img):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        W, H = img.size
        fH, fW = self.height, self.width
        if self.mode == 'train':
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int(np.random.uniform(0, max(0, newH - fH)))
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = 0.48
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop = self.data_aug_conf['crop_val']
            flip = False
            rotate = 0
        
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = get_rot(rotate/180*np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        post_tran_out = torch.zeros(3)
        post_rot_out = torch.eye(3)
        post_tran_out[:2] = post_tran
        post_rot_out[:2, :2] = post_rot
        return img, post_rot_out, post_tran_out


class BEVNuscenesEvalLoader(NuscenesLoader):
    def __init__(self, root_dir, width, height, data_aug_conf, grid_conf):
        super(BEVNuscenesEvalLoader, self).__init__(root_dir, datum_names=CAMERA_NAMES, mode='val')
        self.cams = CAMERA_NAMES
        self.width = width
        self.height = height
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.bev_H = int((grid_conf['xbound'][1] - grid_conf['xbound'][0])/grid_conf['xbound'][2])
        self.bev_W = int((grid_conf['ybound'][1] - grid_conf['ybound'][0])/grid_conf['ybound'][2])
        if self.mode == 'train':
            self.transform = T.Compose([T.ColorJitter(0.5,0.5,0.5,0.2),T.ToTensor()])
        else:
            self.transform = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in self.cams:
            fimage, intrin, rot_CV, trans_CV, box_list = self.get_image_data(index, cam)
            img = Image.open(fimage)
            raw_W, raw_H = img.size
            img = img.resize((self.width, self.height), resample=Image.NEAREST)
            intrin[0, :] *= ( self.width / raw_W)
            intrin[1, :] *= ( self.height / raw_H)
            imgs.append(self.transform(img))
            rots.append(torch.Tensor(rot_CV.rotation_matrix))
            trans.append(torch.Tensor(trans_CV))
            intrins.append(torch.Tensor(intrin))

        bev_rot = torch.Tensor([[1,0,0], [0,1,0], [0, 0, 1]])
        sample = self.samples[index]
        samp = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        egopose_cam = self.nusc.get('ego_pose', samp['ego_pose_token'])
        trans_VW = np.array(egopose_cam['translation'])
        rot_VW = Quaternion(egopose_cam['rotation'])
        data_dict = {
            'image': torch.stack(imgs),
            'rots': torch.stack(rots), 
            'trans': torch.stack(trans),
            'intrins': torch.stack(intrins),
            'rot_pose': torch.Tensor(rot_VW.q),
            'trans_pose': torch.from_numpy(trans_VW),
            'sample_token': sample['token'],
            'bev_rot': bev_rot,
        }
        return data_dict
