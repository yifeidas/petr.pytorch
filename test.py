import torch
from torch.utils import data
import torch.nn.functional as F
from model.petr import PETR
from dataset.nuscenes import BEVNuscenesLoader, CATEGORY_IDS
from utils.vis import draw_voxel_heatmap, pointcloud_to_bevimg, draw_box_in_bev
from utils.tools import merge_six_image_tensors, sample_to_cuda, tensor_to_image, merge_six_images
from geometry.boxes3d import GenericBoxes3D, box3d_ang_to_box3d_quat
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import yaml
import cv2
import numpy as np
import argparse
from thop import profile
import time

parser = argparse.ArgumentParser(description='PyTorch bevdet Testing')
parser.add_argument('--pth', default='checkpoint/petr-nuscenes-backup.pth', type=str, help='model ckpt')
parser.add_argument('--hyp', default='config/hyp.petr.nusc.yaml', type=str, help='learning rate')
parser.add_argument('--batch', default=1, type=int, help='batch size')
parser.add_argument('--detdir', default='/data/04_dataset/nuScenes/nuscenes', type=str, help='detection dataset dir')
args = parser.parse_args()

color_list = [(0, 255, 255), (255, 255, 0), (0, 200, 255), (127, 0, 255), (0, 127, 255), \
    (127, 0, 127), (127, 100, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), ]

torch.set_grad_enabled(False)

with open(args.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    data_arg_conf = {
        'bev_rot': hyp['bev_rot'],
    }
    grid_conf = {
        'xbound': hyp['xbound'],
        'ybound': hyp['ybound'],
        'zbound': hyp['zbound']
    }
valset = BEVNuscenesLoader(args.detdir, None, hyp['width'], hyp['height'], data_arg_conf, grid_conf, False, use_lidar=True,  mode='val')
valloader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

net = PETR(grid_conf, 6, 512, 64, 10, backbone=hyp['backbone'], aux_loss=hyp['aux_loss']).cuda()
checkpoint = torch.load(args.pth, map_location=torch.device('cpu'))
print('epoch: ', checkpoint['epoch'])
net.load_state_dict(checkpoint['net'])
net = net.cuda().eval()
for i, data_dict in enumerate(valloader):
    print(i)
    data_dict = sample_to_cuda(data_dict)
    # if i==0:
    #     flops, params = profile(net, inputs=(data_dict, ))
    #     print("MACs: ", flops / 1e9, " G")
    #     print("params: ", params / 1e6, "M")
    t0 = time.time()
    with torch.cuda.amp.autocast(enabled=True):
        pred = net(data_dict)
    t1 = time.time()
    print("inference time: ", (t1 - t0))
    # objs_list = net.decode(pred, grid_conf, hyp['CANONICAL_BOX3D_SIZES'], thresh=0.35)
    objs_list = net.decode(pred, thresh=0.1)
    print(objs_list[0])
    box3d_preds = box3d_ang_to_box3d_quat(objs_list[0])
    print(box3d_preds.size())
    print(box3d_preds)
    # show in image view
    img_list = []
    for n in range(6):
        img_t = data_dict['image'][0, n]
        img = tensor_to_image(img_t)
        rot = data_dict['rots'][0, n].cpu().double().numpy()
        tran = data_dict['trans'][0, n].cpu().numpy()
        intrinsic = data_dict['intrins'][0, n].cpu().numpy()
        for j in range(box3d_preds.size(0)):
            box3d_pred = box3d_preds[j, :].clone()
            quar, xyz, wlh = box3d_pred[2:6], box3d_pred[6:9], box3d_pred[9:12]
            box3d = Box(xyz.tolist(), wlh.tolist(), Quaternion(quar.tolist()))
            box3d.translate(-tran)
            box3d.rotate(Quaternion._from_matrix(rot, rtol=1e-03, atol=1e-05).inverse)
            uvz = np.dot(intrinsic, box3d.center)
            uv = uvz[:2] / uvz[2]
            if (box3d.center[2] > 0) and (uv[0] >= -30) and (uv[0] < img.shape[1]+30) and (uv[1] >= 0) and (uv[1] < img.shape[0]):
                box3d.render_cv2(img, intrinsic, normalize=True)

        img_list.append(img)
    imgs = merge_six_images(img_list)
    # # show in BEV view
    bev_img = pointcloud_to_bevimg(data_dict['pointcloud'].squeeze(), [-51.2, 51.2, 0.2], [-51.2, 51.2, 0.2], [-10.0, 10.0, 0.2])
    for j in range(box3d_preds.size(0)):
        box3d_pred = box3d_preds[j]
        # print(box3d_pred)
        quar, xyz, wlh = box3d_pred[2:6], box3d_pred[6:9], box3d_pred[9:12]
        cls_ = int(box3d_pred[0].long().numpy())
        box3d = Box(xyz.tolist(), wlh.tolist(), Quaternion(quar.tolist()))
        corners = box3d.bottom_corners()
        bev_img = draw_box_in_bev(corners, bev_img, [-51.2, 51.2, 0.2], [-51.2, 51.2, 0.2], color=color_list[cls_])
    label = data_dict['label'].squeeze(0).cpu()
    mask = label[:, 4:6].sum(dim=1) > 0
    label = box3d_ang_to_box3d_quat(label[mask, :].clone())
    for j in range(label.size(0)):
        box3d_truth = label[j]
        # print(box3d_truth)
        if box3d_truth.sum() == 0:
            break
        quar, xyz, wlh = box3d_truth[1:5], box3d_truth[5:8], box3d_truth[8:11]
        box3d = Box(xyz.tolist(), wlh.tolist(), Quaternion(quar.tolist()))
        corners = box3d.bottom_corners()
        bev_img = draw_box_in_bev(corners, bev_img, [-51.2, 51.2, 0.2], [-51.2, 51.2, 0.2], color=(0,255,0))
    bev_img = bev_img[::-1, ::-1, :].copy()
    # cv2.imshow("img_bev", bev_img)
    # cv2.imshow("img", imgs)
    imgs = cv2.resize(imgs, (1280, 512))
    show_img = np.concatenate([imgs, bev_img], axis=1)
    cv2.imshow("show_img", show_img)
    # cv2.imwrite('show_results/result_%06d.jpg'%i, show_img)
    cv2.waitKey(0)
