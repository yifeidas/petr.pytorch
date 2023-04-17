from cgitb import enable
import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import json
from tqdm import tqdm
from pyquaternion import Quaternion

from model.petr import PETR
from dataset.nuscenes import BEVNuscenesEvalLoader, CATEGORY_IDS
from utils.tools import sample_to_cuda
from geometry.boxes3d import GenericBoxes3D, box3d_ang_to_box3d_quat
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.utils import category_to_detection_name, detection_name_to_rel_attributes
from nuscenes.utils.splits import create_splits_scenes

parser = argparse.ArgumentParser(description='PyTorch petr Testing')
parser.add_argument('--pth', default='checkpoint/petr-nuscenes-backup.pth', type=str, help='model ckpt')
parser.add_argument('--hyp', default='config/hyp.petr.nusc.yaml', type=str, help='learning rate')
parser.add_argument('--batch', default=1, type=int, help='batch size')
parser.add_argument('--detdir', default='/data/04_dataset/nuScenes/nuscenes', type=str, help='detection dataset dir')
args = parser.parse_args()

torch.set_grad_enabled(False)

def random_attr(name: str) -> str:
    """
    This is the most straight-forward way to generate a random attribute.
    Not currently used b/c we want the test fixture to be back-wards compatible.
    """
    # Get relevant attributes.
    rel_attributes = detection_name_to_rel_attributes(name)
    if len(rel_attributes) == 0:
        # Empty string for classes without attributes.
        return ''
    else:
        # Pick a random attribute otherwise.
        return rel_attributes[np.random.randint(0, len(rel_attributes))]

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
valset = BEVNuscenesEvalLoader(args.detdir, hyp['width'], hyp['height'], data_arg_conf, grid_conf)
valloader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

net = PETR(grid_conf, 6, 512, 64, 10, aux_loss=hyp['aux_loss']).cuda()
checkpoint = torch.load(args.pth, map_location=torch.device('cpu'))
print('epoch: ', checkpoint['epoch'])
net.load_state_dict(checkpoint['net'])
net = net.cuda().eval()

results = {}
meta = {
            'use_camera': True,
            'use_lidar': False,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }
        
pbar = tqdm(enumerate(valloader), total=len(valloader))
for i, data_dict in pbar:
    data_dict = sample_to_cuda(data_dict)
    with torch.cuda.amp.autocast(enabled=True):
        pred = net(data_dict)
    # objs_list = net.decode(pred.float(), grid_conf, hyp['CANONICAL_BOX3D_SIZES'], thresh=0.2)
    objs_list = net.decode(pred, thresh=0.2)
    box3d_preds = box3d_ang_to_box3d_quat(objs_list[0])
    sample_token = data_dict['sample_token'][0]
    ego_rot = data_dict['rot_pose'][0].cpu().numpy()
    ego_trans = data_dict['trans_pose'][0].cpu().numpy()
    eval_results = []
    for box3d_pred in box3d_preds:
        quar, xyz, wlh = box3d_pred[2:6], box3d_pred[6:9], box3d_pred[9:12]
        if wlh[0] * wlh[1] * wlh[2] <= 0:
            continue
        detection_name = list( CATEGORY_IDS.keys())[box3d_pred[0].long().item()]
        prob = box3d_pred[1].item()
        box3d = Box(xyz.tolist(), wlh.tolist(), Quaternion(quar.tolist()))
        box3d.rotate(Quaternion(ego_rot))
        box3d.translate(ego_trans)
        eval_results.append(
                    {
                        'sample_token': sample_token,
                        'translation': list(box3d.center),
                        'size': list(box3d.wlh),
                        'rotation': list(box3d.orientation.q),
                        'velocity': [0.0, 0.0],
                        'detection_name': detection_name,
                        'detection_score': prob,
                        'attribute_name': random_attr(detection_name)
                    })
    results[sample_token] = eval_results

cfg = config_factory('detection_cvpr_2019')

submission = {
    'meta': meta,
    'results': results
}

with open('nuscenes_detection_eval.json', 'w') as f:
    json.dump(submission, f)

nusc_eval = DetectionEval(valset.nusc, cfg, 'nuscenes_detection_eval.json', eval_set='val', output_dir='eval_results/', verbose=True)
metrics = nusc_eval.main()