import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torch.multiprocessing as mp
from torch.utils.data import dataloader
from torch.utils import data
import time
import argparse
import numpy as np
import os
from PIL import Image
import sys
import yaml

from dataset.nuscenes import BEVNuscenesLoader, CATEGORY_IDS
from model.petr import PETR
from utils.trainer import Trainer

parser = argparse.ArgumentParser(description='PyTorch DD3D Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt', default='checkpoint/petr-backup.pth', type=str, help='resume checkpoint')
parser.add_argument('--ckpt_output', default='./checkpoint', type=str, help='checkpoint dir')
parser.add_argument('--logdir', default='./runs', type=str, help='resume checkpoint')
parser.add_argument('--hyp', default='config/hyp.petr.nusc.yaml', type=str, help='learning rate')
parser.add_argument('--half', default=0, type=int, help='using half precision training')
parser.add_argument('--adam', default=1, type=int, help='use adam optimizer')
parser.add_argument('--workers', default=4, type=int, help='image height')
parser.add_argument('--batch', default=8, type=int, help='batch size')
parser.add_argument('--detdir', default='/data/04_dataset/nuScenes/nuscenes', type=str, help='detection dataset dir')
parser.add_argument('--disturl', default='tcp://localhost:23456', type=str, help='torch distributed node0 IP and port')
parser.add_argument('--worldsize', default=1, type=int, help='the number of machines for distributed training')
parser.add_argument('--rank', default=0, type=int, help='the order of machines for distributed training')
args = parser.parse_args()

def setup(gpu_rank, world_size):
    ngpus_per_node = torch.cuda.device_count()
    args.worldsize = ngpus_per_node * args.worldsize
    args.rank = args.rank * ngpus_per_node + gpu_rank
    dist.init_process_group("nccl", init_method=args.disturl, rank=args.rank, world_size=args.worldsize)
    print('rank/total: %d/%d'%(args.rank, args.worldsize))

def cleanup():
    dist.destroy_process_group()

def dist_train(rank, world_size):
    setup(rank, world_size)

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

    trainset = BEVNuscenesLoader(args.detdir, None, hyp['width'], hyp['height'], data_arg_conf, grid_conf, hyp['bev_aug'])
    valset = BEVNuscenesLoader(args.detdir, trainset.nusc, hyp['width'], hyp['height'], data_arg_conf, grid_conf, hyp['bev_aug'], mode='val')
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    loader_train = data.DataLoader(trainset, batch_size=args.batch, shuffle=False, \
                    num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    loader_val = data.DataLoader(valset, batch_size=args.batch, shuffle=False, \
                    num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    num_cls = trainset.cls_num

    print('initializing network...')
    network = PETR(grid_conf, 6, 512, 64, num_cls, backbone=hyp['backbone'], bev_aug=hyp['bev_aug'], w_cls=hyp['w_cls'], w_box=hyp['w_box'], aux_loss=hyp['aux_loss'])
    # if hyp['pretrain']:
    #     network.load_backbone()
    torch.backends.cudnn.benchmark = True
    net = network.cuda(rank)
    # #criterion
    # criterion = SimOTALoss(10, grid_conf, w_obj=hyp['w_obj'], w_cls=hyp['w_cls'], w_box=hyp['w_box'], rank=rank).cuda(rank)
    dist.barrier() # sync point

    trainer = Trainer(net, loader_train, loader_val, args, hyp, rank, random_resize=hyp['random_resize'])
    trainer.run()

    cleanup()

if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    mp.spawn(dist_train, args=(ngpus,), nprocs=ngpus, join=True)