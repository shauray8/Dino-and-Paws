import torch
import torch.nn as nn
import sys, time, math, datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import torch.backends.cudnn as cudnn

import model
from model import *
from utils import *


def callable_stuff():
    wargs = sorted(name for name in torchvision_models.__dict__
            if name.islower() and not name.startswith("__")
                and callable(torchvision_models.__dict__[name]))

    kwargs = sorted(name for name in model.__dict__ 
              if name.islower and not name.startswith("  ")
                and callable(model.__dict__[name]))

    return wargs, kwargs

wargs, kwargs = callable_stuff()

## ---------------- arg parse goes here ---------------- ##

## ---------------- ends here ---------------- ##

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vaars(Args)).items())))
    cudnn.benchmark = True

    ## ---------------- Augmenting and preparing the data ---------------- ##

    transform = DataAugmentation(
                args.global_crops_scale,
                args.local_crops_scale,
                args.local_crops_number,
            )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=args.batch_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
    
    print(f"Data loaded: there are {len(dataset)} images")


    ## ---------------- building student and teacher network ---------------- ##
    ## ---------------- for vision transformers ---------------- ##

    if args.arch in model.__dict__.keys():
        student = model.__dict__[args.arch](
                    patch_size = args.patch_size,
                    drop_path_rate=0.1,
                )
        teacher = model.__dict__[arg.arch](patch_size=args.patch_size)
                    
        student.head = DINOHead(
                    student.embed_dim,
                    args.out_dim,
                    use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer,
                )
        teacher.head = DINOHead(teacher.embed_dim, args.out_dim, args.use_bn_in_head)
        
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[arg_arch]()
        teacher = torchvision_models.__dict__[arg_arch]()

        embed_dim = student.fc.weight.shape[1]
        student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer = args_norm_last_layer,
            ))
        teacher = utils.MultiCropWrapper(
                teacher,
                DINOHead(embed_dim, args_out_dim, args.use_bn_in_head),
                )
    else:
        print(f"Unknown Architecture: {args.arch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, teacher = student.to(device), teacher.to(device)

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")


