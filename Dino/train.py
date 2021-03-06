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

## ---------------- LOSS function ---------------- ##
    dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number=2,
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_techer_temp_epochs,
            args.epoch,
            ).to(device)


## ---------------- preparing the optimizer ---------------- ##

    params_group = utils.get_params_groups(student)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()


## ---------------- learning rate schedulers ---------------- ##
    lr_schedule = utils.cosine_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs = args.warmup_epochs,
            )
    wd_scheduler = utils.consine_scheduler(
            args.weight_decay,
            args.weight_Decay_end,
            args.epochs, len(Data_loader),
            )

    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
            args.epochs, len(data_loader))
    print(f"=> LOSS, Optimizer and Scheduler ready")

## ---------------- restore prev training ---------------- ##

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]


## ---------------- TRAINING DINO ---------------- ##
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

## ---------------- training one epoch of DINO ---------------- ##

        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

## ---------------- writing logs ---------------- ##

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

## ---------------- Training 1 epoch ---------------- ##

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
        optimizer, lr_scheduler, wd_schedule, momentum_schedule, epoch,
        f16_scaler, args):
    metric_logger = utils.MatricLogger(delimeter=" ")
    header = f"Epoch: [{epoch}/{arg.epoch}]"

    for it, (image, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        it = len(data_loader) * epoch * it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_Schedule[it]
            if i == 0:
                param_group['weight_decay'] = wd_schedule[it]

        
        # move images to gpu


