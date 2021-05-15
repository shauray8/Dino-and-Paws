import os, sys, time, datetime
import math, random
import subprocess
import torch.distributed as dist
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter, ImageOps


## ------------ Useless (atleast for me) multi GPU setup ------------ ##
def init_distributed_mode(args):
    if torch.cuda.is_available():
        print("will run the code on one GPU")
        args.rank, args.gpu, args.worls_size= 0,0,1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    else:
        print("Does not support training on CPU")
        sys.exit(1)

    dist.init_process_group(
                backend='nccl',
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
    torch.cuda.set_device(args.gpu)
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


## ------------ Fixing a random seed (I dont know why will this help) ------------ ##
def fix_rando_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


## ------------ Data Augmentation ------------ ##
class DataAugmantation(object):
    def __init__(self, global_crop_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
                transforms.RandomHosrizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.COlorJitter(brightness=.4, contrast=.4, saturation=.2, hue=.1)],
                    p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ])
            
        normalize = transforms.Compose([
            transforms=ToTensor(),
            transformers.Normalize((.485, .456, .406),(.229, .224, .225)),
            ])

        # First global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
            ])

        # Second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flib_and_Color_jitter,
            GaussianBlur(.1),
            Solarization(.2),
            normalizw,
            ])

        # Transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale-local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=.5),
            normalize,
            ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_tranfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))

        return crops


## ------------ Gausian Blur ------------ ##
class GaussianBlur(object):
    def __init__(self, p=.5, radius_min=.1, radius_max=2.):
        self.prob=p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
                ImageFilter.GaussianBlur(
                    radius = random.unifor(self.radius_min, self.radius_max)
                    )
                )

## ------------ Solarization ------------ ##
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


## ------------ LOSS function ------------ ##
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
            warmup_teacher_temp_epochs, nepochs, student_temp=.1,
            center_momentum=.9):
        super(DINOLoss, self).__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatinate((
            np.linspace(warmup_teacher_temp, teacher_temp,
                warmup_teacher_temp_epochs),
            np.ones(nepocs - warmup_teacher_temp_epochs) * teacher_temp
            ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((techer_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



def get_params_group(model):
    regularized = []
    not_regualrized = []
    for name, params in model.named_parameters():
        if not params.require_grad:
            continue

        if name.endswith('.bias') or len(param.shape) == 1:
            not_regualarized.append(param)
        else:
            regualrized.append(param)
    return [{"params": regularized}, {"params": not_regualrized, "weight_decay": 0.}]

## ------------ Learning Rate Scheduler ------------ ##

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
        start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epocs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arrange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + .5 * (base_value - final_value) *  \
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
