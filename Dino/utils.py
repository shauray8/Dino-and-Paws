import os, sys, time, datetime
import math, random
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter, ImageOps


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

def fix_rando_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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












