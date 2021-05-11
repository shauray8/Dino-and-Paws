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
    args = sorted(name for name in torchvision_models.__dict__
            if name.islower() and not name.startswith("__")
                and callable(torchvision_models.__dict__[name]))

    kwargs = sorted(name for name in model.__dict__ 
              if name.islower and not name.startswith("  ")
                and callable(model.__dict__[name]))

    return args, kwargs

args, kwargs = callable_stuff()

## ---------------- arg parse goes here ---------------- ##

## ---------------- ends here ---------------- ##

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print(f"git:\n {utils.get_sha()}\n")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vaars(Args)).items())))
    cudnn.benchmark = True

    ## ---------------- Augmenting and preparing the data ---------------- ##



