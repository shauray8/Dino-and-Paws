import torch
import torch.nn as nn
import sys, time, math, datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from model import *
import model


def callable_stuff():
    args = sorted(name for name in torchvision_models.__dict__
            if name.islower() and not name.startswith("__")
                and callable(torchvision_models.__dict__[name]))

    kwargs = sorted(name for name in model.__dict__ 
              if name.islower and not name.startswith("  ")
                and callable(model.__dict__[name]))

    return args, kwargs


args, kwargs = callable_stuff()
print(args, "\n" , kwargs)
