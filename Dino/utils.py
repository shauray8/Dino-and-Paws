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
