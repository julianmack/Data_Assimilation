

import torch
import numpy as np
import random

SEED = 42

def set_seeds(seed = SEED):
    "Fix all seeds"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
