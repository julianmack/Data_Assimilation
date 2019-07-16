import torch
import numpy as np
import random

import os


def set_seeds(seed = None):
    "Fix all seeds"
    if seed == None:
        seed = os.environ.get("SEED")
        if seed == None:
            raise NameError("SEED environment variable not set. Do this manually or initialize a Config class")
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def load_AE(ModelClass, path, **kwargs):
    """Loads an encoder and decoder"""
    model = ModelClass(**kwargs)
    model.load_state_dict(torch.load(path))
    #model.eval()
    encoder = model.encode
    decoder = model.decode

    return encoder, decoder

def get_device(use_gpu=True, device_idx=0):
    """get torch device type"""
    if use_gpu:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device
