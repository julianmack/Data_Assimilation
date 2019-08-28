import torch
import numpy as np
import random

import os
import pickle

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


def load_AE(ModelClass, path, device = None, **kwargs):
    """Loads an encoder and decoder"""
    if device == None:
        device = get_device()


    model = ModelClass(**kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    encoder = model.encode
    decoder = model.decode

    return encoder, decoder

def load_model_from_settings(settings, device=None, device_idx=None):
    """Loads model from settings - if settings.AE_MODEL_FP is set,
    this loads saved weights, otherwise it will init a new model """
    if not device:
        device = get_device(device_idx=device_idx)

    set_seeds()

    model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
    if hasattr(settings, "AE_MODEL_FP"):

        weights = torch.load(settings.AE_MODEL_FP, map_location=device)
        model.load_state_dict(weights)

    model.to(device)
    model.eval()
    return model

def load_model_and_settings_from_dir(dir, device_idx=None, choose_epoch=None):
    if dir[-1] != "/":
        dir += "/"

    #get files
    model, settings = None, None

    max_epoch = 0
    best_fp = None
    for path, subdirs, files in os.walk(dir):
        for file in files:
            if file == "settings.txt":
                with open(os.path.join(path, file), "rb") as f:
                    settings = pickle.load(f)
            if file[-4:] == ".pth":
                if "-" in file:
                    continue
                epoch = int(file.replace(".pth", ""))
                if choose_epoch is not None:
                    if choose_epoch == epoch:
                        best_fp = os.path.join(path, file)
                else:
                    if epoch >= max_epoch:
                        max_epoch = epoch
                        best_fp = os.path.join(path, file)

    if not settings:
        raise ValueError("No settings.txt file in dir")
    if choose_epoch and not best_fp:
        raise ValueError("No file named {}.pth in dir".format(choose_epoch))
    settings.export_env_vars()
    settings.AE_MODEL_FP = best_fp
    model = load_model_from_settings(settings, device_idx=device_idx)

    return model, settings


def get_device(use_gpu=True, device_idx=None):
    """get torch device type"""
    if device_idx is None:
        device_idx = os.environ.get("GPU_DEVICE")
        if device_idx == None:
            raise NameError("GPU_DEVICE environment variable has not been initialized. Do this manually or initialize a Config class")
    if use_gpu:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device
