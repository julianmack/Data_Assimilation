#### ML helper functions
import torch
from pipeline.utils import ML_utils
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

class Jacobian():

    @staticmethod
    def jac_explicit_slow_model(inputs, model, device=None):
        inputs.requires_grad = True
        if device == None:
            device = Jacobian.get_device()
        model.to(device)
        output = model.decode(inputs).flatten()

        print("inputs.shape", inputs.shape)
        print("output.shape", output.shape)

        return Jacobian.jacobian_slow_torch(inputs, output)


    @staticmethod
    def jacobian_slow_torch( inputs, outputs):
        """Computes a jacobian of two torch tensor.
        Uses a loop so linear time-complexity in dimension of output.

        This (slow) function is used to test the much faster .jac_explicit()
        functions in AutoEncoders.py"""
        dims = len(inputs.shape)

        if dims > 1:
            return Jacobian.__batched_jacobian_slow(inputs, outputs)
        else:
            return Jacobian.__no_batch_jacobian_slow(inputs, outputs)
    @staticmethod
    def __batched_jacobian_slow(inputs, outputs):
        dims = len(inputs.shape)
        return torch.transpose(torch.stack([torch.autograd.grad([outputs[:, i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                            for i in range(outputs.size(1))], dim=-1), \
                            dims - 1, dims)
    @staticmethod
    def __no_batch_jacobian_slow(inputs, outputs):
        X = [torch.autograd.grad([outputs[i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                            for i in range(outputs.size(0))]
        X = torch.stack(X, dim=-1)
        return X.t()
