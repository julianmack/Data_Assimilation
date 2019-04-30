

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


class ML_utils():
    """Class to hold ML helper functions"""

    def __init__():
        pass

    @staticmethod
    def training_loop(model, optimizer, loss, train_loader, print_every, epochs, device):
        """Runs a torch model training loop"""
        model.train()
        for epoch in range(epoch):
            for batch_idx, data in enumerate(train_loader):
                pass
