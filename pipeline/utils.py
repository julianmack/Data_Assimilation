"""Helper functions"""
import torch
import numpy as np
import random
import vtktools
import pipeline.config

import vtk.util.numpy_support as nps

SETTINGS = pipeline.config.Config


def set_seeds(seed = SETTINGS.SEED):
    "Fix all seeds"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

class FluidityUtils():
    """Class to hold Fluidity helper functions.
    In theory this should be part of
    vtktools.py but I have kept it seperate to avoid confusion
    as to which is my work """

    def __init__(self):
        pass

    @staticmethod
    def get_3D_grid(fp, field_name, ret_torch=False):
        """Returns numpy array or torch tensor of the vtu file input"""
        ug = vtktools.vtu(fp)
        nx = 100 # =ny
        ny = nx
        nz = 50

        newshape = (nx, ny, nz)

        res = ug.StructuredPointProbe(nx, nx, nz)

        pointdata=res.GetPointData()

        vtkdata = pointdata.GetScalars(field_name)
        np_data = nps.vtk_to_numpy(vtkdata)

        #Fortran order reshape (i.e first index changes fastest):
        result = np.reshape(np_data, newshape, order='F')
        if ret_torch:
            result = torch.Tensor(result)

        return result

class ML_utils():
    """Class to hold ML helper functions"""

    def __init__(self):
        pass

    @staticmethod
    def training_loop_AE(model, optimizer, loss_fn, train_loader, test_loader,
            num_epoch, device=None, print_every=1, test_every=5):
        """Runs a torch AE model training loop.
        NOTE: Ensure that the loss_fn is in mode "sum"
        """
        set_seeds()
        train_losses = []
        test_losses = []
        if device == None:
            device = get_device()
        for epoch in range(num_epoch):
            train_loss = 0
            model.to(device)

            for batch_idx, data in enumerate(train_loader):
                model.train()
                x, = data
                x = x.to(device)
                optimizer.zero_grad()
                y = model(x)

                loss = loss_fn(y, x)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_losses.append((epoch, train_loss / len(train_loader.dataset)))
            if epoch % print_every == 0 or epoch in [0, num_epoch - 1]:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epoch, train_loss / len(train_loader.dataset)))
            if epoch % test_every == 0 or epoch == num_epoch - 1:
                model.eval()
                test_loss = 0
                for batch_idx, data in enumerate(test_loader):
                    x_test, = data
                    x_test = x_test.to(device)
                    y_test = model(x_test)
                    loss = loss_fn(y_test, x_test)
                    test_loss += loss.item()
                print('epoch [{}/{}], validation loss:{:.4f}'.format(epoch + 1, num_epoch, test_loss / len(test_loader.dataset)))
                test_losses.append((epoch, test_loss/len(test_loader.dataset)))
        return train_losses, test_losses

    @staticmethod
    def load_AE(ModelClass, path, **kwargs):
        """Loads an encoder and decoder"""
        model = ModelClass(**kwargs)
        model.load_state_dict(torch.load(path))
        encoder = model.encode
        decoder = model.decode

        return encoder, decoder

    @staticmethod
    def get_device(use_gpu=True, device_idx=0):
        """get torch device type"""
        if use_gpu:
            device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    @staticmethod
    def jacobian_slow_torch(inputs, outputs):
        """Computes a jacobian of two torch tensor.
        Uses a loop so linear complexity in dimension of output"""
        dims = len(inputs.shape)
        return torch.transpose(torch.stack([torch.autograd.grad([outputs[:, i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                            for i in range(outputs.size(1))], dim=-1), \
                            dims - 1, dims)
