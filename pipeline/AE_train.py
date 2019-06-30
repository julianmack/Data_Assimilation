"""Run training for AE"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

import pickle

import pipeline.config as config

from pipeline import utils, DAPipeline
import os

BATCH = 16

class TrainAE():
    def __init__(self, AE_settings, expdir, calc_DA_MAE=False, batch_sz=BATCH):
        """Initilaizes the AE training class.

        ::AE_settings - a config.Config class with the DA settings
        ::expdir - a directory of form `experiments/<possible_path>` to keep logs
        ::calc_DA_MAE - boolean. If True, training will evaluate DA Mean Absolute Error
            during the training cycle. Note: this is *MUCH* slower
        """

        self.settings = AE_settings

        err_msg = """AE_settings must be an AE configuration class"""
        assert self.settings.COMPRESSION_METHOD == "AE", err_msg

        self.expdir = self.__init_expdir(expdir)

        self.test_fp = self.expdir + "test.csv"
        self.train_fp = self.expdir + "train.csv"
        self.settings_fp = self.expdir + "settings.txt"
        self.calc_DA_MAE = calc_DA_MAE
        self.batch_sz = batch_sz


    def train(self, num_epoch = 100, learning_rate = 0.003):
        settings = self.settings
        #data
        loader = utils.DataLoader()
        X = loader.get_X(settings)

        self.train_X, self.test_X, DA_u_c, X_norm,  mean, std = loader.test_train_DA_split_maybe_normalize(X, settings)


        #Add Channel if we are in 3D case
        if settings.THREE_DIM:
            self.train_X = np.expand_dims(self.train_X, 1)
            self.test_X = np.expand_dims(self.test_X, 1)


        #Dataloaders
        train_dataset = TensorDataset(torch.Tensor(self.train_X))
        train_loader = DataLoader(train_dataset, self.batch_sz, shuffle=True, num_workers=6)
        test_dataset = TensorDataset(torch.Tensor(self.test_X))
        test_batch_sz = min(self.test_X.shape[0], self.batch_sz)
        test_loader = DataLoader(test_dataset, test_batch_sz)



        print("train_size = ", len(train_loader.dataset))
        print("test_size = ", len(test_loader.dataset))



        device = utils.ML_utils.get_device()


        loss_fn = torch.nn.L1Loss(reduction='sum')
        model = settings.AE_MODEL_TYPE(**settings.get_kwargs())

        self.model = model
        optimizer = optim.Adam(model.parameters(), learning_rate)

        print(model)

        print("Number of parameters:", sum(p.numel() for p in model.parameters()))

        if settings.SAVE == True:
            model_dir = self.expdir
        else:
            model_dir = None

        train_losses, test_losses = self.training_loop_AE(model, optimizer,
                                loss_fn, train_loader, test_loader,
                                num_epoch, device, print_every=1, test_every=5, model_dir = self.expdir)


        #Save results and settings file (so that it can be exactly reproduced)
        if settings.SAVE == True:
            self.to_csv(train_losses, self.train_fp)
            self.to_csv(test_losses, self.test_fp)
            with open(self.settings_fp, "wb") as f:
                pickle.dump(settings, f)


        return model


    def training_loop_AE(self, model, optimizer, loss_fn, train_loader, test_loader,
            num_epoch, device=None, print_every=1, test_every=5, save_every=5, model_dir=None):
        """Runs a torch AE model training loop.
        NOTE: Ensure that the loss_fn is in mode "sum"
        """
        utils.set_seeds()
        train_losses = []
        test_losses = []
        if device == None:
            device = utils.ML_utils.get_device()
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



            train_DA_MAE, train_DA_ratio = self.maybe_eval_DA_MAE("train")
            train_losses.append((epoch, train_loss / len(train_loader.dataset), train_DA_MAE, train_DA_ratio))
            if epoch % print_every == 0 or epoch in [0, num_epoch - 1]:
                out_str = 'epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epoch, train_loss / len(train_loader.dataset))
                if self.calc_DA_MAE:
                    out_str +  ", DA_MAE:{:.4f}".format(train_DA_MAE)
                print(out_str)
            if epoch % test_every == 0 or epoch == num_epoch - 1:
                model.eval()
                test_loss = 0
                for batch_idx, data in enumerate(test_loader):
                    x_test, = data
                    x_test = x_test.to(device)
                    y_test = model(x_test)
                    loss = loss_fn(y_test, x_test)
                    test_loss += loss.item()
                test_DA_MAE, test_DA_ratio = self.maybe_eval_DA_MAE("test")
                out_str = "epoch [{}/{}], valid: -loss:{:.4f}".format(epoch + 1, num_epoch, test_loss / len(test_loader.dataset))
                if self.calc_DA_MAE:
                    out_str +  ", -DA_MAE:{:.4f}".format(test_DA_MAE)
                print(out_str)
                test_losses.append((epoch, test_loss/len(test_loader.dataset), test_DA_MAE, test_DA_ratio))
            if epoch % save_every == 0 and model_dir != None:
                model_fp_new = "{}{}.pth".format(model_dir, epoch)
                torch.save(model.state_dict(), model_fp_new)
        if epoch % save_every != 0 and model_dir != None:
            #Save model (if new model hasn't just been saved)
            model_fp_new = "{}{}.pth".format(model_dir, epoch)
            torch.save(model.state_dict(), model_fp_new)
        return train_losses, test_losses

    def maybe_eval_DA_MAE(self, test_valid):
        """As the DA procedure is so expensive, only eval on a single state.
        By default this is the final element of the test or train set"""
        if self.calc_DA_MAE:
            if test_valid == "train":
                u_c = self.train_X[-1].flatten()
            elif test_valid == "test":
                u_c = self.test_X[-1].flatten()
            else:
                raise ValueError("Can only evaluate DA_MAE on 'test' or 'train'")

            if not hasattr(self, "DA_data"):
                DA = DAPipeline(self.settings)
                data, std, mean = DA.vda_setup(self.settings)
                self.DA_data = data
                self.__da_data_wipe_some_values()

            #update control state:
            self.DA_data["u_c"] = u_c
            self.DA_data["w_0"] = torch.zeros((self.settings.get_number_modes())).flatten()
            self.DA_data["V_trunc"] = self.model.decode
            if self.settings.JAC_NOT_IMPLEM:
                import warnings
                warnings.warn("Using **Very** slow method of calculating jacobian. Consider disabling DA", UserWarning)
                self.DA_data["V_grad"] = self.slow_jac_wrapper
            else:
                self.DA_data["V_grad"] = self.model.jac_explicit


            DA = DAPipeline(self.settings)

            DA_results = DA.perform_VarDA(self.DA_data, self.settings)
            ref_mae = DA_results["ref_MAE_mean"]
            mae = DA_results["da_MAE_mean"]

            ratio_improve_mae = (ref_mae - mae)/ref_mae
            self.__da_data_wipe_some_values()
            return mae, ratio_improve_mae
        else:
            return "NO_CALC", "NO_CALC"


    def slow_jac_wrapper(self, x):
        return utils.ML_utils.jac_explicit_slow_model(x, self.model)

    def __da_data_wipe_some_values(self):
        #Now wipe some key attributes to prevent overlap between
        #successive calls to maybe_eval_DA_MAE()
        self.DA_data["u_c"] = None
        self.DA_data["w_0"] = None
        self.DA_data["V_trunc"] = None

    def to_csv(self, np_array, fp):
        df = pd.DataFrame(np_array, columns = ["epoch","reconstruction_err","DA_MAE", "DA_ratio_improve_MAE"])
        df.to_csv(fp)


    def __init_expdir(self, expdir):
        expdir = utils.win_to_unix_fp(expdir)
        wd = utils.get_home_dir()
        try:
            dir_ls = expdir.split("/")
            assert "experiments" in dir_ls
        except (AssertionError, KeyError, AttributeError) as e:
            print("~~~~~~~~{}~~~~~~~~~".format(str(e)))
            raise ValueError("expdir must be in the experiments/ directory")

        if expdir[0] == "/":
            expdir = expdir[1:]
        if not expdir[-1] == "/":
            expdir += "/"

        expdir = wd + expdir

        if os.path.isdir(expdir):
            if len(os.listdir(expdir)) > 0:
                raise ValueError("Cannot overwrite files in expdir. Exit-ing.")
        else:
            os.makedirs(expdir)
        return expdir


if __name__ == "__main__":
    settings = config.ToyAEConfig
    main(settings)
