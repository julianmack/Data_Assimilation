"""Run training for AE"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

import pickle

from pipeline import settings


from pipeline import DAPipeline
from pipeline import ML_utils
from pipeline.AEs import Jacobian
from pipeline import GetData, SplitData
import os

BATCH = 16

class TrainAE():
    def __init__(self, AE_settings, expdir, calc_DA_MAE=False, batch_sz=BATCH):
        """Initilaizes the AE training class.

        ::AE_settings - a settings.config.Config class with the DA settings
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
        self.settings.batch_sz =  batch_sz

        ML_utils.set_seeds() #set seeds before init model

        self.model =  AE_settings.AE_MODEL_TYPE(**AE_settings.get_kwargs())
        print("Number of parameters:", sum(p.numel() for p in self.model.parameters()))

        self.device = ML_utils.get_device()


    def train(self, num_epoch = 100, learning_rate = 0.0025, print_every=2,
            test_every=5):

        self.learning_rate = learning_rate
        self.num_epochs = num_epoch

        settings = self.settings

        if settings.SAVE == True:
            self.model_dir = self.expdir
        else:
            self.model_dir = None

        #data
        loader = GetData()
        splitter = SplitData()
        X = loader.get_X(settings)

        self.train_X, self.test_X, DA_u_c, X_norm,  mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)



        #Add Channel if we are in 3D case
        if settings.THREE_DIM:
            self.train_X = np.expand_dims(self.train_X, 1)
            self.test_X = np.expand_dims(self.test_X, 1)


        #Dataloaders
        train_dataset = TensorDataset(torch.Tensor(self.train_X))
        self.train_loader = DataLoader(train_dataset, self.batch_sz, shuffle=True, num_workers=6)
        test_dataset = TensorDataset(torch.Tensor(self.test_X))
        test_batch_sz = min(self.test_X.shape[0], self.batch_sz)
        self.test_loader = DataLoader(test_dataset, test_batch_sz)


        device = ML_utils.get_device()


        self.loss_fn = torch.nn.L1Loss(reduction='sum')


        self.learning_rate, train_losses, test_losses = self.__maybe_cross_val_lr(test_every=test_every)

        settings.learning_rate = self.learning_rate #for logging



        train_losses_, test_losses_ = self.training_loop_AE(self.num_epochs_cv, self.num_epochs, device,
                                        print_every=print_every, test_every=test_every,
                                        model_dir = self.model_dir)
        if train_losses_:
            train_losses.extend(train_losses_)
        if test_losses_:
            test_losses.extend(test_losses_)


        #Save results and settings file (so that it can be exactly reproduced)
        if settings.SAVE == True:
            self.to_csv(train_losses, self.train_fp)
            self.to_csv(test_losses, self.test_fp)
            with open(self.settings_fp, "wb") as f:
                pickle.dump(settings, f)


        return self.model


    def training_loop_AE(self, start_epoch, num_epoch, device=None, print_every=2,
                        test_every=5, save_every=5, model_dir=None):
        """Runs a torch AE model training loop.
        NOTE: Ensure that the loss_fn is in mode "sum"
        """
        model = self.model
        self.model_dir = model_dir

        if device == None:
            device = ML_utils.get_device()
        self.device = device

        ML_utils.set_seeds()
        train_losses = []
        test_losses = []
        epoch = num_epoch - 1 #for case where no training occurs


        for epoch in range(start_epoch, num_epoch):

            train_loss, test_loss = self.train_one_epoch(epoch, print_every, test_every, num_epoch)
            train_losses.append(train_loss)
            if test_loss:
                test_losses.append(test_loss)

        if epoch % save_every != 0 and self.model_dir != None:
            #Save model (if new model hasn't just been saved)
            model_fp_new = "{}{}.pth".format(self.model_dir, epoch)
            torch.save(model.state_dict(), model_fp_new)


        return train_losses, test_losses

    def train_one_epoch(self, epoch, print_every, test_every, num_epoch):
        train_loss_res, test_loss_res = None, None
        train_loss = 0

        self.model.to(self.device)

        for batch_idx, data in enumerate(self.train_loader):
            self.model.train()
            x, = data
            x = x.to(self.device)
            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.loss_fn(y, x)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()



        train_DA_MAE, train_DA_ratio = self.maybe_eval_DA_MAE("train")
        train_loss_res = (epoch, train_loss / len(self.train_loader.dataset), train_DA_MAE, train_DA_ratio)
        if epoch % print_every == 0 or epoch in [0, num_epoch - 1]:
            out_str = 'epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epoch, train_loss / len(self.train_loader.dataset))
            if self.calc_DA_MAE:
                out_str +  ", DA_MAE:{:.4f}".format(train_DA_MAE)
            print(out_str)

        if epoch % test_every == 0 or epoch == num_epoch - 1:
            self.model.eval()
            test_loss = 0
            for batch_idx, data in enumerate(self.test_loader):
                x_test, = data
                x_test = x_test.to(self.device)
                y_test = self.model(x_test)
                loss = self.loss_fn(y_test, x_test)
                test_loss += loss.item()
            test_DA_MAE, test_DA_ratio = self.maybe_eval_DA_MAE("test")
            out_str = "epoch [{}/{}], valid: -loss:{:.4f}".format(epoch + 1, num_epoch, test_loss / len(self.test_loader.dataset))
            if self.calc_DA_MAE:
                out_str +  ", -DA_MAE:{:.4f}".format(test_DA_MAE)
            print(out_str)
            test_loss_res = (epoch, test_loss/len(self.test_loader.dataset), test_DA_MAE, test_DA_ratio)

        if epoch % test_every == 0 and self.model_dir != None:
            model_fp_new = "{}{}.pth".format(self.model_dir, epoch)
            torch.save(self.model.state_dict(), model_fp_new)
        return train_loss_res, test_loss_res

    def __maybe_cross_val_lr(self, test_every, num_epochs_cv = 8):
        if not num_epochs_cv:
            self.num_epochs_cv = 0
            return self.learning_rate
        elif self.num_epochs < num_epochs_cv:
            self.num_epochs_cv = self.num_epochs
        else:
            self.num_epochs_cv = num_epochs_cv

        if self.settings.BATCH_NORM: #i.e. generally larger learning_rate with BN
            lrs = [0.006, 0.003, 0.015, 0.0015]
        else:
            lrs = [0.003, 0.001, 0.006, 0.0006]

        res = []
        optimizers = []


        for idx, lr in enumerate(lrs):
            print("lr:", lr)

            ML_utils.set_seeds() #set seeds before init model
            self.model =  self.settings.AE_MODEL_TYPE(**self.settings.get_kwargs())
            self.optimizer = optim.Adam(self.model.parameters(), lr)
            test_losses = []
            train_losses = []

            for epoch in range(self.num_epochs_cv):
                train, test = self.train_one_epoch(epoch, 1, test_every, self.num_epochs_cv)
                if test:
                    test_losses.append(test)
                train_losses.append(train)

            df = pd.DataFrame(train_losses, columns = ["epoch","reconstruction_err","DA_MAE", "DA_ratio_improve_MAE"])
            train_final = df.tail(1).reconstruction_err

            res.append(train_final.values[0])
            optimizers.append(self.optimizer)

            #save model if best so far

            if res[-1] == min(res):
                best_test = test_losses
                best_train = train_losses
                best_idx = idx
                model_fp_new = "{}{}-{}.pth".format(self.model_dir, epoch, lr)
                torch.save(self.model.state_dict(), model_fp_new)
                best_model = self.model

        self.learning_rate = lrs[best_idx]
        self.optimizer = optimizers[best_idx]
        self.model = best_model
        test_loss = best_test
        train_loss = best_train
        return self.learning_rate, train_loss, test_loss

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
            self.DA_data["model"] = self.model
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
        return Jacobian.accumulated_slow_model(x, self.model, self.DA_data.get("device"))



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
        expdir = settings.helpers.win_to_unix_fp(expdir)
        wd = settings.helpers.get_home_dir()
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
    settings = settings.config.ToyAEConfig
    main(settings)
