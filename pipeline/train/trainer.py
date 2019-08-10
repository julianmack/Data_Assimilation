"""Run training for AE"""
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

import pickle

from pipeline import DAPipeline
from pipeline import ML_utils
from pipeline.AEs import Jacobian
from pipeline.VarDA import VDAInit
from pipeline import GetData
from pipeline.utils.expdir import init_expdir
from pipeline.VarDA.batch_DA import BatchDA

import time
import os

BATCH_MULT = 1
BATCH_UNIT = 16

BATCH = BATCH_MULT * BATCH_UNIT #64
LARGE = 1e30

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

        self.expdir = init_expdir(expdir)


        self.test_fp = self.expdir + "test.csv"
        self.train_fp = self.expdir + "train.csv"
        self.settings_fp = self.expdir + "settings.txt"
        self.calc_DA_MAE = calc_DA_MAE

        self.batch_sz = batch_sz
        self.settings.batch_sz =  batch_sz

        ML_utils.set_seeds() #set seeds before init model

        self.model =  ML_utils.load_model_from_settings(AE_settings)
        print("Number of parameters:", sum(p.numel() for p in self.model.parameters()))

        self.device = ML_utils.get_device()
        self.columns = ["epoch","reconstruction_err","DA_MAE", "DA_ratio_improve_MAE", "time_DA(s)", "time_epoch(s)"]

    def train(self, num_epoch = 100, learning_rate = 0.001, print_every=5,
            test_every=5, num_epochs_cv=8, num_workers=4, small_debug=False):

        self.learning_rate = learning_rate
        self.num_epochs = num_epoch
        self.print_every = print_every
        self.test_every = test_every
        self.num_epoch = num_epoch
        self.small_debug = small_debug

        settings = self.settings

        if settings.SAVE == True:
            self.model_dir = self.expdir
        else:
            self.model_dir = None
        self.loader = GetData()
        self.train_loader, self.test_loader = self.loader.get_train_test_loaders(settings,
                                                            self.batch_sz, num_workers=num_workers,
                                                            small_debug=small_debug)
        #self.loss_fn = torch.nn.L1Loss(reduction='sum')
        self.loss_fn = torch.nn.MSELoss(reduction="sum")


        lr_res = self.__maybe_cross_val_lr(test_every=test_every, num_epochs_cv=num_epochs_cv)


        if not isinstance(lr_res, float):
            #unpack results from __maybe_cross_val_lr()
            self.learning_rate, train_losses, test_losses = lr_res
        else:
            self.learning_rate = lr_res
            train_losses, test_losses = [], []
            #if only lr was returned, no model/optimizers were selected. Init:
            self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)



        settings.learning_rate = self.learning_rate #for logging

        train_losses_, test_losses_ = self.training_loop_AE(self.num_epochs_cv, self.num_epochs, self.device,
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
            self.epoch = epoch

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
        mean_diff = 0
        self.model.to(self.device)
        ###############
        L1 = torch.nn.L1Loss(reduction='sum')
        L1_loss = 0
        ###############

        t_start = time.time()

        for batch_idx, data in enumerate(self.train_loader):
            self.model.train()
            x, = data
            x = x.to(self.device)

            self.optimizer.zero_grad()
            y = self.model(x)

            #to prevent overflow
            # x_red = x / LARGE
            # y_red = y / LARGE


            #loss = self.loss_fn(y_red, x_red) * LARGE ** 2
            loss = self.loss_fn(y, x)
            loss.backward()

            train_loss += loss.item()
            mean_diff += torch.abs((x.mean() - y.mean())) * x.shape[0]
            self.optimizer.step()

            ##############
            self.model.eval()
            #loss1 = L1(y_red, x_red) * LARGE
            loss1 = L1(y, x)
            L1_loss += loss1.item()
            ##############

        self.model.eval()
        train_DA_MAE, train_DA_ratio, train_DA_time = self.maybe_eval_DA_MAE("train")

        t_end = time.time()

        train_loss_res = (epoch, train_loss / len(self.train_loader.dataset), train_DA_MAE, train_DA_ratio, train_DA_time, t_end - t_start)
        if epoch % print_every == 0 or epoch in [0, num_epoch - 1]:
            out_str = 'epoch [{}/{}], TRAIN: -loss:{:.2f}, av_diff: {:.2f}'
            out_str = out_str.format(epoch + 1, num_epoch,
                                train_loss / len(self.train_loader.dataset),
                                mean_diff / len(self.train_loader.dataset) )
            out_str += ", L1 loss:{:.2f}".format(L1_loss / len(self.train_loader.dataset))
            if self.calc_DA_MAE and (epoch % test_every == 0):
                out_str +=  ", DA_ratio:{:.4f}".format(train_DA_ratio)
            out_str += ", time taken (m): {:.2f}m".format( (t_end - t_start) / 60.)
            print(out_str)



        self.model.eval()
        if epoch % test_every == 0 or epoch == num_epoch - 1:
            t_start = time.time()
            test_loss = 0
            for batch_idx, data in enumerate(self.test_loader):
                x_test, = data
                x_test = x_test.to(self.device)
                y_test = self.model(x_test)
                loss = self.loss_fn(y_test, x_test)
                test_loss += loss.item()
            test_DA_MAE, test_DA_ratio, test_DA_time = self.maybe_eval_DA_MAE("test")
            t_end = time.time()
            if epoch % print_every == 0 or epoch == num_epoch - 1:
                out_str = "epoch [{}/{}], TEST: -loss:{:.4f}".format(epoch + 1, num_epoch, test_loss / len(self.test_loader.dataset))
                if self.calc_DA_MAE and (epoch % test_every == 0):
                    out_str +=  ", -DA_ratio:{:.4f}".format(train_DA_MAE)
                out_str += ", time taken(m): {:.2f}m".format( (t_end - t_start) / 60.)
                print(out_str)

            test_loss_res = (epoch, test_loss/len(self.test_loader.dataset), test_DA_MAE, test_DA_ratio, test_DA_time, t_end - t_start)

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

        mult = 1
        if self.settings.BATCH_NORM: #i.e. generally larger learning_rate with BN
            mult = 5

        mult *= BATCH_MULT #linear multiply by size of batch: https://arxiv.org/abs/1706.02677


        lrs_base = [0.0001, 0.0003, 0.001]
        lrs = [mult * x for x in lrs_base]

        res = []
        optimizers = []


        for idx, lr in enumerate(lrs):

            ML_utils.set_seeds() #set seeds before init model
            self.model =  ML_utils.load_model_from_settings(self.settings)
            self.optimizer = optim.Adam(self.model.parameters(), lr)
            test_losses = []
            train_losses = []
            print("learning rate:", lr)
            for epoch in range(self.num_epochs_cv):
                self.epoch = epoch
                train, test = self.train_one_epoch(epoch, self.print_every, test_every, self.num_epochs_cv)
                if test:
                    test_losses.append(test)
                train_losses.append(train)

            df = pd.DataFrame(train_losses, columns = self.columns)
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

        self.learning_rate = lrs[best_idx] * 0.8
        self.optimizer = optimizers[best_idx]
        self.model = best_model
        test_loss = best_test
        train_loss = best_train
        return self.learning_rate, train_loss, test_loss

    def maybe_eval_DA_MAE(self, test_valid):
        """As the DA procedure is so expensive, only eval on a single state.
        By default this is the final element of the test or train set"""
        if self.calc_DA_MAE and (self.epoch % self.test_every == 0 or self.epoch == self.num_epoch - 1):
            if test_valid == "train":
                u_c = self.loader.train_X.copy()
                np.random.shuffle(u_c) #random shuffle
                u_c = u_c[:64]
            elif test_valid == "test":
                u_c = self.loader.test_X
            else:
                raise ValueError("Can only evaluate DA_MAE on 'test' or 'train'")

            if self.settings.THREE_DIM:
                u_c = u_c.squeeze(1)

            if self.small_debug:
                u_c = u_c[:4]

            csv_fp = "{}{}_{}.csv".format(self.expdir, self.epoch, test_valid)
            batcher = BatchDA(self.settings, u_c, csv_fp=csv_fp, AEModel=self.model,
                        reconstruction=True)
            batch_res = batcher.run(self.print_every * 10, True)

            results = batcher.get_tots(batch_res)

            ref_mae = results["ref_MAE_mean"]
            da_mae =  results["da_MAE_mean"]
            ratio_improve_mae = (ref_mae - da_mae)/ref_mae
            time = results["time"]

            return da_mae, ratio_improve_mae, time
        else:
            return "NO_CALC", "NO_CALC", "NO_CALC"

    def slow_jac_wrapper(self, x):
        return Jacobian.accumulated_slow_model(x, self.model, self.DA_data.get("device"))



    def __da_data_wipe_some_values(self):
        #Now wipe some key attributes to prevent overlap between
        #successive calls to maybe_eval_DA_MAE()
        self.DA_data["u_c"] = None
        self.DA_data["w_0"] = None
        self.DA_data["d"] = None

    def to_csv(self, np_array, fp):
        df = pd.DataFrame(np_array, columns = self.columns)
        df.to_csv(fp)




if __name__ == "__main__":
    settings = settings.config.ToyAEConfig
    main(settings)
