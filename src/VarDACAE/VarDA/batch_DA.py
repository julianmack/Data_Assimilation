import torch
import time

from VarDACAE import ML_utils, GetData, SplitData
from VarDACAE.VarDA import DAPipeline
from VarDACAE.VarDA import SVD, VDAInit
from VarDACAE.utils.expdir import init_expdir

import pandas as pd
import numpy as np

class BatchDA():
    def __init__(self, settings, control_states=None, csv_fp=None, AEModel=None,
                reconstruction=True, plot=False):

        self.settings = settings
        self.control_states = control_states
        self.reconstruction = reconstruction
        self.plot = plot
        self.model = AEModel
        self.csv_fp = csv_fp

        if self.csv_fp:
            fps = self.csv_fp.split("/")
            dir = fps[:-1]
            dir = "/".join(fps[:-1])
            self.expdir = init_expdir(dir, True)
            self.file_name = fps[-1]

        if self.control_states is None:
            loader, splitter = GetData(), SplitData()
            X = loader.get_X(settings)

            train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)
            self.control_states = test_X



    def run(self, print_every=10, print_small=True):

        shuffle = self.settings.SHUFFLE_DATA #save value
        self.settings.SHUFFLE_DATA = False

        if self.settings.COMPRESSION_METHOD == "SVD":
            if self.settings.REDUCED_SPACE:
                raise NotImplementedError("Cannot have reduced space SVD")

            fp_base = self.settings.get_X_fp().split("/")[-1][1:]

            U = np.load(self.settings.INTERMEDIATE_FP  + "U" + fp_base)
            s = np.load(self.settings.INTERMEDIATE_FP  + "s" + fp_base)
            W = np.load(self.settings.INTERMEDIATE_FP  + "W" + fp_base)

            num_modes = self.settings.get_number_modes()

            V_trunc = SVD.SVD_V_trunc(U, s, W, modes=num_modes)
            V_trunc_plus = SVD.SVD_V_trunc_plus(U, s, W, modes=num_modes)

            self.DA_pipeline = DAPipeline(self.settings)
            DA_data = self.DA_pipeline.data
            DA_data["V_trunc"] = V_trunc
            DA_data["V"] = None
            DA_data["w_0"] = V_trunc_plus @ DA_data.get("u_0").flatten()
            DA_data["V_grad"] = None

        elif self.settings.COMPRESSION_METHOD == "AE":
            if self.model is None:
                raise ValueError("Must provide an AE torch.nn model if settings.COMPRESSION_METHOD == 'AE'")


            self.DA_pipeline = DAPipeline(self.settings, self.model)
            DA_data = self.DA_pipeline.data

            if self.reconstruction:
                encoder = DA_data.get("encoder")
                decoder = DA_data.get("decoder")

        else:
            raise ValueError("settings.COMPRESSION_METHOD must be in ['AE', 'SVD']")

        self.settings.SHUFFLE_DATA = shuffle

        if self.reconstruction:
            L1 = torch.nn.L1Loss(reduction='sum')
            L2 = torch.nn.MSELoss(reduction="sum")

        totals = {"percent_improvement": 0,
                "ref_MAE_mean": 0,
                "da_MAE_mean": 0,
                "mse_DA": 0,
                "mse_ref": 0,
                "counts": 0,
                "l1_loss": 0,
                "l2_loss": 0,
                "time": 0}

        results = []

        if len(self.control_states.shape) in [1, 3]:
            raise ValueError("This is not batched control_state input")
        else:
            num_states = self.control_states.shape[0]

        for idx in range(num_states):
            u_c = self.control_states[idx]
            if self.settings.REDUCED_SPACE:
                self.DA_pipeline.data = VDAInit.provide_u_c_update_data_reduced_AE(DA_data,
                                                                                    self.settings, u_c)
            else:
                self.DA_pipeline.data = VDAInit.provide_u_c_update_data_full_space(DA_data,
                                                                                        self.settings, u_c)
            t1 = time.time()
            if self.settings.COMPRESSION_METHOD == "AE":
                DA_results = self.DA_pipeline.DA_AE()
            elif self.settings.COMPRESSION_METHOD == "SVD":
                DA_results = self.DA_pipeline.DA_SVD()
            t2 = time.time()
            t_tot = t2 - t1

            if self.reconstruction:
                data_tensor = torch.Tensor(u_c)
                if self.settings.COMPRESSION_METHOD == "AE":
                    device = ML_utils.get_device()
                    #device = ML_utils.get_device(True, 1)

                    data_tensor = data_tensor.to(device)

                    data_hat = decoder(encoder(u_c))
                    data_hat = torch.Tensor(data_hat)
                    data_hat = data_hat.to(device)

                elif self.settings.COMPRESSION_METHOD == "SVD":

                    data_hat = SVD.SVD_reconstruction_trunc(u_c, U, s, W, num_modes)

                    data_hat = torch.Tensor(data_hat)
                with torch.no_grad():
                    l1 = L1(data_hat, data_tensor)
                    l2 = L2(data_hat, data_tensor)
            else:
                l1, l2 = None, None


            result = {}
            result["percent_improvement"] = DA_results["percent_improvement"]
            result["ref_MAE_mean"] =  DA_results["ref_MAE_mean"]
            result["da_MAE_mean"] = DA_results["da_MAE_mean"]
            result["counts"] = DA_results["counts"]
            result["mse_ref"] = DA_results["mse_ref"]
            result["mse_DA"] = DA_results["mse_DA"]
            if self.reconstruction:
                result["l1_loss"] = l1.detach().cpu().numpy()
                result["l2_loss"] = l2.detach().cpu().numpy()
            result["time"] = t2 - t1
            #add to results list (that will become a .csv)
            results.append(result)

            #add to aggregated dict results
            totals = self.__add_result_to_totals(result, totals)

            if idx % print_every == 0 and idx > 0:
                if not print_small:
                    print("idx:", idx)
                self.__print_totals(totals, idx + 1, print_small)
        if not print_small:
            print("------------")
        self.__print_totals(totals, num_states, print_small)
        if not print_small:
            print("------------")


        results_df = pd.DataFrame(results)
        #save to csv
        if self.csv_fp:
            results_df.to_csv(self.expdir + self.file_name)

        if self.plot:
            raise NotImplementedError("plotting functionality not implemented yet")
        return results_df
    @staticmethod
    def get_tots(results_df):
        data = {}
        data["mse_ref"] = results_df["mse_ref"].mean()
        data["mse_DA"] = results_df["mse_DA"].mean()
        data["ref_MAE_mean"] = results_df["ref_MAE_mean"].mean()
        data["da_MAE_mean"] = results_df["da_MAE_mean"].mean()
        data["percent_improvement"] = results_df["percent_improvement"].mean()
        time = results_df["time"]
        time = time[1:] #ignore the first one as this can occud offline

        data["time"] = time.mean()
        return data

    @staticmethod
    def __add_result_to_totals(result, totals):
        for k, v in result.items():
            totals[k] += v
        return totals

    @staticmethod
    def __print_totals(totals, num_states, print_small):
        if not print_small:
            for k, v in totals.items():
                print(k, "{:.2f}".format(v / num_states))
            print()
        else:
            out_str = "DA - - L2: {:.2f}, L1: {:.2f}, % Improve: {:.2f}%, DA_MAE: {:.2f}, mse_ref: {:.2f}, mse_DA: {:.3f}, time(s):  {:.4f}s".format(
                                                    totals["l2_loss"]/ num_states,
                                                    totals["l1_loss"]/ num_states,
                                                    totals["percent_improvement"]/ num_states,
                                                    totals["da_MAE_mean"]/ num_states,
                                                    totals["mse_ref"] / num_states,
                                                    totals["mse_DA"] / num_states,
                                                    totals["time"]/ num_states)
            print(out_str)




