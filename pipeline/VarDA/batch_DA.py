import torch

from pipeline import ML_utils, DAPipeline
from pipeline.VarDA import SVD, VDAInit
import pandas as pd

class BatchDA():
    def __init__(self, settings, control_states, csv_fp=None, AEModel=None, reconstruction=True, plot=True):

        self.settings = settings
        self.control_states = control_states
        self.reconstruction = reconstruction
        self.plot = plot
        self.model = AEModel
        self.csv_fp = csv_fp

    def run(self, print_every=10):


        if settings.COMPRESSION_METHOD == "SVD":
            if settings.REDUCED_SPACE:
                raise NotImplementedError("Cannot have reduced space SVD")

            fp_base = settings.get_X_fp().split("/")[-1][1:]

            U = np.load(self.settings.INTERMEDIATE_FP  + "U" + fp_base)
            s = np.load(self.settings.INTERMEDIATE_FP  + "s" + fp_base)
            W = np.load(self.settings.INTERMEDIATE_FP  + "W" + fp_base)

            V_trunc = SVD.SVD_V_trunc(U, s, W, modes=mode)
            V_trunc_plus = SVD.SVD_V_trunc_plus(U, s, W, modes=mode)

            self.DA_pipeline = DAPipeline(self.settings)
            DA_data = DA_pipeline.data
            DA_data["V_trunc"] = V_trunc
            DA_data["V"] = None
            DA_data["w_0"] = V_trunc_plus @ u_0.flatten()
            DA_data["V_grad"] = None

        elif settings.COMPRESSION_METHOD == "AE":
            if self.model is None:
                raise ValueError("Must provide an AE torch.nn model if settings.COMPRESSION_METHOD == 'AE'")

            self.DA_pipeline = DAPipeline(self.settings, self.model)
            DA_data = DA_pipeline.data

            if self.reconstruction:
                encoder = DA_data.get("encoder")
                decoder = DA_data.get("decoder")

        else:
            raise ValueError("settings.COMPRESSION_METHOD must be in ['AE', 'SVD']")

        if self.reconstruction:
            L1 = torch.nn.L1Loss(reduction='sum')
            L2 = torch.nn.MSELoss(reduction="sum")

        totals = {"percent_improvement": 0,
                "ref_MAE_mean": 0,
                "da_MAE_mean": 0,
                "counts": 0,
                "l1": 0,
                "l2": 0}

        results = []

        for u_c in INSERT_W:

            # if self.settings.THREE_DIM: ???
            #     u_c = u_c.squeeze(0)

            # for idx in range(num_states):
            #     if num_states == 1:
            #         DA_data["u_c"] = data
            #     else:
            #         DA_data["u_c"] = data[idx]

            if self.settings.REDUCED_SPACE:
                self.DA_pipeline.data = VDAInit.provide_u_c_update_data_reduced_AE(self.DA_data,
                                                                                    self.settings, u_c)
            else:
                self.DA_pipeline.data = VDAInit.provide_u_c_update_data_full_space(self.DA_data,
                                                                                        self.settings, u_c)

            if self.settings.COMPRESSION_METHOD == "AE":
                DA_results = self.DA_pipeline.DA_AE()
            elif self.settings.COMPRESSION_METHOD == "SVD":
                DA_results = self.DA_pipeline.DA_SVD()

            if self.reconstruction:

                if self.settings.COMPRESSION_METHOD == "AE":
                    device = ML_utils.get_device()
                    data_tensor = torch.Tensor(u_c)
                    data_tensor = data_tensor.to(device)

                    data_hat = decoder(encoder(data))
                    data_hat = torch.Tensor(data_hat)
                    data_hat = data_hat.to(device)

                elif self.settings.COMPRESSION_METHOD == "SVD":
                    num_modes = self.settings.get_number_modes()
                    data_hat = SVD.SVD_reconstruction_trunc(data, U, s, W, num_modes)

                    data_hat = torch.Tensor(data_hat)

                l1 = L1(data_hat, data_tensor)
                l2 = L2(data_hat, data_tensor)
            else:
                l1, l2 = None, None


            result = {}
            result["percent_improvement"] = DA_results["percent_improvement"]
            result["ref_MAE_mean"] =  DA_results["ref_MAE_mean"]
            result["da_MAE_mean"] = DA_results["da_MAE_mean"]
            result["counts"] = DA_results["counts"]
            result["l1_loss"] = l1
            result["l2_loss"] = l2

            #add to results list (that will become a .csv)
            results.append(result)

            #add to aggregated dict results
            totals = self.__add_result_to_totals(result, totals)

            if idx % print_every == 0:
                print("idx:", idx)
                self.__print_totals(totals, idx + 1)

        print("------------")
        self.__print_totals(totals, num_states)
        print("------------")


        results_df = pd.DataFrame(results)
        #save to csv
        if self.csv_fp:
            pd.to_csv(results_df)

        if self.plot:
            raise NotImplementedError("plotting functionality not implemented yet")
        return results_df

    @staticmethod
    def __add_result_to_totals(result, totals):
        for k, v in result.items():
            totals[k] += v

    @staticmethod
    def __print_totals(totals, num_states):
        for k, v in totals.items():
            print(k, "{:.2f}".format(v / num_states))
        print()


