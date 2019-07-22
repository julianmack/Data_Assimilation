"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import os
import random
import torch
from scipy.optimize import minimize


from pipeline import ML_utils
from pipeline.AEs import Jacobian
from pipeline.settings import config
from pipeline.fluidity import VtkSave
from pipeline import GetData, SplitData
from pipeline.VarDA import VDAInit
from pipeline.VarDA import SVD
from pipeline.VarDA.cost_fn import cost_fn_J, grad_J

class DAPipeline():
    """Class to hold pipeline functions for Variational DA
    """

    def __init__(self, settings, AEmodel=None):
        self.settings = settings
        vda_initilizer = VDAInit(self.settings, AEmodel)
        self.data = vda_initilizer.run()

    def run(self, return_stats=False):
        """Runs the variational DA routine using settings from the passed config class
        (see config.py for example)"""

        settings = self.settings

        if settings.COMPRESSION_METHOD == "SVD":
            DA_results = self.DA_SVD()
        elif settings.COMPRESSION_METHOD == "AE":
            DA_results = self.DA_AE()
        else:
            raise ValueError("COMPRESSION_METHOD must be in {SVD, AE}")
        w_opt = DA_results["w_opt"]

        self.print_DA_results(DA_results)

        if self.settings.SAVE:
            #Save .vtu files so that I can look @ in paraview
            sample_fp = GetData.get_sorted_fps_U(self.settings.DATA_FP)[0]
            out_fp_ref = self.settings.INTERMEDIATE_FP + "ref_MAE.vtu"
            out_fp_DA =  self.settings.INTERMEDIATE_FP + "DA_MAE.vtu"

            VtkSave.save_vtu_file(ref_MAE, "ref_MAE", out_fp_ref, sample_fp)
            VtkSave.save_vtu_file(da_MAE, "DA_MAE", out_fp_DA, sample_fp)


        if return_stats:
            assert return_stats == True, "return_stats must be of type boolean. Here it is type {}".format(type(return_stats))
            stats = {}
            stats["Percent_improvement"] = 100*(DA_results["ref_MAE"] - DA_results["da_MAE_mean"])/DA_results["ref_MAE_mean"]
            stats["ref_MAE_mean"] = DA_results["ref_MAE_mean"]
            stats["da_MAE_mean"] = DA_results["da_MAE_mean"]
            return w_opt, stats

        return w_opt



    def DA_AE(self):
        if self.data.get("model") == None:
            self.model = ML_utils.load_model_from_settings(settings, self.data.get("device"))
            self.data["model"] = self.model
        else:
            self.model = self.data.get("model")

        if self.settings.REDUCED_SPACE:
            V_red = VDAInit.create_V_red(self.data.get("train_X"),
                                        self.data.get("encoder"), self.settings.get_number_modes(),
                                        self.settings)
            self.data["V_grad"] = None
            self.data["V_trunc"] = V_red
        else:

            # Now access explicit gradient function
            self.data["V_grad"] = self.__maybe_get_jacobian()

        #w_0_v1 = torch.zeros((settings.get_number_modes())).to(device)
        self.data["w_0"] = self.data.get("encoder")(self.data.get("u_0"))

        DA_results = self.perform_VarDA(self.data, self.settings)
        return DA_results

    def DA_SVD(self):

        V = VDAInit.create_V_from_X(self.data.get("train_X"), self.settings)

        if self.settings.THREE_DIM:
            #(M x nx x ny x nz)
            V = V.reshape((V.shape[0], -1)).T #(n x M)
        else:
            #(M x n)
            V = V.T #(n x M)
        V_trunc, U, s, W = SVD.TSVD(V, self.settings, self.settings.get_number_modes())

        #Define intial w_0
        s = np.where(s <= 0., 1, s) #remove any zeros (when choosing init point)
        V_plus_trunc = W.T * (1 / s) @  U.T
        w_0 = V_plus_trunc @ self.data["u_0"].flatten() #i.e. this is the value given in Rossella et al (2019).
        #w_0 = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?

        self.data["V_trunc"] = V_trunc
        self.data["V"] = V
        self.data["w_0"] = w_0
        self.data["V_grad"] = None
        DA_results = self.perform_VarDA(self.data, self.settings)
        return DA_results


    @staticmethod
    def perform_VarDA(data, settings):
        """This is a static method so that it can be performed in AE_train with user specified data"""
        args = (data, settings)

        res = minimize(cost_fn_J, data.get("w_0"), args = args, method='L-BFGS-B',
                jac=grad_J, tol=settings.TOL)

        w_opt = res.x
        u_0 = data.get("u_0")
        u_c = data.get("u_c")
        std = data.get("std")
        mean = data.get("mean")

        if settings.COMPRESSION_METHOD == "SVD":
            delta_u_DA = data.get("V_trunc") @ w_opt
            u_0 = u_0.flatten()
            u_c = u_c.flatten()
            std = std.flatten()
            mean = mean.flatten()

        elif settings.COMPRESSION_METHOD == "AE":
            delta_u_DA = data.get("decoder")(w_opt)
            if settings.THREE_DIM:
                delta_u_DA = delta_u_DA.squeeze(0)


        u_DA = u_0 + delta_u_DA


        if settings.UNDO_NORMALIZE:
            std = data.get("std")
            mean = data.get("mean")
            u_DA = (u_DA * std + mean)
            u_c = (u_c * std + mean)
            u_0 = (u_0 * std + mean)
        elif settings.NORMALIZE:
            print("Normalization not undone")

        ref_MAE = np.abs(u_0 - u_c)
        da_MAE = np.abs(u_DA - u_c)
        ref_MAE_mean = np.mean(ref_MAE)
        da_MAE_mean = np.mean(da_MAE)

        if settings.DEBUG:
            # u_0 = u_0[:1, :2, :2]
            # u_c = u_c[:1, :2, :2]
            # u_DA = u_DA[:1, :2, :2]

            size = len(u_0.flatten())
            if size > 5:
                size = 5

            print("std:    ", std.flatten()[:size])
            print("mean:   ", mean.flatten()[:size])
            print("u_0:    ", u_0.flatten()[:size])
            print("u_c:    ", u_c.flatten()[:size])
            print("u_DA:   ", u_DA.flatten()[:size])
            print("ref_MAE:", ref_MAE.flatten()[:size])
            print("da_MAE: ", da_MAE.flatten()[:size])

        results_data = {"ref_MAE": ref_MAE,
                    "da_MAE": da_MAE,
                    "u_DA": u_DA,
                    "ref_MAE_mean": ref_MAE_mean,
                    "da_MAE_mean": da_MAE_mean,
                    "w_opt": w_opt}
        return results_data

    def __maybe_get_jacobian(self):
        jac = None
        if not self.settings.JAC_NOT_IMPLEM:
            try:
                jac = self.model.jac_explicit
            except:
                pass
        else:
            import warnings
            warnings.warn("Using **Very** slow method of calculating jacobian. Consider disabling DA", UserWarning)
            jac = self.slow_jac_wrapper

        if jac == None:
            raise NotImplementedError("This model type does not have a gradient available")
        return jac

    def slow_jac_wrapper(self, x):
        return Jacobian.accumulated_slow_model(x, self.model, self.data.get("device"))

    @staticmethod
    def print_DA_results(DA_results):

        ref_MAE = DA_results["ref_MAE"]
        da_MAE = DA_results["da_MAE"]
        u_DA = DA_results["u_DA"]
        ref_MAE_mean = DA_results["ref_MAE_mean"]
        da_MAE_mean = DA_results["da_MAE_mean"]
        w_opt = DA_results["w_opt"]

        counts = (da_MAE < ref_MAE).sum()
        print("Ref MAE: {:.4f}, DA MAE: {:.4f},".format(ref_MAE_mean, da_MAE_mean), "% improvement: {:.2f}%".format(100*(ref_MAE_mean - da_MAE_mean)/ref_MAE_mean))
        print("DA_MAE < ref_MAE for {}/{} points".format(counts, len(da_MAE.flatten())))
        #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

if __name__ == "__main__":

    settings = config.Config()

    DA = DAPipeline(settings)
    DA.run()
    exit()

    #create X:
    loader = GetData()
    X = loader.get_X(settings)
    np.save(settings.get_X_fp(), X)
