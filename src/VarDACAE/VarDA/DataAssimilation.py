"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import os
import random
import torch
from scipy.optimize import minimize

from VarDACAE import ML_utils
from VarDACAE.AEs import Jacobian
from VarDACAE import fluidity
from VarDACAE import SplitData
from VarDACAE.VarDA import VDAInit
from VarDACAE.VarDA import SVD
from VarDACAE.VarDA.cost_fn import cost_fn_J, grad_J

class DAPipeline():
    """Class to hold pipeline functions for Variational DA
    """

    def __init__(self, settings, AEmodel=None, u_c=None):
        self.settings = settings
        vda_initilizer = VDAInit(self.settings, AEmodel, u_c=u_c)
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



        if return_stats:
            assert return_stats == True, "return_stats must be of type boolean. Here it is type {}".format(type(return_stats))
            stats = {}
            stats["Percent_improvement"] = 100*(DA_results["ref_MAE_mean"] - DA_results["da_MAE_mean"])/DA_results["ref_MAE_mean"]
            stats["ref_MAE_mean"] = DA_results["ref_MAE_mean"]
            stats["da_MAE_mean"] = DA_results["da_MAE_mean"]
            stats["mse_ref"] =  DA_results["mse_ref"]
            stats["mse_DA"] = DA_results["mse_DA"]
            return w_opt, stats

        return w_opt



    def DA_AE(self, force_init=False):
        if self.data.get("model") == None or force_init:
            self.model = ML_utils.load_model_from_settings(self.settings, self.data.get("device"))
            self.data["model"] = self.model
        else:
            self.model = self.data.get("model")

        self.data["model"].eval()
        if self.settings.REDUCED_SPACE:
            if self.data.get("V_trunc") is None or force_init: #only init if not already init
                V_red = VDAInit.create_V_red(self.data.get("train_X"),
                                            self.data.get("encoder"),
                                            self.settings)
                self.data["V_trunc"] = V_red.T #tau x M

                self.data["w_0"] = np.zeros((V_red.shape[0]))
                if self.data["G"] is 1:
                    self.data["G_V"] =self.data["V_trunc"]
                else:
                    self.data["G_V"] = (self.data["G"] @ self.data["V_trunc"] ).astype(float)

            self.data["V_grad"] = None
        else:

            # Now access explicit gradient function
            self.data["V_grad"] = self.__maybe_get_jacobian()

        DA_results = self.perform_VarDA(self.data, self.settings)
        return DA_results

    def DA_SVD(self, force_init=False):
        if self.data.get("V") is None or force_init:
            V = VDAInit.create_V_from_X(self.data.get("train_X"), self.settings)

            if self.settings.THREE_DIM:
                #(M x nx x ny x nz)
                V = V.reshape((V.shape[0], -1)).T #(n x M)
            else:
                #(M x n)
                V = V.T #(n x M)
            V_trunc, U, s, W = SVD.TSVD(V, self.settings, self.settings.get_number_modes())

            #Define intial w_0
            V_trunc_plus = SVD.SVD_V_trunc_plus(U, s, W, self.settings.get_number_modes())
            if self.settings.NORMALIZE:
                w_0 = V_trunc_plus @ np.zeros_like(self.data["u_0"].flatten()) #i.e. this is the value given in Rossella et al (2019).
            else:
                w_0 = V_trunc_plus @ self.data["u_0"].flatten()
            #w_0 = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?

            self.data["V_trunc"] = V_trunc
            self.data["V"] = V
            self.data["w_0"] = w_0
            self.data["V_grad"] = None

            if self.data.get("G") is 1:
                self.data["G_V"] = self.data["V_trunc"]
            elif self.data.get("G") is None:
                assert self.data.get("obs_idx") is not None
                self.data["G_V"] = self.data["V_trunc"][self.data.get("obs_idx")]
            else:
                raise ValueError("G has be deprecated in favour of `obs_idx`. It should be None")
        DA_results = self.perform_VarDA(self.data, self.settings)
        return DA_results


    @staticmethod
    def perform_VarDA(data, settings):
        """This is a static method so that it can be performed in AE_train with user specified data"""
        args = (data, settings)
        w_0 = data.get("w_0")
        if w_0 is None:
            raise ValueError("w_0 was not initialized")

        res = minimize(cost_fn_J, data.get("w_0"), args = args, method='L-BFGS-B',
                jac=grad_J, tol=settings.TOL)

        w_opt = res.x
        u_0 = data.get("u_0")
        u_c = data.get("u_c")
        std = data.get("std")
        mean = data.get("mean")

        if settings.COMPRESSION_METHOD == "SVD":
            delta_u_DA = (data.get("V_trunc") @ w_opt).flatten()
            u_0 = u_0.flatten()
            u_c = u_c.flatten()
            std = std.flatten()
            mean = mean.flatten()
            u_DA = u_0 + delta_u_DA

        elif settings.COMPRESSION_METHOD == "AE" and settings.REDUCED_SPACE:
            #w_0 = data.get("w_0")
            # delta_w_DA = w_opt + w_0
            # u_DA = data.get("decoder")(delta_w_DA)
            # u_DA = u_0 + u_DA


            q_opt = data.get("V_trunc") @ w_opt
            delta_u_DA  = data.get("decoder")(q_opt)

            u_DA = u_0 + delta_u_DA
        elif settings.COMPRESSION_METHOD == "AE":
            delta_u_DA = data.get("decoder")(w_opt)
            if settings.THREE_DIM and len(delta_u_DA.shape) != 3:
                delta_u_DA = delta_u_DA.squeeze(0)

            u_DA = u_0 + delta_u_DA


        if settings.UNDO_NORMALIZE:

            u_DA = (u_DA * std + mean)
            u_c = (u_c * std + mean)
            u_0 = (u_0 * std + mean)
        elif settings.NORMALIZE:
            print("Normalization not undone")

        ref_MAE = np.abs(u_0 - u_c)
        da_MAE = np.abs(u_DA - u_c)
        ref_MAE_mean = np.mean(ref_MAE)
        da_MAE_mean = np.mean(da_MAE)
        percent_improvement = 100 * (ref_MAE_mean - da_MAE_mean)/ref_MAE_mean
        counts = (da_MAE < ref_MAE).sum()
        mse_ref = np.linalg.norm(u_0 - u_c) /  np.linalg.norm(u_c)
        mse_DA = np.linalg.norm(u_DA - u_c) /  np.linalg.norm(u_c)

        mse_percent = 100 * (mse_ref - mse_DA)/mse_ref

        results_data = {"ref_MAE": ref_MAE,
                    "da_MAE": da_MAE,
                    "u_DA": u_DA,
                    "ref_MAE_mean": ref_MAE_mean,
                    "da_MAE_mean": da_MAE_mean,
                    "percent_improvement": percent_improvement,
                    "counts": counts,
                    "w_opt": w_opt,
                    "mse_ref": mse_ref,
                    "mse_DA": mse_DA}


        if settings.SAVE:
            if False:
                out_fp_ref = settings.INTERMEDIATE_FP + "ref_MAE.vtu"
                out_fp_DA =  settings.INTERMEDIATE_FP + "da_MAE.vtu"
                fluidity.utils.save_vtu(settings, out_fp_ref, ref_MAE)
                fluidity.utils.save_vtu(settings, out_fp_DA, da_MAE)

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
            print("%", percent_improvement, "da_MAE", da_MAE_mean,"ref_MAE", ref_MAE_mean)

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
        counts = DA_results["counts"]
        mse_ref = DA_results["mse_ref"]
        mse_DA = DA_results["mse_DA"]
        print("Ref MAE: {:.4f}, DA MAE: {:.4f},".format(ref_MAE_mean, da_MAE_mean), "% improvement: {:.2f}%".format(DA_results["percent_improvement"]))
        print("DA_MAE < ref_MAE for {}/{} points".format(counts, len(da_MAE.flatten())))
        print("mse_ref: {:.4f}, mse_DA: {:.4f}".format(mse_ref, mse_DA))
        #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

if __name__ == "__main__":

    settings = config.Config()

    DA = DAPipeline(settings)
    DA.run()
