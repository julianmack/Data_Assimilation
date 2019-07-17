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
from pipeline.VarDA.SVD import TSVD
class DAPipeline():
    """Class to hold pipeline functions for Variational DA
    """

    def __init__(self, settings):
        self.settings = settings

    def run(self, return_stats=False):
        """Runs the variational DA routine using settings from the passed config class
        (see config.py for example)"""

        vda_initilizer = VDAInit(self.settings)
        self.data, std, mean = vda_initilizer.run()

        settings = self.settings

        if settings.COMPRESSION_METHOD == "SVD":
            DA_results = self.DA_SVD()
        elif settings.COMPRESSION_METHOD == "AE":
            DA_results = self.DA_AE()
        else:
            raise ValueError("COMPRESSION_METHOD must be in {SVD, AE}")

        ref_MAE = DA_results["ref_MAE"]
        da_MAE = DA_results["da_MAE"]
        u_DA = DA_results["u_DA"]
        ref_MAE_mean = DA_results["ref_MAE_mean"]
        da_MAE_mean = DA_results["da_MAE_mean"]
        w_opt = DA_results["w_opt"]


        if self.settings.DEBUG:
            size = len(std)
            if size > 4:
                size = 4
            print("std:    ", std[-size:])
            print("mean:   ", mean[-size:])
            print("u_0:    ", self.data.get("u_0")[-size:])
            print("u_c:    ", self.data.get("u_c")[-size:])
            print("u_DA:   ", u_DA[-size:])
            print("ref_MAE:", ref_MAE[-size:])
            print("da_MAE: ", da_MAE[-size:])

        counts = (ref_MAE > da_MAE).sum()

        print("RESULTS")

        print("Reference MAE: ", ref_MAE_mean)
        print("DA MAE: ", da_MAE_mean)
        print("ref_MAE_mean > da_MAE_mean for {}/{}".format(counts, da_MAE.shape[0]))
        print("If DA has worked, DA MAE > Ref_MAE")
        print("Percentage improvement: {:.2f}%".format(100*(ref_MAE_mean - da_MAE_mean)/ref_MAE_mean))
        #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

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
            stats["Percent_improvement"] = 100*(ref_MAE_mean - da_MAE_mean)/ref_MAE_mean
            stats["ref_MAE_mean"] = ref_MAE_mean
            stats["da_MAE_mean"] = da_MAE_mean
            return w_opt, stats

        return w_opt

    def DA_AE(self):

        device = self.data.get("device")
        self.model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        weights = torch.load(settings.AE_MODEL_FP, map_location=device)
        self.model.load_state_dict(weights)

        self.data["model"] = self.model

        self.data["V_trunc"] = self.model.decode

        #w_0_v1 = torch.zeros((settings.get_number_modes())).to(device)
        self.data["w_0"] = self.model.encode(torch.FloatTensor(self.data.get("u_0_not_flat")).unsqueeze(0))

        # Now access explicit gradient function
        self.data["V_grad"] = self.__maybe_get_jacobian()

        DA_results = self.perform_VarDA(self.data, self.settings)
        return DA_results

    def DA_SVD(self):
        V_trunc, U, s, W = TSVD(self.data["V"], self.settings, self.settings.get_number_modes())

        #Define intial w_0
        s = np.where(s <= 0., 1, s) #remove any zeros (when choosing init point)
        V_plus_trunc = W.T * (1 / s) @  U.T
        w_0 = V_plus_trunc @ self.data["u_0"] #i.e. this is the value given in Rossella et al (2019).
        #w_0 = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?

        self.data["V_trunc"] = V_trunc
        self.data["w_0"] = w_0
        self.data["V_grad"] = None

        DA_results = self.perform_VarDA(self.data, self.settings)
        return DA_results


    @staticmethod
    def perform_VarDA(data, settings):
        """This is a static method so that it can be performed in AE_train with user specified data"""
        args = (data, settings)

        res = minimize(DAPipeline.cost_function_J, data.get("w_0"), args = args, method='L-BFGS-B',
                jac=DAPipeline.grad_J, tol=settings.TOL)

        w_opt = res.x
        if settings.COMPRESSION_METHOD == "SVD":
            delta_u_DA = data.get("V_trunc") @ w_opt
        elif settings.COMPRESSION_METHOD == "AE":
            delta_u_DA = data.get("V_trunc")(torch.Tensor(w_opt)).detach().numpy().flatten()

        u_0 = data.get("u_0")
        u_c = data.get("u_c")
        print(u_c.shape, "NOW")

        u_DA = u_0 + delta_u_DA

        #Undo normalization
        if settings.UNDO_NORMALIZE:
            std = data.get("std")
            mean = data.get("mean")
            u_DA = (u_DA.T * std + mean).T
            u_c = (u_c.T * std + mean).T
            u_0 = (u_0.T * std + mean).T
        elif settings.NORMALIZE:
            print("Normalization not undone")

        ref_MAE = np.abs(u_0 - u_c)
        da_MAE = np.abs(u_DA - u_c)
        ref_MAE_mean = np.mean(ref_MAE)
        da_MAE_mean = np.mean(da_MAE)

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
    def cost_function_J(w, data, settings):
        """Computes VarDA cost function.
        NOTE: eventually - implement this by hand as grad_J and J share quantity Q"""

        device = data.get("device")
        d = data.get("d")
        G = data.get("G")
        V_trunc = data.get("V_trunc")
        V =  V_trunc if V_trunc is not None else data.get("V")
        V_grad = data.get("V_grad")
        R_inv = data.get("R_inv")

        sigma_2 = settings.OBS_VARIANCE
        mode = settings.COMPRESSION_METHOD
        alpha = settings.ALPHA

        if mode == "SVD":
            Q = (G @ V @ w - d)

        elif mode == "AE":
            assert callable(V), "V must be a function if mode=AE is used"

            w_tensor = torch.Tensor(w).to(device)

            V_w = V(w_tensor).detach().cpu().numpy()
            V_w = V_w.flatten()
            Q = (G @ V_w - d)

        else:
            raise ValueError("Invalid mode")

        if sigma_2 and not R_inv:
            #When R is proportional to identity
            J_o = 0.5 / sigma_2 * np.dot(Q, Q)
        elif R_inv:
            J_o = 0.5 * Q.T @ R_inv @ Q
        else:
            raise ValueError("Either R_inv or sigma must be provided")

        J_b = 0.5 * alpha * np.dot(w, w)
        J = J_b + J_o

        if settings.DEBUG:
            print("J_b = {:.2f}, J_o = {:.2f}".format(J_b, J_o))
        return J


    @staticmethod
    def grad_J(w, data, settings):
        device = data.get("device")
        d = data.get("d")
        G = data.get("G")
        V_trunc = data.get("V_trunc")
        V =  V_trunc if V_trunc is not None else data.get("V")
        V_grad = data.get("V_grad")
        R_inv = data.get("R_inv")

        sigma_2 = settings.OBS_VARIANCE
        mode = settings.COMPRESSION_METHOD
        alpha = settings.ALPHA

        if mode == "SVD":
            Q = (G @ V @ w - d)
            P = V.T @ G.T
        elif mode == "AE":
            assert callable(V_grad), "V_grad must be a function if mode=AE is used"
            model = data.get("model").to(device)

            w_tensor = torch.Tensor(w).to(device)


            V_w = V(w_tensor).detach().cpu().numpy()
            V_w = V_w.flatten()
            V_grad_w = V_grad(w_tensor).detach().cpu().numpy()

            Q = (G @ V_w - d)
            P = V_grad_w.T @ G.T
        if not R_inv and sigma_2:
            #When R is proportional to identity
            grad_o = (1.0 / sigma_2 ) * np.dot(P, Q)
        elif R_inv:
            J_o = 1.0 * P @ R_inv @ Q
        else:
            raise ValueError("Either R_inv or sigma must be non-zero")

        grad_J = alpha * w + grad_o

        return grad_J


if __name__ == "__main__":

    settings = config.Config()

    DA = DAPipeline(settings)
    DA.run()
    exit()

    #create X:
    loader = GetData()
    X = loader.get_X(settings)
    np.save(settings.X_FP, X)
