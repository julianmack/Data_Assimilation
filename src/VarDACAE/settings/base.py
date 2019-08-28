"""Configuration file for VarDA. Custom data structure that holds configuration options.

User can create new classes that inherit from class Config and override class variables
in order to create new combinations of config options. Alternatively, individual config
options can be altered one at a time on an ad-hoc basis."""

from VarDACAE.AEs import VanillaAE, ToyAE, CAE_3D
from VarDACAE import ML_utils
from VarDACAE.settings import helpers as setting_helpers


import socket
import os, sys

class Config():

    def __init__(self, loader=None):
        self.HOME_DIR = setting_helpers.get_home_dir()
        self.RESULTS_FP = self.HOME_DIR + "results/"
        self.DATA_FP = self.HOME_DIR + "data_/small3DLSBU/"
        self.INTERMEDIATE_FP = self.HOME_DIR + "data_/small3D_intermediate/"
        self.FIELD_NAME = "Pressure"
        self.FORCE_GEN_X = False
        self.__n = 100040
        self.THREE_DIM = False # i.e. is representation in 3D tensor or 1D array
        self.SAVE = True
        self.DEBUG = True
        self.GPU_DEVICE = 0


        self.LOADER = loader
        self.SEED = 42
        self.NORMALIZE = True #Whether to normalize input data
        self.UNDO_NORMALIZE = self.NORMALIZE

        self.SHUFFLE_DATA = True #Shuffle data (after splitting into historical, test and control state?)
        #config options to divide up data between "History", "observation" and "control_state"
        #User is responsible for checking that these regions do not overlap
        self.HIST_FRAC = 4.0 / 5.0 #fraction of data used as "history"
        self.TDA_IDX_FROM_END = 0 #timestep index of u_c (control state from which
                            #observations are selcted). Value given as integer offset
                            #from final timestep (since number of historical timesteps M is
                            #not known by config file)
        self.OBS_MODE = "rand" #Observation mode: "single_max" or "rand" - i.e. use a single
                         # observation or a random subset
        self.OBS_FRAC = 0.005 # (with OBS_MODE=rand). fraction of state used as "observations".
                        # This is ignored when OBS_MODE = single_max


        #VarDA hyperparams
        self.ALPHA = 0.1
        self.OBS_VARIANCE = 0.05 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)

        self.REDUCED_SPACE = False
        self.COMPRESSION_METHOD = "SVD" # "SVD"/"AE"
        self.NUMBER_MODES = 2 #Number of modes to retain.
            # If NUMBER_MODES = None (and COMPRESSION_METHOD = "SVD"), we use
            # the Rossella et al. method for selection of truncation parameter

        self.TOL = 1e-2 #Tolerance in VarDA minimization routine
        self.JAC_NOT_IMPLEM = True #whether explicit jacobian has been implemented
        self.export_env_vars()

        self.AZURE_STORAGE_ACCOUNT = "vtudata2"
        #self.AZURE_STORAGE_KEY = ""
        self.AZURE_CONTAINER = "x-data"
        self.AZURE_DOWNLOAD = True
        self.AZURE_URL = "https://vtudata2.blob.core.windows.net/x-data/X_3D_Pressure.npy"

    def get_loader(self):
        from VarDACAE import GetData #import here to avoid circular imports

        if hasattr(self, "LOADER") and self.LOADER != None:
            assert iscallable(self.LOADER)
            loader = self.LOADER()
            assert isinstance(loader, GetData())
            return loader
        else: #use fulidity data
            return GetData()

    def get_X_fp(self, force_init=False):
        if hasattr(self, "X_FP_hid") and not force_init:
            return self.X_FP_hid
        else:
            if self.THREE_DIM:
                dim = 3
            else:
                dim = 1
            self.X_FP_hid = self.INTERMEDIATE_FP + "X_{}D_{}.npy".format(dim, self.FIELD_NAME)
            return self.X_FP_hid

    def set_X_fp(self, fp):
        self.X_FP_hid = fp

    def get_n(self):
        return self.__n
    def set_n(self, n):
        self.__n = n

    def get_number_modes(self):
        return self.NUMBER_MODES

    def export_env_vars(self):
        if hasattr(self, "GPU_DEVICE"):
            gpu_idx = self.GPU_DEVICE
        else:
            gpu_idx = 0
        self.env_vars = {"SEED": self.SEED, "GPU_DEVICE": gpu_idx}

        env = os.environ
        for k, v in self.env_vars.items():
            env[str(k)] = str(v)
    def get_channels(self):
        if hasattr(self, "CHANNELS") and self.CHANNELS != None:
            return self.CHANNELS
        elif hasattr(self, "gen_channels"): #gen random channels
            self.CHANNELS = self.gen_channels()
            return self.CHANNELS
        else:
            raise NotImplementedError("No default channel init")

class ConfigExample(Config):
    """Override and add relevant configuration options."""
    def __init__(self, loader=None):
        super(ConfigExample, self).__init__(loader)
        self.ALPHA = 2.0 #override
        self.NEW_OPTION = "FLAG" #Add new


class SmallTestDomain(Config):
    def __init__(self, loader=None):
        super(SmallTestDomain, self).__init__(loader)
        self.SAVE = False
        self.DEBUG = True
        self.NORMALIZE = True
        self.X_FP_hid = self.INTERMEDIATE_FP + "X_small3D_{}_TINY.npy".format(self.FIELD_NAME)
        self.__n = 4
        self.OBS_FRAC = 0.3
        self.NUMBER_MODES = 3
