"""Configuration file for VarDA. Custom data structure that holds configuration options.

User can create new classes that inherit from class Config and override class variables
in order to create new combinations of config options. Alternatively, individual config
options can be altered one at a time on an ad-hoc basis."""

from pipeline.AutoEncoders import VanillaAE, ToyNet
import socket
class Config:
    # Determine which machine this is running on. This is a short-term hack
    # which uses ip address
    ip_lst = socket.gethostbyname(socket.gethostname()).split('.')
    if int(ip_lst[0]) in [129, 192]:
        HOME_DIR = ""
    elif int(ip_lst[0]) == 146:
        #filepaths
        HOME_DIR = "/home/jfm1118/"
    else:
        raise ValueError(("IP address must start with 129 or 146. "
                        "Current IP is {}\n"
                        "Update config.py".format(socket.gethostbyname(socket.gethostname()))))

    RESULTS_FP = HOME_DIR + "results/"
    DATA_FP = HOME_DIR + "data/small3DLSBU/"
    INTERMEDIATE_FP = HOME_DIR + "data/small3D_intermediate/"
    FIELD_NAME = "Pressure"
    X_FP = INTERMEDIATE_FP + "X_small3D_{}.npy".format(FIELD_NAME)
    FORCE_GEN_X = False
    n = 100040
    SAVE = False
    DEBUG = True

    SEED = 42
    NORMALIZE = True #Whether to normalize input data
    UNDO_NORMALIZE = NORMALIZE

    #config options to divide up data between "History", "observation" and "control_state"
    #User is responsible for checking that these regions do not overlap
    HIST_FRAC = 2.0 / 4.0 #fraction of data used as "history"
    TDA_IDX_FROM_END = 0 #timestep index of u_c (control state from which
                        #observations are selcted). Value given as integer offset
                        #from final timestep (since number of historical timesteps M is
                        #not known by config file)
    OBS_MODE = "rand" #Observation mode: "single_max" or "rand" - i.e. use a single
                     # observation or a random subset
    OBS_FRAC = 0.5 # (with OBS_MODE=rand). fraction of state used as "observations".
                    # This is ignored when OBS_MODE = single_max

    #VarDA hyperparams
    ALPHA = 1.0
    OBS_VARIANCE = 0.01 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)

    COMPRESSION_METHOD = "SVD" # "SVD"/"AE"
    NUMBER_MODES = 2 #Number of modes to retain.
        # If NUMBER_MODES = None (and COMPRESSION_METHOD = "SVD"), we use
        # the Rossella et al. method for selection of truncation parameter

    TOL = 1e-3 #Tolerance in VarDA minimization routine

class ConfigExample(Config):
    """Override and add relevant configuration options."""
    ALPHA = 2.0 #override
    NEW_OPTION = "FLAG" #Add new

class ConfigAE(Config):
    NORMALIZE = True
    COMPRESSION_METHOD = "AE"
    NUMBER_MODES = 4 #this must match model above
    AE_MODEL_FP = Config().HOME_DIR + "models/AE_dim{}_epoch120.pth".format(NUMBER_MODES) #AE_dim40_epoch120.pth"
    AE_MODEL_TYPE = VanillaAE #this must match
    kwargs = {"input_size": Config().n, "latent_size": NUMBER_MODES,"hid_layers":[1000, 200]}

class ToyAEConfig(ConfigAE):
    NORMALIZE = True
    UNDO_NORMALIZE  = NORMALIZE
    NUMBER_MODES = 2
    HIDDEN = 32
    AE_MODEL_FP = Config().HOME_DIR + "models/AE_toy_{}_{}_{}.pth".format(NUMBER_MODES, HIDDEN, ConfigAE().FIELD_NAME)
    AE_MODEL_TYPE = ToyNet
    kwargs = {"inn":NUMBER_MODES, "hid":HIDDEN, "out": Config().n}

    DEBUG = True

class SmallTestDomain(Config):
    SAVE = False
    DEBUG = True
    NORMALIZE = True
    X_FP = Config().INTERMEDIATE_FP + "X_small3D_{}_TINY.npy".format(Config().FIELD_NAME)
    n = 4
    OBS_FRAC = 0.3
    NUMBER_MODES = 3
