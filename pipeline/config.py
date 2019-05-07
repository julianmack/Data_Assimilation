"""Configuration file for VarDA. Custom data structure that holds configuration options.

User can create new classes that inherit from class Config and override class variables
in order to create new combinations of config options."""

from AutoEncoders import VanillaAE


class Config:
    #filepaths
    DATA_FP = "data/small3DLSBU/"
    INTERMEDIATE_FP = "data/small3D_intermediate/"
    X_FP = INTERMEDIATE_FP + "X_small3D_Tracer.npy"

    SEED = 42

    #config options to divide up data between "History", "observation" and "control_state"
    #User is responsible for checking that these regions do not overlap
    HIST_FRAC = 2.0 / 3.0 #fraction of data used as "history"
    TDA_IDX_FROM_END = 2 #timestep index of u_c (control state from which
                        #observations are selcted). Value given as integer offset
                        #from final timestep (since number of historical timesteps M is
                        #not known by config file)
    OBS_MODE = "rand" #Observation mode: "single_max" or "rand" - i.e. use a single
                     # observation or a random subset
    OBS_FRAC = 0.01 # (with OBS_MODE=rand). fraction of state used as "observations".
                    # This is ignored when OBS_MODE = single_max

    #VarDA hyperparams
    ALPHA = 1.0
    OBS_VARIANCE = 0.01 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)


    COMPRESSION_METHOD = "SVD" # "SVD"/"AE"
    NUMBER_MODES = 4  #Number of modes to retain.
        # If NUMBER_MODES = None (and COMPRESSION_METHOD = "SVD"), we use
        # the Rossella et al. method for selection of truncation parameter

class ConfigExample(Config):
    """Override relevant configuration options."""
    ALPHA = 2.0

class ConfigAE(Config):
    COMPRESSION_METHOD = "AE"
    NUMBER_MODES = 4

    AE_MODEL_FP = "models/AE_dim2_epoch120.pth" #AE_dim40_epoch120.pth"
    AE_MODEL_TYPE = VanillaAE
