"""Configuration file for VarDA. Custom data structure that holds configuration options.

User can create new classes that inherit from Config() to specify new
combinations of easily accessible config options."""

class Config:

    #filepaths
    DATA_FP = "data/small3DLSBU/"
    X_FP = "data/small3D_intermediate/X_small3D_Tracer.npy"
    INTERMEDIATE_FP = "data/small3D_intermediate/"

    AE_MODEL = "models/AE_dim2_epoch120.pth" #AE_dim40_epoch120.pth"

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
    COMPRESSION_METHOD = "AE" # "SVD"/"AE"
    NUMBER_MODES = 4  #Number of modes to retain.
        # If NUMBER_MODES = None (and COMPRESSION_METHOD = "SVD"), we use
        # the Rossella et al. method for selection of truncation parameter
