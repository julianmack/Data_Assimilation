
#filepaths
DATA_FP = "data/small3DLSBU/"
X_FP = "data/small3D_intermediate/X_small3D_Tracer.npy"
INTERMEDIATE_FP = "data/small3D_intermediate/"

AE_MODEL = "models/AE_dim40_epoch120.pth"

SEED = 42

#Parameters to divide up data between "History", "observation" and "control_state"
HIST_FRAC = 2.0 / 3.0 #fraction of data used as "history"
MODE = "single_max" #"single_max" or "rand" - i.e. use a single observation or a random subset
OBS_FRAC = 0.01 #(with MODE=rand). fraction of state used as "observations"

#VarDA hyperparams
ALPHA = 1
OBS_VARIANCE = 0.01 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)
TRUNCATION_METHOD = "SVD" # "SVD"/"AE"
NUMBER_MODES = 4  #Number of modes to retain.
    # If NUMBER_MODES = None (and TRUNCATION_METHOD = "SVD"), we use
    # the Rossella et al. method for selection of truncation parameter
