import os
from fnmatch import fnmatch
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

#Extract results files from sub directories
def extract_res_from_files(exp_dir_base):
    """Takes a directory and searches recursively for 
    subdirs that have a test train and settings file 
    (meaning a complete experiment was conducted).
    Returns:
        A list of dictionaries where each element in the list 
        is an experiment and the dictionary has the following form:
        
        data_dict = {"train_df": df1, "test_df":df2, 
                    "settings":settings, "path": path}
"""
    
    TEST = "test.csv"
    TRAIN = "train.csv"
    SETTINGS = "settings.txt"
    results = []
    for path, subdirs, files in os.walk(exp_dir_base):
        test, train, settings = None, None, None
        
        for name in files:
            if fnmatch(name, TEST):
                test = os.path.join(path, name)
            elif fnmatch(name, TRAIN):
                train = os.path.join(path, name)
            elif fnmatch(name, SETTINGS):
                settings = os.path.join(path, name)

        if test and train and settings:
            dftest = pd.read_csv(test)
            dftrain = pd.read_csv(train)
            with open(settings, "rb") as f:
                stt = pickle.load(f)
            data_dict = {"train_df": dftrain, "test_df":dftest, "settings":stt, "path": path}
            results.append(data_dict)
    print("{} experiments conducted".format(len(results)))
    return results

def plot_results_loss_epochs(results):
    """Plots subplot with train/valid loss vs number epochs"""
    
    nx = 3
    ny = int(np.ceil(len(results) / nx))
    fig, axs = plt.subplots(ny, nx,  sharey=True)
    fig.set_size_inches(nx * 5, ny * 4) 
    print(axs.shape)
    for idx, ax in enumerate(axs.flatten()):
        try:
            test_df = results[idx]["test_df"]
            train_df = results[idx]["train_df"]
            sttn = results[idx]["settings"]
        except:
            continue
        ax.plot(test_df.epoch, test_df.reconstruction_err, 'ro-')
        ax.plot(train_df.epoch, train_df.reconstruction_err, 'g+-')
        ax.grid(True)

        model_name = sttn.__class__.__name__
        latent = sttn.get_number_modes()
        activation = sttn.ACTIVATION
        
        if hasattr(sttn, "BATCH_NORM"):
            BN = sttn.BATCH_NORM
            if BN: 
                BN = "BN" 
            else: 
                BN = "NBN"
        else:
            BN = "NBN"
        
        
        if hasattr(sttn, "learning_rate"):
            lr = sttn.learning_rate
        else:
            lr = "??"
            
        num_layers = sttn.get_num_layers_decode()

        #ax.set_title(idx)
        ax.set_title("{}: {}, {}, lr={}, latent={}, layers={}".format(model_name, activation, BN, lr, latent, num_layers))

def extract(res):
    """Extracts relevant data to a dataframe from the 'results' dictionary"""
    test_df = res["test_df"]
    train_df = res["train_df"]
    sttn = res["settings"]
    
    valid_loss = min(test_df.reconstruction_err)
    model_name = sttn.__class__.__name__
    latent = sttn.get_number_modes()
    activation = sttn.ACTIVATION
    num_layers = sttn.get_num_layers_decode()
    channels = sttn.get_channels()
    num_channels = sum(channels)
    chan_layer = num_channels/num_layers
    first_channel = channels[1] #get the input channel (this may be a bottleneck)
    
    if hasattr(sttn, "CHANGEOVER_DEFAULT"):
        conv_changeover = sttn.CHANGEOVER_DEFAULT
    else:
        conv_changeover = 10
    
    if hasattr(sttn, "BATCH_NORM"):
        BN = bool(sttn.BATCH_NORM)
    else:
        BN = False
        
    if hasattr(sttn, "learning_rate"):
        lr = sttn.learning_rate
    else:
        lr = "??"
        
    data = {"model":model_name, "valid_loss":valid_loss, "activation":activation, 
            "latent_dims": latent, "num_layers":num_layers, "total_channels":num_channels, 
            "channels/layer":chan_layer, "conv_changeover": conv_changeover, 
            "path": res["path"], "first_channel": first_channel, "batch_norm": BN,
            "channels": channels, "learning_rate": lr}
    return data

def create_res_df(results):
    df_res = pd.DataFrame(columns=["model", "valid_loss", "activation", "latent_dims", "num_layers", "total_channels", "channels/layer"])
    for idx, res in enumerate(results):
        data = extract(res)
        df_res = df_res.append(data, ignore_index=True)
    return df_res