import os
from fnmatch import fnmatch
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from VarDACAE import ML_utils
from collections import OrderedDict

#plt.style.use('seaborn-white')

def get_DA_info(exp_dir_base):
    max_epoch = 0
    last_df = None

    DA_data = []
    for path, subdirs, files in os.walk(exp_dir_base):
        for file in files:
            if file[-9:] == "_test.csv":
                epoch_csv = int(file.replace("_test.csv", ""))
                fp = os.path.join(path, file)
                dfDA = pd.read_csv(fp)

                DA_mean = dfDA["percent_improvement"].mean()
                DA_std = dfDA["percent_improvement"].std()
                res = (epoch_csv, dfDA, DA_mean, DA_std)
                DA_data.append(res)

                if epoch_csv >= max_epoch:
                    max_epoch = epoch_csv
                    last_df = dfDA

    #get DF with best
    mean_DA = [(epoch, mean, std) for (epoch, _, mean, std) in DA_data]
    mean_DA.sort()
    mean_DA = [{"epoch": x, "mean": y, "std1": z, "upper2std": (y + 2 * z), "lower2std": (y - 2 * z), "std2": 2 * z} for (x, y, z) in mean_DA]
    mean_DF = pd.DataFrame(mean_DA)

    return DA_data, mean_DF, last_df



#Extract results files from sub directories
def extract_res_from_files(exp_dir_base):
    """Takes a directory (or directories) and searches recursively for
    subdirs that have a test train and settings file
    (meaning a complete experiment was conducted).
    Returns:
        A list of dictionaries where each element in the list
        is an experiment and the dictionary has the following form:

        data_dict = {"train_df": df1, "test_df":df2,
                    "settings":settings, "path": path}
"""

    if isinstance(exp_dir_base, str):
        exp_dirs = [exp_dir_base]

    elif isinstance(exp_dir_base, list):
        exp_dirs = exp_dir_base
    else:
        raise ValueError("exp_dir_base must be a string or a list")


    TEST = "test.csv"
    TRAIN = "train.csv"
    SETTINGS = "settings.txt"
    results = []


    for exp_dir_base in exp_dirs:
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
                    settings = pickle.load(f)

                test_DA_df = get_DA_info(path)

                model_data = get_model_specific_data(settings, path)

                DA_data, mean_DF, last_df = get_DA_info(path)


                data_dict = {"train_df": dftrain,
                             "test_df":dftest,
                             "test_DA_df_final": last_df,
                             "DA_mean_DF": mean_DF,
                             "settings":settings,
                             "path": path,
                             "model_data": model_data,}
                results.append(data_dict)

    print("{} experiments conducted".format(len(results)))
    sort_res = sorted(results, key = lambda x: x['path'])
    return sort_res




def plot_results_loss_epochs(results, ylim1 = None, ylim2=None):
    """Plots subplot with train/valid loss vs number epochs"""

    nx = 3
    ny = int(np.ceil(len(results) / nx))
    fig, axs = plt.subplots(ny, nx,  sharey=True)
    fig.set_size_inches(nx * 5, ny * 4)
    print(axs.shape)
    color1 = 'tab:red'
    color = 'tab:blue'

    for idx, ax in enumerate(axs.flatten()):
        if idx + 1 > len(results):
            break
        test_df = results[idx]["test_df"]
        train_df = results[idx]["train_df"]
        sttn = results[idx]["settings"]
        DA_mean_DF = results[idx].get("DA_mean_DF")
        model_data = results[idx]["model_data"]

        ax.plot(test_df.epoch, test_df.reconstruction_err, 'ro-')
        ax.plot(train_df.epoch, train_df.reconstruction_err, 'g+-')
        ax.grid(True, axis='y', color=color1 )
        ax.grid(True, axis='x', )
        #############################
        # multiple line plot
        ax.set_ylabel('MSE loss', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.grid(True, axis='y', color=color)
        #set axes:
        if ylim1:
            ax.set_ylim(ylim1[0], ylim1[1])
        if ylim2:
            ax2.set_ylim(ylim2[0], ylim2[1])


        ax2.set_ylabel('Test DA percentage Improvement %', color=color)  # we already handled the x-label with ax1

        ax2.errorbar("epoch", 'mean', yerr=DA_mean_DF.std1, data=DA_mean_DF, marker='+', color=color, )

        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ########################

        try:
            latent = sttn.get_number_modes()
        except:
            latent = "??"
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

        if hasattr(sttn, "AUGMENTATION"):
            aug = sttn.AUGMENTATION
        else:
            aug = False

        if hasattr(sttn, "DROPOUT"):
            drop = sttn.DROPOUT
        else:
            drop = False

        try:
            num_layers = sttn.get_num_layers_decode()
        except:
            num_layers = "??"

        title = "act={}, ".format(activation)

        for idx, (key, value) in enumerate(model_data.items()):
            if idx % 3 == 0 and idx > 0:
                title += "\n"
            if isinstance(value, float):
                title += "{}={:.4f}, ".format(key, value)
            else:
                title += "{}={}, ".format(key, value)
        title = title[:-1]

        ax.set_title(title)
    plt.show()


def extract(res):
    """Extracts relevant data to a dataframe from the 'results' dictionary"""
    test_df = res["test_df"]
    train_df = res["train_df"]
    sttn = res["settings"]

    valid_loss = min(test_df.reconstruction_err)
    model_name = sttn.__class__.__name__
    try:
        latent = sttn.get_number_modes()
    except:
        latent = "??"

    activation = sttn.ACTIVATION
    channels = sttn.get_channels()
    num_channels = sum(channels)

    first_channel = channels[1] #get the input channel (this may be a bottleneck)

    if hasattr(sttn, "get_num_layers_decode"):
        num_layers = sttn.get_num_layers_decode()
        chan_layer = num_channels/num_layers
    else:
        num_layers = "??"
        chan_layer = "??"


    if hasattr(sttn, "CHANGEOVER_DEFAULT"):
        conv_changeover = sttn.CHANGEOVER_DEFAULT
    else:
        conv_changeover = 10

    if hasattr(sttn, "BATCH_NORM"):
        BN = bool(sttn.BATCH_NORM)
    else:
        BN = False
    if hasattr(sttn, "AUGMENTATION"):
        aug = sttn.AUGMENTATION
    else:
        aug = False

    if hasattr(sttn, "DROPOUT"):
        drop = sttn.DROPOUT
    else:
        drop = False

    if hasattr(sttn, "learning_rate"):
        lr = sttn.learning_rate
    else:
        lr = "??"

    data = {"model":model_name, "valid_loss":valid_loss, "activation":activation,
            "latent_dims": latent, "num_layers":num_layers, "total_channels":num_channels,
            "channels/layer":chan_layer, "conv_changeover": conv_changeover,
            "path": res["path"], "first_channel": first_channel, "batch_norm": BN,
            "channels": channels, "learning_rate": lr, "augmentation": aug, "dropout": drop}
    return data

def create_res_df(results, remove_duplicates=False):
    df_res = pd.DataFrame(columns=["model", "valid_loss", "activation", "latent_dims", "num_layers", "total_channels", "channels/layer"])
    for idx, res in enumerate(results):
        data = extract(res)
        df_res = df_res.append(data, ignore_index=True)

    if remove_duplicates:
        df_res_original = df_res.copy() #save original (in case you substitute out)
        columns = list(df_res_original.columns)
        columns.remove("model")
        columns.remove("path")
        df_res_new = df_res_original.loc[df_res_original.astype(str).drop_duplicates(subset=columns, keep="last").index]
        #df_res_new = df_res_original.drop_duplicates(subset=columns, keep="last")
        df_res_new.shape
        df_res = df_res_new

    return df_res

def get_attenuation_from_dir(dir, model=None):
    if not model:
        model, settings = ML_utils.load_model_and_settings_from_dir(dir)
    encode, decode = None, None
    for k, v in model.named_parameters():
        if "attenuate_res" in k:
            if "encode" in k:
                encode =  v.item()
            else:
                decode =  v.item()
    return encode, decode

def get_model_specific_data(settings, dir, model=None):
    """Helper funtion to get model data"""
    cls_name = settings.__class__.__name__
    assert cls_name in ["ResNeXt", "ResStack3", "CLIC", "GRDNBaseline"]
    mod_typ, params = get_block_params(settings)

    results = OrderedDict()
    L = params.get("L")
    N = params.get("N")
    if N:
        results["cardinality"]  = N
    if L:
        results["layers"]  = L
    if (cls_name in ["ResNeXt", "ResStack3"] and (L and N) and (L > 0 and N > 0)) or cls_name in ["CLIC", "GRDNBaseline"]:
        results["block_type"] = params.get("B")

    if mod_typ and mod_typ != "Bespoke":
        results["mod"] = mod_typ
    if params.get("SB"):
        results["sBlock"] = params.get("SB")
    if params.get("S") is not None:
        results["sigmoid"] = params.get("S")
    if params.get("Cstd"):
        results["Cstd"] = params.get("Cstd")
    if params.get("A"):
        results["activation"] = params.get("A")
    if params.get("AS") is not None:
        results["aug_scheme"] = params.get("AS")

    if cls_name in ["ResStack3", "ResNeXt"] and params.get("attenuation") in [None, True]:
        encode_att, decode_att = get_attenuation_from_dir(dir, model)
        if encode_att:
            results["enc"], results["dec"] = encode_att, decode_att

    return results

def get_block_params(settings):
    res_typ, res_params = None, None
    for val in settings.BLOCKS[1:]:
        assert isinstance(val, tuple)
        if len(val) == 2:
            continue
        elif len(val) == 3:
            _, typ, params = val
            if res_typ is not None:
                raise NotImplementedError("This can only deal with a single BLOCK in settings")
            res_typ, res_params = typ, params
        else:
            raise ValueError()
    return res_typ, res_params
