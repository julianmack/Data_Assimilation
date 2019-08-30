"""This is used to check that a new model sucessfully initializes from settings
It should print the number of parameters and channels"""

from VarDACAE.settings.explore.block_models import BaselineRes
from VarDACAE.settings.explore.block_models import Res34AE, Res34AE_Stacked, Cho2019
from VarDACAE.settings.models.resNeXt import Baseline1Block, ResNeXt, ResStack3
from VarDACAE.settings.explore.baseline_explore import Baseline1
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline
from VarDACAE.settings.base import Config
from VarDACAE import ML_utils
from types import ModuleType

ACTIVATION = "prelu"


########
resNext3_k = {"layers": 3, "cardinality": 1, "block_type": "RNAB",
                "module_type": "Bespoke", "sigmoid": True}
clic_k = {"model_name": "Tucodec", "block_type": "NeXt",
        "Cstd": 64, "sigmoid": True,
        "activation": "relu", "aug_scheme": 0}
grdn_k = {"block_type": "NeXt", "Cstd": 2}

CONFIGS = [ResStack3, CLIC, GRDNBaseline]
KWARGS = ( resNext3_k, clic_k, grdn_k)

###########
CONFIGS = CONFIGS[1]
KWARGS = (KWARGS[1],)


PRINT_MODEL = True

def main():

    if isinstance(CONFIGS, list):
        configs = CONFIGS
    else:
        configs = (CONFIGS, ) * len(KWARGS)
    assert len(configs) == len(KWARGS)

    for idx, conf in enumerate(configs):
        check_init(configs[idx], KWARGS[idx], PRINT_MODEL, ACTIVATION)
        print()

def check_init(config, config_kwargs, prnt, activation=None):

    if not config_kwargs:
        config_kwargs = {}
    assert isinstance(config_kwargs, dict)

    settings = config(**config_kwargs)
    settings.DEBUG = False
    #settings.ACTIVATION = activation

    assert isinstance(settings, Config)

    model = ML_utils.load_model_from_settings(settings)

    print(settings.__class__.__name__)
    if config_kwargs != {}:
        for k, v in config_kwargs.items():
            print("{}: {}".format(k, v,), end=", ")
        print(end="\n")
    if prnt:
        print(model.layers_encode)

    num_params = sum(p.numel() for p in model.parameters())
    print("num params", num_params)
    print("CHANNELS", settings.get_channels())



if __name__ == "__main__":
    main()