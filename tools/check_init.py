"""This is used to check that a new model sucessfully initializes
It should print the number of parameters and channels"""

from pipeline.settings.block_models import Res34AE, Res34AE_Stacked, Cho2019
from pipeline.settings.config import Config

from pipeline.settings.models_.resNeXt import Baseline1Block, ResNeXt
from pipeline.settings.baseline_explore import Baseline1
from pipeline import ML_utils
from types import ModuleType


resNext_k = {"layers": 0, "cardinality": 2}

# CONFIGS = [Res34AE, ResNeXt, Baseline1Block, Cho2019]
# KWARGS = (0, resNext_k, 0, 0)

CONFIGS = [ResNeXt,]
KWARGS = (resNext_k,)

PRINT_MODEL = True

def main():

    if isinstance(CONFIGS, list):
        configs = CONFIGS
    else:
        configs = (CONFIGS, ) * len(KWARGS)
    assert len(configs) == len(KWARGS)

    for idx, conf in enumerate(configs):
        check_init(configs[idx], KWARGS[idx], PRINT_MODEL)
        print()

def check_init(config, config_kwargs, prnt):

    if not config_kwargs:
        config_kwargs = {}
    assert isinstance(config_kwargs, dict)

    settings = config(**config_kwargs)
    settings.DEBUG = False
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