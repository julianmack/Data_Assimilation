from pipeline.settings.baseline_explore import Baseline2
from pipeline.settings.block_models import Baseline2_replicate, BaselineBlock
from pipeline import ML_utils


def main():
    settings = Baseline2()
    model = ML_utils.load_model_from_settings(settings)
    print(model.layers_decode)
    num_params = sum(p.numel() for p in model.parameters())
    print("num params", num_params)
    print("CHANNELS", settings.get_channels())

    settings = Baseline2_replicate()
    model = ML_utils.load_model_from_settings(settings)
    print(model.layers_decode)
    num_params = sum(p.numel() for p in model.parameters())
    print("num params", num_params)
    print("CHANNELS", settings.get_channels())

    settings = BaselineBlock()
    model = ML_utils.load_model_from_settings(settings)
    print(model.layers_decode)
    num_params = sum(p.numel() for p in model.parameters())
    print("num params", num_params)
    print("CHANNELS", settings.get_channels())



if __name__ == "__main__":
    main()
