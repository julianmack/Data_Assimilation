from pipeline.settings.baseline import Baseline
from pipeline import ML_utils


def main():
    settings = Baseline()
    model = ML_utils.load_model_from_settings(settings)
    # for name, val in model.named_parameters():
    #     print(name, val.shape)
    # print()
    # for val in model.parameters():
    #     for channel in val:
    #         print(channel.shape, end=", ")
    #     print()

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("num params", num_params)
if __name__ == "__main__":
    main()