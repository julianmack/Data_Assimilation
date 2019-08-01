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

    print(model.layers_encode)
    num_params = sum(p.numel() for p in model.parameters())
    latent_shape = settings.get_kwargs()["latent_sz"]
    latent_size = 1
    for x in latent_shape:
        latent_size *= x

    print("num params", num_params)
    print("latent shape", latent_shape)
    print("latent size", latent_size)
    print("CHANNELS", settings.get_channels())
if __name__ == "__main__":
    main()