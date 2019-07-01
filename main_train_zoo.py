"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE
from pipeline.AEs.CAE_configs import ARCHITECTURES as architectures

def main():
    activations = ["lrelu", "relu"]
    changeover_def = [0, 10]
    chann_sf = [1, 0.5] #scaling factors for final channel
    EPOCHS = 30
    expdir_base = "experiments/CAE_zoo3/"
    exp_idx = 0 #experiment index (for logging)

    for archi in architectures:
        for changeover in changeover_def:
            batch_sz = 16
            settings = archi()
            # use half latent dimension
            settings.CHANGEOVER_DEFAULT = changeover

            channels = settings.get_channels()
            final_channel = settings.CHANNELS[-1]

            for activ in activations:
                settings.ACTIVATION = activ
                for sf in chann_sf:
                    chan = int(final_channel * sf)
                    if chan < 1:
                        chan = 1
                    settings.CHANNELS[-1] = chan
                    expdir = expdir_base + str(exp_idx)

                    settings.final_channel_sf = sf
                    try:

                        batch_sz = 16
                        print(settings.__class__.__name__, settings.CHANNELS, activ, changeover)
                        trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                        model = trainer.train(EPOCHS)


                    except RuntimeError:
                        try:

                            batch_sz = 8
                            trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                            model = trainer.train(EPOCHS)
                        except:
                            try:

                                batch_sz = 2
                                trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                                model = trainer.train(EPOCHS)
                            except Exception as e:
                                print("skipping - memory error: ", str(e))
                    except ValueError: #experiment has already been (at least partly) performed
                        print("skipping - already done")

                    exp_idx += 1

    print("Total Experiments: {}".format(exp_idx + 1))

if __name__ == "__main__":
    main()
