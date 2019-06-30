"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE
from pipeline.AEs.CAE_configs import ARCHITECTURES as archis
from pipeline.AEs.CAE_configs import CAE1A, CAE1B

def main():
    architectures = [CAE1A, CAE1B]
    activations = ["lrelu", "relu"]
    changeover_def = [1, 3, 6, 20]
    chann_sf = [1, 0.5] #scaling factors for final channel
    EPOCHS = 25
    expdir_base = "experiments/CAE_zoo2/"
    exp_idx = 0 #experiment index (for logging)
    for changeover in changeover_def:
        for archi in architectures:
            batch_sz = 16
            settings = archi()
            #use half latent dimension
            settings.CHANGEOVER_DEFAULT = changeover
            channels = settings.get_channels(True)

            final_channel = settings.CHANNELS[-1]

            for activ in activations:
                settings.ACTIVATION = activ
                for sf in chann_sf:
                    chan = int(final_channel * sf)
                    if chan < 1:
                        chan = 1
                    settings.CHANNELS[-1] = chan
                    expdir = expdir_base + str(exp_idx)
                    try:
                        print("batch sz 16")
                        
                        batch_sz = 16
                        trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                        model = trainer.train(EPOCHS)
                    except RuntimeError:
                        try:
                            print("batch sz 8")

                            batch_sz = 8
                            trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                            model = trainer.train(EPOCHS)
                        except:
                            try:
                                print("batch sz 2")

                                batch_sz = 2
                                trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                                model = trainer.train(EPOCHS)
                            except Exception as e:
                                print("skipping - memory error: ", str(e))
                    except ValueError: #experiment has already been (at least partly) performed
                        print("skipping - already done")

                    exp_idx += 1



if __name__ == "__main__":
    main()
