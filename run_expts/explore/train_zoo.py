"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings.CAE_configs import ARCHITECTURES as architectures
TEST_INIT_ONLY = False

#I am seeing Batch norm performing much worse than non bathc norm.
# I have a suspicion this may be beacuse I:
# A) wasn't shuffling the training data (i.e. temporal variations meant that these
#         macro variations were lost in batch norm)
# B) Was applying batch norm to the inputs - again this could mean that relevant
# input batch statistics are lost
# So I'm trying again with the above corrected. To save $$$ I will only run a
# subset of the most successful experiments run previously
# Note: there will be repeated experiments which can be used to get an idea
# of the std of these results (IF there is no noticiable change from A) and B)

def main():
    activations = ["lrelu", "relu"] # first experiments showed "lrelu" much better than "relu"
    chann_sf = [1] #D0 not run the final chapter scale factor change (for $$$)
    batch_norms = [True, False]
    changeover_def = [0]
    EPOCHS = 30
    expdir_base = "experiments/CAE_zooBN2/"
    exp_idx = 0 #experiment index (for logging)

    for archi in architectures:
        for changeover in changeover_def:
            settings = archi()

            #Set changeover default = 0 as previous experiments have shown the
            #smaller networks perfom better
            settings.CHANGEOVER_DEFAULT = 0

            channels = settings.get_channels()
            final_channel = channels[-1]
            
            for activ in activations:
                settings.ACTIVATION = activ
                for sf in chann_sf:
                    for batch_n in batch_norms:
                        settings.BATCH_NORM = batch_n
                        chan = int(final_channel * sf)
                        if chan < 1:
                            chan = 1
                        settings.CHANNELS[-1] = chan

                        if settings.BATCH_NORM == True:
                            BN = "BN"
                        else:
                            BN = "NBN" #"no batch norm"

                        expdir_str = "{}-{}-{}-{}-{}-{}".format(settings.__class__.__name__, len(settings.CHANNELS), activ, changeover, sf, BN)
                        expdir = expdir_base + expdir_str


                        settings.final_channel_sf = sf
                        try:

                            batch_sz = 16
                            print("expt", exp_idx, settings.__class__.__name__, settings.CHANNELS, activ, changeover, sf, BN)

                            trainer = TrainAE(settings, expdir, batch_sz = batch_sz)
                            if TEST_INIT_ONLY:
                                exp_idx += 1
                                continue
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

    print("Total Experiments: {}".format(exp_idx))

if __name__ == "__main__":
    main()
