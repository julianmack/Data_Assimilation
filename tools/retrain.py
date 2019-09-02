from VarDACAE import retrain
from run_expts.expt_config import ExptConfigTest, ExptConfig

TEST = True
EXTRA_EPOCHS = 1
LOSS = "L1"
NEW_EXPDIR = "experiments/retrain/06b/"


OLD_EXPDIR = "experiments/train2/06b2/3"
GPU_DEVICE = 0

def main():
    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    trainer = retrain(OLD_EXPDIR, GPU_DEVICE, NEW_EXPDIR)
    expt.LR /= 4

    model = trainer.train(EXTRA_EPOCHS, test_every=expt.test_every,
                            num_epochs_cv=expt.num_epochs_cv,
                            learning_rate = expt.LR, print_every=expt.print_every,
                            small_debug=expt.SMALL_DEBUG_DOM, loss=LOSS)

if __name__ == "__main__":
    main()