"""Retraining experiments w and w/o L1 fine-tuning"""



from VarDACAE import retrain
from run_expts.expt_config import ExptConfigTest, ExptConfig
import pickle
import shutil
from collections import OrderedDict


TEST = False
GPU_DEVICE = 0
NUM_GPU = 1

NEW_EXPDIR_BASE = "experiments/retrain/"
LOCATION_BASE = "models_/best/"
AUGMENTATION = 1

########################
TRAIN1 = [("L1", 50)]
TRAIN2 = [("L2", 150), ("L1", 50)]

#NOTE: these locs are NOT used
models = OrderedDict([
    #("tucodec_relu_vanilla", {"loc": 'experiments/06a5/12', "sched": TRAIN1}),
    #("tucodec_prelu_next", {"loc": 'experiments/DA3/06a/1/', "sched": TRAIN1}),
    #("RDB3_27_4",  {"loc": 'experiments/09a/09a/0', "sched": TRAIN2}),
    #("ResNeXt_27_1", {"loc": 'experiments/09a/09a/2', "sched": TRAIN2}),
    #("RAB_4_next",   {"loc": 'experiments/03c/10/', "sched": TRAIN2}),
    ("GDRN_CBAM",  {"loc": 'experiments/09c/0', "sched": TRAIN2})])

def main():
    if TEST:
        expt = ExptConfigTest()
        expt.calc_DA_MAE = False
    else:
        expt = ExptConfig()

    idx = 0
    for name, data in models.items():
        idx_ = idx
        idx += 1
        if idx_ % NUM_GPU != GPU_DEVICE:
            continue


        expdir_new = NEW_EXPDIR_BASE + name + "/"
        location = LOCATION_BASE + name + "/"

        trainer = retrain(location, GPU_DEVICE, expdir_new, 8)
        expdir_new = trainer.expdir
        trainer.settings.AUG_SCHEME = AUGMENTATION

        for loss, epochs in data["sched"]:


            if loss == "L1": #use smaller learning rate
                lr = expt.LR / 4
            else:
                lr = expt.LR * 0.8


            if TEST:
                epochs = 2

            print("model: {}, loss: {}, epochs: {}, lr: {:.5f}".format(name, loss, epochs, lr))

            model = trainer.train(epochs, test_every=expt.test_every,
                                    num_epochs_cv=expt.num_epochs_cv,
                                    learning_rate =lr, print_every=expt.print_every,
                                    small_debug=expt.SMALL_DEBUG_DOM,
                                    calc_DA_MAE=expt.calc_DA_MAE, loss=loss)

        loss_sched_fp = expdir_new + "loss_sched.txt"
        with open(loss_sched_fp, "wb") as f:
            pickle.dump(data["sched"], f)


if __name__ == "__main__":
    main()