"""Helper function that retrains from expdir"""
from VarDACAE import ML_utils, TrainAE

def retrain(dir, gpu_device, new_expdir, batch_sz=None):
    """This function accepts an expdir and returns an initialized TrainAE class"""

    model, settings, prev_epoch = ML_utils.load_model_and_settings_from_dir(dir,
                        device_idx= gpu_device, return_epoch=True)

    start_epoch = prev_epoch + 1

    batch_sz = batch_sz if batch_sz is not None else settings.batch_sz
    
    trainer = TrainAE(settings, new_expdir, batch_sz=batch_sz,
                    model=model, start_epoch=start_epoch)


    return trainer