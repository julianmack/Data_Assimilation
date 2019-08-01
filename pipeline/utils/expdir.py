import pipeline
import os

def init_expdir(expdir, ow_permitted=False):
    expdir = pipeline.settings.helpers.win_to_unix_fp(expdir)
    wd = pipeline.settings.helpers.get_home_dir()
    try:
        dir_ls = expdir.split("/")
        assert "experiments" in dir_ls
    except (AssertionError, KeyError, AttributeError) as e:
        print("~~~~~~~~{}~~~~~~~~~".format(str(e)))
        raise ValueError("expdir must be in the experiments/ directory")
    if wd in expdir:
        pass
    else:
        if expdir[0] == "/":
            expdir = expdir[1:]

        expdir = wd + expdir
    if not expdir[-1] == "/":
        expdir += "/"

    if os.path.isdir(expdir):
        if not ow_permitted:
            if len(os.listdir(expdir)) > 0:
                raise ValueError("Cannot overwrite files in expdir. Exit-ing.")
    else:
        os.makedirs(expdir)
    return expdir