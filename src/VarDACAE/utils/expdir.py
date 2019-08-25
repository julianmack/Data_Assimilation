import VarDACAE
import os

def init_expdir(expdir, ow_permitted=False):
    """Helper function to initialize experiment directory
    Arguments
        expdir (str/path) - directory to write experimental results to
        ow_permitted (bool) - can previous results be overwitten (if expdir is already populated)"""
    expdir = VarDACAE.settings.helpers.win_to_unix_fp(expdir)
    wd = VarDACAE.settings.helpers.get_home_dir()
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
            files = os.listdir(expdir)
            if len(files) == 1 and files[0] == 'settings.txt': #allow overwrite
                pass
            elif len(files) > 0:
                raise ValueError("Cannot overwrite files in expdir={}. Exit-ing.".format(expdir))
    else:
        os.makedirs(expdir)
    return expdir