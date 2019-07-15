import sys, os

def win_to_unix_fp(fp):
    if sys.platform[0:3] == 'win': #i.e. windows
        #replace the backslashes with forward slashes
        fp = fp.replace("\\", '/')
        fp = fp.replace("C:", "")
    return fp

def get_home_dir():
    wd = os.getcwd()
    wd = win_to_unix_fp(wd)
    wd += "/"
    return wd