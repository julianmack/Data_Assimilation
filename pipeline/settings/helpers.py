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

def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1

def recursive_set(item, val):
    if type(item) == list:
        return [recursive_set(subitem, val) for subitem in item]
    else:
        return val

def recursive_set_same_struct(item, inputs, idx_=[0]):
    if type(item) == list:
        return [recursive_set_same_struct(subitem, inputs, idx_) for subitem in item]
    else:
        [idx] = idx_
        idx_[0] = idx + 1 #use mutable object
        return inputs[idx]