import sys, os
from copy import deepcopy

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

def flatten_list(nested_list):
    """
    Note: ref: https://gist.github.com/Wilfred/7889868


    Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist

def recursive_set(item, val):
    if type(item) == list:
        return [recursive_set(subitem, val) for subitem in item]
    else:
        return val

def recursive_update(item, update, default=None):
    if type(item) == list:
        return [recursive_update(subitem, update, default) for subitem in item]
    else:
        res = update.get(item) if update.get(item) else default

        return res

def recursive_set_same_struct(item, inputs, idx_=[0], reset_idx=False):
    if reset_idx:
        return recursive_set_same_struct(item, inputs, idx_=[0])

    if type(item) == list:
        return [recursive_set_same_struct(subitem, inputs, idx_) for subitem in item]
    else:
        [idx] = idx_
        idx_[0] = idx + 1 #use mutable object
        return inputs[idx]