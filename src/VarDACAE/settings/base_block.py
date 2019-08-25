from VarDACAE.settings.base_3D import Config3D
from VarDACAE.AEs.AE_general import GenCAE
from VarDACAE.AEs.AE_general import MODES as M
from VarDACAE.ML_utils import ConvScheduler

from itertools import cycle, islice
from VarDACAE.settings.helpers import recursive_len, recursive_set
from VarDACAE.settings.helpers import recursive_set_same_struct, recursive_update
from VarDACAE.settings.helpers import flatten_list

from copy import deepcopy

class Block(Config3D):
    """This model is the baseline CAE in my report and is *similar* to
    the model described in:

    LOSSY IMAGE COMPRESSION WITH COMPRESSIVE AUTOENCODERS (2017), Theis et al.
    http://arxiv.org/abs/1703.00395


    Note: this class is not to be confused with the settings.Config() class
    which is the base cofiguration class """
    def __init__(self):
        super(Block, self).__init__()
        self.REDUCED_SPACE = True
        self.COMPRESSION_METHOD = "AE"
        self.AE_MODEL_TYPE = GenCAE

        self.ACTIVATION = "lrelu"  #default negative_slope = 0.05
        self.BATCH_NORM = False
        self.AUGMENTATION = True
        self.DROPOUT = False
        self.DEBUG = False
        self.REM_FINAL = True

    def get_kwargs(self):
        assert hasattr(self, "BLOCKS"), "Must init self.BLOCKS"

        blocks = self.gen_blocks_with_kwargs()

        latent_sz = None
        rem_final = True if not hasattr(self, "REM_FINAL") else self.REM_FINAL
        kwargs =   {"blocks": blocks,
                    "activation": self.ACTIVATION,
                    "latent_sz": latent_sz,
                    "rem_final": rem_final}
        return kwargs

    def gen_blocks_with_kwargs(self):

        downsample = self.gen_downsample()
        channels = self.get_channels()

        if isinstance(downsample, tuple):
            strides = []
            for down in downsample:
                strides.append(self.gen_strides_flat(down))
            strides = tuple(strides)
        else:
            strides = self.gen_strides_flat(downsample)


        conv_data = ConvScheduler.conv_scheduler3D(self.get_n(), None, 1, self.DEBUG, strides=strides)
        init_data = ConvScheduler.get_init_data_from_schedule(conv_data)

        if isinstance(downsample, tuple):
            downsample = downsample[0]

        init_data_not_flat = recursive_set_same_struct(downsample, init_data, reset_idx=True)

        blocks_w_kwargs = self.gen_block_kwargs_recursive(self.BLOCKS, channels,
                                init_data=init_data_not_flat, reset_idx=True)

        return blocks_w_kwargs

    def gen_channels(self):
        if isinstance(self.BLOCKS, list):
            if self.BLOCKS[0] == M.S:
                structure = self.parse_BLOCKS()
                num_layers_conv = self.recursive_len_conv(structure)
                if hasattr(self, "CHANNELS"):
                    channels_flat = self.CHANNELS
                    assert len(channels_flat) == num_layers_conv + 1
                else:
                    channels_flat = self.channels_default(num_layers_conv)

                #channels = recursive_set_same_struct(structure, channels_flat)
            else:
                raise NotImplementedError("Parallel channel generation not implemented")
        return channels_flat

    def update_channels(self, new):
        assert isinstance(new, list)
        assert all(type(x) == int for x in new)
        old_channels = self.get_channels()
        assert len(old_channels) == len(new)
        self.CHANNELS = new
        return self.CHANNELS


    @staticmethod
    def channels_default(num_layers):
        """Returns default channel schedule of length
        num_layers (all top level list)"""

        idx_half = int((num_layers + 1) / 2)

        channels = [64] * (num_layers + 1)
        channels[idx_half:] = [32] *  len(channels[idx_half:])

        #update bespoke vals
        channels[0] = 1
        channels[1] = 16
        channels[2] = 32
        return channels

    def gen_downsample(self):
        """By default, all layers are downsampling layers (of stride 2)"""

        if not hasattr(self, "DOWNSAMPLE__"):
            structure = self.parse_BLOCKS()
            if hasattr(self, "DOWNSAMPLE"):
                down = self.DOWNSAMPLE
                assert isinstance(down, (list, int, tuple))
                if isinstance(down, int):
                    assert down in [0, 1]
                    schedule = recursive_set(deepcopy(structure), down)
                elif isinstance(down, list):
                    assert all(x in [0, 1] for x in down)
                    schedule = self.gen_downsample_recursive(down, structure)
                else:
                    assert len(down) == 3
                    assert all(len(x) == len(down[0]) for x in down)
                    sched = []
                    for dim_down in down:
                        sched.append(self.gen_downsample_recursive(dim_down, structure))
                    schedule = tuple(sched)
            else:
                update = {"conv": 1, "ResB": 0 }

                schedule = recursive_update(deepcopy(structure), update, 0) #set all to downsample

            assert len(schedule) == len(structure) or isinstance(schedule, tuple)
            self.DOWNSAMPLE__ = schedule

        return self.DOWNSAMPLE__



    def get_number_modes(self):
        raise NotImplementedError("`Number of modes` not implemented for Block class")

    #################### Everything below this point is a helper function

    def parse_BLOCKS(self):
        return self.recursive_parse_BLOCKS(self.BLOCKS)

    def recursive_parse_BLOCKS(self, blocks):
        """Returns list with expanded structure of layers in self.BLOCKS"""
        if isinstance(blocks, str):
            return blocks
        elif isinstance(blocks, list):
            assert len(blocks) > 1, "blocks must be list of structure [<MODE>, block_1, ...]"
            mode = blocks[0]
            assert isinstance(mode, str) , "blocks[0] must be a string"
            assert mode in M.all

            blocks = blocks[1:] #ignore mode
            layers_out  = []
            for idx, block in enumerate(blocks):

                assert isinstance(block, tuple)
                if len(block) == 2:
                    (num, blocks_) = block
                elif len(block) == 3:
                    (num, blocks_, _) = block
                else:
                    raise ValueError("block must be on length 2 or 3")

                layer = []
                for i in range(num):
                    layers_lower = self.recursive_parse_BLOCKS(blocks_)
                    layer.append(layers_lower)

                layers_out.append(layer)

            return layers_out
        else:
            raise ValueError("blocks must be of type str or list. Received type {}".format(type(blocks)))





    def gen_block_kwargs_recursive(self, blocks, channels, init_data, idx_=[0], reset_idx=False):
        if reset_idx:
            return self.gen_block_kwargs_recursive(blocks, channels, init_data, idx_=[0])
        if isinstance(blocks, str):
            return blocks
        elif isinstance(blocks, list):
            assert len(blocks) > 1, "blocks must be list of structure [<MODE>, block_1, ...]"
            mode = blocks[0]
            assert isinstance(mode, str) , "blocks[0] must be a string"
            assert mode in M.all

            blocks = blocks[1:] #ignore mode
            layers_out  = [mode]
            for block_idx, block in enumerate(blocks):

                init_data_lo = deepcopy(init_data[block_idx])
                assert isinstance(block, tuple)
                if len(block) == 2:
                    (num, blocks_) = block

                    if isinstance(blocks_, str):
                        #then generate kwargs
                        if not blocks_ == "conv":
                            layers_out.append((num, blocks_))
                        else:
                            kwargs_ls = []
                            for i in range(num):
                                [idx] = idx_
                                idx_[0] = idx + 1 #use mutable objecT
                                conv_kwargs = {"kernel_size": init_data_lo[i]["kernel_size"],
                                             "padding": init_data_lo[i]["padding"],
                                             "stride": init_data_lo[i]["stride"],
                                             "in_channels": channels[idx],
                                             "out_channels": channels[idx + 1],}
                                kwargs = {"conv_kwargs": conv_kwargs,
                                         "dropout": self.DROPOUT,
                                         "batch_norm": self.BATCH_NORM,}
                                #kwargs = {idx} #EDIT THIS
                                kwargs_ls.append(kwargs)
                            layers_out.append((num, blocks_, kwargs_ls))
                    else:
                        #then go recursively
                        layer = []
                        for i in range(num):
                            if mode == M.S:
                                blocks_lower = self.gen_block_kwargs_recursive(blocks_, channels, init_data_lo[i], idx_)
                                layers_out.append((1, blocks_lower)) #this only works for sequential
                            else:
                                raise NotImplementedError()


                elif len(block) == 3: #kwargs already provided
                    (num, blocks_, kwargs_ls) = block
                    assert isinstance(kwargs_ls, (list, dict))
                    assert isinstance(blocks_, str)
                    if isinstance(kwargs_ls, dict):
                        kwargs_ls = [kwargs_ls] * num
                    else:
                        assert len(kwargs_ls) == num
                    if blocks_ == "conv":
                        [idx] = idx_
                        idx_[0] = idx + num #use mutable objecT
                    layers_out.append((num, blocks_, kwargs_ls))

                else:
                    raise ValueError("block must be on length 2 or 3")

            return layers_out
        else:
            raise ValueError("blocks must be of type str or list. Received type {}".format(type(blocks)))


    @staticmethod
    def gen_strides_flat(downsample):
        downs = deepcopy(downsample)
        update_strides = {1: 2, 0:1}
        strides_not_flat = recursive_update(downs, update_strides)
        return list(flatten_list(strides_not_flat))

    @staticmethod
    def gen_downsample_recursive(down, structure):
        schedule = []
        cycled_down = list(islice(cycle(down), len(structure)))
        for idx, block in enumerate(structure):
            if isinstance(block, str):
                if block == "conv":
                    schedule.append(cycled_down[idx])
                else:
                    pass
            else:
                schedule.append(Block.gen_downsample_recursive(down, block))
        return schedule

    @staticmethod
    def recursive_len_conv(item):
        """Calculates number of layers that are conv (others are ignored)"""
        if type(item) == list:
            return sum(Block.recursive_len_conv(subitem) for subitem in item)
        else:
            if item == "conv":
                return 1
            return 0

