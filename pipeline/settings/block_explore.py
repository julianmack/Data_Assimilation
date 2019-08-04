from pipeline.settings.config import Config3D
from pipeline.AEs.AE_general import GenCAE
from pipeline.AEs.AE_general import MODES as M

from itertools import cycle, islice
from pipeline.settings.helpers import recursive_len, recursive_set, recursive_set_same_struct
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
        self.AUGMENTATION = False
        self.DROPOUT = False

        #I enter:

        self.BLOCKS = [M.S, (2, "conv")]
        self.BLOCKS = [M.S, (2, "conv2"), (3, "conv3"), (1,  [M.S, (2, "conv"), (1, "conv")])]


        self.DOWNSAMPLE = [0, 1, 1, 1]
        #self.DOWNSAMPLE = 0


    def parse_BLOCKS(self):
        return self.recursive_parse_BLOCKS(self.BLOCKS)

    def recursive_parse_BLOCKS(self, blocks):
        """Returns list with expanded structure of layers in self.BLOCKS"""
        if isinstance(blocks, str):
            return 1
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


    def get_num_layers_decode(self):
        print("returning cst number layers")

        return 2

    def gen_channels(self):
        if isinstance(self.BLOCKS, list):
            if self.BLOCKS[0] == M.S:
                structure = self.parse_BLOCKS()
                num_layers_dec = recursive_len(structure)
                #TODO - add check for own channels here
                channels_flat = self.channels_default(num_layers_dec + 1)

                #channels = recursive_set_same_struct(structure, channels_flat)
            else:
                raise NotImplementedError("Parallel channel generation not implemented")
        return channels_flat


    def gen_downsample(self):
        """By default, all layers are downsampling layers (of stride 2)"""
        #num_layers = self.get_num_layers_decode()
        structure = self.parse_BLOCKS()
        if hasattr(self, "DOWNSAMPLE"):
            down = self.DOWNSAMPLE
            assert isinstance(down, (list, int))
            if isinstance(down, int):
                assert down in [0, 1]
                schedule = recursive_set(deepcopy(structure), down)
            else:
                assert all(x in [0, 1] for x in down)
                schedule = self.gen_downsample_recursive(down, structure)
        else:
            schedule = structure
        assert len(schedule) == len(structure)

        return schedule


    def get_kwargs(self):
        print(self.BLOCKS)
        blocks = self.gen_blocks_with_kwargs()
        latent_sz = None
        print(blocks)
        print(self.BLOCKS)



        kwargs =   {"blocks": blocks,
                    "activation": self.ACTIVATION,
                    "latent_sz": latent_sz}
        return kwargs

    def gen_blocks_with_kwargs(self):

        downsample = self.gen_downsample()
        channels = self.gen_channels()
        print(channels)
        print(downsample)
        blocks_w_kwargs = self.gen_block_kwargs_recursive(self.BLOCKS, downsample, channels,)
        return blocks_w_kwargs

        conv1 = deepcopy(conv_kwargs)
        conv2 = deepcopy(conv_kwargs)
        conv1["in_channels"] = 1
        conv1["out_channels"] = 56
        conv2["in_channels"] = 56
        conv2["out_channels"] = 24

        kwargs1 = { "dropout": self.DROPOUT,
                    "batch_norm": self.BATCH_NORM}
        kwargs2 = deepcopy(kwargs1)

        kwargs1["conv_kwargs"] =  conv1
        kwargs2["conv_kwargs"] =  conv2

        kwargs_ls = [kwargs1, kwargs2]



    def gen_block_kwargs_recursive(self, blocks, downsample, channels, idx_=[0]):
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

                print(block, downsample[block_idx])

                downsample_lo = deepcopy(downsample[block_idx])
                assert isinstance(block, tuple)
                if len(block) == 2:
                    (num, blocks_) = block

                    if isinstance(blocks_, str):
                        #then generate kwargs
                        kwargs_ls = []
                        for i in range(num):
                            [idx] = idx_
                            idx_[0] = idx + 1 #use mutable objecT
                            if downsample_lo[i]:
                                stride = (2, 2, 2)
                            else:
                                stride = (1, 1, 1)
                            conv_kwargs = {"kernel_size": (3, 3, 3),
                                         "padding": None,
                                         "stride": stride,
                                         "in_channels": channels[idx],
                                         "out_channels": channels[idx + 1],}
                            kwargs = {"conv_kwargs": conv_kwargs,
                                     "dropout": self.DROPOUT,
                                     "batch_norm": self.BATCH_NORM,}
                            kwargs_ls.append(kwargs)

                        layers_out.append((num, blocks_, kwargs_ls))
                    else:
                        #then go recursively
                        layer = []
                        for i in range(num):
                            blocks_lower = self.gen_block_kwargs_recursive(blocks_, downsample_lo[i], channels, idx_)
                            layer.append(blocks_lower)
                        layers_out.append(layer)

                elif len(block) == 3: #kwargs already provided
                    (num, blocks_, kwargs_ls) = block
                    assert isinstance(kwargs_ls, (list, dict))
                    assert isinstance(blocks_, str)
                    if isinstance(kwargs_ls, dict):
                        kwargs_ls = [kwargs_ls] * num
                    else:
                        assert len(kwargs_ls) == num
                    #in this case, all blocks are
                    #TODO - increment the idx
                    [idx] = idx_
                    idx_[0] = idx + num #use mutable objecT
                    layers_out.append((num, blocks_, kwargs_ls))

                else:
                    raise ValueError("block must be on length 2 or 3")

            return layers_out
        else:
            raise ValueError("blocks must be of type str or list. Received type {}".format(type(blocks)))



    @staticmethod
    def gen_downsample_recursive(down, structure):
        schedule = []
        cycled_down = list(islice(cycle(down), len(structure)))
        for idx, block in enumerate(structure):
            if isinstance(block, int):
                schedule.append(cycled_down[idx])
            else:
                schedule.append(Block.gen_downsample_recursive(down, block))
        return schedule


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

