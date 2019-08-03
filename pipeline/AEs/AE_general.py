import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from pipeline.AEs import BaseAE


class MODES():
    S = "SEQENTIAL"
    PA = "PARALLEL_ADD"
    PC = "PARALLEL_CONCAT"
    all = [S, PA, PC]

class GenCAE(BaseAE):
    """Initializes a general CAE using recursive data structure to define the architecture.

    Args:
        blocks (list/string)    - If string, this string describes an nn.module.
                                - If list, it is a recursive structure with shorthand
            describing the structure of nn.modules in encoder. This list will be mirrored
            and repeated (with relevant changes) in the decoder. The list must have the
            following recursive structure:
                [<MODE>, block_1, block_2, ...]
                where:
                    <MODE> (string) - operation for subsequent blocks. Must be one of:
                                     ["SEQENTIAL", "PARALLEL_ADD", "PARALLEL_CONCAT"]
                    block_x (tuple) - Tuple of length 2/3 of structure:
                        block_x = (number_x, blocks_, kwargs_block_x)
                    where:
                        number_x (int) - is number of times to repeat 'blocks_'
                        blocks_ (list/str) - of same structure as `blocks`
                        kwargs_block_x (dict) - optional. dictionary required to init blocks_
        activation (str)- Activation function between blocks. One of ["relu", "lrelu", "prelu", "GDN"]
        latent_sz (int) - latent size of fixed size input.

        batch_norm (bool) - Whether to use batch normalization between layers (where appropriate).
                               Note: if a block is used that has BN included, this will override this param
        dropout (bool) -  Whether to use dropout between all layers (where appropriate)


    """
    def __init__(self, blocks, activation = "relu", latent_sz=None, batch_norm = False,
                                                                        dropout=False):

        super(GenCAE, self).__init__()

        self.batch_norm = batch_norm
        self.dropout = dropout

        if activation == "lrelu":
            self.act_fn = nn.LeakyReLU(negative_slope = 0.05, inplace=False)
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "prelu":
            raise NotImplementedError()
        elif activation == "GDN":
            raise NotImplementedError()
        else:
            raise NotImplemtedError("Activation function must be in {"relu", "lrelu", "prelu", "GDN"}")

        self.layers_encode = self.parse_blocks(blocks, encode=True)
        self.layers_decode = self.parse_blocks(blocks[::-1], encode=False)
        self.latent_sz = latent_sz

    def parse_blocks(self, blocks, encode, kwargs=None):
        if isinstance(blocks, str):
            return self.parse_blocks_str(blocks, encode, kwargs)
        elif isinstance(blocks, list):
            assert len(blocks) > 1, "blocks must be list of structure [<MODE>, block_1, ...]"
            mode = blocks[0]
            assert isinstance(mode, str) , "blocks[0] must be a string"
            assert mode in MODES.all

            blocks_expanded = []
            for block in blocks[1:]:
                assert isinstance(block, tuple)
                if len(block) == 2:
                    (num, blocks_) = block
                elif len(block) == 3:
                    assert isinstance(blocks, str), "Only give kwargs for node element of blocks data-structure"
                    (num, blocks_, kwargs) = block
                else:
                    raise ValueError("block must be on length 2 or 3")

                layers_lower = parse_blocks(blocks_, encode, kwargs) #nn.module
                layer = OrderedDict()
                for i in range(num):
                    layer.update({i: layers_lower})

                blocks_expanded.append(nn.Sequential(layer))


            if mode == SEQENTIAL:
                layers_out = nn.Sequential(blocks_expanded)
            elif mode == PARALLEL_ADD:
                raise NotImplementedError("")
                layers_out = nnParallelAdd(blocks_expanded)
            elif mode == PARALLEL_CONCAT:
                raise NotImplementedError("")
                layers_out = nnParallelConcat(blocks_expanded)
            else:
                raise ValueError()

            return layers_out
        else:
            raise ValueError("blocks must be of type str or list. Received type {}".format(type(blocks)))

    def parse_blocks_str(self, block, encode, kwargs):
        if block == "conv": #this is poorly named - simple conv?
            return nn.Conv3d(**kwargs) if encode else nn.ConvTranspose3d(**kwargs)
        else:
            raise NotImplementedError("Only conv block is implemented")



    def __conv_maybe_BN_or_drop(self, Cin, Cout, data, transpose, dropout):
        layer = OrderedDict()
        if dropout:
            layer.update({"00": nn.Dropout3d(0.33)})
        if self.batch_norm:
            layer.update({"0": nn.BatchNorm3d(Cin)})
        if not transpose:
            layer.update({"1": nn.Conv3d(Cin, Cout, **data)})
        else:
            layer.update({"1": nn.ConvTranspose3d(Cin, Cout, **data)})
        conv = nn.Sequential(layer)
        return conv
