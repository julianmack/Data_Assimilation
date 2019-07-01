# This file holds CAE configurations that can be run as a batch

from pipeline import config


class CAE1(config.CAEConfig):
    def gen_channels(self):
        channels = list(range(1, self.get_num_layers_decode() + 2))
        channels[0] = 1
        return channels

#
# class CAE2(config.CAEConfig):
#
#     def gen_channels(self):
#         channels = list(range(1, 2*(self.get_num_layers_decode() + 1) + 1, 2))
#         return channels

class CAE3(config.CAEConfig):

    def gen_channels(self):
        channels = [8] * (self.get_num_layers_decode() + 1)

        channels[0] = 1
        return channels


class CAE4(config.CAEConfig):
    def gen_channels(self):
        channels = [4] * (self.get_num_layers_decode() + 1)

        channels[0] = 1
        return channels


class CAE5(config.CAEConfig):
    def gen_channels(self):
        channels = [8] * (self.get_num_layers_decode() + 1)
        half_layers = int((self.get_num_layers_decode() + 1) / 2)
        channels[half_layers:] = [16] *  len(channels[half_layers:])
        channels[-1] = 8

        channels[0] = 1
        return channels


class CAE6(config.CAEConfig):
    def gen_channels(self):
        channels = [2 ** x for x in range (self.get_num_layers_decode() + 1)]

        half_layers = int((self.get_num_layers_decode() + 1) / 2)
        channels[half_layers + 1:] = [16] * len(channels[half_layers + 1:])
        channels[-2] = 8
        channels[-1] = 2

        channels[0] = 1
        return channels



ARCHITECTURES = [CAE1, CAE3, CAE4, CAE5, CAE6]
