# This file holds a range of CAE channel configurations that
# can be used to experiment on the best setup in main_train_zoo.py

from VarDACAE.settings.base_CAE import CAEConfig


class CAE1(CAEConfig):
    def gen_channels(self):
        channels = list(range(1, self.get_num_layers_decode() + 2))
        channels[0] = 1
        return channels

class CAE3(CAEConfig):

    def gen_channels(self):
        channels = [8] * (self.get_num_layers_decode() + 1)

        channels[0] = 1
        return channels


class CAE4(CAEConfig):
    def gen_channels(self):
        channels = [4] * (self.get_num_layers_decode() + 1)

        channels[0] = 1
        return channels


class CAE5(CAEConfig):
    def gen_channels(self):
        channels = [8] * (self.get_num_layers_decode() + 1)
        half_layers = int((self.get_num_layers_decode() + 1) / 2)
        channels[half_layers:] = [16] *  len(channels[half_layers:])
        channels[-1] = 8

        channels[0] = 1
        return channels


class CAE6(CAEConfig):
    def gen_channels(self):
        channels = [2 ** x for x in range (self.get_num_layers_decode() + 1)]

        half_layers = int((self.get_num_layers_decode() + 1) / 2)
        channels[half_layers + 1:] = [16] * len(channels[half_layers + 1:])
        channels[-2] = 8
        channels[-1] = 2

        channels[0] = 1
        return channels



ARCHITECTURES = [CAE1, CAE3, CAE4, CAE5, CAE6]

####### Architectures below this line are no longer used but are retained here
####### in order to enable interpretation of results with these architectures
####### (i.e the config classes are pickled and then loaded at analysis time)

class CAE1A(CAEConfig):
    def gen_channels(self):

        channels = list(range(1, self.get_num_layers_decode() + 2))
        channels = [2 * x for x in channels]
        channels[0] = 1
        return channels

class CAE1B(CAEConfig):
    def gen_channels(self):

        channels = list(range(1, self.get_num_layers_decode() + 2))
        channels = [3 * x for x in channels]
        channels[0] = 1
        return channels


class CAE2(CAEConfig):

    def gen_channels(self):
        channels = list(range(1, 2*(self.get_num_layers_decode() + 1) + 1, 2))
        return channels


