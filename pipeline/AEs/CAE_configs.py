# This file holds CAE configurations that can be run as a batch

from pipeline import config

architectures = [CAE1, CAE2, CAE3, CAE4, CAE5, CAE6]

class CAE1(config.CAEConfig):
    def __init__(self):
        super(CAE1, self).__init__()

        channels = list(range(1, self.LAYERS_DECODE + 2))
        self.CHANNELS = channels

class CAE2(config.CAEConfig):
    def __init__(self):
        super(CAE2, self).__init__()

        channels = list(range(1, 2*(self.LAYERS_DECODE + 1) + 1, 2))
        self.CHANNELS = channels

class CAE3(config.CAEConfig):
    def __init__(self):
        super(CAE3, self).__init__()

        channels = [8] * (self.LAYERS_DECODE + 1)
        channels[0] = 1

        self.CHANNELS = channels

class CAE4(config.CAEConfig):
    def __init__(self):
        super(CAE4, self).__init__()

        channels = [4] * (self.LAYERS_DECODE + 1)
        channels[0] = 1

        self.CHANNELS = channels

class CAE5(config.CAEConfig):
    def __init__(self):
        super(CAE5, self).__init__()

        channels = [8] * (self.LAYERS_DECODE + 1)
        half_layers = int((self.LAYERS_DECODE + 1) / 2)
        channels[half_layers:] = 16
        channels[-1] = 8
        channels[0] = 1

        self.CHANNELS = channels

class CAE6(config.CAEConfig):
    def __init__(self):
        super(CAE6, self).__init__()

        channels = [2 ** x for x in range (self.LAYERS_DECODE + 1)]
        half_layers = int((self.LAYERS_DECODE + 1) / 2)
        channels[half_layers + 1:] = 16
        channels[-2] = 8
        channels[-1] = 2
        channels[0] = 1

        self.CHANNELS = channels