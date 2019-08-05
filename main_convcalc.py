from pipeline import AEs as AutoEncoders
from pipeline.settings import config
import numpy as np
import vtk.util.numpy_support as ns
from pipeline.ML_utils import ConvScheduler


def main():
    n = (91, 85, 32)
    #inx, iny, inz = (91, 85, 32)
    strides = [1, 2, 1, 2,  1, 2, 1, 1]
    conv_data = ConvScheduler.conv_scheduler3D(n, None, 1, True, strides=strides)
    init_data = ConvScheduler.get_init_data_from_schedule(conv_data)


    exit()
    channels = list(range(1, len(init_data) + 2))[::-1]
    cae = AutoEncoders.CAE_3D(init_data, channels)


if __name__ == "__main__":
    main()