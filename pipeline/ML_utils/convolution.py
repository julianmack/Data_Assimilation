from pipeline.settings import config
import os
import pytest
import numpy as np
import torch

class ConvScheduler():
    """"Creates convolutional schedule which can be used to init
    a config.CAE_3D model"""

    @staticmethod
    def conv_formula(inp, stride, pad, kernel):
        x = (inp + 2 * pad - kernel)
        if x < 0:
            raise ValueError("Cannot have (input + 2* padding) < kernel")
        return x  // stride + 1

    @staticmethod
    def conv_scheduler3D(inps, changeovers=None, lowest_outs=1, verbose = True, changeover_out_def=10):
        """Convolutional Scheduler for 3D system"""

        assert inps != None
        arg_tuples = [inps, changeovers, lowest_outs]

        args = []
        for arg in arg_tuples:
            if isinstance(arg, int) or arg == None:
                argument = (arg, arg, arg)
            else:
                assert isinstance(arg, tuple)
                assert len(arg) == 3
                argument = arg
            args.append(argument)

        inps, changeovers, lowest_outs = args[0], args[1], args[2]

        results = []
        for idx, n_i in enumerate(inps):
            res_i = ConvScheduler.conv_scheduler1D(n_i, changeovers[idx], lowest_outs[idx], changeover_out_def)
            results.append(res_i)
        min_len = min([len(i) for i in results])

        intermediate = []
        for dim_results in results:
            intermediate.append(dim_results[: min_len])

        if verbose:
            for idx, _ in enumerate(intermediate[0]):
                for dim in range(3):
                    print(intermediate[dim][idx]["in"], end=", ")
                print("stride=(", end="")
                for dim in range(3):
                    print(intermediate[dim][idx]["stride"], end=", ")
                print(")  ", end="")
                print("kernel_size=(", end="")
                for dim in range(3):
                    print(intermediate[dim][idx]["kernel"], end=", ")
                print(")  ", end="")
                print("padding=(", end="")
                for dim in range(3):
                    print(intermediate[dim][idx]["pad"], end=", ")
                print(")  ", end="")
                print()
            #final out
            for dim in range(3):
                print(results[dim][min_len - 1]["out"], end=", ")

            print("\nNum layers is:", len(intermediate[0]))
        return intermediate

    @staticmethod
    def get_init_data_from_schedule(conv_data):
        """Takes data returned from conv_scheduler3D and creates init data for CAE_3D"""
        init_data = []
        n_dims = len(conv_data)
        n_layers = len(conv_data[0])

        for layer_idx in range(n_layers):
            layer_data = []
            for dim in range(n_dims):
                layer_data.append(conv_data[dim][layer_idx])
            stride = tuple([x["stride"] for x in layer_data])
            padding = tuple([x["pad"] for x in layer_data])
            kernel = tuple([x["kernel"] for x in layer_data])
            init_layer = {"kernel_size": kernel,
                         "padding": padding,
                         "stride": stride}
            init_data.append(init_layer)


        return init_data

    @staticmethod
    def conv_scheduler1D(inp, changeover_out=None, lowest_out=1, changeover_out_def=10):
        """Desired schedule which combines stride=1 layers initially with
        later stride=2 for downsampling
        ::changeover_out - output size at which the schedule changes from stride=1 to stride=2

        """
        if changeover_out == None:
            changeover_out = inp - changeover_out_def # This is a good heuristic if you are not sure
        assert lowest_out >= 1, "lowest_out must be >= 1"
        assert changeover_out > lowest_out, "changeover_out must be > lowest_out"
        res = []
        res_s1 = ConvScheduler.conv_scheduler1D_stride1(inp, changeover_out)
        if len(res_s1) > 0:
            inp = res_s1[-1]["out"]
        res_s2 = ConvScheduler.conv_scheduler1D_stride2(inp, lowest_out)
        res_s1.extend(res_s2)
        return res_s1


    @staticmethod
    def conv_scheduler1D_stride1(inp, lowest_out = 1):
        assert lowest_out >= 1, "lowest_out must be >= 1"
        res = []
        stride = 1
        pad = 0
        kernel = 3
        while inp >= lowest_out and (inp + 2*pad) >= kernel:
            out = ConvScheduler.conv_formula(inp, stride, pad, kernel)
            res.append({"in": inp, "out": out, "stride": stride, "pad": pad, "kernel": kernel})
            inp = out
        return res

    @staticmethod
    def conv_scheduler1D_stride2(inp, lowest_out = 1):
        """Fn to find convolutional schedule that attampts to avoid:
            a) Lots of padding @ latter stages (as this may introduce artefacts)
            b) Any rounding errors in the floor operation (which are particularly
            difficult to reconstruct in the deoder of an AE)

        NOTE: lowest_out is a soft limit - a value may be accepted as part of
        the scheudle if it is slightly lower than this value"""
        res = []
        out = inp
        while inp > 3:
            pad = 0
            stride = 2
            kernel = 3
            if inp % 2 == 0: #input is even
                kernel = 2
                out = ConvScheduler.conv_formula(inp, stride, pad, kernel)
                if out % 2 == 0: #input even and output even
                    pad = 1
                    out = ConvScheduler.conv_formula(inp, stride, pad, kernel)
            else: #input is odd
                out = ConvScheduler.conv_formula(inp, stride, pad, kernel)
                if out % 2 == 0:  #input is and out is even
                    pad = 1
                    out = ConvScheduler.conv_formula(inp, stride, pad, kernel)

            res.append({"in": inp, "out": out, "stride": stride, "pad": pad, "kernel": kernel})
            inp = out
            if out <= lowest_out:
                #break
                return res
        if out <= lowest_out:
            return res

        if inp == 3:
            pad = 0
            stride = 1
            kernel = 2
            out = ConvScheduler.conv_formula(inp, stride, pad, kernel)
            res.append({"in": inp, "out": out, "stride": stride, "pad": pad, "kernel": kernel})
            inp = out
        if out <= lowest_out:
            return res
        if inp == 2:
            pad = 0
            stride = 1
            kernel = 2
            out = ConvScheduler.conv_formula(inp, stride, pad, kernel)
            res.append({"in": inp, "out": out, "stride": stride, "pad": pad, "kernel": kernel})
        return res

