

import torch

from torchvision.transforms import Lambda
from torchvision import transforms

DIM_DICT = {"x": 1,
            "y": 2}


def get_augment(settings):
    """Helper function to choose augmentation scheme"""

    if hasattr(settings, "AUG_SCHEME") and settings.AUG_SCHEME is not None:
        if settings.AUG_SCHEME == -1:
            trnsfrm = transforms.Compose([
                            transforms.RandomApply([FlipHorizontal("x"),
                                                    FlipHorizontal("y")], p=0.5),
                            transforms.RandomChoice([FieldJitter(0.01, 0.1),
                                                    FieldJitter(0.005, 0.5),
                                                    FieldJitter(0., 0.)], ),
                            ])
        elif settings.AUG_SCHEME == 0:
            trnsfrm = None
        elif settings.AUG_SCHEME == 1:
            trnsfrm = transforms.Compose([
                            transforms.RandomChoice([FieldJitter(0.01, 0.1),
                                                    FieldJitter(0.005, 0.5),
                                                    FieldJitter(0., 0.)], ),
                            ])
        elif settings.AUG_SCHEME == 2:
            trnsfrm = transforms.Compose([
                            transforms.RandomChoice([FieldJitter(0.01, 0.1),
                                                    FieldJitter(0.005, 0.5),], ),
                            ])
        elif settings.AUG_SCHEME == 3:
            trnsfrm = transforms.Compose([
                            transforms.RandomChoice([FieldJitter(0.005, 0.5),], ),
                            ])
        elif settings.AUG_SCHEME == 4:
            trnsfrm = transforms.Compose([
                            transforms.RandomChoice([FieldJitter(0.05, 0.25),], ),
                            ])
        elif settings.AUG_SCHEME == 5:
            trnsfrm = transforms.Compose([
                            transforms.RandomChoice([FieldJitter(0.10, 0.5),], ),
                            ])
        elif settings.AUG_SCHEME == 6:
            trnsfrm = transforms.Compose([
                            transforms.RandomChoice([FieldJitter(0.2, 1.0),], ),
                            ])
        else:
            raise ValueError("AUG_SCHEME not recognized")
    else:
        trnsfrm = None
        settings.AUG_SCHEME = 0

    return trnsfrm

class FieldJitter():
    """Jitters 3D field. Analogue of torchvision's ColourJitter transform

    Args:
        jitter_std (float) - standard deviation of the jitter that is applied
        ratio_apply (float) - proportion of locations to which jitter is applied

    """
    def __init__(self, jitter_std, ratio_apply):
        assert isinstance(jitter_std, float)
        assert jitter_std >= 0 and jitter_std < 1.0
        assert isinstance(ratio_apply, float)
        assert ratio_apply >= 0 and ratio_apply <= 1.0

        self.jitter_std = jitter_std
        self.ratio_apply = ratio_apply

    def __call__(self, sample):
        assert len(sample[0].shape) == 4 #each sample must be of shape C, nx, ny, nz

        results = []
        for x in sample:
            apply_jitter = (torch.rand_like(x) < self.ratio_apply).type(torch.float)
            std = torch.randn_like(x) #the data is normalized and mean centred
            std = self.jitter_std * std
            std_apply = std * apply_jitter
            res = x + std_apply

            results.append(res)


        return tuple(results)


class FlipHorizontal():
    """Flip 3D field horizontally (i.e. in the x or y direction). No
    vertical flip allowed.

    Args:
        dimension (string): Dimension to flip. Must be in ["x", "y"]
    """

    def __init__(self, dimension):
        assert isinstance(dimension, str)
        assert dimension in DIM_DICT.keys()
        self.dim_idx = DIM_DICT[dimension]

    def __call__(self, sample):

        assert len(sample[0].shape) == 4 #each sample must be of shape C, nx, ny, nz
        results = []
        for x in sample:
            results.append(torch.flip(x, [self.dim_idx]))
        return tuple(results)

class RotateHorizontal():
    """Rotate 3D field horizontally (i.e. around the z axis).
     All "rotations" must be by 90 degrees to avoid interpolation

     Args:
        degrees (string): Dimension to flip. Must be in ["x", "y"]
    """
    def __init__(self, degrees):
        assert isinstance(degrees, int)
        assert degrees in [0, 90, 180, 270]
        self.degrees = degrees

    def __call__(self, sample):

        assert len(sample[0].shape) == 4 #each sample must be of shape C, nx, ny, nz
        results = []
        for res in sample:
            if self.degrees == 90:
                res = res.permute([0, 2, 1, 3])
                res = torch.flip(res, [DIM_DICT["y"]])
            elif self.degrees == 180:
                res = torch.flip(res, [DIM_DICT["y"]])
                res = torch.flip(res, [DIM_DICT["x"]])
            elif self.degrees == 270:
                res = res.permute([0, 2, 1, 3])
                res = torch.flip(res, [DIM_DICT["x"]])
            else: #0 degrees
                pass

            results.append(res)
        return tuple(results)




"""
All possible rotate/flip transforms are given by:
transforms.Compose([
                transforms.RandomChoice([RotateHorizontal(0),
                                        RotateHorizontal(90),
                                        RotateHorizontal(180),
                                        RotateHorizontal(270)]),
                transforms.RandomApply([FlipHorizontal("x")], p=0.5) ])


BUT: there it doesn't work when the x and y dims are different sizes

"""