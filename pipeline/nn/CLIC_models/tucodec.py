"""Implementation (in 3D) of network in:
 http://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC 2019/Zhou_End-to-end_Optimized_Image_Compression_with_Attention_Mechanism_CVPRW_2019_paper.pdf

 Winner of CLIC 2019

 """
from torch import nn
from pipeline.nn.pytorch_gdn import GDN
from pipeline.nn.RNAB import RNAB
from pipeline.ML_utils import get_device
from pipeline.AEs.AE_Base import BaseAE

class Tucodec(BaseAE):
    def __init__(self, activation_constructor, Block):
        super(Tucodec, self).__init__()
        Cstd = 192

        print("INIT")
        self.layers_encode = nn.Sequential(TucodecEncode(activation_constructor, Block))
        self.layers_decode = nn.Sequential()
        print(self.layers_encode)


class TucodecEncode(nn.Module):
    def __init__(self, activation_constructor, Block):
        super(TucodecEncode, self).__init__()
        Cstd = 192
        device = get_device()
        encode = True

        #main trunk first
        self.conv1 = nn.Conv3d(1, Cstd, kernel_size=3, stride=2)
        self.gdn2 = GDN(Cstd, device, not encode)
        self.conv3 = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=2)
        self.gdn4 = GDN(Cstd, device, not encode)
        self.rnab5 = RNAB(encode, activation_constructor, Cstd, Block)
        self.conv6 = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=2)
        self.gdn7 = GDN(Cstd, device, not encode)
        self.conv8 = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=2)
        self.rnab9 = RNAB(encode, activation_constructor, Cstd, Block)

        #multi-res path
        self.convA = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=8)
        self.convB = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=4)
        self.convC = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=2)

        #final conv
        self.conv10 = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=2)

    def forward(self, x):
        h, xa, xb, xc = trunk(x)

        ha = self.convA(xa)
        hb = self.convB(xb)
        hc = self.convC(xc)

        inp = torch.cat(h, ha, hb, hc)
        print(inp.shape)
        h = self.conv10(inp)
        return h

    def trunk(self, x):
        x = self.conv1(x)
        x = self.gdn2(x)
        xa = x
        x = self.conv3(x)
        x = self.gdn4(x)
        xb = x
        x = self.rnab5(x)
        x = self.conv6(x)
        x = self.gdn7(x)
        xc = x
        x = self.conv8(x)
        x = self.rnab9(x)
        return x, xa, xb, xc
