"""Implementation (in 3D) of network in:
 http://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC 2019/Zhou_End-to-end_Optimized_Image_Compression_with_Attention_Mechanism_CVPRW_2019_paper.pdf

 Winner of CLIC 2019

 """
import torch
from torch import nn
from pipeline.nn.pytorch_gdn import GDN
from pipeline.nn.RAB import RAB
from pipeline.ML_utils import get_device
from pipeline.AEs.AE_Base import BaseAE
from pipeline.nn.explore.empty import Empty


class TucodecEncode(nn.Module):
    def __init__(self, activation_constructor, Block, Cstd, sigmoid=False):
        super(TucodecEncode, self).__init__()

        device = get_device()
        encode = True

        #downsamples and upsamples
        downsample1 = DownUp.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp.upsample1(activation_constructor, Cstd, Cstd)
        downsample2 = DownUp.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp.upsample2(activation_constructor, Cstd, Cstd)

        #main trunk first
        self.conv1 = nn.Conv3d(1, Cstd, kernel_size=(3,3, 2), stride=2, padding=(1, 1, 0))
        self.gdn2 = GDN(Cstd, device, not encode)
        self.conv3 = nn.Conv3d(Cstd, Cstd, kernel_size=(2, 3, 2), stride=2, padding=(0, 1, 0))
        self.gdn4 = GDN(Cstd, device, not encode)
        self.rnab5 = RAB(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.conv6 = nn.Conv3d(Cstd, Cstd, kernel_size=(3,4,2), stride=2)
        self.gdn7 = GDN(Cstd, device, not encode)
        self.conv8 = nn.Conv3d(Cstd, Cstd, kernel_size=(3, 2, 2), stride=2, padding=(1,1,0))
        self.rnab9 = RAB(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)

        #multi-res path
        self.convA = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=8)
        self.convB = nn.Conv3d(Cstd, Cstd, kernel_size=3, stride=4, padding=(0, 1, 0))
        self.convC = nn.Conv3d(Cstd, Cstd, kernel_size=(3, 2, 3), stride=2, padding=1)
        
        #final conv
        self.conv10 = nn.Conv3d(4 * Cstd, Cstd, kernel_size=(2,2,2), stride=2)

    def forward(self, x):
        h, xa, xb, xc = self.trunk(x)

        ha = self.convA(xa)
        hb = self.convB(xb)
        hc = self.convC(xc)

        inp = torch.cat([h, ha, hb, hc], dim=1) #concat on channel
        h = self.conv10(inp)
        # h = self.act11(h)
        # h = self.conv12(h) #to give same latent dim as baseline model
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

class TucodecDecode(nn.Module):
    def __init__(self, activation_constructor, Block, Cstd, sigmoid=False):
        super(TucodecDecode, self).__init__()

        device = get_device()
        encode = False

        #downsamples and upsamples
        downsample2 = DownUp.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp.upsample2(activation_constructor, Cstd, Cstd)
        downsample1 = DownUp.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp.upsample1(activation_constructor, Cstd, Cstd)

        #Keep numbering from Encoder

        self.conv10 = nn.ConvTranspose3d( Cstd, Cstd, kernel_size=(2,2,2), stride=2)

        self.rb10a = Block(encode, activation_constructor, Cstd,)
        self.rb10b = Block(encode, activation_constructor, Cstd,)

        self.rnab9 = RAB(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)
        self.conv8 = nn.ConvTranspose3d(Cstd, Cstd, kernel_size=(3, 2, 2), stride=2, padding=(1,1,0))



        self.gdn7 = GDN(Cstd, device, encode)
        self.conv6 = nn.ConvTranspose3d(Cstd, Cstd, kernel_size=(3,4,2), stride=2,)
        self.rnab5 = RAB(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.gdn4 = GDN(Cstd, device, encode)

        self.conv3 = nn.ConvTranspose3d(Cstd, Cstd, kernel_size=(2, 3, 2), stride=2, padding=(0, 1, 0))
        self.gdn2 = GDN(Cstd, device, encode)
        self.conv1 = nn.ConvTranspose3d(Cstd, 1, kernel_size=(3,3, 2), stride=2, padding=(1, 1, 0))



    def forward(self, x):

        x = self.conv10(x)
        x = self.rb10a(x)
        x = self.rb10b(x)

        x = self.rnab9 (x)
        x = self.conv8 (x)


        x = self.gdn7(x)
        x = self.conv6(x)
        x = self.rnab5(x)
        x = self.gdn4 (x)

        x = self.conv3(x)
        x = self.gdn2(x)
        x = self.conv1(x)
        return x

class DownUp:
    @staticmethod
    def downsample1(activation_constructor, Cin, channel_small):
        """First RAB downsample"""
        conv1 = nn.Conv3d(Cin, Cin, kernel_size=(3, 2, 2), stride=(2,2,2))
        conv2 = nn.Conv3d(Cin, Cin, kernel_size=(3, 3, 2), stride=(2,2,2), padding=(0, 0, 1))
        conv3 = nn.Conv3d(Cin, Cin, kernel_size=(3, 3, 3), stride=(2,2,1), padding=(0, 0, 0))
        return nn.Sequential(conv1, activation_constructor(Cin, False), #Empty("d", 1),
                            conv2, activation_constructor(Cin, False), #Empty("d", 2),
                            conv3, )#Empty("d", 3),)
    @staticmethod
    def upsample1(activation_constructor, Cin, channel_small):
        "First RAB upsample"
        conv1 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 3, 3), stride=(2,2,1), padding=(0, 0, 0))
        conv2 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 3, 2), stride=(2,2,2), padding=(0, 0, 1))
        conv3 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 2, 2), stride=(2,2,2))
        return nn.Sequential(conv1, activation_constructor(Cin, False), #Empty("u", 1),
                            conv2, activation_constructor(Cin, False), #Empty("u", 2),
                            conv3, ) #Empty("u", 3))

    @staticmethod
    def downsample2(activation_constructor, Cin, channel_small):
        """Second RAB downsample"""
        conv1 = nn.Conv3d(Cin, Cin, kernel_size=(2, 2, 2), stride=1,)
        conv2 = nn.Conv3d(Cin, Cin, kernel_size=(3, 3, 1), stride=(2,2,1), padding=0)
        conv3 = nn.Conv3d(Cin, Cin, kernel_size=(2, 2, 1), stride=1, padding=0)
        return nn.Sequential(conv1, activation_constructor(Cin, False),
                            conv2, activation_constructor(Cin, False),
                            conv3, )
    @staticmethod
    def upsample2(activation_constructor, Cin, channel_small):
        """Second RAB upsample"""
        conv1 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(2, 2, 1), stride=1, padding=0)
        conv2 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 3, 1), stride=(2,2,1), padding=0)
        conv3 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(2, 2, 2), stride=1,)
        return nn.Sequential(conv1, activation_constructor(Cin, False),
                            conv2, activation_constructor(Cin, False),
                            conv3, )
