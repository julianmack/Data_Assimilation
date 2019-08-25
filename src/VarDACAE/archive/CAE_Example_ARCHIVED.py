"""Example 3d CAE - don't use this.
all AEs/CAEs should inherit from the BaseAE class"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class CAE_Example(nn.Module):



    def __init__(self, channels):
        super(BaselineCAE, self).__init__()

        C_in = 1
        C1, C2, C3, C4 = channels

        self.C4 = C4 #save this for .decode()

        # encoder layers
        self.conv1 = nn.Conv3d(C_in, C1, 3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm3d(C1)
        self.conv2 = nn.Conv3d(C1, C2, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm3d(C2)
        self.conv3 = nn.Conv3d(C2, C3, 3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm3d(C3)
        self.conv4 = nn.Conv3d(C3, C4, 3, stride=2, padding=1)
        self.norm4 = nn.BatchNorm3d(C4)
        self.fc = nn.Linear(C4 * 2 * 2, hidden_size)
        self.normfc = nn.BatchNorm1d(hidden_size)

        # decoder layers                                      #B, 8, 2, 2
        self.deconv1 = nn.ConvTranspose3d(C4, C3, 3, stride=2) #B, 16, 5
        self.normd1 = nn.BatchNorm3d(C3)
        self.deconv2 = nn.ConvTranspose3d(C3, C2, 3, stride=2, padding=1) #B, 32, 9
        self.normd2 = nn.BatchNorm3d(C2)
        self.deconv3 = nn.ConvTranspose3d(C2, C1, 3, stride=2, padding=1) #B, 16, 17
        self.normd3 =nn.BatchNorm3d(C1)
        self.deconv4 = nn.ConvTranspose3d(C1, C_in, 2, stride=2, padding=1) #B, 3, 32, 32
        self.normd4 = nn.BatchNorm3d(C_in)

        self.sigmoid = nn.Sigmoid()
        # relu
        self.relu = nn.ReLU(True)


    def encode(self, x):
        """
        TODO: Construct the encoder pipeline here. The encoder's
        output will be the laten space representation of x.
        """

        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))
        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1) # b, 8*2*2
        x = self.normfc(self.fc(x)) # b, h_dim
        return x

    def decode(self, z):
        """
        TODO: Construct the decoder pipeline here. The decoder should
        generate an output tensor with equal dimenssions to the
        encoder's input tensor.
        """
        z = z.view(z.size(0), self.C4, 2, 2)
        z = self.normd1(self.relu(self.deconv1(z)))
        z = self.normd2(self.relu(self.deconv2(z)) )
        z = self.normd3(self.relu(self.deconv3(z)) )
        z = self.normd4(self.sigmoid(self.deconv4(z)))

        return z
