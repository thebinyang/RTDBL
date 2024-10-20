import torch
import torch.nn as nn
import random
import numpy as np
from gabor import *
from edgeenhance import *
random.seed(20)
class UNet(nn.Module):
    def __init__(self,ceng):
        super(UNet, self).__init__()
        self.ceng=ceng
        # 定义编码器部分
        # self.encoder1_x1 = self.conv_block(3,100)
        # self.encoder2_x1 = self.conv_block(103, 300)
        # self.encoder3_x1 = self.conv_block(403, 500)
        # # self.encoder6_x1 = self.conv_block(6, 1)
        # #
        # self.encoder1_x2 = self.conv_block(3, 100)
        # self.encoder2_x2 = self.conv_block(103, 300)
        # self.encoder3_x2 = self.conv_block(403, 500)

        self.encoder1_x1 = self.conv_block(3,1)
        self.encoder2_x1 = self.conv_block(4, 1)
        self.encoder3_x1 = self.conv_block(5, 1)
        # self.encoder6_x1 = self.conv_block(6, 1)
        #
        self.encoder1_x2 = self.conv_block(3, 1)
        self.encoder2_x2 = self.conv_block(4, 1)
        self.encoder3_x2 = self.conv_block(5, 1)
        self.linear1 = torch.nn.Linear(6, 128)
        self.linear2 = torch.nn.Linear(128, 2)
        # self.encoder4_x2 = self.conv_block(6, 1)
        # self.encoder5_x2 = self.conv_block(7, 1)
        # self.encoder6_x2 = self.conv_block(8, 1)
    def conv_block1(self, in_channels, out_channels):
        c = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # torch.manual_seed(5)
        c_data = torch.randn(out_channels, in_channels, 1, 1)
        c.weight.data = torch.FloatTensor(c_data)
        # torch.manual_seed(6)
        c_bias = torch.randn(out_channels)
        c.bias.data = torch.FloatTensor(c_bias)

        return nn.Sequential(
            c,
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):

        a=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # torch.manual_seed(1)
        # a_data=torch.randn(out_channels, in_channels, 3, 3)
        # a.weight.data= torch.FloatTensor(a_data)
        # torch.manual_seed(2)
        # a_bias=torch.randn(out_channels)
        # a.bias.data=torch.FloatTensor(a_bias)
        # b=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # torch.manual_seed(3)
        # b_data = torch.randn(out_channels, out_channels, 3, 3)
        # b.weight.data = torch.FloatTensor(b_data)
        # torch.manual_seed(4)
        # b_bias = torch.randn(out_channels)
        # b.bias.data = torch.FloatTensor(b_bias)
        return nn.Sequential(
            a,
            # nn.Sigmoid(),
            # b,
            nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, name,x,x2):

        if name=='sar':
            x = torch.cat((x, x, x), dim=1)
        elif name=='sarca':
            # x = torch.cat((x, x, x,x[:,0:2,:,:]), dim=1)
            x = x[:, 1:4, :, :]
        elif name=='sarca2':
            x = torch.cat((x, x, x,x[:,0:2,:,:]), dim=1)
        random.seed(10)
        # for i in range(self.ceng):
        #     encoder=self.conv_block(x.shape[1], 1)
        #     enc = encoder(x)
        #     x = torch.cat((x, enc), dim=1)

        enc=self.encoder1_x1(x)
        x = torch.cat((x, enc), dim=1)
        enc = self.encoder2_x1(x)
        x = torch.cat((x, enc), dim=1)
        enc = self.encoder3_x1(x)
        x = torch.cat((x, enc), dim=1)

        enc = self.encoder1_x2(x2)
        x2 = torch.cat((x2, enc), dim=1)
        enc = self.encoder2_x2(x2)
        x2 = torch.cat((x2, enc), dim=1)
        enc = self.encoder3_x2(x2)
        x2 = torch.cat((x2, enc), dim=1)

        x3=x-x2
        output = x3
        out=output.permute(0,2,3,1)
        b,h,w,c=out.size()
        outfeature=out.reshape(h* w,c)
        outfeature=self.linear1(outfeature)
        outfeature = self.linear2(outfeature)

        den_x1 = x.permute(0, 2, 3, 1)
        b, h, w, c = den_x1.size()
        den_x1=den_x1.reshape(h, w,c)

        den_x2 = x2.permute(0, 2, 3, 1)
        b, h, w, c = den_x2.size()
        den_x2 = den_x2.reshape(h, w, c)

        return outfeature, den_x1.detach().numpy(),den_x2.detach().numpy()





