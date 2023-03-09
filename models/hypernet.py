import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
        return result
    return wrap

class Conv2dpsuedo(nn.Module):
    def __init__(self, chann, dropprob, dilation=1):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilation, 0), bias=True,groups=chann, dilation=(dilation, 1))

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilation), bias=True,groups=chann, dilation=(1, dilation))

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            2*(dilation), 0), bias=True, groups=chann,dilation=(2*dilation, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 2*(dilation)), bias=True,groups=chann, dilation=(1, 2*dilation))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
    # @timing
    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        # output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        # print("Conv2dpsuedo",input.shape)
        return F.relu(output+input)  # +input = identity (residual connection)



class CSPCompBlock(nn.Module):
    def __init__(self, in_channel, resolution, drop, factor=1) -> None:
        super().__init__()
        dilation = [1, 1, 1]*factor
        self.convhw = Conv2dpsuedo(
            chann=in_channel, dropprob=drop, dilation=dilation[0])
        self.convch = Conv2dpsuedo(
            chann=resolution[1], dropprob=drop, dilation=dilation[1])
        self.convcw = Conv2dpsuedo(
            chann=resolution[0], dropprob=drop, dilation=dilation[2])
    # @timing
    def forward(self, input):
        # input b(0) x c(1) x h(2) x w(3)
        chw = self.convhw(input)
        # b(0) x c(1) x h(2) x w(3)
        wch = chw.permute(0, 3, 1, 2)
        # b(0) x w(1) x c(2) x h(3)
        wch = self.convch(wch)
        # b(0) x w(1) x c(2) x h(3)
        hcw = wch.permute(0, 3, 2, 1)
        # b(0) x h(1) x c(2) x w(3)
        hcw = self.convcw(hcw)
        # b(0) x h(1) x c(2) x w(3)
        output = torch.cat((input, chw, wch.permute(
            0, 2, 3, 1), hcw.permute(0, 2, 1, 3)), dim=1)
        # print("CONV3Dpsuedo")
        return output
    
class ELANCompBlock(nn.Module):
    def __init__(self, in_channel, resolution, drop, factor=1) -> None:
        super().__init__()
        dilation = [1, 1, 1]*factor
        self.convhw = Conv2dpsuedo(
            chann=in_channel, dropprob=drop, dilation=dilation[0])
        self.convch = Conv2dpsuedo(
            chann=resolution[1], dropprob=drop, dilation=dilation[1])
        self.convcw = Conv2dpsuedo(
            chann=resolution[0], dropprob=drop, dilation=dilation[2])
    # @timing
    def forward(self, input):
        # input b(0) x c(1) x h(2) x w(3)
        chw = self.convhw(input)
        # b(0) x c(1) x h(2) x w(3)
        wch = chw.permute(0, 3, 1, 2)
        # b(0) x w(1) x c(2) x h(3)
        wch = self.convch(wch)
        # b(0) x w(1) x c(2) x h(3)
        hcw = wch.permute(0, 3, 2, 1)
        # b(0) x h(1) x c(2) x w(3)
        hcw = self.convcw(hcw)
        # b(0) x h(1) x c(2) x w(3)
        
        output = hcw.permute(0, 2, 1, 3)
        # print("ELANCompBlock")
        return output

class CSPVoV3D(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, drop) -> None:
        super().__init__()
        self.block1 = CSPCompBlock(in_channel, resolution, drop)
        self.agg1 = nn.Conv2d(4*in_channel, in_channel,
                              kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1 = nn.BatchNorm2d(in_channel, eps=1e-03)
        self.block2 = CSPCompBlock(in_channel, resolution, drop, factor=2)
        self.agg2 = nn.Conv2d(4*in_channel, out_channel,
                              kernel_size=1, stride=1, padding=0, bias=True)
        # self.agg2 = nn.Conv2d(5*in_channel, out_channel,
        #                       kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-03)

    # TODO include partialization
    # @timing
    def forward(self, input):
        x = self.block1(input)
        x = self.agg1(x)
        # x = self.bn1(x)
        x = self.block2(x)
        # x = self.agg2(torch.cat((input, x), dim=1))
        x = self.agg2(x)
        x = self.bn2(x)
        # print("CSPVoV3D")
        return F.relu(x)
    
class ELAN3D(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, drop,stage=2) -> None:
        super().__init__()
        mid_channel = int(in_channel/2)
        self.convleft = nn.Conv2d(in_channel, mid_channel, 1, 1)
        self.convright = nn.Conv2d(in_channel, mid_channel, 1, 1)
        self.block1 = ELANCompBlock(mid_channel, resolution, drop)
        self.block2 = ELANCompBlock(mid_channel, resolution, drop, factor=2)
        self.agg = nn.Conv2d(2*in_channel, out_channel,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-03)

    # @timing
    def forward(self, input):
        x_left = self.convleft(input)
        x_right =self.convright(input)
        #computational block
        x1 = self.block1(x_right)
        x2 = self.block2(x1)
        # aggregation
        x = self.agg(torch.cat((x_left,x_right,x1,x2), dim=1))
        x = self.bn2(x)
        # print("ELAN3D")
        return F.relu(x)

class SOCA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Define the Covariance layer
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)
    # @timing
    def forward(self, x):
        # Reshape the input tensor
        batch_size, channels, height, width = x.shape
        # B x C x hw ----> B x hw x C
        x = x.view(batch_size, channels, height*width).transpose(1, 2)

        # Compute the channel-wise mean and covariance
        avg = torch.mean(x, dim=1, keepdim=True)  # B x 1 X C
        x_centered = x - avg  # B x hw x C
        cov = torch.matmul(x_centered.transpose(
            1, 2), x_centered) / (height*width - 1)  # B x C x C

        cov = torch.mean(cov, dim=2, keepdim=True)  # B x C X 1
        # Compute the channel attention weights
        attention = self.conv(cov)  # B x C x 1
        attention = torch.sigmoid(attention)  # B x C x 1

        # Apply the channel attention weights to the input tensor
        x_weighted = x.transpose(1, 2) * attention  # B x C x hw
        out = x_weighted.view(batch_size, channels,
                              height, width)  # B x C x H x W
        # print("SOCA",x.shape)
        return out


class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel-in_channel,
                              (3, 3), stride=2, padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    
class ChannelsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel,
                              (3, 3), stride=2, padding=1, bias=True)
        self.soca = SOCA(in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel,
                              (3, 3), stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)

    def forward(self, input):
        output = self.conv1(input)
        output = self.soca(output)
        output = self.conv2(output)
        output = self.bn(output)
        return F.relu(output)


class UpsamplerBlock (nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        # print("UPsampler")
        return F.relu(output)


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, drop=0.03) -> None:
        super().__init__()
        self.in_channel = in_channel
        # self.cspvov = CSPVoV3D(
        #     self.in_channel, out_channel=self.in_channel, resolution=resolution, drop=drop)
        self.cspvov = ELAN3D(
            self.in_channel, out_channel=self.in_channel, resolution=resolution, drop=drop)
        self.soca = SOCA(self.in_channel)
        self.down = DownsamplerBlock(self.in_channel, out_channel)
    # @timing
    def forward(self, x):
        x = self.cspvov(x)
        x = self.soca(x)
        x = self.down(x)
        # print("Encoder",x.shape)
        return x

class DecoderBlock(nn.Module):
    """
    First upscale the resolution and reduces to half of the input channel. input channel is double 
    of the output from the last layer due to skip connection concatenation. Applies channel 
    attention and then reduces the channel to the output channel by COSVoV net.
    """
    def __init__(self, in_channel, out_channel, resolution, drop=0.03) -> None:
        super().__init__()
        self.up = UpsamplerBlock(in_channel, out_channel)
        resolution = resolution*2
        self.soca = SOCA(out_channel)
        # self.cspvov = CSPVoV3D(
        #     out_channel, out_channel=out_channel, resolution=resolution, drop=drop)
        self.cspvov = ELAN3D(
            out_channel, out_channel=out_channel, resolution=resolution, drop=drop)
    # @timing
    def forward(self, input):
        x = self.up(input)
        x = self.soca(x)
        x = self.cspvov(x)
        # print("Decoder",input.shape)
        return x

class OutputBlock(nn.Module):
    def __init__(self,in_channel,out_channel,drop) -> None:
        super().__init__()
        self.up = UpsamplerBlock(in_channel,in_channel)
        self.soca = SOCA(in_channel)
        self.conv = nn.Conv2d(in_channel,out_channel,3,1,1)
        # self.convdw = Conv2dpsuedo(in_channel,drop)
        # self.convpw = nn.Conv2d(in_channel,out_channel,1,1,0)
        self.bn = nn.BatchNorm2d(out_channel,eps=1e-3)
    # @timing
    def forward(self,x):
        x = self.up(x)
        x = self.soca(x)
        x = self.conv(x)
        # x = self.convdw(x)
        # x = self.convpw(x)
        return self.bn(x)
        
class Skip_connector(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),
                              stride=(1,1))
        self.bn = nn.BatchNorm2d(out_channel,eps=1e-3)
    
    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)
        
        
 
class HyperNet(nn.Module):
    # use encoder to pass pretrained encoder
    def __init__(self, in_channel, num_classes, resolution, feature=32, drop=0.3):
        super().__init__()
        self.stage = 2
        resolution = torch.tensor(resolution)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip = nn.ModuleList()

        self.downsample = ChannelsamplerBlock(in_channel,feature)# C x H --> 32 x H/2
        resolution = torch.div(resolution, 2, rounding_mode='trunc')
        in_channel = feature
        
        for id in range(2,(2*self.stage)+1,2):
            self.encoder.append(EncoderBlock(in_channel, id*feature, resolution)) #32 x H/2 --> (64,128) x (H/4,H/8)
            in_channel = id*feature 
            resolution = torch.div(resolution, 2, rounding_mode='trunc')
        # in_channel =128
        for id in reversed(range(1,self.stage+1)):
            self.decoder.append(DecoderBlock(in_channel,feature*id, resolution))
            in_channel = id*feature
            resolution = resolution*2
            self.skip.append(Skip_connector(2*in_channel,in_channel))
            
        self.output = OutputBlock(feature,num_classes,drop)
        
#     @timing  
    def forward(self, input):
        input = self.downsample(input)
        encode1 = self.encoder[0](input)
        decode = self.encoder[1](encode1)
        decode = self.decoder[0](decode)
        decode = self.skip[0](torch.cat((decode,encode1),dim=1))
        decode = self.decoder[1](decode)
        decode = self.skip[1](torch.cat((decode,input),dim=1))
        decode = self.output(decode)

        return torch.sigmoid(decode)


if __name__ == "__main__":
    import numpy as np

    # create some input data
    input = torch.randn(1, 116, 416, 416)
    model = HyperNet(116, 5, (416, 416))
    model.eval()
    output = model(input)
    # print the shape of the output tensor
    print(output.shape)
    summary(model,(116,416,416))