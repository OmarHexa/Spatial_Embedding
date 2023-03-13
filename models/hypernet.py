import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchsummary import summary
from functools import wraps
from time import time
from torch.profiler import profile, record_function, ProfilerActivity

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
        return result
    return wrap

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob=0.3, groups =2,dilation=1):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilation, 0), bias=True,groups=groups, dilation=(dilation, 1))

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilation), bias=True,groups=groups, dilation=(1, dilation))

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            2*(dilation), 0), bias=True, groups=groups,dilation=(2*dilation, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 2*(dilation)), bias=True,groups=groups, dilation=(1, 2*dilation))

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

    
class EELAN(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        mid_channel = in_channel//2
        self.convleft = nn.Conv2d(in_channel, mid_channel, 1, 1)
        self.convright = nn.Conv2d(in_channel, mid_channel, 1, 1)
        self.nonbt1 = non_bottleneck_1d(mid_channel,dilation=1)
        self.nonbt2 = non_bottleneck_1d(mid_channel, dilation=4)
        self.nonbt3 = non_bottleneck_1d(mid_channel, dilation=1)
        self.nonbt4 = non_bottleneck_1d(mid_channel, dilation=4)
        
        self.agg = nn.Conv2d(2*in_channel, out_channel,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-03)

    # @timing
    def forward(self, input):
        #channel partialization
        x_left = self.convleft(input)
        x_right =self.convright(input)
        #computational block
        x1 = self.nonbt1(x_right)
        x1 = self.nonbt2(x1)
        x2 = self.nonbt3(x1)
        x2 = self.nonbt4(x2)
        
        c_ = x_left.shape[-1]//2
        #channel shuffle
        xg1 = torch.cat((x_left[:,:c_],x_right[:,:c_],x1[:,:c_],x2[:,:c_]),dim=1)
        xg2 = torch.cat((x_left[:,c_:],x_right[:,c_:],x1[:,c_:],x2[:,c_:]),dim=1)
        
        # aggregation
        x = self.agg(torch.cat((xg1,xg2),dim=1))
        x = self.bn2(x)
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
        output = torch.cat([self.conv(input), self.pool(input)], dim=1)
        output = self.bn(output)
        return F.relu(output)
    
class ChannelsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        mid_channel = in_channel//2
        self.conv1 = nn.Conv2d(in_channel, mid_channel,
                              (3, 3), stride=2, padding=1,groups=1, bias=True)
        self.soca = SOCA(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_channel,
                              (1, 1), stride=1, padding=0, bias=True)
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
    def __init__(self, in_channel,out_channel) -> None:
        super().__init__()
        self.compblock = EELAN(in_channel, out_channel=in_channel)
        self.soca = SOCA(in_channel)
        self.down = DownsamplerBlock(in_channel, out_channel)
    # @timing
    def forward(self, x):
        x = self.compblock(x)
        x = self.soca(x)
        x = self.down(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.up = UpsamplerBlock(in_channel, out_channel)
        self.soca = SOCA(out_channel)
        self.nonbt = non_bottleneck_1d(out_channel)
    # @timing
    def forward(self, input):
        x = self.up(input)
        x = self.soca(x)
        x = self.nonbt(x)
        # print("Decoder",input.shape)
        return x
        
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
        
     
 
# class HyperNet(nn.Module):
#     # use encoder to pass pretrained encoder
#     def __init__(self, in_channel, num_classes, feature=32):
#         super().__init__()
#         self.encoder = nn.ModuleList()
#         self.decoder = nn.ModuleList()
#         self.skip = nn.ModuleList()

#         self.downsample = ChannelsamplerBlock(in_channel,feature)# C x H --> 32 x H/2
#         in_channel = feature
        
#         for id in range(2,5,2):
#             self.encoder.append(EncoderBlock(in_channel, id*feature)) #32 x H/2 --> (64,128) x (H/4,H/8)
#             in_channel = id*feature 
            
#         self.decoder.append(DecoderBlock(in_channel,feature*2))
#         self.skip.append(Skip_connector(feature*4,feature*2))
#         self.decoder.append(DecoderBlock(feature*2,feature))
#         self.skip.append(Skip_connector(feature*2,feature))
        
            
#         self.output = UpsamplerBlock(feature,num_classes)
        
#     @timing  
#     def forward(self, input):
#         input = self.downsample(input)
#         encode1 = self.encoder[0](input)
#         decode = self.encoder[1](encode1)
#         decode = self.decoder[0](decode)
#         decode = self.skip[0](torch.cat((decode,encode1),dim=1))
#         decode = self.decoder[1](decode)
#         decode = self.skip[1](torch.cat((decode,input),dim=1))
#         decode = self.output(decode)

#         return torch.sigmoid(decode)


class HyperEncoder(nn.Module):
    def __init__(self, in_channel, feature=32):
        super().__init__()
        self.chsample = ChannelsamplerBlock(in_channel,feature)
        self.encoder = nn.ModuleList()
        in_channel = feature
        for id in range(2,5,2):
            self.encoder.append(EncoderBlock(in_channel, id*feature)) #32 x H/2 --> (64,128) x (H/4,H/8)
            in_channel = id*feature
        
    def forward(self, input):
        input = self.chsample(input)
        encode = self.encoder[0](input)
        encode = self.encoder[1](encode)
        return encode
class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
    
class HyperNet(nn.Module):
    def __init__(self, in_channel,num_classes):  # use encoder to pass pretrained encoder
        super().__init__()

        self.encoder = HyperEncoder(in_channel)
        self.decoder = Decoder(num_classes)
    @timing
    def forward(self, input, only_encode=False):
        output = self.encoder(input)  # predict=False by default
        return self.decoder.forward(output)

def run_model(model, data):
    outputs = model(data)
# taken from pytorch : https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
def ModelSize(model):
    param_size = sum([param.nelement()*param.element_size() for param in model.parameters()])
    buffer_size = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))

if __name__ == "__main__":
    import torch.utils.benchmark as benchmark
    # create some input data
    input = torch.randn(20, 164, 416, 416)
    model = HyperNet(164,5)
    model.eval()
    output = model(input)
    # print the shape of the output tensor
    ModelSize(model)
    print(output.shape)
    # summary(model,(116,416,416))
    # with profile(with_stack=False,activities=[ProfilerActivity.CPU]) as prof:
    #     with record_function("model_inference"):
    #         model(input)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # prof.export_chrome_trace("hypernet.json")
    
    
    

    # num_threads = torch.get_num_threads()
    # t = benchmark.Timer(
    #     stmt = 'run_model(model, data)',
    #     setup = 'from __main__ import run_model',
    #     globals={'model': model, 'data': input},
    #     num_threads=num_threads,
    #     label="Average Inference Duration",
    #     )
    # print(t.timeit(20))
