import torch
import torch.nn as nn
import torch.nn.functional as F
##########################################################################
## dw_conv
class dw_conv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel=3,stride=1,padding=1,scale=2,bias=True,groups=1):
        super().__init__()
        dw_channel = out_ch * scale
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=dw_channel, kernel_size=1, stride=1, padding=0,groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding, groups=out_ch,
                               bias=bias)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x
##########################################################################
## Basic modules
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch=3,kernel=3,stride=1,padding=1,groups=1,norm=False,bias=False):
        super().__init__()
        self.conv = nn.Sequential(
                dw_conv(in_ch,out_ch,kernel=kernel,stride=stride,padding=padding,bias=bias,groups=groups),
                LayerNorm(in_ch) if norm else nn.Identity(),
                nn.PReLU(),
                dw_conv(out_ch,out_ch,kernel=kernel,stride=stride,padding=padding,bias=bias,groups=groups),
                LayerNorm(out_ch) if norm else nn.Identity(),
                nn.PReLU(),
            )

    def forward(self, x):
        return self.conv(x)
################################MLP##############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.PReLU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   

##################################################
####LayerNorm      
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x