import torch
import torch.nn.functional as F
from torch import nn
from scipy.io import savemat
from torchvision import models

LOSS_TP = nn.L1Loss()

EPS = 1e-10

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class SFE(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, int(n_feats/2))
        self.conv21 = conv3x3(int(n_feats/2), n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats, int(n_feats/2))

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        n_feats1 = n_feats
        self.conv12 = conv1x1(n_feats1, n_feats1)
        self.conv13 = conv1x1(n_feats1, n_feats1)

        n_feats2 = int(n_feats/2)
        self.conv21 = conv3x3(n_feats2, n_feats2, 2)
        self.conv23 = conv1x1(n_feats2, n_feats2)

        n_feats3 = int(n_feats/4)
        self.conv31_1 = conv3x3(n_feats3, n_feats3, 2)
        self.conv31_2 = conv3x3(n_feats3, n_feats3, 2)
        self.conv32 = conv3x3(n_feats3, n_feats3, 2)

        self.conv_merge1 = conv3x3(n_feats1*3, n_feats1)
        self.conv_merge2 = conv3x3(n_feats2*3, n_feats2)
        self.conv_merge3 = conv3x3(n_feats3*3, n_feats3)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, int(n_feats/4))
        self.conv23 = conv1x1(int(n_feats/2), int(n_feats/4))
        self.conv_merge = conv3x3(3*int(n_feats/4), out_channels)
        self.conv_tail1 = conv3x3(out_channels, out_channels)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        #x = self.conv_tail2(x)
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, rgb_range=1):
        super(Vgg19, self).__init__()
        
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        for x in range(30):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, X):
        h = self.sub_mean(X)
        h_relu5_1 = self.slice1(h)
        return h_relu5_1

class VGG_LFE(torch.nn.Module):
    def __init__(self, in_channels, requires_grad=True, rgb_range=1):
        super(VGG_LFE, self).__init__()
        
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        #Initial convolutional layer to form RGB image
        self.conv_RGB = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        # Initial convolutional layer to make the RGB image...
        x = self.conv_RGB(x)

        # Extracting VGG Features...
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3


# This function implements the learnable spectral feature extractor (abreviated as LSFE)
# Input:    Hyperspectral or PAN image
# Outputs:  out1 = features at original resolution, out2 = features at original resolution/2, out3 = features at original resolution/4
class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        
        #First level convolutions
        self.conv_64_1  = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(64)
        self.conv_64_2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_64_2    = nn.BatchNorm2d(64)
        
        #Second level convolutions
        self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1   = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_2   = nn.BatchNorm2d(128)
        
        #Third level convolutions
        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_1   = nn.BatchNorm2d(256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_2   = nn.BatchNorm2d(256)
        
        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out1    = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1    = self.bn_64_2(self.conv_64_2(out1))

        #Second level outputs
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2    = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2    = self.bn_128_2(self.conv_128_2(out2))

        #Third level outputs
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3    = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3    = self.bn_256_2(self.conv_256_2(out3))

        return out1, out2, out3


class LFE_lvx(nn.Module):
    def __init__(self, in_channels, n_feates, level):
        super(LFE_lvx, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        self.level = level
        lv1_c = int(n_feates)
        lv2_c = int(n_feates/2)
        lv3_c = int(n_feates/4)

        #First level convolutions
        self.conv_64_1  = nn.Conv2d(in_channels=self.in_channels, out_channels=lv3_c, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(lv3_c)
        self.conv_64_2  = nn.Conv2d(in_channels=lv3_c, out_channels=lv3_c, kernel_size=3, padding=1)
        self.bn_64_2    = nn.BatchNorm2d(lv3_c)
        
        #Second level convolutions
        if self.level == 1 or self.level==2:
            self.conv_128_1 = nn.Conv2d(in_channels=lv3_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_1   = nn.BatchNorm2d(lv2_c)
            self.conv_128_2 = nn.Conv2d(in_channels=lv2_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_2   = nn.BatchNorm2d(lv2_c)
        
        #Third level convolutions
        if  self.level == 1:
            self.conv_256_1 = nn.Conv2d(in_channels=lv2_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_1   = nn.BatchNorm2d(lv1_c)
            self.conv_256_2 = nn.Conv2d(in_channels=lv1_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_2   = nn.BatchNorm2d(lv1_c)
        
        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out    = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out    = self.bn_64_2(self.conv_64_2(out))

        #Second level outputs
        if self.level == 1 or self.level==2:
            out    = self.MaxPool2x2(self.LeakyReLU(out))
            out    = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out)))
            out    = self.bn_128_2(self.conv_128_2(out))

        #Third level outputs
        if  self.level == 1:
            out     = self.MaxPool2x2(self.LeakyReLU(out))
            out     = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out)))
            out     = self.bn_256_2(self.conv_256_2(out))

        return out

#This function implements the multi-head attention
class NoAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self):
        super().__init__()

    def forward(self, v, k, q, mask=None):
        output = v
        return output

class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)


        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)

        #Reshape output to original format
        output  = output.view(b, c, h, w)
        return output

# HyperTransformerResBlock Implementation
class HyperTransformerResBlock(nn.Module):
    ''' Hyperspectral Transformer Residual Block '''

    def __init__(self, HSI_in_c, n_feates, lv, temperature):
        super().__init__()
        self.HSI_in_c = HSI_in_c  #Number of input channels = Number of output channels
        self.lv = lv              #Spatial level 1-> hxw, 2-> 2hx2w, 3-> 3hx3w
        if lv == 1:
            out_channels = int(n_feates)
        elif lv ==2:
            out_channels = int(n_feates/2)
        elif lv ==3:
            out_channels = int(n_feates/4)
        
        #Learnable feature extractors (FE-PAN & FE-HSI)
        self.LFE_HSI    = LFE_lvx(in_channels=self.HSI_in_c, n_feates = n_feates, level=lv)
        self.LFE_PAN    = LFE_lvx(in_channels=1,  n_feates = n_feates, level=lv)

        #Attention
        self.DotProductAttention = ScaledDotProductAttentionOnly(temperature=temperature)

        #Texture & Spectral Mixing
        self.TSmix = nn.Conv2d(in_channels=int(2*out_channels), out_channels=out_channels, kernel_size=3, padding=1)
       
    def forward(self, F, X_PAN, PAN_UD, X_MS_UP):
        # Obtaining Values, Keys, and Queries
        V = self.LFE_PAN(X_PAN)
        K = self.LFE_PAN(PAN_UD)
        Q = self.LFE_HSI(X_MS_UP)

        # Obtaining T (Transfered HR Features)
        T  = self.DotProductAttention(V, K, Q)

        #Concatenating F and T
        FT = torch.cat((F, T), dim=1)

        #Texture spectral mixing
        res = self.TSmix(FT)

        #Output
        output = F + res
        return output

#This function implements the multi-head attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module for Hyperspectral Pansharpening (Image Fusion) '''

    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()
        #Parameters
        self.n_head         = n_head        #No of heads
        self.in_pixels      = in_pixels     #No of pixels in the input image
        self.linear_dim     = linear_dim    #Dim of linear-layer (outputs)

        #Linear layers

        self.w_qs   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for queries
        self.w_ks   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for keys
        self.w_vs   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for values
        self.fc     = nn.Linear(n_head * linear_dim, in_pixels, bias=False) #Final fully connected layer

        #Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

        #Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, v, k, q, mask=None):
        # Reshaping matrixes to 2D
        # q = b, c_q, h*w
        # k = b, c_k, h*w
        # v = b, c_v, h*w
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head          = self.n_head
        linear_dim      = self.linear_dim

        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        #Save V
        output = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)
        
        
        output = output + v_attn
        #output  = v_attn

        #Reshape output to original image format
        output = output.view(b, c, h, w)

        #We can consider batch-normalization here,,,
        #Will complete it later
        output = self.OutBN(output)
        return output

#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
# Experimenting with soft attention
class HyperTransformer(nn.Module):
    def __init__(self, config):
        super(HyperTransformer, self).__init__()
        # Settings
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]
        self.config         = config

        # Parameter setup
        self.num_res_blocks = [16, 4, 4, 4, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        # FE-PAN & FE-HSI
        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        
        # Dimention of each Scaled-Dot-Product-Attention module
        lv1_dim      = config[config["train_dataset"]]["LR_size"]**2
        lv2_dim      = (2*config[config["train_dataset"]]["LR_size"])**2
        lv3_dim      = (4*config[config["train_dataset"]]["LR_size"])**2

        # Number of Heads in Multi-Head Attention Module
        n_head          = config["N_modules"]
        
        # Setting up Multi-Head Attention or Single-Head Attention
        if n_head == 0:
            # No Attention #
            # JUst passing the HR features from PAN image (Values) #
            self.TS_lv3     = NoAttention()
            self.TS_lv2     = NoAttention()
            self.TS_lv1     = NoAttention()
        elif n_head == 1:
            ### Scaled Dot Product Attention ###
            self.TS_lv3     = ScaledDotProductAttentionOnly(temperature=lv1_dim)
            self.TS_lv2     = ScaledDotProductAttentionOnly(temperature=lv2_dim)
            self.TS_lv1     = ScaledDotProductAttentionOnly(temperature=lv3_dim)
        else:   
            ### Multi-Head Attention ###
            lv1_pixels      = config[config["train_dataset"]]["LR_size"]**2
            lv2_pixels      = (2*config[config["train_dataset"]]["LR_size"])**2
            lv3_pixels      = (4*config[config["train_dataset"]]["LR_size"])**2
            self.TS_lv3     = MultiHeadAttention(   n_head= int(n_head), 
                                                    in_pixels = int(lv1_pixels), 
                                                    linear_dim = int(config[config["train_dataset"]]["LR_size"]), 
                                                    num_features = self.n_feats)
            self.TS_lv2     = MultiHeadAttention(   n_head= int(n_head), 
                                                    in_pixels= int(lv2_pixels), 
                                                    linear_dim= int(config[config["train_dataset"]]["LR_size"]), 
                                                    num_features=int(self.n_feats/2))
            self.TS_lv1     = MultiHeadAttention(   n_head= int(n_head), 
                                                    in_pixels = int(lv3_pixels), 
                                                    linear_dim = int(config[config["train_dataset"]]["LR_size"]), 
                                                    num_features=int(self.n_feats/4))
        
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        if config[config["train_dataset"]]["feature_sum"]:
            self.conv11_headSUM    = conv3x3(self.n_feats, self.n_feats)
        else:
            self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)

        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)
        #Residial blocks
        self.RB11           = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                res_scale=self.res_scale))
        self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

        ###############
        ### stage22 ###
        ###############
        if config[config["train_dataset"]]["feature_sum"]:
            self.conv22_headSUM = conv3x3(int(self.n_feats/2), int(self.n_feats/2))
        else:
            self.conv22_head = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)
        #Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
        self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

        ###############
        ### stage33 ###
        ###############
        if config[config["train_dataset"]]["feature_sum"]:
             self.conv33_headSUM   = conv3x3(int(self.n_feats/4), int(self.n_feats/4))
        else:
            self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
        self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

        ##############
        ### FINAL ####
        ##############
        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)

        ###############
        # Batch Norm ##
        ###############
        self.BN_x11 = nn.BatchNorm2d(self.n_feats)
        self.BN_x22 = nn.BatchNorm2d(int(self.n_feats/2))
        self.BN_x33 = nn.BatchNorm2d(int(self.n_feats/4))

        ######################
        # MUlti-Scale-Output #
        ######################
        self.up_conv13 = nn.ConvTranspose2d(in_channels=self.n_feats, out_channels=self.in_channels, kernel_size=3, stride=4, output_padding=1)
        self.up_conv23 = nn.ConvTranspose2d(in_channels=int(self.n_feats/2), out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        ###########################
        # Transfer Periferal Loss #
        ###########################
        self.VGG_LFE_HSI    = VGG_LFE(in_channels=self.in_channels, requires_grad=False)
        self.VGG_LFE_PAN    = VGG_LFE(in_channels=1, requires_grad=False)

    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        T_lv3  = self.TS_lv3(V_lv3, K_lv3, Q_lv3)
        T_lv2  = self.TS_lv2(V_lv2, K_lv2, Q_lv2)
        T_lv1  = self.TS_lv1(V_lv1, K_lv1, Q_lv1)

        #Save feature maps for illustration purpose
        # feature_dic={}
        # feature_dic.update({"V": V_lv3.detach().cpu().numpy()})
        # feature_dic.update({"K": K_lv3.detach().cpu().numpy()})
        # feature_dic.update({"Q": Q_lv3.detach().cpu().numpy()})
        # feature_dic.update({"T": T_lv3.detach().cpu().numpy()})
        # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/soft_attention/multi_head_no_skip_lv3.mat", feature_dic)
        # exit()

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        if self.config[self.config["train_dataset"]]["feature_sum"]:
            x11_res = x11_res + T_lv3
            x11_res = self.conv11_headSUM(x11_res) #F.relu(self.conv11_head(x11_res))
        else:
            x11_res = torch.cat((self.BN_x11(x11_res), T_lv3), dim=1)
            x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11     = x11 + x11_res
        #Residial learning
        x11_res = x11
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        if self.config[self.config["train_dataset"]]["feature_sum"]:
            x22_res = x22_res + T_lv2
            x22_res = self.conv22_headSUM(x22_res) #F.relu(self.conv22_head(x22_res))
        else:
            x22_res = torch.cat((self.BN_x22(x22_res), T_lv2), dim=1)
            x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22     = x22 + x22_res
        #Residial learning
        x22_res = x22
        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22 + x22_res

        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        if self.config[self.config["train_dataset"]]["feature_sum"]:
            x33_res = x33_res + T_lv1
            x33_res = self.conv33_headSUM(x33_res) #F.relu(self.conv33_head(x33_res))
        else:
            x33_res = torch.cat((self.BN_x33(x33_res), T_lv1), dim=1)
            x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33     = x33 + x33_res
        #Residual learning
        x33_res = x33
        for i in range(self.num_res_blocks[3]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33 + x33_res

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        xF      = self.final_conv(xF)
        xF_res  = xF

        #Final resblocks
        for i in range(self.num_res_blocks[4]):
            xF_res = self.RBF[i](xF_res)
        xF_res  = self.convF_tail(xF_res)
        x       = xF + xF_res

        #####################################
        #      Transfer Periferal Loss      #
        #####################################
        #v_vgg_lv1, v_vgg_lv2, v_vgg_lv3 = self.VGG_LFE_HSI(X_MS_UP)
        #q_vgg_lv1, q_vgg_lv2, q_vgg_lv3 = self.VGG_LFE_PAN(X_PAN)
        #loss_tp = LOSS_TP(V_lv1, v_vgg_lv1) + LOSS_TP(V_lv2, v_vgg_lv2) + LOSS_TP(V_lv3, v_vgg_lv3)
        #loss_tp = loss_tp + LOSS_TP(Q_lv1, q_vgg_lv1) + LOSS_TP(Q_lv2, q_vgg_lv2) + LOSS_TP(Q_lv3, q_vgg_lv3)

        Phi_lv1, Phi_lv2, Phi_lv3   = self.LFE_HSI(x.detach())
        Phi_T_lv3  = self.TS_lv3(V_lv3, K_lv3, Phi_lv3)
        Phi_T_lv2  = self.TS_lv2(V_lv2, K_lv2, Phi_lv2)
        Phi_T_lv1  = self.TS_lv1(V_lv1, K_lv1, Phi_lv1)
        loss_tp                                 = LOSS_TP(Phi_T_lv1, T_lv1)+LOSS_TP(Phi_T_lv2, T_lv2)+LOSS_TP(Phi_T_lv3, T_lv3)

        #####################################
        #       Output                      #
        #####################################
        x13 = self.up_conv13(x11)
        x23 = self.up_conv23(x22)
        output = {  "pred": x,
                    "x13": x13,
                    "x23": x23,
                    "tp_loss": loss_tp}
        return output


######################################################
# Hyperspectral Transformer (HSIT) with ResBlocks ####
######################################################
# Experimenting with soft attention
class HyperResTransformer(nn.Module):
    def __init__(self, config):
        super(HyperResTransformer, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]
        self.config         = config

        #Parameter setup
        self.num_res_blocks = [16, config["N_modules"], config["N_modules"], config["N_modules"], 4]
        self.n_feats        = 256
        self.res_scale      = 1
        
        #Dimention of each Scaled-Dot-Product-Attention module
        temp_lv1      = config[config["train_dataset"]]["LR_size"]
        temp_lv2      = (2*config[config["train_dataset"]]["LR_size"])
        temp_lv3      = (4*config[config["train_dataset"]]["LR_size"])

        #Shallow Feature extraction (Backbone)
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)
        #Residial blocks
        self.RB11           = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(HyperTransformerResBlock(  HSI_in_c=config[config["train_dataset"]]["spectral_bands"], 
                                                        n_feates=self.n_feats, lv=1, 
                                                        temperature=temp_lv1))

        ###############
        ### stage22 ###
        ###############
        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)
        #Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(HyperTransformerResBlock(  HSI_in_c=config[config["train_dataset"]]["spectral_bands"], 
                                                        n_feates=self.n_feats, lv=2, 
                                                        temperature=temp_lv2))

        ###############
        ### stage33 ###
        ###############
        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(HyperTransformerResBlock(  HSI_in_c=config[config["train_dataset"]]["spectral_bands"], 
                                                        n_feates=self.n_feats, lv=3, 
                                                        temperature=temp_lv3))

        ##############
        ### FINAL ####
        ##############
        #self.final_conv    = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.conv_final     = nn.Conv2d(in_channels=int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)


        ######################
        # MUlti-Scale-Output #
        ######################
        self.up_conv13 = nn.ConvTranspose2d(in_channels=self.n_feats, out_channels=self.in_channels, kernel_size=3, stride=4, output_padding=1)
        self.up_conv23 = nn.ConvTranspose2d(in_channels=int(self.n_feats/2), out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bicubic')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bicubic')

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        
        #Residial learning
        for i in range(self.num_res_blocks[1]):
            x11 = self.RB11[i](x11, X_PAN, PAN_UD, X_MS_UP)

        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        
        #Residial learning
        for i in range(self.num_res_blocks[2]):
            x22 = self.RB22[i](x22, X_PAN, PAN_UD, X_MS_UP)

        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        
        #Residial learning
        for i in range(self.num_res_blocks[3]):
            x33 = self.RB33[i](x33, X_PAN, PAN_UD, X_MS_UP)

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        xF      = self.conv_final(x33)
        xF_res  = xF

        #Final resblocks
        for i in range(self.num_res_blocks[4]):
            xF_res = self.RBF[i](xF_res)
        xF_res  = self.convF_tail(xF_res)
        x       = xF + xF_res

        #####################################
        #       Output                      #
        #####################################
        x13 = self.up_conv13(x11)
        x23 = self.up_conv23(x22)
        output = {  "pred": x,
                    "x13": x13,
                    "x23": x23}
        return output


#######################################
# Hyperspectral Transformer (HSIT) ####
#         Initial Training         ####
#######################################
# We pre-train this model first and then train the above model with pre-trained weights
class HyperTransformerPre(nn.Module):
    def __init__(self, config):
        super(HyperTransformerPre, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]

        self.num_res_blocks = [16, 1, 1, 1, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        # Scaled dot product attention
        lv1_dim      = config[config["train_dataset"]]["LR_size"]**2
        lv2_dim      = (2*config[config["train_dataset"]]["LR_size"])**2
        lv3_dim      = (4*config[config["train_dataset"]]["LR_size"])**2
        ### Scaled Dot Product Attention ###
        self.TS_lv3     = ScaledDotProductAttentionOnly(temperature=lv1_dim)
        self.TS_lv2     = ScaledDotProductAttentionOnly(temperature=lv2_dim)
        self.TS_lv1     = ScaledDotProductAttentionOnly(temperature=lv3_dim)
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)

        ###############
        ### stage22 ###
        ###############
        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)

        ###############
        ### stage33 ###
        ###############
        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        

        ##############
        ### FINAL ####
        ##############
        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        T_lv3  = self.TS_lv3(V_lv3, K_lv3, Q_lv3)
        T_lv2  = self.TS_lv2(V_lv2, K_lv2, Q_lv2)
        T_lv1  = self.TS_lv1(V_lv1, K_lv1, Q_lv1)

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage11: (L/4, W/4) scale ######
        #####################################
        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11     = x11 + x11_res

        #####################################
        #### stage22: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22     = x22 + x22_res

        #####################################
        ###### stage22: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33     = x33 + x33_res

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        x      = self.final_conv(xF)

        #####################################
        #      Output                       #
        #####################################
        output = {  "pred": x}
        return output


#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
# class HyperTransformer(nn.Module):
#     def __init__(self, config):
#         super(HyperTransformer, self).__init__()
#         self.is_DHP_MS      = config["is_DHP_MS"]
#         self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
#         self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
#         self.factor         = config[config["train_dataset"]]["factor"]

#         self.num_res_blocks = [16, 1, 1, 1, 4]
#         self.n_feats        = 256
#         self.res_scale      = 1

#         self.LFE_HSI    = LFE(in_channels=self.in_channels)
#         self.LFE_PAN    = LFE(in_channels=1)
#         self.TS         = TS_Hard()
#         self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

#         ###############
#         ### stage11 ###
#         ###############
#         self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
#         self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
#         self.ps12           = nn.PixelShuffle(2)
#         #Residial blocks
#         self.RB11           = nn.ModuleList()
#         for i in range(self.num_res_blocks[1]):
#             self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
#                 res_scale=self.res_scale))
#         self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

#         ###############
#         ### stage22 ###
#         ###############
#         self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
#         self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
#         self.ps23           = nn.PixelShuffle(2)
#         #Residual blocks
#         self.RB22 = nn.ModuleList()
#         for i in range(self.num_res_blocks[2]):
#             self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
#         self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

#         ###############
#         ### stage33 ###
#         ###############
#         self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
#         self.RB33 = nn.ModuleList()
#         for i in range(self.num_res_blocks[3]):
#             self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
#         self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

#         ##############
#         ### FINAL ####
#         ##############
#         self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
#         self.RBF = nn.ModuleList()
#         for i in range(self.num_res_blocks[4]):
#             self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
#         self.convF_tail = conv3x3(self.out_channels, self.out_channels)



#     def forward(self, X_MS, X_PAN):
#         with torch.no_grad():
#             if not self.is_DHP_MS:
#                 X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

#             else:
#                 X_MS_UP = X_MS
            
#             # Generating PAN, and PAN (UD) images
#             X_PAN   = X_PAN.unsqueeze(dim=1)
#             PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
#             PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

#         #Extracting T and S at multiple-scales
#         #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
#         V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
#         K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
#         Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

#         T_lv3, S_lv3  = self.TS(V_lv3, K_lv3, Q_lv3)
#         T_lv2, S_lv2  = self.TS(V_lv2, K_lv2, Q_lv2)
#         T_lv1, S_lv1  = self.TS(V_lv1, K_lv1, Q_lv1)

#         #Save feature maps for illustration purpose
#         # feature_dic={}
#         # f = 1e3
#         # feature_dic.update({"V": f*V_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"K": f*K_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"Q": f*Q_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"T": f*T_lv3.detach().cpu().numpy()})
#         # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/features_lv3.mat", feature_dic)
#         # exit()

#         #Shallow Feature Extraction (SFE)
#         x = self.SFE(X_MS)

#         #####################################
#         #### stage1: (L/4, W/4) scale ######
#         #####################################
#         x11 = x
#         #HyperTransformer at (L/4, W/4) scale
#         x11_res = x11
#         x11_res = torch.cat((x11_res, T_lv3), dim=1)
#         x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
#         S_lv3   = S_lv3.view(S_lv3.shape[0], S_lv3.shape[1], 1, 1)
#         x11_res = x11_res * S_lv3.expand_as(x11_res)
#         x11     = x11 + x11_res
#         #Residial learning
#         x11_res = x11
#         for i in range(self.num_res_blocks[1]):
#             x11_res = self.RB11[i](x11_res)
#         x11_res = self.conv11_tail(x11_res)
#         x11 = x11 + x11_res

#         #####################################
#         #### stage2: (L/2, W/2) scale ######
#         #####################################
#         x22 = self.conv12(x11)
#         x22 = F.relu(self.ps12(x22))
#         #HyperTransformer at (L/2, W/2) scale
#         x22_res = x22
#         x22_res = torch.cat((x22_res, T_lv2), dim=1)
#         x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
#         S_lv2   = S_lv2.view(S_lv2.shape[0], S_lv2.shape[1], 1, 1)
#         x22_res = x22_res * S_lv2.expand_as(x22_res)
#         x22     = x22 + x22_res
#         #Residial learning
#         x22_res = x22
#         for i in range(self.num_res_blocks[2]):
#             x22_res = self.RB22[i](x22_res)
#         x22_res = self.conv22_tail(x22_res)
#         x22 = x22 + x22_res

#         #####################################
#         ###### stage3: (L, W) scale ########
#         #####################################
#         x33 = self.conv23(x22)
#         x33 = F.relu(self.ps23(x33))
#         #HyperTransformer at (L, W) scale
#         x33_res = x33
#         x33_res = torch.cat((x33_res, T_lv1), dim=1)
#         x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
#         S_lv1   = S_lv1.view(S_lv1.shape[0], S_lv1.shape[1], 1, 1)
#         x33_res = x33_res * S_lv1.expand_as(x33_res)
#         x33     = x33 + x33_res
#         #Residual learning
#         x33_res = x33
#         for i in range(self.num_res_blocks[3]):
#             x33_res = self.RB33[i](x33_res)
#         x33_res = self.conv33_tail(x33_res)
#         x33 = x33 + x33_res

#         #####################################
#         ############ Feature Pyramid ########
#         #####################################
#         x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
#         x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
#         xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
#         #####################################
#         ####  Final convolution   ###########
#         #####################################
#         xF      = self.final_conv(xF)
#         xF_res  = xF

#         #Final resblocks
#         for i in range(self.num_res_blocks[4]):
#             xF_res = self.RBF[i](xF_res)
#         xF_res  = self.convF_tail(xF_res)
#         x       = xF + xF_res

#         #####################################
#         #      Perceptual loss              #
#         #####################################
#         Phi_LFE_lv1, Phi_LFE_lv2, Phi_LFE_lv3 = self.LFE_HSI(x)
#         #Transferal perceptual loss
#         loss_tp = LOSS_TP(Phi_LFE_lv1, T_lv1)+LOSS_TP(Phi_LFE_lv2, T_lv2)+LOSS_TP(Phi_LFE_lv3, T_lv3)
#         output = {  "pred": x, 
#                     "tp_loss": loss_tp}
#         return output


#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
# class HyperTransformerBASIC(nn.Module):
#     def __init__(self, config):
#         super(HyperTransformerBASIC, self).__init__()
#         self.is_DHP_MS      = config["is_DHP_MS"]
#         self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
#         self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
#         self.factor         = config[config["train_dataset"]]["factor"]
#         self.config         = config

#         self.num_res_blocks = [16, 1, 1, 1, 4]
#         self.n_feats        = 256
#         self.res_scale      = 1

#         self.LFE_HSI    = LFE(in_channels=self.in_channels)
#         self.LFE_PAN    = LFE(in_channels=1)
#         self.TS         = TS_Hard()
#         self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

#         ###############
#         ### stage11 ###
#         ###############
#         self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
#         self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
#         self.ps12           = nn.PixelShuffle(2)
#         #Residial blocks
#         self.RB11           = nn.ModuleList()
#         for i in range(self.num_res_blocks[1]):
#             self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
#                 res_scale=self.res_scale))
#         self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

#         ###############
#         ### stage22 ###
#         ###############
#         self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
#         self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
#         self.ps23           = nn.PixelShuffle(2)
#         #Residual blocks
#         self.RB22 = nn.ModuleList()
#         for i in range(self.num_res_blocks[2]):
#             self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
#         self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

#         ###############
#         ### stage33 ###
#         ###############
#         self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
#         self.RB33 = nn.ModuleList()
#         for i in range(self.num_res_blocks[3]):
#             self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
#         self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

#         ##############
#         ### FINAL ####
#         ##############
#         self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
#         self.RBF = nn.ModuleList()
#         for i in range(self.num_res_blocks[4]):
#             self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
#         self.convF_tail = conv3x3(self.out_channels, self.out_channels)



#     def forward(self, X_MS, X_PAN):
#         with torch.no_grad():
#             if not self.is_DHP_MS:
#                 X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

#             else:
#                 X_MS_UP = X_MS
            
#             # Generating PAN, and PAN (UD) images
#             X_PAN   = X_PAN.unsqueeze(dim=1)
#             PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
#             PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

#         #Extracting T and S at multiple-scales
#         #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
#         V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
#         #K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
#         #Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

#         #T_lv3, S_lv3  = self.TS(V_lv3, K_lv3, Q_lv3)
#         #T_lv2, S_lv2  = self.TS(V_lv2, K_lv2, Q_lv2)
#         #T_lv1, S_lv1  = self.TS(V_lv1, K_lv1, Q_lv1)

#         #Save feature maps for illustration purpose
#         # feature_dic={}
#         # f = 1e3
#         # feature_dic.update({"V": f*V_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"K": f*K_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"Q": f*Q_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"T": f*T_lv3.detach().cpu().numpy()})
#         # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/features_lv3.mat", feature_dic)
#         # exit()

#         #Shallow Feature Extraction (SFE)
#         x = self.SFE(X_MS)

#         #####################################
#         #### stage1: (L/4, W/4) scale ######
#         #####################################
#         x11 = x
#         #HyperTransformer at (L/4, W/4) scale
#         x11_res = x11
#         x11_res = torch.cat((x11_res, V_lv3), dim=1)
#         x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
#         #S_lv3   = S_lv3.view(S_lv3.shape[0], S_lv3.shape[1], 1, 1)
#         #x11_res = x11_res * S_lv3.expand_as(x11_res)
#         x11     = x11 + x11_res
#         #Residial learning
#         x11_res = x11
#         for i in range(self.num_res_blocks[1]):
#             x11_res = self.RB11[i](x11_res)
#         x11_res = self.conv11_tail(x11_res)
#         x11 = x11 + x11_res

#         #####################################
#         #### stage2: (L/2, W/2) scale ######
#         #####################################
#         x22 = self.conv12(x11)
#         x22 = F.relu(self.ps12(x22))
#         #HyperTransformer at (L/2, W/2) scale
#         x22_res = x22
#         x22_res = torch.cat((x22_res, V_lv2), dim=1)
#         x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
#         #S_lv2   = S_lv2.view(S_lv2.shape[0], S_lv2.shape[1], 1, 1)
#         #x22_res = x22_res * S_lv2.expand_as(x22_res)
#         x22     = x22 + x22_res
#         #Residial learning
#         x22_res = x22
#         for i in range(self.num_res_blocks[2]):
#             x22_res = self.RB22[i](x22_res)
#         x22_res = self.conv22_tail(x22_res)
#         x22 = x22 + x22_res

#         #####################################
#         ###### stage3: (L, W) scale ########
#         #####################################
#         x33 = self.conv23(x22)
#         x33 = F.relu(self.ps23(x33))
#         #HyperTransformer at (L, W) scale
#         x33_res = x33
#         x33_res = torch.cat((x33_res, V_lv1), dim=1)
#         x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
#         #S_lv1   = S_lv1.view(S_lv1.shape[0], S_lv1.shape[1], 1, 1)
#         #x33_res = x33_res * S_lv1.expand_as(x33_res)
#         x33     = x33 + x33_res
#         #Residual learning
#         x33_res = x33
#         for i in range(self.num_res_blocks[3]):
#             x33_res = self.RB33[i](x33_res)
#         x33_res = self.conv33_tail(x33_res)
#         x33 = x33 + x33_res

#         #####################################
#         ############ Feature Pyramid ########
#         #####################################
#         x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
#         x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
#         xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
#         #####################################
#         ####  Final convolution   ###########
#         #####################################
#         xF      = self.final_conv(xF)
#         xF_res  = xF

#         #Final resblocks
#         for i in range(self.num_res_blocks[4]):
#             xF_res = self.RBF[i](xF_res)
#         xF_res  = self.convF_tail(xF_res)
#         x       = xF + xF_res

#         #####################################
#         #      Perceptual loss              #
#         #####################################
#         #Phi_LFE_lv1, Phi_LFE_lv2, Phi_LFE_lv3 = self.LFE_HSI(x)
#         #Transferal perceptual loss
#         #if self.config[self.config["train_dataset"]]["Transfer_Periferal_Loss"]:
#         #loss_tp = LOSS_TP(Phi_LFE_lv1, T_lv1)+LOSS_TP(Phi_LFE_lv2, T_lv2)+LOSS_TP(Phi_LFE_lv3, T_lv3)
#         output = {  "pred": x }
#         return output
