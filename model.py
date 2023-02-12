from turtle import forward
import torch

import torch.nn as nn
import segmentation_models_pytorch as smp

class ChannelAteention(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(ChannelAteention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out,_ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim = 1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x

class Unet(nn.Module):
    def __init__(self, cfg_model, no_of_landmarks):
        super(Unet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=cfg_model.ENCODER_NAME,
            encoder_weights=cfg_model.ENCODER_WEIGHTS,
            decoder_channels=cfg_model.DECODER_CHANNELS,
            in_channels=cfg_model.IN_CHANNELS,
            classes=no_of_landmarks,
        )
        # self.CA = ChannelAteention()
        if cfg_model.SPATIAL_ATTENTION:
            self.SA = SpatialAttention()
        else:
            self.SA = nn.Identity()
        self.temperatures = nn.Parameter(torch.ones(1, no_of_landmarks, 1, 1), requires_grad=False)
        self.texture_head = nn.Conv2d(in_channels=cfg_model.DECODER_CHANNELS[-1], out_channels=no_of_landmarks, kernel_size=3, padding=1)
        self.global_head = nn.Conv2d(in_channels=cfg_model.DECODER_CHANNELS[-1], out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # features = self.unet.encoder(x)
        stages = self.unet.encoder.get_stages()
        features = []
        for i in range(self.unet.encoder._depth + 1):
            x = stages[i](x)
            # x = self.CA(x)
            x = self.SA(x)
            features.append(x)
        decoder_output = self.unet.decoder(*features)
        masks = self.unet.segmentation_head(decoder_output)
        texture_output = self.texture_head(decoder_output)
        strcture_output = self.global_head(decoder_output)
        if self.unet.classification_head is not None:
            labels = self.unet.classification_head(features[-1])
            return masks, labels, texture_output, strcture_output
        return masks , texture_output, strcture_output

    def scale(self, x):
        y = x / self.temperatures
        return y

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

def two_d_normalize(x):
    return x / torch.amax(x, dim=(2,3), keepdim=True)

def channel_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=1, keepdim=True)

def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))

def nll_across_channel(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=1))

def mse_across_channel(output, target):
    return torch.mean(torch.sum(torch.square(output-target)))
