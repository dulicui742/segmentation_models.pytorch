import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class Attention_gata(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_gata, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), 
            nn.BatchNorm2d(F_int))
 
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), 
            nn.BatchNorm2d(F_int))
 
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
 
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.attention_gate = Attention_gata(F_g=skip_channels, F_l=in_channels, F_int=out_channels)
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        
    def forward(self, x, skip=None, scale_factor=2, skip_attention=False):
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if skip is not None:
            # print("concat in:", x.shape, skip.shape)
            if skip_attention:
                ## -----attention gate  Attention Unet ------
                x = self.attention_gate(skip, x)
                ## -----attention gate  Attention Unet ------

            x = torch.cat([x, skip], dim=1)
            # print("concat out", x.shape)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        scale_factor=[2, 2, 2, 2],
        skip_attention=True,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.scale_factor = scale_factor
        self.skip_attention = skip_attention
        print("------skip attention: ", self.skip_attention, "-------")

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution (input)
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        # head.shape: torch.Size([1, 448, 64, 64])
        # [ skips.shape
        # torch.Size([1, 160, 64, 64]), 
        # torch.Size([1, 56, 64, 64]), 
        # torch.Size([1, 32, 128, 128]), 
        # torch.Size([1, 48, 256, 256])
        # ]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip, self.scale_factor[i], self.skip_attention)
            # print("decoder: ", i, x.shape)

        return x
