import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        # output_stride=32,
    ):
        super().__init__()
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
        # self.output_stride = output_stride

    # def forward(self, x, skip=None):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
    def forward(self, x, skip=None, scale_factor=2):
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if skip is not None:
            # print("concat in:", x.shape, skip.shape)
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
        # output_stride=32,
        scale_factor=[2, 2, 2, 2],
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
        # kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type, output_stride=output_stride)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.scale_factor = scale_factor

    # def forward(self, *features):

    #     # import pdb; pdb.set_trace()
    #     features = features[1:]  # remove first skip with same spatial resolution (input)
    #     features = features[::-1]  # reverse channels to start from head of encoder

    #     head = features[0]
    #     skips = features[1:]
    #     # head.shape: torch.Size([1, 448, 64, 64])
    #     # [ skips.shape
    #     # torch.Size([1, 160, 64, 64]), 
    #     # torch.Size([1, 56, 64, 64]), 
    #     # torch.Size([1, 32, 128, 128]), 
    #     # torch.Size([1, 48, 256, 256])
    #     # ]
    #     x = self.center(head)
    #     for i, decoder_block in enumerate(self.blocks):
    #         skip = skips[i] if i < len(skips) else None
    #         x = decoder_block(x, skip)
    #         # print("decoder: ", i, x.shape)

    #     return x

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
            x = decoder_block(x, skip, self.scale_factor[i])
            # print("decoder: ", i, x.shape, self.scale_factor[i])

        return x
