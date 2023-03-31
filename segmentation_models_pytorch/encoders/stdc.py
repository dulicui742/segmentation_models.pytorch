import math
import torch.nn as nn

from ._base import EncoderMixin
from .net.stdcnet import(
    STDCNet813, 
    STDCNet1446, 
    CatBottleneck, 
    AddBottleneck,
    ConvX,
)

class SDTCEncoder(STDCNet1446, EncoderMixin):
    def __init__(
        self, 
        out_channels,
        model_name="stdc2", 
        layers=[2, 2, 2], 
        base=64,
        depth=5,
        type="cat",
        block_num=4
    ):
        super().__init__()

        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        
        self._depth = depth
        self._in_channels = 3  ### patch_first_conv will use this parameter
        self._out_channels = out_channels
        self.features = self.re_make_layers(self._in_channels, base, layers, block_num, block)

        if model_name == "stdc1": ## default: stdc2
            self.x8 = nn.Sequential(self.features[2:4])
            self.x16 = nn.Sequential(self.features[4:6])
            self.x32 = nn.Sequential(self.features[6:])

    def re_make_layers(self, in_channels, base, layers, block_num, block):
        features = []
        features += [ConvX(in_channels, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
        return nn.Sequential(*features)
    
    def get_stages(self):
        return [nn.Identity(), self.x2, self.x4, self.x8, self.x16, self.x32]
        
    def forward(self, x):
        """Apply forward pass."""
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        pass 


stdc_encoders = {
    "stdc1": {
        "encoder": SDTCEncoder,
        "params": {
            # "in_channels": 1,
            "base": 64,
            "layers": [2, 2, 2],
            "out_channels": (3, 32, 64, 256, 512, 1024),
            "model_name": "stdc1",
        }

    },

    "stdc2": {
        "encoder": SDTCEncoder,
        "params": {
            # "in_channels": 1,
            "base": 64,
            "layers": [4, 5, 3],
            "out_channels": (3, 32, 64, 256, 512, 1024),
            "model_name": "stdc2",
        }
    }
}
