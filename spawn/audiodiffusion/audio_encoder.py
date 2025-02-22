# audio_encoder.py

import os
import numpy as np
import torch
#from diffusers import ConfigMixin, Mel, ModelMixin
from torch import nn
# Instead of importing from diffusers, import from local mel.py which contains minimal stubs.
from .mel import ConfigMixin, Mel, ModelMixin

try:
    import safetensors.torch
except ImportError:
    raise ImportError("safetensors package is required to load the model weights. Please install it.")


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
            padding=1,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(ConvBlock, self).__init__()
        self.sep_conv = SeparableConv2d(in_channels, out_channels, (3, 3))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.sep_conv(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(DenseBlock, self).__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm1d(out_features, eps=0.001, momentum=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.flatten(x.permute(0, 2, 3, 1))
        x = self.dense(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class AudioEncoder(ModelMixin, ConfigMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = Mel(
            x_res=216,
            y_res=96,
            sample_rate=22050,
            n_fft=2048,
            hop_length=512,
            top_db=80,
        )
        self.conv_blocks = nn.ModuleList([ConvBlock(1, 32, 0.2), ConvBlock(32, 64, 0.3), ConvBlock(64, 128, 0.4)])
        self.dense_block = DenseBlock(41472, 1024, 0.5)
        self.embedding = nn.Linear(1024, 100)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.dense_block(x)
        x = self.embedding(x)
        return x

    @classmethod
    def from_pretrained(cls, model_path=None):
        """
        Load the AudioEncoder from a local directory.
        By default, it loads from the "audio-encoder" subdirectory relative to this file.
        It expects to find at least:
            - config.json
            - diffusion_pytorch_model.safetensors
        """
        if model_path is None:
            # Use the "audio-encoder" folder relative to the audiodiffusion directory
            model_path = os.path.join(os.path.dirname(__file__), "..", "audio-encoder")
        instance = cls()
        weight_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weight file not found at {weight_path}")
        # Load the state dict using safetensors
        state_dict = safetensors.torch.load_file(weight_path)
        instance.load_state_dict(state_dict, strict=False)
        return instance

    @torch.no_grad()
    def encode(self, audio_files, pool="average"):
        self.eval()
        y = []
        for audio_file in audio_files:
            self.mel.load_audio(audio_file)
            x = [
                np.expand_dims(
                    np.frombuffer(self.mel.audio_slice_to_image(slice).tobytes(), dtype="uint8").reshape(
                        (self.mel.y_res, self.mel.x_res)
                    )
                    / 255,
                    axis=0,
                )
                for slice in range(self.mel.get_number_of_slices())
            ]
            y += [self(torch.Tensor(x))]
            if pool == "average":
                y[-1] = torch.mean(y[-1], dim=0)
            elif pool == "max":
                y[-1] = torch.max(y[-1], dim=0)
            else:
                assert pool is None, f"Unknown pooling method {pool}"
        return torch.stack(y)
