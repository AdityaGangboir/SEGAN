# training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for better temporal modeling."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + residual, 0.2)


class DownsampleBlock(nn.Module):
    """Downsampling block with spectral normalization."""
    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.conv(x))


class UpsampleBlock(nn.Module):
    """Upsampling block with batch normalization."""
    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        output_padding = stride - 1
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class UNetGenerator1D(nn.Module):
    """
    Enhanced U-Net generator with residual connections and skip connections.
    Better for preserving fine details in audio.
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = DownsampleBlock(1, base_channels, stride=2)
        self.enc2 = DownsampleBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = DownsampleBlock(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = DownsampleBlock(base_channels * 4, base_channels * 8, stride=2)
        self.enc5 = DownsampleBlock(base_channels * 8, base_channels * 8, stride=2)
        
        # Bottleneck with residual blocks
        self.res1 = ResidualBlock(base_channels * 8, dilation=1)
        self.res2 = ResidualBlock(base_channels * 8, dilation=2)
        self.res3 = ResidualBlock(base_channels * 8, dilation=4)
        
        # Decoder with skip connections
        self.dec5 = UpsampleBlock(base_channels * 8, base_channels * 8, stride=2)
        self.dec4 = UpsampleBlock(base_channels * 16, base_channels * 4, stride=2)
        self.dec3 = UpsampleBlock(base_channels * 8, base_channels * 2, stride=2)
        self.dec2 = UpsampleBlock(base_channels * 4, base_channels, stride=2)
        self.dec1 = UpsampleBlock(base_channels * 2, base_channels, stride=2)
        
        # Output layer
        self.out_conv = nn.Conv1d(base_channels, 1, kernel_size=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        b = self.res1(e5)
        b = self.res2(b)
        b = self.res3(b)
        
        # Decoder with skip connections
        d5 = self.dec5(b)
        d5 = torch.cat([d5, e4], dim=1)
        
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)
        
        # Output
        out = self.tanh(self.out_conv(d1))
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates on different resolutions.
    Provides better gradient flow and captures features at multiple scales.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Three discriminators at different scales
        self.discriminators = nn.ModuleList([
            SubDiscriminator(base_channels),
            SubDiscriminator(base_channels),
            SubDiscriminator(base_channels)
        ])
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        
    def forward(self, noisy, clean):
        outputs = []
        feature_maps = []
        
        # Concatenate noisy and clean
        x = torch.cat([noisy, clean], dim=1)
        
        for i, disc in enumerate(self.discriminators):
            out, features = disc(x)
            outputs.append(out)
            feature_maps.append(features)
            
            # Downsample for next scale
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        
        return outputs, feature_maps


class SubDiscriminator(nn.Module):
    """Single discriminator with spectral normalization."""
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(2, base_channels, kernel_size=15, stride=1, padding=7)),
            spectral_norm(nn.Conv1d(base_channels, base_channels * 2, kernel_size=41, stride=4, padding=20)),
            spectral_norm(nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=41, stride=4, padding=20)),
            spectral_norm(nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=41, stride=4, padding=20)),
            spectral_norm(nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size=41, stride=4, padding=20)),
            spectral_norm(nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size=5, stride=1, padding=2))
        ])
        
        self.out_conv = spectral_norm(nn.Conv1d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)
        
        out = self.out_conv(x)
        
        return out, features


class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
