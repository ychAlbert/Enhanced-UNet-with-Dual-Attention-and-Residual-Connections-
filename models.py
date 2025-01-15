import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class DualAttentionBlock(nn.Module):
    """双重注意力模块：结合空间和通道注意力"""
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_attention = AttentionBlock(in_channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sa = self.spatial_attention(x)
        ca = self.channel_attention(x)
        return sa * ca

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class EnhancedUNet(nn.Module):
    """增强型UNet，集成了注意力机制和残差连接"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, 64),
            DualAttentionBlock(64)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(64, 128),
            DualAttentionBlock(128)
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(128, 256),
            DualAttentionBlock(256)
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(256, 512),
            DualAttentionBlock(512)
        )
        
        # 解码器
        self.dec4 = nn.Sequential(
            ResidualBlock(512 + 256, 256),
            DualAttentionBlock(256)
        )
        self.dec3 = nn.Sequential(
            ResidualBlock(256 + 128, 128),
            DualAttentionBlock(128)
        )
        self.dec2 = nn.Sequential(
            ResidualBlock(128 + 64, 64),
            DualAttentionBlock(64)
        )
        
        # 最终分割头
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # 最大池化
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # 解码路径
        dec4 = self.dec4(torch.cat([F.interpolate(enc4, enc3.shape[2:]), enc3], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc2.shape[2:]), enc2], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc1.shape[2:]), enc1], 1))
        
        return self.final(dec2)