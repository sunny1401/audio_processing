import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DWConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class YOHO(nn.Module):
    def __init__(self, in_shape=(1, 801, 64), num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            # Input: [B, 1, 801, 64]
            ConvBlock(1, 32, kernel_size=3, stride=2, padding=1),          # 401 × 32 × 32
            DWConvBlock(32, kernel_size=3, padding=1),
            ConvBlock(32, 64, kernel_size=1),

            DWConvBlock(64, kernel_size=3, stride=2, padding=1),           # 201 × 16 × 64
            ConvBlock(64, 128, kernel_size=1),
            DWConvBlock(128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=1),

            DWConvBlock(128, kernel_size=3, stride=2, padding=1),          # 101 × 8 × 128
            ConvBlock(128, 256, kernel_size=1),
            DWConvBlock(256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),

            DWConvBlock(256, kernel_size=3, stride=2, padding=1),          # 51 × 4 × 256
            ConvBlock(256, 512, kernel_size=1),

            *[nn.Sequential(
                DWConvBlock(512, kernel_size=3, padding=1),
                ConvBlock(512, 512, kernel_size=1)
            ) for _ in range(5)],                                          # Repeat block ×5

            DWConvBlock(512, kernel_size=3, stride=2, padding=1),          # 26 × 2 × 512
            ConvBlock(512, 1024, kernel_size=1),
            DWConvBlock(1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=1),

            DWConvBlock(1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            DWConvBlock(512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            DWConvBlock(256, kernel_size=3, padding=1),
            ConvBlock(256, 128, kernel_size=1),
        )

        # Final reshape and 1D conv
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((26, 256)), 
            nn.Flatten(start_dim=2),          # [B, 128, 26] → [B, 26, 128]
            nn.Conv1d(256, 6, kernel_size=1), # Final output: [B, 6, 26]
            nn.Softmax(dim=1)                 
        )

    def forward(self, x):
        x = self.model(x)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), 256, 26)  # Reshape: [B, 256, 26]
        x = self.head(x)                                      # → [B, 6, 26]
        return x.permute(0, 2, 1)                              # [B, 26, 6]
