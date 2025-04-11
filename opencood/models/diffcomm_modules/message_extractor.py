import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVForegroundBackgroundExtractor(nn.Module):
    def __init__(self, in_channels=128, out_channels=2):
        super().__init__()
        # Multi-scale: 1x1, 3x3, 5x5
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)
        self.branch3 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)

        # Attention block
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 48, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(48, 192, kernel_size=1),
            nn.Sigmoid()
        )

        # Final fusion and output
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        concat = torch.cat([b1, b3, b5], dim=1)

        attn_weight = self.attn(concat)
        enhanced = concat * attn_weight

        return self.fusion(enhanced)

class MessageExtractor(nn.Module):
    def __init__(self, in_channels=128, out_channels=2):
        super(MessageExtractor, self).__init__()

        self.bev_extractor = BEVForegroundBackgroundExtractor(in_channels, out_channels)

    def forward(self, bev_feature,):

        enhanced_feature = self.bev_extractor(bev_feature)
        return enhanced_feature
    
if __name__ == '__main__':
    # 测试前景增强模块
    in_channels = 128
    reduction_ratio = 16
    bev_feature = torch.randn(4, in_channels, 100, 352)
