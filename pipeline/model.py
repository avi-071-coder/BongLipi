"""
=============================================================================
BONGLIPI - Model Architecture: BongNet
=============================================================================
A custom CNN for Bengali handwritten character recognition.

Architecture: ResNet-inspired with:
  - Stem: initial convolution (1→32 channels)
  - 4 Residual stages with increasing channels (32→64→128→256)
  - Each stage has 2 residual blocks with skip connections
  - Global Average Pooling → Dropout → FC classifier
  - Squeeze-and-Excitation (SE) attention in each block

Designed for 64x64 single-channel (grayscale) inputs.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """
    Residual block with optional downsampling and SE attention.
    
    Structure:
      x → Conv3x3 → BN → ReLU → Conv3x3 → BN → SE → + → ReLU
      |                                               ↑
      └──────── (optional 1x1 conv + downsample) ─────┘
    """
    
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        
        # Skip connection with optional projection
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class BongNet(nn.Module):
    """
    BongNet: Custom CNN for Bengali Character Recognition.
    
    Input: (B, 1, 64, 64) grayscale images
    Output: (B, num_classes) logits
    
    Architecture:
      Stem     → 64x64 → 32ch
      Stage 1  → 32x32 → 64ch  (2 ResBlocks, stride 2)
      Stage 2  → 16x16 → 128ch (2 ResBlocks, stride 2)
      Stage 3  → 8x8   → 256ch (2 ResBlocks, stride 2)
      Stage 4  → 4x4   → 512ch (2 ResBlocks, stride 2)
      GAP → Dropout → FC → num_classes
    """
    
    def __init__(self, num_classes=72, dropout=0.3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Residual stages
        self.stage1 = self._make_stage(32, 64, num_blocks=2, stride=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)
        
        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        
        # Weight initialization
        self._init_weights()
    
    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)      # (B, 32, 64, 64)
        x = self.stage1(x)    # (B, 64, 32, 32)
        x = self.stage2(x)    # (B, 128, 16, 16)
        x = self.stage3(x)    # (B, 256, 8, 8)
        x = self.stage4(x)    # (B, 512, 4, 4)
        
        x = self.pool(x)      # (B, 512, 1, 1)
        x = x.flatten(1)      # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)        # (B, num_classes)
        return x
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BongNetLite(nn.Module):
    """
    Lightweight version for faster CPU training.
    ~1M params instead of ~4M.
    """
    
    def __init__(self, num_classes=72, dropout=0.25):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.stage1 = self._make_stage(16, 32, 2, stride=2)
        self.stage2 = self._make_stage(32, 64, 2, stride=2)
        self.stage3 = self._make_stage(64, 128, 2, stride=2)
        self.stage4 = self._make_stage(128, 256, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride=stride, use_se=True)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1, use_se=True))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    for ModelClass, name in [(BongNet, "BongNet"), (BongNetLite, "BongNetLite")]:
        model = ModelClass(num_classes=72)
        x = torch.randn(4, 1, 64, 64)
        y = model(x)
        print(f"{name}: {model.count_params():,} params | Output: {y.shape}")
