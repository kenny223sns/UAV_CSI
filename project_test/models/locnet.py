"""LocNet updated to support plug‑in backbones (ResNet / FusionNet).
Usage:
    model = LocNet(in_ch=4, backbone='fusion', emb_dim=512, out_dim=2)
"""
from typing import Tuple
import torch.nn as nn
from torchvision.models import resnet34
from models.fusion_net import FusionNet

__all__: Tuple[str,...] = ("LocNet",)

class LocNet(nn.Module):
    def __init__(self, in_ch:int=4, backbone:str='resnet', emb_dim:int=512, out_dim:int=2):
        super().__init__()
        if backbone=='resnet':
            net = resnet34(weights=None)
            net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            net.fc    = nn.Identity()
            self.backbone = net
            final_dim = 512
        elif backbone=='fusion':
            self.backbone = FusionNet(in_ch, emb_dim)
            final_dim = emb_dim
        else:
            raise ValueError(backbone)
        self.head = nn.Sequential(
            nn.Linear(final_dim, final_dim//2), nn.ReLU(inplace=True),
            nn.Linear(final_dim//2, out_dim))
    def forward(self,x):
        f = self.backbone(x)
        return self.head(f)

if __name__ == "__main__":
    m = LocNet(backbone='fusion')
    print(sum(p.numel() for p in m.parameters())/1e6, 'M params')
