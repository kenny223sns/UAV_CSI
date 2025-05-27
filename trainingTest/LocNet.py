import torch.nn as nn
from torchvision.models import resnet34

class LocNet(nn.Module):
    def __init__(self, in_ch=4, out_dim=2):
        super().__init__()
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim)                 # x , y
        )
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
