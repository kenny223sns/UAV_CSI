"""FusionNet: light‑weight multi‑scale backbone for CSI localisation.
Idea: depthwise separable conv stack + SPP + global context.
Produces embedding dim 512, plug‑compatible with LocNetMulti heads.
"""
import torch, torch.nn as nn
import torch.nn.functional as F

class DepthwiseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, act=nn.ReLU):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, k//2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class SPP(nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, in_ch, pool_sizes=(5,9,13)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(ps, stride=1, padding=ps//2),
                          nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False))
            for ps in pool_sizes])
        self.bn = nn.BatchNorm2d(in_ch*(len(pool_sizes)+1))
    def forward(self,x):
        outs=[x]
        for c in self.convs: outs.append(c(x))
        return self.bn(torch.cat(outs,1))

class FusionNet(nn.Module):
    def __init__(self, in_ch=4, emb_dim=512):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            DepthwiseBlock(32,64),
        )
        self.stage2 = nn.Sequential(DepthwiseBlock(64,128, s=2), DepthwiseBlock(128,128))
        self.stage3 = nn.Sequential(DepthwiseBlock(128,256, s=2), DepthwiseBlock(256,256))
        self.stage4 = nn.Sequential(DepthwiseBlock(256,512, s=2), DepthwiseBlock(512,512))
        self.spp    = SPP(512)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512*4, emb_dim)
    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.spp(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    net = FusionNet()
    print(sum(p.numel() for p in net.parameters())/1e6, 'M params')
