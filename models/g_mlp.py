from torch import nn
from torch.nn import functional as F
from .utils import pair, check_sizes

'''
https://github.com/jaketae/g-mlp/blob/master/g_mlp/core.py
'''
#code was copied from https://github.com/liuruiyang98/Jittor-MLP/blob/main/models_pytorch/g_mlp.py and modified

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out

class gMLP(nn.Module):
    def __init__(self, d_model=512, d_ffn=1536, seq_len=256, depth=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)


class gMLPForImageClassification(gMLP):
    def __init__(
        self,
        image_size=32,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        d_model=512,
        d_ffn=1536,
        depth=6,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(d_model, d_ffn, num_patches, depth)
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out
