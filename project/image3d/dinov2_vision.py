import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
import pdb

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features = None,
    ):
        super().__init__()
        assert hidden_features is not None
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class NestedTensorBlock(nn.Module):
    def __init__(
        self,
        dim = 1024,
        num_heads = 16,
        mlp_ratio = 4.0,
    ):
        super().__init__()
        assert dim == 1024
        assert num_heads == 16
        assert mlp_ratio == 4

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
        )
        self.ls2 = LayerScale(dim)


    def forward(self, x):
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size = 224,
        patch_size = 16,
        in_chans = 3,
        embed_dim = 768,
    ):
        super().__init__()
        assert img_size == 518
        assert patch_size == 14
        assert in_chans == 3
        assert embed_dim == 1024

        image_HW = (img_size, img_size)
        patch_HW = (patch_size, patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] # keep !!!
        self.patch_size = patch_HW
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)

    def forward(self, x):
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        # H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        return x

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=4,
    ):
        super().__init__()
        assert img_size == 518
        assert patch_size == 14
        assert in_chans == 3
        assert embed_dim == 1024
        assert depth == 24
        assert num_heads == 16
        assert mlp_ratio == 4
        assert num_register_tokens == 4

        self.num_tokens = 1
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        blocks_list = [
            NestedTensorBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth) # depth === 24
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(embed_dim, 1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim)) # useless !!!

        self.load_weights()

    def load_weights(self, model_path="models/image3d_dinov2.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)

    def forward(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed # self.interpolate_pos_encoding(x, w, h)
        x = torch.cat(
            (
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = F.layer_norm(x, x.shape[-1:]) # features.shape[-1:] == [1024]
        return x # x_prenorm


if __name__ == "__main__":
    model = DinoVisionTransformer()
    model.half().eval().cuda()
    print(model)

    image = torch.randn(1, 3, 518, 518).half().cuda()
    with torch.no_grad():
        features = model(image)
    todos.debug.output_var("features", features)

    # tensor [features] size: [1, 1374, 1024], min: -25.769878, max: 15.378892, mean: 0.0
