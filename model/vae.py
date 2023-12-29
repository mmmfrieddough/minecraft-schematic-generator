import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length,
                         self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (Dim / H).
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        # x: (Batch_Size, Features, Depth, Height, Width)

        residue = x

        # (Batch_Size, Features, Depth, Height, Width) -> (Batch_Size, Features, Depth, Height, Width)
        x = self.groupnorm(x)

        n, c, d, h, w = x.shape

        # (Batch_Size, Features, Depth, Height, Width) -> (Batch_Size, Features, Depth * Height * Width)
        x = x.view((n, c, d * h * w))

        # (Batch_Size, Features, Depth * Height * Width) -> (Batch_Size, Depth * Height * Width, Features)
        x = x.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # (Batch_Size, Depth * Height * Width, Features) -> (Batch_Size, Depth * Height * Width, Features)
        x = self.attention(x)

        # (Batch_Size, Depth * Height * Width, Features) -> (Batch_Size, Features, Depth * Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Depth * Height * Width) -> (Batch_Size, Features, Depth, Height, Width)
        x = x.view((n, c, d, h, w))

        # (Batch_Size, Features, Depth, Height, Width) + (Batch_Size, Features, Depth, Height, Width) -> (Batch_Size, Features, Depth, Height, Width)
        x += residue

        # (Batch_Size, Features, Depth, Height, Width)
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: (Batch_Size, In_Channels, Depth, Height, Width)

        residue = x

        # (Batch_Size, In_Channels, Depth, Height, Width) -> (Batch_Size, In_Channels, Depth, Height, Width)
        x = self.groupnorm_1(x)

        # (Batch_Size, In_Channels, Depth, Height, Width) -> (Batch_Size, In_Channels, Depth, Height, Width)
        x = F.silu(x)

        # (Batch_Size, In_Channels, Depth, Height, Width) -> (Batch_Size, Out_Channels, Depth, Height, Width)
        x = self.conv_1(x)

        # (Batch_Size, Out_Channels, Depth, Height, Width) -> (Batch_Size, Out_Channels, Depth, Height, Width)
        x = self.groupnorm_2(x)

        # (Batch_Size, Out_Channels, Depth, Height, Width) -> (Batch_Size, Out_Channels, Depth, Height, Width)
        x = F.silu(x)

        # (Batch_Size, Out_Channels, Depth, Height, Width) -> (Batch_Size, Out_Channels, Depth, Height, Width)
        x = self.conv_2(x)

        # (Batch_Size, Out_Channels, Depth, Height, Width) -> (Batch_Size, Out_Channels, Depth, Height, Width)
        return x + self.residual_layer(residue)


class VAE_Encoder(nn.Sequential):
    def __init__(self, num_classes):
        super().__init__(
            # (Batch_Size, C, Depth, Height, Width) -> (Batch_Size, 64, Depth, Height, Width)
            nn.Embedding(num_classes, 64),

            # (Batch_Size, 64, Depth, Height, Width) -> (Batch_Size, 128, Depth, Height, Width)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Depth, Height, Width) -> (Batch_Size, 128, Depth, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Depth, Height, Width) -> (Batch_Size, 128, Depth / 2, Height / 2, Width / 2)
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, Depth / 2, Height / 2, Width / 2) -> (Batch_Size, 256, Depth / 2, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Depth / 2, Height / 2, Width / 2) -> (Batch_Size, 256, Depth / 4, Height / 4, Width / 4)
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Depth / 4, Height / 4, Width / 4) -> (Batch_Size, 512, Depth / 4, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # # (Batch_Size, 512, Depth / 4, Height / 4, Width / 4) -> (Batch_Size, 512, Depth / 8, Height / 8, Width / 8)
            # nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=0),

            # VAE_ResidualBlock(512, 512),
            # VAE_ResidualBlock(512, 512),
            # VAE_ResidualBlock(512, 512),
            # VAE_AttentionBlock(512),
            # VAE_ResidualBlock(512, 512),

            # nn.GroupNorm(32, 512),
            # nn.SiLU(),

            # (Batch_Size, 512, Depth / 8, Height / 8, Width / 8) -> (Batch_Size, 8, Depth / 8, Height / 8, Width / 8)
            nn.Conv3d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Depth / 8, Height / 8, Width / 8) -> (Batch_Size, 8, Depth / 8, Height / 8, Width / 8)
            nn.Conv3d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # x: (Batch_Size, Channel, Depth, Height, Width)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2, 2):
                x = F.pad(x, (0, 1, 0, 1, 0, 1))

            x = module(x)

            if isinstance(module, nn.Embedding):
                x = x.permute(0, 4, 1, 2, 3).contiguous()

        mu, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        return mu, logvar


class VAE_Decoder(nn.Sequential):
    def __init__(self, num_classes):
        super().__init__(
            # (Batch_Size, 4, Depth / 8, Height / 8, Width / 8) -> (Batch_Size, 4, Depth / 8, Height / 8, Width / 8)
            nn.Conv3d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Depth / 8, Height / 8, Width / 8) -> (Batch_Size, 512, Depth / 8, Height / 8, Width / 8)
            nn.Conv3d(4, 512, kernel_size=3, padding=1),

            # # 3D Residual and Attention Blocks
            # # (Batch_Size, 512, Depth / 8, Height / 8, Width / 8) -> (Batch_Size, 512, Depth / 8, Height / 8, Width / 8)
            # VAE_ResidualBlock(512, 512),
            # VAE_AttentionBlock(512),
            # VAE_ResidualBlock(512, 512),
            # VAE_ResidualBlock(512, 512),
            # VAE_ResidualBlock(512, 512),

            # # Upsampling for 3D
            # # (Batch_Size, 512, Depth / 8, Height / 8, Width / 8) -> (Batch_Size, 512, Depth / 4, Height / 4, Width / 4)
            # nn.Upsample(scale_factor=(2, 2, 2)),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),

            # (Batch_Size, 512, Depth / 4, Height / 4, Width / 4) -> (Batch_Size, 512, Depth / 4, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Depth / 4, Height / 4, Width / 4) -> (Batch_Size, 512, Depth / 2, Height / 2, Width / 2)
            nn.Upsample(scale_factor=(2, 2, 2)),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),

            # (Batch_Size, 512, Depth / 2, Height / 2, Width / 2) -> (Batch_Size, 256, Depth / 2, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Depth / 2, Height / 2, Width / 2) -> (Batch_Size, 256, Depth, Height, Width)
            nn.Upsample(scale_factor=(2, 2, 2)),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),

            # (Batch_Size, 256, Depth, Height, Width) -> (Batch_Size, 128, Depth, Height, Width)
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Depth, Height, Width) -> (Batch_Size, 128, Depth, Height, Width)
            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # (Batch_Size, 128, Depth, Height, Width) -> (Batch_Size, 64, Depth, Height, Width)
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),

            # (Batch_Size, 64, Depth, Height, Width) -> (Batch_Size, C, Depth, Height, Width)
            nn.Conv3d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Depth / 8, Height / 8, Width / 8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Depth, Height, Width)
        return x


class VAE(nn.Module):
    def __init__(self, num_classes):
        super(VAE, self).__init__()
        self.num_classes = num_classes
        self.encoder = VAE_Encoder(num_classes)
        self.decoder = VAE_Decoder(num_classes)

    def reparameterize(self, mu, logvar):
        variance = logvar.exp()
        stdev = variance.sqrt()
        noise = torch.randn_like(stdev)
        x = mu + stdev * noise
        x *= 0.18215
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar
