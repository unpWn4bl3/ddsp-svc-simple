import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            max_abs = torch.max(-out_min, out_max).float() * torch.max(-gate_min, gate_max).float()
            if max_abs > 1000:
                ratio = (1000 / max_abs).half()
                gate *= ratio
                return (out * gate).clamp(-1000 * ratio, 1000 * ratio) / ratio
        return out * gate


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Transpose(nn.Module):
    def __init__(self, dims: tuple[int, int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.dims)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 1.0, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        _dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            _dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LYNXNet2(nn.Module):
    def __init__(
        self,
        in_dims: int,
        dim_cond: int,
        n_layers: int = 6,
        n_chans: int = 512,
        n_dilates: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        )
        self.residual_layers = nn.ModuleList(
            [LYNXNet2Block(n_chans, n_dilates, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec: torch.Tensor, diffusion_step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = spec
        if x.dim() == 4:
            x = x[:, 0]

        x = self.input_projection(x.transpose(1, 2))
        x = x + self.conditioner_projection(cond.transpose(1, 2))
        x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        for layer in self.residual_layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output_projection(x).transpose(1, 2)

        return x[:, None] if spec.dim() == 4 else x
