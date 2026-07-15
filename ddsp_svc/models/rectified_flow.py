import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .lynxnet import LYNXNet2


class RectifiedFlow(nn.Module):
    def __init__(self, velocity_fn: nn.Module, out_dims: int = 128, spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    def reflow_loss(self, x_1: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond)
        weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
        return torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction="none"))

    def norm_spec(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def forward(self, condition: torch.Tensor, gt_spec: torch.Tensor | None = None,
                infer: bool = True, infer_step: int = 10, method: str = 'euler',
                t_start: float = 0.0, use_tqdm: bool = True):
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device
        if t_start < 0.0:
            t_start = 0.0

        if not infer:
            x_1 = self.norm_spec(gt_spec).transpose(1, 2)[:, None, :, :]
            t = t_start + (1.0 - t_start) * torch.rand(b, device=device)
            t = torch.clip(t, 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, t, cond=cond)

        shape = (b, 1, self.out_dims, cond.shape[2])
        if gt_spec is None:
            x = torch.randn(shape, device=device)
            t = torch.zeros(b, device=device)
            dt = 1.0 / infer_step
        else:
            norm_spec = self.norm_spec(gt_spec).transpose(1, 2)[:, None, :, :]
            x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
            t = torch.full((b,), t_start, device=device)
            dt = (1.0 - t_start) / infer_step

        sample_fn = self.sample_euler if method == 'euler' else self.sample_rk4
        loop = tqdm(range(infer_step), desc='sample') if use_tqdm else range(infer_step)
        for _ in loop:
            x, t = sample_fn(x, t, dt, cond)

        return self.denorm_spec(x.squeeze(1).transpose(1, 2))

    def sample_euler(self, x: torch.Tensor, t: torch.Tensor, dt: float, cond: torch.Tensor):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        return x, t + dt

    def sample_rk4(self, x: torch.Tensor, t: torch.Tensor, dt: float, cond: torch.Tensor):
        k1 = self.velocity_fn(x, 1000 * t, cond)
        k2 = self.velocity_fn(x + 0.5 * dt * k1, 1000 * (t + 0.5 * dt), cond)
        k3 = self.velocity_fn(x + 0.5 * dt * k2, 1000 * (t + 0.5 * dt), cond)
        k4 = self.velocity_fn(x + dt * k3, 1000 * (t + dt), cond)
        x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x, t + dt
