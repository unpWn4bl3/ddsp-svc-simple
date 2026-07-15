import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ──────────────────────────────────────────────
# Core signal processing
# ──────────────────────────────────────────────

def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    return F.interpolate(signal, size=signal.shape[-1] * factor, mode='linear', align_corners=False)


def MaskedAvgPool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)
    summed = F.conv1d(masked_x, kernel, stride=1, padding=0, groups=x.size(1))
    valid = F.conv1d(mask.float(), kernel, stride=1, padding=0, groups=x.size(1)).clamp(min=1)
    return (summed / valid).squeeze(1)


def MedianPool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = F.pad(x.unsqueeze(1), ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect").squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    convolved = ir_size + frame_size - 1
    if power_of_2:
        return int(2 ** np.ceil(np.log2(convolved)))
    return convolved


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    signal_size = signal.shape[-1]
    kernel_size = kernel.shape[-1]
    fft_size = get_fft_size(signal_size, kernel_size)
    signal_pad = F.pad(signal, (0, fft_size - signal_size))
    kernel_pad = F.pad(kernel, (0, fft_size - kernel_size))
    signal_fd = torch.fft.rfft(signal_pad)
    kernel_fd = torch.fft.rfft(kernel_pad)
    output = torch.fft.irfft(signal_fd * kernel_fd, fft_size)
    return output[..., :signal_size + kernel_size - 1]


def frequency_filter(audio: torch.Tensor, magnitudes: torch.Tensor, n_fft: int) -> torch.Tensor:
    audio_fd = torch.fft.rfft(audio, n_fft)
    magnitude_fd = torch.fft.rfft(magnitudes, n_fft)
    return torch.fft.irfft(audio_fd * magnitude_fd, n_fft)


def remove_above_fmax(amplitudes: torch.Tensor, pitch: torch.Tensor, n_freq: int, fmax: float,
                       sample_rate: int) -> torch.Tensor:
    n_harmonic = n_freq - 1
    f0_per_frame = pitch[:, :, :-1].mean(dim=-1)
    harm_freq = torch.arange(1, n_harmonic + 1, device=amplitudes.device).float() * f0_per_frame.unsqueeze(-1)
    above_fmax = harm_freq > fmax
    amplitudes[:, :, :-1] *= (~above_fmax).float()
    return amplitudes


# ──────────────────────────────────────────────
# Conformer Encoder (from Diffusion-SVC)
# ──────────────────────────────────────────────

class _SwiGLU(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            o_min, o_max = torch.aminmax(out.detach())
            g_min, g_max = torch.aminmax(gate.detach())
            max_abs = torch.max(-o_min, o_max).float() * torch.max(-g_min, g_max).float()
            if max_abs > 1000:
                ratio = (1000 / max_abs).half()
                gate *= ratio
                return (out * gate).clamp(-1000 * ratio, 1000 * ratio) / ratio
        return out * gate


class _Transpose(nn.Module):
    def __init__(self, dims: tuple[int, int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.dims)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 1.0, kernel_size: int = 31,
                 dropout: float = 0., use_norm: bool = False):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(dim) if use_norm else nn.Identity(),
            _Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            _Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            _SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            _SwiGLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, num_heads: int = 8, use_norm: bool = False,
                 conv_only: bool = False, conv_dropout: float = 0., atten_dropout: float = 0.1):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model, use_norm=use_norm, dropout=conv_dropout)
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(
                d_model=dim_model, nhead=num_heads, dim_feedforward=dim_model * 4,
                dropout=atten_dropout, activation='gelu', batch_first=True,
            )
            self.norm = nn.LayerNorm(dim_model)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask)
        x = x + self.conformer(x)
        return x


class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, dim_model: int,
                 use_norm: bool = False, conv_only: bool = False,
                 conv_dropout: float = 0., atten_dropout: float = 0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


# ──────────────────────────────────────────────
# Unit2Control
# ──────────────────────────────────────────────

def split_to_dict(tensor: torch.Tensor, splits: dict[str, int]) -> dict[str, torch.Tensor]:
    tensors = torch.split(tensor, list(splits.values()), dim=-1)
    return dict(zip(splits.keys(), tensors))


class Unit2Control(nn.Module):
    def __init__(self, input_channel: int, block_size: int, n_spk: int,
                 output_splits: dict[str, int], num_layers: int = 3, dim_model: int = 256,
                 use_norm: bool = False, use_attention: bool = False, use_pitch_aug: bool = False):
        super().__init__()
        self.output_splits = output_splits
        self.n_spk = n_spk
        self.f0_embed = nn.Linear(1, dim_model)
        self.phase_embed = nn.Linear(1, dim_model)
        self.volume_embed = nn.Linear(1, dim_model)
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, dim_model)
        self.aug_shift_embed = nn.Linear(1, dim_model, bias=False) if use_pitch_aug else None

        self.stack = nn.Sequential(
            weight_norm(nn.Conv1d(input_channel, 512, 3, 1, 1)),
            nn.PReLU(512),
            weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)),
        )
        self.stack2 = nn.Sequential(
            weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)),
            nn.PReLU(512),
            weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)),
        )
        self.decoder = ConformerNaiveEncoder(
            num_layers=num_layers, num_heads=8, dim_model=dim_model,
            use_norm=use_norm, conv_only=not use_attention,
            conv_dropout=0, atten_dropout=0.1,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.n_out = sum(output_splits.values())
        self.dense_out = weight_norm(nn.Linear(dim_model, self.n_out))

    def forward(self, units: torch.Tensor, source: torch.Tensor, noise: torch.Tensor, volume: torch.Tensor,
                spk_id: torch.Tensor | None = None, spk_mix_dict: dict | None = None,
                aug_shift: torch.Tensor | None = None):
        exciter = torch.cat((source, noise), dim=-1).transpose(1, 2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    sid = torch.LongTensor([[k]]).to(units.device)
                    x = x + v * self.spk_embed(sid - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        return split_to_dict(e, self.output_splits), x


# ──────────────────────────────────────────────
# CombSubSuperFast (DDPS synthesizer)
# ──────────────────────────────────────────────

class CombSubSuperFast(nn.Module):
    def __init__(self, sampling_rate: int, block_size: int, win_length: int,
                 n_unit: int = 256, n_spk: int = 1, num_layers: int = 3, dim_model: int = 256,
                 use_norm: bool = False, use_attention: bool = False, use_pitch_aug: bool = False):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))

        split_map = {
            'harmonic_magnitude': win_length // 2 + 1,
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': win_length // 2 + 1,
            'noise_phase': win_length // 2 + 1,
        }
        self.unit2ctrl = Unit2Control(
            n_unit, block_size, n_spk, split_map,
            num_layers=num_layers, dim_model=dim_model,
            use_norm=use_norm, use_attention=use_attention, use_pitch_aug=use_pitch_aug,
        )

    def fast_source_gen(self, f0_frames: torch.Tensor) -> torch.Tensor:
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        return torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)

    def forward(self, units_frames: torch.Tensor, f0_frames: torch.Tensor, volume_frames: torch.Tensor,
                spk_id: torch.Tensor | None = None, spk_mix_dict: dict | None = None,
                aug_shift: torch.Tensor | None = None, infer: bool = True):
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, int(self.block_size), int(self.block_size))
        noise = torch.randn_like(combtooth)
        noise_frames = noise.unfold(1, int(self.block_size), int(self.block_size))
        ctrls, hidden = self.unit2ctrl(units_frames, combtooth_frames, noise_frames, volume_frames,
                                        spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)

        src_filter = torch.exp(ctrls['harmonic_magnitude'].clamp(-10, 10) + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)
        noise_filter = torch.exp(ctrls['noise_magnitude'].clamp(-10, 10) + 1.j * np.pi * ctrls['noise_phase']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)

        pad_mode = 'reflect' if combtooth.shape[-1] > self.win_length // 2 else 'constant'
        combtooth_stft = torch.stft(combtooth, n_fft=int(self.win_length),
                                     win_length=int(self.win_length), hop_length=int(self.block_size),
                                     window=self.window, center=True, return_complex=True, pad_mode=pad_mode)
        noise_stft = torch.stft(noise, n_fft=int(self.win_length),
                                 win_length=int(self.win_length), hop_length=int(self.block_size),
                                 window=self.window, center=True, return_complex=True, pad_mode=pad_mode)

        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)
        signal = torch.istft(signal_stft, n_fft=int(self.win_length),
                              win_length=int(self.win_length), hop_length=int(self.block_size),
                              window=self.window, center=True)
        return signal, hidden


# ──────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────

class SSSLoss(nn.Module):
    def __init__(self, n_fft: int = 111, alpha: float = 1.0, overlap: float = 0, eps: float = 1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        from torchaudio.transforms import Spectrogram
        self.spec = Spectrogram(n_fft=n_fft, hop_length=self.hop_length, power=1, normalized=True, center=False)

    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        S_true = self.spec(x_true) + self.eps
        S_pred = self.spec(x_pred) + self.eps
        converge = torch.mean(torch.linalg.norm(S_true - S_pred, dim=(1, 2)) / torch.linalg.norm(S_true + S_pred, dim=(1, 2)))
        log_term = F.l1_loss(S_true.log(), S_pred.log())
        return converge + self.alpha * log_term


class RSSLoss(nn.Module):
    def __init__(self, fft_min: int, fft_max: int, n_scale: int, alpha: float = 1.0, overlap: float = 0,
                 eps: float = 1e-7, device: str = 'cuda'):
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.n_scale = n_scale
        self.lossdict = {n: SSSLoss(n, alpha, overlap, eps).to(device) for n in range(fft_min, fft_max)}

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        value = 0.
        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            value += self.lossdict[int(n_fft)](x_true, x_pred)
        return value / self.n_scale
