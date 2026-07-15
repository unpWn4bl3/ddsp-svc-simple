import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from torchaudio.transforms import Resample


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class STFT:
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025,
                 clip_val=1e-5):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        key = str(self.fmax) + '_' + str(y.device)
        if key not in self.mel_basis:
            self.mel_basis[key] = torch.from_numpy(
                librosa_mel_fn(sr=self.target_sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
            ).float().to(y.device)

        k_key = str(keyshift) + '_' + str(y.device)
        if k_key not in self.hann_window:
            self.hann_window[k_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        mode = 'reflect' if pad_right < y.size(-1) else 'constant'
        y = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new,
                          window=self.hann_window[k_key], center=center, pad_mode='reflect',
                          normalized=False, onesided=True, return_complex=True)
        spec = spec.abs()
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            if spec.size(1) < size:
                spec = F.pad(spec, (0, 0, 0, size - spec.size(1)))
            spec = spec[:, :size, :] * self.win_size / win_size_new
        spec = torch.matmul(self.mel_basis[key], spec)
        spec = torch.nan_to_num(spec, nan=0.0, posinf=float(self.clip_val), neginf=float(self.clip_val))
        return torch.log(torch.clamp(spec, min=self.clip_val) * 1)


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d[0],
                               padding=get_padding(kernel_size, d[0])))
            for d in [(dilation[0],), (dilation[1],), (dilation[2],)]
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class SineGen(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc, (0, 0, 1, -1))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        return torch.sin(2 * np.pi * rad)

    @torch.no_grad()
    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = self._f02uv(f0)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        return self.l_tanh(self.l_linear(sine_wavs))


class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2:]))
            self.upp = int(np.prod(h.upsample_rates[:2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock1
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(ch * 2, ch, k, u, padding=(k - u) // 2)))
            for j, (ks, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, ks, d))
            if not h.mini_nsf:
                if i + 1 < len(h.upsample_rates):
                    stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=stride_f0 * 2, stride=stride_f0,
                                                   padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0):
        if self.h.mini_nsf:
            har_source = self._fast_sine_gen(f0)
        else:
            har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if not self.h.mini_nsf:
                x_source = self.noise_convs[i](har_source)
                x = x + x_source
            elif i == 1:
                x_source = self.source_conv(har_source)
                x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                out = self.resblocks[idx](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def _fast_sine_gen(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        return torch.sin(2 * np.pi * rad)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def load_generator(model_path: str, device: str = "cpu"):
    model_dir = os.path.split(model_path)[0]
    config_file = os.path.join(os.path.split(model_dir)[0], "config.json")
    if not os.path.exists(config_file):
        config_file = os.path.join(model_dir, "config.json")
    with open(config_file) as f:
        h = AttrDict(json.loads(f.read()))
    generator = Generator(h).to(device)
    cp = torch.load(model_path, map_location=device)
    sd = cp.get("generator", cp.get("state_dict", cp))
    sd = {k.replace("generator.", ""): v for k, v in sd.items() if k.startswith("generator.") or not k.startswith("discriminator.")}
    generator.load_state_dict(sd)
    generator.eval()
    generator.remove_weight_norm()
    return generator, h


class NsfHifiGAN(nn.Module):
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.generator = None
        self._config = None
        self.stft = None

    def _lazy_init(self):
        if self.generator is None:
            self.generator, self._config = load_generator(self.model_path, self.device)
            self.stft = STFT(
                self._config.sampling_rate, self._config.num_mels,
                self._config.n_fft, self._config.win_size,
                self._config.hop_size, self._config.fmin, self._config.fmax,
            )

    def sample_rate(self):
        self._lazy_init()
        return self._config.sampling_rate

    def hop_size(self):
        self._lazy_init()
        return self._config.hop_size

    def dimension(self):
        self._lazy_init()
        return self._config.num_mels

    def extract(self, audio: torch.Tensor, keyshift: float = 0) -> torch.Tensor:
        self._lazy_init()
        return self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        self._lazy_init()
        return self.generator(mel.transpose(1, 2), f0)


class NsfHifiGANLog10(NsfHifiGAN):
    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        self._lazy_init()
        return self.generator(0.434294 * mel.transpose(1, 2), f0)


class Vocoder(nn.Module):
    def __init__(self, vocoder_type: str, vocoder_ckpt: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.resample_kernel = {}
        if vocoder_type == "nsf-hifigan":
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device=device)
        elif vocoder_type == "nsf-hifigan-log10":
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device=device)
        else:
            raise ValueError(f"Unknown vocoder: {vocoder_type}")
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio: torch.Tensor, sample_rate: int = 0, keyshift: float = 0) -> torch.Tensor:
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key = str(sample_rate)
            if key not in self.resample_kernel:
                self.resample_kernel[key] = Resample(
                    sample_rate, self.vocoder_sample_rate, lowpass_filter_width=128
                ).to(self.device)
            audio_res = self.resample_kernel[key](audio)
        return self.vocoder.extract(audio_res, keyshift=keyshift)

    def infer(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        f0 = f0[:, :mel.size(1)].squeeze(-1)
        return self.vocoder(mel, f0)
