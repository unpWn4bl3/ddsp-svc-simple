import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
from transformers import HubertModel, HubertConfig


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class Audio2ContentVec768L12:
    def __init__(self, path: str, device: str = "cpu"):
        self.device = device
        self.hubert = HubertModelWithFinalProj(HubertConfig()).to(device)
        state = torch.load(path, map_location=device)
        consume_prefix_in_state_dict_if_present(state, "module.")
        self.hubert.load_state_dict(state)
        self.hubert.eval()

    @torch.no_grad()
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.hubert(audio)["last_hidden_state"]


class Audio2ContentVec768L12TTA2X:
    def __init__(self, path: str, device: str = "cpu"):
        self.device = device
        self.hubert = HubertModelWithFinalProj(HubertConfig()).to(device)
        state = torch.load(path, map_location=device)
        consume_prefix_in_state_dict_if_present(state, "module.")
        self.hubert.load_state_dict(state)
        self.hubert.eval()

    @torch.no_grad()
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.hubert(audio)["last_hidden_state"]
        audio_pad = F.pad(audio, (160, 0))
        feats2 = self.hubert(audio_pad)["last_hidden_state"]
        n = feats2.shape[1] - feats.shape[1]
        if n > 0:
            feats = F.pad(feats, (0, 0, 0, 1))
        feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
        feats_tta = feats_tta[:, 1:, :]
        if n > 0:
            feats_tta = feats_tta[:, :-1, :]
        return feats_tta


class UnitsEncoder:
    def __init__(self, encoder: str, encoder_ckpt: str,
                 encoder_sample_rate: int = 16000, encoder_hop_size: int = 320,
                 device: str = "cpu"):
        self.device = device
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size

        if encoder == "contentvec768l12":
            self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
        elif encoder == "contentvec768l12tta2x":
            self.model = Audio2ContentVec768L12TTA2X(encoder_ckpt, device=device)
        else:
            raise ValueError(f"Unknown encoder: {encoder}. Supported: contentvec768l12, contentvec768l12tta2x")

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: int, hop_size: int) -> torch.Tensor:
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key = str(sample_rate)
            if key not in self.resample_kernel:
                self.resample_kernel[key] = Resample(
                    sample_rate, self.encoder_sample_rate, lowpass_filter_width=128
                ).to(self.device)
            audio_res = self.resample_kernel[key](audio)

        if audio_res.size(-1) < 400:
            audio_res = F.pad(audio_res, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)

        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames, device=self.device)).long(),
                            max=units.size(1) - 1)
        return torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat(1, 1, units.size(-1)))
