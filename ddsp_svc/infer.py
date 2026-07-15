import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger

from ddsp_svc.config import SvcConfig
from ddsp_svc.device import detect_device, empty_cache
from ddsp_svc.extractors.encoder import UnitsEncoder
from ddsp_svc.extractors.f0 import F0Extractor
from ddsp_svc.extractors.volume import VolumeExtractor
from ddsp_svc.models.unit2wav import Unit2Wav
from ddsp_svc.models.vocoder import Vocoder


def cross_fade(result: np.ndarray, seg: np.ndarray, fade_len: int) -> np.ndarray:
    fade_len = min(fade_len, len(result), len(seg))
    if fade_len <= 0:
        return np.concatenate([result, seg])
    fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
    result[-fade_len:] = result[-fade_len:] * fade_out + seg[:fade_len] * fade_in
    return np.concatenate([result[:-fade_len], seg])


def split(audio: np.ndarray, sample_rate: int, hop_size: int, db_thresh: float = -40., min_len: float = 3.0) -> list:
    import parselmouth
    s = parselmouth.Sound(audio, sample_rate)
    pitch = s.to_pitch_ac(time_step=hop_size / sample_rate, voicing_threshold=0.3,
                          pitch_floor=55, pitch_ceiling=1100)
    f0 = pitch.selected_array['frequency']
    voiced = (f0 > 0).astype(float)

    mean_volume = np.mean(np.abs(audio))
    segments = []
    start = 0
    min_frames = int(min_len * sample_rate / hop_size)
    for i in range(1, len(voiced)):
        if voiced[i] == 0 and voiced[i - 1] == 1:
            if i - start >= min_frames:
                seg_audio = audio[start * hop_size: i * hop_size]
                if np.mean(np.abs(seg_audio)) > mean_volume * 0.1:
                    segments.append((start, seg_audio))
            start = i
    if len(audio) - start * hop_size > min_len * sample_rate:
        segments.append((start, audio[start * hop_size:]))
    return segments or [(0, audio)]


class InferencePipeline:
    def __init__(self, config: SvcConfig):
        self.config = config
        self.device_info = detect_device(config.device)
        self.device = str(self.device_info)
        logger.info(f"Inference device: {self.device_info}")

    def load_model(self, checkpoint_path: str):
        ckpt_path = Path(checkpoint_path)
        config_path = ckpt_path.parent / "config.yaml"
        if config_path.exists():
            from ddsp_svc.config import load_config
            cfg = load_config(config_path)
        else:
            cfg = self.config

        self.vocoder = Vocoder(cfg.vocoder.type, cfg.vocoder.ckpt, device=self.device)
        self.model = Unit2Wav(
            cfg.data.sampling_rate, cfg.data.block_size, cfg.model.win_length,
            cfg.data.encoder_out_channels, cfg.model.n_spk,
            cfg.model.use_norm, cfg.model.use_attention, cfg.model.use_pitch_aug,
            self.vocoder.dimension,
            cfg.model.n_aux_layers, cfg.model.n_aux_chans,
            cfg.model.n_layers, cfg.model.n_chans,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.cfg = cfg

        self.units_encoder = UnitsEncoder(
            cfg.data.encoder, cfg.data.encoder_ckpt,
            cfg.data.encoder_sample_rate, cfg.data.encoder_hop_size,
            device=self.device,
        )
        self.spks = getattr(cfg, "spks", ["speaker0"])
        logger.info(f"Model loaded, speakers: {self.spks}")

    @torch.no_grad()
    def infer(self, audio_path: str, speaker: str | int = "speaker0",
              keychange: int = 0, infer_step: int = 50, method: str = "euler",
              t_start: float = 0.0, threshold: float = -45.0,
              formant_shift: float = 0.0) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=self.cfg.data.sampling_rate, mono=True)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        hop_size = int(self.cfg.data.block_size * sr / self.cfg.data.sampling_rate)

        f0_ext = F0Extractor(self.cfg.data.f0_extractor, sr, hop_size,
                              self.cfg.data.f0_min, self.cfg.data.f0_max, self.device)
        f0 = f0_ext.extract(audio)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * (2 ** (keychange / 12))

        vol_ext = VolumeExtractor(hop_size)
        vol = vol_ext.extract(audio)
        mask = (vol > 10 ** (threshold / 20)).astype(float)
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([mask[n:n + 9].max() for n in range(len(mask) - 8)])
        from ddsp_svc.models.ddsp import upsample
        mask_t = upsample(torch.from_numpy(mask).float().to(self.device).unsqueeze(0).unsqueeze(-1),
                          self.cfg.data.block_size).squeeze(-1)
        vol_t = torch.from_numpy(vol).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sr, hop_size)
        n_enc_frames = units.size(1)

        spk_id = 1
        if isinstance(speaker, str) and speaker in self.spks:
            spk_id = self.spks.index(speaker) + 1
        elif isinstance(speaker, int):
            spk_id = speaker
        spk_id_t = torch.LongTensor([[spk_id]]).to(self.device)
        formant_t = torch.tensor([[formant_shift]]).float().to(self.device)

        segments = split(audio, sr, hop_size)
        logger.info(f"Split into {len(segments)} segments")
        result = np.zeros(0)
        current_len = 0

        for start_frame, seg_audio in segments:
            seg_start_sample = start_frame * self.cfg.data.block_size
            seg_end_sample = seg_start_sample + len(seg_audio)

            enc_start = int(seg_start_sample * self.cfg.data.encoder_sample_rate / sr / self.cfg.data.encoder_hop_size)
            enc_end = int(seg_end_sample * self.cfg.data.encoder_sample_rate / sr / self.cfg.data.encoder_hop_size)
            enc_start = max(0, enc_start)
            enc_end = min(n_enc_frames, enc_end)
            if enc_end <= enc_start:
                continue

            seg_units = units[:, enc_start:enc_end, :]
            n_f0 = min(seg_units.size(1), f0.size(1) - start_frame)
            if n_f0 <= 0:
                continue
            seg_units = seg_units[:, :n_f0, :]

            seg_f0 = f0[:, start_frame:start_frame + n_f0, :]
            seg_vol = vol_t[:, start_frame:start_frame + n_f0, :]
            seg_mask = mask_t[:, start_frame * self.cfg.data.block_size:
                              (start_frame + n_f0) * self.cfg.data.block_size]

            seg_out = self.model(
                seg_units, seg_f0, seg_vol,
                spk_id=spk_id_t, aug_shift=formant_t,
                vocoder=self.vocoder, infer=True, return_wav=True,
                infer_step=infer_step, method=method, t_start=t_start,
            )
            min_len = min(seg_out.size(-1), seg_mask.size(-1))
            seg_out = seg_out[..., :min_len] * seg_mask[..., :min_len]
            seg_np = seg_out.squeeze().cpu().numpy()

            silent = int(start_frame * self.cfg.data.block_size) - current_len
            if silent >= 0:
                result = np.append(result, np.zeros(silent))
                result = np.append(result, seg_np)
            else:
                result = cross_fade(result, seg_np, current_len + silent)
            current_len += silent + len(seg_np)

        empty_cache(self.device_info.type)

        in_rms = np.sqrt(np.mean(audio ** 2) + 1e-8))
        out_rms = np.sqrt(np.mean(result ** 2) + 1e-8)
        if out_rms > 0 and in_rms / out_rms < 100:
            result = result * (in_rms / out_rms)
        return result

    def infer_simple(self, audio_path: str, **kwargs) -> np.ndarray:
        return self.infer(audio_path, **kwargs)
