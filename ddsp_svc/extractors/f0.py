import numpy as np
import parselmouth
import torch
from torchaudio.transforms import Resample

from ddsp_svc.models.ddsp import MedianPool1d, MaskedAvgPool1d

CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}


class F0Extractor:
    def __init__(self, method: str, sample_rate: int = 44100, hop_size: int = 512,
                 f0_min: int = 65, f0_max: int = 800, device: str = "cpu"):
        self.method = method
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device

    def extract_all(self, audio: np.ndarray, uv_interp: bool = True,
                    silence_front: float = 0) -> tuple[np.ndarray, np.ndarray]:
        n_frames = int(len(audio) // self.hop_size) + 1
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence * self.sample_rate)):]

        f0_raw = self._extract(audio, n_frames, start_frame)[:n_frames]
        uv = f0_raw == 0
        f0 = f0_raw.copy()
        if uv_interp and uv.any() and (~uv).any():
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0, uv

    def extract(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        f0, _ = self.extract_all(audio, **kwargs)
        return f0

    def _extract(self, audio: np.ndarray, n_frames: int, start_frame: int) -> np.ndarray:
        m, sr, hs = self.method, self.sample_rate, self.hop_size

        if m == "parselmouth":
            l_pad = int(np.ceil(1.5 / self.f0_min * sr))
            r_pad = int(hs * ((len(audio) - 1) // hs + 1) - len(audio) + l_pad + 1)
            s = parselmouth.Sound(np.pad(audio, (l_pad, r_pad)), sr).to_pitch_ac(
                time_step=hs / sr, voicing_threshold=0.6,
                pitch_floor=self.f0_min, pitch_ceiling=self.f0_max)
            f0 = s.selected_array['frequency']
            return np.pad(f0, (start_frame, max(0, n_frames - len(f0) - start_frame)))

        if m == "crepe":
            return self._crepe(audio, n_frames, start_frame, sr, hs)

        if m == "rmvpe":
            return self._rmvpe(audio, n_frames, start_frame, sr, hs)

        raise ValueError(f"Unknown F0 extractor: {m}, choose from: parselmouth, crepe, rmvpe")

    def _crepe(self, audio: np.ndarray, n_frames: int, start_frame: int,
               sr: int, hs: int) -> np.ndarray:
        import torchcrepe

        key = str(sr)
        if key not in CREPE_RESAMPLE_KERNEL:
            CREPE_RESAMPLE_KERNEL[key] = Resample(sr, 16000, lowpass_filter_width=128).to(self.device)
        wav16k = CREPE_RESAMPLE_KERNEL[key](torch.FloatTensor(audio).unsqueeze(0).to(self.device))

        f0, pd = torchcrepe.predict(wav16k, 16000, 80, self.f0_min, self.f0_max,
                                     pad=True, model='full', batch_size=512,
                                     device=self.device, return_periodicity=True)
        pd = MedianPool1d(pd, 4)
        f0 = torchcrepe.threshold.At(0.05)(f0, pd)
        f0 = MaskedAvgPool1d(f0, 4).squeeze(0).cpu().numpy()
        ratio = hs / sr / 0.005
        idx = [min(int(np.round(n * ratio)), len(f0) - 1) for n in range(n_frames - start_frame)]
        return np.pad(f0[idx], (start_frame, 0))

    def _rmvpe(self, audio: np.ndarray, n_frames: int, start_frame: int,
               sr: int, hs: int) -> np.ndarray:
        if 'rmvpe' not in F0_KERNEL:
            from ddsp_svc.extractors.rmvpe import RMVPE
            F0_KERNEL['rmvpe'] = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
        f0 = F0_KERNEL['rmvpe'].infer_from_audio(audio, sr, device=self.device, thred=0.03, use_viterbi=False)
        uv = f0 == 0
        if (~uv).any():
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        orig = 0.01 * np.arange(len(f0))
        target = hs / sr * np.arange(n_frames - start_frame)
        f0 = np.interp(target, orig, f0)
        uv = np.interp(target, orig, uv.astype(float)) > 0.5
        f0[uv] = 0
        return np.pad(f0, (start_frame, 0))
