import concurrent.futures
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_npy_shape(path: str) -> tuple:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(f)[0]
        elif version == (2, 0):
            return np.lib.format.read_array_header_2_0(f)[0]
        raise ValueError("Unsupported .npy version")


class AudioDataset(Dataset):
    def __init__(self, path_root: str, waveform_sec: float, hop_size: int, sample_rate: int,
                 load_all_data: bool = True, whole_audio: bool = False,
                 extensions: list[str] | None = None, n_spk: int = 1,
                 device: str = "cpu", fp16: bool = False, use_aug: bool = False,
                 spk_map: dict[str, int] | None = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.path_root = Path(path_root)
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.n_spk = n_spk

        audio_dir = self.path_root / "audio"
        extensions = extensions or ["wav"]
        self.paths = sorted([
            str(p.relative_to(audio_dir)) for p in audio_dir.rglob("*")
            if p.suffix[1:] in extensions
        ])

        self.pitch_aug_dict = np.load(
            self.path_root / "pitch_aug_dict.npy", allow_pickle=True
        ).item()

        self.spk_map = spk_map or {}
        self.data_buffer = {}
        self._load_data(load_all_data, device, fp16)

    def _load_data(self, load_all: bool, device: str, fp16: bool):
        def load_one(name_ext: str):
            name = os.path.splitext(name_ext)[0]
            f0 = torch.from_numpy(np.load(self.path_root / "f0" / f"{name}.npy")).float().unsqueeze(-1).to(device)
            volume = torch.from_numpy(np.load(self.path_root / "volume" / f"{name}.npy")).float().unsqueeze(-1).to(device)
            aug_vol = torch.from_numpy(np.load(self.path_root / "aug_vol" / f"{name}.npy")).float().unsqueeze(-1).to(device)

            spk_name = name.split("/")[0] if "/" in name else "speaker0"
            spk_id = self.spk_map.get(spk_name, 1)
            spk_id = torch.LongTensor([spk_id]).to(device)

            mel_len = get_npy_shape(str(self.path_root / "mel" / f"{name}.npy"))[0]
            aug_mel_len = get_npy_shape(str(self.path_root / "aug_mel" / f"{name}.npy"))[0]
            units_len = get_npy_shape(str(self.path_root / "units" / f"{name}.npy"))[0]
            frame_len = min(mel_len, aug_mel_len, units_len, len(f0), len(volume), len(aug_vol))

            if load_all:
                mel = torch.from_numpy(np.load(self.path_root / "mel" / f"{name}.npy")).to(device)
                aug_mel = torch.from_numpy(np.load(self.path_root / "aug_mel" / f"{name}.npy")).to(device)
                units = torch.from_numpy(np.load(self.path_root / "units" / f"{name}.npy")).to(device)
                if fp16:
                    mel, aug_mel, units = mel.half(), aug_mel.half(), units.half()
                data = dict(mel=mel, aug_mel=aug_mel, units=units, f0=f0,
                            volume=volume, aug_vol=aug_vol, spk_id=spk_id, frame_len=frame_len)
            else:
                data = dict(f0=f0, volume=volume, aug_vol=aug_vol, spk_id=spk_id, frame_len=frame_len)
            return name_ext, data

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as ex:
            futures = {ex.submit(load_one, p): p for p in self.paths}
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(self.paths), desc="Loading data"):
                k, v = f.result()
                self.data_buffer[k] = v

    def __getitem__(self, idx: int):
        name_ext = self.paths[idx]
        buf = self.data_buffer[name_ext]
        if buf["frame_len"] < self.crop_len:
            return self.__getitem__((idx + 1) % len(self.paths))
        return self._get_item(name_ext, buf)

    def _get_item(self, name_ext: str, buf: dict) -> dict:
        name = os.path.splitext(name_ext)[0]
        start = 0 if self.whole_audio else torch.randint(0, buf["frame_len"] - self.crop_len + 1, ()).item()
        length = buf["frame_len"] if self.whole_audio else self.crop_len
        aug = self.use_aug and torch.rand(1).item() > 0.5

        mel_key = "aug_mel" if aug else "mel"
        mel = buf.get(mel_key)
        if mel is None:
            mel = torch.from_numpy(np.load(self.path_root / mel_key / f"{name}.npy", mmap_mode='r')[start:start + length].copy()).float()
        else:
            mel = mel[start:start + length]

        units = buf.get("units")
        if units is None:
            units = torch.from_numpy(np.load(self.path_root / "units" / f"{name}.npy", mmap_mode='r')[start:start + length].copy()).float()
        else:
            units = units[start:start + length]

        aug_shift = self.pitch_aug_dict.get(name_ext, self.pitch_aug_dict.get(name + ".wav", 0)) if aug else 0
        f0 = (2 ** (aug_shift / 12)) * buf["f0"][start:start + length]
        vol_key = "aug_vol" if aug else "volume"
        volume = buf[vol_key][start:start + length]

        return dict(
            mel=mel, f0=f0, volume=volume, units=units,
            spk_id=buf["spk_id"],
            aug_shift=torch.tensor([[aug_shift]]).float(),
            name=name, name_ext=name_ext,
        )

    def __len__(self):
        return len(self.paths)


def get_data_loaders(config, whole_audio: bool = False):
    spk_map = {s: i + 1 for i, s in enumerate(config.spks)} if config.spks else {"speaker0": 1}
    train = AudioDataset(
        config.data.train_path, config.data.duration, config.data.block_size,
        config.data.sampling_rate, config.train.cache_all_data, whole_audio,
        extensions=config.data.extensions, n_spk=config.model.n_spk,
        device=config.train.cache_device, fp16=config.train.cache_fp16, use_aug=True,
        spk_map=spk_map,
    )
    loader_train = DataLoader(
        train, batch_size=1 if whole_audio else config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers if config.train.cache_device == "cpu" else 0,
        persistent_workers=config.train.num_workers > 0 and config.train.cache_device == "cpu",
        pin_memory=config.train.cache_device == "cpu",
    )
    valid = AudioDataset(
        config.data.valid_path, config.data.duration, config.data.block_size,
        config.data.sampling_rate, config.train.cache_all_data, whole_audio=True,
        extensions=config.data.extensions, n_spk=config.model.n_spk,
        device=config.train.cache_device, fp16=config.train.cache_fp16,
        spk_map=spk_map,
    )
    loader_valid = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)
    return loader_train, loader_valid
