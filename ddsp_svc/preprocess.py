import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ddsp_svc.config import SvcConfig, save_config
from ddsp_svc.device import detect_device
from ddsp_svc.extractors.encoder import UnitsEncoder
from ddsp_svc.extractors.f0 import F0Extractor
from ddsp_svc.extractors.volume import VolumeExtractor
from ddsp_svc.models.vocoder import Vocoder


def collect_wavs(root: Path) -> list[tuple[Path, str]]:
    result = []
    for p in root.rglob("*.wav"):
        rel = p.relative_to(root)
        result.append((p, str(rel.with_suffix(""))))
    return sorted(result, key=lambda x: x[1])


def split_train_val(files: list[tuple[Path, str]], train_root: Path, val_root: Path, val_ratio: float = 0.05):
    speaker_groups: dict[str, list[tuple[Path, str]]] = {}
    for fp, rel in files:
        speaker = rel.split("/")[0] if "/" in rel else "speaker0"
        speaker_groups.setdefault(speaker, []).append((fp, rel))

    for speaker, group in speaker_groups.items():
        sorted_group = sorted(group, key=lambda x: -x[0].stat().st_size)
        n_val = max(1, int(len(sorted_group) * val_ratio))
        for fp, rel in sorted_group[:n_val]:
            dst = val_root / f"{rel}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, dst)
        for fp, rel in sorted_group[n_val:]:
            dst = train_root / f"{rel}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, dst)

    speakers = sorted(speaker_groups.keys())
    logger.info(f"Split {len(files)} files: speakers={speakers}")
    return speakers


def run_preprocessing(config: SvcConfig, progress_callback=None):
    device_info = detect_device(config.device)
    device = str(device_info)
    logger.info(f"Using device: {device_info}")

    raw_dir = Path(config.data.raw_path)

    if not raw_dir.exists() or not any(raw_dir.rglob("*.wav")):
        raise FileNotFoundError(f"No .wav files found in {raw_dir}")

    files = collect_wavs(raw_dir)
    logger.info(f"Found {len(files)} wav files in {raw_dir}")

    train_audio_root = Path(config.data.train_path) / "audio"
    val_audio_root = Path(config.data.valid_path) / "audio"

    if train_audio_root.exists():
        shutil.rmtree(train_audio_root)
    if val_audio_root.exists():
        shutil.rmtree(val_audio_root)

    speakers = split_train_val(files, train_audio_root, val_audio_root, val_ratio=0.05)
    config.spks = speakers

    f0_ext = F0Extractor(config.data.f0_extractor, config.data.sampling_rate,
                         config.data.block_size, config.data.f0_min, config.data.f0_max, device)
    vol_ext = VolumeExtractor(config.data.block_size)
    units_enc = UnitsEncoder(config.data.encoder, config.data.encoder_ckpt,
                              config.data.encoder_sample_rate, config.data.encoder_hop_size, device)
    vocoder = Vocoder(config.vocoder.type, config.vocoder.ckpt, device=device)

    pitch_aug_dict = {}

    for split_name in ("train", "val"):
        audio_root = train_audio_root if split_name == "train" else val_audio_root
        if not audio_root.exists():
            continue

        wav_files = sorted(audio_root.rglob("*.wav"))
        if not wav_files:
            continue

        feat_root = Path(config.data.train_path if split_name == "train" else config.data.valid_path)

        for wav_path in tqdm(wav_files, desc=f"Processing {split_name}"):
            rel = wav_path.relative_to(audio_root).with_suffix("")
            name = str(rel)

            try:
                audio, sr = librosa.load(str(wav_path), sr=config.data.sampling_rate, mono=True)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
            except Exception as e:
                logger.error(f"Failed to load {wav_path}: {e}")
                continue

            f0 = f0_ext.extract(audio)
            volume = vol_ext.extract(audio)

            aug_shift = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
            pitch_aug_dict[name] = int(aug_shift)

            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            units = units_enc.encode(audio_t, sr, config.data.block_size)
            units_np = units.squeeze(0).cpu().numpy()

            mel = vocoder.extract(audio_t, sr).squeeze(0).cpu().numpy()

            audio_aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=aug_shift, res_type="kaiser_fast")
            audio_aug_t = torch.from_numpy(audio_aug).float().unsqueeze(0).to(device)
            mel_aug = vocoder.extract(audio_aug_t, sr).squeeze(0).cpu().numpy()

            volume_aug = volume * (2 ** (aug_shift / 12)) ** 0.5

            for sub_dir in ("f0", "volume", "mel", "aug_mel", "aug_vol", "units"):
                d = feat_root / sub_dir / rel.parent
                d.mkdir(parents=True, exist_ok=True)

            np.save(feat_root / "f0" / f"{name}.npy", f0)
            np.save(feat_root / "volume" / f"{name}.npy", volume)
            np.save(feat_root / "mel" / f"{name}.npy", mel)
            np.save(feat_root / "aug_mel" / f"{name}.npy", mel_aug)
            np.save(feat_root / "aug_vol" / f"{name}.npy", volume_aug)
            np.save(feat_root / "units" / f"{name}.npy", units_np)

    np.save(Path(config.data.train_path) / "pitch_aug_dict.npy", pitch_aug_dict)
    np.save(Path(config.data.valid_path) / "pitch_aug_dict.npy", pitch_aug_dict)
    config.model.n_spk = len(speakers)

    exp_dir = Path(config.env.expdir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, exp_dir / "config.yaml")
    save_config(config, "configs/default.yaml")

    logger.success(f"Preprocessing complete! {len(speakers)} speaker(s): {speakers}")
