from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    sampling_rate: int = 44100
    block_size: int = 512
    f0_extractor: Literal["parselmouth", "dio", "harvest", "crepe", "rmvpe"] = "rmvpe"
    f0_min: int = 65
    f0_max: int = 800
    encoder: Literal[
        "contentvec768l12tta2x", "contentvec768l12"
    ] = "contentvec768l12tta2x"
    encoder_ckpt: str = "pretrain/contentvec/pytorch_model.bin"
    encoder_sample_rate: int = 16000
    encoder_hop_size: int = 160
    encoder_out_channels: int = 768
    volume_smooth_size: int = 1024
    duration: float = 2.0
    raw_path: str = "data/raw"
    train_path: str = "data/train"
    valid_path: str = "data/val"
    extensions: list[str] = ["wav"]


class ModelConfig(BaseModel):
    type: Literal["RectifiedFlow"] = "RectifiedFlow"
    win_length: int = 2048
    n_layers: int = 6
    n_chans: int = 1024
    n_aux_layers: int = 6
    n_aux_chans: int = 512
    n_spk: int = 1
    t_start: float = 0.0
    use_norm: bool = True
    use_attention: bool = False
    use_pitch_aug: bool = True


class VocoderConfig(BaseModel):
    type: Literal["nsf-hifigan", "nsf-hifigan-log10"] = "nsf-hifigan"
    ckpt: str = "pretrain/nsf_hifigan/model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt"


class InferConfig(BaseModel):
    infer_step: int = 50
    method: Literal["euler", "rk4"] = "euler"


class TrainConfig(BaseModel):
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = 0.0005
    epochs: int = 100000
    num_workers: int = Field(default=0, ge=0)
    amp_dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    cache_all_data: bool = False
    cache_device: Literal["cpu", "cuda"] = "cpu"
    cache_fp16: bool = True
    interval_log: int = 1
    interval_val: int = 500
    interval_force_save: int = 10000
    decay_step: int = 4000
    gamma: float = 0.9
    weight_decay: float = 0.1
    lambda_ddsp: float = 1.0
    save_opt: bool = False


class EnvConfig(BaseModel):
    expdir: str = "exp/default"


class SvcConfig(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    vocoder: VocoderConfig = VocoderConfig()
    infer: InferConfig = InferConfig()
    train: TrainConfig = TrainConfig()
    env: EnvConfig = EnvConfig()
    device: str = "auto"
    spks: list[str] = ["speaker0"]


def load_config(path: str | Path) -> SvcConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SvcConfig(**raw)


def save_config(config: SvcConfig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(),
            f,
            default_flow_style=False,
            allow_unicode=True,
        )
