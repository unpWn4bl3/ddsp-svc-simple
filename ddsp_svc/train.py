import os
import time
import datetime
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ddsp_svc.config import SvcConfig
from ddsp_svc.dataset import get_data_loaders
from ddsp_svc.device import autocast_context, detect_device, empty_cache
from ddsp_svc.models.unit2wav import Unit2Wav
from ddsp_svc.models.vocoder import Vocoder, STFT


class Trainer:
    def __init__(self, config: SvcConfig):
        self.config = config
        self.device_info = detect_device(config.device)
        self.device = str(self.device_info)

        self.vocoder = Vocoder(config.vocoder.type, config.vocoder.ckpt, device=self.device)
        self.model = Unit2Wav(
            config.data.sampling_rate, config.data.block_size, config.model.win_length,
            config.data.encoder_out_channels, config.model.n_spk,
            config.model.use_norm, config.model.use_attention, config.model.use_pitch_aug,
            self.vocoder.dimension,
            config.model.n_aux_layers, config.model.n_aux_chans,
            config.model.n_layers, config.model.n_chans,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.train.learning_rate,
                                            weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.train.decay_step, gamma=config.train.gamma,
        )

        self.global_step = 0
        self.exp_dir = Path(config.env.expdir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.exp_dir / "logs"))
        self.scaler = GradScaler(device=self.device_info.type) if config.train.amp_dtype != "fp32" else None
        self.start_time = time.time()
        self.last_time = time.time()
        self.stop_file = self.exp_dir / "stop.txt"

    def restore(self):
        ckpts = sorted(self.exp_dir.glob("model_*.pt"))
        if not ckpts:
            return 0
        latest = ckpts[-1]
        logger.info(f"Restoring from {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        for _ in range(self.global_step):
            self.scheduler.step()
        return self.global_step

    def save(self, step: int, force: bool = False):
        postfix = f"_{step}"
        path = self.exp_dir / f"model{postfix}.pt"
        torch.save({
            "global_step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.config.train.save_opt else None,
        }, path)
        logger.info(f"Saved {path}")

    def should_stop(self) -> bool:
        return self.stop_file.exists()

    def run(self):
        self.restore()
        if self.stop_file.exists():
            self.stop_file.unlink()
        loader_train, loader_valid = get_data_loaders(self.config, whole_audio=False)

        amp_dtype = self.config.train.amp_dtype
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_dtype, torch.float32)
        n_batches = len(loader_train)
        start_epoch = self.global_step // n_batches

        self.model.train()
        logger.info("Training started")

        for epoch in range(start_epoch, self.config.train.epochs):
            for batch in loader_train:
                if self.should_stop():
                    logger.info("Stop file detected, exiting")
                    return

                self.global_step += 1
                self.optimizer.zero_grad()

                for k in batch:
                    if not k.startswith("name"):
                        batch[k] = batch[k].to(self.device)

                if dtype == torch.float32:
                    ddsp_loss, reflow_loss = self.model(
                        batch["units"].float(), batch["f0"], batch["volume"],
                        batch["spk_id"], aug_shift=batch["aug_shift"],
                        vocoder=self.vocoder, gt_spec=batch["mel"].float(),
                        infer=False, t_start=self.config.model.t_start,
                    )
                else:
                    with autocast(device_type=self.device_info.type, dtype=dtype):
                        ddsp_loss, reflow_loss = self.model(
                            batch["units"], batch["f0"], batch["volume"],
                            batch["spk_id"], aug_shift=batch["aug_shift"],
                            vocoder=self.vocoder, gt_spec=batch["mel"].float(),
                            infer=False, t_start=self.config.model.t_start,
                        )

                if torch.isnan(ddsp_loss):
                    logger.warning("NaN ddsp_loss, skipping")
                    self.optimizer.zero_grad()
                    continue
                if torch.isnan(reflow_loss):
                    raise ValueError("NaN reflow_loss")

                loss = self.config.train.lambda_ddsp * ddsp_loss + reflow_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.scheduler.step()

                if self.global_step % self.config.train.interval_log == 0:
                    now = time.time()
                    speed = self.config.train.interval_log / (now - self.last_time)
                    elapsed = str(datetime.timedelta(seconds=int(now - self.start_time)))
                    logger.info(
                        f"epoch {epoch} | {self.global_step % n_batches}/{n_batches} | "
                        f"batch/s: {speed:.2f} | lr: {self.optimizer.param_groups[0]['lr']:.6} | "
                        f"loss: {loss.item():.3f} | time: {elapsed} | step: {self.global_step}"
                    )
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/ddsp_loss", ddsp_loss.item(), self.global_step)
                    self.writer.add_scalar("train/reflow_loss", reflow_loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                    self.last_time = now

                if self.global_step % self.config.train.interval_val == 0:
                    self.save(self.global_step)
                    self.validate(loader_valid)
                    self.model.train()

    @torch.no_grad()
    def validate(self, loader_valid):
        self.model.eval()
        ddsp_losses, reflow_losses, snrs = [], [], []

        for batch in loader_valid:
            for k in batch:
                if not k.startswith("name"):
                    batch[k] = batch[k].to(self.device)

            ddsp_loss, reflow_loss = self.model(
                batch["units"], batch["f0"], batch["volume"],
                batch["spk_id"], aug_shift=batch["aug_shift"],
                vocoder=self.vocoder, gt_spec=batch["mel"],
                infer=False, t_start=self.config.model.t_start,
            )
            ddsp_losses.append(ddsp_loss.item())
            reflow_losses.append(reflow_loss.item())

            mel = self.model(
                batch["units"], batch["f0"], batch["volume"],
                batch["spk_id"], aug_shift=batch["aug_shift"],
                vocoder=self.vocoder, gt_spec=batch["mel"],
                infer=True, return_wav=False,
                infer_step=self.config.infer.infer_step,
                method=self.config.infer.method,
                t_start=self.config.model.t_start,
            )

            snr = 10 * torch.log10(torch.mean(batch["mel"] ** 2) / torch.var(batch["mel"] - mel))
            snrs.append(snr.item())
            self.writer.add_audio(
                f"val/{batch['name'][0]}",
                self.vocoder.infer(mel, batch["f0"]).squeeze(),
                global_step=self.global_step,
                sample_rate=self.config.data.sampling_rate,
            )

        avg_ddsp = np.mean(ddsp_losses)
        avg_reflow = np.mean(reflow_losses)
        avg_snr = np.mean(snrs)
        avg_loss = self.config.train.lambda_ddsp * avg_ddsp + avg_reflow

        logger.info(f"Validation: loss={avg_loss:.3f} ddsp={avg_ddsp:.3f} reflow={avg_reflow:.3f} snr={avg_snr:.2f}")
        self.writer.add_scalar("validation/loss", avg_loss, self.global_step)
        self.writer.add_scalar("validation/snr", avg_snr, self.global_step)
