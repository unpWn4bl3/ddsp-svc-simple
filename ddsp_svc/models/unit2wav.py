import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddsp import CombSubSuperFast
from .lynxnet import LYNXNet2
from .rectified_flow import RectifiedFlow


class Unit2Wav(nn.Module):
    def __init__(self, sampling_rate: int, block_size: int, win_length: int,
                 n_unit: int, n_spk: int, use_norm: bool = False, use_attention: bool = False,
                 use_pitch_aug: bool = False, out_dims: int = 128,
                 n_aux_layers: int = 3, n_aux_chans: int = 256,
                 n_layers: int = 6, n_chans: int = 512):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.ddsp = CombSubSuperFast(
            sampling_rate, block_size, win_length, n_unit, n_spk,
            n_aux_layers, n_aux_chans, use_norm, use_attention, use_pitch_aug,
        )
        self.reflow = RectifiedFlow(
            LYNXNet2(in_dims=out_dims, dim_cond=out_dims, n_layers=n_layers, n_chans=n_chans),
            out_dims=out_dims,
        )

    def forward(self, units: torch.Tensor, f0: torch.Tensor, volume: torch.Tensor,
                spk_id: torch.Tensor | None = None, spk_mix_dict: dict | None = None,
                aug_shift: torch.Tensor | None = None, vocoder=None,
                gt_spec: torch.Tensor | None = None, infer: bool = True,
                return_wav: bool = False, infer_step: int = 10, method: str = 'euler',
                t_start: float = 0.0, silence_front: float = 0, use_tqdm: bool = True):
        ddsp_wav, hidden = self.ddsp(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                      aug_shift=aug_shift, infer=infer)
        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        ddsp_mel = vocoder.extract(ddsp_wav[:, start_frame * self.block_size:]) if vocoder is not None else None

        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec) if ddsp_mel is not None else torch.tensor(0.0, device=ddsp_wav.device)
            if t_start < 1.0 and ddsp_mel is not None:
                reflow_loss = self.reflow(ddsp_mel, gt_spec=gt_spec, infer=False, t_start=t_start)
            else:
                reflow_loss = torch.tensor(0.0, device=ddsp_wav.device)
            return ddsp_loss, reflow_loss

        if gt_spec is not None and ddsp_mel is None:
            ddsp_mel = gt_spec
        if t_start < 1.0:
            mel = self.reflow(ddsp_mel, gt_spec=ddsp_mel, infer=True,
                              infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
        else:
            mel = ddsp_mel
        if return_wav:
            return vocoder.infer(mel, f0[:, -mel.shape[1]:])
        return mel
