from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch

DeviceType = Literal["cuda", "mps", "cpu"]


@dataclass
class DeviceInfo:
    type: DeviceType
    name: str
    index: int = 0

    def __str__(self):
        if self.type == "cuda":
            return f"cuda:{self.index}"
        return self.type

    def torch_device(self):
        return torch.device(str(self))


def detect_device(prefer: str = "auto") -> DeviceInfo:
    if prefer == "auto":
        if torch.cuda.is_available():
            return DeviceInfo(
                "cuda",
                torch.cuda.get_device_name(0),
            )
        if torch.backends.mps.is_available():
            return DeviceInfo("mps", "Apple Silicon")
        return DeviceInfo("cpu", "CPU")
    if prefer.startswith("cuda"):
        idx = int(prefer.split(":")[1]) if ":" in prefer else 0
        return DeviceInfo("cuda", torch.cuda.get_device_name(idx), idx)
    if prefer == "mps":
        return DeviceInfo("mps", "Apple Silicon")
    return DeviceInfo("cpu", "CPU")


def autocast_context(device_type: DeviceType, amp_dtype: str):
    if device_type == "cpu" or amp_dtype == "fp32":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type, dtype=dtype)


def empty_cache(device_type: DeviceType):
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()
