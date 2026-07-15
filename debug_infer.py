"""Run inference and print debug info at each stage."""
import sys
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

sys.path.insert(0, ".")
from ddsp_svc.config import load_config
from ddsp_svc.infer import InferencePipeline


cfg = load_config("configs/default.yaml")
pipe = InferencePipeline(cfg)

# Load latest checkpoint
ckpts = sorted(Path("exp").rglob("model_*.pt"))
if not ckpts:
    print("No checkpoints found!")
    sys.exit(1)
ckpt_path = str(ckpts[-1])
print(f"Loading: {ckpt_path}")
pipe.load_model(ckpt_path)

# Run inference on a small segment of a test wav
test_wavs = list(Path("data/raw").rglob("*.wav"))
if not test_wavs:
    print("No test wavs in data/raw/")
    sys.exit(1)

audio_path = str(test_wavs[0])
print(f"Input: {audio_path}")

audio, sr = sf.read(audio_path)
audio = audio[:sr * 3]  # first 3 seconds
tmp = "tmp/debug_input.wav"
sf.write(tmp, audio, sr)

output = pipe.infer(
    tmp, speaker=pipe.spks[0], keychange=0,
    infer_step=50, method="euler", t_start=0.0,
)

print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Output min/max/mean: {output.min():.6f} / {output.max():.6f} / {output.mean():.6f}")
print(f"Output RMS: {np.sqrt(np.mean(output**2)):.6f}")

if np.abs(output).max() < 1e-6:
    print("SILENT OUTPUT - checking mask...")
    # Bypass mask and re-run
    output2 = pipe.infer(
        tmp, speaker=pipe.spks[0], keychange=0,
        infer_step=50, method="euler", t_start=0.0, threshold=-999,
    )
    print(f"With threshold=-999: min={output2.min():.6f} max={output2.max():.6f} mean={output2.mean():.6f}")
else:
    sf.write("tmp/debug_out.wav", output, sr)
    print("Output saved to tmp/debug_out.wav")
