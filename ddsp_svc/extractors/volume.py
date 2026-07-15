import numpy as np


class VolumeExtractor:
    def __init__(self, hop_size: int = 512, win_size: int = 2048):
        self.hop_size = hop_size
        self.win_size = win_size

    def extract(self, audio: np.ndarray) -> np.ndarray:
        n_frames = int(len(audio) // self.hop_size) + 1
        audio = np.pad(audio, (int(self.win_size // 2), int((self.win_size + 1) // 2)), mode='reflect')
        audio2 = audio ** 2
        mean = np.array([np.mean(audio[n * self.hop_size: n * self.hop_size + self.win_size]) for n in range(n_frames)])
        mean_sq = np.array([np.mean(audio2[n * self.hop_size: n * self.hop_size + self.win_size]) for n in range(n_frames)])
        return np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))
