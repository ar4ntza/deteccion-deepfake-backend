"""Audio -> espectrograma listo para las redes (mismo procedimiento que SpecDataset en val/test)."""
import librosa
import numpy as np

import config as C


def espectrograma(ruta: str):
    """Devuelve (mel-espectrograma en dB [n_mels, frames], duración en segundos)."""
    y, _ = librosa.load(ruta, sr=C.SR, mono=True)
    if y.size == 0:
        raise ValueError("El audio está vacío.")
    mel = librosa.feature.melspectrogram(
        y=y, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX,
    )
    db = librosa.power_to_db(mel, ref=np.max if C.DB_REF == "max" else 1.0)
    return db.astype(np.float32), y.size / C.SR


def preparar(s: np.ndarray) -> np.ndarray:
    """Ajuste a N_FRAMES (inicio si es largo; relleno con el mínimo si es corto) + z-score."""
    n = C.N_FRAMES
    if s.shape[1] >= n:
        s = s[:, :n]
    else:
        s = np.pad(s, ((0, 0), (0, n - s.shape[1])), constant_values=s.min())
    return ((s - s.mean()) / (s.std() + 1e-8)).astype(np.float32)