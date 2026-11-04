"""Configuración del backend. Todo se puede cambiar con variables de entorno (.env)."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
env = os.getenv

# --- Modelos (para reemplazarlos, solo sobrescribe estos archivos) ---
MODELS_DIR = Path(env("MODELS_DIR", BASE / "modelos"))
CNN_PATH = Path(env("CNN_PATH", MODELS_DIR / "cnn.pt"))
LSTM_PATH = Path(env("LSTM_PATH", MODELS_DIR / "lstm.pt"))

# --- Espectrograma: DEBE ser idéntico al script que generó los .npy de entrenamiento ---
SR = int(env("SR", 16000))
N_FFT = int(env("N_FFT", 1024))
HOP_LENGTH = int(env("HOP_LENGTH", 512))
N_MELS = int(env("N_MELS", 128))
FMIN = float(env("FMIN", 0))
FMAX = float(env("FMAX", 0)) or None   # 0 = sr/2
DB_REF = env("DB_REF", "max")          # "max" -> librosa.power_to_db(ref=np.max); "1" -> ref=1.0
N_FRAMES = int(env("N_FRAMES", 128))

# --- Servidor ---
MAX_BYTES = int(env("MAX_MB", 25)) * 1024 * 1024
ORIGINS = env("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")