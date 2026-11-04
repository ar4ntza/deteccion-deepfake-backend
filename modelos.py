"""Arquitecturas (según el documento de entrenamiento), carga de pesos y predicción."""
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

import config as C

log = logging.getLogger("modelos")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModeloNoDisponible(Exception):
    pass


# ---------------- Arquitecturas ----------------
def _bloque(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class CNN(nn.Module):  # 404,898 parámetros
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_bloque(1, 32), _bloque(32, 64), _bloque(64, 128), _bloque(128, 256),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 2))

    def forward(self, x):                      # (B, 1, n_mels, frames)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.head(self.features(x))


class LSTM(nn.Module):  # 692,674 parámetros
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(C.N_MELS)
        self.lstm = nn.LSTM(C.N_MELS, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.4)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 2))

    def forward(self, x):                      # (B, 1, n_mels, frames) -> (B, frames, n_mels)
        if x.dim() == 4:
            x = x.squeeze(1)
        out, _ = self.lstm(self.norm(x.transpose(1, 2)))
        return self.head(torch.cat([out.mean(1), out.amax(1)], dim=1))


# ---------------- Carga de pesos ----------------
def _cargar(clase, ruta: Path):
    obj = torch.load(ruta, map_location=DEVICE, weights_only=False)  # solo archivos propios y de confianza
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, nn.Module):             # modelo completo guardado con torch.save(model)
        return obj.to(DEVICE).eval()
    net = clase()
    try:
        net.load_state_dict(obj)
    except RuntimeError as err:
        # Si solo difieren los nombres de las capas, se asignan por orden cuando las formas coinciden.
        mias, suyas = list(net.state_dict().items()), list(obj.items())
        if len(mias) == len(suyas) and all(a[1].shape == b[1].shape for a, b in zip(mias, suyas)):
            net.load_state_dict({a[0]: b[1] for a, b in zip(mias, suyas)})
            log.warning("%s: nombres de capas distintos, pesos asignados por orden.", ruta.name)
        else:
            raise RuntimeError(f"Los pesos de {ruta.name} no coinciden con la arquitectura de modelos.py: {err}") from err
    return net.to(DEVICE).eval()


class Modelo:
    """Carga el archivo .pt y lo recarga solo si cambia en disco (para reemplazarlo sin reiniciar)."""

    def __init__(self, nombre, clase, ruta: Path):
        self.nombre, self.clase, self.ruta = nombre, clase, ruta
        self.net, self.visto, self.cargado_en = None, None, None
        self.lock = threading.Lock()

    def asegurar(self):
        try:
            mt = self.ruta.stat().st_mtime
        except FileNotFoundError:
            raise ModeloNoDisponible(f"Falta el archivo {self.ruta.name} en {self.ruta.parent}.")
        if mt == self.visto and self.net is not None:
            return
        with self.lock:
            if mt == self.visto and self.net is not None:
                return
            self.visto = mt
            try:
                self.net = _cargar(self.clase, self.ruta)
                self.cargado_en = datetime.now().isoformat(timespec="seconds")
                log.info("Modelo %s cargado desde %s", self.nombre, self.ruta)
            except Exception:
                if self.net is None:
                    self.visto = None
                    raise
                log.exception("No se pudo cargar la nueva versión de %s; se conserva la anterior.", self.nombre)

    def predecir(self, espectro):
        """espectro: array [n_mels, frames] ya normalizado -> (prob_fake, ms de inferencia)."""
        self.asegurar()
        net = self.net
        x = torch.from_numpy(espectro)[None, None].to(DEVICE)
        t0 = time.perf_counter()
        with torch.inference_mode():
            prob = torch.softmax(net(x), dim=1)[0, 1].item()   # índice 1 = fake
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return prob, (time.perf_counter() - t0) * 1000

    def estado(self):
        existe = self.ruta.exists()
        return {"archivo": str(self.ruta), "existe": existe, "cargado": self.net is not None,
                "modificado": datetime.fromtimestamp(self.ruta.stat().st_mtime).isoformat(timespec="seconds") if existe else None,
                "cargado_en": self.cargado_en}


MODELOS = {"cnn": Modelo("cnn", CNN, C.CNN_PATH), "lstm": Modelo("lstm", LSTM, C.LSTM_PATH)}