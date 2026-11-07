"""API del detector de audio deepfake.

Va en la RAÍZ del proyecto de los modelos, junto a detector/, training/ y resultados_cv/.
Ejecutar desde esa carpeta:   python -m uvicorn main:app --port 8000
"""
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))   # para que se puedan importar `detector` y `training`

import numpy as np                                                    # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                    # noqa: E402

from detector.inference import Detector                               # noqa: E402
from training.preprocesamiento import audio_a_mel                     # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# Configuración opcional por variables de entorno
MAX_BYTES = int(os.getenv("MAX_MB", 25)) * 1024 * 1024
ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
MAX_MIEMBROS = int(os.getenv("MAX_MIEMBROS", 0)) or None    # limita cuántos modelos del conjunto se cargan por etapa
RESULTADOS = os.getenv("RESULTADOS_DIR") or None            # por defecto: <raíz>/resultados_cv

estado = {"detector": None, "error": None}
candado = threading.Lock()      # una inferencia a la vez: evita saturar CPU/GPU con audios simultáneos


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    try:
        estado["detector"] = Detector(resultados=RESULTADOS, max_miembros=MAX_MIEMBROS)
        log.info("Modelos cargados en %.1f s", time.perf_counter() - t0)
    except Exception as e:      # la API arranca igual; /salud y /predict explican qué falló
        estado["error"] = f"{type(e).__name__}: {e}"
        log.exception("No se pudieron cargar los modelos")
    yield


app = FastAPI(title="Detector de audio deepfake", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


def _partes(nombre: str):
    """'CNN · B preentrenado + afinado' -> ('cnn', 'B')"""
    arq, _, resto = nombre.partition(" · ")
    return arq.lower(), resto[:1]


@app.get("/salud")
def salud():
    det = estado["detector"]
    return {
        "ok": det is not None,
        "error": estado["error"],
        "dispositivo": det.device if det else None,
        "modelos": {nombre: len(miembros) for nombre, miembros in det.modelos} if det else {},
    }


@app.post("/predict")
def predict(audio: UploadFile = File(...)):
    det = estado["detector"]
    if det is None:
        raise HTTPException(503, "Los modelos no están cargados. Revisa el registro del servidor.")

    datos = audio.file.read(MAX_BYTES + 1)
    if len(datos) > MAX_BYTES:
        raise HTTPException(413, f"El audio pesa más de {MAX_BYTES // 1024 // 1024} MB.")
    if not datos:
        raise HTTPException(422, "El archivo está vacío.")

    sufijo = Path(audio.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(datos)
        ruta = tmp.name
    try:
        t0 = time.perf_counter()
        try:
            mel = audio_a_mel(ruta)
        except Exception:
            log.warning("Audio ilegible: %s", audio.filename, exc_info=True)
            raise HTTPException(422, "No pudimos leer el audio. Prueba con un WAV, MP3 o FLAC válido.")
        if mel.shape[1] < 10 or np.ptp(mel) < 1e-3:      # misma validación que Detector.analizar
            raise HTTPException(422, "El audio está vacío o es demasiado corto.")
        with candado:
            probs = det.probabilidades(mel)              # {etiqueta: prob_fake 0-1}, ya calibrada y en ensamble
    finally:
        os.unlink(ruta)

    resultados = []
    for nombre, p in probs.items():
        arquitectura, condicion = _partes(nombre)
        fake = round(100 * p, 1)
        resultados.append({"modelo": nombre, "arquitectura": arquitectura, "condicion": condicion,
                           "fake": fake, "real": round(100 - fake, 1)})
    return {"resultados": resultados, "ms_total": round((time.perf_counter() - t0) * 1000)}