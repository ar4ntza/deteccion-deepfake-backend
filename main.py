"""API del detector de audio deepfake.  Ejecutar:  uvicorn main:app --reload --env-file .env"""
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import audio
import config as C
import modelos

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for m in modelos.MODELOS.values():       # precarga; si falta un archivo, la API arranca igual
        try:
            m.asegurar()
        except Exception as e:
            log.error("No se pudo cargar %s: %s", m.nombre, e)
    yield


app = FastAPI(title="Detector de audio deepfake", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=C.ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/salud")
def salud():
    return {"ok": True, "dispositivo": str(modelos.DEVICE), "modelos": {n: m.estado() for n, m in modelos.MODELOS.items()}}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    datos = file.file.read(C.MAX_BYTES + 1)
    if len(datos) > C.MAX_BYTES:
        raise HTTPException(413, f"El audio pesa más de {C.MAX_BYTES // 1024 // 1024} MB.")

    sufijo = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(datos)
        ruta = tmp.name
    try:
        espectro, duracion = audio.espectrograma(ruta)
    except Exception as e:
        log.warning("Audio ilegible (%s): %s", file.filename, e)
        raise HTTPException(422, "No pudimos leer el audio. Prueba con un WAV, MP3 o FLAC válido.")
    finally:
        os.unlink(ruta)

    x = audio.preparar(espectro)
    resp = {}
    for nombre, m in modelos.MODELOS.items():
        try:
            prob, ms = m.predecir(x)
        except modelos.ModeloNoDisponible as e:
            raise HTTPException(503, str(e))
        resp[nombre] = {"prob_fake": round(prob, 4), "ms": round(ms, 1)}
    resp["info"] = {"duracion_seg": round(duracion, 2),
                    "analizados_seg": round(min(duracion, C.N_FRAMES * C.HOP_LENGTH / C.SR), 2)}
    return resp