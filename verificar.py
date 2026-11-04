"""Comprueba que los modelos cargan y que el preprocesamiento coincide con el del entrenamiento.

  python verificar.py --npy datos/espanol/fake/ejemplo.npy     (espectrograma ya generado en el entrenamiento)
  python verificar.py --audio ejemplo.wav                      (el mismo audio, procesado por el backend)

Si ambos dan probabilidades parecidas (y min/max/media similares), el preprocesamiento es correcto.
También puedes comparar con la fila de ese audio en registro_pruebas.csv.
"""
import argparse

import numpy as np

import audio
import modelos

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--npy")
g.add_argument("--audio")
a = ap.parse_args()

s = np.load(a.npy).astype(np.float32) if a.npy else audio.espectrograma(a.audio)[0]
print(f"espectrograma {s.shape}  min={s.min():.1f}  max={s.max():.1f}  media={s.mean():.1f}")
x = audio.preparar(s)
for nombre, m in modelos.MODELOS.items():
    p, ms = m.predecir(x)
    print(f"{nombre}: prob_fake={p:.4f}  ({'fake' if p >= .5 else 'real'})  {ms:.1f} ms")