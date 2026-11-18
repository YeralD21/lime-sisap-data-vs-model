"""Experimento 3: prueba de perturbación (anexo §6.2) con la corrección de un error de scripts/08_exp3.py.

En 08_exp3.py (congelado con el anexo, 512b7cc), `_perturbar_exog` asignaba valores aleatorios decimales a
columnas enteras de las exógenas (p. ej., `eventos`) y pandas lo rechazaba, así que la prueba no llegó a correr.
Este script reutiliza 08_exp3.py sin modificarlo (su SHA-256 sigue coincidiendo) y solo sustituye esa función por
una que convierte antes las exógenas a decimal. Desviación registrada en docs/DESVIACIONES.md (EXP3-1).
Uso: .venv311\\Scripts\\python scripts/09_exp3_perturbacion.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("exp3", AQUI.parent / "08_exp3.py")
e3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e3)


def _perturbar_exog(ex, fecha_origen):
    rng = np.random.default_rng(20250102)
    out = ex.astype(float).copy()                       # única diferencia con 08_exp3.py
    k = out.index > fecha_origen
    for c in out.columns:
        v = out[c].to_numpy(float)
        out.loc[k, c] = rng.uniform(np.nanmin(v), np.nanmax(v), int(k.sum()))
    if "sin_noticias" in out:
        out.loc[k, "n_noticias"] = np.round(out.loc[k, "n_noticias"])
        out["sin_noticias"] = (out.n_noticias == 0).astype(int)
        out.loc[k, "sent"] = out.loc[k, "sent"].where(out.loc[k, "n_noticias"] > 0)
    return out


e3._perturbar_exog = _perturbar_exog
e3.__file__ = str(AQUI)                                  # los subprocesos de entrenamiento vuelven a este script

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo":
        e3.hijo(*sys.argv[2:5])
    else:
        e3.verificar()
        e3.perturbacion()
