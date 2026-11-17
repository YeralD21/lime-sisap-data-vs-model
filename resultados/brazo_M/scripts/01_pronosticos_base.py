"""Brazo M, h = 1: pronósticos de Seasonal Naive, ETS, Theta, SARIMA y Chronos-Bolt-tiny para un escenario.

Uso (desde la raíz del repositorio, con .venv311):
    HF_HUB_OFFLINE=1 .venv311/Scripts/python.exe resultados/brazo_M/scripts/01_pronosticos_base.py --escenario principal
    ... --anios 2023 --N 120 --salida resultados/brazo_M/_prueba     (prueba con una ventana y un N)
Escribe base_pronosticos.csv, base_incidencias.csv y base_tiempos.csv en la carpeta del escenario.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as C  # noqa: E402
from src.datos import ESCENARIOS, serie  # noqa: E402

import modelos_base as MB  # noqa: E402

CARPETAS = {"principal": "principal", "sens1_estricto": "sens1_estricto", "sens7_anomalia_publicada": "sens7_anomalia_publicada"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", required=True, choices=list(CARPETAS))
    ap.add_argument("--anios", nargs="*", type=int, default=list(C.ANIOS_TEST))
    ap.add_argument("--N", nargs="*", default=[str(n) for n in C.TAMANOS["M"]])
    ap.add_argument("--modelos", nargs="*", default=list(MB.MODELOS_BASE))
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    df = serie(ESCENARIOS[a.escenario]["M"])
    destino = Path(a.salida) if a.salida else RAIZ / "resultados" / "brazo_M" / CARPETAS[a.escenario]
    destino.mkdir(parents=True, exist_ok=True)
    Ns = [n if n == "max" else int(n) for n in a.N]
    t0 = time.perf_counter()
    pron, inc, tiempos = MB.correr_base(df, a.escenario, a.anios, Ns, a.modelos, log=lambda s: print(s, flush=True))
    total = time.perf_counter() - t0
    pron.to_csv(destino / "base_pronosticos.csv", index=False, float_format="%.10g")
    inc.to_csv(destino / "base_incidencias.csv", index=False)
    tiempos.to_csv(destino / "base_tiempos.csv", index=False)
    print(f"TOTAL base {a.escenario}: {total:.1f} s; {len(pron)} pronósticos; {len(inc)} incidencias", flush=True)
