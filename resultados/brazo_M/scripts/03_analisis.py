"""Brazo M, h = 1: ensambla pronosticos.csv, incidencias.csv y estados_redes.csv de un escenario y ejecuta el análisis.

Uso (desde la raíz, con .venv311):
    .venv311/Scripts/python.exe resultados/brazo_M/scripts/03_analisis.py --escenario principal
    .venv311/Scripts/python.exe resultados/brazo_M/scripts/03_analisis.py --sens_sin_reentrenar     (sens. 4 y 5 sobre el principal)
Lee base_pronosticos.csv / base_incidencias.csv (01) y resultados/brazo_M/redes_unidades/ (02).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

from src import config as C  # noqa: E402
from src.analisis import COLUMNAS, analizar, escribir  # noqa: E402
from src.datos import ESCENARIOS, serie  # noqa: E402
from src.redes import leer_predicciones  # noqa: E402

BASE = RAIZ / "resultados" / "brazo_M"
RAIZ_REDES = BASE / "redes_unidades"
N_PARAMS_DUAL = 4545     # expected_parameter_count(4, 6)
N_PARAMS_UNI = 1569      # parametros_lstm_univariada()


def pronosticos_redes(nombre: str) -> pd.DataFrame:
    e = ESCENARIOS[nombre]["M"]
    r = leer_predicciones(RAIZ_REDES)
    r = r[(r.brazo == "M") & (r.objetivo == e.objetivo) & (r.anomalia == e.anomalia) & (r.inicio == C.INICIO)
          & (r.particion == "test")]
    return r.assign(escenario=nombre)[["escenario", "brazo", "modelo", "semilla", "N", "anio", "h", "fecha", "y_pred"]]


def estados_redes(nombre: str) -> pd.DataFrame:
    e = ESCENARIOS[nombre]["M"]
    filas = []
    for f in RAIZ_REDES.rglob("estado.json"):
        s = json.loads(f.read_text(encoding="utf-8"))
        es = s["escenario"]
        if es["objetivo"] == e.objetivo and es["anomalia"] == e.anomalia:
            filas.append({"escenario": nombre, "modelo": s["modelo"], "anio": s["anio"], "N": str(s["N"]), "h": s["h"],
                          "semilla": s["semilla"], "epocas": s["epocas"], "mejor_epoca": s["mejor_epoca"],
                          "segundos": s["segundos"], "n_params": s["n_params"], "n_train": s["n_train"],
                          "n_hist": s["n_hist"]})
    return pd.DataFrame(filas).sort_values(["modelo", "anio", "N", "semilla"], kind="mergesort").reset_index(drop=True)


def ensamblar(nombre: str, carpeta: Path) -> pd.DataFrame:
    base = pd.read_csv(carpeta / "base_pronosticos.csv", dtype={"N": str})
    redes = pronosticos_redes(nombre)
    pron = pd.concat([base, redes], ignore_index=True)[COLUMNAS]
    pron["anio"] = pron.anio.astype(int)
    orden_m = {m: i for i, m in enumerate(C.MODELOS)}
    pron = pron.assign(_m=pron.modelo.map(orden_m)).sort_values(["_m", "anio", "N", "semilla", "fecha"], kind="mergesort").drop(columns="_m")
    pron.to_csv(carpeta / "pronosticos.csv", index=False, float_format="%.10g")
    # incidencias
    inc = pd.read_csv(carpeta / "base_incidencias.csv", dtype={"N": str})
    est = estados_redes(nombre)
    est.to_csv(carpeta / "estados_redes.csv", index=False)
    inc.to_csv(carpeta / "incidencias.csv", index=False)
    return pron


def comprobar_completitud(nombre: str, pron: pd.DataFrame) -> None:
    df = serie(ESCENARIOS[nombre]["M"])
    esperado_fechas = sum(len(df[df.index.year == a]) for a in C.ANIOS_TEST)
    for m in C.MODELOS:
        for N in map(str, C.TAMANOS["M"]):
            s = pron[(pron.modelo == m) & (pron.N == N)]
            ns = 10 if m in C.REDES else 1
            if len(s) != esperado_fechas * ns:
                raise AssertionError(f"{nombre} {m} N={N}: {len(s)} filas, se esperaban {esperado_fechas * ns}")
    if not np.isfinite(pron.y_pred).all():
        raise AssertionError("Hay y_pred no finitos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", choices=["principal", "sens1_estricto", "sens7_anomalia_publicada"])
    ap.add_argument("--sens_sin_reentrenar", action="store_true")
    a = ap.parse_args()
    if a.escenario:
        carpeta = BASE / a.escenario
        pron = ensamblar(a.escenario, carpeta)
        comprobar_completitud(a.escenario, pron)
        df = serie(ESCENARIOS[a.escenario]["M"])
        t = analizar(pron, df, "M", 1)
        escribir(t, carpeta)
        print(t["delta"].to_string())
        print(t["decision"].to_string())
    if a.sens_sin_reentrenar:
        pron = pd.read_csv(BASE / "principal" / "pronosticos.csv", dtype={"N": str})
        df = serie(ESCENARIOS["principal"]["M"])
        for sub, kw in (("sens4_pool2023_2025", dict(anios=C.ANIOS_POOL_SENS)), ("sens5_bloques", dict(tipo_boot="bloques"))):
            t = analizar(pron, df, "M", 1, **kw)
            escribir(t, BASE / sub)
            print(sub)
            print(t["delta"].to_string())
            print(t["decision"].to_string())
