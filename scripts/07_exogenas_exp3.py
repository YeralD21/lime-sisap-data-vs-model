"""Experimento 3, fase A1: construye las exógenas alineadas (docs/ANEXO_EXP3.md). No entrena nada.

Escribe (no versionados): data/processed/exogenas_diarias.csv (brazo D) y data/processed/exogenas_mensuales.csv
(brazo M). Versionados: resultados/exp3/pesos_provincias_2016_2019.csv y resultados/exp3/resumen_exogenas.csv
(cobertura y estadísticos descriptivos de las exógenas; ningún dato del SISAP).
Reanudable: la descarga diaria de NASA POWER se guarda por provincia en data/raw/nasa_power_diario/.
Uso: .venv311\\Scripts\\python scripts/07_exogenas_exp3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import exogenas as X3  # noqa: E402


def resumen(df: pd.DataFrame, brazo: str) -> pd.DataFrame:
    filas = []
    for c in df.columns:
        v = df[c]
        filas.append({"brazo": brazo, "variable": c, "inicio": v.index.min().date(), "fin": v.index.max().date(),
                      "n": len(v), "faltantes": int(v.isna().sum()), "media": v.mean(), "de": v.std(),
                      "min": v.min(), "max": v.max(), "ceros": int((v == 0).sum())})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    d = X3.construir_diarias()
    m = X3.construir_mensuales()
    r = pd.concat([resumen(d, "D"), resumen(m, "M")], ignore_index=True)
    r.to_csv(X3.PESOS.parent / "resumen_exogenas.csv", index=False, float_format="%.6g")
    print(r.to_string())
