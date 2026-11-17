"""Auditoría día 3: ¿cambia algo `t_index` de la rama B (DualLSTM, brazo D) si la serie empieza en 2006-01-01 (sens. 6)?

1. Compara las entradas escaladas (rama A y rama B) de la principal y de sens6 para N = 120, 360 y 1000, Y = 2021-2025,
   en float64 y tras convertir a float32 (lo que recibe TensorFlow).
2. Si se pasa --red DIR (unidad sens6 Y=2024 N=360 h=1 DualLSTM semilla 0 reentrenada con src.redes en un directorio
   temporal), compara sus predicciones de prueba con la unidad de la principal publicada en resultados/brazo_D/redes_unidades.
Escribe reportes/auditoria/sens6_tindex.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.datos import Escenario, serie  # noqa: E402
from src.redes import preparar  # noqa: E402
from src.ventanas import Ventana, particion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--red", default=None)
a = ap.parse_args()
pa, pb = serie(Escenario("D")), serie(Escenario("D", inicio="2006-01-01"))
filas = []
for anio in range(2021, 2026):
    for N in (120, 360, 1000):
        Xa, ysa, posa, sca, *_ = preparar(pa, "D", "DualLSTM", particion(pa.index, Ventana(anio), N), 1)
        Xb, ysb, posb, scb, *_ = preparar(pb, "D", "DualLSTM", particion(pb.index, Ventana(anio), N), 1)
        for parte in ("train", "val", "test"):
            for rama in (0, 1):
                d64 = np.max(np.abs(Xa[parte][rama] - Xb[parte][rama]))
                d32 = np.max(np.abs(Xa[parte][rama].astype(np.float32) - Xb[parte][rama].astype(np.float32)))
                n32 = int(np.sum(Xa[parte][rama].astype(np.float32) != Xb[parte][rama].astype(np.float32)))
                filas.append({"comprobacion": "entradas escaladas", "anio": anio, "N": N, "particion": parte,
                              "rama": "A" if rama == 0 else "B", "dif_max_float64": d64, "dif_max_float32": float(d32),
                              "n_valores_float32_distintos": n32, "n_valores": int(Xa[parte][rama].size)})
if a.red:
    r = pd.read_csv(Path(a.red) / "predicciones.csv")
    u = pd.read_csv(RAIZ / "resultados/brazo_D/redes_unidades/D_principal_1997_faltante/dual/Y2024/N360/h1/s00/predicciones.csv")
    m = r.merge(u, on=["particion", "fecha"], suffixes=("_sens6", "_principal"))
    for parte in ("val", "test"):
        s = m[m.particion == parte]
        d = (s.y_pred_sens6 - s.y_pred_principal).abs()
        filas.append({"comprobacion": "predicciones DualLSTM s0 Y2024 N360 h1 (sens6 reentrenada vs principal publicada)",
                      "anio": 2024, "N": 360, "particion": parte, "rama": "", "dif_max_float64": float(d.max()),
                      "dif_max_float32": np.nan, "n_valores_float32_distintos": int((d > 0).sum()), "n_valores": len(d)})
t = pd.DataFrame(filas)
t.to_csv(Path(__file__).resolve().parent / "sens6_tindex.csv", index=False, float_format="%.6g")
print(t.groupby(["comprobacion", "rama"])[["dif_max_float64", "dif_max_float32", "n_valores_float32_distintos"]].max())
