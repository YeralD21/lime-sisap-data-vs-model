"""Prueba previa: cada modelo en una sola ventana y un solo N, con el tiempo real. Escribe resultados/brazo_D/tiempos_una_ventana.csv.
Uso: python resultados/brazo_D/scripts/tiempos_una_ventana.py <anio> <N> <modelo> [<modelo> ...]   (escenario principal, h = 1 y 7)"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estadisticos_lib as L  # noqa: E402

anio, N, modelos = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
df = L.serie(L.escenario_D("principal"))
part = L.particion(df.index, L.Ventana(anio), N if N == "max" else int(N))
filas = []
for m in modelos:
    t0 = time.perf_counter()
    if m == "Seasonal Naive":
        L.seasonal_naive(df, part)
    elif m == "ETS":
        L.ets_unidad(df, part, (1, 7), "principal")
    elif m == "Theta":
        L.theta_unidad(df, part, (1, 7))
    elif m == "SARIMA":
        L.sarima_unidad(df, part, (1, 7))
    else:
        L.chronos_unidad(df, part, (1, 7))
    seg = time.perf_counter() - t0
    filas.append({"modelo": m, "anio": anio, "N": N, "hs": "1,7", "n_hist": part.n_hist, "segundos": round(seg, 1)})
    print(filas[-1], flush=True)
ruta = L.RAIZ / "resultados" / "brazo_D" / "tiempos_una_ventana.csv"
t = pd.DataFrame(filas)
if ruta.exists():
    t = pd.concat([pd.read_csv(ruta), t], ignore_index=True)
t.to_csv(ruta, index=False)
