"""Comprueba que los métodos rápidos de estadisticos_lib equivalen a los del protocolo (parámetros fijos, estado actualizado):
- ETS: smooth(params) sobre el prefijo hasta τ-h + forecast(h)  frente a  smooth sobre toda la serie + simulate(anchor).
- SARIMA: fit.extend(datos hasta τ-h).forecast(h)  frente a  smooth sobre toda la serie + get_prediction(dynamic=0).
- Chronos: contexto de un solo día frente al lote (relleno).
Escribe resultados/brazo_D/equivalencias.csv (diferencia máxima absoluta, en toneladas)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estadisticos_lib as L  # noqa: E402
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa: E402

warnings.simplefilter("ignore")
df = L.serie(L.escenario_D("principal"))
y = df.y.to_numpy()
filas = []
for anio, N in ((2023, 360), (2025, 1000), (2021, "max")):
    part = L.particion(df.index, L.Ventana(anio), N)
    idx = [0, 1, 6, 30, 200, len(part.test) - 1]
    # --- ETS
    spec, fit, _, _ = L.seleccionar_ets(y[part.hist])
    params = np.asarray(fit.params, float)
    rapido, _, _ = L.ets_unidad(df, part, (1, 7), "x")
    for h in (1, 7):
        ref = []
        for j in idx:
            tau = part.test[j]
            r = ETSModel(y[part.ini_hist: tau - h + 1], seasonal_periods=7, initialization_method="estimated", **spec).smooth(params)
            ref.append(float(np.asarray(r.forecast(h), float)[-1]))
        d = np.max(np.abs(np.array(ref) - rapido[h][1][idx]))
        filas.append({"modelo": "ETS", "anio": anio, "N": N, "h": h, "dif_max_abs_t": d, "n_dias": len(idx)})
    # --- SARIMA
    usa_f = part.n_hist >= 730
    ex_h = L.fourier(np.arange(part.ini_hist, part.fin_train + 1)) if usa_f else None
    ajuste = L._sarimax(y[part.hist], ex_h).fit(disp=False, maxiter=500)
    rapido, _ = L.sarima_unidad(df, part, (1, 7))
    for h in (1, 7):
        ref = []
        for j in idx:
            tau = part.test[j]
            nuevos = np.arange(part.fin_train + 1, tau - h + 1)
            fut = np.arange(tau - h + 1, tau + 1)
            ext = ajuste.extend(np.log(y[nuevos]), exog=L.fourier(nuevos) if usa_f else None) if len(nuevos) else ajuste
            f = ext.forecast(h, exog=L.fourier(fut) if usa_f else None)
            ref.append(float(np.exp(np.asarray(f)[-1])))
        d = np.max(np.abs(np.array(ref) - rapido[h][1][idx]))
        filas.append({"modelo": "SARIMA", "anio": anio, "N": N, "h": h, "dif_max_abs_t": d, "n_dias": len(idx)})
# --- Chronos: individual frente a lote
part = L.particion(df.index, L.Ventana(2023), "max")
P = L._pipeline()
lote, _ = L.chronos_unidad(df, part, (1, 7))
for h in (1, 7):
    ref = []
    idx = [0, 100, 300]
    for j in idx:
        tau = part.test[j]
        c = y[: tau - h + 1][-P["ctx"]:]
        q = P["p"].predict([P["torch"].tensor(c, dtype=P["torch"].float32)], prediction_length=h).numpy()
        ref.append(float(q[0, P["imed"], h - 1]))
    filas.append({"modelo": "Chronos", "anio": 2023, "N": "max", "h": h,
                  "dif_max_abs_t": float(np.max(np.abs(np.array(ref) - lote[h][1][idx]))), "n_dias": len(idx)})
t = pd.DataFrame(filas)
print(t)
t.to_csv(L.RAIZ / "resultados" / "brazo_D" / "equivalencias.csv", index=False)
