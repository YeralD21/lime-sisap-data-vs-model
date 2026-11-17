"""Auditoría día 3: reproducción independiente del brazo M (escenario principal), Y = 2023, N = 120, h = 1.

Escrito a partir de docs/PROTOCOLO.md §3-§5, sin importar los scripts del brazo M. Solo usa src.datos y src.ventanas.
- Seasonal Naive: ŷ_t = y_{t-12}.
- ETS: 12 especificaciones, menor AICc en la historia N; ETSModel(y[ini:t]).smooth(params).forecast(1) para cada mes t.
- SARIMA (1,1,1)(1,1,0)_12, trend = "c", opciones del artículo, ajustado en la historia N; estado sin reestimar por una vía
  DISTINTA a la del brazo (que usa append): SARIMAX(y[ini:fin_prueba]).smooth(params) y predicción a un paso no dinámica.
- Chronos-Bolt-tiny: los 12 meses, uno a uno, contexto = últimas 120 observaciones hasta t-1.
- DualLSTM semilla 0: reentrenada aparte con src.redes en un directorio temporal (--red DIR).
Uso: .venv311\\Scripts\\python.exe reportes/auditoria/reproducir_M.py [--red DIR]
Escribe reportes/auditoria/reproduccion_M.csv y reproduccion_M_detalle.csv.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.datos import Escenario, serie  # noqa: E402
from src.ventanas import Ventana, particion  # noqa: E402

SAL = Path(__file__).resolve().parent
warnings.simplefilter("ignore")
REJ = [dict(error=e, trend=t, damped_trend=d, seasonal=s) for e, (t, d), s in
       itertools.product(("add", "mul"), ((None, False), ("add", False), ("add", True)), ("add", "mul"))]
ANIO, N = 2023, 120


def comparar(filas, det, modelo, fechas, mio, pub, semilla=""):
    d = np.abs(np.asarray(mio, float) - np.asarray(pub, float))
    filas.append({"brazo": "M", "escenario": "principal", "modelo": modelo, "semilla": semilla, "anio": ANIO, "N": N, "h": 1,
                  "n_comparados": len(d), "dif_max_abs_t": float(d.max()), "dif_media_abs_t": float(d.mean())})
    det.append(pd.DataFrame({"modelo": modelo, "fecha": fechas, "y_pred_auditor": mio, "y_pred_publicado": pub}))
    print(filas[-1], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", default=None)
    a = ap.parse_args()
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = serie(Escenario("M"))
    y = df.y.to_numpy(float)
    part = particion(df.index, Ventana(ANIO), N)
    fechas = df.index[part.test].strftime("%Y-%m-%d")
    pron = pd.read_csv(RAIZ / "resultados" / "brazo_M" / "principal" / "pronosticos.csv", dtype={"N": str})

    def pub(modelo, semilla=None):
        s = pron[(pron.modelo == modelo) & (pron.anio == ANIO) & (pron.N == str(N)) & (pron.h == 1)]
        if semilla is not None:
            s = s[s.semilla == semilla]
        s = s.set_index("fecha").y_pred
        assert not s.index.duplicated().any()
        return s.reindex(fechas).to_numpy(float)

    filas, det = [], []
    comparar(filas, det, "Seasonal Naive", fechas, y[part.test - 12], pub("Seasonal Naive"))
    # ETS
    mejor = None
    for i, s in enumerate(REJ):
        try:
            f = ETSModel(y[part.hist], seasonal_periods=12, initialization_method="estimated", **s).fit(disp=False, maxiter=1000)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(f.aicc) and (mejor is None or f.aicc < mejor[0]):
            mejor = (float(f.aicc), i, s, np.asarray(f.params, float))
    _, i_ets, s_ets, par = mejor
    e = [float(np.asarray(ETSModel(y[part.ini_hist:t], seasonal_periods=12, initialization_method="estimated", **s_ets)
                          .smooth(par).forecast(1))[-1]) for t in part.test]
    comparar(filas, det, f"ETS (indice rejilla {i_ets})", fechas, e, pub("ETS"))
    # SARIMA por smooth + predicción a un paso
    op = dict(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), trend="c", enforce_stationarity=False,
              enforce_invertibility=False, initialization="approximate_diffuse")
    fit = SARIMAX(y[part.hist], **op).fit(disp=False, maxiter=500)
    tot = y[part.ini_hist: part.test[-1] + 1]
    res = SARIMAX(tot, **op).smooth(fit.params)
    sp = np.asarray(res.get_prediction(start=part.test[0] - part.ini_hist, end=part.test[-1] - part.ini_hist).predicted_mean)
    comparar(filas, det, "SARIMA", fechas, sp, pub("SARIMA"))
    # Chronos
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(2)
    p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", revision="a0e552de83495b5c28c14c71c374f3e33280b340",
                                            device_map="cpu", torch_dtype=torch.float32)
    imed = [float(q) for q in p.quantiles].index(0.5)
    c = [float(p.predict([torch.tensor(y[t - N:t], dtype=torch.float32)], prediction_length=1).numpy()[0, imed, 0])
         for t in part.test]
    comparar(filas, det, "Chronos-Bolt-tiny", fechas, c, pub("Chronos-Bolt-tiny"))
    if a.red:
        r = pd.read_csv(Path(a.red) / "predicciones.csv")
        r = r[r.particion == "test"].set_index("fecha").reindex(fechas)
        comparar(filas, det, "DualLSTM", fechas, r.y_pred.to_numpy(float), pub("DualLSTM", semilla=0), semilla=0)
    pd.DataFrame(filas).to_csv(SAL / "reproduccion_M.csv", index=False, float_format="%.6g")
    pd.concat(det).to_csv(SAL / "reproduccion_M_detalle.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
