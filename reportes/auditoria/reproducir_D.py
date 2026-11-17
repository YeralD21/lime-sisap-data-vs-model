"""Auditoría día 3: reproducción independiente del brazo D (escenario principal).

Escrito a partir de docs/PROTOCOLO.md §3-§5, sin importar los scripts del brazo D. Solo usa src.datos (serie imputada)
y src.ventanas (partición). Procedimiento LITERAL del protocolo (no el atajo de un solo smooth del brazo):
- Seasonal Naive: ŷ_τ = y_{τ-7} (h = 1 y h = 7).
- ETS: 12 especificaciones, menor AICc en la historia N (empate: orden de la rejilla); para cada día τ de la prueba,
  ETSModel(y[ini:τ-h+1]).smooth(params).forecast(h)[-1]  (parámetros fijos, estado recalculado con datos hasta τ-h).
- Chronos-Bolt-tiny: un pronóstico por fecha (sin lotes), contexto = últimas N observaciones hasta τ-h, mediana.
Ventanas: Y = 2024, N = 360 (h = 1 y 7) y, para revisar la inestabilidad de ETS, Y = 2023, N = 120 (h = 1).
La red (DualLSTM, semilla 0, Y = 2024, N = 360, h = 1) se reentrena aparte con src.redes en un directorio temporal
y se compara aquí si se pasa --red DIR.

Uso: .venv311\\Scripts\\python.exe reportes/auditoria/reproducir_D.py [--red DIR]
Escribe reportes/auditoria/reproduccion_D.csv y reproduccion_D_detalle.csv.
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


def ets_aicc(y):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    mejor = None
    for i, s in enumerate(REJ):
        try:
            f = ETSModel(y, seasonal_periods=7, initialization_method="estimated", **s).fit(disp=False, maxiter=1000)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(f.aicc) and (mejor is None or f.aicc < mejor[0]):
            mejor = (float(f.aicc), i, s, np.asarray(f.params, float))
    return mejor


def ets_literal(y, part, h):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    aicc, i, s, par = ets_aicc(y[part.hist])
    pred = []
    for tau in part.test:
        r = ETSModel(y[part.ini_hist: tau - h + 1], seasonal_periods=7, initialization_method="estimated", **s).smooth(par)
        pred.append(float(np.asarray(r.forecast(h), float)[-1]))
    return np.array(pred), i, aicc


def chronos_pred(y, taus, N, h):
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(2)
    p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", revision="a0e552de83495b5c28c14c71c374f3e33280b340",
                                            device_map="cpu", torch_dtype=torch.float32)
    imed = [float(q) for q in p.quantiles].index(0.5)
    out = []
    for tau in taus:
        fin = tau - h + 1
        ctx = y[fin - N: fin]
        q = p.predict([torch.tensor(ctx, dtype=torch.float32)], prediction_length=h).numpy()
        out.append(float(q[0, imed, h - 1]))
    return np.array(out)


def comparar(filas, det, modelo, anio, N, h, fechas, mio, pub, semilla=""):
    d = np.abs(np.asarray(mio, float) - np.asarray(pub, float))
    filas.append({"brazo": "D", "escenario": "principal", "modelo": modelo, "semilla": semilla, "anio": anio, "N": N, "h": h,
                  "n_comparados": len(d), "dif_max_abs_t": float(d.max()), "dif_media_abs_t": float(d.mean())})
    det.append(pd.DataFrame({"modelo": modelo, "anio": anio, "N": N, "h": h, "fecha": fechas, "y_pred_auditor": mio,
                             "y_pred_publicado": pub}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", default=None)
    a = ap.parse_args()
    df = serie(Escenario("D"))
    y = df.y.to_numpy(float)
    pron = pd.read_csv(RAIZ / "resultados" / "brazo_D" / "principal" / "pronosticos.csv", dtype={"N": str})

    def pub(modelo, anio, N, h, fechas, semilla=None):
        s = pron[(pron.modelo == modelo) & (pron.anio == anio) & (pron.N == str(N)) & (pron.h == h)]
        if semilla is not None:
            s = s[s.semilla == semilla]
        s = s.set_index("fecha").y_pred
        if s.index.duplicated().any():
            raise AssertionError(f"filas duplicadas en pronosticos.csv: {modelo} {anio} {N} {h}")
        return s.reindex(fechas).to_numpy(float)

    filas, det, info = [], [], []
    for anio, N, hs in ((2024, 360, (1, 7)), (2023, 120, (1,))):
        part = particion(df.index, Ventana(anio), N)
        fechas = df.index[part.test].strftime("%Y-%m-%d")
        for h in hs:
            sn = y[part.test - 7]
            comparar(filas, det, "Seasonal Naive", anio, N, h, fechas, sn, pub("Seasonal Naive", anio, N, h, fechas))
            e, i, aicc = ets_literal(y, part, h)
            comparar(filas, det, "ETS", anio, N, h, fechas, e, pub("ETS", anio, N, h, fechas))
            info.append({"anio": anio, "N": N, "h": h, "ets_indice_rejilla": i, "ets_aicc": aicc})
            print(filas[-1], flush=True)
    # Chronos: pocas fechas, Y = 2024, N = 360, h = 1 y 7
    part = particion(df.index, Ventana(2024), 360)
    sel = part.test[[0, 1, 59, 182, 300, 365]]
    fechas = df.index[sel].strftime("%Y-%m-%d")
    for h in (1, 7):
        c = chronos_pred(y, sel, 360, h)
        comparar(filas, det, "Chronos-Bolt-tiny", 2024, 360, h, fechas, c, pub("Chronos-Bolt-tiny", 2024, 360, h, fechas))
        print(filas[-1], flush=True)
    if a.red:
        r = pd.read_csv(Path(a.red) / "predicciones.csv")
        r = r[r.particion == "test"]
        fechas = r.fecha.to_numpy()
        comparar(filas, det, "DualLSTM", 2024, 360, 1, fechas, r.y_pred.to_numpy(float),
                 pub("DualLSTM", 2024, 360, 1, fechas, semilla=0), semilla=0)
        u = pd.read_csv(RAIZ / "resultados/brazo_D/redes_unidades/D_principal_1997_faltante/dual/Y2024/N360/h1/s00/predicciones.csv")
        m = r.merge(u[u.particion == "test"], on="fecha", suffixes=("_aud", "_unidad"))
        comparar(filas, det, "DualLSTM (unidad publicada en redes_unidades)", 2024, 360, 1, m.fecha.to_numpy(),
                 m.y_pred_aud.to_numpy(float), m.y_pred_unidad.to_numpy(float), semilla=0)
        print(filas[-2:], flush=True)
    pd.DataFrame(filas).to_csv(SAL / "reproduccion_D.csv", index=False, float_format="%.6g")
    pd.concat(det).to_csv(SAL / "reproduccion_D_detalle.csv", index=False, float_format="%.6f")
    pd.DataFrame(info).to_csv(SAL / "reproduccion_D_ets_seleccion.csv", index=False)


if __name__ == "__main__":
    main()
