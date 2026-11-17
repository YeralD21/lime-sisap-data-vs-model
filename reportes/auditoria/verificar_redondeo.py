"""Auditoría día 3: redondeo de las tablas Δ(N) de los informes frente al valor EXACTO (no al CSV redondeado a 6 cifras).

Recalcula MAE_DualLSTM, MAE_ref* y Δ a precisión completa desde pronosticos.csv (y_true de src.datos.serie) y compara
con la cifra del informe redondeada a los decimales mostrados. Lista solo las celdas en que el informe no es el redondeo
correcto del valor exacto. Escribe reportes/auditoria/redondeo_delta.csv.
"""
from __future__ import annotations

import re
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.datos import Escenario, serie  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verificar_cifras import num, ruta_csv, tablas  # noqa: E402

ESC = {"principal": ("principal", dict(), 1, range(2021, 2026)), "sens1_estricto": ("sens1_estricto", dict(objetivo="estricto"), 1, range(2021, 2026)),
       "sens2_bolsa": ("sens2_bolsa", dict(objetivo="bolsa"), 1, range(2021, 2026)), "sens3_h7": ("principal", dict(), 7, range(2021, 2026)),
       "sens4_pool2023_2025": ("principal", dict(), 1, range(2023, 2026)), "sens5_bloques": ("principal", dict(), 1, range(2021, 2026)),
       "sens6_desde2006": ("sens6_desde2006", dict(inicio="2006-01-01"), 1, range(2021, 2026)),
       "sens7_anomalia_publicada": ("sens7_anomalia_publicada", dict(anomalia="publicada"), 1, range(2021, 2026))}
filas = []
for brazo in ("M", "D"):
    md = (RAIZ / "reportes" / f"brazo_{brazo}.md").read_text(encoding="utf-8")
    cache = {}
    for linea, cab, fs, ctx in tablas(md):
        if "delta.csv" not in ctx or cab[0] != "N":
            continue
        esc = ruta_csv(ctx, brazo, "delta.csv")
        carp, kw, h, anios = ESC[esc]
        if (carp, h) not in cache:
            df = serie(Escenario(brazo, **kw))
            y = pd.Series(df.y_eval.to_numpy(float), index=df.index.strftime("%Y-%m-%d"))
            p = pd.read_csv(RAIZ / "resultados" / f"brazo_{brazo}" / carp / "pronosticos.csv", dtype={"N": str})
            p = p[p.h == h].copy()
            p["yt"] = p.fecha.map(y)
            p = p.dropna(subset=["yt"])
            p["ae"] = (p.y_pred - p.yt).abs()
            cache[(carp, h)] = p
        p = cache[(carp, h)]
        p = p[p.anio.isin(list(anios))]
        for f in fs:
            s = p[p.N == f[0]]
            mae = s.groupby(["modelo", "fecha"]).ae.mean().groupby("modelo").mean()
            exactos = {"MAE_DualLSTM": mae["DualLSTM"], "MAE_ref": mae[f[1]], "delta": mae["DualLSTM"] - mae[f[1]]}
            for k, (col, v) in zip((2, 3, 4), exactos.items()):
                x, dec = num(f[k])
                correcto = round(v, dec)
                ok = abs(x - correcto) < 10 ** (-dec) / 2
                if not ok:
                    filas.append({"brazo": brazo, "linea_md": linea, "escenario": esc, "N": f[0], "columna": col,
                                  "mostrado": f[k], "valor_exacto": repr(float(v)), "redondeo_correcto": f"{correcto:.{dec}f}"})
t = pd.DataFrame(filas, columns=["brazo", "linea_md", "escenario", "N", "columna", "mostrado", "valor_exacto", "redondeo_correcto"])
t.to_csv(Path(__file__).resolve().parent / "redondeo_delta.csv", index=False)
print(t.to_string())
