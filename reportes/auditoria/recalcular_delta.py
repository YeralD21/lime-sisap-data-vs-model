"""Auditoría día 3: recálculo independiente del MAE pool, ref*(N), Δ(N), su IC 95 % y la etiqueta de la regla (§1, §6).

No importa src.analisis ni src.evaluacion. y_true sale de src.datos.serie(esc).y_eval (NaN = no evaluable).
- MAE por modelo: error absoluto promediado entre semillas por fecha (DA8) y media sobre los días o meses evaluables.
- ref*(N): menor MAE pool entre las 5 referencias.
- IC de Δ: bootstrap pareado iid estratificado por año, B = 10 000, semilla 42 (estratos en orden de año; mismo
  esquema de sorteo que el artículo: un rng.integers(0, n_a, (B, n_a)) por año). Por bloques (sens. 5): bloques
  móviles circulares de 7 días (D) o 3 meses (M) dentro de cada año.
- Etiqueta: regla de §1 aplicada a mano a Δ e IC.
Compara con delta.csv, metricas.csv y decision.csv de cada escenario. Escribe reportes/auditoria/recalculo_delta_{D,M}.csv
y recalculo_decision.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.datos import Escenario, serie  # noqa: E402

SAL = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
REFS = ["Seasonal Naive", "ETS", "Theta", "SARIMA", "Chronos-Bolt-tiny"]
NS = {"D": ["120", "360", "1000", "max"], "M": ["60", "120", "240", "max"]}
ESC = {"principal": dict(), "sens1_estricto": dict(objetivo="estricto"), "sens2_bolsa": dict(objetivo="bolsa"),
       "sens6_desde2006": dict(inicio="2006-01-01"), "sens7_anomalia_publicada": dict(anomalia="publicada")}
# carpeta -> (escenario de la serie, carpeta de pronosticos, h, años, bootstrap)
CASOS = {
    "D": {"principal": ("principal", "principal", 1, range(2021, 2026), "iid"),
          "sens1_estricto": ("sens1_estricto", "sens1_estricto", 1, range(2021, 2026), "iid"),
          "sens2_bolsa": ("sens2_bolsa", "sens2_bolsa", 1, range(2021, 2026), "iid"),
          "sens3_h7": ("principal", "principal", 7, range(2021, 2026), "iid"),
          "sens4_pool2023_2025": ("principal", "principal", 1, range(2023, 2026), "iid"),
          "sens5_bloques": ("principal", "principal", 1, range(2021, 2026), "bloques"),
          "sens6_desde2006": ("sens6_desde2006", "sens6_desde2006", 1, range(2021, 2026), "iid"),
          "sens7_anomalia_publicada": ("sens7_anomalia_publicada", "sens7_anomalia_publicada", 1, range(2021, 2026), "iid")},
    "M": {"principal": ("principal", "principal", 1, range(2021, 2026), "iid"),
          "sens1_estricto": ("sens1_estricto", "sens1_estricto", 1, range(2021, 2026), "iid"),
          "sens4_pool2023_2025": ("principal", "principal", 1, range(2023, 2026), "iid"),
          "sens5_bloques": ("principal", "principal", 1, range(2021, 2026), "bloques"),
          "sens7_anomalia_publicada": ("sens7_anomalia_publicada", "sens7_anomalia_publicada", 1, range(2021, 2026), "iid")},
}
BLOQUE = {"D": 7, "M": 3}


def boot_dif(d, anios_obs, tipo, largo, B=10_000):
    rng = np.random.default_rng(42)
    cols = []
    for a in pd.unique(anios_obs):
        pos = np.flatnonzero(anios_obs == a)
        n = len(pos)
        if tipo == "iid":
            cols.append(pos[rng.integers(0, n, size=(B, n))])
        else:
            k = int(np.ceil(n / largo))
            ini = rng.integers(0, n, size=(B, k))
            rel = (ini[:, :, None] + np.arange(largo)[None, None, :]) % n
            cols.append(pos[rel.reshape(B, -1)[:, :n]])
    idx = np.concatenate(cols, axis=1)
    est = d[idx].mean(axis=1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def etiqueta(filas):
    d = [f["delta"] for f in filas]
    sig = [f["ic_lo"] > 0 or f["ic_hi"] < 0 for f in filas]
    sigpos = [s and x > 0 for s, x in zip(sig, d)]
    mono = all(a > b for a, b in zip(d[:-1], d[1:]))
    if d[-1] > 0 and sig[-1]:
        return "el modelo"
    datos = mono and (not sig[-1] or d[-1] < 0)
    if not any(sigpos):
        return "no concluyente"            # subcaso (también si datos=True: la premisa no se cumple)
    return "los datos" if datos else "no concluyente"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dec_rows = []
    for brazo, casos in CASOS.items():
        out = []
        cache = {}
        for carpeta, (esc, carp_pron, h, anios, tipo) in casos.items():
            df = serie(Escenario(brazo, **ESC[esc]))
            yev = pd.Series(df.y_eval.to_numpy(float), index=df.index.strftime("%Y-%m-%d"))
            if carp_pron not in cache:
                cache[carp_pron] = pd.read_csv(RES / f"brazo_{brazo}" / carp_pron / "pronosticos.csv", dtype={"N": str})
            pron = cache[carp_pron]
            pron = pron[(pron.h == h) & pron.anio.isin(list(anios))].copy()
            pron["y_true"] = pron.fecha.map(yev)
            pron = pron.dropna(subset=["y_true"])
            pron["ae"] = (pron.y_pred - pron.y_true).abs()
            delta_pub = pd.read_csv(RES / f"brazo_{brazo}" / carpeta / "delta.csv", dtype={"N": str}).set_index("N")
            met_pub = pd.read_csv(RES / f"brazo_{brazo}" / carpeta / "metricas.csv", dtype={"N": str, "anio": str})
            met_pub = met_pub[met_pub.anio == "pool"].set_index(["N", "modelo"])
            filas = []
            for N in NS[brazo]:
                s = pron[pron.N == N]
                ae = s.groupby(["modelo", "fecha"]).ae.mean().unstack(0)      # fechas x modelos (media entre semillas)
                if ae.isna().any().any():
                    raise AssertionError(f"{brazo} {carpeta} N={N}: faltan pronósticos")
                mae = ae.mean()
                ref = min(REFS, key=lambda m: (mae[m], REFS.index(m)))
                anios_obs = pd.to_datetime(ae.index).year.to_numpy()
                dif = (ae["DualLSTM"] - ae[ref]).to_numpy()
                lo, hi = boot_dif(dif, anios_obs, tipo, BLOQUE[brazo])
                mae_dif = max(abs(mae[m] - met_pub.loc[(N, m), "MAE"]) for m in mae.index)
                f = {"brazo": brazo, "escenario": carpeta, "N": N, "n_obs": len(ae), "ref_estrella": ref,
                     "MAE_DualLSTM": mae["DualLSTM"], "MAE_ref": mae[ref], "delta": mae["DualLSTM"] - mae[ref],
                     "ic_lo": lo, "ic_hi": hi,
                     "pub_ref_estrella": delta_pub.loc[N, "ref_estrella"], "pub_delta": delta_pub.loc[N, "delta"],
                     "pub_ic_lo": delta_pub.loc[N, "ic_lo"], "pub_ic_hi": delta_pub.loc[N, "ic_hi"], "pub_n": delta_pub.loc[N, "n"],
                     "dif_max_MAE_vs_metricas": mae_dif}
                f["coincide_ref"] = f["ref_estrella"] == f["pub_ref_estrella"]
                f["dif_delta"] = abs(f["delta"] - f["pub_delta"])
                f["dif_ic"] = max(abs(lo - f["pub_ic_lo"]), abs(hi - f["pub_ic_hi"]))
                filas.append(f)
                print({k: f[k] for k in ("brazo", "escenario", "N", "ref_estrella", "delta", "dif_delta", "dif_ic",
                                         "dif_max_MAE_vs_metricas")}, flush=True)
            out += filas
            et = etiqueta(filas)
            dpub = pd.read_csv(RES / f"brazo_{brazo}" / carpeta / "decision.csv").iloc[0]
            dec_rows.append({"brazo": brazo, "escenario": carpeta, "etiqueta_auditor": et, "etiqueta_publicada": dpub.etiqueta,
                             "detalle_publicado": dpub.detalle, "coincide": et == dpub.etiqueta})
        pd.DataFrame(out).to_csv(SAL / f"recalculo_delta_{brazo}.csv", index=False, float_format="%.6g")
    pd.DataFrame(dec_rows).to_csv(SAL / "recalculo_decision.csv", index=False)
    print(pd.DataFrame(dec_rows).to_string())


if __name__ == "__main__":
    main()
