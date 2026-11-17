"""Análisis común a los dos brazos a partir de `pronosticos.csv` (docs/PROTOCOLO.md §1, §6, §7).

Formato de `pronosticos.csv` (una fila por pronóstico de prueba):
    escenario, brazo, modelo, semilla (vacía en modelos deterministas), N ("120", ..., "max"), anio, h,
    fecha (fecha OBJETIVO, YYYY-MM-DD), y_pred (toneladas)
SIN y_true: es dato del SISAP (sin licencia de redistribución, §2) y `resultados/` se versiona. El análisis toma
y_true de data/processed con `completar_y_true` (NaN en los días o meses no evaluables).

`analizar(pron, df, brazo, h, ...)` devuelve tablas listas para escribir:
- metricas:        MAE (IC 95 % bootstrap), RMSE, MASE por N y modelo, en el pool y por año;
- delta:           Δ(N) = MAE_DualLSTM − MAE_ref*(N) con IC 95 % (bootstrap pareado) y DM-HLN;
- ranking:         ranking y empates por N (todas las observaciones evaluables);
- top_ranked:      «consistently top-ranked» por N (P50, P60, P75, P90);
- semillas:        dispersión entre semillas de las redes;
- decision:        etiqueta de la regla de decisión (§1).
Los errores de las redes se promedian entre semillas por fecha antes del bootstrap y del DM (§5, DA8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import evaluacion as E
from src.ventanas import Ventana, escala_mase, evaluables, particion

COLUMNAS = ["escenario", "brazo", "modelo", "semilla", "N", "anio", "h", "fecha", "y_pred"]


def completar_y_true(pron: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Añade y_true = y_eval de la serie del escenario (NaN si no evaluable). Si ya existe, se reemplaza."""
    y = pd.Series(df.y_eval.to_numpy(float), index=df.index.strftime("%Y-%m-%d"))
    return pron.drop(columns=["y_true"], errors="ignore").assign(y_true=pron.fecha.astype(str).map(y).to_numpy(float))


def _orden_N(brazo):
    return [str(n) for n in C.TAMANOS[brazo]]


def fechas_evaluables(df: pd.DataFrame, anios) -> pd.DataFrame:
    filas = []
    for a in anios:
        p = particion(df.index, Ventana(a), "max")
        for pos in evaluables(df, p):
            filas.append({"anio": a, "fecha": df.index[pos].strftime("%Y-%m-%d"), "pos": pos,
                          "y_true": float(df.y_eval.iloc[pos])})
    return pd.DataFrame(filas)


def matriz_ae(pron: pd.DataFrame, df: pd.DataFrame, N: str, h: int, anios, modelos) -> tuple[np.ndarray, pd.DataFrame]:
    """AE promediado entre semillas: (modelos, observaciones evaluables). Falla si falta algún pronóstico o si y_true
    no coincide con la serie (tolerancia 1e-6 t)."""
    ref = fechas_evaluables(df, anios)
    sub = pron[(pron.N.astype(str) == N) & (pron.h == h) & pron.anio.isin(anios) & pron.modelo.isin(modelos)]
    sub = sub.assign(ae=(sub.y_pred - sub.y_true).abs())
    g = sub.dropna(subset=["y_true"]).groupby(["modelo", "fecha"]).agg(ae=("ae", "mean"), y_true=("y_true", "first"))
    AE = np.full((len(modelos), len(ref)), np.nan)
    for i, m in enumerate(modelos):
        gm = g.xs(m, level="modelo") if m in g.index.get_level_values(0) else pd.DataFrame(columns=["ae", "y_true"])
        al = gm.reindex(ref.fecha)
        if al.ae.isna().any():
            raise AssertionError(f"N={N} h={h}: faltan {int(al.ae.isna().sum())} pronósticos de {m}.")
        if not np.allclose(al.y_true.to_numpy(float), ref.y_true.to_numpy(float), atol=1e-6, rtol=0):
            raise AssertionError(f"N={N} h={h}: y_true de {m} no coincide con la serie.")
        AE[i] = al.ae.to_numpy(float)
    return AE, ref


def _rmse_modelo(pron, N, h, anios, modelo, fechas):
    s = pron[(pron.N.astype(str) == N) & (pron.h == h) & pron.anio.isin(anios) & (pron.modelo == modelo)
             & pron.fecha.isin(fechas)]
    e = s.y_pred - s.y_true
    if s.semilla.notna().any():                       # RMSE por semilla y media entre semillas (como el artículo)
        return float(s.assign(e=e).groupby("semilla").e.apply(lambda x: np.sqrt(np.mean(x ** 2))).mean())
    return E.rmse(e.to_numpy())


def analizar(pron: pd.DataFrame, df: pd.DataFrame, brazo: str, h: int = 1, anios=C.ANIOS_TEST,
             tipo_boot: str = "iid", modelos=C.MODELOS) -> dict[str, pd.DataFrame]:
    pron = completar_y_true(pron, df)
    anios = [a for a in anios if len(evaluables(df, particion(df.index, Ventana(a), "max")))]
    largo = C.BLOQUE[brazo] if tipo_boot == "bloques" else None
    modelos = list(modelos)
    escalas = {a: escala_mase(df, brazo, Ventana(a)) for a in anios}
    t_met, t_delta, t_rank, t_top, t_top_det, t_sem = [], [], [], [], [], []
    for N in _orden_N(brazo):
        AE, ref = matriz_ae(pron, df, N, h, anios, modelos)
        estr = ref.anio.to_numpy()
        esc = ref.anio.map(escalas).to_numpy()
        boot = E.remuestreos(AE, estr, None, tipo_boot, largo)
        for i, m in enumerate(modelos):
            lo, hi = E.ic(boot[i])
            t_met.append({"N": N, "modelo": m, "anio": "pool", "MAE": AE[i].mean(), "MAE_ic_lo": lo, "MAE_ic_hi": hi,
                          "RMSE": _rmse_modelo(pron, N, h, anios, m, set(ref.fecha)), "MASE": E.mase(AE[i], esc),
                          "n": AE.shape[1]})
            for a in anios:
                k = estr == a
                t_met.append({"N": N, "modelo": m, "anio": a, "MAE": AE[i, k].mean(), "MAE_ic_lo": np.nan,
                              "MAE_ic_hi": np.nan, "RMSE": _rmse_modelo(pron, N, h, [a], m, set(ref.fecha[k])),
                              "MASE": E.mase(AE[i, k], escalas[a]), "n": int(k.sum())})
        mae = {m: float(AE[i].mean()) for i, m in enumerate(modelos)}
        refm = E.mejor_referencia(mae)
        i_d, i_r = modelos.index("DualLSTM"), modelos.index(refm)
        lo, hi = E.ic(boot[i_d] - boot[i_r])
        dm = E.dm_hln(AE[i_d], AE[i_r], h)
        t_delta.append({"N": N, "ref_estrella": refm, "MAE_DualLSTM": mae["DualLSTM"], "MAE_ref": mae[refm],
                        "delta": mae["DualLSTM"] - mae[refm], "ic_lo": lo, "ic_hi": hi,
                        "significativa": bool(lo > 0 or hi < 0), "DM": dm["DM"], "HLN": dm["HLN"], "p_DM": dm["p"],
                        "n": AE.shape[1]})
        r = E.ranking(AE, modelos, estr, None, tipo_boot, largo)
        t_rank.append(r.assign(N=N))
        masks = {}
        for P in C.UMBRALES_P:
            mk = np.zeros(len(ref), bool)
            for a in anios:
                p = particion(df.index, Ventana(a), "max")
                k = (estr == a)
                mk[k], _ = E.mascara_shock(df, brazo, p.val[-1], ref.pos.to_numpy()[k], P)
            masks[P] = mk
        det, res = E.top_ranked(AE, modelos, estr, masks)
        t_top_det.append(det.assign(N=N))
        t_top.append(res.assign(N=N))
        for m in ("LSTM univariada", "DualLSTM"):
            s = pron[(pron.N.astype(str) == N) & (pron.h == h) & pron.anio.isin(anios) & (pron.modelo == m)
                     & pron.fecha.isin(set(ref.fecha))]
            ms = s.assign(ae=(s.y_pred - s.y_true).abs()).groupby("semilla").ae.mean()
            t_sem.append({"N": N, "modelo": m, "n_semillas": len(ms), "MAE_media": ms.mean(), "MAE_de": ms.std(ddof=1),
                          "MAE_min": ms.min(), "MAE_max": ms.max(),
                          "MAE_promedio_semillas": mae[m]})
    delta = pd.DataFrame(t_delta)
    dec = E.regla_decision(delta.to_dict("records"))
    return {"metricas": pd.DataFrame(t_met), "delta": delta, "ranking": pd.concat(t_rank, ignore_index=True),
            "top_ranked": pd.concat(t_top, ignore_index=True), "top_ranked_detalle": pd.concat(t_top_det, ignore_index=True),
            "semillas": pd.DataFrame(t_sem), "decision": pd.DataFrame([{**dec, "brazo": brazo, "h": h,
                                                                        "anios": "-".join(map(str, anios)),
                                                                        "bootstrap": tipo_boot}])}


def escribir(tablas: dict[str, pd.DataFrame], carpeta, sufijo: str = "") -> None:
    from pathlib import Path
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for k, t in tablas.items():
        t.to_csv(carpeta / f"{k}{sufijo}.csv", index=False, float_format="%.6g")
