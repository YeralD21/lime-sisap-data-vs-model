"""Métricas, bootstrap pareado, Diebold-Mariano HLN, «consistently top-ranked» y regla de decisión
(docs/PROTOCOLO.md §1, §5 «Agregación de semillas», §6).

Implementación del bootstrap, ranking y DM-HLN copiada de lib/evaluacion.py del artículo
(experimentos_backtesting_rodante/02_scripts/lib/evaluacion.py, commit eee9a71). Cambios:
- DM-HLN admite h > 1 con autocovarianzas hasta h-1 (el artículo solo tenía h = 1; para h = 1 es idéntico);
- la longitud del bloque es un parámetro (7 días en D, 3 meses en M).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C


# ----------------------------------------------------------------------------- métricas puntuales

def mae(ae: np.ndarray) -> float:
    return float(np.mean(ae))


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def mase(ae: np.ndarray, escala: np.ndarray | float) -> float:
    return float(np.mean(np.asarray(ae, float) / escala))


def promediar_semillas(pron: pd.DataFrame, claves=("fecha",)) -> pd.DataFrame:
    """§5 (DA8): error absoluto promediado entre semillas. `pron` con columnas y_true, y_pred, semilla."""
    p = pron.assign(ae=(pron.y_pred - pron.y_true).abs())
    return p.groupby(list(claves), as_index=False).agg(ae=("ae", "mean"), y_true=("y_true", "first"),
                                                       n_semillas=("semilla", "nunique"))


# ----------------------------------------------------------------------------- bootstrap

def indices_iid(estratos: np.ndarray, subconjunto: np.ndarray, rng: np.random.Generator, b: int = C.B) -> np.ndarray:
    """Remuestreo iid pareado, estratificado por ventana, dentro de `subconjunto` (bool)."""
    cols = []
    for g in pd.unique(estratos):
        pos = np.flatnonzero((estratos == g) & subconjunto)
        if len(pos):
            cols.append(pos[rng.integers(0, len(pos), size=(b, len(pos)))])
    return np.concatenate(cols, axis=1)


def indices_bloques(estratos: np.ndarray, rng: np.random.Generator, largo: int, b: int = C.B) -> np.ndarray:
    """Bloques móviles circulares de `largo` observaciones dentro de cada ventana (sensibilidad 5)."""
    cols = []
    for g in pd.unique(estratos):
        pos = np.flatnonzero(estratos == g)
        n = len(pos)
        k = int(np.ceil(n / largo))
        inicios = rng.integers(0, n, size=(b, k))
        rel = (inicios[:, :, None] + np.arange(largo)[None, None, :]) % n
        cols.append(pos[rel.reshape(b, -1)[:, :n]])
    return np.concatenate(cols, axis=1)


def remuestrear(AE: np.ndarray, peso: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Media ponderada (peso 0/1) de cada modelo en cada remuestreo: (modelos, B)."""
    w = peso[idx].astype(float)
    num = np.einsum("mbn,bn->mb", AE[:, idx], w)
    den = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den[None, :]


def ic(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def remuestreos(AE: np.ndarray, estratos: np.ndarray, subconj: np.ndarray | None = None,
                tipo: str = "iid", largo_bloque: int | None = None) -> np.ndarray:
    """MAE de cada modelo (filas de AE) en cada uno de los B remuestreos; mismos índices para todos (pareado).
    Cada llamada reinicia el generador con la semilla 42, como en el artículo."""
    subconj = np.ones(AE.shape[1], bool) if subconj is None else np.asarray(subconj, bool)
    rng = np.random.default_rng(C.SEMILLA_BOOT)
    if tipo == "iid":
        return AE[:, indices_iid(estratos, subconj, rng)].mean(axis=2)
    if largo_bloque is None:
        raise ValueError("largo_bloque es obligatorio con tipo='bloques'.")
    return remuestrear(AE, subconj, indices_bloques(estratos, rng, largo_bloque))


def diferencia(AE: np.ndarray, i: int, j: int, estratos: np.ndarray, subconj=None, tipo="iid", largo_bloque=None) -> dict:
    """MAE_i - MAE_j con IC 95 % por percentiles del bootstrap pareado. Significativa si el IC excluye 0 (§1)."""
    sub = np.ones(AE.shape[1], bool) if subconj is None else np.asarray(subconj, bool)
    est = remuestreos(AE, estratos, sub, tipo, largo_bloque)
    lo, hi = ic(est[i] - est[j])
    d = float(AE[i, sub].mean() - AE[j, sub].mean())
    return {"delta": d, "ic_lo": lo, "ic_hi": hi, "significativa": bool(lo > 0 or hi < 0), "n": int(sub.sum())}


# ----------------------------------------------------------------------------- Diebold-Mariano

def dm_hln(ae_a: np.ndarray, ae_b: np.ndarray, h: int = 1) -> dict:
    """DM con corrección HLN, pérdida absoluta, bilateral, t(n-1). Varianza de largo plazo con autocovarianzas
    hasta h-1 (para h = 1 coincide con el artículo)."""
    d = np.asarray(ae_a, float) - np.asarray(ae_b, float)
    n = len(d)
    if n < 3:
        return {"n": n, "d_media": float(d.mean()) if n else np.nan, "DM": np.nan, "HLN": np.nan, "p": np.nan}
    dc = d - d.mean()
    gam = [float(np.mean(dc[k:] * dc[:n - k])) for k in range(h)]
    var = gam[0] + 2 * sum(gam[1:])
    if var <= 0:
        return {"n": n, "d_media": float(d.mean()), "DM": np.nan, "HLN": np.nan, "p": np.nan}
    dm = d.mean() / np.sqrt(var / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return {"n": n, "d_media": float(d.mean()), "DM": float(dm), "HLN": float(hln), "p": float(p)}


# ----------------------------------------------------------------------------- ranking y top-ranked

def ranking(AE: np.ndarray, nombres, estratos: np.ndarray, subconj: np.ndarray | None = None,
            tipo: str = "iid", largo_bloque: int | None = None) -> pd.DataFrame:
    """Ranking por MAE en `subconj`. «primero» = el mejor y los que empatan con él (IC 95 % de
    MAE_modelo - MAE_mejor contiene 0)."""
    sub = np.ones(AE.shape[1], bool) if subconj is None else np.asarray(subconj, bool)
    if not sub.any():
        return pd.DataFrame({"modelo": list(nombres), "MAE": np.nan, "posicion": pd.NA, "dif_vs_mejor_lo": np.nan,
                             "dif_vs_mejor_hi": np.nan, "primero": False, "n": 0})
    est = remuestreos(AE, estratos, sub, tipo, largo_bloque)
    m = AE[:, sub].mean(axis=1)
    mejor = int(np.argmin(m))
    filas = []
    for i, nom in enumerate(nombres):
        lo, hi = ic(est[i] - est[mejor])
        filas.append({"modelo": nom, "MAE": float(m[i]), "dif_vs_mejor_lo": lo, "dif_vs_mejor_hi": hi,
                      "primero": bool(i == mejor or (lo <= 0 <= hi)), "n": int(sub.sum())})
    out = pd.DataFrame(filas)
    out.insert(2, "posicion", out.MAE.rank(method="min").astype(int))
    return out


def variacion_relativa(y: np.ndarray, k: int) -> np.ndarray:
    """|y_t - y_{t-k}| / y_{t-k}; NaN en las primeras k posiciones. k = 7 en D (adaptación declarada), 1 en M."""
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    out[k:] = np.abs(y[k:] - y[:-k]) / y[:-k]
    return out


def mascara_shock(df: pd.DataFrame, brazo: str, origen_pos: int, pos_eval: np.ndarray, percentil: int) -> tuple[np.ndarray, float]:
    """§6: umbral = percentil lineal de la variación relativa con la historia hasta el origen (fin de Y-1, incluido),
    calculada sobre la serie imputada; día o mes de prueba en shock si su variación (valor real frente a t-k
    imputado) es estrictamente mayor que el umbral."""
    k = 7 if brazo == "D" else 1
    hist = variacion_relativa(df.y.to_numpy()[: origen_pos + 1], k)
    umbral = float(np.percentile(hist[np.isfinite(hist)], percentil, method="linear"))
    y_eval = df.y_eval.to_numpy()
    y_imp = df.y.to_numpy()
    var = np.abs(y_eval[pos_eval] - y_imp[pos_eval - k]) / y_imp[pos_eval - k]
    return var > umbral, umbral


def top_ranked(AE: np.ndarray, nombres, estratos: np.ndarray, mascaras: dict[int, np.ndarray]) -> pd.DataFrame:
    """«Consistently top-ranked»: primero o empatado con el primero en al menos 3 de los 4 umbrales."""
    tablas = []
    for p, sub in mascaras.items():
        r = ranking(AE, nombres, estratos, sub)
        r.insert(0, "percentil", p)
        tablas.append(r)
    t = pd.concat(tablas, ignore_index=True)
    res = t.groupby("modelo", sort=False).primero.sum().rename("umbrales_primero").reset_index()
    res["consistently_top_ranked"] = res.umbrales_primero >= C.MIN_UMBRALES_TOP
    return t, res


# ----------------------------------------------------------------------------- regla de decisión

def mejor_referencia(mae_por_modelo: dict[str, float]) -> str:
    """§1: ref*(N) = referencia con menor MAE agregado en ese N (empate: orden de C.REFERENCIAS)."""
    cands = [(mae_por_modelo[m], k, m) for k, m in enumerate(C.REFERENCIAS) if m in mae_por_modelo]
    if len(cands) != len(C.REFERENCIAS):
        faltan = set(C.REFERENCIAS) - set(mae_por_modelo)
        raise ValueError(f"Faltan referencias para elegir ref*: {faltan}")
    return min(cands)[2]


def regla_decision(deltas: list[dict]) -> dict:
    """§1. `deltas`: lista ordenada por N creciente (el último es N = máx) con claves delta, ic_lo, ic_hi.
    Significativa = IC 95 % excluye 0. Orden de aplicación:
      1. «el modelo»: Δ(máx) > 0 y significativa;
      2. subcaso: Δ no es significativamente positiva en ningún N → «no concluyente (no se reproduce la desventaja)»;
      3. «los datos»: Δ estrictamente decreciente en N y, en N = máx, no significativa o Δ(máx) < 0;
      4. resto: «no concluyente»."""
    d = [x["delta"] for x in deltas]
    sig = [(x["ic_lo"] > 0) or (x["ic_hi"] < 0) for x in deltas]
    sig_pos = [s and x["delta"] > 0 for s, x in zip(sig, deltas)]
    monotona = all(a > b for a, b in zip(d[:-1], d[1:]))
    ultimo, sig_ult = d[-1], sig[-1]
    base = {"monotona_decreciente": monotona, "delta_max": ultimo, "significativa_max": sig_ult,
            "alguna_significativa_positiva": any(sig_pos)}
    if ultimo > 0 and sig_ult:
        return {**base, "etiqueta": "el modelo", "detalle": "Δ(máx) > 0 y significativa"}
    if not any(sig_pos):
        return {**base, "etiqueta": "no concluyente", "detalle": "no se reproduce la desventaja"}
    if monotona and (not sig_ult or ultimo < 0):
        return {**base, "etiqueta": "los datos", "detalle": "Δ estrictamente decreciente y Δ(máx) no significativa o < 0"}
    return {**base, "etiqueta": "no concluyente", "detalle": "otro patrón"}


def conclusion_conjunta(etiqueta_D: str, etiqueta_M: str) -> str:
    return etiqueta_D if etiqueta_D == etiqueta_M else "no concluyente"
