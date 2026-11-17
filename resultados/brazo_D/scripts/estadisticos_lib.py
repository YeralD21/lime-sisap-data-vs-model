"""Modelos estadísticos y Chronos del brazo D (docs/PROTOCOLO.md §3-§5).

Una «unidad» es (escenario, modelo, año de prueba Y, N) y produce los pronósticos de TODOS los días del año Y
para los horizontes pedidos (h = 1 y/o 7), junto con las incidencias (convergencia, especificación ETS, truncamiento).
Reutiliza `src` (serie, partición, N); no reimplementa imputación ni ventanas.

Equivalencia con el protocolo («parámetros una vez por ventana, estado actualizado con los datos reales, sin reestimar»):
- ETS: `smooth(params)` sobre la serie hasta el final de la prueba y `simulate(h, anchor=τ-h (última observación usada))` con errores cero. El filtro
  de ETS es causal: el estado en τ-h solo depende de los datos hasta τ-h, igual que `smooth(params)` sobre el prefijo.
- SARIMA: `smooth(params)` sobre toda la serie y predicción dinámica desde τ-h+1 (equivale a `append(refit=False)`).
  Las dos equivalencias se comprueban numéricamente en `comprobar_equivalencias.py`.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config as C  # noqa: E402
from src.datos import ESCENARIOS, serie  # noqa: E402
from src.ventanas import Ventana, particion  # noqa: E402

PERIODO = 7
SALIDA = RAIZ / "resultados" / "brazo_D" / "estadisticos"
MODELOS_EST = ("Seasonal Naive", "ETS", "Theta", "SARIMA", "Chronos-Bolt-tiny")
SLUG = {"Seasonal Naive": "SN", "ETS": "ETS", "Theta": "Theta", "SARIMA": "SARIMA", "Chronos-Bolt-tiny": "Chronos"}
REJILLA_ETS = [dict(error=e, trend=t, damped_trend=d, seasonal=s)
               for e, (t, d), s in itertools.product(("add", "mul"), ((None, False), ("add", False), ("add", True)), ("add", "mul"))]


def escenario_D(nombre: str):
    return ESCENARIOS[nombre]["D"]


def nombre_ets(spec: dict) -> str:
    t = "N" if spec["trend"] is None else ("Ad" if spec["damped_trend"] else "A")
    return f"ETS({spec['error'][0].upper()},{t},{spec['seasonal'][0].upper()})"


def _fila(escenario, modelo, N, anio, h, df, pos, pred, semilla=np.nan):
    return pd.DataFrame({"escenario": escenario, "brazo": "D", "modelo": modelo, "semilla": semilla, "N": str(N),
                         "anio": anio, "h": h, "fecha": df.index[pos].strftime("%Y-%m-%d"), "y_pred": np.asarray(pred, float)})


# ----------------------------------------------------------------------------- Seasonal Naive
def seasonal_naive(df, part):
    y = df.y.to_numpy()
    return part.test, y[part.test - 7]          # h = 1 y h = 7: y_{τ-7}


# ----------------------------------------------------------------------------- ETS
def _ajustar_ets(y, spec):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        mod = ETSModel(y, seasonal_periods=PERIODO, initialization_method="estimated", **spec)
        fit = mod.fit(disp=False, maxiter=1000)
    conv = bool(fit.mle_retvals.get("converged", False)) if isinstance(fit.mle_retvals, dict) else None
    return fit, conv, sorted({a.category.__name__ for a in avisos})


def seleccionar_ets(y):
    """Mínimo AICc en la historia N entre las 12 especificaciones (empate: orden de la rejilla)."""
    filas, ajustes = [], {}
    for i, spec in enumerate(REJILLA_ETS):
        fila = {"indice": i, "especificacion": nombre_ets(spec), "aicc": np.nan, "convergio": None, "avisos": "",
                "error_ajuste": "", "segundos": np.nan}
        t0 = time.perf_counter()
        try:
            fit, conv, av = _ajustar_ets(y, spec)
            fila.update(aicc=float(fit.aicc), convergio=conv, avisos=";".join(av))
            ajustes[i] = fit
        except Exception as exc:  # noqa: BLE001
            fila["error_ajuste"] = f"{type(exc).__name__}: {exc}"[:300]
        fila["segundos"] = round(time.perf_counter() - t0, 2)
        filas.append(fila)
    rej = pd.DataFrame(filas)
    ok = rej[np.isfinite(rej.aicc)]
    if ok.empty:
        raise RuntimeError("ETS: ninguna especificación se ajustó")
    el = ok.sort_values(["aicc", "indice"], kind="mergesort").iloc[0]
    return REJILLA_ETS[int(el.indice)], ajustes[int(el.indice)], rej, el


def ets_unidad(df, part, hs, escenario):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    y = df.y.to_numpy()
    spec, fit, rej, el = seleccionar_ets(y[part.hist])
    params = np.asarray(fit.params, float).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # control del ajuste: smooth(params) sobre la historia reproduce forecast(1) de statsmodels
        f0 = float(np.asarray(fit.forecast(1), float)[0])
        r0 = ETSModel(y[part.hist], seasonal_periods=PERIODO, initialization_method="estimated", **spec).smooth(params)
        if not np.isclose(float(np.asarray(r0.forecast(1), float)[0]), f0, rtol=1e-9, atol=1e-6):
            raise AssertionError("ETS: smooth(params) no reproduce forecast(1)")
        y_all = y[part.ini_hist: part.test[-1] + 1]
        res = ETSModel(y_all, seasonal_periods=PERIODO, initialization_method="estimated", **spec).smooth(params)
        out = {}
        for h in hs:
            pred = []
            for tau in part.test:
                a = int(tau - h - part.ini_hist)   # anchor = última observación usada (τ-h)
                s = res.simulate(h, anchor=a, random_errors=np.zeros((h, 1)))
                pred.append(float(np.asarray(s, float).reshape(-1)[-1]))
            out[h] = (part.test, np.array(pred))
    inc = {"especificacion": nombre_ets(spec), "aicc": float(el.aicc), "convergio": el.convergio,
           "n_fallidas": int((rej.error_ajuste != "").sum()), "n_no_convergidas": int((rej.convergio == False).sum()),  # noqa: E712
           "avisos": el.avisos}
    return out, inc, rej.assign(escenario=escenario, anio=part.ventana.anio, N=str(part.N))


# ----------------------------------------------------------------------------- Theta
class ThetaFijo:
    """Como r2lib.ThetaFijo del artículo, con period = 7: b0, alpha y factores estacionales fijos; estado SES actualizado."""

    def __init__(self, y_train):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self.mod = ThetaModel(y_train, period=PERIODO)
        self.res = self.mod.fit()
        self.b0, self.alpha = float(self.res.params["b0"]), float(self.res.params["alpha"])
        self.estacional = np.asarray(self.res._seasonal, dtype=float)
        self.metodo = self.mod.method
        self.desest = bool(self.mod.deseasonalize and self.estacional.shape[0])
        self.n, self.nivel = 0, np.nan
        for v in y_train:
            self.agregar(float(v))
        if not np.isclose(self.nivel, float(np.asarray(self.res._one_step)[0]), rtol=1e-9, atol=1e-6):
            raise AssertionError("Theta: la recursión SES no reproduce one_step")
        for h in (1, 7):
            if not np.isclose(self.pronostico(h), float(np.asarray(self.res.forecast(h))[-1]), rtol=1e-9, atol=1e-6):
                raise AssertionError(f"Theta: pronóstico manual a h={h} no reproduce ThetaModelResults.forecast")

    def _factor(self, i):
        return float(self.estacional[i % PERIODO])

    def agregar(self, y):
        x = (y / self._factor(self.n) if self.metodo.startswith("mul") else y - self._factor(self.n)) if self.desest else y
        if self.n == 0:
            self.nivel = x
        self.nivel = self.alpha * x + (1 - self.alpha) * self.nivel
        self.n += 1

    def pronostico(self, h=1):
        hh = (h - 1) + (1 / self.alpha - (1 - self.alpha) ** self.n / self.alpha if self.alpha > 0 else 0.0)
        f = 0.5 * self.b0 * hh + self.nivel
        if self.desest:
            i = self.n + h - 1
            f = f * self._factor(i) if self.metodo.startswith("mul") else f + self._factor(i)
        return float(f)


def theta_unidad(df, part, hs):
    y = df.y.to_numpy()
    out, inc = {}, {}
    for h in hs:
        th = ThetaFijo(y[part.hist])
        p0 = (th.b0, th.alpha, th.estacional.copy())
        sig, pred = part.fin_train + 1, []
        for tau in part.test:
            while sig <= tau - h:
                th.agregar(float(y[sig]))
                sig += 1
            pred.append(th.pronostico(h))
        if not (p0[0] == th.b0 and p0[1] == th.alpha and np.array_equal(p0[2], th.estacional)):
            raise AssertionError("Theta: los parámetros cambiaron")
        out[h] = (part.test, np.array(pred))
        inc = {"b0": th.b0, "alpha": th.alpha, "metodo_estacional": th.metodo, "desestacionaliza": th.desest}
    return out, inc


# ----------------------------------------------------------------------------- SARIMA
def fourier(pos, K=3):
    t = np.asarray(pos, float)
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t / 365.25), np.cos(2 * np.pi * k * t / 365.25)]
    return np.column_stack(cols)


def _sarimax(y, ex):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    return SARIMAX(np.log(y), exog=ex, order=(1, 0, 1), seasonal_order=(0, 1, 1, PERIODO), enforce_stationarity=False,
                   enforce_invertibility=False, initialization="approximate_diffuse")


def sarima_unidad(df, part, hs):
    y = df.y.to_numpy()
    usa_f = part.n_hist >= 730
    ex_h = fourier(np.arange(part.ini_hist, part.fin_train + 1)) if usa_f else None
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        fit = _sarimax(y[part.hist], ex_h).fit(disp=False, maxiter=500)
    rv = fit.mle_retvals if isinstance(fit.mle_retvals, dict) else {}
    inc = {"convergio": bool(rv.get("converged", False)), "iteraciones": rv.get("iterations"), "warnflag": rv.get("warnflag"),
           "llf": float(fit.llf), "fourier": usa_f, "avisos": ";".join(sorted({a.category.__name__ for a in avisos})),
           "params": ";".join(f"{n}={v:.6g}" for n, v in zip(fit.param_names, fit.params))}
    pos_all = np.arange(part.ini_hist, part.test[-1] + 1)
    ex_a = fourier(pos_all) if usa_f else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _sarimax(y[pos_all], ex_a).smooth(fit.params)
        out = {}
        for h in hs:
            if h == 1:
                p = np.asarray(res.get_prediction(start=part.test[0] - part.ini_hist, end=part.test[-1] - part.ini_hist).predicted_mean)
            else:
                p = np.array([float(np.asarray(res.get_prediction(start=int(tau - h + 1 - part.ini_hist), end=int(tau - part.ini_hist),
                                                                  dynamic=0).predicted_mean)[-1]) for tau in part.test])
            out[h] = (part.test, np.exp(p))
    return out, inc


# ----------------------------------------------------------------------------- Chronos
_PIPE = {}


def _pipeline():
    if "p" not in _PIPE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(2)
        p = BaseChronosPipeline.from_pretrained(C.CHRONOS_ID, revision=C.CHRONOS_SNAPSHOT, device_map="cpu",
                                                torch_dtype=torch.float32)
        q = getattr(p, "quantiles", None) or getattr(p.model, "quantiles", None)
        _PIPE.update(p=p, torch=torch, imed=[float(x) for x in q].index(0.5),
                     ctx=int(p.model.config.chronos_config["context_length"]))
    return _PIPE


def chronos_unidad(df, part, hs, lote=64):
    P = _pipeline()
    torch, pipe = P["torch"], P["p"]
    y = df.y.to_numpy()
    out, inc = {}, {"context_length_modelo": P["ctx"]}
    for h in hs:
        ctxs, trunc = [], 0
        for tau in part.test:
            fin = tau - h + 1
            ini = 0 if part.N == "max" else fin - int(part.N)
            c = y[ini:fin]
            if len(c) > P["ctx"]:
                trunc += 1
                c = c[-P["ctx"]:]
            ctxs.append(torch.tensor(c, dtype=torch.float32))
        pred = []
        for i in range(0, len(ctxs), lote):
            with torch.inference_mode():
                q = pipe.predict(ctxs[i:i + lote], prediction_length=h).detach().cpu().numpy()
            pred.extend(q[:, P["imed"], h - 1].astype(float))
        out[h] = (part.test, np.array(pred))
        inc[f"n_truncados_h{h}"] = trunc
        inc[f"n_pronosticos_h{h}"] = len(part.test)
    return out, inc


# ----------------------------------------------------------------------------- unidad
def archivos(escenario: str, modelo: str, anio: int, N):
    d = SALIDA / escenario / "unidades"
    base = f"{SLUG[modelo]}_Y{anio}_N{N}"
    return d / f"{base}.csv", d / f"{base}.json", d / f"{base}_rejilla.csv"


def correr_unidad(escenario: str, modelo: str, anio: int, N, hs) -> dict:
    """Ejecuta y guarda una unidad (reanudable). Devuelve el resumen."""
    f_csv, f_json, f_rej = archivos(escenario, modelo, anio, N)
    if f_csv.exists() and f_json.exists():
        return {"reanudada": True, "unidad": f_csv.name}
    f_csv.parent.mkdir(parents=True, exist_ok=True)
    df = serie(escenario_D(escenario))
    Nn = N if N == "max" else int(N)
    part = particion(df.index, Ventana(anio), Nn)
    t0 = time.perf_counter()
    rej = None
    if modelo == "Seasonal Naive":
        pos, pr = seasonal_naive(df, part)
        out, inc = {h: (pos, pr) for h in hs}, {}
    elif modelo == "ETS":
        out, inc, rej = ets_unidad(df, part, hs, escenario)
    elif modelo == "Theta":
        out, inc = theta_unidad(df, part, hs)
    elif modelo == "SARIMA":
        out, inc = sarima_unidad(df, part, hs)
    elif modelo == "Chronos-Bolt-tiny":
        out, inc = chronos_unidad(df, part, hs)
    else:
        raise ValueError(modelo)
    seg = time.perf_counter() - t0
    pron = pd.concat([_fila(escenario, modelo, N, anio, h, df, pos, pr) for h, (pos, pr) in out.items()], ignore_index=True)
    if not np.isfinite(pron.y_pred).all():
        raise AssertionError(f"{modelo} Y{anio} N{N}: pronósticos no finitos")
    if rej is not None:
        rej.to_csv(f_rej, index=False)
    pron.to_csv(f_csv, index=False, float_format="%.10g")
    info = {"escenario": escenario, "modelo": modelo, "anio": anio, "N": str(N), "hs": list(hs), "n_hist": part.n_hist,
            "segundos": round(seg, 2), **inc}
    f_json.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return info
