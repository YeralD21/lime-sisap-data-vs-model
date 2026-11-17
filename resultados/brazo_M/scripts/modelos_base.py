"""Modelos de referencia del brazo M a h = 1 (docs/PROTOCOLO.md §5, columna «Brazo M»).

Seasonal Naive, ETS (12 especificaciones, menor AICc), Theta (θ = 2), SARIMA congelado del artículo y Chronos-Bolt-tiny.
Todos reciben la misma serie imputada `src.datos.serie(esc).y`. Los parámetros se estiman una vez por ventana con la
historia de tamaño N (`src.ventanas.particion`) y el estado se actualiza con los datos reales hasta t-1, sin reestimar.

Referencias de implementación (solo lectura, repositorio del artículo, commit eee9a71):
- ETS: `_ajustar_ets`, `_pronostico_ets`, `seleccionar_ets`, `ets_rolling` de
  experimentos_backtesting_rodante/13_revision_r2/scripts/r2lib.py;
- Theta: `ThetaFijo` del mismo archivo (copiada aquí);
- SARIMA: `lib/modelos_series.py` (`_ajustar`, `_a_un_paso`): forecast(steps=1) y append(refit=False).
"""
from __future__ import annotations

import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd

from src import config as C
from src.ventanas import Ventana, particion

PERIODO = 12
REJILLA_ETS = [dict(error=e, trend=t, damped_trend=d, seasonal=s)
               for e, (t, d), s in itertools.product(("add", "mul"), ((None, False), ("add", False), ("add", True)), ("add", "mul"))]
SARIMA_ORDEN, SARIMA_ESTACIONAL = (1, 1, 1), (1, 1, 0, 12)
SARIMA_OPCIONES = dict(trend="c", enforce_stationarity=False, enforce_invertibility=False,
                       initialization="approximate_diffuse")
SARIMA_MAXITER = 500
CHRONOS_CONTEXTO_MAX = 2048


def nombre_ets(spec: dict) -> str:
    t = "N" if spec["trend"] is None else ("Ad" if spec["damped_trend"] else "A")
    return f"ETS({spec['error'][0].upper()},{t},{spec['seasonal'][0].upper()})"


# ----------------------------------------------------------------------------- Seasonal Naive

def seasonal_naive(y: np.ndarray, p) -> tuple[np.ndarray, list]:
    return np.array([y[t - PERIODO] for t in p.test], float), []


# ----------------------------------------------------------------------------- ETS

def _ajustar_ets(y: np.ndarray, spec: dict):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        mod = ETSModel(y, seasonal_periods=PERIODO, initialization_method="estimated", **spec)
        fit = mod.fit(disp=False, maxiter=1000)
    conv = bool(fit.mle_retvals.get("converged", False)) if isinstance(fit.mle_retvals, dict) else None
    return fit, conv, sorted({a.category.__name__ for a in avisos})


def _pronostico_ets(y_hasta: np.ndarray, spec: dict, params: np.ndarray) -> float:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ETSModel(y_hasta, seasonal_periods=PERIODO, initialization_method="estimated", **spec).smooth(params)
        return float(np.asarray(r.forecast(1), dtype=float)[-1])


def seleccionar_ets(y: np.ndarray):
    """Mínimo AICc en la historia N entre las 12 especificaciones (empate: orden de la rejilla)."""
    filas, ajustes = [], {}
    for i, spec in enumerate(REJILLA_ETS):
        fila = {"indice": i, "modelo": nombre_ets(spec), "aicc": np.nan, "convergio": None, "avisos": "", "error_ajuste": ""}
        try:
            fit, conv, av = _ajustar_ets(y, spec)
            aicc = float(fit.aicc)
            if np.isfinite(aicc):
                ajustes[i] = fit
            else:
                fila["error_ajuste"] = "AICc no finito"
            fila.update(aicc=aicc, convergio=conv, avisos=";".join(av))
        except Exception as exc:  # una especificación que falla se descarta y se registra
            fila["error_ajuste"] = f"{type(exc).__name__}: {exc}"[:300]
        filas.append(fila)
    rejilla = pd.DataFrame(filas)
    validos = rejilla[np.isfinite(rejilla.aicc)]
    if validos.empty:
        raise RuntimeError("Ninguna especificación ETS ajustó")
    el = validos.sort_values(["aicc", "indice"], kind="mergesort").iloc[0]
    i = int(el.indice)
    return REJILLA_ETS[i], ajustes[i], rejilla, el


def ets(y: np.ndarray, p) -> tuple[np.ndarray, list]:
    yh = y[p.hist]
    spec, fit, rejilla, el = seleccionar_ets(yh)
    params = np.asarray(fit.params, dtype=float).copy()
    f0 = float(np.asarray(fit.forecast(1), dtype=float)[0])           # control como r2lib.ets_rolling
    if not np.isclose(_pronostico_ets(yh, spec, params), f0, rtol=1e-9, atol=1e-6):
        raise AssertionError("ETS: smooth(params) no reproduce forecast(1) del ajuste")
    pred = np.array([_pronostico_ets(y[p.ini_hist:t], spec, params) for t in p.test], float)
    n_fall = int((rejilla.error_ajuste != "").sum())
    n_noc = int((rejilla.convergio == False).sum())  # noqa: E712
    inc = [("ets_seleccion", nombre_ets(spec),
            f"aicc={el.aicc:.4f}; convergio={el.convergio}; n_especificaciones=12; n_fallidas={n_fall}; n_no_convergidas={n_noc}")]
    for r in rejilla.itertuples():
        if r.error_ajuste:
            inc.append(("ets_especificacion_fallida", r.modelo, r.error_ajuste))
        elif r.convergio is False:
            inc.append(("ets_especificacion_no_convergio", r.modelo, f"aicc={r.aicc:.4f}; avisos={r.avisos}"))
    if not np.isfinite(pred).all():
        inc.append(("ets_pronostico_no_finito", nombre_ets(spec), f"{int((~np.isfinite(pred)).sum())} pronósticos no finitos"))
    return pred, inc


# ----------------------------------------------------------------------------- Theta (copia de r2lib.ThetaFijo)

class ThetaFijo:
    """Theta (θ = 2) con b0, α y factores estacionales fijados en la historia N; estado SES actualizado con cada dato."""

    def __init__(self, y_train: np.ndarray):
        from statsmodels.tsa.forecasting.theta import ThetaModel

        self.mod = ThetaModel(y_train, period=PERIODO)
        self.res = self.mod.fit()
        self.b0, self.alpha = float(self.res.params["b0"]), float(self.res.params["alpha"])
        self.estacional = np.asarray(self.res._seasonal, dtype=float)
        self.metodo = self.mod.method
        self.desestacionaliza = bool(self.mod.deseasonalize and self.estacional.shape[0])
        self.n = 0
        self.nivel = np.nan
        for v in y_train:
            self.agregar(float(v))
        if not np.isclose(self.nivel, float(np.asarray(self.res._one_step)[0]), rtol=1e-9, atol=1e-6):
            raise AssertionError("Theta: la recursión SES no reproduce el one_step de statsmodels")
        if not np.isclose(self.pronostico(), float(np.asarray(self.res.forecast(1))[0]), rtol=1e-9, atol=1e-6):
            raise AssertionError("Theta: el pronóstico manual no reproduce ThetaModelResults.forecast(1)")

    def _factor(self, i: int) -> float:
        return float(self.estacional[i % PERIODO])

    def agregar(self, y: float) -> None:
        if self.desestacionaliza:
            x = y / self._factor(self.n) if self.metodo.startswith("mul") else y - self._factor(self.n)
        else:
            x = y
        if self.n == 0:
            self.nivel = x
        self.nivel = self.alpha * x + (1 - self.alpha) * self.nivel
        self.n += 1

    def pronostico(self, h: int = 1) -> float:
        hh = (h - 1) + (1 / self.alpha - (1 - self.alpha) ** self.n / self.alpha if self.alpha > 0 else 0.0)
        f = 0.5 * self.b0 * hh + self.nivel
        i = self.n + h - 1
        if self.desestacionaliza:
            f = f * self._factor(i) if self.metodo.startswith("mul") else f + self._factor(i)
        return float(f)


def theta(y: np.ndarray, p) -> tuple[np.ndarray, list]:
    th = ThetaFijo(y[p.hist])
    p0 = (th.b0, th.alpha, th.estacional.copy())
    pred = []
    for t in range(p.fin_train + 1, p.test[-1] + 1):       # validación y prueba: estado con datos reales hasta t-1
        if t >= p.test[0]:
            pred.append(th.pronostico())
        th.agregar(float(y[t]))
    if not (p0[0] == th.b0 and p0[1] == th.alpha and np.array_equal(p0[2], th.estacional)):
        raise AssertionError("Theta: los parámetros cambiaron durante la actualización")
    return np.array(pred, float), [("theta_parametros", th.metodo,
                                    f"alpha={th.alpha:.6f}; b0={th.b0:.6f}; desestacionaliza={th.desestacionaliza}")]


# ----------------------------------------------------------------------------- SARIMA congelado

def sarima(y: np.ndarray, p) -> tuple[np.ndarray, list]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        fit = SARIMAX(y[p.hist], order=SARIMA_ORDEN, seasonal_order=SARIMA_ESTACIONAL, **SARIMA_OPCIONES).fit(
            disp=False, maxiter=SARIMA_MAXITER)
    conv = bool(fit.mle_retvals.get("converged", False)) if isinstance(fit.mle_retvals, dict) else None
    av = sorted({f"{a.category.__name__}: {str(a.message)[:120]}" for a in avisos})
    estado = fit.append(y[p.val], refit=False)             # la validación solo actualiza el estado
    pred = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in p.test:
            pred.append(float(np.asarray(estado.forecast(steps=1), dtype=float)[0]))
            estado = estado.append([float(y[t])], refit=False)
    inc = [("sarima_convergencia", str(conv), f"aic={fit.aic:.4f}; avisos={' | '.join(av) if av else 'ninguno'}")]
    return np.array(pred, float), inc


# ----------------------------------------------------------------------------- Chronos-Bolt-tiny

class ChronosFijo:
    """Chronos-Bolt-tiny con la revisión fijada (protocolo §5), sin red (HF_HUB_OFFLINE=1), CPU, float32, 2 hilos."""

    def __init__(self) -> None:
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE debe ser 1")
        import torch
        from chronos import BaseChronosPipeline

        self.torch = torch
        torch.set_num_threads(2)
        self.pipeline = BaseChronosPipeline.from_pretrained(C.CHRONOS_ID, revision=C.CHRONOS_SNAPSHOT, device_map="cpu",
                                                            torch_dtype=torch.float32)
        q = getattr(self.pipeline, "quantiles", None) or getattr(getattr(self.pipeline, "model", None), "quantiles", None)
        self.cuantiles = [float(x) for x in q]
        self.i_mediana = self.cuantiles.index(0.5)

    def predecir(self, contexto: np.ndarray) -> float:
        x = self.torch.tensor(np.asarray(contexto, dtype=float), dtype=self.torch.float32)
        with self.torch.inference_mode():
            arr = self.pipeline.predict(x, prediction_length=1).detach().cpu().numpy()
        return float(arr[0, self.i_mediana, 0] if arr.ndim == 3 else arr[self.i_mediana, 0])


_CHRONOS = None


def chronos(y: np.ndarray, p) -> tuple[np.ndarray, list]:
    """Contexto = las últimas N observaciones hasta t-1 (N = n_hist de la partición; en N = max, la longitud de la
    historia máxima de la ventana). Se registra si se trunca (máximo del modelo: 2048)."""
    global _CHRONOS
    if _CHRONOS is None:
        _CHRONOS = ChronosFijo()
    n = p.n_hist
    ctx = min(n, CHRONOS_CONTEXTO_MAX)
    pred = np.array([_CHRONOS.predecir(y[t - n:t][-ctx:]) for t in p.test], float)
    return pred, [("chronos_contexto", str(n), f"contexto_solicitado={n}; contexto_usado={ctx}; truncado={n > CHRONOS_CONTEXTO_MAX}")]


MODELOS_BASE = {"Seasonal Naive": seasonal_naive, "ETS": ets, "Theta": theta, "SARIMA": sarima, "Chronos-Bolt-tiny": chronos}


def correr_base(df: pd.DataFrame, nombre_escenario: str, anios, Ns, modelos=tuple(MODELOS_BASE), log=print):
    """Devuelve (pronósticos, incidencias, tiempos) como DataFrames."""
    y = df.y.to_numpy(float)
    filas, incs, tiempos = [], [], []
    for anio in anios:
        for N in Ns:
            p = particion(df.index, Ventana(anio), N)
            for m in modelos:
                t0 = time.perf_counter()
                try:
                    pred, inc = MODELOS_BASE[m](y, p)
                except Exception as exc:  # fallo total: se registra y se deja sin pronóstico (analizar fallará: hay que verlo)
                    log(f"FALLO {nombre_escenario} {m} Y={anio} N={N}: {type(exc).__name__}: {exc}")
                    incs.append({"escenario": nombre_escenario, "anio": anio, "N": str(N), "modelo": m, "tipo": "fallo_total",
                                 "valor": type(exc).__name__, "detalle": str(exc)[:300]})
                    continue
                seg = time.perf_counter() - t0
                for t, v in zip(p.test, pred):
                    filas.append({"escenario": nombre_escenario, "brazo": "M", "modelo": m, "semilla": np.nan, "N": str(N),
                                  "anio": anio, "h": 1, "fecha": df.index[t].strftime("%Y-%m-%d"), "y_pred": float(v)})
                for tipo, valor, detalle in inc:
                    incs.append({"escenario": nombre_escenario, "anio": anio, "N": str(N), "modelo": m, "tipo": tipo,
                                 "valor": valor, "detalle": detalle})
                tiempos.append({"escenario": nombre_escenario, "anio": anio, "N": str(N), "modelo": m, "segundos": round(seg, 3)})
                log(f"{nombre_escenario} Y={anio} N={N} {m}: {seg:.1f}s")
    return pd.DataFrame(filas), pd.DataFrame(incs), pd.DataFrame(tiempos)
