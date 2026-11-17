"""Pruebas mínimas del código común (src/). Ejecutar: .venv311\\Scripts\\python -m pytest -q tests"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import evaluacion as E
from src import redes as R
from src.datos import ESCENARIOS, Escenario, imputar_t7, serie
from src.ventanas import Ventana, escala_mase, evaluables, particion


@pytest.fixture(scope="module")
def series():
    return {(k, b): serie(e) for k, d in ESCENARIOS.items() for b, e in d.items()}


def _por_anio(df):
    return [len(evaluables(df, particion(df.index, Ventana(a), "max"))) for a in C.ANIOS_TEST]


# ----------------------------------------------------------------------------- conteos del protocolo (§2.2)

def test_evaluables_coinciden_con_protocolo(series):
    assert _por_anio(series["principal", "D"]) == [364, 365, 360, 365, 364]
    assert _por_anio(series["sens1_estricto", "D"]) == [298, 268, 284, 270, 232]
    assert _por_anio(series["sens2_bolsa", "D"]) == [364, 363, 360, 364, 364]
    assert _por_anio(series["principal", "M"]) == [11, 12, 12, 12, 12]
    # Sens. 1 de M: el protocolo (tabla §2.2) da 7 en 2021 porque cuenta 2021-02; aquí se excluye por §2.2
    # (mes de la anomalía, no se evalúa). Desviación registrada en docs/DESVIACIONES.md.
    assert _por_anio(series["sens1_estricto", "M"]) == [6, 2, 5, 4, 0]


def test_anomalia(series):
    d, d7 = series["principal", "D"], series["sens7_anomalia_publicada", "D"]
    f = pd.Timestamp("2021-02-01")
    assert np.isnan(d.y_eval[f]) and d.y[f] == d.y[f - pd.Timedelta(days=7)]
    assert d7.y_eval[f] == 2351.0 and d7.y[f] == 2351.0
    m7 = series["sens7_anomalia_publicada", "M"]
    assert m7.y_eval[pd.Timestamp("2021-02-01")] == 16990.0
    assert np.isnan(series["principal", "M"].y_eval[pd.Timestamp("2021-02-01")])


def test_historia_maxima_protocolo(series):
    d, m = series["principal", "D"], series["principal", "M"]
    assert particion(d.index, Ventana(2021), "max").n_hist == 8127
    assert particion(d.index, Ventana(2025), "max").n_hist == 9588
    assert particion(m.index, Ventana(2021), "max").n_hist == 267
    assert particion(m.index, Ventana(2025), "max").n_hist == 315


def test_sin_datos_2026_y_desde_1997_10(series):
    for df in series.values():
        assert df.index.max() <= pd.Timestamp(C.FIN_DATOS)
    assert series["principal", "D"].index.min() == pd.Timestamp("1997-10-01")
    assert series["sens6_desde2006", "D"].index.min() == pd.Timestamp("2006-01-01")


# ----------------------------------------------------------------------------- mismos días de prueba para todo N

@pytest.mark.parametrize("brazo", ["D", "M"])
def test_mismos_dias_prueba_en_todos_los_N(series, brazo):
    df = series["principal", brazo]
    for a in C.ANIOS_TEST:
        ref = None
        for N in C.TAMANOS[brazo]:
            p = particion(df.index, Ventana(a), N)
            assert p.fin_train == particion(df.index, Ventana(a), "max").fin_train
            ev = evaluables(df, p)
            assert (df.index[ev].year == a).all()
            if ref is None:
                ref = (ev, p.val)
            assert np.array_equal(ev, ref[0]) and np.array_equal(p.val, ref[1])
            if N != "max":
                assert p.n_hist == N


# ----------------------------------------------------------------------------- sin fuga

def test_imputacion_solo_usa_pasado():
    rng = np.random.default_rng(0)
    v = rng.uniform(1, 10, 200)
    v[rng.choice(np.arange(7, 200), 40, replace=False)] = np.nan
    base = imputar_t7(v)
    for corte in (50, 120, 180):
        w = v.copy()
        w[corte + 1:] = rng.uniform(100, 200, len(w) - corte - 1)
        assert np.array_equal(imputar_t7(w)[: corte + 1], base[: corte + 1])


def _perturbar_despues(df, pos):
    out = df.copy()
    k = out.columns.get_loc
    out.iloc[pos + 1:, k("y")] = out.y.to_numpy()[pos + 1:] * 7.0 + 1000.0
    out.iloc[pos + 1:, k("precio")] = out.precio.to_numpy()[pos + 1:] * 3.0 + 5.0
    return out


@pytest.mark.parametrize("brazo,modelo,h", [("D", "DualLSTM", 1), ("D", "DualLSTM", 7), ("D", "LSTM univariada", 1),
                                            ("M", "DualLSTM", 1), ("M", "LSTM univariada", 1)])
def test_redes_sin_fuga_en_ajuste_ni_escalado(series, brazo, modelo, h):
    """Cambiar cualquier dato posterior al fin del entrenamiento no altera el escalador ni las secuencias y
    objetivos de entrenamiento. Cambiar datos posteriores a t-1 no altera la entrada del pronóstico de t."""
    df = series["principal", brazo]
    for N in C.TAMANOS[brazo]:
        p = particion(df.index, Ventana(2023), N)
        X, ys, pos, ms, _, _ = R.preparar(df, brazo, modelo, p, h)
        X2, ys2, pos2, ms2, _, _ = R.preparar(_perturbar_despues(df, p.fin_train), brazo, modelo, p, h)
        assert ms == ms2
        assert np.array_equal(pos["train"], pos2["train"])
        assert np.array_equal(ys[pos["train"]], ys2[pos2["train"]])
        for a, b in zip(X["train"], X2["train"]):
            assert np.array_equal(a, b)
        assert pos["train"].max() <= p.fin_train
        assert pos["train"].min() - (h - 1) - 12 >= p.ini_hist
    # entrada de prueba: el pronóstico del objetivo τ solo usa datos hasta τ - h
    p = particion(df.index, Ventana(2023), "max")
    X, _, pos, _, _, _ = R.preparar(df, brazo, modelo, p, h)
    j = len(pos["test"]) // 2
    tau = pos["test"][j]
    X2, _, _, _, _, _ = R.preparar(_perturbar_despues(df, tau - h), brazo, modelo, p, h)
    for a, b in zip(X["test"], X2["test"]):
        assert np.array_equal(a[j], b[j])


def test_escala_mase_fija_por_ventana(series):
    df = series["principal", "D"]
    s = escala_mase(df, "D", Ventana(2023))
    df2 = _perturbar_despues(df, particion(df.index, Ventana(2023), "max").fin_train)
    assert s == escala_mase(df2, "D", Ventana(2023)) and s > 0


def test_parametros_red():
    assert R.expected_parameter_count(4, 33) == 6273 and R.expected_parameter_count(4, 39) == 6657  # artículo
    assert R.expected_parameter_count(4, 8) == 4673


# ----------------------------------------------------------------------------- evaluación

def test_dm_hln_h1_igual_que_articulo():
    rng = np.random.default_rng(1)
    a, b = rng.gamma(2, 1, 100), rng.gamma(2, 1.1, 100)
    d = a - b
    n = len(d)
    g0 = np.mean((d - d.mean()) ** 2)
    hln = d.mean() / np.sqrt(g0 / n) * np.sqrt((n + 1 - 2 + 0) / n)
    assert E.dm_hln(a, b, 1)["HLN"] == pytest.approx(hln)


def test_bootstrap_pareado_reproducible_y_estratificado():
    rng = np.random.default_rng(2)
    AE = rng.gamma(2, 1, (3, 60))
    est = np.repeat([2021, 2022, 2023], 20)
    r1, r2 = E.remuestreos(AE, est), E.remuestreos(AE, est)
    assert r1.shape == (3, C.B) and np.array_equal(r1, r2)
    idx = E.indices_iid(est, np.ones(60, bool), np.random.default_rng(42), b=50)
    assert ((idx[:, :20] < 20).all() and ((idx[:, 20:40] >= 20) & (idx[:, 20:40] < 40)).all())
    d = E.diferencia(AE, 0, 0, est)
    assert d["delta"] == 0 and not d["significativa"]


def test_regla_decision():
    def x(d, lo, hi):
        return {"delta": d, "ic_lo": lo, "ic_hi": hi}
    assert E.regla_decision([x(5, 1, 9), x(3, 1, 5), x(2, 0.5, 4), x(1, 0.1, 2)])["etiqueta"] == "el modelo"
    assert E.regla_decision([x(5, 1, 9), x(3, 1, 5), x(2, 0.5, 4), x(0.5, -1, 2)])["etiqueta"] == "los datos"
    assert E.regla_decision([x(5, 1, 9), x(3, 1, 5), x(2, 0.5, 4), x(-3, -5, -1)])["etiqueta"] == "los datos"
    assert E.regla_decision([x(5, 1, 9), x(6, 1, 9), x(2, 0.5, 4), x(0.5, -1, 2)])["etiqueta"] == "no concluyente"
    r = E.regla_decision([x(1, -1, 3), x(0.5, -1, 2), x(0.2, -1, 1), x(-3, -5, -1)])
    assert r["etiqueta"] == "no concluyente" and r["detalle"] == "no se reproduce la desventaja"


def test_mejor_referencia_y_conjunta():
    m = {"Seasonal Naive": 5, "ETS": 3, "Theta": 3, "SARIMA": 4, "Chronos-Bolt-tiny": 6, "DualLSTM": 1}
    assert E.mejor_referencia(m) == "ETS"
    assert E.conclusion_conjunta("los datos", "los datos") == "los datos"
    assert E.conclusion_conjunta("los datos", "el modelo") == "no concluyente"


def test_mascara_shock_usa_historia_hasta_origen(series):
    df = series["principal", "D"]
    p = particion(df.index, Ventana(2023), "max")
    ev = evaluables(df, p)
    origen = p.val[-1]
    m1, u1 = E.mascara_shock(df, "D", origen, ev, 75)
    df2 = df.copy()
    df2.iloc[origen + 1:, df2.columns.get_loc("y")] *= 100   # el umbral no depende de la prueba
    _, u2 = E.mascara_shock(df2, "D", origen, ev, 75)
    assert u1 == u2 and m1.dtype == bool and 0 < m1.sum() < len(ev)


def test_analisis_humo(series):
    """Pronósticos sintéticos (y_true + ruido) para todos los modelos del brazo M: el análisis corre completo,
    detecta faltantes y elige ref* entre las referencias."""
    from src.analisis import analizar, fechas_evaluables
    df = series["principal", "M"]
    rng = np.random.default_rng(0)
    filas = []
    for N in map(str, C.TAMANOS["M"]):
        for a in C.ANIOS_TEST:
            p = particion(df.index, Ventana(a), "max")
            for pos in p.test:
                yt = df.y_eval.iloc[pos]
                for k, m in enumerate(C.MODELOS):
                    for s in (range(2) if m in C.REDES else [np.nan]):
                        filas.append({"escenario": "principal", "brazo": "M", "modelo": m, "semilla": s, "N": N,
                                      "anio": a, "h": 1, "fecha": df.index[pos].strftime("%Y-%m-%d"),
                                      "y_pred": (df.y.iloc[pos]) * (1 + rng.normal(0, 0.05 * (k + 1)))})
    pron = pd.DataFrame(filas)
    t = analizar(pron, df, "M", 1)
    assert set(t["delta"].ref_estrella) <= set(C.REFERENCIAS)
    assert (t["metricas"].query("anio == 'pool'").n == 59).all()
    assert t["decision"].etiqueta.iloc[0] in ("los datos", "el modelo", "no concluyente")
    with pytest.raises(AssertionError):
        analizar(pron[~((pron.modelo == "ETS") & (pron.fecha == "2023-05-01"))], df, "M", 1)
