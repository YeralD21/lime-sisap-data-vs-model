"""Pruebas del experimento 3 (docs/ANEXO_EXP3.md): exógenas alineadas y sin fuga.
Necesitan data/processed/exogenas_{diarias,mensuales}.csv (scripts/07_exogenas_exp3.py).
Ejecutar: .venv311\\Scripts\\python -m pytest -q tests"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import exogenas as X3
from src import redes as R
from src.ventanas import Ventana, particion

pytestmark = pytest.mark.skipif(not (X3.EXOG_DIARIAS.exists() and X3.EXOG_MENSUALES.exists()),
                                reason="faltan las exógenas (scripts/07_exogenas_exp3.py)")


@pytest.fixture(scope="module")
def datos():
    return {b: (X3.serie_exp3(b), X3.cargar(b)) for b in "DM"}


def test_cobertura_2016_2025(datos):
    d, ed = datos["D"]
    m, em = datos["M"]
    assert ed.index.min() == pd.Timestamp("2016-01-01") and ed.index.max() == pd.Timestamp("2025-12-31")
    assert len(ed) == len(pd.date_range("2016-01-01", "2025-12-31")) and ed.index.is_monotonic_increasing
    assert len(em) == 120 and em.index.max() == pd.Timestamp("2025-12-01")
    assert d.index.min() == pd.Timestamp("2016-01-01") and d.index.max() == pd.Timestamp("2025-12-31")
    assert not ed[["prec", "tmax", "tmin", "rh", "eventos", "afectados", "n_noticias"]].isna().any().any()
    assert not em.isna().any().any()
    # 0 noticias = media faltante con indicador
    assert (ed.sent.isna() == (ed.n_noticias == 0)).all() and (ed.sin_noticias == (ed.n_noticias == 0)).all()


def test_pesos_solo_2016_2019():
    w = X3.pesos_provinciales()
    assert w.peso.sum() == pytest.approx(1.0) and (w.peso > 0).all()
    p = pd.read_csv(X3.PROD_PROVINCIAL, encoding="utf-8-sig")
    q = p[p["año"].between(2016, 2019)].groupby(["departamento", "provincia"]).produccion_t.sum()
    q = q[q > 0] / q[q > 0].sum()
    assert np.allclose(w.set_index(["departamento", "provincia"]).peso.reindex(q.index).to_numpy(), q.to_numpy())


def test_noticias_diarias_reproducen_mensual_del_articulo(datos):
    _, ed = datos["D"]
    art = pd.read_csv(X3.SENT_MENSUAL, encoding="utf-8-sig")
    n = ed.n_noticias.groupby([ed.index.year, ed.index.month]).sum().to_numpy()
    assert np.array_equal(n, art.n_noticias.to_numpy())


def test_indeci_diario_reproduce_panel_mensual_del_articulo(datos):
    _, ed = datos["D"]
    _, em = datos["M"]
    assert np.array_equal(ed.eventos.resample("MS").sum().to_numpy(), em.num_emergencias.to_numpy())
    assert np.allclose(ed.afectados.resample("MS").sum().to_numpy(), em.total_afectados.to_numpy())


def _perturbar(ex, desde):
    out = ex.copy()
    rng = np.random.default_rng(7)
    for c in out.columns:
        v = out[c].to_numpy(float).copy()
        v[desde:] = rng.uniform(0, 1000, len(v) - desde)
        out[c] = v
    if "sin_noticias" in out:                             # mantener coherencia del indicador
        out["n_noticias"] = np.round(out.n_noticias)
        out["sin_noticias"] = (out.n_noticias == 0).astype(int)
        out["sent"] = out.sent.where(out.n_noticias > 0)
    return out


@pytest.mark.parametrize("brazo", ["D", "M"])
@pytest.mark.parametrize("variante", ["V1", "V2", "V3", "V4"])
def test_exogenas_del_dia_t_no_entran_al_pronostico_de_t(datos, brazo, variante):
    """Cambiar las exógenas del día (mes) objetivo t y posteriores no altera la entrada de la red para t;
    cambiar las de t−1 sí la altera (las exógenas entran rezagadas un paso)."""
    df, ex = datos[brazo]
    fam = X3.VARIANTES[variante]
    p = particion(df.index, Ventana(2023), "max")
    marg = X3.MARGEN_FILAS[brazo]
    extra = X3.matriz_B(df.index, brazo, fam, ex)
    X, _, pos, _, _, _ = R.preparar(df, brazo, "DualLSTM", p, 1, extra, marg)
    for j in (0, len(pos["test"]) // 2, len(pos["test"]) - 1):
        tau = int(pos["test"][j])
        e2 = X3.matriz_B(df.index, brazo, fam, _perturbar(ex, tau))
        X2, _, _, _, _, _ = R.preparar(df, brazo, "DualLSTM", p, 1, e2, marg)
        assert np.array_equal(X["test"][1][j], X2["test"][1][j])
        e3 = X3.matriz_B(df.index, brazo, fam, _perturbar(ex, tau - 1))
        X3_, _, _, _, _, _ = R.preparar(df, brazo, "DualLSTM", p, 1, e3, marg)
        assert not np.array_equal(X["test"][1][j], X3_["test"][1][j])


@pytest.mark.parametrize("brazo", ["D", "M"])
def test_exogenas_posteriores_al_entrenamiento_no_cambian_ajuste(datos, brazo):
    df, ex = datos[brazo]
    marg = X3.MARGEN_FILAS[brazo]
    for anio in C.ANIOS_TEST:
        p = particion(df.index, Ventana(anio), "max")
        e1 = X3.matriz_B(df.index, brazo, X3.VARIANTES["V4"], ex)
        e2 = X3.matriz_B(df.index, brazo, X3.VARIANTES["V4"], _perturbar(ex, p.fin_train + 1))
        X, ys, pos, ms, _, B = R.preparar(df, brazo, "DualLSTM", p, 1, e1, marg)
        Y, ys2, pos2, ms2, _, B2 = R.preparar(df, brazo, "DualLSTM", p, 1, e2, marg)
        assert ms == ms2 and np.array_equal(pos["train"], pos2["train"])
        for a, b in zip(X["train"], Y["train"]):
            assert np.array_equal(a, b)
        sc1 = R.Escalador(B, p.ini_hist + max(6, marg), p.fin_train)
        sc2 = R.Escalador(B2, p.ini_hist + max(6, marg), p.fin_train)
        assert np.array_equal(sc1.media, sc2.media) and np.array_equal(sc1.escala, sc2.escala)


@pytest.mark.parametrize("brazo", ["D", "M"])
def test_variantes_mismos_objetivos_y_entradas_finitas(datos, brazo):
    df, ex = datos[brazo]
    marg = X3.MARGEN_FILAS[brazo]
    for anio in C.ANIOS_TEST:
        p = particion(df.index, Ventana(anio), "max")
        assert df.index[p.ini_hist] == pd.Timestamp("2016-01-01")
        ref = None
        for v, fam in X3.VARIANTES.items():
            X, ys, pos, _, A, B = R.preparar(df, brazo, "DualLSTM", p, 1, X3.matriz_B(df.index, brazo, fam, ex), marg)
            assert B.shape[1] == (8 if brazo == "D" else 6) + len(X3.columnas_B(brazo, fam))
            for k in ("val", "test"):
                assert all(np.isfinite(x).all() for x in X[k])
            if ref is None:
                ref = pos
            for k in ("train", "val", "test"):
                assert np.array_equal(pos[k], ref[k])


def test_dia3_sin_cambios_con_valores_por_defecto():
    from src.datos import ESCENARIOS, serie
    df = serie(ESCENARIOS["principal"]["D"])
    p = particion(df.index, Ventana(2023), 1000)
    a = R.preparar(df, "D", "DualLSTM", p, 1)
    b = R.preparar(df, "D", "DualLSTM", p, 1, None, 0)
    assert all(np.array_equal(x, y) for x, y in zip(a[0]["train"], b[0]["train"])) and a[3] == b[3]
