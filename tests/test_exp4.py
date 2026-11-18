"""Pruebas del experimento 4 (docs/ANEXO_EXP4.md): nombres y grupos de columnas, fondo, permutación por bloques,
agregación de SHAP, atención en NumPy y viabilidad de GradientExplainer con la capa de atención.
Las pruebas con TensorFlow usan una red SIN ENTRENAR (pesos aleatorios): no explican ningún modelo del estudio.
Ejecutar: .venv311\\Scripts\\python -m pytest -q tests"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src import explicabilidad as XP
from src import exogenas as X3
from src import redes as R

EXOG = X3.EXOG_DIARIAS.exists() and X3.EXOG_MENSUALES.exists()


@pytest.mark.parametrize("brazo", ["D", "M"])
@pytest.mark.parametrize("variante", [None, "V0", "V1", "V2", "V3", "V4"])
def test_nombres_y_grupos_cubren_la_rama_B(brazo, variante):
    nb = XP.nombres_B(brazo, variante)
    esperado = {"D": {None: 8, "V0": 8, "V1": 20, "V2": 14, "V3": 17, "V4": 35},
                "M": {None: 6, "V0": 6, "V1": 21, "V2": 24, "V3": 12, "V4": 45}}[brazo][variante]
    assert len(nb) == esperado == len(set(nb))                      # anexo del exp. 3 §2.3
    gr = XP.mapa_grupos(nb, lambda n: XP.grupo_repo(n, brazo))
    idx_b = np.concatenate([i for g, (r, i) in gr.items() if r == "B"])
    assert sorted(idx_b) == list(range(len(nb)))                   # partición exacta, sin solapes
    assert set(gr) >= {"historia", "calendario", "precio"}
    fams = X3.VARIANTES[variante] if variante else ()
    assert {g for g in gr if g not in ("historia", "calendario", "precio")} == set(fams)


@pytest.mark.skipif(not EXOG, reason="faltan las exógenas (scripts/07_exogenas_exp3.py)")
@pytest.mark.parametrize("brazo", ["D", "M"])
def test_nombres_coinciden_con_la_matriz_de_la_red(brazo):
    df = X3.serie_exp3(brazo)
    extra = X3.matriz_B(df.index, brazo, X3.VARIANTES["V4"])
    _, B = R.variables(df, brazo, "DualLSTM", extra)
    assert B.shape[1] == len(XP.nombres_B(brazo, "V4"))


def test_grupos_del_articulo():
    gc3 = (["mes_sin", "mes_cos", "t_index"]
           + [f"{v}_lag{k}" for v in ("T2M", "T2M_MAX", "WS2M", "PRECTOTCORR", "RH2M") for k in (1, 3, 6)]
           + [f"{v}_lag{k}" for v in ("num_emergencias", "personas_afectadas", "personas_damnificadas",
                                      "hectareas_cultivo_perdidas", "hectareas_cultivo_afectadas") for k in (1, 3, 6)])
    ge = gc3 + [f"{v}_lag{k}" for v in ("avg_sentiment", "n_noticias") for k in (1, 3, 6)]
    for nombres, n_exo in ((gc3, {"clima": 15, "indeci": 15}), (ge, {"clima": 15, "indeci": 15, "noticias": 6})):
        gr = XP.mapa_grupos(nombres, XP.grupo_articulo)
        assert {g: len(i) for g, (r, i) in gr.items() if g in n_exo} == n_exo
        assert len(gr["calendario"][1]) == 3 and "precio" not in gr


def test_fondo_solo_entrenamiento_y_determinista():
    a = XP.indices_fondo(2887)
    assert len(a) == 200 and len(set(a)) == 200 and a.min() >= 0 and a.max() < 2887
    assert np.array_equal(a, XP.indices_fondo(2887))
    assert np.array_equal(XP.indices_fondo(84), np.arange(84))     # menos de 200: todas


@pytest.mark.parametrize("n,b", [(365, 7), (12, 3), (10, 3)])
def test_permutacion_por_bloques(n, b):
    p = XP.permutacion_bloques(n, b, np.random.default_rng(42))
    assert sorted(p) == list(range(n))
    # dentro de cada bloque original se conserva el orden: cada inicio de bloque va seguido de su bloque
    for ini in range(0, n, b):
        k = int(np.flatnonzero(p == ini)[0])
        largo = min(b, n - ini)
        assert list(p[k:k + largo]) == list(range(ini, ini + largo))
    q = XP.permutacion_bloques(n, b, np.random.default_rng(42))
    assert np.array_equal(p, q)


def test_permutar_grupo_solo_toca_sus_columnas():
    rng = np.random.default_rng(0)
    XA, XB = rng.normal(size=(20, 6, 4)), rng.normal(size=(20, 6, 9))
    perm = XP.permutacion_bloques(20, 7, np.random.default_rng(1))
    a2, b2 = XP.permutar_grupo(XA, XB, "B", np.array([2, 5]), perm)
    assert np.array_equal(a2, XA)
    assert np.array_equal(np.delete(b2, [2, 5], axis=2), np.delete(XB, [2, 5], axis=2))
    assert np.array_equal(b2[:, :, [2, 5]], XB[perm][:, :, [2, 5]])
    a3, b3 = XP.permutar_grupo(XA, XB, "A", np.arange(4), perm)
    assert np.array_equal(a3, XA[perm]) and np.array_equal(b3, XB)


def test_agregacion_shap_es_aditiva():
    rng = np.random.default_rng(0)
    phiA, phiB = rng.normal(size=(15, 6, 4)), rng.normal(size=(15, 6, 8))
    gr = XP.mapa_grupos(XP.nombres_B("D", None), lambda n: XP.grupo_repo(n, "D"))
    c = XP.contribucion_grupos(phiA, phiB, gr)
    total = sum(v.sum(axis=1) for v in c.values())
    assert np.allclose(total, phiA.sum(axis=(1, 2)) + phiB.sum(axis=(1, 2)))
    imp = XP.importancia_grupos(phiA, phiB, gr)
    assert all(v["abs_total"] <= v["suma_abs"] + 1e-12 for v in imp.values())


# ----------------------------------------------------------------------------- con TensorFlow (red sin entrenar)
@pytest.fixture(scope="module")
def red_aleatoria():
    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(0)
    red, Att = R._construir("DualLSTM", 4, 35, n_b_libre=True)
    rng = np.random.default_rng(0)
    return red, rng.normal(size=(40, 6, 4)).astype("float32"), rng.normal(size=(40, 6, 35)).astype("float32")


def test_atencion_numpy_reproduce_la_capa(red_aleatoria):
    from tensorflow import keras
    red, XA, XB = red_aleatoria
    sub = keras.Model(red.inputs, [red.get_layer("lstm_b").output, red.get_layer("attention_b").output])
    h, c = (np.asarray(t, float) for t in sub.predict([XA, XB], verbose=0))
    L = red.get_layer("attention_b")
    al = XP.pesos_atencion(h, *(np.asarray(w.kernel.numpy(), float) for w in (L.W_query, L.W_values, L.V)))
    assert np.allclose(al.sum(axis=1), 1)
    assert np.abs((al[..., None] * h).sum(axis=1) - c).max() < 1e-5


def test_gradient_explainer_funciona_con_la_capa_de_atencion(red_aleatoria):
    shap = pytest.importorskip("shap")
    red, XA, XB = red_aleatoria
    ex = shap.GradientExplainer(red, [XA[:30], XB[:30]])
    sv = ex.shap_values([XA[30:], XB[30:]], nsamples=200, rseed=42)
    phiA, phiB = (np.asarray(v, float) for v in sv)
    phiA = phiA.reshape(XA[30:].shape) if phiA.size == XA[30:].size else phiA[..., 0]
    phiB = phiB.reshape(XB[30:].shape) if phiB.size == XB[30:].size else phiB[..., 0]
    assert phiA.shape == (10, 6, 4) and phiB.shape == (10, 6, 35)
    fx = red.predict([XA[30:], XB[30:]], verbose=0).reshape(-1)
    base = red.predict([XA[:30], XB[:30]], verbose=0).mean()
    err = fx - base - phiA.sum(axis=(1, 2)) - phiB.sum(axis=(1, 2))
    # expected gradients cumple la completitud solo en esperanza: error pequeño frente a la salida
    assert np.abs(err).mean() < 0.25 * np.abs(fx - base).mean() + 1e-3


def test_hijo_conserva_el_checkpoint_y_borra_lo_demas(tmp_path, monkeypatch):
    """El parche de 10_exp4.py solo cambia el borrado de checkpoint_best.keras (lo renombra a modelo.keras)."""
    import pathlib
    spec = importlib.util.spec_from_file_location("exp4", Path(__file__).resolve().parents[1] / "scripts" / "10_exp4.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    llamadas = {}

    def falso_entrenar(u, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "checkpoint_best.keras").write_text("x")
        (dest / "otro.tmp").write_text("y")
        (dest / "checkpoint_best.keras").unlink(missing_ok=True)
        (dest / "otro.tmp").unlink()
        llamadas["ok"] = True

    monkeypatch.setattr(R, "entrenar_y_predecir", falso_entrenar)
    monkeypatch.setattr(m, "guardar_entradas_repo", lambda u, d: None)
    m.hijo_repo('{"semilla": 0}', str(tmp_path / "u"))
    assert llamadas["ok"]
    assert (tmp_path / "u" / "modelo.keras").read_text() == "x"
    assert not (tmp_path / "u" / "otro.tmp").exists()
    assert pathlib.Path.unlink.__name__ == "unlink" and pathlib.Path.unlink.__module__ == "pathlib"
