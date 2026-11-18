"""Experimento 4 (docs/ANEXO_EXP4.md): explicabilidad DESCRIPTIVA / POSTERIOR de la DualLSTM ya entrenada.

Pasos:
  --verificar   el anexo está en HEAD sin cambios y los SHA-256 coinciden con docs/hashes_anexo_exp4.json
  --modelos     reentrena SOLO las unidades del anexo §2 que no tienen pesos guardados (ninguna los tiene),
                guarda el modelo y sus entradas y comprueba que los pronósticos de 2025 coinciden con los
                publicados (anexo §2.4). Si alguno no coincide, se detiene. (Fase A, permitido por el anexo.)
  --explicar    SHAP (GradientExplainer), permutación por bloques y atención por unidad (fase B; exige --verificar)
  --analizar    agrega por grupo, posición y semilla -> resultados/exp4/*.csv (fase B)
  --figuras     x1-x6 (--idioma es|en; matplotlib solo está en .venv)
Uso:
  .venv311\\Scripts\\python scripts/10_exp4.py --modelos --n-proc 6
  .venv311\\Scripts\\python scripts/10_exp4.py --verificar --explicar --n-proc 6 --analizar
  .venv\\Scripts\\python scripts/10_exp4.py --figuras --idioma es    (y en)
Todo es reanudable: una unidad con sus archivos de salida completos no se vuelve a calcular.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src import explicabilidad as XP  # noqa: E402

SAL = RAIZ / "resultados" / "exp4"
MOD = SAL / "modelos"                      # no versionado (modelos, entradas escaladas y SHAP por muestra)
ANEXO = RAIZ / "docs" / "ANEXO_EXP4.md"
HASHES = RAIZ / "docs" / "hashes_anexo_exp4.json"
ART = C.ARTICULO_RAIZ
ART_PY = ART / "venv" / "Scripts" / "python.exe"
ART_MOD = ART / "v2_reentrenamiento" / "experimentos_backtesting_rodante" / "04_modelos"
ANIO = 2025

# caso -> (fuente, brazo, variante o modelo del artículo)
CASOS = {
    "exp3_V4_D": ("exp3", "D", "V4"),
    "exp3_V4_M": ("exp3", "M", "V4"),
    "exp2_Nmax_D": ("exp2", "D", None),
    "exp2_Nmax_M": ("exp2", "M", None),
    "exp1_GC3": ("exp1", "M", "GC3"),
    "exp1_GE": ("exp1", "M", "GE"),
}
VARIANTE_GRUPO = {"clima": "V1", "indeci": "V2", "noticias": "V3"}   # ablación del experimento 3


def sha256_lf(ruta: Path) -> str:
    b = Path(ruta).read_bytes()
    if Path(ruta).suffix.lower() in {".md", ".csv", ".py", ".json", ".txt"}:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def verificar() -> None:
    h = json.loads(HASHES.read_text(encoding="utf-8"))
    for ruta, info in h["archivos"].items():
        if sha256_lf(RAIZ / ruta) != info["sha256"]:
            raise SystemExit(f"SHA-256 distinto: {ruta}")
    r = subprocess.run(["git", "ls-files", "--error-unmatch", "docs/ANEXO_EXP4.md"], cwd=RAIZ, capture_output=True)
    d = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "docs/ANEXO_EXP4.md", "docs/hashes_anexo_exp4.json"],
                       cwd=RAIZ)
    if r.returncode or d.returncode:
        raise SystemExit("El anexo del experimento 4 no está en HEAD o tiene cambios sin commit: no se calcula.")
    print("Anexo del experimento 4 en HEAD y hashes OK.")


# ----------------------------------------------------------------------------- unidades
def unidad_repo(caso: str, semilla: int) -> dict:
    fuente, brazo, variante = CASOS[caso]
    u = {"escenario": {"brazo": brazo, "objetivo": "principal", "inicio": C.INICIO, "anomalia": "faltante"},
         "anio": ANIO, "N": "max", "h": 1, "modelo": "DualLSTM", "semilla": semilla}
    if fuente == "exp3":
        u["variante"] = variante
    return u


def carpeta(caso: str, semilla: int) -> Path:
    return MOD / caso / f"s{semilla:02d}"


def pred_publicadas(caso: str, semilla: int) -> tuple[pd.DataFrame, pd.DataFrame | None, str | None]:
    """(predicciones de prueba por unidad, a precisión completa; filas de pronosticos.csv; su formato)."""
    fuente, b, v = CASOS[caso]
    if fuente == "exp3":
        u = pd.read_csv(RAIZ / f"resultados/exp3/unidades/exp3_{b}/{v}/Y{ANIO}/s{semilla:02d}/predicciones.csv")
        p = pd.read_csv(RAIZ / f"resultados/exp3/{b}/pronosticos.csv")
        p = p[(p.variante == v) & (p.semilla == semilla) & (p.anio == ANIO)]
        return u[u.particion == "test"], p, "%.6g"           # 08_exp3.py guarda con float_format="%.6g"
    if fuente == "exp2":
        u = pd.read_csv(RAIZ / f"resultados/brazo_{b}/redes_unidades/{b}_principal_1997_faltante/dual/Y{ANIO}/Nmax/h1/"
                        f"s{semilla:02d}/predicciones.csv")
        p = pd.read_csv(RAIZ / f"resultados/brazo_{b}/principal/pronosticos.csv", low_memory=False)
        p = p[(p.modelo == "DualLSTM") & (p.semilla == semilla) & (p.N.astype(str) == "max") & (p.anio == ANIO)
              & (p.h == 1)]
        return u[u.particion == "test"], p, ("%.8g" if b == "D" else "%.10g")
    carp = {"GC3": "gc3_lstm_sin_nlp", "GE": "ge_lstm_con_nlp"}[v]
    u = pd.read_csv(ART_MOD / carp / "sutil" / str(ANIO) / f"seed_{semilla:02d}" / "predicciones.csv")
    u = u[u.particion == "test"].assign(fecha=lambda x: x.fecha + "-01")
    return u, None, None


def hijo_repo(unidad_json: str, destino: str) -> None:
    """Proceso nuevo (con src.redes.VARIABLES_ENTORNO). Entrena con src.redes.entrenar_y_predecir SIN cambiarla:
    solo se intercepta el borrado de checkpoint_best.keras para conservarlo como modelo.keras."""
    import pathlib

    original = pathlib.Path.unlink

    def unlink(self, missing_ok=False):
        if self.name == "checkpoint_best.keras":
            self.replace(self.with_name("modelo.keras"))
            return None
        return original(self, missing_ok=missing_ok)

    pathlib.Path.unlink = unlink
    from src import redes as R
    u = json.loads(unidad_json)
    dest = Path(destino)
    R.entrenar_y_predecir(u, dest)
    pathlib.Path.unlink = original
    if not (dest / "modelo.keras").exists():
        raise SystemExit("No quedó modelo.keras")
    guardar_entradas_repo(u, dest)


def guardar_entradas_repo(u: dict, dest: Path) -> None:
    """Las mismas secuencias que usó entrenar_y_predecir (misma selección de serie, exógenas y margen)."""
    from src import redes as R
    from src.datos import Escenario, serie
    from src.ventanas import Ventana, particion
    esc = Escenario(**u["escenario"])
    variante = u.get("variante")
    if variante is None:
        df, extra, margen = serie(esc), None, 0
    else:
        from src import exogenas as X3
        df = X3.serie_exp3(esc.brazo)
        extra = X3.matriz_B(df.index, esc.brazo, X3.VARIANTES[variante])
        margen = X3.MARGEN_FILAS[esc.brazo]
    part = particion(df.index, Ventana(int(u["anio"])), u["N"])
    X, ys, pos, (mu, sd), A, B = R.preparar(df, esc.brazo, u["modelo"], part, int(u["h"]), extra, margen)
    nb = XP.nombres_B(esc.brazo, variante)
    if len(nb) != B.shape[1]:
        raise AssertionError(f"Nombres de la rama B ({len(nb)}) != columnas ({B.shape[1]})")
    np.savez(dest / "entradas.npz",
             XA_train=X["train"][0], XB_train=X["train"][1], XA_val=X["val"][0], XB_val=X["val"][1],
             XA_test=X["test"][0], XB_test=X["test"][1], ys_test=ys[pos["test"]],
             y_eval_test=df.y_eval.to_numpy(float)[pos["test"]],
             fechas_test=np.array([df.index[p].strftime("%Y-%m-%d") for p in pos["test"]]),
             mu=mu, sd=sd, nombres_A=np.array(XP.NOMBRES_A), nombres_B=np.array(nb))


def _correr(cmd: list[str], env: dict | None = None, cwd: Path = RAIZ) -> None:
    r = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"código {r.returncode}\n{r.stderr[-2500:]}")


def reentrenar_unidad(caso: str, semilla: int) -> str:
    dest = carpeta(caso, semilla)
    if (dest / "modelo.keras").exists() and (dest / "entradas.npz").exists() and (dest / "estado.json").exists():
        return "reanudada"
    import shutil
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fuente = CASOS[caso][0]
    if fuente == "exp1":
        if not ART_PY.exists():
            raise RuntimeError(f"No existe el intérprete del artículo: {ART_PY}")
        _correr([str(ART_PY), str(RAIZ / "scripts" / "10_exp4_articulo.py"), CASOS[caso][2], str(semilla), str(dest)],
                env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)})
        ck = dest / "checkpoint_best.keras"
        if not ck.exists():
            raise RuntimeError("El código del artículo no dejó checkpoint_best.keras")
        ck.replace(dest / "modelo.keras")
    else:
        from src.redes import VARIABLES_ENTORNO, python_redes
        _correr([python_redes(), str(Path(__file__).resolve()), "--hijo-repo", json.dumps(unidad_repo(caso, semilla)),
                 str(dest)], env={**os.environ, **VARIABLES_ENTORNO})
    return "entrenada"


def comprobar_reproduccion(caso: str, semilla: int) -> dict:
    """Anexo §2.4: pronósticos de prueba de 2025 del modelo reentrenado frente a los publicados."""
    dest = carpeta(caso, semilla)
    nuevo = pd.read_csv(dest / "predicciones.csv")
    nuevo = nuevo[nuevo.particion == "test"]
    if CASOS[caso][0] == "exp1":
        nuevo = nuevo.assign(fecha=lambda x: x.fecha + "-01")
    unidad, pron, fmt = pred_publicadas(caso, semilla)
    m = nuevo.merge(unidad[["fecha", "y_pred"]], on="fecha", suffixes=("", "_pub"), validate="1:1")
    if len(m) != len(nuevo) or len(m) != len(unidad):
        raise RuntimeError(f"{caso} s{semilla}: fechas distintas ({len(nuevo)} vs {len(unidad)})")
    d_unidad = float((m.y_pred - m.y_pred_pub).abs().max())
    fila = {"caso": caso, "semilla": semilla, "n_test": len(m), "dif_max_unidad_t": d_unidad,
            "ok_unidad": d_unidad <= 1e-3}
    if pron is not None:
        q = nuevo.merge(pron[["fecha", "y_pred"]], on="fecha", suffixes=("", "_pub"), validate="1:1")
        dif = (q.y_pred - q.y_pred_pub).abs()
        redondeo = np.array([float(fmt % a) == b for a, b in zip(q.y_pred, q.y_pred_pub)])
        fila |= {"formato_pronosticos": fmt, "dif_max_pronosticos_t": float(dif.max()),
                 "ok_pronosticos": bool(((dif <= 1e-3) | redondeo).all() and len(q) == len(nuevo))}
    return fila


def modelos(n_proc: int) -> None:
    SAL.mkdir(parents=True, exist_ok=True)
    log = open(SAL / "modelos.log", "a", encoding="utf-8")

    def escribir(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()
    # 1. ¿hay pesos guardados? (anexo §2.2)
    zipf = Path(r"D:\respaldo_dia3_unidades.zip")
    pesos = [p for p in (RAIZ / "resultados").rglob("*.keras") if MOD not in p.parents]
    en_zip = 0
    if zipf.exists():
        import zipfile
        with zipfile.ZipFile(zipf) as z:
            en_zip = sum(n.endswith((".keras", ".h5", ".weights.h5")) for n in z.namelist())
    en_art = len(list(ART_MOD.rglob("*.keras"))) if ART_MOD.exists() else 0
    escribir(f"Pesos guardados: resultados/ fuera de exp4 = {len(pesos)}; zip = {en_zip}; artículo 04_modelos = {en_art}")
    # 2. reentrenar (reanudable) y comprobar
    tareas = [(c, s) for c in CASOS for s in C.SEMILLAS]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_proc) as ex:
        futs = {ex.submit(reentrenar_unidad, c, s): (c, s) for c, s in tareas}
        for i, f in enumerate(futs, 1):
            c, s = futs[f]
            try:
                escribir(f"[{i}/{len(tareas)}] {c} s{s:02d}: {f.result()}")
            except Exception as exc:
                escribir(f"[{i}/{len(tareas)}] FALLO {c} s{s:02d}: {exc}")
    escribir(f"Pared: {time.time() - t0:.0f} s")
    filas = []
    for c, s in tareas:
        try:
            filas.append(comprobar_reproduccion(c, s))
        except Exception as exc:
            filas.append({"caso": c, "semilla": s, "error": str(exc), "ok_unidad": False})
    rep = pd.DataFrame(filas)
    rep.to_csv(SAL / "reproduccion.csv", index=False, float_format="%.6g")
    malos = rep[~rep.ok_unidad.astype(bool) | (rep.get("ok_pronosticos", True) == False)]  # noqa: E712
    escribir(rep.groupby("caso").agg(dif_max_unidad_t=("dif_max_unidad_t", "max"), ok=("ok_unidad", "all"))
             .to_string())
    if len(malos):
        escribir(f"NO COINCIDEN {len(malos)} unidades: ver resultados/exp4/reproduccion.csv. Se detiene.")
        raise SystemExit(1)
    escribir("Reproducción OK en todas las unidades.")


# ----------------------------------------------------------------------------- explicación (fase B)
def _cargar(dest: Path):
    from tensorflow import keras

    from src import redes as R
    z = dict(np.load(dest / "entradas.npz", allow_pickle=False))
    _, Att = R._construir("DualLSTM", 4, z["XB_test"].shape[2], n_b_libre=True)
    red = keras.models.load_model(dest / "modelo.keras", custom_objects={"BahdanauAttention": Att}, compile=False)
    return red, z


def hijo_explicar(caso: str, semilla: str) -> None:
    """Proceso nuevo por unidad: SHAP, permutación y atención. Escribe explicacion.npz y permutacion.csv."""
    import random

    import tensorflow as tf
    s = int(semilla)
    random.seed(XP.SEMILLA_EXP4)
    np.random.seed(XP.SEMILLA_EXP4)
    tf.random.set_seed(XP.SEMILLA_EXP4)
    tf.config.experimental.enable_op_determinism()
    dest = carpeta(caso, s)
    red, z = _cargar(dest)
    brazo = CASOS[caso][1]
    XA, XB = z["XA_test"], z["XB_test"]
    sd, mu = float(z["sd"]), float(z["mu"])

    def f(a, b):
        return red.predict([a, b], verbose=0, batch_size=512).reshape(-1)
    fx = f(XA, XB)
    # 0. el modelo cargado reproduce sus propias predicciones guardadas
    pr = pd.read_csv(dest / "predicciones.csv")
    pr = pr[pr.particion == "test"].y_pred.to_numpy()
    dif_carga = float(np.abs(fx * sd + mu - pr).max())
    # 1. SHAP: fondo solo de entrenamiento
    ib = XP.indices_fondo(len(z["XA_train"]))
    bgA, bgB = z["XA_train"][ib], z["XB_train"][ib]
    metodo = "GradientExplainer"
    try:
        import shap
        ex = shap.GradientExplainer(red, [bgA, bgB])
        sv = ex.shap_values([XA, XB], nsamples=200, rseed=XP.SEMILLA_EXP4)
        phiA, phiB = (np.asarray(v, float) for v in sv)
        phiA = phiA.reshape(XA.shape) if phiA.size == XA.size else phiA[..., 0]
        phiB = phiB.reshape(XB.shape) if phiB.size == XB.size else phiB[..., 0]
    except Exception as exc:  # anexo §3.1: alternativa declarada
        print(f"GradientExplainer falló ({exc}); se usa KernelExplainer", file=sys.stderr)
        metodo = "KernelExplainer"
        import shap
        na, nb = XA[0].size, XB[0].size

        def g(flat):
            return f(flat[:, :na].reshape((-1,) + XA.shape[1:]), flat[:, na:].reshape((-1,) + XB.shape[1:]))
        bg = np.hstack([bgA.reshape(len(bgA), -1), bgB.reshape(len(bgB), -1)])
        kx = shap.KernelExplainer(g, shap.kmeans(bg, 20))
        ph = np.asarray(kx.shap_values(np.hstack([XA.reshape(len(XA), -1), XB.reshape(len(XB), -1)]),
                                       nsamples=1000, silent=True))
        phiA, phiB = ph[:, :na].reshape(XA.shape), ph[:, na:].reshape(XB.shape)
    base = float(f(bgA, bgB).mean())
    err = fx - base - phiA.sum(axis=(1, 2)) - phiB.sum(axis=(1, 2))
    # 2. atención (NumPy) y comprobación contra la capa
    from tensorflow import keras
    sub = keras.Model(red.inputs, [red.get_layer("lstm_a").output, red.get_layer("lstm_b").output,
                                   red.get_layer("attention_a").output, red.get_layer("attention_b").output])
    ha, hb, ca, cb = (np.asarray(t, float) for t in sub.predict([XA, XB], verbose=0, batch_size=512))
    alphas, dif_att = {}, 0.0
    for rama, h, c in (("A", ha, ca), ("B", hb, cb)):
        L = red.get_layer(f"attention_{rama.lower()}")
        al = XP.pesos_atencion(h, *(np.asarray(w.kernel.numpy(), float) for w in (L.W_query, L.W_values, L.V)))
        dif_att = max(dif_att, float(np.abs((al[..., None] * h).sum(axis=1) - c).max()))
        alphas[rama] = al
    np.savez(dest / "explicacion.npz", phiA=phiA, phiB=phiB, fx=fx, base=base, err=err, alpha_A=alphas["A"],
             alpha_B=alphas["B"], fondo=ib)
    # 3. permutación por bloques
    nombres_b = [str(n) for n in z["nombres_B"]]
    gr = XP.mapa_grupos(nombres_b, XP.grupo_articulo if caso.startswith("exp1") else
                        (lambda n: XP.grupo_repo(n, brazo)))
    y = z["y_eval_test"]
    ok = np.isfinite(y)
    mae0 = float(np.abs(fx[ok] * sd + mu - y[ok]).mean())
    filas = []
    for g, (rama, idx) in gr.items():
        rng = np.random.default_rng(XP.SEMILLA_EXP4)        # mismas permutaciones para todos los grupos y semillas
        for r in range(XP.N_REPETICIONES):
            perm = XP.permutacion_bloques(len(XA), XP.BLOQUE_PERM[brazo], rng)
            a2, b2 = XP.permutar_grupo(XA, XB, rama, idx, perm)
            mae = float(np.abs(f(a2, b2)[ok] * sd + mu - y[ok]).mean())
            filas.append({"caso": caso, "semilla": s, "grupo": g, "rep": r, "identidad": bool((perm == np.arange(len(perm))).all()),
                          "mae_orig": mae0, "mae_perm": mae, "delta": mae - mae0})
    pd.DataFrame(filas).to_csv(dest / "permutacion.csv", index=False)
    (dest / "explicacion.json").write_text(json.dumps({
        "caso": caso, "semilla": s, "metodo_shap": metodo, "n_fondo": int(len(ib)), "n_explicadas": int(len(XA)),
        "n_evaluables": int(ok.sum()), "sd": sd, "dif_carga_t": dif_carga, "dif_atencion": dif_att,
        "base_t": base * sd + mu, "err_medio_abs_t": float(np.abs(err).mean() * sd),
        "err_max_abs_t": float(np.abs(err).max() * sd),
        "salida_media_abs_t": float(np.abs(fx - base).mean() * sd)}, indent=2), encoding="utf-8")


def explicar_unidad(caso: str, semilla: int) -> str:
    dest = carpeta(caso, semilla)
    if all((dest / n).exists() for n in ("explicacion.npz", "permutacion.csv", "explicacion.json")):
        return "reanudada"
    from src.redes import VARIABLES_ENTORNO, python_redes
    _correr([python_redes(), str(Path(__file__).resolve()), "--hijo-explicar", caso, str(semilla)],
            env={**os.environ, **VARIABLES_ENTORNO})
    return "calculada"


def explicar(n_proc: int) -> None:
    log = open(SAL / "explicacion.log", "a", encoding="utf-8")
    tareas = [(c, s) for c in CASOS for s in C.SEMILLAS]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_proc) as ex:
        futs = {ex.submit(explicar_unidad, c, s): (c, s) for c, s in tareas}
        for i, f in enumerate(futs, 1):
            c, s = futs[f]
            try:
                m = f"[{i}/{len(tareas)}] {c} s{s:02d}: {f.result()}"
            except Exception as exc:
                m = f"[{i}/{len(tareas)}] FALLO {c} s{s:02d}: {exc}"
            print(m, flush=True)
            log.write(m + "\n")
            log.flush()
    m = f"Pared: {time.time() - t0:.0f} s"
    print(m)
    log.write(m + "\n")


# ----------------------------------------------------------------------------- agregación (fase B)
def _grupos_caso(caso: str, z) -> dict:
    nb = [str(n) for n in z["nombres_B"]]
    if caso.startswith("exp1"):
        return XP.mapa_grupos(nb, XP.grupo_articulo)
    return XP.mapa_grupos(nb, lambda n: XP.grupo_repo(n, CASOS[caso][1]))


def _rango(serie: pd.Series) -> pd.Series:
    return serie.rank(ascending=False, method="min").astype(int)


def analizar() -> None:
    gsem, pos, var, att, cord = [], [], [], [], []
    perm = []
    for caso in CASOS:
        for s in C.SEMILLAS:
            d = carpeta(caso, s)
            z = np.load(d / "entradas.npz", allow_pickle=False)
            e = np.load(d / "explicacion.npz", allow_pickle=False)
            info = json.loads((d / "explicacion.json").read_text(encoding="utf-8"))
            sd = float(z["sd"])
            gr = _grupos_caso(caso, z)
            imp = XP.importancia_grupos(e["phiA"], e["phiB"], gr)
            for g, v in imp.items():
                gsem.append({"caso": caso, "semilla": s, "grupo": g, "n_columnas": len(gr[g][1]),
                             "shap_abs_t": v["abs_total"] * sd, "shap_suma_abs_t": v["suma_abs"] * sd})
                for k, a in enumerate(v["abs_pos"]):
                    pos.append({"caso": caso, "semilla": s, "grupo": g, "posicion": f"t-{len(v['abs_pos']) - k}",
                                "shap_abs_t": a * sd})
            nombres = [str(n) for n in z["nombres_A"]] + [str(n) for n in z["nombres_B"]]
            phi = np.concatenate([e["phiA"], e["phiB"]], axis=2).sum(axis=1)       # (n, variables)
            gvar = ["historia"] * len(z["nombres_A"]) + [
                (XP.grupo_articulo(str(n)) if caso.startswith("exp1") else XP.grupo_repo(str(n), CASOS[caso][1]))
                for n in z["nombres_B"]]
            for j, n in enumerate(nombres):
                var.append({"caso": caso, "semilla": s, "variable": n, "grupo": gvar[j],
                            "shap_abs_t": float(np.abs(phi[:, j]).mean() * sd)})
            for rama in "AB":
                for k, a in enumerate(e[f"alpha_{rama}"].mean(axis=0)):
                    att.append({"caso": caso, "semilla": s, "rama": rama, "posicion": f"t-{6 - k}", "alpha": a})
            cord.append({k: info[k] for k in ("caso", "semilla", "metodo_shap", "n_fondo", "n_explicadas",
                                              "n_evaluables", "dif_carga_t", "dif_atencion", "err_medio_abs_t",
                                              "err_max_abs_t", "salida_media_abs_t")})
            perm.append(pd.read_csv(d / "permutacion.csv"))
            if caso == "exp3_V4_D":
                if s == 0:
                    bee_phi, bee_x = np.zeros_like(phi), np.concatenate([z["XA_test"], z["XB_test"]], axis=2)[:, -1, :]
                    bee_nombres, bee_grupos = nombres, gvar
                bee_phi += phi * sd / len(C.SEMILLAS)
    np.savez(MOD / "beeswarm_exp3_V4_D.npz", phi=bee_phi, x=bee_x, nombres=np.array(bee_nombres),
             grupos=np.array(bee_grupos))                                     # no versionado (entradas escaladas)
    G = pd.DataFrame(gsem)
    G.to_csv(SAL / "shap_grupo_semillas.csv", index=False, float_format="%.6g")
    g = G.groupby(["caso", "grupo"], sort=False).agg(n_columnas=("n_columnas", "first"),
                                                       shap_abs_t=("shap_abs_t", "mean"),
                                                       shap_abs_sd_semillas=("shap_abs_t", "std"),
                                                       shap_suma_abs_t=("shap_suma_abs_t", "mean"),
                                                       shap_suma_abs_sd_semillas=("shap_suma_abs_t", "std")).reset_index()
    g["rango_shap"] = g.groupby("caso").shap_abs_t.transform(_rango)
    g.to_csv(SAL / "shap_grupo.csv", index=False, float_format="%.6g")
    P = pd.DataFrame(pos).groupby(["caso", "grupo", "posicion"], sort=False).shap_abs_t.agg(["mean", "std"])
    P.rename(columns={"mean": "shap_abs_t", "std": "sd_semillas"}).reset_index().to_csv(
        SAL / "shap_rezago.csv", index=False, float_format="%.6g")
    V = pd.DataFrame(var).groupby(["caso", "variable", "grupo"], sort=False).shap_abs_t.agg(["mean", "std"])
    V.rename(columns={"mean": "shap_abs_t", "std": "sd_semillas"}).reset_index().to_csv(
        SAL / "shap_variable.csv", index=False, float_format="%.6g")
    A = pd.DataFrame(att).groupby(["caso", "rama", "posicion"], sort=False).alpha.agg(["mean", "std"])
    A.rename(columns={"mean": "alpha", "std": "sd_semillas"}).reset_index().to_csv(
        SAL / "atencion.csv", index=False, float_format="%.6g")
    pd.DataFrame(cord).to_csv(SAL / "cordura.csv", index=False, float_format="%.6g")
    R = pd.concat(perm, ignore_index=True)
    R.to_csv(SAL / "permutacion_repeticiones.csv", index=False, float_format="%.6g")
    p = R.groupby(["caso", "grupo"], sort=False).agg(
        mae_orig_t=("mae_orig", "mean"), delta_mae_t=("delta", "mean"),
        ic_lo=("delta", lambda x: np.percentile(x, 2.5)), ic_hi=("delta", lambda x: np.percentile(x, 97.5)),
        n=("delta", "size"), repeticiones_identidad=("identidad", "sum")).reset_index()
    sdsem = R.groupby(["caso", "grupo", "semilla"]).delta.mean().groupby(["caso", "grupo"]).std()
    p["sd_semillas"] = [sdsem.loc[(c, gg)] for c, gg in zip(p.caso, p.grupo)]
    p["rango_perm"] = p.groupby("caso").delta_mae_t.transform(_rango)
    p.to_csv(SAL / "permutacion.csv", index=False, float_format="%.6g")
    # comparación con la ablación del experimento 3 (solo casos exp3_V4)
    filas = []
    for caso in CASOS:
        b = CASOS[caso][1]
        gs, ps = g[g.caso == caso].set_index("grupo"), p[p.caso == caso].set_index("grupo")
        exo = [x for x in ("clima", "indeci", "noticias") if x in gs.index]
        r_exo_shap = _rango(gs.loc[exo, "shap_abs_t"]) if exo else pd.Series(dtype=int)
        r_exo_perm = _rango(ps.loc[exo, "delta_mae_t"]) if exo else pd.Series(dtype=int)
        abl = None
        if caso.startswith("exp3"):
            abl = pd.read_csv(RAIZ / "resultados" / "exp3" / b / "delta.csv").set_index("variante")
            r_abl = (abl.loc[[VARIANTE_GRUPO[x] for x in exo], "delta"].rank(method="min").astype(int))
        for gg in gs.index:
            fila = {"caso": caso, "grupo": gg, "shap_abs_t": gs.loc[gg, "shap_abs_t"],
                    "rango_shap": gs.loc[gg, "rango_shap"], "delta_perm_t": ps.loc[gg, "delta_mae_t"],
                    "perm_ic_lo": ps.loc[gg, "ic_lo"], "perm_ic_hi": ps.loc[gg, "ic_hi"],
                    "rango_perm": ps.loc[gg, "rango_perm"],
                    "rango_shap_exogenas": r_exo_shap.get(gg, np.nan), "rango_perm_exogenas": r_exo_perm.get(gg, np.nan)}
            if abl is not None and gg in VARIANTE_GRUPO:
                v = VARIANTE_GRUPO[gg]
                fila |= {"ablacion_variante": v, "ablacion_delta_mae_t": abl.loc[v, "delta"],
                         "ablacion_ic_lo": abl.loc[v, "ic_lo"], "ablacion_ic_hi": abl.loc[v, "ic_hi"],
                         "ablacion_p_holm": abl.loc[v, "p_holm"], "ablacion_etiqueta": abl.loc[v, "etiqueta"],
                         "rango_ablacion_exogenas": int(r_abl.loc[v])}
            filas.append(fila)
        if abl is not None:
            filas.append({"caso": caso, "grupo": "todas_exogenas (V4)", "ablacion_variante": "V4",
                          "ablacion_delta_mae_t": abl.loc["V4", "delta"], "ablacion_ic_lo": abl.loc["V4", "ic_lo"],
                          "ablacion_ic_hi": abl.loc["V4", "ic_hi"], "ablacion_p_holm": abl.loc["V4", "p_holm"],
                          "ablacion_etiqueta": abl.loc["V4", "etiqueta"]})
    pd.DataFrame(filas).to_csv(SAL / "comparacion_ablacion.csv", index=False, float_format="%.6g")
    print("Agregados en", SAL)


# ----------------------------------------------------------------------------- figuras (fase B)
COLOR_G = {"historia": "#000000", "calendario": "#999999", "precio": "#E69F00", "clima": "#0072B2",
           "indeci": "#009E73", "noticias": "#CC79A7"}
TXT = {"es": {"caso": {"exp3_V4_D": "Exp. 3 V4, brazo D", "exp3_V4_M": "Exp. 3 V4, brazo M",
                       "exp2_Nmax_D": "Exp. 2 N = máx, brazo D", "exp2_Nmax_M": "Exp. 2 N = máx, brazo M",
                       "exp1_GC3": "Exp. 1 GC3 (mensual)", "exp1_GE": "Exp. 1 GE (mensual)"},
              "g": {"historia": "historia (A)", "calendario": "calendario", "precio": "precio", "clima": "clima",
                    "indeci": "INDECI", "noticias": "noticias"},
              "shap": "|SHAP| medio del grupo (t)", "pos": "Posición en la secuencia", "perm": "Aumento de MAE (t)",
              "rk": ["SHAP", "Permutación", "Ablación exp. 3"], "noabl": "no ablacionado",
              "bee": "Valor SHAP (t), suma de las 6 posiciones", "valor": "valor de la variable en t−1 (escalado)",
              "alpha": "Peso de atención medio", "ramaA": "rama A", "ramaB": "rama B",
              "et": {"ayuda": "ayuda", "perjudica": "perjudica", "sin efecto": "sin efecto"}},
       "en": {"caso": {"exp3_V4_D": "Exp. 3 V4, arm D", "exp3_V4_M": "Exp. 3 V4, arm M",
                       "exp2_Nmax_D": "Exp. 2 N = max, arm D", "exp2_Nmax_M": "Exp. 2 N = max, arm M",
                       "exp1_GC3": "Exp. 1 GC3 (monthly)", "exp1_GE": "Exp. 1 GE (monthly)"},
              "g": {"historia": "history (A)", "calendario": "calendar", "precio": "price", "clima": "climate",
                    "indeci": "INDECI", "noticias": "news"},
              "shap": "Mean group |SHAP| (t)", "pos": "Position in the sequence", "perm": "MAE increase (t)",
              "rk": ["SHAP", "Permutation", "Exp. 3 ablation"], "noabl": "not ablated",
              "bee": "SHAP value (t), sum over the 6 positions", "valor": "feature value at t−1 (scaled)",
              "alpha": "Mean attention weight", "ramaA": "branch A", "ramaB": "branch B",
              "et": {"ayuda": "helps", "perjudica": "hurts", "sin efecto": "no effect"}}}


def figuras(idioma: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = TXT[idioma]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.bbox": "tight"})
    out = RAIZ / "reportes" / "figuras" / idioma / "exp4"
    out.mkdir(parents=True, exist_ok=True)
    casos = list(CASOS)

    def guardar(fig, n):
        fig.savefig(out / f"{n}.pdf")
        fig.savefig(out / f"{n}.png", dpi=300)
        plt.close(fig)

    G = pd.read_csv(SAL / "shap_grupo.csv")
    P = pd.read_csv(SAL / "permutacion.csv")
    # x1
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.6))
    for ax, c in zip(axs.ravel(), casos):
        s = G[G.caso == c]
        x = np.arange(len(s))
        ax.bar(x, s.shap_abs_t, yerr=s.shap_abs_sd_semillas, color=[COLOR_G[g] for g in s.grupo], capsize=2)
        ax.set_xticks(x, [L["g"][g] for g in s.grupo], rotation=30, ha="right")
        ax.set_title(L["caso"][c])
        ax.set_ylabel(L["shap"])
    fig.tight_layout()
    guardar(fig, "x1_shap_por_grupo")
    # x2
    R = pd.read_csv(SAL / "shap_rezago.csv")
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.6))
    for ax, c in zip(axs.ravel(), casos):
        for gg, s in R[R.caso == c].groupby("grupo", sort=False):
            ax.errorbar(np.arange(len(s)), s.shap_abs_t, yerr=s.sd_semillas, color=COLOR_G[gg], marker="o", ms=3,
                        lw=1, capsize=2, label=L["g"][gg])
            ax.set_xticks(np.arange(len(s)), [p.replace("-", "−") for p in s.posicion])
        ax.set_title(L["caso"][c])
        ax.set_ylabel(L["shap"])
        ax.set_xlabel(L["pos"])
    axs.ravel()[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    guardar(fig, "x2_shap_por_rezago")
    # x3
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.6))
    for ax, c in zip(axs.ravel(), casos):
        s = P[P.caso == c]
        x = np.arange(len(s))
        ax.axhline(0, color="k", lw=0.6)
        ax.errorbar(x, s.delta_mae_t, yerr=[s.delta_mae_t - s.ic_lo, s.ic_hi - s.delta_mae_t], fmt="none",
                    ecolor="k", capsize=2, lw=0.8)
        ax.scatter(x, s.delta_mae_t, c=[COLOR_G[g] for g in s.grupo], s=22, zorder=3)
        ax.set_xticks(x, [L["g"][g] for g in s.grupo], rotation=30, ha="right")
        ax.set_title(L["caso"][c])
        ax.set_ylabel(L["perm"])
    fig.tight_layout()
    guardar(fig, "x3_permutacion")
    # x4: rankings lado a lado (exp. 3 V4)
    K = pd.read_csv(SAL / "comparacion_ablacion.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, c in zip(axs, ("exp3_V4_D", "exp3_V4_M")):
        k = K[(K.caso == c) & K.rango_shap.notna()]
        n = len(k)
        for _, r in k.iterrows():
            ys = [r.rango_shap, r.rango_perm]
            ax.plot([0, 1], ys, color=COLOR_G[r.grupo], lw=1.2, marker="o", ms=4)
            ax.text(-0.06, r.rango_shap, L["g"][r.grupo], ha="right", va="center", fontsize=7, color=COLOR_G[r.grupo])
            if isinstance(r.get("ablacion_variante"), str):
                ya = r.rango_ablacion_exogenas      # 1-3: orden por ΔMAE (el más favorable arriba)
                ax.plot([1, 2], [r.rango_perm, ya], color=COLOR_G[r.grupo], lw=1.2, ls=":", marker="o", ms=4)
                ax.text(2.06, ya, f"{L['g'][r.grupo]}: Δ = {r.ablacion_delta_mae_t:+.1f} t ({L['et'][r.ablacion_etiqueta]})",
                        ha="left", va="center", fontsize=6.5, color=COLOR_G[r.grupo])
        ax.text(2.06, n - 0.6, L["noabl"] + ":\n" + ", ".join(L["g"][g] for g in ("historia", "calendario", "precio")),
                ha="left", va="center", fontsize=6, color="#666666")
        ax.set_xticks([0, 1, 2], L["rk"])
        ax.set_yticks(range(1, n + 1))
        ax.set_ylim(n + 0.5, 0.5)
        ax.set_xlim(-0.9, 3.3)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(L["caso"][c])
    fig.tight_layout()
    guardar(fig, "x4_shap_vs_ablacion")
    # x5: beeswarm (exp. 3 V4 D), 20 variables con mayor |SHAP| medio; SHAP medio entre las 10 semillas
    z = np.load(MOD / "beeswarm_exp3_V4_D.npz", allow_pickle=False)
    phi, xv, nom, grp = z["phi"], z["x"], [str(n) for n in z["nombres"]], [str(g) for g in z["grupos"]]
    orden = np.argsort(-np.abs(phi).mean(axis=0))[:20][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    rng = np.random.default_rng(XP.SEMILLA_EXP4)
    for i, j in enumerate(orden):
        v = xv[:, j]
        lo, hi = np.nanpercentile(v, [5, 95])
        col = np.clip((v - lo) / (hi - lo if hi > lo else 1), 0, 1)
        sc = ax.scatter(phi[:, j], i + rng.uniform(-0.3, 0.3, len(v)), c=col, cmap="coolwarm", s=3, lw=0,
                        rasterized=True)
    ax.set_yticks(range(len(orden)), [f"{nom[j]} [{L['g'][grp[j]]}]" for j in orden], fontsize=6.5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(L["bee"])
    ax.set_title(L["caso"]["exp3_V4_D"])
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(L["valor"])
    cb.set_ticks([0, 1], labels=["p5", "p95"])
    fig.tight_layout()
    guardar(fig, "x5_beeswarm_V4_D")
    # x6
    A = pd.read_csv(SAL / "atencion.csv")
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.6))
    for ax, c in zip(axs.ravel(), casos):
        for rama, col in (("A", "#000000"), ("B", "#D55E00")):
            s = A[(A.caso == c) & (A.rama == rama)]
            ax.errorbar(np.arange(len(s)), s.alpha, yerr=s.sd_semillas, color=col, marker="o", ms=3, lw=1, capsize=2,
                        label=L[f"rama{rama}"])
            ax.set_xticks(np.arange(len(s)), [p.replace("-", "−") for p in s.posicion])
        ax.axhline(1 / 6, color="#999999", lw=0.6, ls="--")
        ax.set_title(L["caso"][c])
        ax.set_ylabel(L["alpha"])
        ax.set_xlabel(L["pos"])
    axs.ravel()[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    guardar(fig, "x6_atencion")
    print("Figuras en", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo-repo":
        hijo_repo(sys.argv[2], sys.argv[3])
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo-explicar":
        hijo_explicar(sys.argv[2], sys.argv[3])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--modelos", action="store_true")
    ap.add_argument("--explicar", action="store_true")
    ap.add_argument("--analizar", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    ap.add_argument("--idioma", default="es", choices=("es", "en"))
    ap.add_argument("--n-proc", type=int, default=6)
    a = ap.parse_args()
    if a.modelos:
        modelos(a.n_proc)
    if a.verificar or a.explicar or a.analizar:
        verificar()
    if a.explicar:
        explicar(a.n_proc)
    if a.analizar:
        analizar()
    if a.figuras:
        figuras(a.idioma)
