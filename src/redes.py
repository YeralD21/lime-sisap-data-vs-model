"""Redes del experimento: DualLSTM-Bahdanau y LSTM univariada (docs/PROTOCOLO.md §5).

Arquitectura copiada de `v2_reentrenamiento/src/models/dual_lstm_attention.py` del repositorio del artículo,
commit eee9a71 (futura v2.2-forecasting), sin copiar pesos. Única modificación: el validador de dimensiones
(rama B con 8 variables en D y 6 en M) y el conteo esperado, que se calcula con `expected_parameter_count`.

Cada corrida (escenario, ventana, N, horizonte, modelo, semilla) se ejecuta en un PROCESO NUEVO para que la
semilla se fije antes de importar TensorFlow (como `07_lstm.py` y `determinism.py` del artículo). Uso:

    .venv311\\Scripts\\python -m src.redes --unidad '<json>' --destino DIR      (una corrida)
    from src.redes import correr_unidades; correr_unidades(lista, raiz, n_proc=6)  (orquestador reanudable)

Entradas (§5):
- rama A (4): y_t, y_{t-1}, y_{t-3}, y_{t-6}; LSTM univariada: solo y_t;
- rama B D (8): sen/cos día de la semana, sen/cos día del año, t_index, precio rezagado 1, 3, 6;
- rama B M (6): sen/cos mes, t_index, precio rezagado 1, 3, 6.
Secuencia: filas t-6..t-1 para el objetivo y_t (h = 1) o y_{t+6} (h = 7, estrategia directa).
Escalado: media y desviación poblacional (StandardScaler) ajustadas SOLO con las filas de la historia de tamaño N.
Las secuencias de entrenamiento usan solo filas de la historia N (los primeros 12 pasos se consumen como rezagos).
Validación (early stopping) con objetivos del año Y-1 de la serie imputada; prueba con objetivos del año Y.
Mejores pesos: ModelCheckpoint(save_best_only) sobre val_loss, como en el artículo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

VARIABLES_ENTORNO = {
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_DETERMINISTIC_OPS": "1",
    "PYTHONHASHSEED": "42",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "CUDA_VISIBLE_DEVICES": "-1",
}
HILOS_INTRA, HILOS_INTER = 2, 1
REZAGOS_A = (0, 1, 3, 6)
REZAGOS_PRECIO = (1, 3, 6)
MARGEN = C.RED["lookback"] + max(REZAGOS_A)   # 12 pasos consumidos como rezagos


# ----------------------------------------------------------------------------- variables

def variables(df: pd.DataFrame, brazo: str, modelo: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Matrices de variables sin escalar, alineadas con df (fila j usa datos hasta j). NaN al inicio por rezagos."""
    y = df.y
    A = np.column_stack([y.shift(k).to_numpy() for k in REZAGOS_A])
    if modelo == "LSTM univariada":
        return A[:, :1], None
    t = np.arange(len(df), dtype=float)
    p = df.precio
    lag_p = [p.shift(k).to_numpy() for k in REZAGOS_PRECIO]
    if brazo == "D":
        dow = df.index.dayofweek.to_numpy()
        doy = df.index.dayofyear.to_numpy()
        cal = [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
               np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), t]
    else:
        mes = df.index.month.to_numpy()
        cal = [np.sin(2 * np.pi * mes / 12), np.cos(2 * np.pi * mes / 12), t]
    return A, np.column_stack(cal + lag_p)


class Escalador:
    """StandardScaler (ddof = 0) ajustado con las filas [ini, fin] de la historia; ignora los NaN de los rezagos."""

    def __init__(self, X: np.ndarray, ini: int, fin: int):
        Z = X[ini:fin + 1]
        self.media = np.nanmean(Z, axis=0)
        self.escala = np.nanstd(Z, axis=0)
        self.escala[self.escala == 0] = 1.0
        self.filas = (ini, fin)

    def __call__(self, X):
        return (X - self.media) / self.escala


def objetivos(part, h: int) -> dict[str, np.ndarray]:
    """Posiciones del objetivo por partición. h = 1: objetivo t con contexto t-6..t-1.
    h = 7: objetivo τ = t+6 con contexto t-6..t-1 (último dato τ-7)."""
    lb = C.RED["lookback"]
    paso = h - 1
    ini_ctx = part.ini_hist + max(REZAGOS_A)          # primera fila con todos los rezagos dentro de la historia
    primero = ini_ctx + lb + paso                      # primer objetivo cuyo contexto cabe en la historia
    train = np.arange(primero, part.fin_train + 1)
    return {"train": train, "val": part.val, "test": part.test}


def secuencias(Xs: np.ndarray, pos: np.ndarray, h: int) -> np.ndarray:
    lb = C.RED["lookback"]
    fin_ctx = pos - (h - 1)                            # el contexto termina en fin_ctx - 1
    return np.stack([Xs[f - lb:f] for f in fin_ctx])



def preparar(df: pd.DataFrame, brazo: str, modelo: str, part, h: int):
    """Secuencias escaladas y objetivo escalado. El escalador se ajusta con las filas [ini_hist + 6, fin_train]:
    todas sus variables (incluidos los rezagos) caen dentro de la historia de tamaño N."""
    A, B = variables(df, brazo, modelo)
    ini_fit = part.ini_hist + max(REZAGOS_A)
    ea = Escalador(A, ini_fit, part.fin_train)
    As = ea(A)
    Bs = Escalador(B, ini_fit, part.fin_train)(B) if B is not None else None
    mu, sd = float(ea.media[0]), float(ea.escala[0])   # columna 0 de A = y_t: mismo escalador para el objetivo
    ys = (df.y.to_numpy() - mu) / sd
    pos = objetivos(part, h)
    X = {k: [secuencias(As, v, h)] + ([secuencias(Bs, v, h)] if Bs is not None else []) for k, v in pos.items()}
    if not np.isfinite(np.concatenate([x.ravel() for x in X["train"]])).all():
        raise AssertionError("NaN en las secuencias de entrenamiento: el contexto se sale de la historia N.")
    if pos["train"].max() > part.fin_train or pos["train"].min() - (h - 1) - C.RED["lookback"] - max(REZAGOS_A) < part.ini_hist:
        raise AssertionError("Fuga: objetivo o contexto de entrenamiento fuera de la historia N.")
    return X, ys, pos, (mu, sd), A, B


# ----------------------------------------------------------------------------- arquitectura (copiada)

def expected_parameter_count(n_features_a: int, n_features_b: int) -> int:
    """Fórmula del artículo (dual_lstm_attention.py)."""
    u, a, d1, d2 = C.RED["lstm_units"], C.RED["attention_units"], *C.RED["dense_units"]

    def lstm(i):
        return 4 * u * (i + u + 1)

    att = 2 * u * a + a
    return lstm(n_features_a) + lstm(n_features_b) + 2 * att + (2 * u * d1 + d1) + (d1 * d2 + d2) + (d2 + 1)


def parametros_lstm_univariada() -> int:
    u, (d1, d2) = C.RED["lstm_units"], C.RED["dense_units"]
    return 4 * u * (1 + u + 1) + (u * d1 + d1) + (d1 * d2 + d2) + (d2 + 1)


def _construir(modelo: str, n_a: int, n_b: int | None):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    R = C.RED

    class BahdanauAttention(keras.layers.Layer):
        """Copia literal de la capa del artículo."""

        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W_query = layers.Dense(units, use_bias=False, name="W_query")
            self.W_values = layers.Dense(units, use_bias=False, name="W_values")
            self.V = layers.Dense(1, use_bias=False, name="V")

        def call(self, query, values):
            q_expanded = tf.expand_dims(self.W_query(query), axis=1)
            energy = self.V(tf.nn.tanh(self.W_values(values) + q_expanded))
            alpha = tf.nn.softmax(energy, axis=1)
            return tf.reduce_sum(alpha * values, axis=1)

    lb = R["lookback"]
    if modelo == "DualLSTM":
        if n_a != 4 or n_b not in (6, 8):                 # validador modificado (§5)
            raise ValueError(f"Dimensiones no previstas: A={n_a}, B={n_b}")
        ia = keras.Input(shape=(lb, n_a), name="rama_a")
        ha = layers.LSTM(R["lstm_units"], return_sequences=True, name="lstm_a")(ia)
        ca = BahdanauAttention(R["attention_units"], name="attention_a")(ha[:, -1, :], ha)
        ib = keras.Input(shape=(lb, n_b), name="rama_b")
        hb = layers.LSTM(R["lstm_units"], return_sequences=True, name="lstm_b")(ib)
        cb = BahdanauAttention(R["attention_units"], name="attention_b")(hb[:, -1, :], hb)
        x = layers.Concatenate(name="concatenate")([ca, cb])
        entradas, esperado = [ia, ib], expected_parameter_count(n_a, n_b)
    else:
        ia = keras.Input(shape=(lb, 1), name="rama_a")
        x = layers.LSTM(R["lstm_units"], name="lstm_a")(ia)
        entradas, esperado = [ia], parametros_lstm_univariada()
    x = layers.Dense(R["dense_units"][0], activation="relu", name="dense_16")(x)
    x = layers.Dropout(R["dropout"], name="dropout_020")(x)
    x = layers.Dense(R["dense_units"][1], activation="relu", name="dense_8")(x)
    out = layers.Dense(1, name="output")(x)
    red = keras.Model(inputs=entradas, outputs=out)
    red.compile(optimizer=keras.optimizers.Adam(learning_rate=R["learning_rate"]), loss=R["loss"], metrics=["mae"])
    if red.count_params() != esperado:
        raise ValueError(f"Parámetros {red.count_params()} != {esperado}")
    return red, BahdanauAttention


# ----------------------------------------------------------------------------- una corrida

def _determinismo(semilla: int) -> dict:
    import random

    random.seed(C.SEMILLA_GLOBAL)
    np.random.seed(C.SEMILLA_GLOBAL)
    import tensorflow as tf
    from tensorflow import keras

    tf.config.threading.set_intra_op_parallelism_threads(HILOS_INTRA)
    tf.config.threading.set_inter_op_parallelism_threads(HILOS_INTER)
    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    keras.utils.set_random_seed(semilla)
    tf.config.experimental.enable_op_determinism()
    return {"semilla": semilla, "semilla_global": C.SEMILLA_GLOBAL, "intra_op": HILOS_INTRA, "inter_op": HILOS_INTER,
            "op_determinism": True, **{k: os.environ.get(k) for k in VARIABLES_ENTORNO}}


def entrenar_y_predecir(unidad: dict, destino: Path) -> dict:
    """unidad: {escenario: {brazo, objetivo, inicio, anomalia}, anio, N, h, modelo, semilla}."""
    det = _determinismo(int(unidad["semilla"]))
    from tensorflow import keras

    from src.datos import Escenario, serie
    from src.ventanas import Ventana, particion

    esc = Escenario(**unidad["escenario"])
    df = serie(esc)
    h, modelo = int(unidad["h"]), unidad["modelo"]
    N = unidad["N"] if unidad["N"] == "max" else int(unidad["N"])
    part = particion(df.index, Ventana(int(unidad["anio"])), N)
    X, ys, pos, (mu, sd), A, B = preparar(df, esc.brazo, modelo, part, h)
    red, Att = _construir(modelo, A.shape[1], None if B is None else B.shape[1])
    R = C.RED
    destino.mkdir(parents=True, exist_ok=True)
    ckpt = destino / "checkpoint_best.keras"
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=R["early_stopping_patience"], restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=R["reduce_lr_factor"], patience=R["reduce_lr_patience"],
                                             min_lr=R["min_lr"]),
           keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True)]
    t0 = time.perf_counter()
    hist = red.fit(X["train"], ys[pos["train"]], validation_data=(X["val"], ys[pos["val"]]),
                   epochs=R["max_epochs"], batch_size=R["batch_size"], shuffle=R["shuffle"], callbacks=cbs, verbose=0)
    seg = time.perf_counter() - t0
    mejor = keras.models.load_model(ckpt, custom_objects={"BahdanauAttention": Att}, compile=False)
    filas = []
    for parte in ("val", "test"):
        yp = mejor.predict(X[parte], verbose=0).reshape(-1) * sd + mu
        for p_, v in zip(pos[parte], yp):
            filas.append({"particion": parte, "fecha": df.index[p_].strftime("%Y-%m-%d"), "y_pred": float(v)})
    pd.DataFrame(filas).to_csv(destino / "predicciones.csv", index=False)
    ckpt.unlink(missing_ok=True)
    vl = hist.history["val_loss"]
    estado = {**unidad, "n_params": int(red.count_params()), "n_train": int(len(pos["train"])),
              "n_val": int(len(pos["val"])), "n_test": int(len(pos["test"])), "n_hist": part.n_hist,
              "epocas": len(vl), "mejor_epoca": int(np.argmin(vl)) + 1, "segundos": round(seg, 2),
              "escalador_filas": [df.index[part.ini_hist + max(REZAGOS_A)].strftime("%Y-%m-%d"), df.index[part.fin_train].strftime("%Y-%m-%d")],
              "determinismo": det}
    (destino / "estado.json").write_text(json.dumps(estado, indent=2, default=str), encoding="utf-8")
    return estado


# ----------------------------------------------------------------------------- orquestación

def nombre_unidad(u: dict) -> str:
    e = u["escenario"]
    esc = f"{e['brazo']}_{e['objetivo']}_{e.get('inicio', C.INICIO)[:4]}_{e.get('anomalia', 'faltante')}"
    mod = "dual" if u["modelo"] == "DualLSTM" else "uni"
    return f"{esc}/{mod}/Y{u['anio']}/N{u['N']}/h{u['h']}/s{int(u['semilla']):02d}"


def python_redes() -> str:
    """Intérprete de .venv311 (TensorFlow no tiene distribución para Python 3.14; ver docs/DESVIACIONES.md)."""
    p = C.RAIZ / ".venv311" / "Scripts" / "python.exe"
    return str(p) if p.exists() else sys.executable


def correr_unidad_subproceso(u: dict, raiz: Path) -> dict:
    destino = Path(raiz) / nombre_unidad(u)
    if (destino / "estado.json").exists():
        return json.loads((destino / "estado.json").read_text(encoding="utf-8")) | {"reanudada": True}
    env = {**os.environ, **VARIABLES_ENTORNO}
    tmp = Path(tempfile.mkdtemp(prefix="red_", dir=Path(raiz)))
    r = subprocess.run([python_redes(), "-m", "src.redes", "--unidad", json.dumps(u), "--destino", str(tmp)],
                       cwd=str(C.RAIZ), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{nombre_unidad(u)}: código {r.returncode}\n{r.stderr[-2000:]}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        import shutil
        shutil.rmtree(destino)
    tmp.replace(destino)
    return json.loads((destino / "estado.json").read_text(encoding="utf-8"))


def correr_unidades(unidades: list[dict], raiz: Path, n_proc: int = 6, log=print) -> list[dict]:
    """Ejecuta las unidades en `n_proc` procesos paralelos, cada una en un proceso nuevo. Reanudable.
    El resultado de cada unidad no depende del orden ni del paralelismo (semilla fija y op determinism)."""
    raiz = Path(raiz)
    raiz.mkdir(parents=True, exist_ok=True)
    out = []
    with ThreadPoolExecutor(max_workers=n_proc) as ex:
        futs = {ex.submit(correr_unidad_subproceso, u, raiz): u for u in unidades}
        for i, f in enumerate(futs, 1):
            try:
                e = f.result()
                out.append(e)
                log(f"[{i}/{len(unidades)}] {nombre_unidad(futs[f])} epocas={e.get('epocas')} s={e.get('segundos')}")
            except Exception as exc:  # se registra y se sigue; el orquestador debe revisar los fallos
                log(f"[{i}/{len(unidades)}] FALLO {nombre_unidad(futs[f])}: {exc}")
                out.append({**futs[f], "error": str(exc)})
    return out


def leer_predicciones(raiz: Path) -> pd.DataFrame:
    """Une predicciones.csv y estado.json de todas las unidades bajo `raiz`."""
    filas = []
    for est in Path(raiz).rglob("estado.json"):
        e = json.loads(est.read_text(encoding="utf-8"))
        p = pd.read_csv(est.parent / "predicciones.csv")
        esc = e["escenario"]
        p = p.assign(modelo=e["modelo"], semilla=e["semilla"], N=str(e["N"]), anio=e["anio"], h=e["h"],
                     brazo=esc["brazo"], objetivo=esc["objetivo"], inicio=esc.get("inicio", C.INICIO),
                     anomalia=esc.get("anomalia", "faltante"))
        filas.append(p)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--unidad", required=True)
    ap.add_argument("--destino", required=True)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != VARIABLES_ENTORNO["PYTHONHASHSEED"]:
        sys.exit(subprocess.run([sys.executable, "-m", "src.redes", *sys.argv[1:]], cwd=str(C.RAIZ),
                                env={**os.environ, **VARIABLES_ENTORNO}).returncode)
    print(json.dumps(entrenar_y_predecir(json.loads(a.unidad), Path(a.destino)), default=str))
