"""Experimento 4 (docs/ANEXO_EXP4.md): explicabilidad descriptiva de la DualLSTM ya entrenada.

Funciones puras (sin TensorFlow) para nombrar y agrupar las columnas de entrada, elegir el fondo de SHAP,
permutar en bloques y agregar los valores SHAP; y una función con NumPy que reproduce los pesos de la
atención Bahdanau de `src/redes.py` (copia de eee9a71). Nada de este módulo entrena ni cambia los modelos.

SHAP y la permutación describen el comportamiento del modelo, no causas en el mundo real.
"""
from __future__ import annotations

import numpy as np

from src import exogenas as X3

GRUPOS = ("historia", "calendario", "precio", "clima", "indeci", "noticias")
SEMILLA_EXP4 = 42
N_FONDO = 200
N_REPETICIONES = 30
BLOQUE_PERM = {"D": 7, "M": 3}

# Rama A (src/redes.REZAGOS_A): la fila j lleva y(j), y(j-1), y(j-3), y(j-6)
NOMBRES_A = ("y_l0", "y_l1", "y_l3", "y_l6")
CAL_D = ("dow_sin", "dow_cos", "doy_sin", "doy_cos", "t_index")
CAL_M = ("mes_sin", "mes_cos", "t_index")
PRECIO = ("precio_l1", "precio_l3", "precio_l6")


# ----------------------------------------------------------------------------- nombres y grupos

def nombres_B(brazo: str, variante: str | None) -> list[str]:
    """Columnas de la rama B en el orden de `src/redes.variables` (+ `src/exogenas.matriz_B` si hay variante).
    variante None = rama B del experimento 2 (calendario + precio)."""
    base = list(CAL_D if brazo == "D" else CAL_M) + list(PRECIO)
    fams = X3.VARIANTES[variante] if variante else ()
    return base + X3.columnas_B(brazo, fams)


def grupo_repo(nombre: str, brazo: str) -> str:
    """Grupo de una columna de la rama B de este repositorio (exp. 2 y 3)."""
    if nombre in CAL_D or nombre in CAL_M:
        return "calendario"
    if nombre in PRECIO:
        return "precio"
    raiz = nombre.rsplit("_", 1)[0]
    if raiz in ("sent", "sin"):
        return "noticias"
    for fam, cols in X3.FAMILIAS[brazo].items():
        if raiz in cols:
            return fam
    raise KeyError(f"Columna sin grupo: {nombre}")


def grupo_articulo(nombre: str) -> str:
    """Grupo de una columna de la rama B del artículo (features_gc3_ge.py, eee9a71)."""
    if nombre in ("mes_sin", "mes_cos", "t_index"):
        return "calendario"
    if nombre.startswith(("T2M", "WS2M", "PRECTOTCORR", "RH2M")):
        return "clima"
    if nombre.startswith(("num_emergencias", "personas_", "hectareas_")):
        return "indeci"
    if nombre.startswith(("avg_sentiment", "n_noticias")):
        return "noticias"
    raise KeyError(f"Columna del artículo sin grupo: {nombre}")


def mapa_grupos(nombres_b: list[str], grupo_de) -> dict[str, tuple[str, np.ndarray]]:
    """{grupo: (rama, índices de columna)}. La rama A entera es «historia». Solo grupos presentes."""
    out = {"historia": ("A", np.arange(len(NOMBRES_A)))}
    gb = np.array([grupo_de(n) for n in nombres_b])
    for g in GRUPOS[1:]:
        idx = np.flatnonzero(gb == g)
        if len(idx):
            out[g] = ("B", idx)
    return out


# ----------------------------------------------------------------------------- fondo y permutación

def indices_fondo(n_train: int, n: int = N_FONDO, semilla: int = SEMILLA_EXP4) -> np.ndarray:
    """Índices (ordenados) de las secuencias de ENTRENAMIENTO usadas como fondo de SHAP.
    Si hay menos de n secuencias de entrenamiento, se usan todas."""
    if n_train <= n:
        return np.arange(n_train)
    return np.sort(np.random.default_rng(semilla).choice(n_train, size=n, replace=False))


def permutacion_bloques(n: int, bloque: int, rng: np.random.Generator) -> np.ndarray:
    """Permutación de 0..n-1 que baraja bloques contiguos de tamaño `bloque` (el último puede ser más corto)
    y conserva el orden dentro de cada bloque."""
    bloques = [np.arange(i, min(i + bloque, n)) for i in range(0, n, bloque)]
    orden = rng.permutation(len(bloques))
    return np.concatenate([bloques[k] for k in orden])


def permutar_grupo(XA: np.ndarray, XB: np.ndarray | None, rama: str, idx: np.ndarray, perm: np.ndarray):
    """Copia de las entradas con las columnas del grupo (las 6 posiciones de la secuencia) tomadas de la muestra
    perm[i] en lugar de la i. Las demás columnas no cambian."""
    XA2 = XA.copy()
    XB2 = None if XB is None else XB.copy()
    if rama == "A":
        XA2[:, :, idx] = XA[perm][:, :, idx]
    else:
        XB2[:, :, idx] = XB[perm][:, :, idx]
    return XA2, XB2


# ----------------------------------------------------------------------------- agregación de SHAP

def contribucion_grupos(phiA: np.ndarray, phiB: np.ndarray | None, grupos: dict) -> dict[str, np.ndarray]:
    """Contribución con signo de cada grupo por muestra y posición: {grupo: (n, lookback)}.
    Es la suma de los SHAP de sus columnas (aditividad), sin valor absoluto todavía."""
    out = {}
    for g, (rama, idx) in grupos.items():
        phi = phiA if rama == "A" else phiB
        out[g] = phi[:, :, idx].sum(axis=2)
    return out


def importancia_grupos(phiA, phiB, grupos) -> dict[str, dict]:
    """Por grupo:
    - abs_total: media sobre muestras de |Σ_{columnas, posiciones} φ| (medida principal del anexo);
    - suma_abs: media de Σ |φ| (secundaria; crece con el número de columnas);
    - abs_pos: media de |Σ_columnas φ| en cada posición de la secuencia (vector de longitud lookback)."""
    c = contribucion_grupos(phiA, phiB, grupos)
    out = {}
    for g, (rama, idx) in grupos.items():
        phi = (phiA if rama == "A" else phiB)[:, :, idx]
        out[g] = {"abs_total": float(np.abs(c[g].sum(axis=1)).mean()),
                  "suma_abs": float(np.abs(phi).sum(axis=(1, 2)).mean()),
                  "abs_pos": np.abs(c[g]).mean(axis=0)}
    return out


# ----------------------------------------------------------------------------- atención (NumPy)

def pesos_atencion(h: np.ndarray, Wq: np.ndarray, Wv: np.ndarray, V: np.ndarray) -> np.ndarray:
    """α de la capa BahdanauAttention de src/redes.py con query = último estado oculto.
    h: (n, lookback, unidades). Devuelve (n, lookback); cada fila suma 1."""
    q = h[:, -1, :] @ Wq                                  # (n, a)
    e = np.tanh(h @ Wv + q[:, None, :]) @ V              # (n, lookback, 1)
    e = e[..., 0]
    e = e - e.max(axis=1, keepdims=True)
    a = np.exp(e)
    return a / a.sum(axis=1, keepdims=True)
