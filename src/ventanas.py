"""Ventanas rolling origin y tamaños de historia (docs/PROTOCOLO.md §3 y §4).

Todo se expresa en posiciones enteras sobre el índice de `datos.serie(esc)`.
- entrenamiento: hasta el 31-12 de Y-2 (en M, el mes 12 de Y-2);
- historia de tamaño N: las últimas N observaciones del entrenamiento (N = "max": desde el inicio de la serie);
- validación: año Y-1 completo; prueba: año Y completo. Iguales para todos los N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C


@dataclass(frozen=True)
class Ventana:
    anio: int

    def __post_init__(self):
        if self.anio not in C.ANIOS_TEST:
            raise ValueError(f"Año de prueba fuera del preregistro: {self.anio}")

    @property
    def fin_train(self) -> pd.Timestamp:
        return pd.Timestamp(self.anio - 2, 12, 31)

    @property
    def origen(self) -> pd.Timestamp:
        """Último dato antes de la prueba (fin de la validación). Se usa para los umbrales de shock (§6)."""
        return pd.Timestamp(self.anio - 1, 12, 31)


@dataclass(frozen=True)
class Particion:
    ventana: Ventana
    N: int | str
    ini_hist: int          # primera posición de la historia de tamaño N
    fin_train: int         # última posición del entrenamiento (incluida)
    val: np.ndarray        # posiciones del año Y-1
    test: np.ndarray       # posiciones del año Y

    @property
    def n_hist(self) -> int:
        return self.fin_train - self.ini_hist + 1

    @property
    def hist(self) -> slice:
        return slice(self.ini_hist, self.fin_train + 1)


def particion(indice: pd.DatetimeIndex, ventana: Ventana, N: int | str) -> Particion:
    anio = indice.year
    fin = int(np.flatnonzero(indice <= ventana.fin_train)[-1])
    if indice[fin].year != ventana.anio - 2 or indice[fin].month != 12:
        raise AssertionError("El entrenamiento no termina en diciembre de Y-2.")
    ini = 0 if N == "max" else fin - int(N) + 1
    if ini < 0:
        raise ValueError(f"N={N} supera la historia disponible ({fin + 1}).")
    val = np.flatnonzero(anio == ventana.anio - 1)
    test = np.flatnonzero(anio == ventana.anio)
    if len(val) == 0 or len(test) == 0 or val[0] != fin + 1 or test[0] != val[-1] + 1:
        raise AssertionError("Validación o prueba no contiguas al entrenamiento.")
    return Particion(ventana, N, ini, fin, val, test)


def evaluables(df: pd.DataFrame, p: Particion) -> np.ndarray:
    """Posiciones de prueba evaluables (§2.2): y_eval no faltante. No depende de N, del modelo ni del horizonte
    (a h = 7 el objetivo es el día t+7 de la prueba y se evalúa si no es faltante)."""
    return p.test[df.y_eval.notna().to_numpy()[p.test]]


def escala_mase(df: pd.DataFrame, brazo: str, ventana: Ventana) -> float:
    """§6: MAE dentro de la muestra del Seasonal Naive (m = 7 o 12) sobre la historia MÁXIMA de entrenamiento
    de la ventana (serie imputada), igual para todos los N."""
    p = particion(df.index, ventana, "max")
    y = df.y.to_numpy()[p.hist]
    m = C.PERIODO[brazo]
    return float(np.mean(np.abs(y[m:] - y[:-m])))
