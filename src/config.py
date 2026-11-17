"""Constantes del protocolo (docs/PROTOCOLO.md, congelado en 9ede96c). Ningún valor se afina con datos."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIARIO = RAIZ / "data" / "processed" / "sutil_diario.csv"
MENSUAL = RAIZ / "data" / "processed" / "sutil_mensual.csv"
RESULTADOS = RAIZ / "resultados"

# §2: inicio de ambos brazos y fecha final de uso (2026 no se usa)
INICIO = "1997-10-01"
INICIO_SENS6 = "2006-01-01"          # §9, sensibilidad 6 (solo D)
FIN_DATOS = "2025-12-31"

# §3: ventanas rolling origin
ANIOS_TEST = (2021, 2022, 2023, 2024, 2025)
ANIOS_POOL_SENS = (2023, 2024, 2025)  # §9, sensibilidad 4

# §4: tamaños de historia ("max" = desde INICIO)
TAMANOS = {"D": (120, 360, 1000, "max"), "M": (60, 120, 240, "max")}

# §5 / §6: periodo estacional
PERIODO = {"D": 7, "M": 12}
HORIZONTES = {"D": (1, 7), "M": (1,)}

# §5: modelos
REFERENCIAS = ("Seasonal Naive", "ETS", "Theta", "SARIMA", "Chronos-Bolt-tiny")
REDES = ("LSTM univariada", "DualLSTM")
MODELOS = REFERENCIAS + REDES

# Hiperparámetros congelados del artículo (hiperparametros_congelados.json, gc3_ge)
RED = {
    "lookback": 6, "lstm_units": 16, "attention_units": 16, "dense_units": (16, 8), "dropout": 0.20,
    "learning_rate": 0.001, "loss": "mse", "batch_size": 8, "max_epochs": 300, "shuffle": False,
    "early_stopping_patience": 15, "reduce_lr_patience": 8, "reduce_lr_factor": 0.5, "min_lr": 1e-6,
}
SEMILLAS = tuple(range(10))
SEMILLA_GLOBAL = 42

# §6: bootstrap y top-ranked
B = 10_000
SEMILLA_BOOT = 42
BLOQUE = {"D": 7, "M": 3}            # §6, sensibilidad 5
UMBRALES_P = (50, 60, 75, 90)
MIN_UMBRALES_TOP = 3

# §5: Chronos
CHRONOS_ID = "amazon/chronos-bolt-tiny"
CHRONOS_SNAPSHOT = "a0e552de83495b5c28c14c71c374f3e33280b340"

# Código del artículo (solo lectura). Ver docs/DESVIACIONES.md (versión del artículo).
ARTICULO_COMMIT = "eee9a71 (futura v2.2-forecasting)"
ARTICULO_RAIZ = Path(r"C:\Machine-learming\duallstm-lime-forecasting-peru")
