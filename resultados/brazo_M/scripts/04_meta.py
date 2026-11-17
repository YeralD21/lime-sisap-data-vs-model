"""Escribe versiones.json y semillas.json del brazo M (ejecutar con .venv311 desde la raíz)."""
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
from src import config as C  # noqa: E402

BASE = RAIZ / "resultados" / "brazo_M"
paquetes = ["numpy", "pandas", "scipy", "statsmodels", "tensorflow", "keras", "torch", "chronos-forecasting", "transformers",
            "scikit-learn"]
v = {"python": platform.python_version(), "plataforma": platform.platform()}
for p in paquetes:
    v[p] = metadata.version(p)
v["chronos_modelo"] = C.CHRONOS_ID
v["chronos_snapshot"] = C.CHRONOS_SNAPSHOT
(BASE / "versiones.json").write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
(BASE / "semillas.json").write_text(json.dumps({
    "semillas_redes": list(C.SEMILLAS), "semilla_global": C.SEMILLA_GLOBAL, "semilla_bootstrap": C.SEMILLA_BOOT,
    "B": C.B, "PYTHONHASHSEED": 42,
    "nota": "Cada unidad de red fija su semilla antes de entrenar; ETS, Theta, SARIMA, Seasonal Naive y Chronos son deterministas."},
    indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(v, indent=1))
