"""Escribe resultados/brazo_D/versiones.json y semillas.json."""
import json, sys, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config as C
import numpy, pandas, statsmodels, scipy, torch, transformers, tensorflow, keras, chronos
from importlib.metadata import version
v = {"python": sys.version, "plataforma": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__,
     "statsmodels": statsmodels.__version__, "scipy": scipy.__version__, "tensorflow": tensorflow.__version__,
     "keras": keras.__version__, "torch": torch.__version__, "chronos-forecasting": version("chronos-forecasting"),
     "transformers": transformers.__version__, "chronos_snapshot": C.CHRONOS_SNAPSHOT}
b = C.RAIZ / "resultados" / "brazo_D"
(b / "versiones.json").write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
s = {"semillas_redes": list(C.SEMILLAS), "semilla_global": C.SEMILLA_GLOBAL, "semilla_bootstrap": C.SEMILLA_BOOT, "B": C.B,
     "PYTHONHASHSEED": 42, "nota": "modelos estadisticos y Chronos son deterministas (sin semilla)"}
(b / "semillas.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
print(v)
