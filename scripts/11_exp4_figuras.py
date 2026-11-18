"""Experimento 4: figuras x1-x6 con una corrección de maquetación de scripts/10_exp4.py (desviación EXP4-2).

En x4_shap_vs_ablacion, las etiquetas del eje x «Permutación» y «Ablación exp. 3» se solapan. 10_exp4.py está
congelado con el anexo (b052f2d), así que este script lo reutiliza SIN modificarlo (su SHA-256 sigue
coincidiendo con docs/hashes_anexo_exp4.json) y solo parte en dos líneas esas etiquetas. Las cifras no cambian.
Uso: .venv\\Scripts\\python scripts/11_exp4_figuras.py --idioma es   (y en)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

AQUI = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("exp4", AQUI.parent / "10_exp4.py")
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)

e4.TXT["es"]["rk"] = ["SHAP", "Permuta-\nción", "Ablación\nexp. 3"]       # única diferencia con 10_exp4.py
e4.TXT["en"]["rk"] = ["SHAP", "Permu-\ntation", "Exp. 3\nablation"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idioma", default="es", choices=("es", "en"))
    e4.figuras(ap.parse_args().idioma)
