"""Consolida en resultados/tiempos_dia3.csv los tiempos de pared del día 3 a partir de los registros de los brazos.

Fuentes: resultados/brazo_D/logs/redes_cadena.log (líneas FASE_FIN), resultados/brazo_D/logs/est_*.log (TOTAL),
resultados/brazo_M/logs/redes_*.log y base_*.log (TOTAL). Ejecutar: .venv\\Scripts\\python resultados/tiempos_dia3.py
"""
import csv
import re
from pathlib import Path

R = Path(__file__).resolve().parent
filas = []

for linea in (R / "brazo_D/logs/redes_cadena.log").read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.match(r"FASE_FIN (\S+) h=\[(\d+)\] unidades=(\d+) fallos=(\d+) segundos=(\d+)", linea)
    if m:
        filas.append({"brazo": "D", "bloque": "redes", "escenario": m[1], "h": m[2], "unidades": m[3],
                      "fallos": m[4], "segundos_pared": m[5], "fuente": "brazo_D/logs/redes_cadena.log"})

for f in sorted((R / "brazo_D/logs").glob("est_*.log")):
    m = re.search(r"TOTAL (\d+(?:\.\d+)?) s", f.read_text(encoding="utf-8", errors="replace"))
    filas.append({"brazo": "D", "bloque": "estadisticos_y_chronos", "escenario": f.stem[4:], "h": "1,7",
                  "unidades": 100, "fallos": "", "segundos_pared": m[1], "fuente": f"brazo_D/logs/{f.name}"})

for f in sorted((R / "brazo_M/logs").glob("*.log")):
    t = f.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"TOTAL redes (\S+): ([\d.]+) s de pared; (\d+) unidades; \d+ reanudadas; (\d+) fallos", t)
    if m:
        filas.append({"brazo": "M", "bloque": "redes", "escenario": m[1], "h": "1", "unidades": m[3], "fallos": m[4],
                      "segundos_pared": m[2], "fuente": f"brazo_M/logs/{f.name}"})
    m = re.search(r"TOTAL base (\S+): ([\d.]+) s; (\d+) pron", t)
    if m:
        filas.append({"brazo": "M", "bloque": "estadisticos_y_chronos", "escenario": m[1], "h": "1", "unidades": "",
                      "fallos": "", "segundos_pared": m[2], "fuente": f"brazo_M/logs/{f.name}"})

with open(R / "tiempos_dia3.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
print(len(filas), "filas")
