"""Auditoría día 3: pronósticos negativos (toneladas < 0) por escenario, modelo, N y h en resultados/brazo_*/<esc>/pronosticos.csv.
Escribe reportes/auditoria/pronosticos_negativos.csv."""
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[2] / "resultados"
filas = []
for b, es in (("M", ("principal", "sens1_estricto", "sens7_anomalia_publicada")),
              ("D", ("principal", "sens1_estricto", "sens2_bolsa", "sens6_desde2006", "sens7_anomalia_publicada"))):
    for e in es:
        p = pd.read_csv(R / f"brazo_{b}" / e / "pronosticos.csv", dtype={"N": str})
        g = p.groupby(["modelo", "N", "h"]).y_pred.agg(n="size", n_negativos=lambda s: int((s < 0).sum()), minimo="min").reset_index()
        filas.append(g.assign(brazo=b, escenario=e))
t = pd.concat(filas)[["brazo", "escenario", "modelo", "N", "h", "n", "n_negativos", "minimo"]]
t.to_csv(Path(__file__).resolve().parent / "pronosticos_negativos.csv", index=False, float_format="%.6g")
print(t.groupby(["brazo", "escenario"])[["n", "n_negativos"]].sum())
print(t[(t.escenario == "principal") & (t.brazo == "D")].to_string())
