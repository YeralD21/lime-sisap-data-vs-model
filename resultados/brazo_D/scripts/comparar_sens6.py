"""Sensibilidad 6: diferencia maxima absoluta entre los pronosticos de la principal y los de sens6 para N = 120, 360, 1000 (h = 1),
por modelo. Escribe resultados/brazo_D/sens6_vs_principal.csv. (Las redes de sens6 con N <= 1000 SON las de la principal: diferencia 0 por construccion.)"""
import pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parents[1]
a = pd.read_csv(B / "principal" / "pronosticos.csv", dtype={"N": str})
b = pd.read_csv(B / "sens6_desde2006" / "pronosticos.csv", dtype={"N": str})
a = a[(a.h == 1) & a.N.isin(["120", "360", "1000"])]
b = b[b.N.isin(["120", "360", "1000"])]
m = a.merge(b, on=["modelo", "semilla", "N", "anio", "fecha"], suffixes=("_a", "_b"))
m["d"] = (m.y_pred_a - m.y_pred_b).abs()
t = m.groupby(["modelo", "N"]).d.agg(["max", "mean", "size"]).reset_index().rename(
    columns={"max": "dif_max_abs_t", "mean": "dif_media_abs_t", "size": "n_pronosticos"})
t.to_csv(B / "sens6_vs_principal.csv", index=False)
print(t.to_string())
