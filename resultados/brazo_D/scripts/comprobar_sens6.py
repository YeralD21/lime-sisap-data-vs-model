"""Sensibilidad 6: comprueba que serie(sens6).y y .precio coinciden con los de la principal desde 2006-01-01
(y, por tanto, que las historias de N = 120, 360 y 1000 son idénticas). Escribe resultados/brazo_D/sens6_comprobacion.csv."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config as C
from src.datos import ESCENARIOS, serie
from src.ventanas import Ventana, particion

a, b = serie(ESCENARIOS["principal"]["D"]), serie(ESCENARIOS["sens6_desde2006"]["D"])
a2 = a[a.index >= C.INICIO_SENS6]
filas = [{"comprobacion": "mismo indice desde 2006-01-01", "resultado": bool(a2.index.equals(b.index))},
         {"comprobacion": "y identico (max abs dif)", "resultado": float(np.max(np.abs(a2.y.to_numpy() - b.y.to_numpy())))},
         {"comprobacion": "precio identico (max abs dif)", "resultado": float(np.max(np.abs(a2.precio.to_numpy() - b.precio.to_numpy())))},
         {"comprobacion": "y_eval identico (NaN en las mismas posiciones y valores iguales)",
          "resultado": bool(np.array_equal(a2.y_eval.to_numpy(), b.y_eval.to_numpy(), equal_nan=True))}]
for y in C.ANIOS_TEST:
    for N in (120, 360, 1000):
        pa, pb = particion(a.index, Ventana(y), N), particion(b.index, Ventana(y), N)
        ok = (np.array_equal(a.y.to_numpy()[pa.hist], b.y.to_numpy()[pb.hist]) and
              np.array_equal(a.precio.to_numpy()[pa.hist], b.precio.to_numpy()[pb.hist]) and
              a.index[pa.hist][0] == b.index[pb.hist][0] and np.array_equal(a.index[pa.test], b.index[pb.test]))
        filas.append({"comprobacion": f"historia, validacion y prueba identicas Y={y} N={N}", "resultado": bool(ok)})
t = pd.DataFrame(filas)
print(t.to_string())
t.to_csv(C.RAIZ / "resultados" / "brazo_D" / "sens6_comprobacion.csv", index=False)
