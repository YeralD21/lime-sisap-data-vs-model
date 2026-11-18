"""Experimento 4, diagnóstico POSTERIOR no previsto en el anexo (desviación EXP4-3).

Comprueba si el |SHAP| mayor en la posición t-6 (x2) es del modelo o del método: en cada unidad sustituye una
posición completa de la secuencia (todas las columnas de las ramas A y B) por la misma posición de secuencias del
fondo de SHAP (entrenamiento, 10 sorteos, semilla 42) y mide el cambio absoluto medio del pronóstico (t).
Salida: resultados/exp4/diagnostico_posicion.csv.  Uso: .venv311\Scripts\python scripts/12_exp4_diagnostico_posicion.py
"""
import importlib.util
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
spec = importlib.util.spec_from_file_location("e4", RAIZ / "scripts" / "10_exp4.py")
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)
from src import explicabilidad as XP  # noqa: E402

filas = []
for caso in e4.CASOS:
    for s in range(10):
        red, z = e4._cargar(e4.carpeta(caso, s))
        XA, XB, sd = z["XA_test"], z["XB_test"], float(z["sd"])
        ib = XP.indices_fondo(len(z["XA_train"])); bA, bB = z["XA_train"][ib], z["XB_train"][ib]
        f = lambda a, b: red.predict([a, b], verbose=0, batch_size=1024).reshape(-1)
        f0 = f(XA, XB); rng = np.random.default_rng(42)
        for k in range(6):
            d = []
            for r in range(10):
                j = rng.integers(0, len(bA), len(XA))
                a2, b2 = XA.copy(), XB.copy(); a2[:, k] = bA[j, k]; b2[:, k] = bB[j, k]
                d.append(np.abs(f(a2, b2) - f0).mean() * sd)
            filas.append({"caso": caso, "semilla": s, "posicion": f"t-{6-k}", "cambio_abs_medio_t": float(np.mean(d))})
D = pd.DataFrame(filas)
D.to_csv(RAIZ / "resultados" / "exp4" / "diagnostico_posicion.csv", index=False, float_format="%.6g")
print(D.groupby(["caso", "posicion"], sort=False).cambio_abs_medio_t.mean().unstack().round(1).to_string())
