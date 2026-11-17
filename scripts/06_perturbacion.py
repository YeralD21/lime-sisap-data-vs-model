"""Día 4, T3: prueba de perturbación (ANÁLISIS DIAGNÓSTICO POSTERIOR; no modifica ninguna etiqueta).

Para una unidad de red por brazo (principal, Y = 2025, N = máx, DualLSTM, semilla 0) se alteran al azar todas las
observaciones posteriores al origen (fin de la validación, 2024-12-31): y, y_eval y precio. Luego se repiten el
escalado, la construcción de secuencias, el entrenamiento y el pronóstico, y se compara con la serie original:
- escalador (medias y escalas), secuencias y objetivos de entrenamiento y validación, entrada del primer paso de prueba;
- pronóstico del primer día (o mes) de prueba;
- valor observado de ese día (debe cambiar).

Cada entrenamiento corre en un subproceso nuevo con el mismo determinismo que el día 3 (src/redes.py, D4).
Salida: reportes/figuras/datos/perturbacion.csv (solo diferencias; ningún valor de la serie del SISAP).
Uso: .venv311\\Scripts\\python scripts/06_perturbacion.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SALIDA = RAIZ / "reportes" / "figuras" / "datos" / "perturbacion.csv"
SEMILLA_PERTURBACION = 20250101
UNIDADES = {
    "D": "resultados/brazo_D/redes_unidades/D_principal_1997_faltante/dual/Y2025/Nmax/h1/s00",
    "M": "resultados/brazo_M/redes_unidades/M_principal_1997_faltante/dual/Y2025/Nmax/h1/s00",
}


def perturbar(df: pd.DataFrame, origen: int) -> pd.DataFrame:
    """Sustituye y, y_eval y precio posteriores al origen por valores aleatorios en el rango de la serie."""
    rng = np.random.default_rng(SEMILLA_PERTURBACION)
    out = df.copy()
    n = len(out) - origen - 1
    for col in ("y", "y_eval", "precio"):
        v = out[col].to_numpy(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        nuevo = rng.uniform(lo, hi, n)
        if col == "y_eval":
            nuevo = np.where(np.isnan(v[origen + 1:]), np.nan, nuevo)
        out.iloc[origen + 1:, out.columns.get_loc(col)] = nuevo
    return out


def _serie_y_particion(unidad):
    from src.datos import Escenario, serie
    from src.ventanas import Ventana, particion
    esc = Escenario(**unidad["escenario"])
    df = serie(esc)
    part = particion(df.index, Ventana(int(unidad["anio"])), unidad["N"])
    return esc, df, part


def hijo(unidad_json: str, perturbada: str, destino: str) -> None:
    """Subproceso: entrena y predice con la serie original o perturbada (parchea src.datos.serie)."""
    import src.datos as D
    from src import redes as R
    unidad = json.loads(Path(unidad_json).read_text(encoding="utf-8"))
    _, df, part = _serie_y_particion(unidad)
    if perturbada == "1":
        df = perturbar(df, int(part.val[-1]))
    D.serie = lambda esc, _df=df: _df
    R.entrenar_y_predecir(unidad, Path(destino))


def principal() -> None:
    from src import redes as R
    filas = []
    for brazo, ruta in UNIDADES.items():
        estado = json.loads((RAIZ / ruta / "estado.json").read_text(encoding="utf-8"))
        unidad = {k: estado[k] for k in ("escenario", "anio", "N", "h", "modelo", "semilla")}
        esc, df, part = _serie_y_particion(unidad)
        origen = int(part.val[-1])
        dfp = perturbar(df, origen)
        h = int(unidad["h"])
        X, ys, pos, ms, _, _ = R.preparar(df, esc.brazo, unidad["modelo"], part, h)
        Xp, ysp, posp, msp, _, _ = R.preparar(dfp, esc.brazo, unidad["modelo"], part, h)
        t0 = int(pos["test"][0])

        def dif(a, b):
            return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))

        n_cambiados = int((dfp.y.to_numpy()[origen + 1:] != df.y.to_numpy()[origen + 1:]).sum())
        comp = {
            "escalador_y (media, escala)": dif(ms, msp),
            "secuencias_entrenamiento": max(dif(a, b) for a, b in zip(X["train"], Xp["train"])),
            "objetivos_entrenamiento": dif(ys[pos["train"]], ysp[posp["train"]]),
            "secuencias_validacion": max(dif(a, b) for a, b in zip(X["val"], Xp["val"])),
            "objetivos_validacion": dif(ys[pos["val"]], ysp[posp["val"]]),
            "entrada_primer_paso_prueba": max(dif(a[0], b[0]) for a, b in zip(X["test"], Xp["test"])),
            "entrada_ultimo_paso_prueba": max(dif(a[-1], b[-1]) for a, b in zip(X["test"], Xp["test"])),
            "observado_primer_paso_prueba": abs(float(dfp.y_eval.iloc[t0] - df.y_eval.iloc[t0])),
        }
        # entrenamientos en subprocesos nuevos (original y perturbada)
        pred = {}
        with tempfile.TemporaryDirectory() as tmp:
            uj = Path(tmp) / "unidad.json"
            uj.write_text(json.dumps(unidad), encoding="utf-8")
            for etiqueta in ("0", "1"):
                dest = Path(tmp) / f"p{etiqueta}"
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--hijo", str(uj), etiqueta, str(dest)],
                               check=True, cwd=RAIZ, env={**os.environ, **R.VARIABLES_ENTORNO})
                pred[etiqueta] = pd.read_csv(dest / "predicciones.csv")
                pred[etiqueta + "_estado"] = json.loads((dest / "estado.json").read_text(encoding="utf-8"))
        publ = pd.read_csv(RAIZ / ruta / "predicciones.csv")
        f0 = df.index[t0].strftime("%Y-%m-%d")
        def _p(t):
            return float(t.loc[(t.particion == "test") & (t.fecha == f0), "y_pred"].iloc[0])
        p0 = {k: _p(pred[k]) for k in ("0", "1")}
        pp = _p(publ)
        comp.update({
            "pronostico_primer_paso_prueba (perturbada vs original)": abs(p0["1"] - p0["0"]),
            "pronostico_primer_paso_prueba (original vs publicado dia 3)": abs(p0["0"] - pp),
            "pronosticos_validacion (perturbada vs original)": dif(pred["1"].query("particion == 'val'").y_pred,
                                                                    pred["0"].query("particion == 'val'").y_pred),
            "pronosticos_prueba_resto (perturbada vs original)": dif(pred["1"].query("particion == 'test'").y_pred.iloc[1:],
                                                                      pred["0"].query("particion == 'test'").y_pred.iloc[1:]),
            "epocas (perturbada - original)": float(pred["1_estado"]["epocas"] - pred["0_estado"]["epocas"]),
            "mejor_epoca (perturbada - original)": float(pred["1_estado"]["mejor_epoca"] - pred["0_estado"]["mejor_epoca"]),
        })
        for k, v in comp.items():
            filas.append({"brazo": brazo, "unidad": ruta, "fecha_origen": df.index[origen].strftime("%Y-%m-%d"),
                          "primer_paso_prueba": f0, "observaciones_alteradas": n_cambiados,
                          "comparacion": k, "dif_max_abs": v})
        print(brazo, "listo")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA, index=False, float_format="%.6g")
    print(pd.DataFrame(filas)[["brazo", "comparacion", "dif_max_abs"]].to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo":
        hijo(*sys.argv[2:5])
    else:
        principal()
