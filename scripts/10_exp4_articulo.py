"""Experimento 4 (docs/ANEXO_EXP4.md §2.3): reentrena UNA unidad GC3 o GE de limón sutil, Y = 2025, con el código
del artículo (duallstm-lime-forecasting-peru, eee9a71) SIN modificarlo, y guarda sus entradas.

Se ejecuta con el intérprete del artículo (su venv), en un proceso nuevo por unidad:
    C:\\Machine-learming\\duallstm-lime-forecasting-peru\\venv\\Scripts\\python scripts/10_exp4_articulo.py GC3 0 DESTINO

- Llama a `lib.entorno.preparar_entorno` (variables de entorno y PYTHONHASHSEED = 42, como `07_lstm.py`) y a
  `lib.integridad.Integridad().verificar()` (hashes de los insumos del artículo), igual que `07_lstm.py una`.
- Llama a `lib.modelo_lstm.entrenar_y_predecir(2025, "sutil", modelo, semilla, DESTINO)`. Esa función deja
  `checkpoint_best.keras` en DESTINO (no lo borra), así que no hace falta tocar nada del artículo.
- Recalcula las secuencias con las mismas funciones del artículo y las guarda en DESTINO/entradas.npz.
DESTINO debe estar fuera del repositorio del artículo (aquí: resultados/exp4/modelos/, no versionado).
"""
import sys
from pathlib import Path

ARTICULO = Path(r"C:\Machine-learming\duallstm-lime-forecasting-peru")
sys.path.insert(0, str(ARTICULO / "v2_reentrenamiento" / "experimentos_backtesting_rodante" / "02_scripts"))

from lib.entorno import preparar_entorno  # noqa: E402

preparar_entorno()          # relanza el proceso si PYTHONHASHSEED != 42 (antes de importar numpy o TensorFlow)

import json  # noqa: E402

ANIO, CULTIVAR = 2025, "sutil"


def main(modelo: str, semilla: int, destino: Path) -> None:
    if ARTICULO in destino.resolve().parents:
        raise SystemExit("DESTINO no puede estar dentro del repositorio del artículo.")
    from lib.integridad import Integridad
    from lib.modelo_lstm import entrenar_y_predecir

    Integridad().verificar()
    estado = entrenar_y_predecir(ANIO, CULTIVAR, modelo, semilla, destino)

    import numpy as np

    from lib.datos import ConstructorDatosVentana
    from lib.preparacion import EscaladorVentana, secuencias
    from lib.ventanas import Ventana
    from v2_reentrenamiento.src.models.features_gc3_ge import build_feature_spec
    from v2_reentrenamiento.src.training.config_gc3_ge import OFFICIAL_TRAINING_CONFIG as CFG

    v = Ventana(ANIO)
    esc = EscaladorVentana(v)
    escalado = esc.ajustar_transformar(ConstructorDatosVentana().features(CULTIVAR, v.anio_fin_train))
    spec = build_feature_spec(CULTIVAR, modelo)
    p = secuencias(escalado, spec.rama_a, spec.rama_b, spec.target, v, lookback=CFG.lookback)
    mu, sd = esc.media_escala(spec.target)
    np.savez(destino / "entradas.npz",
             XA_train=p["train"].X_a, XB_train=p["train"].X_b, XA_val=p["val"].X_a, XB_val=p["val"].X_b,
             XA_test=p["test"].X_a, XB_test=p["test"].X_b, ys_test=p["test"].y,
             y_eval_test=esc.invertir(spec.target, p["test"].y),
             fechas_test=np.array([f"{f:%Y-%m-%d}" for f in p["test"].fechas_objetivo]),
             mu=mu, sd=sd, nombres_A=np.array(spec.rama_a), nombres_B=np.array(spec.rama_b))
    (destino / "estado.json").write_text(json.dumps({"fuente": "articulo eee9a71", "modelo": modelo,
                                                      "semilla": semilla, "anio": ANIO, **estado}, indent=2,
                                                     default=str), encoding="utf-8")
    print(json.dumps({"ok": True, "modelo": modelo, "semilla": semilla}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
