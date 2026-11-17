"""Lanza las unidades de modelos estadísticos y Chronos con a lo sumo 2 procesos. Reanudable.
Uso: python resultados/brazo_D/scripts/estadisticos_run.py --escenario principal --hs 1 7 [--N 120 360 1000 max]
     [--modelos ETS SARIMA ...] [--anios 2021 ...] [--workers 2]"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estadisticos_lib as L  # noqa: E402


def _tarea(args):
    return L.correr_unidad(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", required=True)
    ap.add_argument("--hs", type=int, nargs="+", default=[1])
    ap.add_argument("--N", nargs="+", default=["120", "360", "1000", "max"])
    ap.add_argument("--modelos", nargs="+", default=list(L.MODELOS_EST))
    ap.add_argument("--anios", type=int, nargs="+", default=list(L.C.ANIOS_TEST))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.workers > 2:
        sys.exit("Presupuesto: como máximo 2 procesos.")
    orden_N = {"max": 0, "1000": 1, "360": 2, "120": 3}
    orden_m = {"ETS": 0, "SARIMA": 1, "Chronos-Bolt-tiny": 2, "Theta": 3, "Seasonal Naive": 4}
    tareas = sorted([(a.escenario, m, y, n, tuple(a.hs)) for m in a.modelos for y in a.anios for n in a.N],
                    key=lambda t: (orden_N[t[3]], orden_m[t[1]], t[2]))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_tarea, t): t for t in tareas}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                r = f.result()
                print(f"[{i}/{len(tareas)}] {t[1]} Y{t[2]} N{t[3]} {r.get('segundos', 'reanudada')} s  (t={time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(tareas)}] FALLO {t[1]} Y{t[2]} N{t[3]}: {type(e).__name__}: {e}", flush=True)
    print(f"TOTAL {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
