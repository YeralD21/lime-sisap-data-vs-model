"""Brazo M, h = 1: LSTM univariada y DualLSTM-Bahdanau (10 semillas) con src.redes.correr_unidades (n_proc = 2).

Uso (desde la raíz, con .venv311):
    .venv311/Scripts/python.exe resultados/brazo_M/scripts/02_redes.py --escenario principal
    ... --anios 2023 --N 120 --semillas 0 --modelos DualLSTM --raiz resultados/brazo_M/_prueba/redes_unidades  (prueba)
Es reanudable: las unidades con estado.json se saltan. Escribe en <raiz> (por defecto resultados/brazo_M/redes_unidades/)
y los tiempos de pared en resultados/brazo_M/<carpeta>/redes_tiempo_pared.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

from src import config as C  # noqa: E402
from src.datos import ESCENARIOS  # noqa: E402
from src.redes import correr_unidades  # noqa: E402

CARPETAS = {"principal": "principal", "sens1_estricto": "sens1_estricto", "sens7_anomalia_publicada": "sens7_anomalia_publicada"}


def unidades(nombre: str, anios, Ns, semillas, modelos) -> list[dict]:
    e = ESCENARIOS[nombre]["M"]
    esc = {"brazo": "M", "objetivo": e.objetivo, "anomalia": e.anomalia}
    return [{"escenario": esc, "anio": int(y), "N": N if N == "max" else int(N), "h": 1, "modelo": m, "semilla": int(s)}
            for y in anios for N in Ns for m in modelos for s in semillas]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", required=True, choices=list(CARPETAS))
    ap.add_argument("--anios", nargs="*", type=int, default=list(C.ANIOS_TEST))
    ap.add_argument("--N", nargs="*", default=[str(n) for n in C.TAMANOS["M"]])
    ap.add_argument("--semillas", nargs="*", type=int, default=list(C.SEMILLAS))
    ap.add_argument("--modelos", nargs="*", default=list(C.REDES))
    ap.add_argument("--raiz", default=str(RAIZ / "resultados" / "brazo_M" / "redes_unidades"))
    ap.add_argument("--n_proc", type=int, default=2)
    a = ap.parse_args()
    us = unidades(a.escenario, a.anios, a.N, a.semillas, a.modelos)
    print(f"{len(us)} unidades de {a.escenario}", flush=True)
    t0 = time.perf_counter()
    out = correr_unidades(us, Path(a.raiz), n_proc=min(a.n_proc, 2), log=lambda s: print(s, flush=True))
    seg = time.perf_counter() - t0
    fallos = [o for o in out if "error" in o]
    reanudadas = sum(1 for o in out if o.get("reanudada"))
    print(f"TOTAL redes {a.escenario}: {seg:.1f} s de pared; {len(out)} unidades; {reanudadas} reanudadas; {len(fallos)} fallos", flush=True)
    if a.raiz.endswith("redes_unidades"):
        ruta = Path(a.raiz).parent / CARPETAS[a.escenario] / "redes_tiempo_pared.json"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        previo = json.loads(ruta.read_text()) if ruta.exists() else {"corridas": []}
        previo["corridas"].append({"segundos_pared": round(seg, 1), "unidades": len(out), "reanudadas": reanudadas,
                                   "fallos": len(fallos), "n_proc": min(a.n_proc, 2)})
        ruta.write_text(json.dumps(previo, indent=1), encoding="utf-8")
    if fallos:
        for f in fallos:
            print("FALLO", f["error"][:500], flush=True)
        sys.exit(1)
