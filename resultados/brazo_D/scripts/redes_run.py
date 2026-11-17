"""Lanza las unidades de redes (LSTM univariada y DualLSTM, 10 semillas) con src.redes.correr_unidades (reanudable).
Uso: python resultados/brazo_D/scripts/redes_run.py --escenario principal --h 1 [--N ...] [--anios ...] [--modelos ...]
     [--semillas ...] [--n_proc 4]
Escenarios: nombres de src.datos.ESCENARIOS (principal, sens1_estricto, sens2_bolsa, sens6_desde2006, sens7_anomalia_publicada)."""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

from src import config as C  # noqa: E402
from src.datos import ESCENARIOS  # noqa: E402
from src.redes import correr_unidades  # noqa: E402

RAIZ_UNIDADES = RAIZ / "resultados" / "brazo_D" / "redes_unidades"


def unidades(escenario, hs, Ns, anios, modelos, semillas):
    esc = dataclasses.asdict(ESCENARIOS[escenario]["D"])
    esc = {k: esc[k] for k in ("brazo", "objetivo", "inicio", "anomalia")}
    orden_N = {"max": 0, "1000": 1, "360": 2, "120": 3}
    us = [{"escenario": esc, "anio": y, "N": n, "h": h, "modelo": m, "semilla": s}
          for h in hs for n in Ns for y in anios for m in modelos for s in semillas]
    return sorted(us, key=lambda u: (orden_N[str(u["N"])], u["h"], u["anio"], u["modelo"], u["semilla"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", required=True)
    ap.add_argument("--h", type=int, nargs="+", default=[1])
    ap.add_argument("--N", nargs="+", default=[str(n) for n in C.TAMANOS["D"]])
    ap.add_argument("--anios", type=int, nargs="+", default=list(C.ANIOS_TEST))
    ap.add_argument("--modelos", nargs="+", default=list(C.REDES))
    ap.add_argument("--semillas", type=int, nargs="+", default=list(C.SEMILLAS))
    ap.add_argument("--n_proc", type=int, default=4)
    a = ap.parse_args()
    if a.n_proc > 4:
        sys.exit("Presupuesto: como máximo 4 procesos de red.")
    us = unidades(a.escenario, a.h, a.N, a.anios, a.modelos, a.semillas)
    t0 = time.time()
    print(f"FASE_INICIO {a.escenario} h={a.h} unidades={len(us)}", flush=True)
    res = correr_unidades(us, RAIZ_UNIDADES, n_proc=a.n_proc, log=lambda s: print(s, flush=True))
    fallos = [r for r in res if "error" in r]
    print(f"FASE_FIN {a.escenario} h={a.h} unidades={len(us)} fallos={len(fallos)} segundos={time.time()-t0:.0f}", flush=True)
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
