"""Dia 2 - T6: SHA-256 del protocolo y de las series procesadas (preparacion de la congelacion).

Escribe docs/hashes_protocolo.json. NO congela nada: la congelacion es el commit de YesaR.

Metodo de hash: archivos de texto (.md .csv .py .yaml .json) con CRLF -> LF y luego SHA-256
(mismo criterio que el "metodo 2" del repositorio del articulo); el repositorio fuerza LF
con .gitattributes, asi que el hash no depende del sistema operativo del checkout.

Uso:
  python scripts/04_hashes_protocolo.py              # (re)escribe docs/hashes_protocolo.json
  python scripts/04_hashes_protocolo.py --verificar  # compara contra el JSON; sale con 1 si algo difiere
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "hashes_protocolo.json"
TEXTO = {".md", ".csv", ".py", ".yaml", ".json", ".txt"}

ARCHIVOS = {
    "protocolo": ["docs/PROTOCOLO.md"],
    "series_procesadas": ["data/processed/sutil_diario.csv", "data/processed/sutil_mensual.csv",
                          "data/processed/sutil_diario_procedencia.csv"],
    "procedencia_de_las_series": ["reportes/manifiesto_crudos.csv", "config/fuentes.yaml",
                                  "scripts/sisap_comun.py", "scripts/02_descarga_sisap.py", "scripts/03_series.py"],
}


def sha256_lf(ruta: Path) -> str:
    b = ruta.read_bytes()
    if ruta.suffix.lower() in TEXTO:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def calcular() -> dict:
    out = {}
    for grupo, rutas in ARCHIVOS.items():
        out[grupo] = {}
        for r in rutas:
            p = RAIZ / r
            if not p.exists():
                out[grupo][r] = {"sha256": None, "bytes_lf": None}
                continue
            out[grupo][r] = {"sha256": sha256_lf(p), "bytes_lf": len(p.read_bytes().replace(b"\r\n", b"\n"))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    args = ap.parse_args()
    actual = calcular()
    faltan = [r for d in actual.values() for r, v in d.items() if v["sha256"] is None]
    if faltan and not args.verificar:
        raise SystemExit(f"No se pueden escribir los hashes: faltan {faltan}")
    if args.verificar:
        guardado = json.loads(SALIDA.read_text(encoding="utf-8"))
        malos = [(g, r) for g, d in actual.items() for r, v in d.items()
                 if guardado.get("archivos", {}).get(g, {}).get(r, {}).get("sha256") != v["sha256"]]
        for g, r in malos:
            print(f"{'FALTA' if actual[g][r]['sha256'] is None else 'DIFIERE'}: {r}")
        if any(actual[g][r]["sha256"] is None for g, r in malos):
            print("data/processed/*.csv no se versiona: regenerelo con scripts/02_descarga_sisap.py y scripts/03_series.py")
        print("OK: todos los hashes coinciden" if not malos else f"{len(malos)} archivo(s) difieren")
        return 1 if malos else 0
    SALIDA.write_text(json.dumps({
        "descripcion": "SHA-256 para congelar el protocolo del Experimento 2. Lo congela YesaR con su commit; "
                       "este archivo solo prepara la congelacion.",
        "metodo": "texto con CRLF->LF y luego SHA-256 (metodo 2 del articulo); binarios tal cual",
        "fecha_calculo": dt.datetime.now().isoformat(timespec="seconds"),
        "estado": "pendiente de congelar (commit de YesaR)",
        "archivos": actual,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for g, d in actual.items():
        for r, v in d.items():
            print(f"{v['sha256']}  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
