"""Une los pronósticos estadísticos, Chronos y redes de un escenario del brazo D, escribe pronosticos.csv, incidencias.csv,
estados_redes.csv y las tablas de src.analisis.analizar.
Uso: python resultados/brazo_D/scripts/ensamblar_D.py <parte> [<parte> ...]
Partes: principal (h = 1 y 7 en pronosticos.csv; analiza h = 1), sens4, sens5, sens3_h7, sens1_estricto, sens2_bolsa,
        sens6_desde2006, sens7_anomalia_publicada."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estadisticos_lib as L  # noqa: E402
from src import config as C  # noqa: E402
from src.analisis import COLUMNAS, analizar, escribir  # noqa: E402
from src.datos import ESCENARIOS, serie  # noqa: E402
from src.redes import leer_predicciones  # noqa: E402

BASE = C.RAIZ / "resultados" / "brazo_D"
RAIZ_REDES = BASE / "redes_unidades"
TEXTO_REUSO = ("Los pronósticos, incidencias y estados de redes de esta sensibilidad son los de {origen}; "
               "aquí solo están las tablas del análisis ({que}).\n")


def pron_estadisticos(escenario: str, hs) -> pd.DataFrame:
    d = L.SALIDA / escenario / "unidades"
    fs = [f for f in sorted(d.glob("*.csv")) if not f.name.endswith("_rejilla.csv")]
    p = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return p[p.h.isin(hs)]


def pron_redes(nombre_esc: str, esc_dict, hs, N=None, etiqueta=None):
    r = leer_predicciones(RAIZ_REDES)
    if r.empty:
        return pd.DataFrame(columns=COLUMNAS), pd.DataFrame()
    e = esc_dict
    r = r[(r.objetivo == e.objetivo) & (r.inicio == e.inicio) & (r.anomalia == e.anomalia) & (r.particion == "test") & r.h.isin(hs)]
    if N is not None:
        r = r[r.N.isin(N)]
    r = r.assign(escenario=etiqueta or nombre_esc)
    return r[COLUMNAS], r


def estados_redes(esc_dict, hs, N=None) -> pd.DataFrame:
    filas = []
    for f in RAIZ_REDES.rglob("estado.json"):
        e = json.loads(f.read_text(encoding="utf-8"))
        s = e["escenario"]
        if (s["objetivo"], s.get("inicio", C.INICIO), s.get("anomalia", "faltante")) != (esc_dict.objetivo, esc_dict.inicio, esc_dict.anomalia):
            continue
        if e["h"] not in hs or (N is not None and str(e["N"]) not in N):
            continue
        filas.append({"modelo": e["modelo"], "anio": e["anio"], "N": str(e["N"]), "h": e["h"], "semilla": e["semilla"],
                      "n_params": e["n_params"], "n_hist": e["n_hist"], "n_train": e["n_train"], "n_val": e["n_val"],
                      "n_test": e["n_test"], "epocas": e["epocas"], "mejor_epoca": e["mejor_epoca"], "segundos": e["segundos"]})
    return pd.DataFrame(filas).sort_values(["modelo", "h", "N", "anio", "semilla"]).reset_index(drop=True)


def incidencias(escenario: str, hs) -> pd.DataFrame:
    d = L.SALIDA / escenario / "unidades"
    filas = []
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        if not set(hs) & set(j["hs"]):
            continue
        j.pop("hs")
        for h in (1, 7):
            if h not in hs:
                j.pop(f"n_truncados_h{h}", None)
                j.pop(f"n_pronosticos_h{h}", None)
        filas.append(j)
    t = pd.DataFrame(filas)
    return t.sort_values(["modelo", "N", "anio"], key=lambda s: s.astype(str)).reset_index(drop=True)


def ensamblar(carpeta: str, escenario: str, hs, tabla_h: int | None, anios=C.ANIOS_TEST, tipo_boot="iid",
              reusar_redes_N=None, escribir_pron=True):
    """tabla_h: horizonte que se analiza. reusar_redes_N: tamaños N cuyas redes se toman de la principal (sens. 6)."""
    esc = ESCENARIOS[escenario]["D"]
    df = serie(esc)
    est = pron_estadisticos(escenario, hs).assign(escenario=carpeta)
    if reusar_redes_N:
        otros = [str(n) for n in C.TAMANOS["D"] if str(n) not in reusar_redes_N]
        r1, _ = pron_redes(escenario, esc, hs, N=otros, etiqueta=carpeta)
        r2, _ = pron_redes("principal", ESCENARIOS["principal"]["D"], hs, N=reusar_redes_N, etiqueta=carpeta)
        redes = pd.concat([r1, r2], ignore_index=True)
    else:
        redes, _ = pron_redes(escenario, esc, hs, etiqueta=carpeta)
    redes = redes.assign(brazo="D")
    pron = pd.concat([est[COLUMNAS], redes[COLUMNAS]], ignore_index=True)
    pron = pron.sort_values(["h", "N", "modelo", "semilla", "anio", "fecha"], key=lambda s: s.astype(str), kind="stable")
    carp = BASE / carpeta
    carp.mkdir(parents=True, exist_ok=True)
    if escribir_pron:
        pron.to_csv(carp / "pronosticos.csv", index=False, float_format="%.8g")
        incidencias(escenario, hs).to_csv(carp / "incidencias.csv", index=False)
        er = estados_redes(esc, hs, N=reusar_redes_N and None)
        if reusar_redes_N:      # sens. 6: las unidades de N <= 1000 vienen de la principal
            er = pd.concat([er, estados_redes(ESCENARIOS["principal"]["D"], hs, N=reusar_redes_N)], ignore_index=True)
        er.to_csv(carp / "estados_redes.csv", index=False)
        d = er[er.modelo == "DualLSTM"]
        assert (d.n_params == 4673).all(), "n_params de la DualLSTM != 4673"
        print(f"{carpeta}: pronosticos {len(pron)} filas; redes {len(er)} unidades; n_params DualLSTM = {sorted(d.n_params.unique())}")
    if tabla_h is not None:
        t = analizar(pron, df, "D", tabla_h, anios=anios, tipo_boot=tipo_boot)
        escribir(t, carp)
        print(carpeta, t["decision"].to_string())
        print(t["delta"][["N", "ref_estrella", "MAE_DualLSTM", "MAE_ref", "delta", "ic_lo", "ic_hi", "p_DM"]].to_string())
    return pron


def main():
    for parte in sys.argv[1:]:
        if parte == "principal":
            ensamblar("principal", "principal", (1, 7), 1)
        elif parte == "sens4":
            ensamblar_tablas("sens4_pool2023_2025", anios=C.ANIOS_POOL_SENS)
        elif parte == "sens5":
            ensamblar_tablas("sens5_bloques", tipo_boot="bloques")
        elif parte == "sens3_h7":
            ensamblar_tablas("sens3_h7", h=7)
        elif parte in ("sens1_estricto", "sens2_bolsa", "sens7_anomalia_publicada"):
            ensamblar(parte, parte, (1,), 1)
        elif parte == "sens6_desde2006":
            ensamblar(parte, parte, (1,), 1, reusar_redes_N=["120", "360", "1000"])
        else:
            raise SystemExit(f"parte desconocida: {parte}")


def ensamblar_tablas(carpeta, h=1, anios=C.ANIOS_TEST, tipo_boot="iid"):
    """Análisis sobre los pronósticos de la principal (sin copiarlos): sens. 3, 4 y 5."""
    df = serie(ESCENARIOS["principal"]["D"])
    pron = pd.read_csv(BASE / "principal" / "pronosticos.csv", dtype={"N": str})
    t = analizar(pron, df, "D", h, anios=anios, tipo_boot=tipo_boot)
    escribir(t, BASE / carpeta)
    (BASE / carpeta / "LEEME.txt").write_text(TEXTO_REUSO.format(
        origen="resultados/brazo_D/principal/ (pronosticos.csv, incidencias.csv, estados_redes.csv)",
        que=f"h = {h}, años {'-'.join(map(str, anios))}, bootstrap {tipo_boot}"), encoding="utf-8")
    print(carpeta, t["decision"].to_string())
    print(t["delta"][["N", "ref_estrella", "MAE_DualLSTM", "MAE_ref", "delta", "ic_lo", "ic_hi", "p_DM"]].to_string())


if __name__ == "__main__":
    main()
