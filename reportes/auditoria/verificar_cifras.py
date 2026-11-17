"""Auditoría día 3: verificación de las cifras de reportes/brazo_M.md y reportes/brazo_D.md contra los CSV de resultados/.

Dos niveles:
1. Tablas estructuradas (cada tabla se asocia a la ruta CSV citada en las líneas anteriores):
   - MAE [IC] / RMSE / MASE por modelo y N (metricas.csv, anio = pool);
   - MAE por año (metricas.csv);
   - Δ(N) (delta.csv);
   - dispersión entre semillas (semillas.csv);
   - ranking y top-ranked (ranking.csv, top_ranked.csv).
   Cada celda se compara con el valor del CSV redondeado a los decimales mostrados (tolerancia: medio último dígito
   + 1e-9 relativo; el CSV ya está redondeado a 6 cifras significativas).
2. Barrido de todos los números del texto frente a todos los valores numéricos de los CSV del brazo
   (excepto pronosticos.csv): un número sin ningún valor CSV compatible se lista como «sin respaldo directo».
Escribe reportes/auditoria/verificacion_cifras_{M,D}.csv (discrepancias de tablas) y cifras_sin_respaldo_{M,D}.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SAL = Path(__file__).resolve().parent
SEP = "[    ]"
NUM = re.compile(rf"(?<![\w.])-?\d{{1,3}}(?:{SEP}\d{{3}})+(?:[.,]\d+)?(?![\w])|(?<![\w.])-?\d+(?:[.,]\d+)?(?:e-?\d+)?(?![\w])")


def num(s: str):
    s = s.strip()
    s2 = re.sub(SEP, "", s).replace(",", ".")
    try:
        v = float(s2)
    except ValueError:
        return None, None
    m = re.match(r"-?\d+\.(\d+)$", s2)
    dec = len(m.group(1)) if m else 0
    if "e" in s2:
        dec = None
    return v, dec


def tol(v, dec):
    return (0.5 * 10 ** (-dec) if dec is not None else abs(v) * 0.05) + 1e-9 * max(1, abs(v))


def tablas(md: str):
    lineas = md.splitlines()
    out, i = [], 0
    while i < len(lineas):
        if lineas[i].startswith("|") and i + 1 < len(lineas) and re.match(r"^\|[-| ]+\|$", lineas[i + 1]):
            cab = [c.strip() for c in lineas[i].strip("|").split("|")]
            filas, j = [], i + 2
            while j < len(lineas) and lineas[j].startswith("|"):
                filas.append([c.strip() for c in lineas[j].strip("|").split("|")])
                j += 1
            contexto = "\n".join(lineas[max(0, i - 4):i])
            out.append((i + 1, cab, filas, contexto))
            i = j
        else:
            i += 1
    return out


def ruta_csv(contexto, brazo, nombre):
    rs = re.findall(rf"resultados/brazo_{brazo}/([\w<>]+)/{nombre}", contexto)
    return rs[-1] if rs else None


def comparar(disc, brazo, linea, que, mostrado, real):
    v, dec = num(mostrado)
    if v is None:
        disc.append({"brazo": brazo, "linea_md": linea, "celda": que, "mostrado": mostrado, "csv": real, "estado": "no numérico"})
        return
    if real is None or (isinstance(real, float) and np.isnan(real)):
        disc.append({"brazo": brazo, "linea_md": linea, "celda": que, "mostrado": mostrado, "csv": real, "estado": "sin valor en CSV"})
        return
    if abs(v - float(real)) > tol(float(real), dec):
        disc.append({"brazo": brazo, "linea_md": linea, "celda": que, "mostrado": mostrado, "csv": real, "estado": "DISCREPA"})


def verificar_tablas(brazo, md):
    base = RAIZ / "resultados" / f"brazo_{brazo}"
    disc, n_celdas = [], 0
    for linea, cab, filas, ctx in tablas(md):
        cab0 = cab[0].lower()
        if "metricas.csv" in ctx and "ranking.csv" not in ctx and cab0 == "modelo" and cab[1].startswith("N="):
            esc = ruta_csv(ctx, brazo, "metricas.csv")
            if esc is None:            # tablas de RMSE / MASE: «Fuente» en la línea anterior sin carpeta → principal
                esc = "principal"
            met = pd.read_csv(base / esc / "metricas.csv", dtype={"N": str, "anio": str})
            met = met[met.anio == "pool"].set_index(["N", "modelo"])
            col = "RMSE" if "RMSE" in ctx.split("\n")[-1] or "RMSE" in ctx[-200:] else ("MASE" if "MASE" in ctx[-250:] else "MAE")
            for f in filas:
                for k, c in enumerate(cab[1:], 1):
                    N = c.split("=")[1]
                    m = re.match(r"(.+?)\s*\[(.+?);\s*(.+?)\]", f[k])
                    if m:
                        for txt, cc in zip(m.groups(), ("MAE", "MAE_ic_lo", "MAE_ic_hi")):
                            comparar(disc, brazo, linea, f"{esc}/{f[0]}/N={N}/{cc}", txt, met.loc[(N, f[0]), cc]); n_celdas += 1
                    else:
                        comparar(disc, brazo, linea, f"{esc}/{f[0]}/N={N}/{col}", f[k], met.loc[(N, f[0]), col]); n_celdas += 1
        elif "metricas.csv" in ctx and cab0 == "modelo" and cab[1] == "N" and cab[2] == "2021":
            esc = ruta_csv(ctx, brazo, "metricas.csv") or "principal"
            met = pd.read_csv(base / esc / "metricas.csv", dtype={"N": str, "anio": str}).set_index(["N", "modelo", "anio"])
            for f in filas:
                for k, a in enumerate(cab[2:], 2):
                    comparar(disc, brazo, linea, f"{esc}/{f[0]}/N={f[1]}/{a}", f[k], met.loc[(f[1], f[0], a), "MAE"]); n_celdas += 1
        elif "delta.csv" in ctx and cab0 == "n":
            esc = ruta_csv(ctx, brazo, "delta.csv")
            d = pd.read_csv(base / esc / "delta.csv", dtype={"N": str}).set_index("N")
            for f in filas:
                r = d.loc[f[0]]
                if f[1] != r.ref_estrella:
                    disc.append({"brazo": brazo, "linea_md": linea, "celda": f"{esc}/N={f[0]}/ref", "mostrado": f[1],
                                 "csv": r.ref_estrella, "estado": "DISCREPA"})
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/MAE_Dual", f[2], r.MAE_DualLSTM)
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/MAE_ref", f[3], r.MAE_ref)
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/delta", f[4], r.delta)
                lo, hi = re.match(r"\[(.+?);\s*(.+?)\]", f[5]).groups()
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/ic_lo", lo, r.ic_lo)
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/ic_hi", hi, r.ic_hi)
                sig = f[6] == "sí"
                if sig != bool(r.significativa):
                    disc.append({"brazo": brazo, "linea_md": linea, "celda": f"{esc}/N={f[0]}/sig", "mostrado": f[6],
                                 "csv": r.significativa, "estado": "DISCREPA"})
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/HLN", f[7], r.HLN)
                p = f[8]
                if p.startswith("<"):
                    v, _ = num(p[1:])
                    if not r.p_DM < v:
                        disc.append({"brazo": brazo, "linea_md": linea, "celda": f"{esc}/N={f[0]}/p", "mostrado": p,
                                     "csv": r.p_DM, "estado": "DISCREPA"})
                else:
                    comparar(disc, brazo, linea, f"{esc}/N={f[0]}/p", p, r.p_DM)
                comparar(disc, brazo, linea, f"{esc}/N={f[0]}/n", f[9], r.n)
                n_celdas += 12
        elif "semillas.csv" in ctx:
            esc = ruta_csv(ctx, brazo, "semillas.csv") or "principal"
            s = pd.read_csv(base / esc / "semillas.csv", dtype={"N": str}).set_index(["N", "modelo"])
            iN, im = (cab.index("N"), cab.index("modelo") if "modelo" in cab else cab.index("Modelo"))
            cols = {"semillas": "n_semillas", "n_semillas": "n_semillas", "MAE medio": "MAE_media", "MAE_media": "MAE_media",
                    "DE entre semillas": "MAE_de", "MAE_de": "MAE_de", "mín": "MAE_min", "MAE_min": "MAE_min", "máx": "MAE_max",
                    "MAE_max": "MAE_max", "MAE del promedio de semillas": "MAE_promedio_semillas",
                    "MAE_promedio_semillas": "MAE_promedio_semillas"}
            for f in filas:
                for k, c in enumerate(cab):
                    if c in cols:
                        comparar(disc, brazo, linea, f"{esc}/{f[im]}/N={f[iN]}/{c}", f[k], s.loc[(f[iN], f[im]), cols[c]]); n_celdas += 1
        elif "ranking.csv" in ctx and cab0 == "modelo":
            esc = ruta_csv(ctx, brazo, "ranking.csv")
            rk = pd.read_csv(base / esc / "ranking.csv", dtype={"N": str}).set_index(["N", "modelo"])
            tp = pd.read_csv(base / esc / "top_ranked.csv", dtype={"N": str}).set_index(["N", "modelo"])
            for f in filas:
                for k, c in enumerate(cab[1:], 1):
                    N = c.split("=")[1]
                    t = f[k]
                    pos = int(re.search(r"(\d+)", t).group(1))
                    prim = ("primero/empate" in t) or bool(re.match(r"\d+\*", t))
                    umb = int(re.search(r"(\d)(?:/4| ?✔| ?$| TOP)", t.split("/")[-2] if "/4" in t else t.split("/")[-1]).group(1))
                    top = ("TOP" in t) or ("✔" in t)
                    r, q = rk.loc[(N, f[0])], tp.loc[(N, f[0])]
                    for nom, a, b in (("posicion", pos, int(r.posicion)), ("primero", prim, bool(r.primero)),
                                      ("umbrales", umb, int(q.umbrales_primero)), ("top", top, bool(q.consistently_top_ranked))):
                        n_celdas += 1
                        if a != b:
                            disc.append({"brazo": brazo, "linea_md": linea, "celda": f"{esc}/{f[0]}/N={N}/{nom}", "mostrado": t,
                                         "csv": b, "estado": "DISCREPA"})
    return disc, n_celdas


def valores_csv(brazo):
    vals = []
    for f in (RAIZ / "resultados" / f"brazo_{brazo}").rglob("*.csv"):
        if f.name in ("pronosticos.csv", "base_pronosticos.csv") or "redes_unidades" in f.parts or "unidades" in f.parts:
            continue
        t = pd.read_csv(f)
        for c in t.columns:
            v = pd.to_numeric(t[c], errors="coerce").dropna().to_numpy(float)
            vals.append(v)
    # sumas y conteos que suelen citarse (segundos, unidades)
    v = np.unique(np.concatenate(vals))
    return v


def barrido(brazo, md):
    v = valores_csv(brazo)
    filas = []
    for i, l in enumerate(md.splitlines(), 1):
        for m in NUM.finditer(l):
            x, dec = num(m.group(0))
            if x is None:
                continue
            ok = np.any(np.abs(v - x) <= tol(x, dec if dec is not None else 3)) or np.any(np.abs(-v - x) <= tol(x, dec or 0))
            if not ok:
                filas.append({"brazo": brazo, "linea_md": i, "numero": m.group(0), "linea": l.strip()[:160]})
    return pd.DataFrame(filas)


def main():
    res = []
    for brazo in ("M", "D"):
        md = (RAIZ / "reportes" / f"brazo_{brazo}.md").read_text(encoding="utf-8")
        disc, n = verificar_tablas(brazo, md)
        pd.DataFrame(disc, columns=["brazo", "linea_md", "celda", "mostrado", "csv", "estado"]).to_csv(
            SAL / f"verificacion_cifras_{brazo}.csv", index=False)
        b = barrido(brazo, md)
        b.to_csv(SAL / f"cifras_sin_respaldo_{brazo}.csv", index=False)
        res.append({"brazo": brazo, "celdas_de_tabla_comprobadas": n, "discrepancias": len(disc),
                     "numeros_sin_valor_csv_compatible": len(b)})
    pd.DataFrame(res).to_csv(SAL / "verificacion_cifras_resumen.csv", index=False)
    print(pd.DataFrame(res).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
