"""Dia 2 - series procesadas, calidad de datos y diagnostico descriptivo (sin modelos).

Lee los crudos de 02_descarga_sisap.py (data/raw/sisap_dia2, data/raw/gobpe_dia2,
data/raw/gobpe) y la produccion nacional de SISAGRI del repositorio del articulo
(solo lectura; ruta en config `sisagri`). No calcula ningun error de pronostico.

Salidas
  data/processed/sutil_diario.csv              serie diaria (ver DICCIONARIO abajo)
  data/processed/sutil_diario_procedencia.csv  volumen diario por procedencia (formato largo)
  data/processed/sutil_mensual.csv             serie mensual SISAP + agregado del diario; objetivo del
                                               brazo M = vol_total (con dias_faltantes y flag_faltante)
  (data/processed/*.csv no se versiona: falta licencia de redistribucion; se regenera con este script)
  reportes/dia2_*.csv / dia2_diagnostico.json  tablas y cifras del informe
  reportes/figuras_dia2/*.png

Convenciones
  * '...' del SISAP = FALTANTE (NaN), nunca cero. Ver T3 en reportes/INFORME_DIA2.md.
  * vol_total (principal) = vol_bolsa + vol_cajon con '...' en UNA presentacion = 0; NaN si faltan ambas.
    vol_total_estricto (sensibilidad) = NaN si falta alguna. vol_bolsa/vol_cajon: valores originales.
  * Sin imputacion aqui: la regla de imputacion esta en docs/PROTOCOLO.md.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pdfplumber  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sisap_comun import CFG, PROCESSED, RAIZ, RAW, REPORTES, parsear_tabla_sisap, sha256_archivo  # noqa: E402

FIG = REPORTES / "figuras_dia2"
CORTE = pd.Timestamp(CFG["descarga"]["fecha_corte"])
INICIO = pd.Timestamp("1997-01-01")
BOLSA, CAJON = "Limon Sutil Bolsa", "Limon Sutil Cajon"
MESES_ABR = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AGO": 8,
             "SEP": 9, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12}
DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
UMBRAL_MES_INCOMPLETO = 2   # dias sin volumen de bolsa (excepto miercoles) en el mes
UMBRAL_DIAS_FALTANTES = 0.20    # brazo M: mes faltante si mas de esta fraccion de dias es faltante
UMBRAL_DIAS_OBSERVADOS = 0.80   # sensibilidad (tratamiento anterior): fraccion minima de dias observados
PRUEBA = ("2021-01-01", "2025-12-31")

# Paleta (dataviz, validada: categorica azul/naranja/aqua; secuencial azul)
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, SUPERFICIE, GRILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _var(nombre: str) -> str:
    """Normaliza el nombre de variable del SISAP (los acentos vienen en ISO-8859-1)."""
    n = nombre.lower()
    if n.startswith("volumen"):
        return "vol"
    if "prom" in n:
        return "precio_prom"
    if "nimo" in n:
        return "precio_min"
    if "ximo" in n:
        return "precio_max"
    raise ValueError(nombre)


# ------------------------------------------------------------------ lectura
def leer_diario() -> pd.DataFrame:
    filas = []
    archivos = sorted(glob.glob(str(RAW / "sisap_dia2" / "diario" / "diario_*.html")))
    if not archivos:
        raise SystemExit("No hay crudos diarios: ejecute antes scripts/02_descarga_sisap.py")
    for f in archivos:
        largo, _ = parsear_tabla_sisap(Path(f).read_bytes().decode("latin-1"))
        filas += largo
    d = pd.DataFrame(filas)
    d["fecha"] = pd.to_datetime(d["periodo"], format="%d/%m/%Y")
    d["var"] = d["variable"].map(_var)
    d = d[d["fecha"] <= CORTE]
    dup = d.duplicated(["fecha", "producto", "var", "procedencia"]).sum()
    if dup:
        raise ValueError(f"{dup} filas duplicadas en el diario")
    return d


def leer_mensual() -> pd.DataFrame:
    f = sorted(glob.glob(str(RAW / "sisap_dia2" / "mensual" / "mensual_*.html")))[-1]
    largo, _ = parsear_tabla_sisap(Path(f).read_bytes().decode("latin-1"))
    m = pd.DataFrame(largo)
    per = m["periodo"].str.extract(r"([A-Z]{3})-(\d{4})")
    m["anio"] = per[1].astype(int)
    m["mes"] = per[0].map(MESES_ABR)
    m["var"] = m["variable"].map(_var)
    return m


def construir_diario(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tot = d[(d["procedencia"] == "Total") | (d["var"] != "vol")]
    ancho = tot.pivot_table(index="fecha", columns=["producto", "var"], values="valor", aggfunc="first", dropna=False)
    calendario = pd.date_range(INICIO, CORTE, freq="D")
    en_tabla = pd.Series(calendario.isin(ancho.index), index=calendario)
    ancho = ancho.reindex(calendario)

    def col(p, v):
        return ancho[(p, v)] if (p, v) in ancho.columns else pd.Series(np.nan, index=calendario)

    s = pd.DataFrame(index=calendario)
    s.index.name = "fecha"
    s["vol_bolsa"] = col(BOLSA, "vol")
    s["vol_cajon"] = col(CAJON, "vol")
    # vol_bolsa / vol_cajon: valores ORIGINALES del SISAP ('...' = NaN).
    # Supuesto principal (protocolo 2.1): '...' en una sola presentacion = ingreso cero de esa presentacion;
    # si ambas tienen '...' (o la fecha no esta en la tabla), el dia es faltante.
    ambas = s["vol_bolsa"].isna() & s["vol_cajon"].isna()
    for p in ("bolsa", "cajon"):
        o = s[f"vol_{p}"]
        s[f"estado_{p}"] = np.where(o.notna(), "observado", np.where(ambas, "faltante", "cero_supuesto"))
        s[f"vol_{p}_regla"] = o.where(o.notna() | ambas, 0.0)
    s["vol_total"] = s["vol_bolsa_regla"] + s["vol_cajon_regla"]       # objetivo principal; NaN si faltan ambas
    s["vol_total_estricto"] = s["vol_bolsa"] + s["vol_cajon"]          # tratamiento anterior: NaN si falta alguna
    # Precio de la serie secundaria: presentacion bolsa (dominante desde ~2008 y casi sin huecos).
    s["precio_prom"] = col(BOLSA, "precio_prom")
    s["precio_min"] = col(BOLSA, "precio_min")
    s["precio_max"] = col(BOLSA, "precio_max")
    s["precio_prom_cajon"] = col(CAJON, "precio_prom")
    s["precio_min_cajon"] = col(CAJON, "precio_min")
    s["precio_max_cajon"] = col(CAJON, "precio_max")
    s["dia_en_tabla"] = en_tabla.astype(int)                  # 0 = la fecha no aparece en la respuesta
    s["falta_vol_bolsa"] = s["vol_bolsa"].isna().astype(int)
    s["falta_vol_cajon"] = s["vol_cajon"].isna().astype(int)
    s["falta_vol_total"] = s["vol_total"].isna().astype(int)             # ambas presentaciones en '...'
    s["falta_vol_total_estricto"] = s["vol_total_estricto"].isna().astype(int)
    s["falta_precio"] = s["precio_prom"].isna().astype(int)
    s["dia_semana"] = s.index.day_of_week  # 0 = lunes
    # Anomalias con evidencia documental (config): se marcan, no se corrigen.
    s["anomalia_documentada"] = 0
    for an in CFG.get("anomalias_documentadas", []):
        f = pd.Timestamp(an["fecha"])
        if f in s.index:
            s.loc[f, "anomalia_documentada"] = 1

    proc = d[(d["var"] == "vol") & (d["procedencia"] != "Total") & d["valor"].notna()]
    proc = proc.assign(presentacion=proc["producto"].map({BOLSA: "bolsa", CAJON: "cajon"}))[
        ["fecha", "presentacion", "procedencia", "valor"]].rename(columns={"valor": "vol_t"})
    return s, proc.sort_values(["fecha", "presentacion", "procedencia"])


def construir_mensual(m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    tot = m[(m["procedencia"] == "Total") | (m["var"] != "vol")]
    ancho = tot.pivot_table(index=["anio", "mes"], columns=["producto", "var"], values="valor", aggfunc="first",
                            dropna=False)
    idx = pd.MultiIndex.from_tuples([(t.year, t.month) for t in pd.period_range("1997-01", CORTE, freq="M")],
                                    names=["anio", "mes"])
    ancho = ancho.reindex(idx)
    out = pd.DataFrame(index=idx)
    out["vol_bolsa"] = ancho[(BOLSA, "vol")]
    out["vol_cajon"] = ancho[(CAJON, "vol")]
    out["vol_total_sisap"] = out["vol_bolsa"] + out["vol_cajon"]      # suma publicada por el SISAP (referencia)
    out["precio_prom"] = ancho[(BOLSA, "precio_prom")]
    out["precio_prom_cajon"] = ancho[(CAJON, "precio_prom")]
    g = s.groupby([s.index.year, s.index.month])
    out["vol_bolsa_suma_diario"] = g["vol_bolsa"].sum(min_count=1).values
    out["vol_cajon_suma_diario"] = g["vol_cajon"].sum(min_count=1).values
    out["dias_mes"] = g.size().values
    out["dias_con_vol_bolsa"] = g["vol_bolsa"].count().values
    out["dias_con_vol_cajon"] = g["vol_cajon"].count().values
    out["dias_con_vol_total_estricto"] = g["vol_total_estricto"].count().values
    out["dias_fuera_de_tabla"] = (1 - g["dia_en_tabla"].mean()).mul(g.size()).round().astype(int).values
    # Mes incompleto (definicion del dia 2, solo descriptiva): falta el volumen de bolsa en mas de 2 dias
    # que no son miercoles.
    falta_no_mie = (s["falta_vol_bolsa"] * (s["dia_semana"] != 2)).groupby([s.index.year, s.index.month]).sum()
    out["dias_sin_bolsa_no_miercoles"] = falta_no_mie.values
    out["mes_incompleto"] = (out["dias_sin_bolsa_no_miercoles"] > UMBRAL_MES_INCOMPLETO).astype(int)
    ultimo_mes = (CORTE + pd.Timedelta(days=1)).month != CORTE.month
    out["mes_completo"] = 1
    if not ultimo_mes:
        out.loc[(CORTE.year, CORTE.month), "mes_completo"] = 0
    out["falta_vol_total_sisap"] = out["vol_total_sisap"].isna().astype(int)
    dias_cal = np.array([pd.Period(year=a, month=mm, freq="M").days_in_month for a, mm in idx])
    out["dias_del_mes"] = dias_cal
    # Objetivo PRINCIPAL del brazo M (protocolo 2.1): suma de vol_total diario con la regla
    # ('...' en una presentacion = 0). Los dias faltantes (ambas en '...', o posteriores al corte) no
    # suman. Mes faltante si mas del 20 % de sus dias son faltantes, o si contiene una anomalia documentada.
    out["vol_total"] = g["vol_total"].sum(min_count=1).values
    out["dias_faltantes"] = g["falta_vol_total"].sum().values + (dias_cal - out["dias_mes"].values)
    out["mes_con_anomalia"] = g["anomalia_documentada"].max().values
    out["flag_faltante"] = ((out["dias_faltantes"] > UMBRAL_DIAS_FALTANTES * dias_cal)
                            | (out["mes_con_anomalia"] == 1)).astype(int)
    # SENSIBILIDAD (tratamiento anterior, '...' = faltante): promedio diario de vol_total_estricto en los
    # dias observados x dias del mes; la anomalia no cuenta como observada; faltante con < 80 % observados.
    vt = s["vol_total_estricto"].where(s["anomalia_documentada"] == 0)
    gv = vt.groupby([s.index.year, s.index.month])
    out["dias_observados_estricto"] = gv.count().values
    out["vol_total_escalado"] = gv.mean().values * dias_cal
    out["flag_faltante_estricto"] = (out["dias_observados_estricto"] < UMBRAL_DIAS_OBSERVADOS * dias_cal).astype(int)
    return out.reset_index()


# ------------------------------------------------------------------ T3: '...'
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
            "septiembre", "octubre", "noviembre", "diciembre"]


def control_puntos_suspensivos(s: pd.DataFrame) -> pd.DataFrame:
    """Compara los promedios del PDF (7 dias y 4 mismos dias de la semana) con el SISAP
    bajo dos lecturas de '...': faltante (promedio indefinido) o cero."""
    pdfs = sorted(glob.glob(str(RAW / "gobpe_dia2" / "pdf_gmml_*.pdf"))) + \
        sorted(glob.glob(str(RAW / "gobpe" / "pdf_gmml_*.pdf")))
    filas = []
    for f in pdfs:
        with pdfplumber.open(f) as pdf:
            texto = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
        m = re.search(r"(\d{1,2}) de (\w+) de (\d{4})", texto.splitlines()[0])
        fecha = pd.Timestamp(int(m.group(3)), MESES_ES.index(m.group(2).lower().replace("setiembre", "septiembre")) + 1,
                             int(m.group(1)))
        for linea in texto.splitlines():
            mm = re.match(r"Limon Sutil (Bolsa|Cajon) (\S+) (\S+) (\S+) (\S+) ", linea)
            if not mm:
                continue
            pres = mm.group(1).lower()
            serie = s[f"vol_{pres}"]
            valores = dict(zip(["ayer", "hoy", "prom7", "prom4"], mm.group(2, 3, 4, 5)))
            ventanas = {
                "ayer": [fecha - pd.Timedelta(days=1)],
                "hoy": [fecha],
                "prom7": list(pd.date_range(end=fecha - pd.Timedelta(days=1), periods=7)),
                "prom4": [fecha - pd.Timedelta(weeks=k) for k in range(1, 5)],
            }
            for campo, idx in ventanas.items():
                x = serie.reindex(idx)
                faltante = round(float(x.mean()), 1) if x.notna().all() else None
                cero = round(float(x.fillna(0).mean()), 1)
                pdf_v = None if valores[campo] == ":" else float(valores[campo])
                filas.append({
                    "fecha_pdf": fecha.date().isoformat(), "presentacion": pres, "campo": campo,
                    "ventana": f"{idx[0].date()}..{idx[-1].date()}", "n_puntos_suspensivos_en_ventana": int(x.isna().sum()),
                    "valor_pdf": "" if pdf_v is None else pdf_v,
                    "sisap_si_faltante": "" if faltante is None else faltante, "sisap_si_cero": cero,
                    "coincide_faltante": (pdf_v is None and faltante is None) or (
                        pdf_v is not None and faltante is not None and abs(pdf_v - faltante) <= 0.5),
                    "coincide_cero": pdf_v is not None and abs(pdf_v - cero) <= 0.5,
                    "archivo": Path(f).relative_to(RAIZ).as_posix(),
                })
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ T3: ruptura
def ruptura(s: pd.DataFrame, mens: pd.DataFrame) -> dict:
    obs = pd.concat([s["vol_bolsa"], s["vol_cajon"]]).dropna()
    frac = (obs % 1).abs() > 1e-9
    por_mes = frac.groupby([obs.index.year, obs.index.month]).mean()
    con_dec = por_mes[por_mes > 0]
    ultimo_dec = con_dec.index.max()
    primer_entero = None
    for (a, mth), v in por_mes.items():
        if (a, mth) > ultimo_dec and v == 0:
            primer_entero = (a, mth)
            break
    corte_r = pd.Timestamp(primer_entero[0], primer_entero[1], 1)
    res = {"ultimo_mes_con_decimales": f"{ultimo_dec[0]}-{ultimo_dec[1]:02d}",
           "primer_mes_todo_entero": corte_r.strftime("%Y-%m"),
           "fraccion_valores_con_decimales_por_anio": {
               int(a): round(float(v), 3) for a, v in frac.groupby(obs.index.year).mean().items() if a <= 2008},
           "parte_decimal_media_t_antes": round(float((obs[obs.index < corte_r] % 1).mean()), 3)}
    # Niveles y varianza: 3 anios antes vs 3 anios despues
    a0, a1 = corte_r - pd.DateOffset(years=3), corte_r + pd.DateOffset(years=3)
    comp = {}
    for nombre, sub in {"antes": s.loc[a0:corte_r - pd.Timedelta(days=1)], "despues": s.loc[corte_r:a1 - pd.Timedelta(days=1)]}.items():
        v = sub["vol_total_estricto"].dropna()
        lv = np.log(v[v > 0])
        comp[nombre] = {"periodo": f"{sub.index.min().date()}..{sub.index.max().date()}",
                        "n_dias_vol_total": int(v.size), "media_t": round(float(v.mean()), 1),
                        "mediana_t": round(float(v.median()), 1), "desv_t": round(float(v.std()), 1),
                        "cv": round(float(v.std() / v.mean()), 3), "desv_log": round(float(lv.std()), 3)}
    res["diario_3_anios"] = comp
    lev = stats.levene(s.loc[a0:corte_r - pd.Timedelta(days=1), "vol_total_estricto"].dropna(),
                       s.loc[corte_r:a1 - pd.Timedelta(days=1), "vol_total_estricto"].dropna(), center="median")
    res["levene_brown_forsythe_diario"] = {"estadistico": round(float(lev.statistic), 3), "p": float(lev.pvalue)}
    # Escalon de nivel en el log del volumen mensual: tendencia + meses + escalon, HAC(12)
    mm = mens[(mens["anio"] >= corte_r.year - 5) & (mens["anio"] <= corte_r.year + 4)].dropna(subset=["vol_total_sisap"]).copy()
    mm["t"] = (mm["anio"] - mm["anio"].min()) * 12 + mm["mes"]
    mm["escalon"] = ((mm["anio"] * 100 + mm["mes"]) >= corte_r.year * 100 + corte_r.month).astype(float)
    X = pd.get_dummies(mm["mes"].astype(int), prefix="m", drop_first=True).astype(float)
    X["t"] = mm["t"].astype(float)
    X["escalon"] = mm["escalon"]
    X = sm.add_constant(X)
    fit = sm.OLS(np.log(mm["vol_total_sisap"].astype(float)), X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    b, (lo, hi) = fit.params["escalon"], fit.conf_int().loc["escalon"]
    res["escalon_log_mensual"] = {
        "muestra": f"{mm['anio'].min()}-{mm['mes'].iloc[0]:02d}..{mm['anio'].max()}-{mm['mes'].iloc[-1]:02d}",
        "n_meses": int(len(mm)), "coef": round(float(b), 4), "ic95": [round(float(lo), 4), round(float(hi), 4)],
        "efecto_pct": round(float(np.expm1(b) * 100), 1), "p": float(fit.pvalues["escalon"]),
        "especificacion": "log(vol_total mensual) ~ const + 11 dummies de mes + tendencia lineal + escalon; errores HAC (Newey-West, 12 rezagos)"}
    # Varianza de las variaciones mensuales (log) antes / despues
    lm = np.log(mens.set_index(pd.to_datetime(dict(year=mens.anio, month=mens.mes, day=1)))["vol_total_sisap"]).diff()
    res["desv_dlog_mensual"] = {
        "antes": round(float(lm[a0:corte_r - pd.Timedelta(days=1)].std()), 3),
        "despues": round(float(lm[corte_r:a1 - pd.Timedelta(days=1)].std()), 3)}
    return res


# ------------------------------------------------------------------ excluidos vs evaluados (brazo D)
def excluidos_vs_evaluados(s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dias de prueba 2021-2025: evaluados (vol_total observado y sin anomalia) frente a excluidos.
    Descriptivo; la regla que usa el resultado esta en docs/PROTOCOLO.md (2.3)."""
    p = s.loc[PRUEBA[0]:PRUEBA[1]].copy()
    p["grupo"] = np.where(p["vol_total_estricto"].notna() & (p["anomalia_documentada"] == 0), "evaluado", "excluido")
    filas = []
    for g, sub in p.groupby("grupo"):
        n = len(sub)
        for k in range(7):
            c = int((sub["dia_semana"] == k).sum())
            filas.append({"grupo": g, "dimension": "dia_semana", "categoria": DIAS[k], "n": c,
                          "pct": round(100 * c / n, 1)})
        for k in range(1, 13):
            c = int((sub.index.month == k).sum())
            filas.append({"grupo": g, "dimension": "mes", "categoria": k, "n": c, "pct": round(100 * c / n, 1)})
        vb = sub["vol_bolsa"].dropna()
        filas.append({"grupo": g, "dimension": "vol_bolsa", "categoria": "n_dias_con_bolsa", "n": int(vb.size), "pct": None})
        for nombre, v in [("media", vb.mean()), ("p10", vb.quantile(.10)), ("p25", vb.quantile(.25)),
                          ("mediana", vb.median()), ("p75", vb.quantile(.75)), ("p90", vb.quantile(.90))]:
            filas.append({"grupo": g, "dimension": "vol_bolsa", "categoria": nombre, "n": round(float(v), 1), "pct": None})
        for k in range(7):
            v = sub.loc[sub["dia_semana"] == k, "vol_bolsa"].median()
            filas.append({"grupo": g, "dimension": "vol_bolsa_mediana_por_dia", "categoria": DIAS[k],
                          "n": None if pd.isna(v) else round(float(v), 1), "pct": None})
    tabla = pd.DataFrame(filas)
    ev = p.loc[p["grupo"] == "evaluado", "vol_bolsa"].dropna()
    ex = p.loc[p["grupo"] == "excluido", "vol_bolsa"].dropna()
    mw = stats.mannwhitneyu(ex, ev, alternative="two-sided")
    no_mie = p["dia_semana"] != 2
    mw2 = stats.mannwhitneyu(p.loc[(p["grupo"] == "excluido") & no_mie, "vol_bolsa"].dropna(),
                             p.loc[(p["grupo"] == "evaluado") & no_mie, "vol_bolsa"].dropna(), alternative="two-sided")
    res = {
        "periodo": f"{PRUEBA[0]}..{PRUEBA[1]}",
        "n_evaluados": int((p["grupo"] == "evaluado").sum()), "n_excluidos": int((p["grupo"] == "excluido").sum()),
        "n_excluidos_con_bolsa": int(ex.size),
        "excluidos_por_anio": {int(k): int(v) for k, v in (p["grupo"] == "excluido").groupby(p.index.year).sum().items()},
        "mediana_vol_bolsa_evaluados": float(ev.median()), "mediana_vol_bolsa_excluidos": float(ex.median()),
        "mann_whitney_vol_bolsa": {"U": float(mw.statistic), "p": float(mw.pvalue), "alternativa": "bilateral"},
        "regla_bolsa_sensibilidad_principal": bool(mw.pvalue < 0.05),
        "complemento_sin_miercoles_no_aplica_a_la_regla": {
            "U": float(mw2.statistic), "p": float(mw2.pvalue),
            "mediana_evaluados": float(p.loc[(p["grupo"] == "evaluado") & no_mie, "vol_bolsa"].median()),
            "mediana_excluidos": float(p.loc[(p["grupo"] == "excluido") & no_mie, "vol_bolsa"].median())},
    }
    return tabla, res


def mensual_vs_sisap(mens: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compara el objetivo mensual principal (vol_total, regla '...' en una presentacion = 0) y la
    sensibilidad (vol_total_escalado) con la suma publicada por el SISAP. Descriptivo."""
    m = mens[mens["mes_completo"] == 1].copy()
    m["dif_t"] = m["vol_total"] - m["vol_total_sisap"]
    m["dif_pct"] = 100 * m["dif_t"] / m["vol_total_sisap"]
    m["dif_gt_5pct"] = (m["dif_pct"].abs() > 5).astype(int)
    m["dif_escalado_pct"] = 100 * (m["vol_total_escalado"] - m["vol_total_sisap"]) / m["vol_total_sisap"]
    cols = ["anio", "mes", "dias_del_mes", "dias_faltantes", "mes_con_anomalia", "flag_faltante", "vol_total_sisap",
            "vol_total", "dif_t", "dif_pct", "dif_gt_5pct", "dias_observados_estricto", "flag_faltante_estricto",
            "vol_total_escalado", "dif_escalado_pct"]
    t = m[cols]
    ok = t.dropna(subset=["dif_t"])
    rango = t[(t.anio * 100 + t.mes >= 199710) & (t.anio <= 2025)]
    res = {
        "rango_brazo_m": "1997-10..2025-12", "n_meses_rango": int(len(rango)),
        "n_faltantes_rango": int(rango["flag_faltante"].sum()),
        "faltantes_por_anio": {int(k): int(v) for k, v in rango.groupby("anio")["flag_faltante"].sum().items() if v},
        "meses_faltantes": [f"{a}-{mm:02d} ({d} d faltantes{', anomalia' if an else ''})" for a, mm, f, d, an in
                            zip(rango.anio, rango.mes, rango.flag_faltante, rango.dias_faltantes, rango.mes_con_anomalia) if f],
        "n_meses_comparables": int(len(ok)),
        "dif_abs_max_t": round(float(ok["dif_t"].abs().max()), 2),
        "n_dif_gt_0_5t": int((ok["dif_t"].abs() > 0.5).sum()),
        "meses_dif_gt_0_5t": {f"{a}-{mm:02d}": round(float(d), 2) for a, mm, d in zip(ok.anio, ok.mes, ok.dif_t) if abs(d) > 0.5},
        "n_dif_gt_5pct": int(ok["dif_gt_5pct"].sum()),
        "faltantes_estricto_rango": int(rango["flag_faltante_estricto"].sum()),
    }
    return t, res


def feriados_peru(anios) -> dict:
    """Feriados nacionales de larga data (lista fija; no incluye los agregados desde 2022 ni los dias no
    laborables decretados). Solo para describir en que fechas caen los dias faltantes."""
    from dateutil.easter import easter
    fijos = {(1, 1): "Año Nuevo", (5, 1): "Día del Trabajo", (6, 29): "San Pedro y San Pablo",
             (7, 28): "Fiestas Patrias", (7, 29): "Fiestas Patrias", (8, 30): "Santa Rosa de Lima",
             (10, 8): "Combate de Angamos", (11, 1): "Todos los Santos", (12, 8): "Inmaculada Concepción",
             (12, 25): "Navidad"}
    out = {}
    for a in anios:
        for (mm, dd), n in fijos.items():
            out[pd.Timestamp(a, mm, dd)] = n
        pascua = pd.Timestamp(easter(a))
        out[pascua - pd.Timedelta(days=3)] = "Jueves Santo"
        out[pascua - pd.Timedelta(days=2)] = "Viernes Santo"
    return out


def dias_ambas_faltantes(s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f = s[s["falta_vol_total"] == 1]
    fer = feriados_peru(range(INICIO.year, CORTE.year + 1))
    filas = []
    for d, r in f.iterrows():
        cerca = [f"{n} ({(d - k).days:+d} d)" for k, n in fer.items() if abs((d - k).days) == 1]
        filas.append({"fecha": d.date().isoformat(), "dia": DIAS[d.day_of_week], "en_tabla": int(r["dia_en_tabla"]),
                      "feriado": fer.get(d, ""), "junto_a_feriado": "; ".join(cerca)})
    tab = pd.DataFrame(filas)
    y = pd.to_datetime(tab["fecha"]).dt.year
    prueba = tab[(y >= 2021) & (y <= 2025)]
    res = {"n_total": int(len(tab)), "por_anio": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
           "n_fuera_de_tabla": int((tab["en_tabla"] == 0).sum()),
           "n_feriado": int((tab["feriado"] != "").sum()),
           "n_junto_a_feriado": int(((tab["feriado"] == "") & (tab["junto_a_feriado"] != "")).sum()),
           "por_dia_semana": {d: int((tab["dia"] == d).sum()) for d in DIAS},
           "prueba_2021_2025": prueba.to_dict(orient="records")}
    return tab, res


def evaluables_por_anio(s: pd.DataFrame, mens: pd.DataFrame) -> pd.DataFrame:
    p = s.loc[PRUEBA[0]:PRUEBA[1]]
    sin_an = p["anomalia_documentada"] == 0
    y = p.index.year
    m = mens[(mens.anio >= 2021) & (mens.anio <= 2025)].set_index("anio")
    t = pd.DataFrame({
        "dias": p.groupby(y).size(),
        "D_dias_evaluables": (p["vol_total"].notna() & sin_an).groupby(y).sum(),
        "D_dias_cero_supuesto_bolsa": (p["estado_bolsa"] == "cero_supuesto").groupby(y).sum(),
        "D_dias_cero_supuesto_cajon": (p["estado_cajon"] == "cero_supuesto").groupby(y).sum(),
        "D_dias_evaluables_estricto": (p["vol_total_estricto"].notna() & sin_an).groupby(y).sum(),
        "D_dias_evaluables_bolsa": p["vol_bolsa"].notna().groupby(y).sum(),
        "M_meses_evaluables": (1 - m["flag_faltante"]).groupby(level=0).sum(),
        "M_meses_evaluables_estricto": (1 - m["flag_faltante_estricto"]).groupby(level=0).sum(),
    })
    t.index.name = "anio"
    t.loc["total"] = t.sum()
    return t.astype(int)


# ------------------------------------------------------------------ figuras
def _estilo(ax, titulo, ylabel=None):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", fontsize=11, color=TINTA)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRILLA)
    ax.tick_params(colors=TINTA2, labelsize=8)
    ax.grid(axis="y", color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=TINTA2, fontsize=9)


def _guardar(fig, nombre):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(FIG / nombre, dpi=150)
    plt.close(fig)


def figuras(s, mens, semanal, sisagri_cmp, rup):
    m = mens[mens["mes_completo"] == 1].copy()
    m["f"] = pd.to_datetime(dict(year=m.anio, month=m.mes, day=1))
    # 1. Mensual por presentacion
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(m["f"], m["vol_bolsa"] / 1000, color=AZUL, lw=1.5, label="Bolsa")
    ax.plot(m["f"], m["vol_cajon"] / 1000, color=NARANJA, lw=1.5, label="Cajón")
    ax.axvline(pd.Timestamp(rup["primer_mes_todo_entero"]), color=TINTA2, lw=0.8, ls=":")
    ax.text(pd.Timestamp(rup["primer_mes_todo_entero"]), ax.get_ylim()[1] * 0.95, " volúmenes enteros", fontsize=7, color=TINTA2)
    ax.text(m["f"].iloc[-1], m["vol_bolsa"].iloc[-1] / 1000, " Bolsa", color=TINTA, fontsize=8, va="center")
    ax.text(m["f"].iloc[-1], m["vol_cajon"].iloc[-1] / 1000, " Cajón", color=TINTA, fontsize=8, va="center")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _estilo(ax, "Ingreso mensual de limón sutil al GMML por presentación (SISAP)", "miles de t por mes")
    _guardar(fig, "fig1_mensual_presentacion.png")

    # 2. Mapa de calor: mediana por dia de la semana y anio (vol_total)
    piv = semanal.pivot(index="dia", columns="anio", values="mediana_vol_total").reindex(DIAS)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
    ax.set_yticks(range(7), DIAS)
    ax.set_xticks(range(0, piv.shape[1], 2), piv.columns[::2], rotation=90)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("mediana t/día", fontsize=8, color=TINTA2)
    cb.ax.tick_params(labelsize=7, colors=TINTA2)
    _estilo(ax, "Mediana diaria de vol_total (bolsa + cajón) por día de la semana y año")
    ax.grid(False)
    _guardar(fig, "fig2_semanal_mapa_calor.png")

    # 3. Razon miercoles / resto por anio
    r = semanal.pivot(index="anio", columns="dia", values="mediana_vol_bolsa")
    razon = r["mie"] / r[[d for d in DIAS if d != "mie"]].median(axis=1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(razon.index, razon.values, color=AZUL, lw=1.5, marker="o", ms=4)
    _estilo(ax, "Mediana del miércoles / mediana de los otros días (volumen de bolsa)", "razón")
    _guardar(fig, "fig3_efecto_miercoles.png")

    # 4. Perfil estacional por periodo
    fig, ax = plt.subplots(figsize=(7, 3.4))
    periodos = [(1998, 2005, AZUL), (2006, 2015, NARANJA), (2016, 2025, AQUA)]
    for a0, a1, c in periodos:
        sub = m[(m.anio >= a0) & (m.anio <= a1)].copy()
        sub["ind"] = sub["vol_total_sisap"] / sub.groupby("anio")["vol_total_sisap"].transform("mean")
        perfil = sub.groupby("mes")["ind"].median()
        ax.plot(perfil.index, perfil.values, color=c, lw=1.8, marker="o", ms=4, label=f"{a0}–{a1}")
    ax.axhline(1, color=TINTA2, lw=0.6)
    ax.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.legend(frameon=False, fontsize=8)
    _estilo(ax, "Perfil estacional: vol_total mensual / media del año (mediana)", "índice")
    _guardar(fig, "fig4_perfil_estacional.png")

    # 5. Totales anuales (tendencia)
    an = m[m.anio <= CORTE.year - 1].groupby("anio")[["vol_bolsa", "vol_cajon"]].sum() / 1000
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(an.index, an["vol_bolsa"], color=AZUL, width=0.8, label="Bolsa", edgecolor=SUPERFICIE, linewidth=1)
    ax.bar(an.index, an["vol_cajon"], bottom=an["vol_bolsa"], color=NARANJA, width=0.8, label="Cajón",
           edgecolor=SUPERFICIE, linewidth=1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _estilo(ax, "Ingreso anual de limón sutil al GMML (años completos)", "miles de t")
    _guardar(fig, "fig5_anual.png")

    # 6. GMML vs SISAGRI, indexados (sin doble eje)
    if sisagri_cmp is not None:
        c = sisagri_cmp.copy()
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(c["f"], 100 * c["gmml"] / c["gmml"].mean(), color=AZUL, lw=1.5, label="GMML (SISAP), vol_total")
        ax.plot(c["f"], 100 * c["sisagri"] / c["sisagri"].mean(), color=NARANJA, lw=1.5, label="Producción nacional (SISAGRI)")
        inc = c[c["mes_incompleto"] == 1]
        ax.scatter(inc["f"], 100 * inc["gmml"] / c["gmml"].mean(), s=40, facecolors="none", edgecolors=TINTA, lw=1,
                   zorder=3, label="GMML: mes incompleto")
        ax.axhline(100, color=TINTA2, lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _estilo(ax, "Mensual 2016–2025, índice (media del periodo = 100)", "índice")
        _guardar(fig, "fig6_gmml_vs_sisagri.png")


# ------------------------------------------------------------------ main
def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d = leer_diario()
    s, proc = construir_diario(d)
    mens = construir_mensual(leer_mensual(), s)

    s_out = s.copy()
    s_out.index = s_out.index.strftime("%Y-%m-%d")
    s_out.to_csv(PROCESSED / "sutil_diario.csv", float_format="%.2f", lineterminator="\n")
    proc.assign(fecha=proc["fecha"].dt.strftime("%Y-%m-%d")).to_csv(
        PROCESSED / "sutil_diario_procedencia.csv", index=False, float_format="%.2f", lineterminator="\n")
    mens.to_csv(PROCESSED / "sutil_mensual.csv", index=False, float_format="%.2f", lineterminator="\n")

    diag = {"corte": CORTE.date().isoformat()}
    # Inicio de los datos
    diag["primer_dia_en_tabla"] = s.index[s["dia_en_tabla"] == 1].min().date().isoformat()
    diag["primer_dia_vol_bolsa"] = s["vol_bolsa"].first_valid_index().date().isoformat()
    diag["primer_dia_vol_cajon"] = s["vol_cajon"].first_valid_index().date().isoformat()
    diag["primer_dia_vol_total"] = s["vol_total_estricto"].first_valid_index().date().isoformat()
    diag["primer_dia_precio_bolsa"] = s["precio_prom"].first_valid_index().date().isoformat()
    diag["primer_dia_precio_cajon"] = s["precio_prom_cajon"].first_valid_index().date().isoformat()
    diag["volumen_cero_en_sisap"] = int(((d["var"] == "vol") & (d["valor"] == 0)).sum())
    diag["volumen_minimo_observado_t"] = float(d.loc[(d["var"] == "vol") & (d["procedencia"] == "Total"), "valor"].min())
    tot = d[(d["var"] == "vol") & (d["procedencia"] == "Total")].set_index(["fecha", "producto"])["valor"].dropna()
    suma = d[(d["var"] == "vol") & (d["procedencia"] != "Total")].groupby(["fecha", "producto"])["valor"].sum(min_count=1)
    cmpt = pd.concat([tot, suma], axis=1, keys=["t", "s"]).dropna()
    diag["total_vs_suma_procedencias"] = {"n": int(len(cmpt)), "dif_max_t": round(float((cmpt.t - cmpt.s).abs().max()), 3),
                                         "n_dif_mayor_0_011": int(((cmpt.t - cmpt.s).abs() > 0.011).sum())}

    # Huecos por anio
    y = s.index.year
    huecos = pd.DataFrame({
        "dias": s.groupby(y).size(),
        "dias_fuera_de_tabla": (1 - s["dia_en_tabla"]).groupby(y).sum(),
        "falta_vol_bolsa": s["falta_vol_bolsa"].groupby(y).sum(),
        "falta_vol_cajon": s["falta_vol_cajon"].groupby(y).sum(),
        "falta_vol_total": s["falta_vol_total_estricto"].groupby(y).sum(),
        "falta_precio_bolsa": s["falta_precio"].groupby(y).sum(),
        "falta_vol_total_miercoles": (s["falta_vol_total_estricto"] * (s["dia_semana"] == 2)).groupby(y).sum(),
        "faltan_ambas_presentaciones": s["falta_vol_total"].groupby(y).sum(),
        "participacion_cajon_pct": (100 * s["vol_cajon"].groupby(y).sum()
                                    / (s["vol_bolsa"].groupby(y).sum() + s["vol_cajon"].groupby(y).sum())).round(1),
    })
    huecos.index.name = "anio"
    huecos.to_csv(REPORTES / "dia2_huecos_por_anio.csv", lineterminator="\n")
    # rachas de dias fuera de tabla
    fuera = s.index[s["dia_en_tabla"] == 0]
    rachas, ini, prev = [], None, None
    for f in fuera:
        if prev is None or (f - prev).days > 1:
            if ini is not None:
                rachas.append((ini, prev))
            ini = f
        prev = f
    if ini is not None:
        rachas.append((ini, prev))
    diag["rachas_fuera_de_tabla_ge3"] = [f"{a.date()}..{b.date()} ({(b - a).days + 1} d)" for a, b in rachas
                                         if (b - a).days + 1 >= 3]
    diag["n_dias_fuera_de_tabla"] = int(len(fuera))

    # Cribado descriptivo de atipicos (no se corrige nada): valor > 4 veces la mediana del mismo dia de la
    # semana en las 4 semanas anteriores y las 4 siguientes (ventana centrada, solo para describir).
    cand = []
    for col in ("vol_bolsa", "vol_cajon"):
        x = s[col]
        vecinos = pd.concat([x.shift(7 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)], axis=1)
        med = vecinos.median(axis=1, skipna=True)
        razon = x / med
        sel = razon[(razon > 4) & (x > 100)]
        for f, r in sel.items():
            cand.append({"fecha": f.date().isoformat(), "serie": col, "valor_t": x[f], "mediana_vecinos_t": med[f],
                         "razon": round(float(r), 2), "anomalia_documentada": int(s.loc[f, "anomalia_documentada"])})
    cand = pd.DataFrame(cand).sort_values("razon", ascending=False)
    cand.to_csv(REPORTES / "dia2_candidatos_atipicos.csv", index=False, float_format="%.2f", lineterminator="\n")
    diag["candidatos_atipicos"] = {"criterio": "valor > 4 x mediana del mismo dia de la semana (+-4 semanas) y > 100 t",
                                   "n": int(len(cand)), "n_en_2021_2025": int(cand["fecha"].between("2021-01-01", "2025-12-31").sum()),
                                   "top10": cand.head(10).to_dict(orient="records")}

    # '...'
    ctrl = control_puntos_suspensivos(s)
    ctrl.to_csv(REPORTES / "dia2_control_puntos_suspensivos.csv", index=False, lineterminator="\n")
    disc = ctrl[ctrl["coincide_faltante"] != ctrl["coincide_cero"]]
    diag["control_puntos_suspensivos"] = {
        "n_pdf": int(ctrl["fecha_pdf"].nunique()), "n_comparaciones": int(len(ctrl)),
        "n_coincide_faltante": int(ctrl["coincide_faltante"].sum()), "n_coincide_cero": int(ctrl["coincide_cero"].sum()),
        "n_discriminantes": int(len(disc)), "discriminantes_a_favor_faltante": int(disc["coincide_faltante"].sum()),
        "dias_con_puntos_suspensivos_comparados_ayer_hoy": int(((ctrl["campo"].isin(["ayer", "hoy"]))
                                                                & (ctrl["n_puntos_suspensivos_en_ventana"] == 1)).sum()),
        "pdf_muestra_dos_puntos_en_esos_dias": int(((ctrl["campo"].isin(["ayer", "hoy"]))
                                                     & (ctrl["n_puntos_suspensivos_en_ventana"] == 1)
                                                     & (ctrl["valor_pdf"] == "")).sum()),
    }

    # Ruptura
    diag["ruptura_2005_2006"] = ruptura(s, mens)

    # Consistencia diario vs mensual
    mc = mens[mens["mes_completo"] == 1].copy()
    for p in ("bolsa", "cajon"):
        mc[f"dif_{p}"] = mc[f"vol_{p}"] - mc[f"vol_{p}_suma_diario"]
    mc[["anio", "mes", "vol_bolsa", "vol_bolsa_suma_diario", "dif_bolsa", "vol_cajon", "vol_cajon_suma_diario",
        "dif_cajon", "dias_mes", "dias_con_vol_bolsa", "dias_con_vol_cajon"]].to_csv(
        REPORTES / "dia2_consistencia_mensual.csv", index=False, float_format="%.2f", lineterminator="\n")
    cons = {}
    for p in ("bolsa", "cajon"):
        dif = mc[f"dif_{p}"].dropna().abs()
        cons[p] = {"n_meses_comparables": int(dif.size), "n_iguales_tol_0_5t": int((dif <= 0.5).sum()),
                   "dif_abs_max_t": round(float(dif.max()), 2),
                   "meses_con_dif_gt_0_5t": [f"{a}-{mm:02d}" for a, mm, v in
                                            zip(mc["anio"], mc["mes"], mc[f"dif_{p}"].abs()) if v > 0.5][:20],
                   "meses_mensual_nan_diario_no": [f"{a}-{mm:02d}" for a, mm, x, z in
                                                   zip(mc["anio"], mc["mes"], mc[f"vol_{p}"], mc[f"vol_{p}_suma_diario"])
                                                   if pd.isna(x) and not pd.isna(z)]}
    pp = mc[["anio", "mes", "precio_prom"]].copy()
    pd_m = s.groupby([s.index.year, s.index.month])["precio_prom"].mean()
    pp["media_diaria"] = pd_m.reindex(list(zip(pp.anio, pp.mes))).values
    cons["precio_bolsa_mensual_vs_media_diaria"] = {
        "dif_abs_mediana": round(float((pp.precio_prom - pp.media_diaria).abs().median()), 4),
        "dif_abs_max": round(float((pp.precio_prom - pp.media_diaria).abs().max()), 4)}
    diag["consistencia_diario_mensual"] = cons

    # Patron semanal por anio
    reg = []
    for (a, dow), g in s.groupby([s.index.year, s["dia_semana"]]):
        reg.append({"anio": a, "dia": DIAS[dow], "mediana_vol_total": g["vol_total_estricto"].median(),
                    "mediana_vol_bolsa": g["vol_bolsa"].median(), "mediana_vol_cajon": g["vol_cajon"].median(),
                    "n_vol_total": int(g["vol_total_estricto"].count())})
    semanal = pd.DataFrame(reg)
    semanal.to_csv(REPORTES / "dia2_semanal_mediana.csv", index=False, float_format="%.1f", lineterminator="\n")
    r = semanal.pivot(index="anio", columns="dia", values="mediana_vol_bolsa")
    razon = (r["mie"] / r[[x for x in DIAS if x != "mie"]].median(axis=1)).round(3)
    diag["razon_miercoles_bolsa_por_anio"] = {int(k): float(v) for k, v in razon.items()}
    for per, (a0, a1) in {"1997_2005": (1997, 2005), "2006_2015": (2006, 2015), "2016_2026": (2016, 2026)}.items():
        sub = s[(s.index.year >= a0) & (s.index.year <= a1)]
        diag[f"mediana_vol_total_por_dia_{per}"] = {DIAS[k]: float(v) for k, v in sub.groupby("dia_semana")["vol_total_estricto"].median().items()}

    # Patron anual y tendencia
    m = mens[mens["mes_completo"] == 1]
    an = m.groupby("anio")["vol_total_sisap"].sum(min_count=12)
    diag["vol_total_anual_t"] = {int(k): (None if pd.isna(v) else round(float(v))) for k, v in an.items()}
    perfil = {}
    for a0, a1 in [(1998, 2005), (2006, 2015), (2016, 2025)]:
        sub = m[(m.anio >= a0) & (m.anio <= a1)].copy()
        sub["ind"] = sub["vol_total_sisap"] / sub.groupby("anio")["vol_total_sisap"].transform("mean")
        perfil[f"{a0}_{a1}"] = {int(k): round(float(v), 3) for k, v in sub.groupby("mes")["ind"].median().items()}
    diag["perfil_estacional_mediana"] = perfil
    a_ok = an.dropna()
    a_ok = a_ok[a_ok.index >= 1998]
    tasa = np.polyfit(a_ok.index, np.log(a_ok.values), 1)[0]
    diag["tendencia_log_lineal_anual_pct"] = round(float(np.expm1(tasa) * 100), 2)

    # SISAGRI
    ruta = Path(os.environ.get("SISAGRI_CSV", CFG["sisagri"]["csv_nacional_sutil"]))
    sis_cmp = None
    if ruta.exists():
        sg = pd.read_csv(ruta, dtype={"mes": int})
        sg = sg[(sg["año"] >= 2016) & (sg["año"] <= 2025)]
        c = m[(m.anio >= 2016) & (m.anio <= 2025)][["anio", "mes", "vol_total_sisap", "mes_incompleto"]].merge(
            sg.rename(columns={"año": "anio"})[["anio", "mes", "produccion_t_nacional"]], on=["anio", "mes"])
        c = c.rename(columns={"vol_total_sisap": "gmml", "produccion_t_nacional": "sisagri"})
        c["f"] = pd.to_datetime(dict(year=c.anio, month=c.mes, day=1))
        c = c.sort_values("f").reset_index(drop=True)
        filas = []
        transf = {"niveles": lambda x: x, "dif_mensual": lambda x: x.diff(),
                  "var_pct_mensual": lambda x: x.pct_change() * 100}
        for muestra in ("solo_meses_completos", "todos"):
          for nombre, f in transf.items():
            for rez in (0, 1, 2):
                g_ = c["gmml"].where(c["mes_incompleto"] == 0) if muestra == "solo_meses_completos" else c["gmml"]
                a, b = f(g_), f(c["sisagri"]).shift(rez)   # rez > 0: produccion de t-rez frente a GMML de t
                ok = a.notna() & b.notna()
                pr = stats.pearsonr(a[ok], b[ok])
                sp = stats.spearmanr(a[ok], b[ok])
                filas.append({"muestra": muestra, "transformacion": nombre, "rezago_sisagri_meses": rez, "n": int(ok.sum()),
                              "pearson_r": round(float(pr.statistic), 3), "pearson_p": float(pr.pvalue),
                              "spearman_rho": round(float(sp.statistic), 3), "spearman_p": float(sp.pvalue)})
        corr = pd.DataFrame(filas)
        corr.to_csv(REPORTES / "dia2_correlacion_sisagri.csv", index=False, lineterminator="\n")
        diag["sisagri"] = {"archivo": str(ruta), "sha256": sha256_archivo(ruta), "n_meses": int(len(c)),
                           "nota": "copia de solo lectura del repositorio del articulo; no se modifica"}
        sis_cmp = c
    else:
        diag["sisagri"] = {"archivo": str(ruta), "estado": "no encontrado: correlacion no calculada"}

    figuras(s, mens, semanal, sis_cmp, diag["ruptura_2005_2006"])

    # Brazo D: dias de prueba excluidos frente a evaluados (protocolo 2.3)
    tab_ex, diag["excluidos_vs_evaluados"] = excluidos_vs_evaluados(s)
    tab_ex.to_csv(REPORTES / "dia2_excluidos_vs_evaluados.csv", index=False, lineterminator="\n")
    # Brazo M: objetivo mensual nuevo frente a la suma del SISAP (protocolo 2.1)
    tab_m, diag["mensual_vs_sisap"] = mensual_vs_sisap(mens)
    tab_m.to_csv(REPORTES / "dia2_mensual_vs_sisap.csv", index=False, float_format="%.2f", lineterminator="\n")
    # Regla '...' en una presentacion = 0: dias con ambas en '...' y evaluables por anio de prueba
    tab_a, diag["dias_ambas_faltantes"] = dias_ambas_faltantes(s)
    tab_a.to_csv(REPORTES / "dia2_dias_ambas_faltantes.csv", index=False, lineterminator="\n")
    ev = evaluables_por_anio(s, mens)
    ev.to_csv(REPORTES / "dia2_evaluables_por_anio.csv", lineterminator="\n")
    diag["evaluables_por_anio"] = {str(k): {c: int(v) for c, v in r.items()} for k, r in ev.iterrows()}

    # Resumen de cobertura de la serie principal
    diag["cobertura"] = {
        "dias_calendario": int(len(s)), "dias_en_tabla": int(s["dia_en_tabla"].sum()),
        "dias_con_vol_total": int(s["vol_total_estricto"].notna().sum()), "dias_con_vol_bolsa": int(s["vol_bolsa"].notna().sum()),
        "dias_con_precio_bolsa": int(s["precio_prom"].notna().sum()),
        "meses_completos": int(len(m)), "meses_con_vol_total": int(m["vol_total_sisap"].notna().sum()),
        "meses_sin_vol_total": [f"{a}-{mm:02d}" for a, mm, v in zip(m.anio, m.mes, m.vol_total_sisap) if pd.isna(v)],
        "meses_incompletos": [f"{a}-{mm:02d}" for a, mm, v in zip(m.anio, m.mes, m.mes_incompleto) if v == 1]}
    (REPORTES / "dia2_diagnostico.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1, default=str),
                                                    encoding="utf-8", newline="\n")
    print(json.dumps({k: diag[k] for k in ["primer_dia_vol_total", "primer_dia_precio_bolsa", "cobertura"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
