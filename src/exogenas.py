"""Experimento 3 (posterior al protocolo; docs/ANEXO_EXP3.md): exógenas alineadas para la rama B de la DualLSTM.

Fuentes (solo lectura; no se copian a este repositorio):
- producción provincial de limón sutil (SISAGRI, ETL del artículo): pesos por provincia = participación en la
  producción de 2016–2019, fijos para todas las ventanas;
- NASA POWER diario (descarga propia, community AG) en las coordenadas provinciales del artículo;
- INDECI, BD_2003-2025_EMERGENCIAS.csv (SHA-256 de data_sources.json del artículo), mismas reglas que
  etl/02_indeci_a_panel.py del artículo, pero por día (fecha de la emergencia);
- sentimiento por noticia (RoBERTuito, score = P(pos) − P(neg)); solo las columnas fecha y score: el texto no se lee;
- brazo M: series mensuales del artículo (clima, INDECI y sentimiento), agregadas con los mismos pesos.

Convención de rezagos (anclada en el objetivo t): la última fila del contexto de la red es la del día (mes) t−1 y
lleva exógenas de t−1, t−7 y la media de t−30…t−1 (D) o de t−1, t−3 y t−6 (M). Por tanto, la fila j lleva x(j),
x(j−6) y media(x(j−29..j)) en D, y x(j), x(j−2), x(j−5) en M. Ningún pronóstico usa exógenas del día (mes) t.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

ART = C.ARTICULO_RAIZ / "v2_reentrenamiento" / "data"
ART_PRIV = Path(r"C:\Machine-learming\limon-forecasting-mdpi\v2_reentrenamiento\data")   # copia privada del artículo
PROD_PROVINCIAL = ART / "interim" / "limon_sutil_provincia.csv"
NASA_MENSUAL = ART / "raw" / "nasa_power" / "por_provincia" / "clima_nasa_power_2016_2025.csv"
INDECI_MENSUAL = ART / "interim" / "indeci_limpio_2016_2025.csv"
SENT_MENSUAL = ART / "interim" / "noticias" / "sentimiento_mensual_2016_2025.csv"
INDECI_CRUDO = ART_PRIV / "raw" / "indeci" / "oficial_2016_2018" / "BD_2003-2025_EMERGENCIAS.csv"
INDECI_SHA = "a1fe63f0a7c836b8ff7dd1c87271d3bd3144f8444e41f93b7db205c85fef2687"   # data_sources.json del artículo
NOTICIAS_ITEM = ART_PRIV / "interim" / "noticias" / "noticias_por_item.csv"

NASA_CACHE = C.RAIZ / "data" / "raw" / "nasa_power_diario"          # no se versiona (data/raw/*)
EXOG_DIARIAS = C.RAIZ / "data" / "processed" / "exogenas_diarias.csv"     # no se versiona
EXOG_MENSUALES = C.RAIZ / "data" / "processed" / "exogenas_mensuales.csv"  # no se versiona
PESOS = C.RAIZ / "resultados" / "exp3" / "pesos_provincias_2016_2019.csv"

INICIO_EXP3, FIN_EXP3 = "2016-01-01", "2025-12-31"
ANIOS_PESOS = (2016, 2019)
NASA_PARAMS = ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M")
PELIGROS_CLIMATICOS = ["LLUVIA INTENSA", "VIENTOS FUERTES", "BAJAS TEMPERATURAS", "INUNDACION", "DESLIZAMIENTO",
                       "HUAYCO", "SEQUIA", "EROSION", "TORMENTA ELECTRICA", "MAREJADA"]

# columnas base (sin rezagar) por familia y brazo
FAMILIAS = {
    "D": {"clima": ["prec", "tmax", "tmin", "rh"],
          "indeci": ["eventos", "afectados"],
          "noticias": ["sent", "n_noticias"]},
    "M": {"clima": ["T2M", "T2M_MAX", "WS2M", "PRECTOTCORR", "RH2M"],
          "indeci": ["num_emergencias", "personas_afectadas", "personas_damnificadas", "total_afectados",
                     "hectareas_cultivo_perdidas", "hectareas_cultivo_afectadas"],
          "noticias": ["avg_sentiment", "n_noticias"]},
}
VARIANTES = {"V0": (), "V1": ("clima",), "V2": ("indeci",), "V3": ("noticias",), "V4": ("clima", "indeci", "noticias")}
MARGEN_FILAS = {"D": 29, "M": 5}       # filas de calentamiento de las exógenas (media de 30 días; rezago 6 meses)


# ----------------------------------------------------------------------------- pesos
def pesos_provinciales() -> pd.DataFrame:
    p = pd.read_csv(PROD_PROVINCIAL, encoding="utf-8-sig")
    p = p[(p["año"] >= ANIOS_PESOS[0]) & (p["año"] <= ANIOS_PESOS[1])]
    if p["año"].max() > ANIOS_PESOS[1]:
        raise AssertionError("Los pesos usan datos posteriores a 2019.")
    q = p.groupby(["departamento", "provincia"], as_index=False).produccion_t.sum()
    q = q[q.produccion_t > 0].copy()
    q["peso"] = q.produccion_t / q.produccion_t.sum()
    coord = (pd.read_csv(NASA_MENSUAL, encoding="utf-8-sig")[["departamento", "provincia", "lat", "lon"]]
             .drop_duplicates(["departamento", "provincia"]))
    q = q.merge(coord, on=["departamento", "provincia"], how="left")
    if q.lat.isna().any():
        raise AssertionError("Provincias sin coordenadas en el archivo NASA del artículo.")
    return q.sort_values("peso", ascending=False).reset_index(drop=True)


def _ponderar(panel: pd.DataFrame, pesos: pd.DataFrame, clave: list[str], cols: list[str]) -> pd.DataFrame:
    """Media ponderada por fecha; si falta una provincia en una fecha, se renormalizan los pesos."""
    x = panel.merge(pesos[["departamento", "provincia", "peso"]], on=["departamento", "provincia"], how="inner")
    out = {}
    for c in cols:
        v = x[c].astype(float)
        ok = v.notna()
        num = (v.where(ok, 0) * x.peso).groupby([x[k] for k in clave]).sum()
        den = (x.peso.where(ok, 0)).groupby([x[k] for k in clave]).sum()
        out[c] = num / den.replace(0, np.nan)
    return pd.DataFrame(out).reset_index()


# ----------------------------------------------------------------------------- fuentes diarias
def descargar_nasa(pesos: pd.DataFrame, log=print) -> pd.DataFrame:
    """NASA POWER diario por provincia (cache en data/raw/nasa_power_diario/, reanudable)."""
    import json
    import time

    import requests
    NASA_CACHE.mkdir(parents=True, exist_ok=True)
    filas = []
    for _, r in pesos.iterrows():
        f = NASA_CACHE / f"{r.departamento}_{r.provincia}.json".replace(" ", "_")
        if not f.exists():
            params = {"parameters": ",".join(NASA_PARAMS), "community": "AG", "longitude": round(r.lon, 4),
                      "latitude": round(r.lat, 4), "start": INICIO_EXP3.replace("-", ""),
                      "end": FIN_EXP3.replace("-", ""), "format": "JSON"}
            for intento in range(5):
                try:
                    resp = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params,
                                        timeout=120)
                    resp.raise_for_status()
                    f.write_text(resp.text, encoding="utf-8")
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"  reintento {intento + 1} {r.provincia}: {e}")
                    time.sleep(5 * (intento + 1))
            else:
                raise RuntimeError(f"NASA POWER no respondió para {r.departamento}/{r.provincia}")
            time.sleep(0.7)
        d = json.loads(f.read_text(encoding="utf-8"))
        par = d["properties"]["parameter"]
        fill = d.get("header", {}).get("fill_value", -999.0)
        fechas = sorted(par[NASA_PARAMS[0]].keys())
        t = pd.DataFrame({"fecha": pd.to_datetime(fechas, format="%Y%m%d"),
                          **{k: [par[k][x] for x in fechas] for k in NASA_PARAMS}})
        t = t.replace(fill, np.nan)
        t["departamento"], t["provincia"] = r.departamento, r.provincia
        filas.append(t)
    return pd.concat(filas, ignore_index=True)


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def indeci_diario(pesos: pd.DataFrame) -> pd.DataFrame:
    if _sha256(INDECI_CRUDO) != INDECI_SHA:
        raise AssertionError("BD_2003-2025_EMERGENCIAS.csv no coincide con el SHA-256 del artículo.")
    r = pd.read_csv(INDECI_CRUDO, sep=";", encoding="latin-1", dtype=str)
    r.columns = [c.strip() for c in r.columns]
    fcol = [c for c in r.columns if c.upper().startswith("FECHA")][0]
    for c in ("DPTO.", "PROV.", "PELIGRO"):
        r[c] = r[c].str.strip()
    # Fecha: dd/mm/aaaa en 2003-2024; los registros de 2025 vienen como mm/dd/aa. Se exige que mes y año
    # coincidan con las columnas MES y AÑO del propio registro.
    s = r[fcol].str.strip()
    corto = s.str.fullmatch(r"\d\d/\d\d/\d\d").fillna(False)
    r["fecha"] = pd.to_datetime(s.where(~corto), format="%d/%m/%Y", errors="coerce")
    r.loc[corto, "fecha"] = pd.to_datetime(s[corto], format="%m/%d/%y", errors="coerce")
    meses = {m: i + 1 for i, m in enumerate(["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO", "AGOSTO",
                                             "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"])}
    anio = pd.to_numeric(r[r.columns[2]], errors="coerce")          # columna AÑO
    mes = r.MES.str.strip().map(meses)
    en_rango = anio.between(2016, 2025)
    if r.loc[en_rango, "fecha"].isna().any():
        raise AssertionError("INDECI: fechas sin interpretar en 2016-2025.")
    if not ((r.fecha.dt.year == anio) & (r.fecha.dt.month == mes))[en_rango].all():
        raise AssertionError("INDECI: la fecha no coincide con las columnas AÑO y MES.")
    r = r[r.fecha.between(INICIO_EXP3, FIN_EXP3) & r.PELIGRO.isin(PELIGROS_CLIMATICOS)].copy()
    r["departamento"] = r["DPTO."].replace({"LIMA METROPOLITANA": "LIMA", "LIMA PROVINCIAS": "LIMA"})
    r["provincia"] = r["PROV."].replace({"LIMA METROPOLITANA": "LIMA"})
    for c in ("AFECTADOS", "DAMNIFICADOS"):
        r[c] = pd.to_numeric(r[c].str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    r = r.merge(pesos[["departamento", "provincia"]], on=["departamento", "provincia"], how="inner")
    g = r.groupby("fecha").agg(eventos=("PELIGRO", "size"), afectados=("AFECTADOS", "sum"),
                               damnificados=("DAMNIFICADOS", "sum"))
    g["afectados"] = g.afectados + g.damnificados        # = total_afectados del artículo
    return g[["eventos", "afectados"]]


def noticias_diarias() -> pd.DataFrame:
    n = pd.read_csv(NOTICIAS_ITEM, usecols=["fecha", "score"])          # nunca se leen otras columnas
    n["fecha"] = pd.to_datetime(n.fecha).dt.normalize()
    g = n.groupby("fecha").score.agg(["sum", "count"]).rename(columns={"sum": "sent_suma", "count": "n_noticias"})
    return g


def construir_diarias(log=print) -> pd.DataFrame:
    pesos = pesos_provinciales()
    PESOS.parent.mkdir(parents=True, exist_ok=True)
    pesos.to_csv(PESOS, index=False, float_format="%.8g")
    idx = pd.date_range(INICIO_EXP3, FIN_EXP3, freq="D", name="fecha")
    nasa = descargar_nasa(pesos, log)
    clima = _ponderar(nasa, pesos, ["fecha"], list(NASA_PARAMS)).set_index("fecha").reindex(idx)
    clima.columns = ["prec", "tmax", "tmin", "rh"]
    ind = indeci_diario(pesos).reindex(idx).fillna(0)                   # día sin registro = 0 (regla del artículo)
    nt = noticias_diarias().reindex(idx)
    nt["n_noticias"] = nt.n_noticias.fillna(0)
    nt["sent_suma"] = nt.sent_suma.fillna(0)
    out = pd.concat([clima, ind, nt], axis=1)
    out["sent"] = (out.sent_suma / out.n_noticias).where(out.n_noticias > 0)    # 0 noticias = faltante
    out["sin_noticias"] = (out.n_noticias == 0).astype(int)
    out = out[["prec", "tmax", "tmin", "rh", "eventos", "afectados", "sent", "n_noticias", "sent_suma", "sin_noticias"]]
    out.to_csv(EXOG_DIARIAS, float_format="%.6g")
    return out


def construir_mensuales() -> pd.DataFrame:
    pesos = pesos_provinciales()
    nasa = pd.read_csv(NASA_MENSUAL, encoding="utf-8-sig")
    nasa["fecha"] = pd.to_datetime(nasa.DATE)
    clima = _ponderar(nasa, pesos, ["fecha"], FAMILIAS["M"]["clima"]).set_index("fecha")
    ind = pd.read_csv(INDECI_MENSUAL, encoding="utf-8-sig")
    ind["fecha"] = pd.to_datetime(dict(year=ind.anio, month=ind.mes, day=1))
    ind = ind.merge(pesos[["departamento", "provincia"]], on=["departamento", "provincia"], how="inner")
    indeci = ind.groupby("fecha")[FAMILIAS["M"]["indeci"]].sum()          # recuento en las provincias productoras
    s = pd.read_csv(SENT_MENSUAL, encoding="utf-8-sig")
    s["fecha"] = pd.to_datetime(dict(year=s["año"], month=s.mes, day=1))
    sent = s.set_index("fecha")[["avg_sentiment", "n_noticias"]]          # tal como el artículo (0 si no hay)
    idx = pd.date_range(INICIO_EXP3, FIN_EXP3, freq="MS", name="fecha")
    out = pd.concat([clima.reindex(idx), indeci.reindex(idx), sent.reindex(idx)], axis=1)
    out.to_csv(EXOG_MENSUALES, float_format="%.6g")
    return out


# ----------------------------------------------------------------------------- variables de la rama B
def cargar(brazo: str) -> pd.DataFrame:
    ruta = EXOG_DIARIAS if brazo == "D" else EXOG_MENSUALES
    return pd.read_csv(ruta, parse_dates=["fecha"], index_col="fecha")


def columnas_B(brazo: str, familias) -> list[str]:
    nombres = []
    for fam in familias:
        for c in FAMILIAS[brazo][fam]:
            if brazo == "D":
                if c == "sent":
                    nombres += ["sent_t1", "sin_t1", "sent_t7", "sin_t7", "sent_m30", "sin_m30"]
                else:
                    nombres += [f"{c}_t1", f"{c}_t7", f"{c}_m30"]
            else:
                nombres += [f"{c}_t1", f"{c}_t3", f"{c}_t6"]
    return nombres


def matriz_B(index: pd.DatetimeIndex, brazo: str, familias, ex: pd.DataFrame | None = None) -> np.ndarray | None:
    """Exógenas transformadas, alineadas fila a fila con `index` (convención del encabezado). None si no hay familias."""
    if not familias:
        return None
    ex = cargar(brazo) if ex is None else ex
    ex = ex.reindex(index)
    cols = []
    for fam in familias:
        for c in FAMILIAS[brazo][fam]:
            if brazo == "D":
                if c == "sent":
                    # sentimiento: valor del día (0 si no hay noticias) + indicador; media de 30 días ponderada
                    # por número de noticias (suma de scores / número de noticias)
                    v, sin = ex.sent.fillna(0.0), ex.sin_noticias.astype(float)
                    s30 = ex.sent_suma.rolling(30, min_periods=30).sum()
                    n30 = ex.n_noticias.rolling(30, min_periods=30).sum()
                    m30 = (s30 / n30).where(n30 > 0)
                    cols += [v, sin, v.shift(6), sin.shift(6), m30.fillna(0.0).where(n30.notna()),
                             (n30 == 0).astype(float).where(n30.notna())]
                else:
                    x = ex[c].astype(float)
                    cols += [x, x.shift(6), x.rolling(30, min_periods=30).mean()]
            else:
                x = ex[c].astype(float)
                cols += [x, x.shift(2), x.shift(5)]
    return np.column_stack([c.to_numpy(float) for c in cols])


def serie_exp3(brazo: str) -> pd.DataFrame:
    """Serie principal del protocolo (imputación del día 3) recortada a 2016–2025."""
    from src.datos import ESCENARIOS, serie
    df = serie(ESCENARIOS["principal"][brazo])
    return df[(df.index >= INICIO_EXP3) & (df.index <= FIN_EXP3)].copy()
