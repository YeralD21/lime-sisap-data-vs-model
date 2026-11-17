"""Carga de series e imputación (docs/PROTOCOLO.md §2.1, §2.2, §9).

`serie(esc)` devuelve un DataFrame indexado por fecha (días en D, primer día del mes en M) con:
- `y`: objetivo imputado, la entrada de todos los modelos (§2.2). Solo usa información pasada.
- `y_eval`: valor real del objetivo si el día o mes es evaluable; NaN si no (§2.2, «En la evaluación»).
- `precio`: precio exógeno imputado (D: precio_prom diario con t-7; M: precio mensual del SISAP).
- `faltante`: True donde `y` es imputado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C

OBJETIVOS = {"D": ("principal", "estricto", "bolsa"), "M": ("principal", "estricto")}


@dataclass(frozen=True)
class Escenario:
    """Una serie del experimento. principal: §2.1; estricto: sensibilidad 1; bolsa: sensibilidad 2 (solo D);
    inicio=INICIO_SENS6: sensibilidad 6 (solo D); anomalia="publicada": sensibilidad 7."""
    brazo: str = "D"
    objetivo: str = "principal"
    inicio: str = C.INICIO
    anomalia: str = "faltante"

    def __post_init__(self):
        if self.brazo not in OBJETIVOS or self.objetivo not in OBJETIVOS[self.brazo]:
            raise ValueError(f"Escenario no preregistrado: {self}")
        if self.anomalia not in ("faltante", "publicada"):
            raise ValueError(self.anomalia)
        if self.brazo == "M" and self.inicio != C.INICIO:
            raise ValueError("La sensibilidad 6 es solo del brazo D (§9).")

    @property
    def nombre(self) -> str:
        partes = [self.brazo, self.objetivo]
        if self.inicio != C.INICIO:
            partes.append("desde" + self.inicio[:4])
        if self.anomalia == "publicada":
            partes.append("anomalia_publicada")
        return "_".join(partes)


def imputar_t7(v: np.ndarray) -> np.ndarray:
    """§2.2: cada faltante se reemplaza por el valor imputado de t-7, en orden cronológico."""
    v = np.asarray(v, dtype=float).copy()
    for i in np.flatnonzero(np.isnan(v)):
        if i < 7:
            raise ValueError(f"Faltante en la posición {i}: no hay t-7 dentro de la serie.")
        v[i] = v[i - 7]
    return v


def _diario() -> pd.DataFrame:
    d = pd.read_csv(C.DIARIO, parse_dates=["fecha"])
    d = d[d.fecha <= C.FIN_DATOS].set_index("fecha").asfreq("D")
    if d.index.min() > pd.Timestamp(C.INICIO):
        raise AssertionError("La serie diaria empieza después de 1997-10-01.")
    return d


def _diaria(esc: Escenario) -> pd.DataFrame:
    d = _diario()
    d = d[d.index >= esc.inicio]
    anom = d.anomalia_documentada.eq(1) & (esc.anomalia == "faltante")
    if esc.objetivo == "principal":
        raw, falta = d.vol_total, d.falta_vol_total.eq(1) | anom
    elif esc.objetivo == "estricto":
        raw, falta = d.vol_total_estricto, d.falta_vol_total_estricto.eq(1) | anom
    else:  # bolsa: la anomalía es del cajón, no afecta a la bolsa
        raw, falta = d.vol_bolsa, d.falta_vol_bolsa.eq(1)
    falta = falta | raw.isna()
    y_obs = raw.where(~falta)
    precio = d.precio_prom.where(d.falta_precio.eq(0) & d.precio_prom.notna())
    return pd.DataFrame({"y": imputar_t7(y_obs.to_numpy()), "y_eval": y_obs,
                         "precio": imputar_t7(precio.to_numpy()), "faltante": falta.to_numpy()}, index=d.index)


def _mensual(esc: Escenario) -> pd.DataFrame:
    m = pd.read_csv(C.MENSUAL)
    m.index = pd.to_datetime(dict(year=m.anio, month=m.mes, day=1))
    m = m[(m.index >= esc.inicio) & (m.index <= C.FIN_DATOS)].asfreq("MS")
    if esc.objetivo == "principal":
        raw, falta = m.vol_total, m.flag_faltante.eq(1)
    else:  # sensibilidad 1: vol_total_escalado, faltante con < 80 % de días observados
        raw, falta = m.vol_total_escalado, m.flag_faltante_estricto.eq(1)
    if esc.anomalia == "faltante":
        falta = falta | m.mes_con_anomalia.eq(1)          # §2.2: 2021-02 no se evalúa
    else:
        falta = falta & ~(m.mes_con_anomalia.eq(1) & m.dias_faltantes.eq(0))  # §9, sens. 7
    falta = falta | raw.isna()
    # §2.2: el mes faltante se imputa con la suma mensual de la serie diaria imputada (misma regla del brazo D)
    dia = _diaria(Escenario("D", esc.objetivo, C.INICIO, esc.anomalia))
    suma = dia.y.resample("MS").sum().reindex(m.index)
    y_obs = raw.where(~falta)
    y = y_obs.fillna(suma)
    if y.isna().any():
        raise AssertionError("Mes faltante sin suma diaria imputada.")
    precio = m.precio_prom.astype(float)
    if precio.isna().any():
        raise AssertionError("Precio mensual faltante: el protocolo no define imputación para M.")
    return pd.DataFrame({"y": y.to_numpy(float), "y_eval": y_obs, "precio": precio.to_numpy(float),
                         "faltante": falta.to_numpy()}, index=m.index)


def serie(esc: Escenario) -> pd.DataFrame:
    return _diaria(esc) if esc.brazo == "D" else _mensual(esc)


ESCENARIOS = {
    "principal": {"D": Escenario("D"), "M": Escenario("M")},
    "sens1_estricto": {"D": Escenario("D", "estricto"), "M": Escenario("M", "estricto")},
    "sens2_bolsa": {"D": Escenario("D", "bolsa")},
    "sens6_desde2006": {"D": Escenario("D", inicio=C.INICIO_SENS6)},
    "sens7_anomalia_publicada": {"D": Escenario("D", anomalia="publicada"), "M": Escenario("M", anomalia="publicada")},
}
