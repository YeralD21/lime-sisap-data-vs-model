"""Día 4: figuras, tablas y diagnósticos descriptivos del Experimento 2 (no entrena ni recalcula la regla de decisión).

Lee los resultados guardados del día 3 (resultados/brazo_*/), los registros y, solo para dibujar, la serie observada de
data/processed. Escribe:
- reportes/figuras/<idioma>/fNN_*.pdf (vectorial) y .png (300 ppp), con --idioma en|es;
- reportes/figuras/tablas/*.csv y *.<idioma>.tex;
- reportes/figuras/datos/*.csv: datos agregados de cada figura (sin valores de la serie del SISAP).

Reanudable: una figura o tabla cuyos archivos ya existen no se rehace (salvo --forzar); los datos derivados de
reportes/figuras/datos/ se reutilizan si existen. Las figuras T5 (f08–f10) necesitan data/processed.

Uso: .venv\\Scripts\\python scripts/05_figuras.py --idioma en   (y luego --idioma es)
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src import config as C  # noqa: E402
from src import evaluacion as E  # noqa: E402
from src.analisis import completar_y_true, fechas_evaluables, matriz_ae  # noqa: E402
from src.datos import ESCENARIOS, serie  # noqa: E402
from src.ventanas import Ventana, particion  # noqa: E402

RES = RAIZ / "resultados"
FIG = RAIZ / "reportes" / "figuras"
DATOS = FIG / "datos"
TABLAS = FIG / "tablas"
SISAGRI_CSV = C.ARTICULO_RAIZ / "v2_reentrenamiento" / "data" / "interim" / "limon_sutil_nacional_mensual.csv"

# ----------------------------------------------------------------------------- estilo
# Okabe-Ito (apta para daltónicos); el mismo color y marcador por modelo en todas las figuras.
COLOR = {"Seasonal Naive": "#999999", "ETS": "#0072B2", "Theta": "#56B4E9", "SARIMA": "#009E73",
         "Chronos-Bolt-tiny": "#CC79A7", "LSTM univariada": "#E69F00", "DualLSTM": "#D55E00", "obs": "#000000"}
MARCA = {"Seasonal Naive": "v", "ETS": "s", "Theta": "D", "SARIMA": "^", "Chronos-Bolt-tiny": "P",
         "LSTM univariada": "X", "DualLSTM": "o"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "figure.dpi": 100})

TXT = {
    "es": {"LSTM univariada": "LSTM univariada", "N": "Tamaño de historia N", "N_D": "N (días)", "N_M": "N (meses)",
           "mae": "MAE (t)", "brazo_D": "Brazo D (diario, h = 1)", "brazo_M": "Brazo M (mensual, h = 1)",
           "delta": "Δ(N) (t)", "delta_pct": "Δ(N) (% del MAE de ref*)", "ref_nota": "Δ(N) = MAE DualLSTM − MAE ref*(N); bajo cada punto, ref*(N)",
           "dm_sig": "* DM-HLN p < 0,05", "train": "Entrenamiento (historia N)", "val": "Validación (Y−1)",
           "test": "Prueba (Y)", "anio_prueba": "Año de prueba Y", "fecha": "Fecha", "obs": "Observado",
           "dual": "DualLSTM (media de 10 semillas, N = máx)", "sn": "Seasonal Naive", "ref": "{r} (ref*, N = máx)",
           "t": "Toneladas", "epoca_parada": "Época de parada", "mejor_epoca": "Mejor época (early stopping)",
           "semillas": "MAE de prueba por semilla (t)", "ref_linea": "ref*(N): {r}", "dia": "Día de la semana",
           "dias": ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"], "mes": "Mes del año",
           "meses": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
           "acf": "ACF de los residuos", "rezago_D": "Rezago (días)", "rezago_M": "Rezago (meses)",
           "shock": "Shock (variación > P{p})", "normal": "Normal", "umbral": "Umbral de shock",
           "dm_titulo": "p-valor DM-HLN: DualLSTM frente a cada referencia", "referencia": "Referencia",
           "segundos": "Segundos por unidad (mediana)", "zoom": "Detalle: {a} a {b}",
           "perturb": "Diferencia máxima absoluta (perturbada − original)", "identico": "idéntico (0)",
           "gmml": "SISAP (GMML, ingreso mensual)", "sisagri": "SISAGRI (producción nacional)",
           "indice": "Índice (media 2016–2025 = 100)", "brecha_120_max": "Brecha de la DualLSTM en el brazo M",
           "peor": "DualLSTM peor", "mejor": "DualLSTM mejor"},
    "en": {"LSTM univariada": "Univariate LSTM", "N": "History length N", "N_D": "N (days)", "N_M": "N (months)",
           "mae": "MAE (t)", "brazo_D": "Arm D (daily, h = 1)", "brazo_M": "Arm M (monthly, h = 1)",
           "delta": "Δ(N) (t)", "delta_pct": "Δ(N) (% of ref* MAE)", "ref_nota": "Δ(N) = MAE DualLSTM − MAE ref*(N); ref*(N) shown below each point",
           "dm_sig": "* DM-HLN p < 0.05", "train": "Training (history N)", "val": "Validation (Y−1)",
           "test": "Test (Y)", "anio_prueba": "Test year Y", "fecha": "Date", "obs": "Observed",
           "dual": "DualLSTM (mean of 10 seeds, N = max)", "sn": "Seasonal Naive", "ref": "{r} (ref*, N = max)",
           "t": "Tonnes", "epoca_parada": "Stopping epoch", "mejor_epoca": "Best epoch (early stopping)",
           "semillas": "Test MAE per seed (t)", "ref_linea": "ref*(N): {r}", "dia": "Day of week",
           "dias": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "mes": "Month of year",
           "meses": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           "acf": "Residual ACF", "rezago_D": "Lag (days)", "rezago_M": "Lag (months)",
           "shock": "Shock (change > P{p})", "normal": "Normal", "umbral": "Shock threshold",
           "dm_titulo": "DM-HLN p-value: DualLSTM vs each reference", "referencia": "Reference",
           "segundos": "Seconds per unit (median)", "zoom": "Detail: {a} to {b}",
           "perturb": "Maximum absolute difference (perturbed − original)", "identico": "identical (0)",
           "gmml": "SISAP (GMML, monthly arrivals)", "sisagri": "SISAGRI (national production)",
           "indice": "Index (2016–2025 mean = 100)", "brecha_120_max": "DualLSTM gap in arm M",
           "peor": "DualLSTM worse", "mejor": "DualLSTM better"},
}
L: dict = {}


def nm(m: str) -> str:
    return L.get(m, m)


def _orden_N(b):
    return [str(n) for n in C.TAMANOS[b]]


def _x_N(b):
    """Posición en el eje log: N numérico; 'max' = mediana de la historia máxima de las 5 ventanas."""
    df = _serie(b)
    nmax = int(np.median([particion(df.index, Ventana(a), "max").n_hist for a in C.ANIOS_TEST]))
    return {str(n): (nmax if n == "max" else int(n)) for n in C.TAMANOS[b]}, nmax


def _etiquetas_N(b):
    xs, nmax = _x_N(b)
    mx = "máx" if IDIOMA == "es" else "max"
    return [xs[n] for n in _orden_N(b)], [n if n != "max" else f"{mx}\n(~{nmax})" for n in _orden_N(b)]


def eje_N(ax, b):
    """Eje x logarítmico con los tamaños de la curva; en M separa las etiquetas de 240 y máx (muy cercanas)."""
    ax.set_xscale("log")
    t, e = _etiquetas_N(b)
    ax.set_xticks(t, e)
    ax.minorticks_off()
    if b == "M":
        lab = ax.get_xticklabels()
        lab[-2].set_ha("right")
        lab[-1].set_ha("left")
    ax.set_xlim(t[0] / 1.25, t[-1] * 1.25)


def guardar(fig, nombre, forzar=False):
    d = FIG / IDIOMA
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{nombre}.pdf")
    fig.savefig(d / f"{nombre}.png", dpi=300)
    plt.close(fig)
    print("  ", d / nombre)


def hecho(nombre, forzar):
    d = FIG / IDIOMA
    return not forzar and (d / f"{nombre}.pdf").exists() and (d / f"{nombre}.png").exists()


def cache(nombre, f, forzar=False):
    ruta = DATOS / f"{nombre}.csv"
    if ruta.exists() and not forzar:
        return pd.read_csv(ruta, dtype={"N": str})
    DATOS.mkdir(parents=True, exist_ok=True)
    t = f()
    t.to_csv(ruta, index=False, float_format="%.6g")
    return pd.read_csv(ruta, dtype={"N": str})


# ----------------------------------------------------------------------------- carga (memoria; no se escribe la serie)
_SERIES, _PRON = {}, {}


def _serie(b, esc="principal"):
    if (b, esc) not in _SERIES:
        _SERIES[b, esc] = serie(ESCENARIOS[esc][b])
    return _SERIES[b, esc]


def _pron(b):
    if b not in _PRON:
        p = pd.read_csv(RES / f"brazo_{b}" / "principal" / "pronosticos.csv", dtype={"N": str}, low_memory=False)
        _PRON[b] = completar_y_true(p, _serie(b))
    return _PRON[b]


def _csv(b, esc, nombre):
    return pd.read_csv(RES / f"brazo_{b}" / esc / f"{nombre}.csv", dtype={"N": str})


def _ref_max(b):
    d = _csv(b, "principal", "delta")
    return d.loc[d.N == "max", "ref_estrella"].iloc[0]


def _ae_por_fecha(b, N, modelos, h=1):
    """AE promediado entre semillas por fecha evaluable (en memoria)."""
    AE, ref = matriz_ae(_pron(b), _serie(b), N, h, list(C.ANIOS_TEST), list(modelos))
    return AE, ref


# ============================================================================= T1
def f01_curva(forzar):
    if hecho("f01_curva_aprendizaje", forzar):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, b in zip(axs, "DM"):
        m = _csv(b, "principal", "metricas").query("anio == 'pool'")
        xs, _ = _x_N(b)
        for mod in C.MODELOS:
            s = m[m.modelo == mod].set_index("N").loc[_orden_N(b)]
            x = [xs[n] for n in s.index]
            ax.plot(x, s.MAE, marker=MARCA[mod], ms=4, color=COLOR[mod], lw=1.4 if mod == "DualLSTM" else 1,
                    label=nm(mod), zorder=3 if mod == "DualLSTM" else 2)
            ax.fill_between(x, s.MAE_ic_lo, s.MAE_ic_hi, color=COLOR[mod], alpha=0.12, lw=0)
        eje_N(ax, b)
        ax.set_xlabel(L[f"N_{b}"])
        ax.set_ylabel(L["mae"])
        ax.set_title(L[f"brazo_{b}"])
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    guardar(fig, "f01_curva_aprendizaje")


def _delta(b):
    d = _csv(b, "principal", "delta").set_index("N").loc[_orden_N(b)].reset_index()
    d["delta_pct"] = 100 * d.delta / d.MAE_ref
    d["ic_lo_pct"] = 100 * d.ic_lo / d.MAE_ref
    d["ic_hi_pct"] = 100 * d.ic_hi / d.MAE_ref
    return d


def f02_f03_brecha(forzar):
    for nombre, col, lo, hi, ylab in (("f02_brecha", "delta", "ic_lo", "ic_hi", "delta"),
                                      ("f03_brecha_pct", "delta_pct", "ic_lo_pct", "ic_hi_pct", "delta_pct")):
        if hecho(nombre, forzar):
            continue
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
        for ax, b in zip(axs, "DM"):
            d = _delta(b)
            xs, _ = _x_N(b)
            x = [xs[n] for n in d.N]
            ax.axhline(0, color="k", lw=0.8)
            ax.errorbar(x, d[col], yerr=[d[col] - d[lo], d[hi] - d[col]], color=COLOR["DualLSTM"], marker="o",
                        ms=5, capsize=3, lw=1.2)
            for xi, loi, hii, p, r in zip(x, d[lo], d[hi], d.p_DM, d.ref_estrella):
                if p < 0.05:
                    ax.annotate("*", (xi, hii), ha="center", va="bottom", fontsize=10)
                ax.annotate(nm(r), (xi, loi), ha="center", va="top", fontsize=6.5, xytext=(0, -3),
                            textcoords="offset points", color="0.3")
            eje_N(ax, b)
            ax.set_xlabel(L[f"N_{b}"])
            ax.set_ylabel(L[ylab])
            ax.set_title(L[f"brazo_{b}"])
            ax.set_ylim(min(0, d[lo].min()) * 1.1 - 0.08 * d[hi].max(), d[hi].max() * 1.15)
        fig.tight_layout()
        fig.text(0.01, -0.03, L["dm_sig"] + "; " + L["ref_nota"], ha="left", fontsize=7)
        guardar(fig, nombre)


def tablas_t1(forzar):
    TABLAS.mkdir(parents=True, exist_ok=True)
    filas = []
    for b in "DM":
        d = _delta(b)
        for _, r in d.iterrows():
            filas.append({"brazo": b, "N": r.N, "ref_estrella": r.ref_estrella, "MAE_DualLSTM": r.MAE_DualLSTM,
                          "MAE_ref": r.MAE_ref, "delta": r.delta, "ic_lo": r.ic_lo, "ic_hi": r.ic_hi,
                          "delta_pct": r.delta_pct, "ic_lo_pct": r.ic_lo_pct, "ic_hi_pct": r.ic_hi_pct,
                          "p_DM": r.p_DM, "n": int(r.n)})
    t = pd.DataFrame(filas)
    t.to_csv(TABLAS / "t1_mae_dual_ref_brecha.csv", index=False, float_format="%.6g")
    tex = TABLAS / f"t1_mae_dual_ref_brecha.{IDIOMA}.tex"
    if tex.exists() and not forzar:
        return
    es = IDIOMA == "es"
    cab = (r"Brazo & $N$ & ref$^*$ & MAE DualLSTM & MAE ref$^*$ & $\Delta$ [IC 95\,\%] & $\Delta$ (\%) [IC 95\,\%] \\"
           if es else r"Arm & $N$ & ref$^*$ & DualLSTM MAE & ref$^*$ MAE & $\Delta$ [95\,\% CI] & $\Delta$ (\%) [95\,\% CI] \\")
    lin = [r"\begin{tabular}{lllrrll}", r"\toprule", cab, r"\midrule"]
    for _, r in t.iterrows():
        n = ("máx" if es else "max") if r.N == "max" else r.N
        lin.append(f"{r.brazo} & {n} & {nm(r.ref_estrella)} & {r.MAE_DualLSTM:.1f} & {r.MAE_ref:.1f} & "
                   f"{r.delta:.1f} [{r.ic_lo:.1f}; {r.ic_hi:.1f}] & {r.delta_pct:.1f} [{r.ic_lo_pct:.1f}; {r.ic_hi_pct:.1f}] \\\\")
    lin += [r"\bottomrule", r"\end{tabular}"]
    tex.write_text("\n".join(lin) + "\n", encoding="utf-8")


# ============================================================================= T2
def f04_rolling(forzar):
    if hecho("f04_rolling_origin", forzar):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.2), sharey=False)
    filas = []
    for ax, b in zip(axs, "DM"):
        df = _serie(b)
        yk, etiquetas = [], []
        k = 0
        for a in C.ANIOS_TEST:
            for N in C.TAMANOS[b]:
                p = particion(df.index, Ventana(a), N)
                i0, f0 = df.index[p.ini_hist], df.index[p.fin_train]
                v0, v1 = df.index[p.val[0]], df.index[p.val[-1]]
                t0, t1 = df.index[p.test[0]], df.index[p.test[-1]]
                filas.append({"brazo": b, "anio": a, "N": str(N), "ini_hist": i0.date(), "fin_train": f0.date(),
                              "ini_val": v0.date(), "fin_val": v1.date(), "ini_test": t0.date(),
                              "fin_test": t1.date(), "n_hist": p.n_hist})
                y = -k
                ax.barh(y, (f0 - i0).days + 1, left=i0, height=0.7, color=COLOR["ETS"], alpha=0.75)
                ax.barh(y, (v1 - v0).days + 1, left=v0, height=0.7, color=COLOR["LSTM univariada"])
                ax.barh(y, (t1 - t0).days + 1, left=t0, height=0.7, color=COLOR["DualLSTM"])
                yk.append(y)
                etiquetas.append(f"{a}  N={'máx' if (N == 'max' and IDIOMA == 'es') else N}")
                k += 1
            k += 0.6
        ax.set_yticks(yk, etiquetas, fontsize=6.5)
        ax.set_title(L[f"brazo_{b}"].split(",")[0] + ")" if False else L[f"brazo_{b}"])
        ax.set_xlabel(L["fecha"])
        ax.set_xlim(pd.Timestamp("1997-01-01"), pd.Timestamp("2026-03-01"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLOR["ETS"], alpha=0.75, label=L["train"]),
                        Patch(color=COLOR["LSTM univariada"], label=L["val"]),
                        Patch(color=COLOR["DualLSTM"], label=L["test"])],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    DATOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(DATOS / "f04_ventanas.csv", index=False)
    guardar(fig, "f04_rolling_origin")


DESCRIPCION_PRUEBAS = {
    "test_evaluables_coinciden_con_protocolo": ("Días/meses evaluables por año = tabla de §2.2 (salvo D5)",
                                                "Evaluable days/months per year = §2.2 table (except D5)"),
    "test_anomalia": ("Anomalía 2021-02-01: faltante en principal; valor publicado en sens. 7",
                      "2021-02-01 anomaly: missing in main analysis; published value in sensitivity 7"),
    "test_historia_maxima_protocolo": ("Historia máxima por ventana = §4 (8 127–9 588 días; 267–315 meses)",
                                       "Maximum history per window = §4 (8,127–9,588 days; 267–315 months)"),
    "test_sin_datos_2026_y_desde_1997_10": ("Sin datos de 2026; inicio 1997-10-01 (2006-01-01 en sens. 6)",
                                            "No 2026 data; start 1997-10-01 (2006-01-01 in sensitivity 6)"),
    "test_mismos_dias_prueba_en_todos_los_N": ("Mismo fin de entrenamiento, validación y días de prueba para todo N",
                                               "Same training end, validation and test days for every N"),
    "test_imputacion_solo_usa_pasado": ("Imputación t−7 causal: cambiar el futuro no altera el pasado imputado",
                                        "Causal t−7 imputation: changing the future does not alter past imputed values"),
    "test_redes_sin_fuga_en_ajuste_ni_escalado": ("Perturbar datos posteriores al fin del entrenamiento no cambia escalador, secuencias ni objetivos; la entrada de t solo usa datos hasta t−h",
                                                  "Perturbing data after the training end leaves scaler, sequences and targets unchanged; the input for t uses data up to t−h only"),
    "test_escala_mase_fija_por_ventana": ("Escala MASE fija por ventana y sin datos posteriores al entrenamiento",
                                          "MASE scale fixed per window and free of post-training data"),
    "test_parametros_red": ("Recuento de parámetros de la DualLSTM (artículo y brazo D)",
                            "DualLSTM parameter count (article and arm D)"),
    "test_dm_hln_h1_igual_que_articulo": ("DM-HLN a h = 1 igual a la fórmula del artículo",
                                          "DM-HLN at h = 1 equals the article's formula"),
    "test_bootstrap_pareado_reproducible_y_estratificado": ("Bootstrap pareado reproducible (semilla 42) y estratificado por ventana",
                                                            "Paired bootstrap is reproducible (seed 42) and stratified by window"),
    "test_regla_decision": ("Regla de decisión de §1 en casos sintéticos", "§1 decision rule on synthetic cases"),
    "test_mejor_referencia_y_conjunta": ("Elección de ref* y conclusión conjunta", "Choice of ref* and joint conclusion"),
    "test_mascara_shock_usa_historia_hasta_origen": ("Umbral de shock calculado solo con la historia hasta el origen",
                                                     "Shock threshold computed from history up to the origin only"),
    "test_analisis_humo": ("Análisis completo con pronósticos sintéticos; falla si falta un pronóstico",
                           "Full analysis on synthetic forecasts; fails if a forecast is missing"),
}


def tabla_pruebas(forzar):
    xml = DATOS / "pytest_junit.xml"
    if not xml.exists():
        raise SystemExit(f"Falta {xml}: ejecute antes .venv311\\Scripts\\python -m pytest -q tests --junitxml {xml}")
    filas = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        nombre = tc.get("name")
        base = nombre.split("[")[0]
        res = "fallo" if tc.find("failure") is not None or tc.find("error") is not None else (
            "omitida" if tc.find("skipped") is not None else "pasa")
        es, en = DESCRIPCION_PRUEBAS.get(base, ("", ""))
        filas.append({"prueba": nombre, "comprueba_es": es, "comprueba_en": en, "resultado": res,
                      "segundos": float(tc.get("time"))})
    t = pd.DataFrame(filas)
    TABLAS.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLAS / "t2_pruebas.csv", index=False)
    col = "comprueba_es" if IDIOMA == "es" else "comprueba_en"
    res = {"pasa": "pasa", "fallo": "falla", "omitida": "omitida"} if IDIOMA == "es" else \
        {"pasa": "pass", "fallo": "fail", "omitida": "skipped"}
    cab = r"Prueba & Qué comprueba & Resultado \\" if IDIOMA == "es" else r"Test & What it checks & Result \\"
    lin = [r"\begin{tabular}{p{5.2cm}p{8cm}l}", r"\toprule", cab, r"\midrule"]
    for _, r in t.iterrows():
        lin.append(f"\\texttt{{{_tex(r.prueba)}}} & {_tex(r[col])} & {res[r.resultado]} \\\\")
    lin += [r"\bottomrule", r"\end{tabular}"]
    (TABLAS / f"t2_pruebas.{IDIOMA}.tex").write_text("\n".join(lin) + "\n", encoding="utf-8")


def _tex(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tabla_hashes(forzar):
    import importlib.util
    spec = importlib.util.spec_from_file_location("h04", RAIZ / "scripts" / "04_hashes_protocolo.py")
    h04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h04)
    congelado = json.loads((RAIZ / "docs" / "hashes_protocolo.json").read_text(encoding="utf-8"))
    filas = []
    for grupo, arch in congelado["archivos"].items():
        for ruta, info in arch.items():
            actual = h04.sha256_lf(RAIZ / ruta)
            filas.append({"grupo": grupo, "archivo": ruta, "sha256_congelado": info["sha256"],
                          "sha256_actual": actual, "coincide": actual == info["sha256"]})
    t = pd.DataFrame(filas)
    TABLAS.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLAS / "t2_hashes.csv", index=False)
    cab = (r"Archivo & SHA-256 congelado (9ede96c) & SHA-256 actual & Coincide \\" if IDIOMA == "es"
           else r"File & Frozen SHA-256 (9ede96c) & Current SHA-256 & Match \\")
    si = ("sí", "no") if IDIOMA == "es" else ("yes", "no")
    lin = [r"\begin{tabular}{llll}", r"\toprule", cab, r"\midrule"]
    for _, r in t.iterrows():
        lin.append(f"\\texttt{{{_tex(r.archivo)}}} & \\texttt{{{r.sha256_congelado[:12]}…}} & "
                   f"\\texttt{{{r.sha256_actual[:12]}…}} & {si[0] if r.coincide else si[1]} \\\\")
    lin += [r"\bottomrule", r"\end{tabular}"]
    (TABLAS / f"t2_hashes.{IDIOMA}.tex").write_text("\n".join(lin) + "\n", encoding="utf-8")
    if not t.coincide.all():
        print("ATENCIÓN: hay hashes que no coinciden")


# ============================================================================= T3
def f05_perturbacion(forzar):
    if hecho("f05_perturbacion", forzar):
        return
    ruta = DATOS / "perturbacion.csv"
    if not ruta.exists():
        print("  falta", ruta, "(ejecute scripts/06_perturbacion.py con .venv311)")
        return
    t = pd.read_csv(ruta)
    sel = ["escalador_y (media, escala)", "secuencias_entrenamiento", "objetivos_entrenamiento",
           "secuencias_validacion", "objetivos_validacion", "entrada_primer_paso_prueba",
           "pronostico_primer_paso_prueba (perturbada vs original)",
           "pronostico_primer_paso_prueba (original vs publicado dia 3)",
           "observado_primer_paso_prueba", "entrada_ultimo_paso_prueba",
           "pronosticos_prueba_resto (perturbada vs original)"]
    nombres = {
        "es": ["Escalador (media y escala de y)", "Secuencias de entrenamiento", "Objetivos de entrenamiento",
               "Secuencias de validación", "Objetivos de validación", "Entrada del 1.er paso de prueba",
               "Pronóstico del 1.er paso de prueba", "Pronóstico 1.er paso: original vs publicado (día 3)",
               "Valor observado del 1.er paso (t)", "Entrada del último paso de prueba",
               "Pronósticos del resto de la prueba (t)"],
        "en": ["Scaler (mean and scale of y)", "Training sequences", "Training targets", "Validation sequences",
               "Validation targets", "Input of first test step", "Forecast of first test step",
               "First-step forecast: original vs published (day 3)", "Observed value of first step (t)",
               "Input of last test step", "Forecasts for the rest of the test year (t)"]}[IDIOMA]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, b in zip(axs, "DM"):
        s = t[t.brazo == b].set_index("comparacion").loc[sel, "dif_max_abs"].to_numpy()
        y = np.arange(len(sel))[::-1]
        col = [COLOR["SARIMA"] if v == 0 else COLOR["DualLSTM"] for v in s]
        ax.scatter(np.where(s == 0, 1e-3, s), y, c=col, s=28, zorder=3)
        for yi, v in zip(y, s):
            ax.annotate(L["identico"] if v == 0 else (f"{v:.0f}" if v >= 10 else f"{v:.3g}"), (max(v, 1e-3), yi), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7)
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_xlim(-0.001, 1e5)
        ax.axhline(y[7] - 0.5, color="0.6", lw=0.6, ls="--")
        ax.set_yticks(y, nombres)
        ax.set_title(L[f"brazo_{b}"])
    fig.supxlabel(L["perturb"], fontsize=9)
    fig.tight_layout()
    guardar(fig, "f05_perturbacion")


# ============================================================================= T4
def _estados(b):
    e = _csv(b, "principal", "estados_redes")
    e = e[e.h == 1].copy()
    e["N"] = e.N.astype(str)
    return e


def f06_epocas(forzar):
    if hecho("f06_epoca_parada", forzar):
        return
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2))
    filas = []
    for j, b in enumerate("DM"):
        e = _estados(b)
        for i, mod in enumerate(("DualLSTM", "LSTM univariada")):
            ax = axs[i, j]
            datos, datos2 = [], []
            for N in _orden_N(b):
                s = e[(e.modelo == mod) & (e.N == N)]
                datos.append(s.mejor_epoca.to_numpy())
                datos2.append(s.epocas.to_numpy())
                filas.append({"brazo": b, "modelo": mod, "N": N, "n_unidades": len(s),
                              "mejor_epoca_mediana": s.mejor_epoca.median(), "mejor_epoca_min": s.mejor_epoca.min(),
                              "mejor_epoca_max": s.mejor_epoca.max(), "epocas_mediana": s.epocas.median(),
                              "epocas_min": s.epocas.min(), "epocas_max": s.epocas.max()})
            pos = np.arange(len(datos))
            ax.boxplot(datos, positions=pos - 0.17, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor=COLOR[mod], alpha=0.6), medianprops=dict(color="k"), flierprops=dict(ms=2))
            ax.boxplot(datos2, positions=pos + 0.17, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=COLOR[mod]), medianprops=dict(color=COLOR[mod]),
                       flierprops=dict(ms=2))
            ax.set_xticks(pos, [("máx" if IDIOMA == "es" else "max") if n == "max" else n for n in _orden_N(b)])
            ax.set_title(f"{L[f'brazo_{b}'].split(' (')[0]} — {nm(mod)}")
            ax.set_xlabel(L[f"N_{b}"])
            ax.set_ylabel(L["epoca_parada"] if j == 0 else "")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="0.5", alpha=0.6, label=L["mejor_epoca"]),
                        Patch(facecolor="white", edgecolor="0.3", label=L["epoca_parada"])],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pd.DataFrame(filas).to_csv(DATOS / "f06_epocas.csv", index=False, float_format="%.6g")
    guardar(fig, "f06_epoca_parada")


def _mae_semillas():
    filas = []
    for b in "DM":
        df = _serie(b)
        ref = fechas_evaluables(df, list(C.ANIOS_TEST))
        fe = set(ref.fecha)
        p = _pron(b)
        for N in _orden_N(b):
            for mod in C.REDES:
                s = p[(p.N == N) & (p.h == 1) & (p.modelo == mod) & p.fecha.isin(fe)]
                g = s.assign(ae=(s.y_pred - s.y_true).abs()).groupby("semilla").ae.agg(["mean", "size"])
                for sem, r in g.iterrows():
                    filas.append({"brazo": b, "N": N, "modelo": mod, "semilla": int(sem), "MAE": r["mean"],
                                  "n": int(r["size"])})
    return pd.DataFrame(filas)


def f07_semillas(forzar):
    if hecho("f07_variabilidad_semillas", forzar):
        return
    t = cache("f07_mae_por_semilla", _mae_semillas)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, b in zip(axs, "DM"):
        d = _delta(b)
        orden = _orden_N(b)
        for k, N in enumerate(orden):
            for dx, mod in ((-0.15, "LSTM univariada"), (0.15, "DualLSTM")):
                v = t[(t.brazo == b) & (t.N == N) & (t.modelo == mod)].MAE.to_numpy()
                ax.scatter(np.full(len(v), k + dx) + np.linspace(-0.05, 0.05, len(v)), v, s=9,
                           color=COLOR[mod], marker=MARCA[mod], label=nm(mod) if k == 0 else None, zorder=3)
            r = d.set_index("N").loc[N]
            ax.hlines(r.MAE_ref, k - 0.35, k + 0.35, color=COLOR[r.ref_estrella], lw=2)
            ax.annotate(nm(r.ref_estrella), (k + 0.36, r.MAE_ref), fontsize=6.5, va="center",
                        color=COLOR[r.ref_estrella])
        ax.set_xticks(range(len(orden)), [("máx" if IDIOMA == "es" else "max") if n == "max" else n for n in orden])
        ax.set_xlim(-0.5, len(orden) - 0.1)
        ax.set_xlabel(L[f"N_{b}"])
        ax.set_ylabel(L["semillas"])
        ax.set_title(L[f"brazo_{b}"])
    axs[0].legend(frameon=False, loc="upper right")
    guardar(fig, "f07_variabilidad_semillas")


# ============================================================================= T5 (usa la serie observada solo para dibujar)
def _linea_tiempo(b, ini, fin):
    df = _serie(b)
    p = _pron(b)
    ref = _ref_max(b)
    s = p[(p.N == "max") & (p.h == 1)]
    s = s[(s.fecha >= ini) & (s.fecha <= fin)]
    dual = s[s.modelo == "DualLSTM"].groupby("fecha").y_pred.mean()
    out = {"DualLSTM": dual, ref: s[s.modelo == ref].set_index("fecha").y_pred,
           "Seasonal Naive": s[s.modelo == "Seasonal Naive"].set_index("fecha").y_pred}
    obs = df.y_eval[(df.index >= ini) & (df.index <= fin)]
    return obs, {k: v.set_axis(pd.to_datetime(v.index)) for k, v in out.items()}, ref


def _dibujar_tiempo(ax, b, ini, fin, marcas=False):
    obs, mods, ref = _linea_tiempo(b, ini, fin)
    ax.plot(obs.index, obs.values, color=COLOR["obs"], lw=1.1, label=L["obs"], marker="o" if marcas else None, ms=2.5)
    ax.plot(mods["Seasonal Naive"].index, mods["Seasonal Naive"].values, color=COLOR["Seasonal Naive"], lw=0.8,
            ls=":", label=L["sn"])
    ax.plot(mods[ref].index, mods[ref].values, color=COLOR[ref], lw=0.9, label=L["ref"].format(r=nm(ref)))
    ax.plot(mods["DualLSTM"].index, mods["DualLSTM"].values, color=COLOR["DualLSTM"], lw=0.9, label=L["dual"])
    ax.set_ylabel(L["t"])
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def f08_d2025(forzar):
    if hecho("f08_D_2025", forzar):
        return
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0))
    _dibujar_tiempo(axs[0], "D", "2025-01-01", "2025-12-31")
    a, z = "2025-01-06", "2025-03-02"     # zoom fijo: las 8 primeras semanas completas (lunes a domingo) de 2025
    axs[0].axvspan(pd.Timestamp(a), pd.Timestamp(z), color="0.9", zorder=0)
    axs[0].set_title(L["brazo_D"] + " — 2025")
    _dibujar_tiempo(axs[1], "D", a, z, marcas=True)
    axs[1].set_title(L["zoom"].format(a=a, b=z))
    axs[0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left")
    fig.tight_layout()
    guardar(fig, "f08_D_2025")


def f09_d2023(forzar):
    if hecho("f09_D_2023_episodio", forzar):
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    _dibujar_tiempo(ax, "D", "2023-02-01", "2023-06-30")
    ax.set_title(L["brazo_D"] + " — 2023-02 / 2023-06")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    guardar(fig, "f09_D_2023_episodio")


def f10_m(forzar):
    if hecho("f10_M_2021_2025", forzar):
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    _dibujar_tiempo(ax, "M", "2021-01-01", "2025-12-31", marcas=True)
    ax.set_title(L["brazo_M"] + " — 2021–2025")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    guardar(fig, "f10_M_2021_2025")


# ============================================================================= T6
def _errores_max():
    """Agregados por día de la semana y mes (N = máx, h = 1) para DualLSTM, ref*(máx) y Seasonal Naive."""
    filas = []
    for b in "DM":
        ref = _ref_max(b)
        mods = ["DualLSTM", ref, "Seasonal Naive"]
        AE, fr = _ae_por_fecha(b, "max", mods)
        f = pd.DatetimeIndex(pd.to_datetime(fr.fecha))
        for i, m in enumerate(mods):
            for tipo, clave in (("dia_semana", f.dayofweek), ("mes", f.month)):
                if b == "M" and tipo == "dia_semana":
                    continue
                g = pd.Series(AE[i]).groupby(np.asarray(clave)).agg(["mean", "size"])
                for k, r in g.iterrows():
                    filas.append({"brazo": b, "modelo": m, "rol": "ref*" if m == ref else m, "tipo": tipo,
                                  "clave": int(k), "MAE": r["mean"], "n": int(r["size"])})
    return pd.DataFrame(filas)


def f11_f12_calendario(forzar):
    t = cache("f11_f12_mae_calendario", _errores_max)
    if not hecho("f11_mae_dia_semana_D", forzar):
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        s = t[(t.brazo == "D") & (t.tipo == "dia_semana")]
        mods = list(dict.fromkeys(s.modelo))
        w = 0.27
        for j, m in enumerate(mods):
            v = s[s.modelo == m].sort_values("clave")
            ax.bar(v.clave + (j - 1) * w, v.MAE, width=w, color=COLOR[m], label=nm(m) + (" (ref*)" if m == _ref_max("D") else ""))
        ax.set_xticks(range(7), L["dias"])
        ax.set_xlabel(L["dia"])
        ax.set_ylabel(L["mae"])
        ax.set_title(L["brazo_D"] + ", N = " + ("máx" if IDIOMA == "es" else "max"))
        ax.legend(frameon=False, fontsize=7)
        guardar(fig, "f11_mae_dia_semana_D")
    if not hecho("f12_mae_mes", forzar):
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
        for ax, b in zip(axs, "DM"):
            s = t[(t.brazo == b) & (t.tipo == "mes")]
            for m in dict.fromkeys(s.modelo):
                v = s[s.modelo == m].sort_values("clave")
                ax.plot(v.clave, v.MAE, marker=MARCA[m], ms=3.5, color=COLOR[m],
                        ls=":" if m == "Seasonal Naive" else "-",
                        label=nm(m) + (" (ref*)" if m == _ref_max(b) else ""))
            ax.set_xticks(range(1, 13), L["meses"], fontsize=7)
            ax.set_xlabel(L["mes"])
            ax.set_ylabel(L["mae"])
            ax.set_title(L[f"brazo_{b}"] + ", N = " + ("máx" if IDIOMA == "es" else "max"))
            ax.legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        guardar(fig, "f12_mae_mes")


def _acf():
    from statsmodels.tsa.stattools import acf
    filas = []
    for b, nl in (("D", 28), ("M", 12)):
        df = _serie(b)
        ref = _ref_max(b)
        p = _pron(b)
        s = p[(p.N == "max") & (p.h == 1)]
        idx = df.index[(df.index >= "2021-01-01") & (df.index <= "2025-12-31")].strftime("%Y-%m-%d")
        for m in ("DualLSTM", ref):
            e = s[s.modelo == m].groupby("fecha").apply(lambda g: (g.y_pred - g.y_true).mean(), include_groups=False)
            r = e.reindex(idx).to_numpy(float)
            v = acf(r, nlags=nl, missing="conservative", fft=False)
            n = int(np.isfinite(r).sum())
            for k, a in enumerate(v):
                filas.append({"brazo": b, "modelo": m, "rol": "ref*" if m == ref else m, "rezago": k, "acf": a,
                              "n": n, "banda95": 1.96 / np.sqrt(n)})
    return pd.DataFrame(filas)


def f13_acf(forzar):
    if hecho("f13_acf_residuos", forzar):
        return
    t = cache("f13_acf_residuos", _acf)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, b in zip(axs, "DM"):
        s = t[(t.brazo == b) & (t.rezago > 0)]
        for j, m in enumerate(dict.fromkeys(s.modelo)):
            v = s[s.modelo == m]
            ax.vlines(v.rezago + (j - 0.5) * 0.3, 0, v.acf, color=COLOR[m], lw=1.6,
                      label=nm(m) + (" (ref*)" if m == _ref_max(b) else ""))
        bd = s.banda95.iloc[0]
        ax.axhspan(-bd, bd, color="0.85", zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(L[f"rezago_{b}"])
        ax.set_ylabel(L["acf"])
        ax.set_title(L[f"brazo_{b}"] + ", N = " + ("máx" if IDIOMA == "es" else "max"))
        ax.legend(frameon=False, fontsize=7)
        ax.set_xticks(range(0, 29, 7) if b == "D" else range(1, 13))
    fig.tight_layout()
    guardar(fig, "f13_acf_residuos")


def _shocks():
    filas = []
    for b in "DM":
        df = _serie(b)
        ref = _ref_max(b)
        mods = ["DualLSTM", ref, "Seasonal Naive"]
        AE, fr = _ae_por_fecha(b, "max", mods)
        estr = fr.anio.to_numpy()
        for P in C.UMBRALES_P:
            mk = np.zeros(len(fr), bool)
            for a in C.ANIOS_TEST:
                k = estr == a
                if not k.any():
                    continue
                p = particion(df.index, Ventana(a), "max")
                mk[k], _ = E.mascara_shock(df, b, p.val[-1], fr.pos.to_numpy()[k], P)
            for i, m in enumerate(mods):
                for grupo, sel in (("shock", mk), ("normal", ~mk)):
                    filas.append({"brazo": b, "P": P, "modelo": m, "rol": "ref*" if m == ref else m, "grupo": grupo,
                                  "MAE": AE[i, sel].mean() if sel.any() else np.nan, "n": int(sel.sum())})
    return pd.DataFrame(filas)


def f14_shocks(forzar):
    if hecho("f14_shock_vs_normal", forzar):
        return
    t = cache("f14_shock_vs_normal", _shocks)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, b in zip(axs, "DM"):
        s = t[t.brazo == b]
        mods = list(dict.fromkeys(s.modelo))
        x = np.arange(len(C.UMBRALES_P))
        w = 0.13
        for j, m in enumerate(mods):
            for g, (hatch, alpha) in enumerate(((None, 1.0), ("//", 0.35))):
                grupo = ("shock", "normal")[g]
                v = s[(s.modelo == m) & (s.grupo == grupo)].set_index("P").loc[list(C.UMBRALES_P)]
                ax.bar(x + (j * 2 + g - 2.5) * w, v.MAE, width=w, color=COLOR[m], alpha=alpha, hatch=hatch,
                       edgecolor=COLOR[m], label=f"{nm(m)} — {L['shock'].split(' (')[0] if g == 0 else L['normal']}")
        ax.set_xticks(x, [f"P{p}" for p in C.UMBRALES_P])
        ax.set_xlabel(L["umbral"])
        ax.set_ylabel(L["mae"])
        ax.set_title(L[f"brazo_{b}"] + ", N = " + ("máx" if IDIOMA == "es" else "max"))
    hs, ls = [], []
    for ax in axs:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_)
                ls.append(l_)
    fig.legend(hs, ls, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.99, 0.95))
    fig.tight_layout()
    guardar(fig, "f14_shock_vs_normal")


# ============================================================================= T7
def _dm_pares():
    filas = []
    for b in "DM":
        for N in _orden_N(b):
            AE, fr = _ae_por_fecha(b, N, C.MODELOS)
            i_d = C.MODELOS.index("DualLSTM")
            for r in C.REFERENCIAS:
                i_r = C.MODELOS.index(r)
                dm = E.dm_hln(AE[i_d], AE[i_r], 1)
                p = 2 * stats.t.sf(abs(dm["HLN"]), df=dm["n"] - 1)    # cola con sf: sin subdesbordamiento
                filas.append({"brazo": b, "N": N, "referencia": r, "d_media": dm["d_media"], "HLN": dm["HLN"],
                              "p": p, "n": dm["n"]})
    return pd.DataFrame(filas)


def f15_dm(forzar):
    t = cache("f15_dm_pares", _dm_pares)
    if not hecho("f15_dm_heatmap", forzar):
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
        cmap = plt.get_cmap("cividis_r")
        for ax, b in zip(axs, "DM"):
            s = t[t.brazo == b]
            orden = _orden_N(b)
            M = np.array([[s[(s.N == N) & (s.referencia == r)].p.iloc[0] for N in orden] for r in C.REFERENCIAS])
            D = np.array([[s[(s.N == N) & (s.referencia == r)].d_media.iloc[0] for N in orden] for r in C.REFERENCIAS])
            Z = -np.log10(np.clip(M, 1e-300, 1))
            im = ax.imshow(np.minimum(Z, 10), cmap=cmap, vmin=0, vmax=10, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    txt = ("<1e-10" if M[i, j] < 1e-10 else f"{M[i, j]:.2g}") + ("\n+" if D[i, j] > 0 else "\n−")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="white" if Z[i, j] > 6 else "black")
            ax.set_xticks(range(len(orden)), [("máx" if IDIOMA == "es" else "max") if n == "max" else n for n in orden])
            ax.set_yticks(range(len(C.REFERENCIAS)), [nm(r) for r in C.REFERENCIAS] if b == "D" else [])
            ax.set_xlabel(L[f"N_{b}"])
            ax.set_title(L[f"brazo_{b}"])
        cb = fig.colorbar(im, ax=axs, shrink=0.85)
        cb.set_label("−log10(p) (" + ("tope 10" if IDIOMA == "es" else "capped at 10") + ")")
        fig.suptitle(L["dm_titulo"] + ("   (+: " + L["peor"] + "; −: " + L["mejor"] + ")"), fontsize=8, y=1.02)
        guardar(fig, "f15_dm_heatmap")
    # tabla consistently top-ranked: principal y sens. 5 (sens. 5 calculada con bootstrap iid, D9)
    filas = []
    for b in "DM":
        for esc in ("principal", "sens5_bloques"):
            s = _csv(b, esc, "top_ranked")
            for _, r in s.iterrows():
                filas.append({"brazo": b, "escenario": esc, "bootstrap_top_ranked": "iid",
                              "nota": "sens5: tabla calculada con bootstrap iid (D9)" if esc != "principal" else "",
                              **r.to_dict()})
    tt = pd.DataFrame(filas)
    TABLAS.mkdir(parents=True, exist_ok=True)
    tt.to_csv(TABLAS / "t7_top_ranked.csv", index=False)
    es = IDIOMA == "es"
    lin = [r"\begin{tabular}{llcccc}", r"\toprule",
           (r"Brazo & Modelo & \multicolumn{4}{c}{Umbrales en que es primero o empatado (de 4); $\checkmark$ = consistently top-ranked} \\"
            if es else r"Arm & Model & \multicolumn{4}{c}{Thresholds where first or tied (of 4); $\checkmark$ = consistently top-ranked} \\")]
    for b in "DM":
        orden = _orden_N(b)
        lin.append(r"\midrule")
        lin.append(f" & & " + " & ".join(f"$N$ = {('máx' if es else 'max') if n == 'max' else n}" for n in orden) + r" \\")
        s = tt[(tt.brazo == b) & (tt.escenario == "principal")]
        for m in C.MODELOS:
            cel = []
            for N in orden:
                r = s[(s.modelo == m) & (s.N.astype(str) == N)].iloc[0]
                cel.append(f"{int(r.umbrales_primero)}" + (r" $\checkmark$" if r.consistently_top_ranked else ""))
            lin.append(f"{b} & {nm(m)} & " + " & ".join(cel) + r" \\")
    lin += [r"\bottomrule", r"\end{tabular}",
            ("% Sensibilidad 5: tabla calculada con bootstrap iid (D9); idéntica a la principal." if es else
             "% Sensitivity 5: table computed with iid bootstrap (D9); identical to the main analysis.")]
    (TABLAS / f"t7_top_ranked.{IDIOMA}.tex").write_text("\n".join(lin) + "\n", encoding="utf-8")


# ============================================================================= T8
def f16_m120(forzar):
    if hecho("f16_M_N120_vs_max", forzar):
        return
    d = _delta("M").set_index("N").loc[["120", "max"]]
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.8))
    for ax, (col, lo, hi, lab) in zip(axs, (("delta", "ic_lo", "ic_hi", "delta"),
                                            ("delta_pct", "ic_lo_pct", "ic_hi_pct", "delta_pct"))):
        x = np.arange(2)
        ax.bar(x, d[col], color=COLOR["DualLSTM"], alpha=0.8, width=0.5)
        ax.errorbar(x, d[col], yerr=[d[col] - d[lo], d[hi] - d[col]], fmt="none", ecolor="k", capsize=4)
        for xi, (_, r) in zip(x, d.iterrows()):
            ax.annotate(f"ref* = {nm(r.ref_estrella)}", (xi, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color="white")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, ["N = 120", "N = " + ("máx" if IDIOMA == "es" else "max")])
        ax.set_ylabel(L[lab])
    fig.suptitle(L["brecha_120_max"], fontsize=9)
    fig.tight_layout()
    guardar(fig, "f16_M_N120_vs_max")


def f17_sisagri(forzar):
    if hecho("f17_sisap_vs_sisagri", forzar):
        return
    m = pd.read_csv(RAIZ / "reportes" / "dia2_mensual_vs_sisap.csv")
    m = m[(m.anio >= 2016) & (m.anio <= 2025)]
    if not SISAGRI_CSV.exists():
        print("  falta", SISAGRI_CSV)
        return
    sg = pd.read_csv(SISAGRI_CSV, dtype={"mes": int}).rename(columns={"año": "anio"})
    c = m.merge(sg[["anio", "mes", "produccion_t_nacional"]], on=["anio", "mes"])
    c["f"] = pd.to_datetime(dict(year=c.anio, month=c.mes, day=1))
    c = c.sort_values("f")
    corr = pd.read_csv(RAIZ / "reportes" / "dia2_correlacion_sisagri.csv")
    r = corr.query("muestra == 'solo_meses_completos' and transformacion == 'niveles' and rezago_sisagri_meses == 0").iloc[0]
    rd = corr.query("muestra == 'solo_meses_completos' and transformacion == 'dif_mensual' and rezago_sisagri_meses == 0").iloc[0]
    gm = c.vol_total_sisap.where(c.flag_faltante == 0)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(c.f, 100 * gm / gm.mean(), color=COLOR["ETS"], lw=1.3, label=L["gmml"])
    ax.plot(c.f, 100 * c.produccion_t_nacional / c.produccion_t_nacional.mean(), color=COLOR["LSTM univariada"],
            lw=1.3, label=L["sisagri"])
    ax.set_ylabel(L["indice"])
    dec = "," if IDIOMA == "es" else "."
    ax.set_title(f"r = {r.pearson_r:.2f}".replace(".", dec) + f" (n = {int(r.n)}); "
                 + ("diferencias" if IDIOMA == "es" else "differences") + f": r = {rd.pearson_r:.2f}".replace(".", dec))
    ax.legend(frameon=False, fontsize=7)
    guardar(fig, "f17_sisap_vs_sisagri")


def _tiempos_unidad():
    filas = []
    for b in "DM":
        e = _estados(b)
        for _, r in e.iterrows():
            filas.append({"brazo": b, "modelo": r.modelo, "N": r.N, "anio": r.anio, "semilla": r.semilla,
                          "segundos": r.segundos, "fuente": f"resultados/brazo_{b}/principal/estados_redes.csv"})
    for f in sorted((RES / "brazo_D" / "estadisticos" / "principal" / "unidades").glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        filas.append({"brazo": "D", "modelo": j["modelo"], "N": str(j["N"]), "anio": j["anio"], "semilla": np.nan,
                      "segundos": j["segundos"], "fuente": "resultados/brazo_D/estadisticos/principal/unidades/*.json (h = 1 y 7 juntos)"})
    bt = pd.read_csv(RES / "brazo_M" / "principal" / "base_tiempos.csv", dtype={"N": str})
    for _, r in bt.iterrows():
        filas.append({"brazo": "M", "modelo": r.modelo, "N": r.N, "anio": r.anio, "semilla": np.nan,
                      "segundos": r.segundos, "fuente": "resultados/brazo_M/principal/base_tiempos.csv"})
    return pd.DataFrame(filas)


def f18_tiempos(forzar):
    if hecho("f18_tiempos", forzar):
        return
    t = cache("f18_tiempos_por_unidad", _tiempos_unidad)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, b in zip(axs, "DM"):
        xs, _ = _x_N(b)
        for m in C.MODELOS:
            s = t[(t.brazo == b) & (t.modelo == m)].groupby("N").segundos.median().reindex(_orden_N(b))
            ax.plot([xs[n] for n in s.index], np.maximum(s.values, 0.01), marker=MARCA[m], ms=4, color=COLOR[m],
                    label=nm(m))
        eje_N(ax, b)
        ax.set_yscale("log")
        ax.set_xlabel(L[f"N_{b}"])
        ax.set_ylabel(L["segundos"])
        ax.set_title(L[f"brazo_{b}"].split(",")[0] + (")" if "(" in L[f"brazo_{b}"] else ""))
    axs[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    guardar(fig, "f18_tiempos")


# ============================================================================= principal
TAREAS = [("T1", lambda f: (f01_curva(f), f02_f03_brecha(f), tablas_t1(f))),
          ("T2", lambda f: (f04_rolling(f), tabla_pruebas(f), tabla_hashes(f))),
          ("T3", f05_perturbacion), ("T4", lambda f: (f06_epocas(f), f07_semillas(f))),
          ("T5", lambda f: (f08_d2025(f), f09_d2023(f), f10_m(f))),
          ("T6", lambda f: (f11_f12_calendario(f), f13_acf(f), f14_shocks(f))),
          ("T7", f15_dm), ("T8", lambda f: (f16_m120(f), f17_sisagri(f), f18_tiempos(f)))]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idioma", choices=("en", "es"), required=True)
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--solo", nargs="*", help="T1 ... T8")
    a = ap.parse_args()
    IDIOMA = a.idioma
    L.update(TXT[IDIOMA])
    for nombre, f in TAREAS:
        if a.solo and nombre not in a.solo:
            continue
        print(nombre)
        f(a.forzar)
