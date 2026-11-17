"""Genera reportes/brazo_D.md solo a partir de los CSV de resultados/brazo_D/ (cada tabla cita su ruta).
Las secciones de texto libre (desviaciones, notas) están al final de este archivo, en PLANTILLA_*."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
B = RAIZ / "resultados" / "brazo_D"
REL = "resultados/brazo_D"
MODELOS = ["Seasonal Naive", "ETS", "Theta", "SARIMA", "Chronos-Bolt-tiny", "LSTM univariada", "DualLSTM"]
NS = ["120", "360", "1000", "max"]


def f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "sí" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float):
        if x != 0 and abs(x) < 1e-3:
            return f"{x:.1e}"
        return f"{x:.{d}f}"
    return str(x)


def md(df, d=2):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c], d) for c in cols) + " |")
    return "\n".join(out)


def rd(carpeta, nombre):
    return pd.read_csv(B / carpeta / f"{nombre}.csv", dtype={"N": str})


def curva(carpeta, metrica="MAE", ic=False):
    m = rd(carpeta, "metricas")
    m = m[m.anio.astype(str) == "pool"]
    filas = []
    for mod in MODELOS:
        fila = {"Modelo": mod}
        for N in NS:
            r = m[(m.N == N) & (m.modelo == mod)].iloc[0]
            fila[f"N={N}"] = (f"{r.MAE:.1f} [{r.MAE_ic_lo:.1f}; {r.MAE_ic_hi:.1f}]" if ic else f(float(r[metrica]), 3 if metrica == "MASE" else 1))
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_delta(carpeta):
    d = rd(carpeta, "delta")
    return pd.DataFrame({"N": d.N, "ref*(N)": d.ref_estrella, "MAE DualLSTM": d.MAE_DualLSTM, "MAE ref*": d.MAE_ref,
                         "Δ (t)": d.delta, "IC 95 % Δ": [f"[{a:.1f}; {b:.1f}]" for a, b in zip(d.ic_lo, d.ic_hi)],
                         "significativa": d.significativa, "HLN": d.HLN, "p DM-HLN": d.p_DM, "n": d.n})


def etiqueta(carpeta):
    d = rd(carpeta, "decision").iloc[0]
    return d


def txt_etiqueta(carpeta, nombre):
    d = etiqueta(carpeta)
    return (f"**{nombre}: «{d.etiqueta}»** (detalle de `decision.csv`: «{d.detalle}»; monótona decreciente = {f(bool(d.monotona_decreciente))}; "
            f"Δ(máx) = {float(d.delta_max):.2f} t; significativa en máx = {f(bool(d.significativa_max))}; "
            f"alguna Δ significativamente positiva = {f(bool(d.alguna_significativa_positiva))}; h = {d.h}; años {d.anios}; bootstrap {d.bootstrap}). "
            f"Fuente: `{REL}/{carpeta}/decision.csv`.")


def tabla_ranking(carpeta):
    r = rd(carpeta, "ranking")
    t = rd(carpeta, "top_ranked")
    filas = []
    for mod in MODELOS:
        fila = {"Modelo": mod}
        for N in NS:
            a = r[(r.N == N) & (r.modelo == mod)].iloc[0]
            b = t[(t.N == N) & (t.modelo == mod)].iloc[0]
            fila[f"N={N}"] = f"{int(a.posicion)}{'*' if a.primero else ''} / {int(b.umbrales_primero)}{' ✔' if b.consistently_top_ranked else ''}"
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_anios(carpeta):
    m = rd(carpeta, "metricas")
    m = m[m.anio.astype(str) != "pool"]
    filas = []
    for mod in MODELOS:
        for N in NS:
            fila = {"Modelo": mod, "N": N}
            for a in sorted(m.anio.unique(), key=lambda v: int(v)):
                fila[str(a)] = float(m[(m.N == N) & (m.modelo == mod) & (m.anio == a)].MAE.iloc[0])
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_semillas(carpeta):
    s = rd(carpeta, "semillas")
    return s[["N", "modelo", "n_semillas", "MAE_media", "MAE_de", "MAE_min", "MAE_max", "MAE_promedio_semillas"]]


def tiempos():
    h = {}
    for log in sorted((B / "logs").glob("est_*.log")):
        for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.startswith("TOTAL"):
                h[log.stem[4:]] = float(ln.split()[1])
    est = pd.DataFrame({"escenario": list(h), "segundos (pared, 2 procesos)": list(h.values())})
    fases = []
    for ln in (B / "logs" / "redes_cadena.log").read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith("FASE_FIN"):
            p = dict(x.split("=") for x in ln.split()[2:] if "=" in x)
            fases.append({"fase": ln.split()[1] + " h=" + p["h"], "unidades": int(p["unidades"]), "fallos": int(p["fallos"]),
                          "segundos (pared, 4 procesos)": float(p["segundos"])})
    return est, pd.DataFrame(fases)


def main():
    S = []
    add = S.append
    add("# Brazo D (diario) — resultados del día 3\n")
    add("Todas las cifras salen de CSV de `resultados/brazo_D/` (ruta citada en cada tabla). Los pronósticos no llevan `y_true`: "
        "`src.analisis.completar_y_true` lo toma de `data/processed` al analizar. Este informe lo genera "
        "`resultados/brazo_D/scripts/generar_informe.py`.\n")
    # ---------------------------------------------------------------- 1
    add("## 1. Qué se ejecutó y tiempo de cómputo\n")
    add(PLANTILLA_EJECUCION)
    est, fases = tiempos()
    add("\n**Modelos estadísticos y Chronos** (SN, ETS, Theta, SARIMA, Chronos; 100 unidades por escenario = 5 años × 4 N × 5 modelos), "
        f"tiempos de pared de cada escenario, de `{REL}/logs/est_*.log`:\n")
    add(md(est, 0))
    add(f"\n**Redes** (LSTM univariada y DualLSTM; 10 semillas), tiempos de pared de cada fase, de `{REL}/logs/redes_cadena.log`:\n")
    add(md(fases, 0))
    tu = pd.read_csv(B / "tiempos_una_ventana.csv")
    add(f"\nPrueba previa de una ventana (Y = 2025; h = 1 y 7 juntos), `{REL}/tiempos_una_ventana.csv`:\n")
    add(md(tu, 1))
    er = pd.read_csv(B / "principal" / "estados_redes.csv", dtype={"N": str})
    add(f"\nSegundos de entrenamiento de las redes (suma de `segundos` de cada unidad, `{REL}/principal/estados_redes.csv`, h = 1 y 7): "
        f"{er.segundos.sum()/3600:.2f} h de CPU de proceso en {len(er)} unidades.\n")
    # ---------------------------------------------------------------- 2
    add("## 2. Curva de aprendizaje a h = 1 (escenario principal, pool 2021–2025)\n")
    add(f"MAE en toneladas/día con IC 95 % (bootstrap pareado iid estratificado por año, B = 10 000, semilla 42). Fuente: `{REL}/principal/metricas.csv` (anio = pool).\n")
    add(md(curva("principal", ic=True)))
    add(f"\nRMSE (t). Fuente: `{REL}/principal/metricas.csv`.\n")
    add(md(curva("principal", "RMSE")))
    add(f"\nMASE (escala fija por ventana: Seasonal Naive m = 7 en la historia máxima). Fuente: `{REL}/principal/metricas.csv`.\n")
    add(md(curva("principal", "MASE")))
    # ---------------------------------------------------------------- 3
    add("\n## 3. ref*(N) y Δ(N) a h = 1\n")
    add(f"Δ(N) = MAE_DualLSTM − MAE_ref*(N). Significativa si el IC 95 % del bootstrap pareado excluye 0. DM-HLN: prueba secundaria. Fuente: `{REL}/principal/delta.csv`.\n")
    add(md(tabla_delta("principal")))
    # ---------------------------------------------------------------- 4
    add("\n## 4. Etiqueta de la regla de decisión (brazo D, h = 1)\n")
    add(txt_etiqueta("principal", "Etiqueta del brazo D"))
    add("\nLa conclusión conjunta con el brazo M no se calcula aquí.\n")
    # ---------------------------------------------------------------- 5
    add("## 5. Ranking, «consistently top-ranked», años y semillas (h = 1)\n")
    add(f"Cada celda: posición por MAE en el pool (`*` = primero o empatado con el primero, IC 95 % de la diferencia con el mejor contiene 0) / "
        f"número de umbrales P50, P60, P75, P90 en que el modelo es primero o empata (`✔` = «consistently top-ranked», ≥ 3 de 4). "
        f"Fuente: `{REL}/principal/ranking.csv` y `{REL}/principal/top_ranked.csv` (detalle por umbral: `top_ranked_detalle.csv`).\n")
    add(md(tabla_ranking("principal")))
    add(f"\nMAE por año de prueba (t). Fuente: `{REL}/principal/metricas.csv`.\n")
    add(md(tabla_anios("principal"), 1))
    add(f"\nDispersión entre semillas de las redes (MAE de cada semilla sobre los días evaluables; `MAE_promedio_semillas` es el MAE del error promediado entre semillas que usa el bootstrap). Fuente: `{REL}/principal/semillas.csv`.\n")
    add(md(tabla_semillas("principal"), 2))
    # ---------------------------------------------------------------- 6
    add("\n## 6. Sensibilidades (no cambian la etiqueta principal)\n")
    for carpeta, titulo, nota in SENS:
        if not (B / carpeta / "decision.csv").exists():
            add(f"### {titulo}\n\n_No disponible (no se ejecutó)._\n")
            continue
        add(f"### {titulo}\n")
        add(nota + "\n")
        add(txt_etiqueta(carpeta, f"Etiqueta ({carpeta})") + "\n")
        add(f"Δ(N). Fuente: `{REL}/{carpeta}/delta.csv`.\n")
        add(md(tabla_delta(carpeta)))
        add(f"\nMAE por N y modelo (t; pool). Fuente: `{REL}/{carpeta}/metricas.csv`.\n")
        add(md(curva(carpeta, ic=True)))
        add("")
    # ---------------------------------------------------------------- 7
    add("\n## 7. Incidencias\n")
    add(incidencias_texto())
    # ---------------------------------------------------------------- 8
    add("\n## 8. Desviaciones propuestas\n")
    add(PLANTILLA_DESVIACIONES)
    out = RAIZ / "reportes" / "brazo_D.md"
    out.write_text("\n".join(S) + "\n", encoding="utf-8")
    print("escrito", out)


SENS = [
    ("sens1_estricto", "Sensibilidad 1: «...» como faltante (`vol_total_estricto`)", "1 352 días evaluables en 2021–2025; h = 1."),
    ("sens2_bolsa", "Sensibilidad 2: objetivo `vol_bolsa`", "1 815 días evaluables; h = 1."),
    ("sens3_h7", "Sensibilidad 3: h = 7 (estrategia directa en las redes)", "Escenario principal; los pronósticos están en `principal/pronosticos.csv` (columna `h` = 7); DM-HLN con autocovarianzas hasta h − 1 = 6."),
    ("sens4_pool2023_2025", "Sensibilidad 4: pool de prueba 2023–2025", "Escenario principal, h = 1, sin reentrenar; umbrales de shock solo con esos años."),
    ("sens5_bloques", "Sensibilidad 5: bootstrap por bloques de 7 días", "Escenario principal, h = 1, sin reentrenar; bloques móviles circulares dentro de cada año."),
    ("sens6_desde2006", "Sensibilidad 6: historia máxima desde 2006-01-01", "h = 1. La escala MASE es la de la historia máxima desde 2006."),
    ("sens7_anomalia_publicada", "Sensibilidad 7: anomalía de 2021-02-01 con el valor publicado (2 351 t)", "h = 1; todos los modelos reentrenados con la serie con el valor publicado."),
]


def incidencias_texto():
    t = []
    for carpeta in ("principal", "sens1_estricto", "sens2_bolsa", "sens6_desde2006", "sens7_anomalia_publicada"):
        p = B / carpeta / "incidencias.csv"
        if not p.exists():
            continue
        i = pd.read_csv(p, dtype={"N": str})
        e = i[i.modelo == "ETS"]
        s = i[i.modelo == "SARIMA"]
        c = i[i.modelo == "Chronos-Bolt-tiny"]
        fila = {"escenario": carpeta,
                "ETS: ajustes finales sin convergir": f"{int((e.convergio == False).sum())} de {len(e)}",  # noqa: E712
                "ETS: especificaciones no convergidas (de 12 por unidad)": int(e.n_no_convergidas.sum()),
                "ETS: especificaciones que fallaron": int(e.n_fallidas.sum()),
                "SARIMA: sin convergir": f"{int((s.convergio == False).sum())} de {len(s)}",  # noqa: E712
                "Chronos: unidades con truncamiento": f"{int((c.n_truncados_h1 > 0).sum())} de {len(c)} (todas con N = max)" if len(c) else ""}
        t.append(fila)
    r = [f"Resumen por escenario. Fuente: `{REL}/<escenario>/incidencias.csv` (detalle por modelo, año y N).\n", md(pd.DataFrame(t), 0)]
    i = pd.read_csv(B / "principal" / "incidencias.csv", dtype={"N": str})
    c = i[i.modelo == "Chronos-Bolt-tiny"]
    r.append(f"\n**Chronos.** La longitud máxima de contexto del modelo es {int(c.context_length_modelo.iloc[0])}. Con N = max el contexto "
             f"(desde 1997-10-01, miles de días) se trunca a los últimos {int(c.context_length_modelo.iloc[0])} días en todos los pronósticos "
             f"(h = 1: {int(c.n_truncados_h1.sum())} de {int(c.n_pronosticos_h1.sum())} pronósticos de la principal, todos con N = max; h = 7: {int(c.n_truncados_h7.sum())} de {int(c.n_pronosticos_h7.sum())}). "
             f"Con N = 120, 360 y 1000 no hay truncamiento. Fuente: `{REL}/principal/incidencias.csv`.")
    e = i[i.modelo == "ETS"].copy()
    tab = e.groupby(["N", "especificacion"]).size().rename("ventanas").reset_index()
    r.append(f"\n**ETS: especificación elegida por AICc** (número de ventanas de 5 por N). Fuente: `{REL}/principal/incidencias.csv`.\n")
    r.append(md(tab, 0))
    filas = []
    for carpeta in ("principal", "sens1_estricto", "sens2_bolsa", "sens6_desde2006", "sens7_anomalia_publicada"):
        m = pd.read_csv(B / carpeta / "metricas.csv", dtype={"N": str})
        m = m[(m.modelo == "ETS") & (m.anio.astype(str) != "pool") & (m.MAE > 250)]
        ii = pd.read_csv(B / carpeta / "incidencias.csv", dtype={"N": str})
        ii = ii[ii.modelo == "ETS"].set_index(["anio", "N"])
        for _, x in m.iterrows():
            filas.append({"escenario": carpeta, "año": int(x.anio), "N": x.N, "MAE ETS (t)": float(x.MAE),
                          "especificación elegida": ii.loc[(int(x.anio), x.N), "especificacion"]})
    if filas:
        r.append("\n**ETS inestable en ventanas cortas.** Ventanas (año de prueba) en que el MAE anual del ETS supera 250 t "
                 "(el Seasonal Naive tiene entre 96 y 136 t por año en la principal): los pronósticos son finitos, pero la especificación "
                 f"elegida con N = 120 extrapola mal. Fuente: `{REL}/<escenario>/metricas.csv` e `incidencias.csv`.\n")
        r.append(md(pd.DataFrame(filas), 1))
    s = i[i.modelo == "SARIMA"]
    r.append(f"\n**SARIMA** (principal): {int(s.convergio.sum())} de {len(s)} ajustes convergieron; iteraciones del optimizador entre {int(s.iteraciones.min())} y {int(s.iteraciones.max())} (máximo permitido 500). "
             f"Con N < 730 (N = 120, 360) no se usan Fourier. Fuente: `{REL}/principal/incidencias.csv`.")
    for nombre, ruta, que in (("Equivalencia de los métodos rápidos con los del protocolo", "equivalencias.csv", "diferencia máxima absoluta en t"),
                              ("Sensibilidad 6: comprobación de series idénticas", "sens6_comprobacion.csv", "")):
        d = pd.read_csv(B / ruta)
        r.append(f"\n**{nombre}** (`{REL}/{ruta}`{'; ' + que if que else ''}):\n")
        r.append(md(d, 6))
    est = pd.concat([pd.read_csv(B / c / "estados_redes.csv", dtype={"N": str}).assign(escenario=c)
                     for c in ("principal", "sens1_estricto", "sens2_bolsa", "sens6_desde2006", "sens7_anomalia_publicada") if (B / c / "estados_redes.csv").exists()])
    r.append(f"\n**Redes.** Parámetros de la DualLSTM: {sorted(est[est.modelo == 'DualLSTM'].n_params.unique())}; "
             f"LSTM univariada: {sorted(est[est.modelo == 'LSTM univariada'].n_params.unique())}. "
             f"Unidades de red por escenario (estados_redes.csv de cada carpeta): "
             f"{est.groupby('escenario').size().to_dict()}. Épocas entre {int(est.epocas.min())} y {int(est.epocas.max())} (máximo 300). "
             f"Fuente: `{REL}/<escenario>/estados_redes.csv`.")
    ex = BASE_FALLOS
    r.append(f"\n{ex}")
    return "\n".join(r)


BASE_FALLOS = "Fallos de unidades de red durante las corridas: ver la tabla de fases de la sección 1 (columna `fallos`)."

PLANTILLA_EJECUCION = """Intérprete `.venv311` (Python 3.11.9; versiones en `resultados/brazo_D/versiones.json`, semillas en `resultados/brazo_D/semillas.json`).

- **Seasonal Naive, ETS, Theta, SARIMA y Chronos-Bolt-tiny:** `resultados/brazo_D/scripts/estadisticos_lib.py` y `estadisticos_run.py` (2 procesos). Un ajuste por ventana y N, con la serie `src.datos.serie(esc).y` y las particiones de `src.ventanas`.
- **LSTM univariada y DualLSTM:** `src.redes.correr_unidades` con 4 procesos (`scripts/redes_run.py`, `scripts/cadena_redes.sh`); 10 semillas (0–9); unidades en `resultados/brazo_D/redes_unidades/`.
- **Ensamblado y análisis:** `scripts/ensamblar_D.py` (usa `src.analisis.analizar` y `escribir`); informe: `scripts/generar_informe.py`.
- **Orden:** principal h = 1 → sensibilidades 4 y 5 (sin reentrenar) → sensibilidad 3 (h = 7) → 1 → 2 → 6 → 7."""

PLANTILLA_DESVIACIONES = """Ninguna cambia la etiqueta ni el diseño. Todas se decidieron el 2026-11-17, **antes de ver errores de prueba** (la comprobación de equivalencias se hizo con pronósticos, sin calcular ningún error; los errores se calcularon por primera vez al terminar la corrida principal h = 1 de las redes). No están en `docs/DESVIACIONES.md` (no se modifica `docs/`); YesaR decide si se registran.

1. **ETS y SARIMA: cálculo equivalente, no literal, del «estado actualizado sin reestimar».** El protocolo (§3) pide parámetros fijos y estado actualizado con cada dato. Para no reajustar el modelo 365 veces por ventana se aplica `smooth(params)` una sola vez sobre la serie hasta el final de la prueba y se leen los pronósticos a h pasos desde el estado en τ−h (ETS: `simulate(h, anchor=τ−h)` con errores cero; SARIMA: `get_prediction(dynamic=0)`). Se comprobó contra el procedimiento literal (ETS: `smooth` sobre el prefijo y `forecast(h)`; SARIMA: `extend` y `forecast(h)`) en 3 ventanas, 6 días y h = 1 y 7: diferencia máxima 0 en todas (`resultados/brazo_D/equivalencias.csv`). Efecto: ninguno.
2. **Chronos en lotes.** Los contextos se agrupan en lotes de 64 con relleno por la izquierda. Frente al pronóstico de un solo contexto, la diferencia máxima es de 1,2×10⁻⁴ t (`equivalencias.csv`, fila Chronos). Efecto: despreciable.
3. **Chronos con N = max usa solo los últimos 2 048 días.** Es el caso que el protocolo (§5) dejó como «no comprobado»: la longitud máxima de contexto es 2 048 y con N = max el contexto se trunca en todos los pronósticos (sección 7). Por tanto Chronos con N = max equivale a N = 2 048. No es una desviación, se declara por transparencia.
4. **Theta: clase propia con `period = 7`.** `r2lib.ThetaFijo` del artículo tiene el periodo 12 fijo en el módulo y arrastra rutas de importación del repositorio del artículo; se copió la lógica a `resultados/brazo_D/scripts/estadisticos_lib.py` con periodo 7 y se mantuvieron sus comprobaciones internas (la recursión SES y el pronóstico manual a h = 1 y 7 reproducen `ThetaModelResults`). `ThetaModel` usa los valores por defecto de statsmodels (`deseasonalize`, `method = auto`, prueba de estacionalidad); el método estacional elegido queda en `incidencias.csv`.
5. **ETS: `maxiter = 1000`** (valor por defecto de statsmodels y de `r2lib`). El protocolo no lo fija. Efecto: ninguno sobre la selección, que es por AICc.
6. **SARIMA: pronóstico = exp(media en escala log).** Es la mediana, no la media, de la distribución en toneladas. Regresores de Fourier con `t` = posición del día en la serie del escenario (en la sensibilidad 6 empieza en 2006-01-01); con N < 730 se omiten.
7. **Sensibilidad 6: las redes con N = 120, 360 y 1 000 son las de la principal.** Se comprobó que `serie(esc).y` y `.precio` de la sensibilidad 6 coinciden exactamente con los de la principal desde 2006-01-01 y que historia, validación y prueba son idénticas en cada (Y, N) (`resultados/brazo_D/sens6_comprobacion.csv`). La única diferencia es la variable `t_index` de la rama B (posición en la serie, desplazada por la fecha de inicio), que tras el `StandardScaler` es invariante salvo redondeo; por eso no se reentrenaron esas 300 unidades (solo se reentrenó N = max, 100 unidades). Los modelos estadísticos y Chronos sí se reejecutaron en todos los N: son idénticos a la principal salvo SARIMA con N = 1 000, cuyos Fourier tienen otra fase de origen (diferencia máxima 0,16 t, media 0,007 t; `resultados/brazo_D/sens6_vs_principal.csv`).
8. **Sensibilidades 3, 4 y 5 no duplican `pronosticos.csv`, `incidencias.csv` ni `estados_redes.csv`.** Usan los de `principal/` (h = 1 y 7 en el mismo archivo); cada carpeta tiene un `LEEME.txt`. Las sensibilidades 4 y 5 son solo otro análisis de los mismos pronósticos.
9. **`pronosticos.csv` incluye todos los días de cada año de prueba** (365 o 366), también los no evaluables; `src.analisis.analizar` evalúa solo los 1 818 evaluables del pool principal.
10. **Cómputo.** Las redes se ejecutaron con 4 procesos en una máquina compartida con el brazo M: la DualLSTM con N = max tardó entre 58 y 257 s por unidad (mediana 100 s; `principal/estados_redes.csv`), más que los 60–70 s medidos aisladamente. Los tiempos reales están en la sección 1. Resultados deterministas por unidad (D4), sin efecto.
11. **Error de implementación corregido antes de correr nada.** La primera versión de ETS usaba un ancla de `simulate` desplazada en un día; la comprobación de equivalencias lo detectó (diferencias de cientos de toneladas) y se corrigió antes de lanzar ninguna corrida; el CSV de equivalencias es el de la versión corregida."""


if __name__ == "__main__":
    main()
