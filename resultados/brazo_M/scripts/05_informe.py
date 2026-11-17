"""Genera reportes/brazo_M.md solo a partir de los CSV de resultados/brazo_M/ (ninguna cifra se escribe a mano).

Uso (desde la raíz, con .venv311):  .venv311/Scripts/python.exe resultados/brazo_M/scripts/05_informe.py
Las secciones de texto fijo (qué se ejecutó, desviaciones propuestas) están en `TEXTO_*` al final del archivo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
from src import config as C  # noqa: E402

B = RAIZ / "resultados" / "brazo_M"
REL = "resultados/brazo_M"
ORDEN_N = [str(n) for n in C.TAMANOS["M"]]
ESCENARIOS = {"principal": "Principal", "sens1_estricto": "Sensibilidad 1 («...» = faltante)",
              "sens7_anomalia_publicada": "Sensibilidad 7 (anomalía con el valor publicado)",
              "sens4_pool2023_2025": "Sensibilidad 4 (pool 2023–2025)", "sens5_bloques": "Sensibilidad 5 (bootstrap por bloques de 3 meses)"}


def f(x, d=1):
    return "—" if pd.isna(x) else f"{x:,.{d}f}".replace(",", " ")


def p(x):
    return "—" if pd.isna(x) else (f"{x:.4f}" if x >= 0.0001 else "<0,0001")


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def leer(carpeta: str, t: str) -> pd.DataFrame:
    return pd.read_csv(B / carpeta / f"{t}.csv", dtype={"N": str})


def tabla_mae(m: pd.DataFrame) -> str:
    pool = m[m.anio == "pool"]
    modelos = [x for x in C.MODELOS if x in set(pool.modelo)]
    filas = []
    for mod in modelos:
        fila = {"Modelo": mod}
        for N in ORDEN_N:
            r = pool[(pool.N == N) & (pool.modelo == mod)].iloc[0]
            fila[f"N={N}"] = f"{f(r.MAE)} [{f(r.MAE_ic_lo)}; {f(r.MAE_ic_hi)}]"
        filas.append(fila)
    return md(pd.DataFrame(filas))


def tabla_otra(m: pd.DataFrame, col: str, d: int) -> str:
    pool = m[m.anio == "pool"]
    filas = []
    for mod in [x for x in C.MODELOS if x in set(pool.modelo)]:
        fila = {"Modelo": mod}
        for N in ORDEN_N:
            fila[f"N={N}"] = f(pool[(pool.N == N) & (pool.modelo == mod)][col].iloc[0], d)
        filas.append(fila)
    return md(pd.DataFrame(filas))


def tabla_delta(d: pd.DataFrame) -> str:
    t = pd.DataFrame({"N": d.N, "ref*(N)": d.ref_estrella, "MAE DualLSTM": d.MAE_DualLSTM.map(f), "MAE ref*": d.MAE_ref.map(f),
                      "Δ(N)": d.delta.map(f), "IC 95 % de Δ": [f"[{f(a)}; {f(b)}]" for a, b in zip(d.ic_lo, d.ic_hi)],
                      "IC excluye 0": d.significativa.map({True: "sí", False: "no"}),
                      "DM-HLN (estadístico)": d.HLN.map(lambda x: f(x, 3)), "p DM-HLN": d.p_DM.map(p), "n": d.n})
    return md(t)


def bloque_decision(carpeta: str) -> str:
    d = leer(carpeta, "decision")
    r = d.iloc[0]
    return (f"Etiqueta (`{REL}/{carpeta}/decision.csv`): **«{r.etiqueta}»**. Detalle: {r.detalle}. "
            f"Monótona decreciente: {r.monotona_decreciente}; Δ(máx) = {f(r.delta_max)} t; significativa en N = máx: "
            f"{r.significativa_max}; alguna Δ significativamente positiva: {r.alguna_significativa_positiva}; "
            f"años: {r.anios}; bootstrap: {r.bootstrap}.")


def tabla_ranking(carpeta: str) -> str:
    r = leer(carpeta, "ranking")
    t = leer(carpeta, "top_ranked")
    filas = []
    for mod in [x for x in C.MODELOS]:
        fila = {"Modelo": mod}
        for N in ORDEN_N:
            a = r[(r.N == N) & (r.modelo == mod)].iloc[0]
            b = t[(t.N == N) & (t.modelo == mod)].iloc[0]
            fila[f"N={N}"] = f"pos {a.posicion}{' (primero/empate)' if a.primero else ''}; umbrales {b.umbrales_primero}/4{' TOP' if b.consistently_top_ranked else ''}"
        filas.append(fila)
    return md(pd.DataFrame(filas))


def tabla_anio(m: pd.DataFrame) -> str:
    a = m[m.anio != "pool"].copy()
    filas = []
    for mod in C.MODELOS:
        for N in ORDEN_N:
            fila = {"Modelo": mod, "N": N}
            for y in sorted(a.anio.unique()):
                r = a[(a.modelo == mod) & (a.N == N) & (a.anio == y)]
                fila[str(y)] = f(r.MAE.iloc[0]) if len(r) else "—"
            filas.append(fila)
    return md(pd.DataFrame(filas))


def tabla_semillas(s: pd.DataFrame) -> str:
    t = pd.DataFrame({"Modelo": s.modelo, "N": s.N, "semillas": s.n_semillas, "MAE medio": s.MAE_media.map(f),
                      "DE entre semillas": s.MAE_de.map(f), "mín": s.MAE_min.map(f), "máx": s.MAE_max.map(f),
                      "MAE del promedio de semillas": s.MAE_promedio_semillas.map(f)})
    return md(t)


def resumen_incidencias(carpeta: str) -> str:
    inc = pd.read_csv(B / carpeta / "incidencias.csv", dtype={"N": str})
    out = []
    sel = inc[inc.tipo == "ets_seleccion"]
    tab = sel.pivot_table(index="anio", columns="N", values="valor", aggfunc="first")[ORDEN_N]
    out.append("Especificación ETS elegida por AICc (`incidencias.csv`, tipo `ets_seleccion`):\n\n" + md(tab.reset_index().rename(columns={"anio": "Año"}).rename(columns={n: f"N={n}" for n in ORDEN_N})))
    noconv = sel[sel.detalle.str.contains("convergio=False")]
    out.append(f"\nETS elegida que no convergió: {len(noconv)} de {len(sel)} ventana×N.")
    fall = inc[inc.tipo == "ets_especificacion_fallida"]
    nc = inc[inc.tipo == "ets_especificacion_no_convergio"]
    out.append(f"Especificaciones ETS con fallo de ajuste: {len(fall)} (sobre {12 * len(sel)} ajustes); sin convergencia: {len(nc)}.")
    if len(fall):
        out.append("\n" + md(fall.groupby(["valor"]).size().rename("veces").reset_index().rename(columns={"valor": "Especificación"})))
    if len(nc):
        out.append("\n" + md(nc.groupby(["valor"]).size().rename("veces").reset_index().rename(columns={"valor": "Especificación sin convergencia"})))
    sa = inc[inc.tipo == "sarima_convergencia"]
    out.append(f"\nSARIMA: {int((sa.valor == 'True').sum())} de {len(sa)} ajustes convergieron; "
               f"{int((sa.valor != 'True').sum())} no. Avisos distintos de «ninguno»: {int((~sa.detalle.str.endswith('avisos=ninguno')).sum())}.")
    bad = sa[sa.valor != "True"]
    if len(bad):
        out.append("\n" + md(bad[["anio", "N", "detalle"]].rename(columns={"anio": "Año"})))
    warn = sa[(sa.valor == "True") & ~sa.detalle.str.endswith("avisos=ninguno")]
    if len(warn):
        out.append("\nSARIMA con convergencia pero con avisos:\n\n" + md(warn[["anio", "N", "detalle"]].rename(columns={"anio": "Año"})))
    ch = inc[inc.tipo == "chronos_contexto"]
    out.append(f"\nChronos: {int(ch.detalle.str.contains('truncado=True').sum())} de {len(ch)} ventana×N con contexto truncado "
               f"(contexto máximo solicitado: {ch.valor.astype(int).max()} observaciones; límite del modelo: 2048).")
    tf = inc[inc.tipo.isin(["fallo_total", "ets_pronostico_no_finito"])]
    out.append(f"Fallos totales o pronósticos ETS no finitos: {len(tf)}.")
    return "\n".join(out)


def resumen_redes(carpeta: str) -> str:
    e = pd.read_csv(B / carpeta / "estados_redes.csv", dtype={"N": str})
    g = e.groupby("modelo").agg(unidades=("semilla", "size"), n_params=("n_params", lambda x: ", ".join(map(str, sorted(x.unique())))),
                                epocas_mediana=("epocas", "median"), epocas_max=("epocas", "max"),
                                mejor_epoca_mediana=("mejor_epoca", "median"), segundos_suma=("segundos", "sum")).reset_index()
    g["segundos_suma"] = g.segundos_suma.map(lambda x: f(x, 0))
    return md(g)


def seccion_escenario(carpeta: str, titulo: str, completo: bool) -> str:
    out = [f"### {titulo}\n", bloque_decision(carpeta) + "\n", f"Δ(N) (`{REL}/{carpeta}/delta.csv`), toneladas:\n", tabla_delta(leer(carpeta, "delta")) + "\n"]
    return "\n".join(out)


def main():
    pr = "principal"
    m = leer(pr, "metricas")
    partes = ["# Brazo M (mensual), h = 1: informe del día 3\n",
              "Todas las cifras salen de los CSV de `resultados/brazo_M/` (ruta citada junto a cada tabla). "
              "Este informe lo genera `resultados/brazo_M/scripts/05_informe.py`; la sección 8 contiene cifras escritas a mano (conteos de parámetros, 267–315 meses, 80 unidades, 1,8e-12 t) que no salen de un CSV: la del 1,8e-12 t es una comprobación puntual hecha en una consola (Y = 2023, N = 120) y no está guardada; el resto se puede comprobar en los CSV citados y en `src/`.\n"]
    partes.append(TEXTO_EJECUCION(B))
    partes.append("## 2. Curva de aprendizaje (escenario principal, pool de prueba 2021–2025)\n")
    partes.append(f"MAE en toneladas con IC 95 % del bootstrap iid pareado (B = 10 000, semilla 42) entre corchetes (`{REL}/principal/metricas.csv`, `anio == pool`):\n")
    partes.append(tabla_mae(m) + "\n")
    partes.append(f"RMSE en toneladas (`{REL}/principal/metricas.csv`):\n")
    partes.append(tabla_otra(m, "RMSE", 1) + "\n")
    partes.append(f"MASE, escala fija por ventana (`{REL}/principal/metricas.csv`):\n")
    partes.append(tabla_otra(m, "MASE", 3) + "\n")
    partes.append("## 3. ref*(N), Δ(N) y DM-HLN\n")
    partes.append(f"Δ(N) = MAE_DualLSTM(N) − MAE_ref*(N), IC 95 % del bootstrap pareado y prueba Diebold-Mariano con corrección HLN (`{REL}/principal/delta.csv`):\n")
    partes.append(tabla_delta(leer(pr, "delta")) + "\n")
    partes.append("## 4. Etiqueta de la regla de decisión (escenario principal)\n")
    partes.append(bloque_decision(pr) + "\n")
    partes.append(f"Copia literal de `{REL}/principal/decision.csv`:\n\n```\n{(B / pr / 'decision.csv').read_text(encoding='utf-8').strip()}\n```\n")
    partes.append("## 5. Ranking, «consistently top-ranked», años y semillas\n")
    partes.append(f"Posición por MAE en el pool, si es primero o empata con el primero, y número de umbrales (P50, P60, P75, P90) en que es primero o empata; TOP = «consistently top-ranked» (≥ 3 de 4) (`{REL}/principal/ranking.csv`, `top_ranked.csv`):\n")
    partes.append(tabla_ranking(pr) + "\n")
    partes.append(f"MAE por año de prueba, toneladas (`{REL}/principal/metricas.csv`):\n")
    partes.append(tabla_anio(m) + "\n")
    partes.append(f"Dispersión entre semillas de las redes (`{REL}/principal/semillas.csv`):\n")
    partes.append(tabla_semillas(leer(pr, "semillas")) + "\n")
    partes.append(f"Entrenamiento de las redes (`{REL}/principal/estados_redes.csv`):\n")
    partes.append(resumen_redes(pr) + "\n")
    partes.append("## 6. Sensibilidades (no cambian la etiqueta principal)\n")
    for k, tit in [("sens1_estricto", ESCENARIOS["sens1_estricto"]), ("sens4_pool2023_2025", ESCENARIOS["sens4_pool2023_2025"]),
                   ("sens5_bloques", ESCENARIOS["sens5_bloques"]), ("sens7_anomalia_publicada", ESCENARIOS["sens7_anomalia_publicada"])]:
        partes.append(seccion_escenario(k, tit, True))
        mm = leer(k, "metricas")
        partes.append(f"MAE por modelo y N, pool (`{REL}/{k}/metricas.csv`):\n\n" + tabla_mae(mm) + "\n")
        if k == "sens1_estricto":
            n = mm[(mm.anio != "pool") & (mm.N == "max") & (mm.modelo == "Seasonal Naive")][["anio", "n"]]
            partes.append("Meses evaluables por año (`" + f"{REL}/sens1_estricto/metricas.csv`): " + "; ".join(f"{a}: {b}" for a, b in zip(n.anio, n.n))
                          + ". El año 2025 no tiene meses evaluables y queda fuera del pool (ver D5).\n")
        if k in ("sens1_estricto", "sens7_anomalia_publicada"):
            partes.append(f"Ranking y top-ranked (`{REL}/{k}/ranking.csv`, `top_ranked.csv`):\n\n" + tabla_ranking(k) + "\n")
    partes.append("## 7. Incidencias\n")
    partes.append(f"Principal (`{REL}/principal/incidencias.csv`):\n")
    partes.append(resumen_incidencias(pr) + "\n")
    for k in ("sens1_estricto", "sens7_anomalia_publicada"):
        partes.append(f"\n{ESCENARIOS[k]} (`{REL}/{k}/incidencias.csv`):\n")
        partes.append(resumen_incidencias(k) + "\n")
    partes.append(TEXTO_DESVIACIONES)
    (RAIZ / "reportes" / "brazo_M.md").write_text("\n".join(partes), encoding="utf-8")
    print("escrito reportes/brazo_M.md")


def TEXTO_EJECUCION(B: Path) -> str:
    filas = []
    tot_pared = 0.0
    for k in ("principal", "sens1_estricto", "sens7_anomalia_publicada"):
        e = pd.read_csv(B / k / "estados_redes.csv")
        t = pd.read_csv(B / k / "base_tiempos.csv")
        pw = json.loads((B / k / "redes_tiempo_pared.json").read_text())["corridas"]
        pared = sum(c["segundos_pared"] for c in pw)
        tot_pared += pared
        pron = pd.read_csv(B / k / "pronosticos.csv")
        filas.append({"Escenario": k, "Unidades de red (estado.json)": len(e), "Filas de pronosticos.csv": len(pron),
                      "Segundos de pared de las redes (2 procesos)": f(pared, 0),
                      "Segundos de entrenamiento sumados (estado.json)": f(e.segundos.sum(), 0),
                      "Segundos de los 5 modelos de referencia": f(t.segundos.sum(), 0)})
    tabla = md(pd.DataFrame(filas))
    es = pd.read_csv(B / "principal" / "estados_redes.csv")
    npd = ", ".join(map(str, sorted(es[es.modelo == "DualLSTM"].n_params.unique())))
    npu = ", ".join(map(str, sorted(es[es.modelo == "LSTM univariada"].n_params.unique())))
    return f"""## 1. Qué se ejecutó

Brazo M (mensual), h = 1, años de prueba 2021–2025, N ∈ {{60, 120, 240, max}}, rolling origin de `src.ventanas.particion`. Modelos: Seasonal Naive, ETS (12 especificaciones, menor AICc), Theta (θ = 2), SARIMA (1,1,1)(1,1,0)₁₂ congelado del artículo, Chronos-Bolt-tiny (revisión `{C.CHRONOS_SNAPSHOT}`, cero disparos, cuantil 0,5), LSTM univariada y DualLSTM-Bahdanau (10 semillas, 0–9). Escenarios con modelos reentrenados: principal, sensibilidad 1 y sensibilidad 7; las sensibilidades 4 y 5 reanalizan los pronósticos principales (no tienen `pronosticos.csv` propio). Código: `{REL}/scripts/` (modelos de referencia en `modelos_base.py`; redes con `src.redes.correr_unidades`, n_proc = 2; análisis con `src.analisis`).

Cómputo medido (`{REL}/<escenario>/estados_redes.csv`, `redes_tiempo_pared.json`, `base_tiempos.csv`):

{tabla}

Tiempo de pared total de las redes: {f(tot_pared, 0)} s ({f(tot_pared / 3600, 2)} h). Cada escenario tiene 5 años × 4 N × 2 redes × 10 semillas = 400 unidades. Los tiempos de las redes incluyen la competencia por CPU con el brazo D y el arranque de un proceso por unidad; los de los modelos de referencia se midieron con un proceso de un hilo (Chronos con 2 hilos).

Parámetros de las redes (`{REL}/principal/estados_redes.csv`): DualLSTM {npd}; LSTM univariada {npu}. `expected_parameter_count(4, 6)` da 4545 (ver sección 8, nota 1).
"""


TEXTO_DESVIACIONES = """## 8. Desviaciones propuestas

Ninguna de las decisiones siguientes cambia un valor fijado por `docs/PROTOCOLO.md`; son interpretaciones donde el protocolo es ambiguo o aclaraciones de ejecución. Todas se tomaron el 2026-11-17 antes de lanzar la corrida completa y antes de calcular cualquier error de prueba de la corrida final. Una prueba previa con una sola ventana (Y = 2023, N = 120, escenario principal) imprimió el MAE de los cinco modelos de referencia en esa ventana; no se cambió nada después de verlo y los archivos de la prueba se borraron. Para el integrador, que decidirá cuáles pasar a `docs/DESVIACIONES.md`:

1. **Conteo de parámetros de la DualLSTM (no es una desviación del protocolo).** El encargo decía 4 289, pero `src.redes.expected_parameter_count(4, 6)` da 4 545 (LSTM rama A 1 344, rama B 1 472, dos atenciones 1 056, Dense(16) 528, Dense(8) 136, salida 9) y `tests/test_comun.py` fija `expected_parameter_count(4, 8) = 4 673` (= 4 545 + 128). Las 400 unidades del escenario principal reportan 4 545 (`resultados/brazo_M/principal/estados_redes.csv`); la fórmula del artículo es la que manda (§5). El 4 289 del encargo parece un error de transcripción.
2. **Contexto de Chronos en N = max.** §5 dice «las últimas N observaciones hasta t − 1». Para N numérico se usa exactamente eso. Para N = max se usó el tamaño de la historia máxima de la ventana (`Particion.n_hist`, 267–315 meses) como longitud fija del contexto, que termina en t − 1 y por tanto incluye validación y los meses de prueba ya observados. La alternativa (historia creciente desde 1997-10) no se ejecutó. Efecto: ninguna truncación (máximo 315 frente al límite de 2048).
3. **Selección de ETS.** Se elige la especificación de menor AICc entre las que producen AICc finito, aunque el optimizador no haya convergido (como `r2lib.seleccionar_ets`); la convergencia de la elegida y el número de especificaciones fallidas o sin convergencia se registran en `incidencias.csv`. El protocolo no dice qué hacer con las no convergidas.
4. **Estado inicial de los modelos de espacio de estado.** ETS, Theta y SARIMA se ajustan con la historia de tamaño N y su estado se recalcula desde el primer mes de esa historia con los datos reales (imputados donde falte) hasta t − 1, sin reestimar parámetros. En Theta, SARIMA (`append(refit=False)`) y ETS (`smooth(params)`) el procedimiento es el de `r2lib.py` y `lib/modelos_series.py` del artículo. Se comprobó que añadir de una vez la validación y la prueba a SARIMA da los mismos pronósticos a un paso que añadirlos uno a uno (diferencia máxima 1,8e-12 t en Y = 2023, N = 120).
5. **Sensibilidad 1 con la ventana 2025.** Se entrenaron y pronosticaron también las 80 unidades de red y los modelos de referencia de 2025 en la sensibilidad 1, aunque esa ventana no tiene meses evaluables y no entra en el pool. Efecto sobre resultados: ninguno (solo cómputo).
6. **Sensibilidades 4 y 5.** No tienen `pronosticos.csv` propio: reutilizan `resultados/brazo_M/principal/pronosticos.csv`.
7. **CPU.** Los modelos de referencia (un proceso de un hilo, Chronos con 2 hilos) se ejecutaron mientras corrían las redes con 2 procesos, así que hubo brevemente 3 procesos de este brazo a la vez. Efecto sobre resultados: ninguno (determinismo verificado en D4).
8. **Sensibilidad 1, 17 meses evaluables (D5).** Ya registrado: 6, 2, 5, 4, 0 meses en 2021–2025 (`resultados/brazo_M/sens1_estricto/metricas.csv`).
"""

if __name__ == "__main__":
    main()
