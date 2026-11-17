# Leyendas de las figuras y tablas del día 4 / Figure and table captions (day 4)

Generadas con `scripts/05_figuras.py --idioma en|es` (`.venv`). Figuras en `reportes/figuras/<en|es>/`, en PDF vectorial y PNG a 300 ppp. Los datos agregados de cada figura están en `reportes/figuras/datos/` y las tablas en `reportes/figuras/tablas/`. Paleta Okabe-Ito, apta para daltónicos; cada modelo tiene el mismo color y marcador en todas las figuras. Brazo D: h = 1, salvo que se indique otra cosa. Pool de prueba: 2021–2025.

Generated with `scripts/05_figuras.py --idioma en|es`. Okabe-Ito colour-blind-safe palette; each model keeps the same colour and marker across figures. Arm D at h = 1 unless stated. Test pool 2021–2025.

En el eje de N, «máx» se sitúa en la mediana de la historia máxima de las 5 ventanas: ~8 858 días en D y ~291 meses en M. La historia real es de 8 127–9 588 días y 267–315 meses según la ventana.

On the N axis, "max" is placed at the median maximum history across the 5 windows (~8,858 days in D, ~291 months in M); the actual history ranges over 8,127–9,588 days and 267–315 months depending on the window.

---

## T1. Resultado principal / Main result

**f01_curva_aprendizaje**

- **ES.** Curva de aprendizaje: MAE de prueba frente al tamaño de historia N (escala log) para los siete modelos, con bandas de IC 95 % (bootstrap pareado estratificado por ventana, B = 10 000, semilla 42). La DualLSTM y la LSTM univariada se promedian entre 10 semillas por fecha antes de calcular el MAE. Izquierda: brazo D (n = 1 818 días). Derecha: brazo M (n = 59 meses).
- **EN.** Learning curves: test MAE versus history length N (log scale) for all seven models, with 95 % CI bands (paired bootstrap stratified by window, B = 10,000, seed 42). Network errors are averaged over 10 seeds per date before computing MAE. Left: arm D (n = 1,818 days). Right: arm M (n = 59 months).
- **Datos / Data:** `resultados/brazo_D/principal/metricas.csv`, `resultados/brazo_M/principal/metricas.csv` (filas / rows `anio = pool`).

**f02_brecha**

- **ES.** Brecha Δ(N) = MAE(DualLSTM) − MAE(ref*(N)), donde ref*(N) es la referencia con menor MAE en cada N (nombre bajo cada punto). Barras: IC 95 % bootstrap de Δ. Línea en 0. *: DM-HLN con p < 0,05.
- **EN.** Gap Δ(N) = MAE(DualLSTM) − MAE(ref*(N)), where ref*(N) is the reference with the lowest MAE at each N (named below each point). Bars: 95 % bootstrap CI of Δ. Line at 0. *: DM-HLN p < 0.05.
- **Datos / Data:** `resultados/brazo_*/principal/delta.csv`.

**f03_brecha_pct**

- **ES.** La brecha de f02 en porcentaje del MAE de ref*(N): 100·Δ/MAE(ref*). El intervalo es el IC de Δ dividido por MAE(ref*); no es un bootstrap del cociente.
- **EN.** The f02 gap as a percentage of ref*(N) MAE: 100·Δ/MAE(ref*). The interval is the Δ CI divided by MAE(ref*); it is not a bootstrap of the ratio.
- **Datos / Data:** `reportes/figuras/tablas/t1_mae_dual_ref_brecha.csv`.

**t1_mae_dual_ref_brecha** (`.csv`, `.en.tex`, `.es.tex`)

- **ES.** MAE de la DualLSTM y de ref*(N) por N y brazo, Δ con IC 95 %, Δ en % y p DM-HLN.
- **EN.** DualLSTM and ref*(N) MAE by N and arm, Δ with 95 % CI, Δ in %, DM-HLN p-value.
- **Datos / Data:** `resultados/brazo_*/principal/delta.csv`.

## T2. Diseño sin fuga / Leakage-free design

**f04_rolling_origin**

- **ES.** Ventanas rolling-origin. Para cada año de prueba Y y cada N: historia de entrenamiento (las últimas N observaciones hasta el 31-12 de Y−2), validación (Y−1) y prueba (Y). La historia termina siempre en el mismo punto y la validación y la prueba son las mismas para todo N.
- **EN.** Rolling-origin windows. For each test year Y and each N: training history (the last N observations up to 31 Dec of Y−2), validation (Y−1) and test (Y). The history always ends at the same point, and validation and test are identical for every N.
- **Datos / Data:** `reportes/figuras/datos/f04_ventanas.csv` (de / from `src/ventanas.py`).

**t2_pruebas**

- **ES.** Las 20 pruebas de `tests/test_comun.py`, qué comprueba cada una y su resultado (20 pasan).
- **EN.** The 20 tests in `tests/test_comun.py`, what each checks and its result (20 pass).
- **Datos / Data:** `reportes/figuras/datos/pytest_junit.xml`, `reportes/figuras/tablas/t2_pruebas.csv`.

**t2_hashes**

- **ES.** SHA-256 congelados en 9ede96c (`docs/hashes_protocolo.json`) frente a los recalculados al terminar el día 4 con el mismo método (CRLF→LF): coinciden los 9.
- **EN.** SHA-256 hashes frozen at 9ede96c versus those recomputed at the end of day 4 with the same method (CRLF→LF): all 9 match.
- **Datos / Data:** `reportes/figuras/tablas/t2_hashes.csv`.

## T3. Prueba de perturbación / Perturbation test (diagnóstico posterior / post-hoc diagnostic)

**f05_perturbacion**

- **ES.** Una unidad por brazo (principal, Y = 2025, N = máx, DualLSTM, semilla 0). Se sustituyen al azar todas las observaciones posteriores al origen (fin de 2024) y se repiten el escalado, las secuencias, el entrenamiento y el pronóstico. Verde: diferencia máxima 0 (idéntico). Naranja: cambia. El escalador, las secuencias y objetivos de entrenamiento y validación, la entrada y el pronóstico del primer paso de prueba son idénticos, y el pronóstico coincide con el publicado el día 3. El valor observado y los pasos posteriores, que usan datos posteriores al origen, sí cambian. Eje symlog. Diagnóstico posterior (D12).
- **EN.** One unit per arm (main analysis, Y = 2025, N = max, DualLSTM, seed 0). All observations after the forecast origin (end of 2024) are randomly replaced, and scaling, sequence construction, training and forecasting are rerun. Green: maximum difference 0 (identical). Orange: changes. Scaler, training and validation sequences and targets, and the first test step's input and forecast are identical; that forecast also equals the one published on day 3. The observed value and later steps, which use post-origin data, change. Symlog axis. Post-hoc diagnostic (D12).
- **Datos / Data:** `reportes/figuras/datos/perturbacion.csv` (`scripts/06_perturbacion.py`).

## T4. Entrenamiento / Training

**f06_epoca_parada**

- **ES.** Distribución de la mejor época (rellena; early stopping con val_loss) y de la época de parada (contorno) por N, sobre 5 ventanas × 10 semillas = 50 unidades por caja (principal, h = 1). La parada llega 15 épocas después de la mejor (paciencia 15).
- **EN.** Distribution of the best epoch (filled; early stopping on val_loss) and the stopping epoch (outline) by N, over 5 windows × 10 seeds = 50 units per box (main analysis, h = 1). Stopping occurs 15 epochs after the best epoch (patience 15).
- **Datos / Data:** `resultados/brazo_*/principal/estados_redes.csv`; resumen / summary `reportes/figuras/datos/f06_epocas.csv`.

**f07_variabilidad_semillas**

- **ES.** MAE de prueba (pool 2021–2025) de cada una de las 10 semillas, por N, para la LSTM univariada y la DualLSTM. La línea horizontal es el MAE de ref*(N).
- **EN.** Test MAE (2021–2025 pool) of each of the 10 seeds, by N, for the univariate LSTM and the DualLSTM. The horizontal line is the ref*(N) MAE.
- **Datos / Data:** `reportes/figuras/datos/f07_mae_por_semilla.csv` (de / from `pronosticos.csv` + `data/processed`); ref*: `resultados/brazo_*/principal/delta.csv`.

## T5. Pronósticos frente a observado / Forecasts vs observed

(Contienen la serie del SISAP: se versiona solo el PNG, D13. / Contain SISAP series: only the PNG is versioned, D13.)

**f08_D_2025**

- **ES.** Brazo D, 2025: observado, DualLSTM (media de 10 semillas, N = máx), SARIMA (ref*(máx)) y Seasonal Naive, h = 1. Abajo: detalle de las 8 primeras semanas completas (2025-01-06 a 2025-03-02), elegidas por regla fija y no por el error.
- **EN.** Arm D, 2025: observed, DualLSTM (mean of 10 seeds, N = max), SARIMA (ref*(max)) and Seasonal Naive, h = 1. Bottom: the first 8 complete weeks (2025-01-06 to 2025-03-02), chosen by a fixed rule, not by error.
- **Datos / Data:** `resultados/brazo_D/principal/pronosticos.csv` (no versionado / not versioned); observado / observed: `data/processed/sutil_diario.csv`.

**f09_D_2023_episodio**

- **ES.** Brazo D, episodio de febrero a junio de 2023, con los mismos modelos que f08.
- **EN.** Arm D, February–June 2023 episode, same models as f08.
- **Datos / Data:** como / as f08.

**f10_M_2021_2025**

- **ES.** Brazo M, 2021–2025: observado, DualLSTM (media de semillas, N = máx), ETS (ref*(máx)) y Seasonal Naive. El hueco de 2021-02 es el mes de la anomalía, que no se evalúa.
- **EN.** Arm M, 2021–2025: observed, DualLSTM (seed mean, N = max), ETS (ref*(max)) and Seasonal Naive. The gap at 2021-02 is the anomaly month, which is not evaluated.
- **Datos / Data:** `resultados/brazo_M/principal/pronosticos.csv`; `data/processed/sutil_mensual.csv`.

## T6. Diagnóstico de errores / Error diagnostics

**f11_mae_dia_semana_D**

- **ES.** Brazo D, N = máx: MAE por día de la semana de la DualLSTM, SARIMA (ref*) y Seasonal Naive.
- **EN.** Arm D, N = max: MAE by day of week for DualLSTM, SARIMA (ref*) and Seasonal Naive.
- **Datos / Data:** `reportes/figuras/datos/f11_f12_mae_calendario.csv`.

**f12_mae_mes**

- **ES.** MAE por mes del año (N = máx) en los dos brazos, para la DualLSTM, ref*(máx) y Seasonal Naive. En M cada mes tiene 4 o 5 observaciones.
- **EN.** MAE by month of year (N = max) in both arms for DualLSTM, ref*(max) and Seasonal Naive. In M each month has 4–5 observations.
- **Datos / Data:** `reportes/figuras/datos/f11_f12_mae_calendario.csv`.

**f13_acf_residuos**

- **ES.** ACF de los residuos de prueba (pronóstico − observado; N = máx, h = 1) de la DualLSTM y de ref*(máx), sobre 2021–2025 con los días no evaluables como faltantes (`missing = "conservative"`). Banda gris: ±1,96/√n.
- **EN.** ACF of test residuals (forecast − observed; N = max, h = 1) for DualLSTM and ref*(max) over 2021–2025, with non-evaluable days treated as missing (`missing = "conservative"`). Grey band: ±1.96/√n.
- **Datos / Data:** `reportes/figuras/datos/f13_acf_residuos.csv`.

**f14_shock_vs_normal**

- **ES.** MAE en días o meses de shock frente a normales (N = máx). Shock según §6 del protocolo: la variación relativa frente a t−7 (D) o t−1 (M) supera el percentil P (50, 60, 75, 90) de la historia hasta el origen de cada ventana. En M, con P90 hay solo 2 meses de shock.
- **EN.** MAE on shock versus normal days/months (N = max). Shock per protocol §6: relative change versus t−7 (D) or t−1 (M) exceeds the P-th percentile (50, 60, 75, 90) of the history up to each window's origin. In M, only 2 months are shocks at P90.
- **Datos / Data:** `reportes/figuras/datos/f14_shock_vs_normal.csv` (con `n` por grupo / with `n` per group).

## T7. Comparación entre modelos / Model comparison

**f15_dm_heatmap**

- **ES.** p-valor DM-HLN (h = 1, bilateral) de la DualLSTM frente a cada referencia, por N y brazo. Color: −log10(p), con tope en 10. +: la DualLSTM tiene más error; −: menos. Descriptivo (no es la regla de decisión, que solo usa ref*). El p se calcula con la cola `t.sf` para evitar el subdesbordamiento del `delta.csv` (p = 0).
- **EN.** DM-HLN p-value (h = 1, two-sided) of DualLSTM versus each reference, by N and arm. Colour: −log10(p), capped at 10. +: DualLSTM has higher error; −: lower. Descriptive (the decision rule only uses ref*). p computed with `t.sf` to avoid the underflow (p = 0) in `delta.csv`.
- **Datos / Data:** `reportes/figuras/datos/f15_dm_pares.csv`.

**t7_top_ranked**

- **ES.** «Consistently top-ranked» por N: número de umbrales (de 4) en que cada modelo es primero o empata (✓ si son 3 o más). El CSV incluye la sensibilidad 5, marcada como calculada con bootstrap iid (D9); es idéntica a la principal.
- **EN.** "Consistently top-ranked" by N: number of thresholds (of 4) where each model is first or tied (✓ if ≥ 3). The CSV includes sensitivity 5, flagged as computed with an iid bootstrap (D9); it is identical to the main analysis.
- **Datos / Data:** `resultados/brazo_*/{principal,sens5_bloques}/top_ranked.csv`.

## T8. Puente con el artículo / Bridge to the article

**f16_M_N120_vs_max**

- **ES.** Brazo M: brecha de la DualLSTM con N = 120 (el tamaño de historia del artículo) y con N = máx, en toneladas y en % de ref*, con IC 95 %.
- **EN.** Arm M: DualLSTM gap at N = 120 (the article's history length) and at N = max, in tonnes and in % of ref*, with 95 % CI.
- **Datos / Data:** `resultados/brazo_M/principal/delta.csv`.

**f17_sisap_vs_sisagri**

- **ES.** Ingreso mensual al GMML (SISAP) frente a la producción nacional de limón sutil (SISAGRI), 2016–2025, como índices (media = 100). Los meses faltantes del SISAP no se dibujan. r de Pearson del día 2, en niveles y en diferencias, solo con meses completos.
- **EN.** Monthly arrivals at GMML (SISAP) versus national sutil lime production (SISAGRI), 2016–2025, as indices (mean = 100). Missing SISAP months are not drawn. Pearson r from day 2, in levels and differences, complete months only.
- **Datos / Data:** `reportes/dia2_mensual_vs_sisap.csv`, `reportes/dia2_correlacion_sisagri.csv`; SISAGRI: `limon_sutil_nacional_mensual.csv` del repositorio del artículo / of the article repository.

**f18_tiempos**

- **ES.** Tiempo de cómputo por unidad (mediana sobre ventanas y semillas) por N y modelo, en escala log-log. Redes: un ajuste y una predicción por unidad (h = 1). Estadísticos y Chronos en D: un ajuste con h = 1 y 7 juntos. Los valores menores de 0,01 s se dibujan en 0,01. Las redes corrieron en paralelo en una máquina compartida.
- **EN.** Compute time per unit (median over windows and seeds) by N and model, log-log scale. Networks: one fit and prediction per unit (h = 1). Statistical models and Chronos in D: one fit covering h = 1 and 7. Values below 0.01 s are drawn at 0.01. Networks ran in parallel on a shared machine.
- **Datos / Data:** `reportes/figuras/datos/f18_tiempos_por_unidad.csv` (de / from `estados_redes.csv`, `resultados/brazo_D/estadisticos/principal/unidades/*.json`, `resultados/brazo_M/principal/base_tiempos.csv`); totales por fase / phase totals: `resultados/tiempos_dia3.csv`.
