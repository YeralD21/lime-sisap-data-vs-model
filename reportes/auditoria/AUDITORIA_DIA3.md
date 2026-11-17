# Auditoría del día 3 (Experimento 2, curvas de aprendizaje)

Auditor: agente 3. Fecha: 2026-11-17. Autoridad: `docs/PROTOCOLO.md` (congelado en 9ede96c).
Todo lo que escribí está en `reportes/auditoria/`. No modifiqué nada de `src/`, `tests/`, `resultados/`, `docs/`, `data/` ni de los informes de los brazos. No ejecuté la curva completa. Las únicas redes que reentrené fueron 3 unidades sueltas, en un directorio temporal fuera del repositorio.

Orden seguido: protocolo → código de `src/` y de los dos brazos → fugas → reproducción → **después** lectura de `reportes/brazo_M.md` y `reportes/brazo_D.md` → `--verificar` y pytest.

## 1. Resumen

- **Ninguna etiqueta cambia.** Recalculé de forma independiente MAE, ref\*(N), Δ(N), el IC 95 % del bootstrap y la etiqueta de los 13 análisis con etiqueta (8 de D y 5 de M) a partir de `pronosticos.csv`. Las 13 etiquetas coinciden con `decision.csv` (`reportes/auditoria/recalculo_decision.csv`):
  - D: «el modelo» en la principal y en las sensibilidades 1 a 7;
  - M: «el modelo» en la principal y en las sensibilidades 4, 5 y 7, y «no concluyente (no se reproduce la desventaja)» en la sensibilidad 1.
- **La reproducción independiente coincide** hasta el redondeo con el que se escribió `pronosticos.csv`:
  - Seasonal Naive, ETS (procedimiento literal, día a día), SARIMA (M, por una vía de cálculo distinta), Chronos y una semilla de la DualLSTM por brazo;
  - las diferencias máximas son ≤ 9,3 × 10⁻⁵ t en D y ≤ 5 × 10⁻⁶ t en M (§4).
- **No encontré fuga de información.** La prueba solo se usa para evaluar, para definir los días de shock (como prevé §6) y como estado o entrada ya observada (como prevé §3).
- **Cifras de los informes:** comparé 2 652 celdas de tabla con los CSV citados y no hay ninguna discrepancia. Aun así:
  - hay 5 redondeos dobles de una unidad en el último dígito;
  - hay algunas cifras sin respaldo en un CSV.
- **Hallazgos:** 0 bloqueantes, 3 importantes y 11 menores (§6).
- `04_hashes_protocolo.py --verificar`: OK. pytest: 20 tests pasados (§7).

## 2. Protocolo → código

Leyenda del estado: **C** = conforme; **DR** = desviación registrada (con su número en `docs/DESVIACIONES.md`); **NR** = interpretación o desviación no registrada (remite al hallazgo de §6).

| Apartado del protocolo | Dónde se implementa | Estado |
|---|---|---|
| §2 Datos: archivos con hash, corte 2025-12-31, inicio 1997-10-01 | `src/config.py`; `src/datos.py:_diario`; `tests/test_sin_datos_2026...` | C (`--verificar` OK) |
| §2.1 Supuesto «...» = 0; objetivo D `vol_total`; M = suma diaria; mes faltante con más del 20 % de días faltantes o con la anomalía | `src/datos.py:_diaria`, `_mensual` (columnas de `data/processed`) | C. Días y meses evaluables: 1 818 y 59, iguales a §2.2 |
| §2.2 Imputación D con t−7, en orden cronológico | `src/datos.py:imputar_t7` | C (solo pasado) |
| §2.2 Imputación M: suma de la serie diaria imputada | `src/datos.py:_mensual` (`suma = dia.y.resample("MS").sum()`) | C. 2021-02 imputado = 15 881 t (valor comprobado por mí en consola, sin CSV) |
| §2.2 Anomalía 2021-02-01: faltante, se imputa y no se evalúa; M 2021-02 faltante | `src/datos.py:_diaria` (`anom`), `_mensual` (`mes_con_anomalia`) | C (`tests/test_anomalia`) |
| §2.2 Tabla de evaluables (sensibilidad 1 de M: 18) | `src/datos.py:_mensual` | DR (D5: 17). Coincido con D5: la regla escrita de §2.2 prevalece |
| §3 Ventanas: entrenamiento hasta Y−2, validación Y−1, prueba Y | `src/ventanas.py:particion` | C |
| §3 Parámetros una vez por ventana; estado con datos reales, sin reestimar | M: `modelos_base.py` (append, smooth, `ThetaFijo`); D: `estadisticos_lib.py` (un solo `smooth`, `simulate` y `get_prediction`) | C (M) / DR (D7, atajo equivalente; lo reproduje por la vía literal, §4) |
| §4 N = últimas N observaciones del entrenamiento; N = máx desde 1997-10 (8 127–9 588 días; 267–315 meses) | `src/ventanas.py:particion`; `tests/test_historia_maxima_protocolo` | C |
| §4 Secuencias de las redes solo dentro de la historia N | `src/redes.py:objetivos`, `preparar` (con aserción de fuga) | C |
| §5 Seasonal Naive (m = 7 / 12; a h = 7, ŷ = y_{τ−7}) | `estadisticos_lib.py:seasonal_naive`; `modelos_base.py:seasonal_naive` | C |
| §5 ETS: 12 especificaciones, menor AICc en el entrenamiento, empate por orden de la rejilla, `smooth(params)` | `estadisticos_lib.py:seleccionar_ets`, `ets_unidad`; `modelos_base.py:seleccionar_ets`, `ets` | C. Elegir entre especificaciones no convergidas: DR en M (D6) y NR en D (hallazgo m5). `maxiter = 1000`: DR (D7) |
| §5 Theta (θ = 2) con b0, α y factores estacionales fijos y SES actualizado | `ThetaFijo` en los dos brazos (comprobado internamente contra statsmodels) | C (DR D7 por copiar la clase) |
| §5 SARIMA D: (1,0,1)(0,1,1)₇ sobre log, Fourier K = 3 si N ≥ 730, opciones y maxiter 500 | `estadisticos_lib.py:_sarimax`, `sarima_unidad` | C. exp(media log) = mediana: DR (D7) |
| §5 SARIMA M: (1,1,1)(1,1,0)₁₂, `trend = "c"`, en niveles | `modelos_base.py:sarima` | C. Igual que `lib/modelos_series.py` del artículo, que trabaja en niveles |
| §5 Chronos-Bolt-tiny: snapshot `a0e552de…`, mediana, CPU, float32, contexto = últimas N hasta t−1 | `estadisticos_lib.py:chronos_unidad`; `modelos_base.py:chronos` | C. D con N = máx: truncado a 2 048, como prevé §5 («usa solo la cola y se reporta»), y reportado en `incidencias.csv`. M con N = máx: ventana fija de n_hist (DR D6). Lotes en D: DR (D7) |
| §5 DualLSTM: lookback 6, LSTM(16), atención aditiva de 16 unidades sin sesgo, Dense 16/8/1, Dropout 0,2, Adam 1e-3, MSE, batch 8, 300 épocas, sin barajar, ES 15 con restauración, RLROP 8/0,5/1e-6, semillas 0–9 y global 42, checkpoint | `src/config.py:RED`; `src/redes.py:_construir`, `entrenar_y_predecir`, `_determinismo` | C |
| §5 Ramas A y B, escalado solo con el entrenamiento, devolución a toneladas | `src/redes.py:variables`, `Escalador`, `preparar` | C |
| §5 Parámetros 4 673 (D), 4 545 (M) y 1 569 (LSTM univariada) | `src/redes.py:expected_parameter_count`; `estados_redes.csv` | C (todas las unidades de todos los escenarios) |
| §5 h = 7 directo: objetivo y_{t+6} con entradas hasta t−1 | `src/redes.py:objetivos`, `secuencias` | C |
| §5 Agregación de semillas (DA8) | `src/analisis.py:matriz_ae` | C. RMSE de las redes como media del RMSE de cada semilla: NR (m11) |
| §6 MAE, RMSE y MASE con escala SN m = 7/12 sobre la historia máxima | `src/ventanas.py:escala_mase`; `src/evaluacion.py` | C |
| §6 Bootstrap iid pareado estratificado por año, B = 10 000, semilla 42 | `src/evaluacion.py:indices_iid`, `remuestreos` (igual que `lib/evaluacion.py` del artículo) | C |
| §6 Bloques (7 días / 3 meses) | `src/evaluacion.py:indices_bloques`; `config.BLOQUE` | C para Δ y ranking. Top-ranked sigue con iid: NR (m4) |
| §6 DM-HLN, pérdida absoluta, h = 1 (h = 7 en la sensibilidad 3) | `src/evaluacion.py:dm_hln` | C |
| §6 «Consistently top-ranked»: ≥ 3 de 4 umbrales, percentil lineal con la historia hasta el origen, comparación estricta, D semana a semana | `src/evaluacion.py:mascara_shock`, `top_ranked` | C |
| §1 ref\* = menor MAE pool entre las 5 referencias | `src/evaluacion.py:mejor_referencia` | C |
| §1 Regla de decisión | `src/evaluacion.py:regla_decision` | C en el resultado. El orden entre el subcaso y «los datos» es NR (I1) |
| §9.1 «...» = faltante (D: 1 352 días; M: escalado y menos del 80 % de días observados) | `src/datos.py` | C / DR (D5) |
| §9.2 Bolsa (1 815 días) | `src/datos.py:_diaria` (objetivo `bolsa`) | C |
| §9.3 D a h = 7 | `ensamblar_D.py` (`sens3_h7`) | C |
| §9.4 Pool 2023–2025 | `analizar(anios=...)` en los dos brazos | C |
| §9.5 Bloques | `analizar(tipo_boot="bloques")` | C (salvo m4) |
| §9.6 D desde 2006-01-01 | `Escenario(inicio=...)`; `ensamblar_D.py` reutiliza las redes con N ≤ 1000 | DR (D7). Válido: ver el apartado sobre la sensibilidad 6 más abajo |
| §9.7 Anomalía con el valor publicado | `Escenario(anomalia="publicada")` | C (D: 2021-02-01 = 2 351 t evaluado; M: 2021-02 = 16 990 t, mes no faltante) |
| §7.6 Incidencias de convergencia y truncamiento de Chronos | `incidencias.csv` de cada escenario | C |

**Revisión de `docs/DESVIACIONES.md`.**
- D1–D6 son correctas y no ocultan nada.
- D7 tiene dos frases inexactas (m3):
  - dice que el límite de 2 048 lo «fija §5», pero §5 lo deja como «no comprobado». El valor sale de la configuración del modelo y está en `incidencias.csv`, en la columna `context_length_modelo`;
  - dice que la reutilización de redes en la sensibilidad 6 es «aproximada (solo por redondeo)», pero es exacta (`reportes/auditoria/sens6_tindex.csv`).
- Faltan por registrar:
  - el orden de aplicación de la regla de decisión (I1);
  - el criterio de selección de ETS en D (m5);
  - el bootstrap iid del top-ranked en la sensibilidad 5 (m4);
  - los objetivos imputados en la validación del early stopping (m6);
  - la agregación del RMSE (m11).
- Ninguna de estas omisiones afecta a una etiqueta.

**Puntos concretos del encargo.**
- **Contexto de Chronos con N = máx.**
  - **D:** el código toma toda la serie hasta τ−h y la trunca a 2 048 (`estadisticos_lib.py:252-256`). La lectura literal («las últimas N observaciones», con N = n_hist = 8 127–9 588) también se trunca a los mismos 2 048 días, así que las dos lecturas dan pronósticos idénticos. §5 prevé el truncamiento y pide reportarlo, y se reportó.
  - **M:** usa una ventana fija de n_hist (267–315 meses, sin truncar). Es la lectura literal y está registrada en D6.
  - Conforme en los dos brazos.
- **Sensibilidad 6 de D.** Es válida.
  - Según §9.6, solo cambia la historia máxima. Las historias, validaciones y pruebas de N = 120, 360 y 1000 son idénticas (`resultados/brazo_D/sens6_comprobacion.csv`).
  - Comprobé que, tras el escalado, las entradas de las ramas A y B (incluido `t_index`) son idénticas bit a bit en float64 y float32 para los 15 pares (Y, N) y las tres particiones.
  - Reentrené la unidad sens6 (Y = 2024, N = 360, h = 1, DualLSTM, semilla 0): sus 731 predicciones (365 de validación y 366 de prueba) son idénticas a las de la unidad principal publicada, con diferencia 0 (`reportes/auditoria/sens6_tindex.csv`).
  - La reutilización es por tanto exacta, no aproximada.
- **ETS de D con N = 120.** Es un resultado legítimo del procedimiento preregistrado, no un error.
  - Reproduje literalmente Y = 2023, N = 120. Mi selección por AICc elige el índice 3 de la rejilla, ETS(A,A,M), con AICc 1 563,67 (`reportes/auditoria/reproduccion_D_ets_seleccion.csv`). Es la misma especificación de `incidencias.csv`.
  - Los 365 pronósticos coinciden con diferencia máxima de 4,9 × 10⁻⁵ t (`reportes/auditoria/reproduccion_D.csv`).
  - Con tendencia aditiva y estacionalidad multiplicativa estimadas en 120 días, el filtro con parámetros fijos extrapola mal: hay pronósticos entre −4 348,5 y 7 025,7 t (`reportes/auditoria/reproduccion_D_detalle.csv`; mínimo en `pronosticos_negativos.csv`).
  - No afecta a Δ(120), porque ref\*(120) = SARIMA.
- **Sensibilidad 1 de M (17 meses).** Coincido con D5. El test `test_evaluables_coinciden_con_protocolo` fija los conteos 6, 2, 5, 4 y 0.
- **ETS y SARIMA de D con un solo `smooth`.**
  - `equivalencias.csv` da diferencia 0.
  - Además, mi ETS literal (prefijo + `smooth` + `forecast(h)` día a día) coincide en 3 combinaciones (Y = 2024, N = 360, h = 1 y 7; Y = 2023, N = 120, h = 1): diferencia ≤ 4,9 × 10⁻⁵ t, que es el redondeo `%.8g` de `pronosticos.csv`.
- **Paso 0 (D4).** No se guardó ningún artefacto de la comprobación (m10), pero mi reejecución confirma el determinismo:
  - D: la unidad Y = 2024, N = 360, s0 corrió sola, en otro momento, y es idéntica a la unidad publicada, que corrió con 4 procesos paralelos (diferencia 0,0 en 366 días; `reproduccion_D.csv`);
  - M: la unidad Y = 2023, N = 120, s0 también es idéntica (diferencia 0,0 en las 24 predicciones de validación y prueba; comprobado en consola contra `redes_unidades/.../predicciones.csv`).
- **Consistencia entre brazos.**
  - Los dos usan `src.analisis.analizar` con los mismos argumentos (`ensamblar_D.py:217`, `03_analisis.py:204`) y el mismo bootstrap (B = 10 000, semilla 42, estratificado por año).
  - La sensibilidad 5 usa bloques de 7 días en D y de 3 meses en M para Δ y el ranking. El top-ranked usa iid (m4).
  - Recuento de parámetros: 4 673, 4 545 y 1 569 en todas las unidades (en todos los `estados_redes.csv`, 3 800 unidades en total).

## 3. Fuga de información

| Punto | Comprobación | Resultado |
|---|---|---|
| Escalado de las redes | `Escalador` se ajusta con las filas `[ini_hist+6, fin_train]` (`src/redes.py:110-114`); el objetivo usa el mismo escalador que y_t | Sin fuga |
| Selección de modelo | ETS por AICc sobre `y[part.hist]` en los dos brazos; early stopping y checkpoint con `val_loss` de Y−1 (`src/redes.py:232-238`) | Sin fuga (es lo previsto) |
| Umbrales de shock | `mascara_shock` usa `df.y[:origen+1]`, con origen = fin de Y−1 (`src/analisis.py:118`); `tests/test_mascara_shock_usa_historia_hasta_origen` | Sin fuga. El valor real del día de prueba solo decide si ese día es de shock (§6) |
| Escala MASE | `escala_mase` usa la historia máxima hasta Y−2 | Sin fuga |
| Imputación | `imputar_t7` es cronológica; en M, la suma diaria del propio mes faltante (información hasta el fin de ese mes, que solo se usa como dato de ese mes) | Sin fuga |
| Contexto de Chronos | D: `y[ini:τ−h+1]`; M: `y[t−n:t]` | Sin fuga (hasta τ−1, o τ−7 a h = 7) |
| Estado de ETS y SARIMA | D: el filtro es causal (un pronóstico a un paso no dinámico o `dynamic=0` desde τ−h+1); M: append y `smooth` sobre `y[ini:t]`. Reproducido día a día por la vía literal (§4) | Sin fuga |
| Entradas de la red a h = 7 | `secuencias`: filas τ−12…τ−7; cada fila j usa y_j…y_{j−6} y el precio de j−1…j−6, así que el dato más reciente es τ−7 (`src/redes.py:99-102`); `tests/test_redes_sin_fuga...` | Sin fuga |
| Uso de la prueba | Solo como estado o entrada ya observada (§3), en `y_true` del análisis y en la máscara de shock. Ningún hiperparámetro ni especificación depende de la prueba. En M, la prueba previa de una ventana (D6) no cambió nada, según su declaración | Sin fuga |

## 4. Reproducción independiente

Mi código está en `reportes/auditoria/`:
- `reproducir_D.py` y `reproducir_M.py`: solo importan `src.datos` y `src.ventanas`, más statsmodels y chronos;
- `recalcular_delta.py`: no importa `src.analisis` ni `src.evaluacion`;
- `comprobar_sens6_tindex.py`.

La red se reentrenó con `python -m src.redes --unidad ... --destino <temporal>`, fuera del repositorio.

**Elección de ventanas.**
- D: Y = 2024, N = 360 (h = 1 y 7), y además Y = 2023, N = 120 para el caso de ETS inestable.
- M: Y = 2023, N = 120.

Son tamaños intermedios y baratos, que cubren el horizonte h = 7 y el caso extremo de ETS.

**D: `reportes/auditoria/reproduccion_D.csv`** (frente a `resultados/brazo_D/principal/pronosticos.csv`):

| Modelo | Y | N | h | n | dif. máx. (t) | dif. media (t) |
|---|---|---|---|---|---|---|
| Seasonal Naive | 2024 | 360 | 1 | 366 | 0 | 0 |
| Seasonal Naive | 2024 | 360 | 7 | 366 | 0 | 0 |
| ETS (literal, día a día) | 2024 | 360 | 1 | 366 | 4,89e-05 | 3,10e-06 |
| ETS (literal) | 2024 | 360 | 7 | 366 | 4,88e-05 | 2,88e-06 |
| Seasonal Naive | 2023 | 120 | 1 | 365 | 0 | 0 |
| ETS (literal) | 2023 | 120 | 1 | 365 | 4,90e-05 | 1,13e-05 |
| Chronos (sin lotes, 6 fechas) | 2024 | 360 | 1 | 6 | 9,26e-05 | 4,29e-05 |
| Chronos (sin lotes, 6 fechas) | 2024 | 360 | 7 | 6 | 5,95e-05 | 2,97e-05 |
| DualLSTM, semilla 0 (reentrenada) | 2024 | 360 | 1 | 366 | 3,59e-05 | 2,51e-06 |
| DualLSTM, semilla 0, frente a la unidad publicada | 2024 | 360 | 1 | 366 | 0 | 0 |

**M: `reportes/auditoria/reproduccion_M.csv`** (frente a `resultados/brazo_M/principal/pronosticos.csv`, Y = 2023, N = 120, h = 1, 12 meses):

| Modelo | dif. máx. (t) | dif. media (t) |
|---|---|---|
| Seasonal Naive | 0 | 0 |
| ETS (índice 1 de la rejilla, literal) | 4,42e-06 | 2,14e-06 |
| SARIMA (smooth + pronóstico a un paso, vía distinta del append del brazo) | 4,70e-06 | 2,43e-06 |
| Chronos (12 meses, sin lotes) | 4,69e-06 | 2,84e-06 |
| DualLSTM, semilla 0 (reentrenada) | 5,0e-06 | 2,58e-06 |

Todas las diferencias están por debajo del redondeo de `pronosticos.csv` (`%.8g` en D y `%.10g` en M).

**Recálculo de MAE pool y Δ** (`reportes/auditoria/recalculo_delta_D.csv`, `recalculo_delta_M.csv`). Cubre todos los N y todos los análisis con etiqueta: 32 filas en D y 20 en M.
- ref\* coincide en las 52 filas.
- n coincide en todas.
- Diferencia máxima frente a los CSV publicados (que están redondeados a 6 cifras significativas):

  | Brazo | Δ | IC 95 % | MAE de `metricas.csv` |
  |---|---|---|---|
  | D | 4,8 × 10⁻⁴ t | 3,8 × 10⁻⁴ t | 5,0 × 10⁻⁴ t |
  | M | 4,4 × 10⁻⁴ t | 4,5 × 10⁻³ t | 5,0 × 10⁻³ t |

  Todas están dentro del redondeo `%.6g`.
- Escenario principal:

  | Brazo | N | ref\* | MAE DualLSTM | MAE ref\* | Δ | IC 95 % |
  |---|---|---|---|---|---|---|
  | D | 120 | SARIMA | 206,901 | 97,5179 | 109,383 | [103,656; 115,119] |
  | D | 360 | ETS | 156,777 | 93,2197 | 63,5573 | [58,0556; 69,0123] |
  | D | 1000 | ETS | 137,419 | 90,3601 | 47,0592 | [42,4605; 51,7197] |
  | D | máx | SARIMA | 121,625 | 89,3265 | 32,2986 | [27,8316; 36,7042] |
  | M | 60 | Theta | 1 842,21 | 956,185 | 886,026 | [575,454; 1 196,51] |
  | M | 120 | ETS | 1 436,66 | 976,544 | 460,116 | [220,557; 696,614] |
  | M | 240 | Theta | 1 354,02 | 1 011,96 | 342,066 | [126,890; 553,382] |
  | M | máx | ETS | 1 333,12 | 972,512 | 360,608 | [159,307; 561,736] |

- Etiquetas: mi aplicación de la regla de §1 a cada `delta.csv` coincide con `decision.csv` en los 13 casos (`reportes/auditoria/recalculo_decision.csv`).

## 5. Verificación de las cifras de los informes

Script: `reportes/auditoria/verificar_cifras.py`. Resumen en `reportes/auditoria/verificacion_cifras_resumen.csv`.

- **Tablas.** Cada celda se compara con el CSV que cita la tabla (`metricas.csv`, `delta.csv`, `semillas.csv`, `ranking.csv` y `top_ranked.csv`), al número de decimales mostrado:

  | Informe | Celdas comprobadas | Discrepancias |
  |---|---|---|
  | M | 1 240 | 0 |
  | D | 1 412 | 0 |

  Detalle (vacío) en `verificacion_cifras_M.csv` y `verificacion_cifras_D.csv`.
- **Redondeo frente al valor exacto** (`reportes/auditoria/redondeo_delta.csv`, script `verificar_redondeo.py`). Hay 5 celdas con redondeo doble: el informe redondea el valor ya redondeado del CSV y queda una unidad por debajo en el último dígito.
  - D, principal y sensibilidad 5, N = máx, MAE DualLSTM: el informe da 121.62; el valor exacto es 121,6251, así que debería ser 121.63.
  - D, sensibilidad 4, N = máx: el informe da 107.48; el valor exacto es 107,4855, así que debería ser 107.49.
  - M, sensibilidad 1, N = 120: el informe da 1 572.5; el valor exacto es 1 572,447, así que debería ser 1 572.4.
  - M, sensibilidad 7, N = 60: el informe da 1 826.5; el valor exacto es 1 826,447, así que debería ser 1 826.4.
  - Las mismas cifras se repiten en las tablas de MAE pool de esos escenarios. Esto no afecta a Δ ni a ninguna etiqueta.
- **Etiquetas.** Cada `decision.csv` es la aplicación literal de la regla a su `delta.csv` (§4).
- **Números sin un valor CSV compatible.** El barrido está en `cifras_sin_respaldo_M.csv` (37) y `cifras_sin_respaldo_D.csv` (23). Los revisé uno por uno:
  - **Correctos, pero tomados de JSON o de logs, no de un CSV:**
    - tiempos de pared de M: 3 281, 4 648 y 3 138 s, y el total de 11 067 s, en `redes_tiempo_pared.json`;
    - tiempos de pared de D: 6 295, 4 502, 6 834, 4 132, 1 189 y 3 028 s, en `logs/redes_cadena.log`; y 122, 109, 104, 92 y 109 s, en `logs/est_*.log`.
  - **Correctos, agregados de CSV:**
    - M: sumas de segundos 3 933, 5 343 y 3 950; medianas de épocas 31,5 y 52,5 y de mejor época 16,5 y 37,5; sumas de segundos 2 099 y 1 833;
    - D: 9,59 h y 800 unidades; 7 304 = 4 × 1 826; tiempos de la DualLSTM con N = máx de 58 a 257 s, mediana 100 s.
    - Lo comprobé en consola contra `estados_redes.csv` y `base_tiempos.csv`.
  - **Constantes y fechas del protocolo:** 2 048, 730, 500, 2006, 1997 y 10 000.
  - **2 351 t:** sale de `data/processed`, no de `resultados/`.
  - **Cifras sin ningún respaldo en archivo** (hallazgo I3):
    - M: «1,8e-12 t» (§8.4; el propio informe dice que no se guardó);
    - D: «60–70 s medidos aisladamente» (§8.10);
    - D: «diferencias de cientos de toneladas» (§8.11);
    - M: 4 289 y el desglose de parámetros (§8.1). Este desglose se puede rederivar de `src/redes.py` y el informe lo declara como escrito a mano.

## 6. Hallazgos

### Bloqueantes

Ninguno. Las 13 etiquetas se reproducen. En todos los escenarios, Δ(máx) es positiva y su IC excluye 0 con mucho margen, salvo la sensibilidad 1 de M, que es no significativa en todos los N. Ningún hallazgo puede cambiar una etiqueta.

### Importantes

**I1. El orden de aplicación de la regla de decisión no está registrado.**
- **Evidencia:** `src/evaluacion.py:202-208`. El subcaso «no se reproduce la desventaja» se evalúa **antes** que «los datos». Si Δ baja de forma monótona, Δ(máx) no es significativa y ningún N tiene Δ significativamente positiva, el código da «no concluyente», y no «los datos». La tabla de §1 no fija esa precedencia, aunque el ejemplo «Δ(máx) < 0 significativa sin desventaja con N pequeño» la sugiere. `tests/test_comun.py:test_regla_decision` la fija.
- **Efecto:** ninguno en las etiquetas actuales. Ningún escenario cumple esa condición: la sensibilidad 1 de M no es monótona (`resultados/brazo_M/sens1_estricto/delta.csv`, 198,8 < 309,6).
- **Propuesta:** registrarlo en `docs/DESVIACIONES.md` como interpretación de §1, con la fecha y la indicación de que se decidió antes de ver errores.

**I2. Los pronósticos negativos no se declaran.**
- **Evidencia:** `reportes/auditoria/pronosticos_negativos.csv`. En D principal hay 4 638 de 365 200 pronósticos (h = 1 y 7) por debajo de 0 t:
  - ETS con N = 120 y h = 1: 125, con mínimo −4 348,5 t;
  - DualLSTM con N = 120 y h = 1: 620;
  - LSTM univariada con N = 120 y h = 1: 503.

  En los otros escenarios de D hay entre 1 606 y 3 078 por escenario; en M, 0. Ninguno de los dos informes ni `docs/DESVIACIONES.md` lo menciona.
- **Efecto:** ninguno en la etiqueta. El protocolo no prevé truncar en 0, así que no truncar es el procedimiento literal. Pero el dato explica la inestabilidad del ETS con N = 120 y forma parte del MAE de las redes con N pequeño.
- **Propuesta:** declararlo en el informe de D (§7, incidencias) y en la nota del ETS con N = 120. No truncar a posteriori: sería una desviación.

**I3. Hay cifras de los informes sin respaldo en un CSV, contra la regla obligatoria.**
- **Evidencia:**
  - `reportes/brazo_M.md:3` y `:347` («1,8e-12 t», no guardado);
  - `reportes/brazo_D.md:460` («60–70 s medidos aisladamente»);
  - `reportes/brazo_D.md:461` («cientos de toneladas»);
  - `reportes/brazo_D.md:3` afirma «Todas las cifras salen de CSV», pero las tablas de tiempos de su §1 salen de `logs/*.log`, y las de M, de `redes_tiempo_pared.json`.
- **Efecto:** ninguno en los resultados. Las cifras de los logs y del JSON son correctas.
- **Propuesta:**
  - guardar la comprobación de 1,8e-12 t en un CSV, o quitar la cifra;
  - quitar o respaldar «60–70 s» y «cientos de toneladas»;
  - pasar los tiempos de logs y JSON a un CSV, o corregir la frase de la cabecera del informe de D.

### Menores

- **m1. Cinco redondeos dobles de una unidad en el último dígito.**
  - Evidencia: `reportes/auditoria/redondeo_delta.csv`; `reportes/brazo_D.md:98`, `:262`, `:289`; `reportes/brazo_M.md:159`, `:247`.
  - Propuesta: formatear desde el valor exacto o desde CSV con más cifras.
- **m2. `p DM-HLN = 0` en D es un subdesbordamiento.**
  - Evidencia: `delta.csv` da p_DM = 0 (por ejemplo, con HLN = 36,56 y n = 1 818); `reportes/brazo_D.md:95-98` y siguientes.
  - Además, el formato no es homogéneo: «0», «0.00» y «9.4e-12» en D; «<0,0001» con coma decimal en M, mientras el resto usa punto.
  - Propuesta: escribir «< 1e-15», o el umbral que corresponda, y unificar el separador decimal.
- **m3. Dos frases inexactas en D7** (`docs/DESVIACIONES.md:71` y `:80`).
  - El 2 048 no lo «fija §5». La sensibilidad 6 no es «aproximada»: es exacta (`reportes/auditoria/sens6_tindex.csv`: entradas escaladas idénticas, diferencia 0, y 731 predicciones idénticas).
  - El informe de D lo repite en `:457` («invariante salvo redondeo»).
  - Propuesta: corregir la redacción.
- **m4. El top-ranked de la sensibilidad 5 usa bootstrap iid.**
  - Evidencia: `src/analisis.py:120` → `src/evaluacion.py:168`, donde `ranking(...)` se llama sin `tipo`. Δ y el ranking sí usan bloques (`src/analisis.py:110`).
  - Propuesta: registrarlo o pasar `tipo`/`largo` a `top_ranked`. No afecta a la etiqueta.
- **m5. El criterio de selección de ETS en D no está registrado.**
  - Evidencia: D elige también entre especificaciones con AICc finito aunque no hayan convergido (`resultados/brazo_D/scripts/estadisticos_lib.py:89`). D6 lo registra solo para M.
  - Sin efecto: según `reportes/brazo_D.md` §7, las especificaciones elegidas no convergidas son «0 de 20» en todos los escenarios.
  - Propuesta: ampliar D7.
- **m6. El early stopping usa objetivos de validación imputados.**
  - Evidencia: `src/redes.py:237`, `ys[pos["val"]]` con `df.y` imputada. Por ejemplo, 2021-02-01 en la validación de Y = 2022. El protocolo no lo precisa.
  - Propuesta: registrarlo como interpretación.
- **m7. La tabla «ETS inestable» del informe D omite la sensibilidad 3.**
  - Evidencia: `reportes/brazo_D.md:389-396`. Falta la fila de la sensibilidad 3 (h = 7): 2023, N = 120, MAE 1 195,69 t (`resultados/brazo_D/sens3_h7/metricas.csv`).
  - Propuesta: añadirla.
- **m8. Redacción inexacta en la sensibilidad 4 de D.**
  - Evidencia: `reportes/brazo_D.md:251` dice «umbrales de shock solo con esos años». Los umbrales son los de cada ventana, con la historia hasta el origen; lo único que cambia es qué años de prueba entran.
  - Propuesta: corregir la frase.
- **m9. Restos de formato de Python en el informe D.**
  - Evidencia: `reportes/brazo_D.md:443`, «[np.int64(4673)]» y un diccionario de Python.
  - Propuesta: limpiar el formato.
- **m10. D4 no dejó ningún artefacto de su comprobación.**
  - Evidencia: D4 dice que se hizo en un directorio temporal.
  - Mi reejecución (`reportes/auditoria/reproduccion_D.csv`, fila «unidad publicada»: diferencia 0) la respalda ahora.
  - Propuesta: citar este CSV en D4.
- **m11. El RMSE de las redes es la media del RMSE de cada semilla.**
  - Evidencia: `src/analisis.py:73-74`. El MAE, en cambio, sigue DA8 (error promediado entre semillas). El protocolo no fija el RMSE.
  - Propuesta: declararlo en los informes. El RMSE no entra en la regla.

## 7. `--verificar` y pytest

- `.venv\Scripts\python.exe scripts/04_hashes_protocolo.py --verificar` → «OK: todos los hashes coinciden» (código de salida 0).
- `.venv311\Scripts\python.exe -m pytest -q tests` → «20 passed in 2.07s» (código de salida 0).

## Archivos de la auditoría

- `reportes/auditoria/AUDITORIA_DIA3.md` (este informe).
- Scripts:
  - `reproducir_D.py`
  - `reproducir_M.py`
  - `recalcular_delta.py`
  - `comprobar_sens6_tindex.py`
  - `verificar_cifras.py`
  - `verificar_redondeo.py`
  - `contar_negativos.py`
- CSV:
  - `reproduccion_D.csv`, `reproduccion_D_detalle.csv`, `reproduccion_D_ets_seleccion.csv`
  - `reproduccion_M.csv`, `reproduccion_M_detalle.csv`
  - `recalculo_delta_D.csv`, `recalculo_delta_M.csv`, `recalculo_decision.csv`
  - `sens6_tindex.csv`
  - `verificacion_cifras_M.csv`, `verificacion_cifras_D.csv`, `verificacion_cifras_resumen.csv`
  - `cifras_sin_respaldo_M.csv`, `cifras_sin_respaldo_D.csv`
  - `redondeo_delta.csv`
  - `pronosticos_negativos.csv`
