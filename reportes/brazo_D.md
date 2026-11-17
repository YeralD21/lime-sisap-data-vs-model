# Brazo D (diario) — resultados del día 3

Las cifras salen de CSV de `resultados/brazo_D/` (ruta citada en cada tabla), salvo los tiempos de pared de la sección 1, que salen de los registros de `resultados/brazo_D/logs/` (ruta citada). Los pronósticos no llevan `y_true`: `src.analisis.completar_y_true` lo toma de `data/processed` al analizar. Este informe lo genera `resultados/brazo_D/scripts/generar_informe.py`.

## 1. Qué se ejecutó y tiempo de cómputo

Intérprete `.venv311` (Python 3.11.9; versiones en `resultados/brazo_D/versiones.json`, semillas en `resultados/brazo_D/semillas.json`).

- **Seasonal Naive, ETS, Theta, SARIMA y Chronos-Bolt-tiny:** `resultados/brazo_D/scripts/estadisticos_lib.py` y `estadisticos_run.py` (2 procesos). Un ajuste por ventana y N, con la serie `src.datos.serie(esc).y` y las particiones de `src.ventanas`.
- **LSTM univariada y DualLSTM:** `src.redes.correr_unidades` con 4 procesos (`scripts/redes_run.py`, `scripts/cadena_redes.sh`); 10 semillas (0–9); unidades en `resultados/brazo_D/redes_unidades/`.
- **Ensamblado y análisis:** `scripts/ensamblar_D.py` (usa `src.analisis.analizar` y `escribir`); informe: `scripts/generar_informe.py`.
- **Orden:** principal h = 1 → sensibilidades 4 y 5 (sin reentrenar) → sensibilidad 3 (h = 7) → 1 → 2 → 6 → 7.

**Modelos estadísticos y Chronos** (SN, ETS, Theta, SARIMA, Chronos; 100 unidades por escenario = 5 años × 4 N × 5 modelos), tiempos de pared de cada escenario, de `resultados/brazo_D/logs/est_*.log`:

| escenario | segundos (pared, 2 procesos) |
|---|---|
| principal | 122 |
| sens1_estricto | 109 |
| sens2_bolsa | 104 |
| sens6_desde2006 | 92 |
| sens7_anomalia_publicada | 109 |

**Redes** (LSTM univariada y DualLSTM; 10 semillas), tiempos de pared de cada fase, de `resultados/brazo_D/logs/redes_cadena.log`:

| fase | unidades | fallos | segundos (pared, 4 procesos) |
|---|---|---|---|
| principal h=[1] | 400 | 0 | 6295 |
| principal h=[7] | 400 | 0 | 4502 |
| sens1_estricto h=[1] | 400 | 0 | 6834 |
| sens2_bolsa h=[1] | 400 | 0 | 4132 |
| sens6_desde2006 h=[1] | 100 | 0 | 1189 |
| sens7_anomalia_publicada h=[1] | 400 | 0 | 3028 |

Prueba previa de una ventana (Y = 2025; h = 1 y 7 juntos), `resultados/brazo_D/tiempos_una_ventana.csv`:

| modelo | anio | N | hs | n_hist | segundos |
|---|---|---|---|---|---|
| Seasonal Naive | 2025 | 1000 | 1,7 | 1000 | 0.0 |
| Theta | 2025 | 1000 | 1,7 | 1000 | 0.8 |
| SARIMA | 2025 | 1000 | 1,7 | 1000 | 1.1 |
| Chronos-Bolt-tiny | 2025 | 1000 | 1,7 | 1000 | 8.2 |
| ETS | 2025 | 1000 | 1,7 | 1000 | 3.1 |
| ETS | 2025 | max | 1,7 | 9588 | 9.2 |
| Theta | 2025 | max | 1,7 | 9588 | 0.4 |
| SARIMA | 2025 | max | 1,7 | 9588 | 10.8 |
| Chronos-Bolt-tiny | 2025 | max | 1,7 | 9588 | 12.1 |

Segundos de entrenamiento de las redes (suma de `segundos` de cada unidad, `resultados/brazo_D/principal/estados_redes.csv`, h = 1 y 7): 9.59 h de CPU de proceso en 800 unidades.

## 2. Curva de aprendizaje a h = 1 (escenario principal, pool 2021–2025)

MAE en toneladas/día con IC 95 % (bootstrap pareado iid estratificado por año, B = 10 000, semilla 42). Fuente: `resultados/brazo_D/principal/metricas.csv` (anio = pool).

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] |
| ETS | 300.3 [275.1; 326.4] | 93.2 [87.7; 98.9] | 90.4 [85.3; 95.6] | 91.0 [85.9; 96.1] |
| Theta | 192.8 [182.0; 204.0] | 177.2 [164.7; 190.1] | 125.7 [119.6; 131.9] | 159.5 [152.8; 166.2] |
| SARIMA | 97.5 [91.7; 103.6] | 95.5 [89.7; 101.5] | 96.6 [90.9; 102.4] | 89.3 [83.8; 95.0] |
| Chronos-Bolt-tiny | 272.3 [264.3; 280.4] | 262.9 [254.4; 271.4] | 256.8 [247.9; 265.6] | 255.5 [246.7; 264.3] |
| LSTM univariada | 235.6 [227.8; 243.5] | 152.6 [146.0; 159.5] | 135.3 [129.4; 141.5] | 125.8 [119.9; 131.8] |
| DualLSTM | 206.9 [200.8; 213.2] | 156.8 [151.1; 162.5] | 137.4 [132.0; 142.9] | 121.6 [116.4; 127.1] |

RMSE (t). Fuente: `resultados/brazo_D/principal/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 188.2 | 188.2 | 188.2 | 188.2 |
| ETS | 740.5 | 152.7 | 143.7 | 142.6 |
| Theta | 311.6 | 336.6 | 188.1 | 216.6 |
| SARIMA | 161.1 | 159.3 | 157.0 | 149.4 |
| Chronos-Bolt-tiny | 324.0 | 321.8 | 321.8 | 319.8 |
| LSTM univariada | 301.8 | 217.7 | 196.9 | 187.9 |
| DualLSTM | 266.5 | 212.1 | 188.4 | 174.4 |

MASE (escala fija por ventana: Seasonal Naive m = 7 en la historia máxima). Fuente: `resultados/brazo_D/principal/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 1.729 | 1.729 | 1.729 | 1.729 |
| ETS | 4.417 | 1.382 | 1.342 | 1.352 |
| Theta | 2.870 | 2.642 | 1.888 | 2.371 |
| SARIMA | 1.445 | 1.418 | 1.435 | 1.327 |
| Chronos-Bolt-tiny | 4.039 | 3.899 | 3.807 | 3.788 |
| LSTM univariada | 3.517 | 2.280 | 2.025 | 1.885 |
| DualLSTM | 3.058 | 2.322 | 2.040 | 1.818 |

## 3. ref*(N) y Δ(N) a h = 1

Δ(N) = MAE_DualLSTM − MAE_ref*(N). Significativa si el IC 95 % del bootstrap pareado excluye 0. DM-HLN: prueba secundaria. Fuente: `resultados/brazo_D/principal/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 206.90 | 97.52 | 109.38 | [103.7; 115.1] | sí | 36.56 | 0 | 1818 |
| 360 | ETS | 156.78 | 93.22 | 63.56 | [58.1; 69.0] | sí | 22.66 | 0 | 1818 |
| 1000 | ETS | 137.42 | 90.36 | 47.06 | [42.5; 51.7] | sí | 19.97 | 0 | 1818 |
| max | SARIMA | 121.63 | 89.33 | 32.30 | [27.8; 36.7] | sí | 13.79 | 0 | 1818 |

## 4. Etiqueta de la regla de decisión (brazo D, h = 1)

**Etiqueta del brazo D: «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 32.30 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2021-2022-2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/principal/decision.csv`.

La conclusión conjunta con el brazo M no se calcula aquí.

## 5. Ranking, «consistently top-ranked», años y semillas (h = 1)

Cada celda: posición por MAE en el pool (`*` = primero o empatado con el primero, IC 95 % de la diferencia con el mejor contiene 0) / número de umbrales P50, P60, P75, P90 en que el modelo es primero o empata (`✔` = «consistently top-ranked», ≥ 3 de 4). Fuente: `resultados/brazo_D/principal/ranking.csv` y `resultados/brazo_D/principal/top_ranked.csv` (detalle por umbral: `top_ranked_detalle.csv`).

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 2 / 0 | 3 / 0 | 3 / 0 | 3 / 0 |
| ETS | 7 / 0 | 1* / 4 ✔ | 1* / 4 ✔ | 2* / 1 |
| Theta | 3 / 1 | 6 / 0 | 4 / 2 | 6 / 1 |
| SARIMA | 1* / 4 ✔ | 2* / 4 ✔ | 2 / 3 ✔ | 1* / 4 ✔ |
| Chronos-Bolt-tiny | 6 / 0 | 7 / 0 | 7 / 0 | 7 / 0 |
| LSTM univariada | 5 / 0 | 4 / 0 | 5 / 0 | 5 / 1 |
| DualLSTM | 4 / 0 | 5 / 0 | 6 / 0 | 4 / 0 |

MAE por año de prueba (t). Fuente: `resultados/brazo_D/principal/metricas.csv`.

| Modelo | N | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Seasonal Naive | 120 | 134.3 | 96.5 | 135.6 | 105.0 | 111.3 |
| Seasonal Naive | 360 | 134.3 | 96.5 | 135.6 | 105.0 | 111.3 |
| Seasonal Naive | 1000 | 134.3 | 96.5 | 135.6 | 105.0 | 111.3 |
| Seasonal Naive | max | 134.3 | 96.5 | 135.6 | 105.0 | 111.3 |
| ETS | 120 | 117.4 | 112.4 | 1059.4 | 100.4 | 121.1 |
| ETS | 360 | 102.5 | 79.0 | 104.6 | 90.1 | 90.1 |
| ETS | 1000 | 102.5 | 79.2 | 109.1 | 78.2 | 83.1 |
| ETS | max | 106.9 | 79.8 | 104.3 | 79.6 | 84.4 |
| Theta | 120 | 214.5 | 158.7 | 301.5 | 173.8 | 116.7 |
| Theta | 360 | 230.1 | 112.8 | 302.9 | 126.7 | 115.1 |
| Theta | 1000 | 197.5 | 125.9 | 134.9 | 84.3 | 86.2 |
| Theta | max | 191.6 | 151.1 | 145.5 | 152.0 | 157.1 |
| SARIMA | 120 | 109.7 | 79.0 | 108.6 | 86.1 | 104.5 |
| SARIMA | 360 | 109.4 | 80.9 | 107.3 | 83.3 | 97.1 |
| SARIMA | 1000 | 106.6 | 89.2 | 115.6 | 84.6 | 87.3 |
| SARIMA | max | 106.0 | 74.3 | 103.4 | 80.2 | 82.9 |
| Chronos-Bolt-tiny | 120 | 305.3 | 249.8 | 250.4 | 273.8 | 282.3 |
| Chronos-Bolt-tiny | 360 | 292.3 | 241.6 | 242.5 | 264.7 | 273.4 |
| Chronos-Bolt-tiny | 1000 | 279.7 | 237.4 | 237.6 | 259.3 | 269.9 |
| Chronos-Bolt-tiny | max | 279.7 | 235.8 | 236.5 | 257.4 | 268.1 |
| LSTM univariada | 120 | 300.0 | 213.2 | 287.8 | 228.1 | 149.6 |
| LSTM univariada | 360 | 202.1 | 153.8 | 179.4 | 108.8 | 119.2 |
| LSTM univariada | 1000 | 202.4 | 120.3 | 148.2 | 103.2 | 102.6 |
| LSTM univariada | max | 206.4 | 107.1 | 109.5 | 98.0 | 107.7 |
| DualLSTM | 120 | 182.3 | 218.9 | 206.3 | 187.4 | 239.6 |
| DualLSTM | 360 | 154.7 | 164.2 | 162.9 | 123.6 | 178.6 |
| DualLSTM | 1000 | 145.7 | 133.8 | 149.5 | 134.4 | 123.7 |
| DualLSTM | max | 178.7 | 106.9 | 117.1 | 99.1 | 106.4 |

Dispersión entre semillas de las redes (MAE de cada semilla sobre los días evaluables; `MAE_promedio_semillas` es el MAE del error promediado entre semillas que usa el bootstrap). Fuente: `resultados/brazo_D/principal/semillas.csv`.

| N | modelo | n_semillas | MAE_media | MAE_de | MAE_min | MAE_max | MAE_promedio_semillas |
|---|---|---|---|---|---|---|---|
| 120 | LSTM univariada | 10 | 235.59 | 10.14 | 222.73 | 252.71 | 235.59 |
| 120 | DualLSTM | 10 | 206.90 | 22.39 | 161.59 | 232.91 | 206.90 |
| 360 | LSTM univariada | 10 | 152.59 | 2.94 | 148.28 | 158.39 | 152.59 |
| 360 | DualLSTM | 10 | 156.78 | 11.67 | 139.99 | 175.32 | 156.78 |
| 1000 | LSTM univariada | 10 | 135.28 | 3.52 | 131.06 | 139.96 | 135.28 |
| 1000 | DualLSTM | 10 | 137.42 | 8.03 | 127.55 | 148.11 | 137.42 |
| max | LSTM univariada | 10 | 125.75 | 3.50 | 120.01 | 131.82 | 125.75 |
| max | DualLSTM | 10 | 121.63 | 3.80 | 115.15 | 126.93 | 121.63 |

## 6. Sensibilidades (no cambian la etiqueta principal)

### Sensibilidad 1: «...» como faltante (`vol_total_estricto`)

1 352 días evaluables en 2021–2025; h = 1.

**Etiqueta (sens1_estricto): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 28.72 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2021-2022-2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/sens1_estricto/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens1_estricto/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 203.77 | 116.91 | 86.86 | [79.3; 94.2] | sí | 22.75 | 0 | 1352 |
| 360 | ETS | 161.45 | 107.66 | 53.79 | [47.0; 60.6] | sí | 15.42 | 0 | 1352 |
| 1000 | ETS | 142.06 | 103.98 | 38.08 | [32.0; 43.8] | sí | 12.57 | 0 | 1352 |
| max | ETS | 129.60 | 100.88 | 28.72 | [24.0; 33.5] | sí | 11.49 | 0 | 1352 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens1_estricto/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 129.7 [122.1; 137.5] | 129.7 [122.1; 137.5] | 129.7 [122.1; 137.5] | 129.7 [122.1; 137.5] |
| ETS | 148.7 [128.9; 175.4] | 107.7 [101.8; 113.7] | 104.0 [97.8; 110.4] | 100.9 [95.2; 106.7] |
| Theta | 378.6 [353.3; 404.7] | 290.5 [256.4; 326.6] | 175.9 [167.2; 185.1] | 181.4 [173.9; 189.0] |
| SARIMA | 116.9 [110.1; 124.1] | 110.9 [104.2; 117.8] | 118.4 [111.4; 125.4] | 102.6 [96.5; 109.0] |
| Chronos-Bolt-tiny | 271.8 [261.9; 282.0] | 268.8 [258.3; 279.4] | 267.3 [256.5; 278.5] | 265.9 [255.2; 276.9] |
| LSTM univariada | 229.1 [220.3; 238.2] | 162.7 [155.0; 170.6] | 146.7 [139.8; 154.0] | 143.4 [136.6; 150.5] |
| DualLSTM | 203.8 [196.7; 211.1] | 161.4 [154.8; 168.3] | 142.1 [135.9; 148.3] | 129.6 [123.7; 135.7] |

### Sensibilidad 2: objetivo `vol_bolsa`

1 815 días evaluables; h = 1.

**Etiqueta (sens2_bolsa): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 35.08 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2021-2022-2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/sens2_bolsa/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens2_bolsa/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 209.74 | 97.34 | 112.41 | [106.3; 118.5] | sí | 34.86 | 0 | 1815 |
| 360 | SARIMA | 165.08 | 96.89 | 68.19 | [63.0; 73.5] | sí | 24.46 | 0 | 1815 |
| 1000 | SARIMA | 146.23 | 95.51 | 50.72 | [45.8; 55.6] | sí | 19.62 | 0 | 1815 |
| max | SARIMA | 128.23 | 93.15 | 35.08 | [30.2; 39.9] | sí | 13.51 | 0 | 1815 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens2_bolsa/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 119.2 [112.7; 125.6] | 119.2 [112.7; 125.6] | 119.2 [112.7; 125.6] | 119.2 [112.7; 125.6] |
| ETS | 245.3 [231.0; 259.2] | 114.0 [107.6; 120.8] | 98.5 [92.9; 104.2] | 144.3 [138.0; 150.5] |
| Theta | 161.5 [151.9; 171.6] | 164.0 [152.3; 176.7] | 129.9 [123.9; 136.0] | 186.3 [179.2; 193.2] |
| SARIMA | 97.3 [91.9; 102.8] | 96.9 [91.4; 102.3] | 95.5 [90.2; 100.9] | 93.2 [87.9; 98.4] |
| Chronos-Bolt-tiny | 243.3 [235.9; 250.8] | 234.9 [227.1; 242.8] | 229.2 [221.1; 237.5] | 228.3 [220.2; 236.6] |
| LSTM univariada | 215.8 [209.0; 222.6] | 144.5 [138.3; 150.9] | 127.3 [121.3; 133.5] | 137.1 [130.8; 143.5] |
| DualLSTM | 209.7 [203.1; 216.5] | 165.1 [159.0; 171.3] | 146.2 [140.7; 151.7] | 128.2 [122.7; 133.8] |

### Sensibilidad 3: h = 7 (estrategia directa en las redes)

Escenario principal; los pronósticos están en `principal/pronosticos.csv` (columna `h` = 7); DM-HLN con autocovarianzas hasta h − 1 = 6.

**Etiqueta (sens3_h7): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 28.08 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 7; años 2021-2022-2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/sens3_h7/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens3_h7/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 271.05 | 95.03 | 176.03 | [169.1; 183.1] | sí | 42.93 | 0 | 1818 |
| 360 | ETS | 187.02 | 92.11 | 94.91 | [89.8; 100.0] | sí | 27.64 | 0 | 1818 |
| 1000 | ETS | 148.91 | 90.74 | 58.18 | [53.6; 62.8] | sí | 17.40 | 0 | 1818 |
| max | ETS | 119.21 | 91.13 | 28.08 | [24.5; 31.6] | sí | 11.17 | 0 | 1818 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens3_h7/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] |
| ETS | 321.6 [293.4; 350.7] | 92.1 [87.0; 97.4] | 90.7 [85.8; 95.9] | 91.1 [86.1; 96.3] |
| Theta | 196.5 [186.6; 206.6] | 176.4 [164.5; 189.1] | 123.1 [117.2; 129.2] | 154.3 [147.8; 160.9] |
| SARIMA | 95.0 [89.5; 100.7] | 95.2 [89.6; 100.9] | 97.2 [91.5; 102.9] | 91.5 [86.0; 97.2] |
| Chronos-Bolt-tiny | 260.4 [253.0; 267.9] | 255.8 [248.1; 263.6] | 252.3 [244.4; 260.2] | 251.9 [244.0; 259.8] |
| LSTM univariada | 128.5 [122.4; 134.6] | 121.6 [115.6; 127.6] | 114.1 [108.2; 119.9] | 116.3 [110.6; 122.0] |
| DualLSTM | 271.1 [264.6; 277.7] | 187.0 [181.5; 192.6] | 148.9 [143.3; 154.5] | 119.2 [114.1; 124.3] |

### Sensibilidad 4: pool de prueba 2023–2025

Escenario principal, h = 1, sin reentrenar; los umbrales de shock son los de cada ventana (historia hasta el origen) y solo cambian los años de prueba que entran en el pool.

**Etiqueta (sens4_pool2023_2025): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 18.72 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/sens4_pool2023_2025/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens4_pool2023_2025/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 211.11 | 99.66 | 111.45 | [103.5; 119.2] | sí | 27.33 | 0.00 | 1089 |
| 360 | ETS | 154.98 | 94.88 | 60.11 | [52.5; 67.7] | sí | 15.34 | 0.00 | 1089 |
| 1000 | ETS | 135.84 | 90.05 | 45.80 | [39.6; 52.1] | sí | 14.41 | 0.00 | 1089 |
| max | SARIMA | 107.49 | 88.77 | 18.72 | [13.4; 24.0] | sí | 6.89 | 9.4e-12 | 1089 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens4_pool2023_2025/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 117.2 [109.2; 125.7] | 117.2 [109.2; 125.7] | 117.2 [109.2; 125.7] | 117.2 [109.2; 125.7] |
| ETS | 424.3 [383.0; 467.1] | 94.9 [87.8; 102.4] | 90.0 [84.0; 96.3] | 89.4 [83.4; 95.5] |
| Theta | 196.9 [182.5; 212.5] | 181.1 [164.5; 198.8] | 101.7 [95.0; 108.8] | 151.5 [143.4; 160.1] |
| SARIMA | 99.7 [92.4; 107.3] | 95.8 [88.8; 103.2] | 95.8 [89.3; 102.7] | 88.8 [82.3; 95.6] |
| Chronos-Bolt-tiny | 268.9 [258.9; 279.2] | 260.3 [249.7; 271.2] | 255.7 [244.6; 267.2] | 254.1 [243.1; 265.6] |
| LSTM univariada | 221.6 [212.3; 230.9] | 135.6 [126.9; 144.4] | 117.9 [110.3; 125.5] | 105.1 [98.1; 112.3] |
| DualLSTM | 211.1 [203.0; 219.7] | 155.0 [147.6; 162.6] | 135.8 [128.8; 143.1] | 107.5 [101.0; 114.2] |

### Sensibilidad 5: bootstrap por bloques de 7 días

Escenario principal, h = 1, sin reentrenar; bloques móviles circulares dentro de cada año.

**Etiqueta (sens5_bloques): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 32.30 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2021-2022-2023-2024-2025; bootstrap bloques). Fuente: `resultados/brazo_D/sens5_bloques/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens5_bloques/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 206.90 | 97.52 | 109.38 | [102.9; 115.8] | sí | 36.56 | 0 | 1818 |
| 360 | ETS | 156.78 | 93.22 | 63.56 | [56.9; 70.1] | sí | 22.66 | 0 | 1818 |
| 1000 | ETS | 137.42 | 90.36 | 47.06 | [41.5; 52.5] | sí | 19.97 | 0 | 1818 |
| max | SARIMA | 121.63 | 89.33 | 32.30 | [27.8; 36.7] | sí | 13.79 | 0 | 1818 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens5_bloques/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 116.5 [107.3; 126.5] | 116.5 [107.3; 126.5] | 116.5 [107.3; 126.5] | 116.5 [107.3; 126.5] |
| ETS | 300.3 [259.8; 344.7] | 93.2 [85.7; 101.6] | 90.4 [83.2; 97.9] | 91.0 [83.9; 98.6] |
| Theta | 192.8 [177.1; 210.1] | 177.2 [156.0; 200.6] | 125.7 [118.3; 133.7] | 159.5 [152.8; 166.4] |
| SARIMA | 97.5 [89.7; 106.1] | 95.5 [87.6; 104.1] | 96.6 [88.9; 104.9] | 89.3 [81.8; 97.4] |
| Chronos-Bolt-tiny | 272.3 [264.9; 279.6] | 262.9 [255.7; 270.2] | 256.8 [249.6; 264.1] | 255.5 [248.4; 262.8] |
| LSTM univariada | 235.6 [226.2; 245.2] | 152.6 [142.1; 163.5] | 135.3 [125.8; 145.3] | 125.8 [117.3; 134.7] |
| DualLSTM | 206.9 [199.6; 214.5] | 156.8 [149.3; 164.6] | 137.4 [129.9; 145.2] | 121.6 [114.1; 129.5] |

### Sensibilidad 6: historia máxima desde 2006-01-01

h = 1. La escala MASE es la de la historia máxima desde 2006.

**Etiqueta (sens6_desde2006): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 31.37 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2021-2022-2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/sens6_desde2006/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens6_desde2006/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 206.90 | 97.52 | 109.38 | [103.7; 115.1] | sí | 36.56 | 0 | 1818 |
| 360 | ETS | 156.78 | 93.22 | 63.56 | [58.1; 69.0] | sí | 22.66 | 0 | 1818 |
| 1000 | ETS | 137.42 | 90.36 | 47.06 | [42.5; 51.7] | sí | 19.97 | 0 | 1818 |
| max | SARIMA | 120.70 | 89.32 | 31.37 | [27.1; 35.6] | sí | 14.06 | 0 | 1818 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens6_desde2006/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] |
| ETS | 300.3 [275.1; 326.4] | 93.2 [87.7; 98.9] | 90.4 [85.3; 95.6] | 92.1 [87.0; 97.3] |
| Theta | 192.8 [182.0; 204.0] | 177.2 [164.7; 190.1] | 125.7 [119.6; 131.9] | 154.7 [148.0; 161.4] |
| SARIMA | 97.5 [91.7; 103.6] | 95.5 [89.7; 101.5] | 96.6 [90.9; 102.4] | 89.3 [83.8; 94.9] |
| Chronos-Bolt-tiny | 272.3 [264.3; 280.4] | 262.9 [254.4; 271.4] | 256.8 [247.9; 265.6] | 255.5 [246.7; 264.3] |
| LSTM univariada | 235.6 [227.8; 243.5] | 152.6 [146.0; 159.5] | 135.3 [129.4; 141.5] | 127.1 [121.3; 133.1] |
| DualLSTM | 206.9 [200.8; 213.2] | 156.8 [151.1; 162.5] | 137.4 [132.0; 142.9] | 120.7 [115.3; 126.2] |

### Sensibilidad 7: anomalía de 2021-02-01 con el valor publicado (2 351 t)

h = 1; todos los modelos reentrenados con la serie con el valor publicado.

**Etiqueta (sens7_anomalia_publicada): «el modelo»** (detalle de `decision.csv`: «Δ(máx) > 0 y significativa»; monótona decreciente = sí; Δ(máx) = 32.67 t; significativa en máx = sí; alguna Δ significativamente positiva = sí; h = 1; años 2021-2022-2023-2024-2025; bootstrap iid). Fuente: `resultados/brazo_D/sens7_anomalia_publicada/decision.csv`.

Δ(N). Fuente: `resultados/brazo_D/sens7_anomalia_publicada/delta.csv`.

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ (t) | IC 95 % Δ | significativa | HLN | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 120 | SARIMA | 207.87 | 98.23 | 109.64 | [103.9; 115.4] | sí | 36.51 | 0 | 1819 |
| 360 | ETS | 157.66 | 93.99 | 63.67 | [58.3; 69.0] | sí | 22.56 | 0 | 1819 |
| 1000 | ETS | 139.84 | 92.58 | 47.27 | [42.7; 52.0] | sí | 19.47 | 0 | 1819 |
| max | SARIMA | 122.66 | 90.00 | 32.67 | [28.1; 37.2] | sí | 13.75 | 0 | 1819 |

MAE por N y modelo (t; pool). Fuente: `resultados/brazo_D/sens7_anomalia_publicada/metricas.csv`.

| Modelo | N=120 | N=360 | N=1000 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 117.7 [110.8; 124.7] | 117.7 [110.8; 124.7] | 117.7 [110.8; 124.7] | 117.7 [110.8; 124.7] |
| ETS | 301.0 [275.7; 327.0] | 94.0 [88.3; 99.8] | 92.6 [87.3; 97.9] | 91.9 [86.6; 97.1] |
| Theta | 193.3 [182.7; 204.4] | 178.0 [165.7; 191.2] | 126.5 [120.3; 132.8] | 160.1 [153.2; 166.9] |
| SARIMA | 98.2 [92.3; 104.3] | 96.3 [90.4; 102.3] | 97.4 [91.7; 103.2] | 90.0 [84.3; 95.7] |
| Chronos-Bolt-tiny | 273.2 [264.9; 281.2] | 263.8 [255.0; 272.3] | 257.8 [248.5; 266.6] | 256.5 [247.3; 265.2] |
| LSTM univariada | 236.8 [228.7; 244.9] | 153.9 [147.1; 160.8] | 136.7 [130.5; 143.0] | 125.8 [120.0; 131.9] |
| DualLSTM | 207.9 [201.7; 214.2] | 157.7 [152.0; 163.6] | 139.8 [134.2; 145.5] | 122.7 [117.2; 128.3] |


## 7. Incidencias

Resumen por escenario. Fuente: `resultados/brazo_D/<escenario>/incidencias.csv` (detalle por modelo, año y N).

| escenario | ETS: ajustes finales sin convergir | ETS: especificaciones no convergidas (de 12 por unidad) | ETS: especificaciones que fallaron | SARIMA: sin convergir | Chronos: unidades con truncamiento |
|---|---|---|---|---|---|
| principal | 0 de 20 | 4 | 0 | 0 de 20 | 5 de 20 (todas con N = max) |
| sens1_estricto | 0 de 20 | 1 | 0 | 0 de 20 | 5 de 20 (todas con N = max) |
| sens2_bolsa | 0 de 20 | 3 | 0 | 0 de 20 | 5 de 20 (todas con N = max) |
| sens6_desde2006 | 0 de 20 | 2 | 0 | 0 de 20 | 5 de 20 (todas con N = max) |
| sens7_anomalia_publicada | 0 de 20 | 2 | 0 | 0 de 20 | 5 de 20 (todas con N = max) |

**Chronos.** La longitud máxima de contexto del modelo es 2048. Con N = max el contexto (desde 1997-10-01, miles de días) se trunca a los últimos 2048 días en todos los pronósticos (h = 1: 1826 de 7304 pronósticos de la principal, todos con N = max; h = 7: 1826 de 7304). Con N = 120, 360 y 1000 no hay truncamiento. Fuente: `resultados/brazo_D/principal/incidencias.csv`.

**ETS: especificación elegida por AICc** (número de ventanas de 5 por N). Fuente: `resultados/brazo_D/principal/incidencias.csv`.

| N | especificacion | ventanas |
|---|---|---|
| 1000 | ETS(A,A,A) | 1 |
| 1000 | ETS(A,Ad,M) | 2 |
| 1000 | ETS(A,N,M) | 2 |
| 120 | ETS(A,A,M) | 1 |
| 120 | ETS(A,N,A) | 1 |
| 120 | ETS(A,N,M) | 2 |
| 120 | ETS(M,Ad,M) | 1 |
| 360 | ETS(A,Ad,A) | 1 |
| 360 | ETS(A,Ad,M) | 3 |
| 360 | ETS(M,N,M) | 1 |
| max | ETS(A,Ad,A) | 3 |
| max | ETS(A,N,A) | 2 |

**ETS inestable en ventanas cortas.** Ventanas (año de prueba) en que el MAE anual del ETS supera 250 t (el Seasonal Naive tiene entre 96 y 136 t por año en la principal): los pronósticos son finitos, pero la especificación elegida con N = 120 extrapola mal. Fuente: `resultados/brazo_D/<escenario>/metricas.csv` e `incidencias.csv`.

| escenario | año | N | MAE ETS (t) | especificación elegida |
|---|---|---|---|---|
| principal | 2023 | 120 | 1059.4 | ETS(A,A,M) |
| sens2_bolsa | 2024 | 120 | 773.9 | ETS(M,A,M) |
| sens3_h7 (h = 7) | 2023 | 120 | 1195.7 | ETS(A,A,M) (misma especificación que la principal) |
| sens6_desde2006 | 2023 | 120 | 1059.4 | ETS(A,A,M) |
| sens7_anomalia_publicada | 2023 | 120 | 1059.4 | ETS(A,A,M) |

**Pronósticos negativos (añadido tras la auditoría, hallazgo I2).** No se truncan en 0, porque el protocolo no lo prevé (`docs/DESVIACIONES.md`, D10). En la principal hay 4 638 pronósticos negativos de 365 200 (h = 1 y 7): ETS con N = 120 y h = 1 tiene 125, con mínimo −4 348,5 t; la DualLSTM con N = 120 y h = 1, 620; la LSTM univariada con N = 120 y h = 1, 503. Fuente: `reportes/auditoria/pronosticos_negativos.csv`.

**SARIMA** (principal): 20 de 20 ajustes convergieron; iteraciones del optimizador entre 6 y 91 (máximo permitido 500). Con N < 730 (N = 120, 360) no se usan Fourier. Fuente: `resultados/brazo_D/principal/incidencias.csv`.

**Equivalencia de los métodos rápidos con los del protocolo** (`resultados/brazo_D/equivalencias.csv`; diferencia máxima absoluta en t):

| modelo | anio | N | h | dif_max_abs_t | n_dias |
|---|---|---|---|---|---|
| ETS | 2023 | 360 | 1 | 0.000000 | 6 |
| ETS | 2023 | 360 | 7 | 0.000000 | 6 |
| SARIMA | 2023 | 360 | 1 | 0.000000 | 6 |
| SARIMA | 2023 | 360 | 7 | 0.000000 | 6 |
| ETS | 2025 | 1000 | 1 | 0.000000 | 6 |
| ETS | 2025 | 1000 | 7 | 0.000000 | 6 |
| SARIMA | 2025 | 1000 | 1 | 0.000000 | 6 |
| SARIMA | 2025 | 1000 | 7 | 0.000000 | 6 |
| ETS | 2021 | max | 1 | 0.000000 | 6 |
| ETS | 2021 | max | 7 | 0.000000 | 6 |
| SARIMA | 2021 | max | 1 | 0.000000 | 6 |
| SARIMA | 2021 | max | 7 | 0.000000 | 6 |
| Chronos | 2023 | max | 1 | 1.2e-04 | 3 |
| Chronos | 2023 | max | 7 | 3.1e-05 | 3 |

**Sensibilidad 6: comprobación de series idénticas** (`resultados/brazo_D/sens6_comprobacion.csv`):

| comprobacion | resultado |
|---|---|
| mismo indice desde 2006-01-01 | True |
| y identico (max abs dif) | 0.0 |
| precio identico (max abs dif) | 0.0 |
| y_eval identico (NaN en las mismas posiciones y valores iguales) | True |
| historia, validacion y prueba identicas Y=2021 N=120 | True |
| historia, validacion y prueba identicas Y=2021 N=360 | True |
| historia, validacion y prueba identicas Y=2021 N=1000 | True |
| historia, validacion y prueba identicas Y=2022 N=120 | True |
| historia, validacion y prueba identicas Y=2022 N=360 | True |
| historia, validacion y prueba identicas Y=2022 N=1000 | True |
| historia, validacion y prueba identicas Y=2023 N=120 | True |
| historia, validacion y prueba identicas Y=2023 N=360 | True |
| historia, validacion y prueba identicas Y=2023 N=1000 | True |
| historia, validacion y prueba identicas Y=2024 N=120 | True |
| historia, validacion y prueba identicas Y=2024 N=360 | True |
| historia, validacion y prueba identicas Y=2024 N=1000 | True |
| historia, validacion y prueba identicas Y=2025 N=120 | True |
| historia, validacion y prueba identicas Y=2025 N=360 | True |
| historia, validacion y prueba identicas Y=2025 N=1000 | True |

**Redes.** Parámetros de la DualLSTM: 4 673; LSTM univariada: 1 569. Unidades de red por escenario (estados_redes.csv de cada carpeta): {'principal': 800, 'sens1_estricto': 400, 'sens2_bolsa': 400, 'sens6_desde2006': 400, 'sens7_anomalia_publicada': 400}. Épocas entre 16 y 247 (máximo 300). Fuente: `resultados/brazo_D/<escenario>/estados_redes.csv`.

Fallos de unidades de red durante las corridas: ver la tabla de fases de la sección 1 (columna `fallos`).

## 8. Desviaciones propuestas

Ninguna cambia la etiqueta ni el diseño. Todas se decidieron el 2026-11-17, **antes de ver errores de prueba** (la comprobación de equivalencias se hizo con pronósticos, sin calcular ningún error; los errores se calcularon por primera vez al terminar la corrida principal h = 1 de las redes). No están en `docs/DESVIACIONES.md` (no se modifica `docs/`); YesaR decide si se registran.

1. **ETS y SARIMA: cálculo equivalente, no literal, del «estado actualizado sin reestimar».** El protocolo (§3) pide parámetros fijos y estado actualizado con cada dato. Para no reajustar el modelo 365 veces por ventana se aplica `smooth(params)` una sola vez sobre la serie hasta el final de la prueba y se leen los pronósticos a h pasos desde el estado en τ−h (ETS: `simulate(h, anchor=τ−h)` con errores cero; SARIMA: `get_prediction(dynamic=0)`). Se comprobó contra el procedimiento literal (ETS: `smooth` sobre el prefijo y `forecast(h)`; SARIMA: `extend` y `forecast(h)`) en 3 ventanas, 6 días y h = 1 y 7: diferencia máxima 0 en todas (`resultados/brazo_D/equivalencias.csv`). Efecto: ninguno.
2. **Chronos en lotes.** Los contextos se agrupan en lotes de 64 con relleno por la izquierda. Frente al pronóstico de un solo contexto, la diferencia máxima es de 1,2×10⁻⁴ t (`equivalencias.csv`, fila Chronos). Efecto: despreciable.
3. **Chronos con N = max usa solo los últimos 2 048 días.** Es el caso que el protocolo (§5) dejó como «no comprobado»: la longitud máxima de contexto es 2 048 y con N = max el contexto se trunca en todos los pronósticos (sección 7). Por tanto Chronos con N = max equivale a N = 2 048. No es una desviación, se declara por transparencia.
4. **Theta: clase propia con `period = 7`.** `r2lib.ThetaFijo` del artículo tiene el periodo 12 fijo en el módulo y arrastra rutas de importación del repositorio del artículo; se copió la lógica a `resultados/brazo_D/scripts/estadisticos_lib.py` con periodo 7 y se mantuvieron sus comprobaciones internas (la recursión SES y el pronóstico manual a h = 1 y 7 reproducen `ThetaModelResults`). `ThetaModel` usa los valores por defecto de statsmodels (`deseasonalize`, `method = auto`, prueba de estacionalidad); el método estacional elegido queda en `incidencias.csv`.
5. **ETS: `maxiter = 1000`** (valor por defecto de statsmodels y de `r2lib`). El protocolo no lo fija. Efecto: ninguno sobre la selección, que es por AICc.
6. **SARIMA: pronóstico = exp(media en escala log).** Es la mediana, no la media, de la distribución en toneladas. Regresores de Fourier con `t` = posición del día en la serie del escenario (en la sensibilidad 6 empieza en 2006-01-01); con N < 730 se omiten.
7. **Sensibilidad 6: las redes con N = 120, 360 y 1 000 son las de la principal.** Se comprobó que `serie(esc).y` y `.precio` de la sensibilidad 6 coinciden exactamente con los de la principal desde 2006-01-01 y que historia, validación y prueba son idénticas en cada (Y, N) (`resultados/brazo_D/sens6_comprobacion.csv`). La única diferencia es la variable `t_index` de la rama B (posición en la serie, desplazada por la fecha de inicio), que tras el `StandardScaler` da entradas idénticas bit a bit (el auditor lo comprobó y una unidad reentrenada da diferencia 0: `reportes/auditoria/sens6_tindex.csv`); por eso no se reentrenaron esas 300 unidades (solo se reentrenó N = max, 100 unidades). Los modelos estadísticos y Chronos sí se reejecutaron en todos los N: son idénticos a la principal salvo SARIMA con N = 1 000, cuyos Fourier tienen otra fase de origen (diferencia máxima 0,16 t, media 0,007 t; `resultados/brazo_D/sens6_vs_principal.csv`).
8. **Sensibilidades 3, 4 y 5 no duplican `pronosticos.csv`, `incidencias.csv` ni `estados_redes.csv`.** Usan los de `principal/` (h = 1 y 7 en el mismo archivo); cada carpeta tiene un `LEEME.txt`. Las sensibilidades 4 y 5 son solo otro análisis de los mismos pronósticos.
9. **`pronosticos.csv` incluye todos los días de cada año de prueba** (365 o 366), también los no evaluables; `src.analisis.analizar` evalúa solo los 1 818 evaluables del pool principal.
10. **Cómputo.** Las redes se ejecutaron con 4 procesos en una máquina compartida con el brazo M: la DualLSTM con N = max tardó entre 58 y 257 s por unidad (mediana 100 s; `principal/estados_redes.csv`), más que lo medido con la máquina libre en el paso 0 (medición no guardada en CSV). Los tiempos reales están en la sección 1. Resultados deterministas por unidad (D4), sin efecto.
11. **Error de implementación corregido antes de correr nada.** La primera versión de ETS usaba un ancla de `simulate` desplazada en un día; la comprobación de equivalencias lo detectó y se corrigió antes de lanzar ninguna corrida; el CSV de equivalencias es el de la versión corregida.
