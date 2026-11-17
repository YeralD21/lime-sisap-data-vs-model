# Brazo M (mensual), h = 1: informe del día 3

Todas las cifras salen de los CSV (y, en los tiempos de pared, del JSON `redes_tiempo_pared.json`) de `resultados/brazo_M/` (ruta citada junto a cada tabla). Este informe lo genera `resultados/brazo_M/scripts/05_informe.py`; la sección 8 contiene cifras escritas a mano (conteos de parámetros, 267–315 meses, 80 unidades) que no salen de un CSV; el resto se puede comprobar en los CSV citados y en `src/`.

## 1. Qué se ejecutó

Brazo M (mensual), h = 1, años de prueba 2021–2025, N ∈ {60, 120, 240, max}, rolling origin de `src.ventanas.particion`. Modelos: Seasonal Naive, ETS (12 especificaciones, menor AICc), Theta (θ = 2), SARIMA (1,1,1)(1,1,0)₁₂ congelado del artículo, Chronos-Bolt-tiny (revisión `a0e552de83495b5c28c14c71c374f3e33280b340`, cero disparos, cuantil 0,5), LSTM univariada y DualLSTM-Bahdanau (10 semillas, 0–9). Escenarios con modelos reentrenados: principal, sensibilidad 1 y sensibilidad 7; las sensibilidades 4 y 5 reanalizan los pronósticos principales (no tienen `pronosticos.csv` propio). Código: `resultados/brazo_M/scripts/` (modelos de referencia en `modelos_base.py`; redes con `src.redes.correr_unidades`, n_proc = 2; análisis con `src.analisis`).

Cómputo medido (`resultados/brazo_M/<escenario>/estados_redes.csv`, `redes_tiempo_pared.json`, `base_tiempos.csv`):

| Escenario | Unidades de red (estado.json) | Filas de pronosticos.csv | Segundos de pared de las redes (2 procesos) | Segundos de entrenamiento sumados (estado.json) | Segundos de los 5 modelos de referencia |
|---|---|---|---|---|---|
| principal | 400 | 6000 | 3 281 | 3 933 | 39 |
| sens1_estricto | 400 | 6000 | 4 648 | 5 343 | 38 |
| sens7_anomalia_publicada | 400 | 6000 | 3 138 | 3 950 | 39 |

Tiempo de pared total de las redes: 11 067 s (3.07 h). Cada escenario tiene 5 años × 4 N × 2 redes × 10 semillas = 400 unidades. Los tiempos de las redes incluyen la competencia por CPU con el brazo D y el arranque de un proceso por unidad; los de los modelos de referencia se midieron con un proceso de un hilo (Chronos con 2 hilos).

Parámetros de las redes (`resultados/brazo_M/principal/estados_redes.csv`): DualLSTM 4545; LSTM univariada 1569. `expected_parameter_count(4, 6)` da 4545 (ver sección 8, nota 1).

## 2. Curva de aprendizaje (escenario principal, pool de prueba 2021–2025)

MAE en toneladas con IC 95 % del bootstrap iid pareado (B = 10 000, semilla 42) entre corchetes (`resultados/brazo_M/principal/metricas.csv`, `anio == pool`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 1 523.0 [1 196.4; 1 863.0] | 1 523.0 [1 196.4; 1 863.0] | 1 523.0 [1 196.4; 1 863.0] | 1 523.0 [1 196.4; 1 863.0] |
| ETS | 967.0 [797.1; 1 147.0] | 976.5 [792.7; 1 172.0] | 1 017.8 [824.3; 1 229.4] | 972.5 [794.0; 1 158.5] |
| Theta | 956.2 [782.7; 1 140.1] | 988.9 [811.3; 1 175.8] | 1 012.0 [836.7; 1 201.5] | 1 050.9 [875.2; 1 234.8] |
| SARIMA | 2 130.9 [1 834.4; 2 439.3] | 1 334.5 [1 116.8; 1 558.0] | 1 336.3 [1 117.0; 1 560.4] | 1 324.2 [1 079.4; 1 581.1] |
| Chronos-Bolt-tiny | 1 101.5 [877.4; 1 342.9] | 1 038.7 [841.5; 1 252.8] | 1 030.2 [841.2; 1 232.8] | 1 029.5 [838.7; 1 236.5] |
| LSTM univariada | 1 396.3 [1 138.4; 1 679.0] | 1 218.8 [1 003.9; 1 448.1] | 1 266.1 [1 074.1; 1 473.4] | 1 288.2 [1 086.0; 1 505.7] |
| DualLSTM | 1 842.2 [1 545.6; 2 159.0] | 1 436.7 [1 191.8; 1 697.9] | 1 354.0 [1 133.4; 1 595.1] | 1 333.1 [1 128.9; 1 553.9] |

RMSE en toneladas (`resultados/brazo_M/principal/metricas.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 2 094.5 | 2 094.5 | 2 094.5 | 2 094.5 |
| ETS | 1 228.1 | 1 249.6 | 1 314.6 | 1 235.1 |
| Theta | 1 237.2 | 1 244.2 | 1 263.0 | 1 285.0 |
| SARIMA | 3 062.6 | 1 667.4 | 1 671.8 | 1 729.0 |
| Chronos-Bolt-tiny | 1 442.3 | 1 320.7 | 1 295.8 | 1 299.0 |
| LSTM univariada | 1 813.4 | 1 537.7 | 1 506.0 | 1 550.1 |
| DualLSTM | 2 291.9 | 1 827.5 | 1 751.8 | 1 725.0 |

MASE, escala fija por ventana (`resultados/brazo_M/principal/metricas.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 1.409 | 1.409 | 1.409 | 1.409 |
| ETS | 0.900 | 0.908 | 0.946 | 0.904 |
| Theta | 0.891 | 0.919 | 0.941 | 0.977 |
| SARIMA | 1.968 | 1.239 | 1.241 | 1.230 |
| Chronos-Bolt-tiny | 1.024 | 0.965 | 0.957 | 0.957 |
| LSTM univariada | 1.299 | 1.133 | 1.177 | 1.197 |
| DualLSTM | 1.708 | 1.331 | 1.256 | 1.237 |

## 3. ref*(N), Δ(N) y DM-HLN

Δ(N) = MAE_DualLSTM(N) − MAE_ref*(N), IC 95 % del bootstrap pareado y prueba Diebold-Mariano con corrección HLN (`resultados/brazo_M/principal/delta.csv`):

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ(N) | IC 95 % de Δ | IC excluye 0 | DM-HLN (estadístico) | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Theta | 1 842.2 | 956.2 | 886.0 | [575.5; 1 196.5] | sí | 5.065 | <0,0001 | 59 |
| 120 | ETS | 1 436.7 | 976.5 | 460.1 | [220.6; 696.6] | sí | 3.539 | 0.0008 | 59 |
| 240 | Theta | 1 354.0 | 1 012.0 | 342.1 | [126.9; 553.4] | sí | 2.940 | 0.0047 | 59 |
| max | ETS | 1 333.1 | 972.5 | 360.6 | [159.3; 561.7] | sí | 3.264 | 0.0018 | 59 |

## 4. Etiqueta de la regla de decisión (escenario principal)

Etiqueta (`resultados/brazo_M/principal/decision.csv`): **«el modelo»**. Detalle: Δ(máx) > 0 y significativa. Monótona decreciente: False; Δ(máx) = 360.6 t; significativa en N = máx: True; alguna Δ significativamente positiva: True; años: 2021-2022-2023-2024-2025; bootstrap: iid.

Copia literal de `resultados/brazo_M/principal/decision.csv`:

```
monotona_decreciente,delta_max,significativa_max,alguna_significativa_positiva,etiqueta,detalle,brazo,h,anios,bootstrap
False,360.608,True,True,el modelo,Δ(máx) > 0 y significativa,M,1,2021-2022-2023-2024-2025,iid
```

## 5. Ranking, «consistently top-ranked», años y semillas

Posición por MAE en el pool, si es primero o empata con el primero, y número de umbrales (P50, P60, P75, P90) en que es primero o empata; TOP = «consistently top-ranked» (≥ 3 de 4) (`resultados/brazo_M/principal/ranking.csv`, `top_ranked.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | pos 5; umbrales 3/4 TOP | pos 7; umbrales 4/4 TOP | pos 7; umbrales 4/4 TOP | pos 7; umbrales 4/4 TOP |
| ETS | pos 2 (primero/empate); umbrales 1/4 | pos 1 (primero/empate); umbrales 3/4 TOP | pos 2 (primero/empate); umbrales 3/4 TOP | pos 1 (primero/empate); umbrales 3/4 TOP |
| Theta | pos 1 (primero/empate); umbrales 3/4 TOP | pos 2 (primero/empate); umbrales 3/4 TOP | pos 1 (primero/empate); umbrales 3/4 TOP | pos 3; umbrales 3/4 TOP |
| SARIMA | pos 7; umbrales 0/4 | pos 5; umbrales 3/4 TOP | pos 5; umbrales 3/4 TOP | pos 5; umbrales 3/4 TOP |
| Chronos-Bolt-tiny | pos 3 (primero/empate); umbrales 0/4 | pos 3 (primero/empate); umbrales 0/4 | pos 3 (primero/empate); umbrales 0/4 | pos 2 (primero/empate); umbrales 0/4 |
| LSTM univariada | pos 4; umbrales 0/4 | pos 4; umbrales 0/4 | pos 4; umbrales 1/4 | pos 4; umbrales 0/4 |
| DualLSTM | pos 6; umbrales 0/4 | pos 6; umbrales 2/4 | pos 6; umbrales 1/4 | pos 6; umbrales 2/4 |

MAE por año de prueba, toneladas (`resultados/brazo_M/principal/metricas.csv`):

| Modelo | N | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Seasonal Naive | 60 | 765.7 | 1 821.2 | 1 394.1 | 2 528.4 | 1 042.2 |
| Seasonal Naive | 120 | 765.7 | 1 821.2 | 1 394.1 | 2 528.4 | 1 042.2 |
| Seasonal Naive | 240 | 765.7 | 1 821.2 | 1 394.1 | 2 528.4 | 1 042.2 |
| Seasonal Naive | max | 765.7 | 1 821.2 | 1 394.1 | 2 528.4 | 1 042.2 |
| ETS | 60 | 949.8 | 900.6 | 1 307.5 | 1 273.6 | 402.2 |
| ETS | 120 | 861.1 | 1 032.6 | 1 313.7 | 1 120.0 | 545.6 |
| ETS | 240 | 921.3 | 1 039.6 | 1 299.4 | 1 174.4 | 646.1 |
| ETS | max | 860.1 | 967.7 | 1 350.5 | 1 111.3 | 563.5 |
| Theta | 60 | 1 122.5 | 809.8 | 1 402.1 | 1 104.8 | 355.5 |
| Theta | 120 | 931.3 | 983.3 | 1 296.5 | 1 183.3 | 545.1 |
| Theta | 240 | 945.7 | 996.9 | 1 380.6 | 1 097.3 | 633.7 |
| Theta | max | 1 012.4 | 988.6 | 1 344.2 | 1 220.8 | 685.2 |
| SARIMA | 60 | 910.1 | 1 736.0 | 1 642.2 | 5 671.7 | 593.0 |
| SARIMA | 120 | 968.0 | 1 813.2 | 1 411.4 | 1 865.5 | 583.9 |
| SARIMA | 240 | 982.4 | 1 944.5 | 1 368.4 | 1 760.8 | 596.0 |
| SARIMA | max | 980.0 | 1 937.9 | 1 361.9 | 1 749.0 | 563.4 |
| Chronos-Bolt-tiny | 60 | 989.7 | 1 216.1 | 1 410.5 | 981.1 | 900.8 |
| Chronos-Bolt-tiny | 120 | 952.3 | 1 130.1 | 1 225.2 | 1 044.9 | 833.9 |
| Chronos-Bolt-tiny | 240 | 988.5 | 1 130.2 | 1 254.2 | 947.4 | 827.2 |
| Chronos-Bolt-tiny | max | 982.4 | 1 121.6 | 1 232.0 | 976.8 | 830.8 |
| LSTM univariada | 60 | 1 663.5 | 1 152.2 | 1 877.9 | 970.5 | 1 339.6 |
| LSTM univariada | 120 | 1 366.5 | 1 222.4 | 1 501.2 | 885.6 | 1 130.5 |
| LSTM univariada | 240 | 1 401.6 | 1 182.7 | 1 524.0 | 1 083.6 | 1 150.0 |
| LSTM univariada | max | 1 428.2 | 1 144.6 | 1 549.9 | 1 162.5 | 1 167.4 |
| DualLSTM | 60 | 1 057.9 | 2 532.6 | 1 978.8 | 2 145.1 | 1 431.3 |
| DualLSTM | 120 | 737.5 | 2 030.3 | 1 572.0 | 1 543.6 | 1 241.6 |
| DualLSTM | 240 | 816.1 | 1 760.8 | 1 550.0 | 1 758.5 | 839.8 |
| DualLSTM | max | 789.9 | 1 687.8 | 1 535.6 | 1 851.1 | 755.9 |

Dispersión entre semillas de las redes (`resultados/brazo_M/principal/semillas.csv`):

| Modelo | N | semillas | MAE medio | DE entre semillas | mín | máx | MAE del promedio de semillas |
|---|---|---|---|---|---|---|---|
| LSTM univariada | 60 | 10 | 1 396.3 | 27.8 | 1 356.4 | 1 437.5 | 1 396.3 |
| DualLSTM | 60 | 10 | 1 842.2 | 97.4 | 1 725.4 | 1 985.5 | 1 842.2 |
| LSTM univariada | 120 | 10 | 1 218.8 | 18.1 | 1 187.7 | 1 244.6 | 1 218.8 |
| DualLSTM | 120 | 10 | 1 436.7 | 129.2 | 1 259.3 | 1 672.9 | 1 436.7 |
| LSTM univariada | 240 | 10 | 1 266.1 | 16.9 | 1 236.6 | 1 289.1 | 1 266.1 |
| DualLSTM | 240 | 10 | 1 354.0 | 163.0 | 1 197.2 | 1 602.7 | 1 354.0 |
| LSTM univariada | max | 10 | 1 288.2 | 28.4 | 1 260.3 | 1 330.5 | 1 288.2 |
| DualLSTM | max | 10 | 1 333.1 | 188.2 | 1 132.6 | 1 642.3 | 1 333.1 |

Entrenamiento de las redes (`resultados/brazo_M/principal/estados_redes.csv`):

| modelo | unidades | n_params | epocas_mediana | epocas_max | mejor_epoca_mediana | segundos_suma |
|---|---|---|---|---|---|---|
| DualLSTM | 200 | 4545 | 31.5 | 167 | 16.5 | 2 099 |
| LSTM univariada | 200 | 1569 | 52.5 | 199 | 37.5 | 1 833 |

## 6. Sensibilidades (no cambian la etiqueta principal)

### Sensibilidad 1 («...» = faltante)

Etiqueta (`resultados/brazo_M/sens1_estricto/decision.csv`): **«no concluyente»**. Detalle: no se reproduce la desventaja. Monótona decreciente: False; Δ(máx) = 205.4 t; significativa en N = máx: False; alguna Δ significativamente positiva: False; años: 2021-2022-2023-2024; bootstrap: iid.

Δ(N) (`resultados/brazo_M/sens1_estricto/delta.csv`), toneladas:

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ(N) | IC 95 % de Δ | IC excluye 0 | DM-HLN (estadístico) | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Theta | 1 758.6 | 1 559.8 | 198.8 | [-546.2; 907.2] | no | 0.473 | 0.6426 | 17 |
| 120 | Theta | 1 572.4 | 1 262.8 | 309.6 | [-216.6; 833.1] | no | 0.949 | 0.3569 | 17 |
| 240 | ETS | 1 621.2 | 1 275.9 | 345.3 | [-139.6; 827.1] | no | 1.126 | 0.2767 | 17 |
| max | ETS | 1 572.6 | 1 367.2 | 205.4 | [-293.2; 700.3] | no | 0.626 | 0.5400 | 17 |

MAE por modelo y N, pool (`resultados/brazo_M/sens1_estricto/metricas.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 2 468.2 [1 809.9; 3 132.9] | 2 468.2 [1 809.9; 3 132.9] | 2 468.2 [1 809.9; 3 132.9] | 2 468.2 [1 809.9; 3 132.9] |
| ETS | 1 702.1 [1 160.4; 2 297.5] | 1 334.7 [785.6; 1 936.3] | 1 275.9 [717.9; 1 912.8] | 1 367.2 [821.0; 1 977.3] |
| Theta | 1 559.8 [1 022.5; 2 144.3] | 1 262.8 [729.4; 1 862.5] | 1 290.7 [756.2; 1 902.0] | 1 432.5 [882.8; 2 070.6] |
| SARIMA | 3 404.9 [2 706.5; 4 187.5] | 2 565.3 [1 978.1; 3 163.5] | 2 358.9 [1 791.9; 2 953.9] | 2 340.8 [1 767.7; 2 934.8] |
| Chronos-Bolt-tiny | 1 637.2 [1 149.9; 2 195.9] | 1 409.5 [888.6; 1 989.1] | 1 402.6 [913.7; 1 924.0] | 1 387.0 [892.7; 1 914.1] |
| LSTM univariada | 1 552.3 [1 059.5; 2 059.0] | 1 552.8 [1 085.1; 2 048.5] | 1 696.6 [1 243.1; 2 168.0] | 1 807.4 [1 336.5; 2 290.9] |
| DualLSTM | 1 758.6 [1 407.2; 2 119.5] | 1 572.4 [1 279.5; 1 865.7] | 1 621.2 [1 323.1; 1 931.5] | 1 572.6 [1 247.5; 1 908.7] |

Meses evaluables por año (`resultados/brazo_M/sens1_estricto/metricas.csv`): 2021: 6; 2022: 2; 2023: 5; 2024: 4. El año 2025 no tiene meses evaluables y queda fuera del pool (ver D5).

Ranking y top-ranked (`resultados/brazo_M/sens1_estricto/ranking.csv`, `top_ranked.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | pos 6; umbrales 3/4 TOP | pos 6; umbrales 1/4 | pos 7; umbrales 2/4 | pos 7; umbrales 1/4 |
| ETS | pos 4 (primero/empate); umbrales 2/4 | pos 2 (primero/empate); umbrales 2/4 | pos 1 (primero/empate); umbrales 2/4 | pos 1 (primero/empate); umbrales 2/4 |
| Theta | pos 2 (primero/empate); umbrales 3/4 TOP | pos 1 (primero/empate); umbrales 2/4 | pos 2 (primero/empate); umbrales 2/4 | pos 3 (primero/empate); umbrales 2/4 |
| SARIMA | pos 7; umbrales 0/4 | pos 7; umbrales 0/4 | pos 6; umbrales 0/4 | pos 6; umbrales 0/4 |
| Chronos-Bolt-tiny | pos 3 (primero/empate); umbrales 2/4 | pos 3 (primero/empate); umbrales 2/4 | pos 3 (primero/empate); umbrales 2/4 | pos 2 (primero/empate); umbrales 1/4 |
| LSTM univariada | pos 1 (primero/empate); umbrales 0/4 | pos 4 (primero/empate); umbrales 0/4 | pos 5 (primero/empate); umbrales 1/4 | pos 5 (primero/empate); umbrales 0/4 |
| DualLSTM | pos 5 (primero/empate); umbrales 2/4 | pos 5 (primero/empate); umbrales 3/4 TOP | pos 4 (primero/empate); umbrales 3/4 TOP | pos 4 (primero/empate); umbrales 4/4 TOP |

### Sensibilidad 4 (pool 2023–2025)

Etiqueta (`resultados/brazo_M/sens4_pool2023_2025/decision.csv`): **«el modelo»**. Detalle: Δ(máx) > 0 y significativa. Monótona decreciente: True; Δ(máx) = 372.4 t; significativa en N = máx: True; alguna Δ significativamente positiva: True; años: 2023-2024-2025; bootstrap: iid.

Δ(N) (`resultados/brazo_M/sens4_pool2023_2025/delta.csv`), toneladas:

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ(N) | IC 95 % de Δ | IC excluye 0 | DM-HLN (estadístico) | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Theta | 1 851.7 | 954.1 | 897.6 | [517.2; 1 298.5] | sí | 4.297 | 0.0001 | 36 |
| 120 | ETS | 1 452.4 | 993.1 | 459.3 | [185.3; 747.7] | sí | 3.053 | 0.0043 | 36 |
| 240 | Chronos-Bolt-tiny | 1 382.8 | 1 009.6 | 373.2 | [160.8; 595.5] | sí | 2.962 | 0.0055 | 36 |
| max | ETS | 1 380.9 | 1 008.5 | 372.4 | [169.7; 591.5] | sí | 3.173 | 0.0031 | 36 |

MAE por modelo y N, pool (`resultados/brazo_M/sens4_pool2023_2025/metricas.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 1 654.9 [1 212.6; 2 137.9] | 1 654.9 [1 212.6; 2 137.9] | 1 654.9 [1 212.6; 2 137.9] | 1 654.9 [1 212.6; 2 137.9] |
| ETS | 994.4 [797.2; 1 215.0] | 993.1 [774.2; 1 237.2] | 1 040.0 [788.3; 1 312.5] | 1 008.5 [792.8; 1 242.3] |
| Theta | 954.1 [759.5; 1 165.7] | 1 008.3 [789.1; 1 249.9] | 1 037.2 [816.7; 1 273.1] | 1 083.4 [870.8; 1 311.8] |
| SARIMA | 2 635.6 [2 194.7; 3 083.8] | 1 286.9 [1 022.6; 1 591.1] | 1 241.7 [972.2; 1 541.9] | 1 224.8 [894.0; 1 583.2] |
| Chronos-Bolt-tiny | 1 097.4 [832.4; 1 383.7] | 1 034.7 [812.1; 1 281.4] | 1 009.6 [788.7; 1 252.7] | 1 013.2 [787.5; 1 259.9] |
| LSTM univariada | 1 396.0 [1 047.1; 1 793.8] | 1 172.4 [928.1; 1 451.3] | 1 252.5 [1 026.9; 1 486.1] | 1 293.3 [1 059.0; 1 537.9] |
| DualLSTM | 1 851.7 [1 441.8; 2 297.2] | 1 452.4 [1 123.4; 1 813.7] | 1 382.8 [1 094.9; 1 714.1] | 1 380.9 [1 122.8; 1 679.1] |

### Sensibilidad 5 (bootstrap por bloques de 3 meses)

Etiqueta (`resultados/brazo_M/sens5_bloques/decision.csv`): **«el modelo»**. Detalle: Δ(máx) > 0 y significativa. Monótona decreciente: False; Δ(máx) = 360.6 t; significativa en N = máx: True; alguna Δ significativamente positiva: True; años: 2021-2022-2023-2024-2025; bootstrap: bloques.

Δ(N) (`resultados/brazo_M/sens5_bloques/delta.csv`), toneladas:

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ(N) | IC 95 % de Δ | IC excluye 0 | DM-HLN (estadístico) | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Theta | 1 842.2 | 956.2 | 886.0 | [544.1; 1 254.8] | sí | 5.065 | <0,0001 | 59 |
| 120 | ETS | 1 436.7 | 976.5 | 460.1 | [197.6; 734.9] | sí | 3.539 | 0.0008 | 59 |
| 240 | Theta | 1 354.0 | 1 012.0 | 342.1 | [127.5; 571.5] | sí | 2.940 | 0.0047 | 59 |
| max | ETS | 1 333.1 | 972.5 | 360.6 | [180.1; 556.4] | sí | 3.264 | 0.0018 | 59 |

MAE por modelo y N, pool (`resultados/brazo_M/sens5_bloques/metricas.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 1 523.0 [1 170.8; 1 926.3] | 1 523.0 [1 170.8; 1 926.3] | 1 523.0 [1 170.8; 1 926.3] | 1 523.0 [1 170.8; 1 926.3] |
| ETS | 967.0 [813.9; 1 127.4] | 976.5 [787.9; 1 172.6] | 1 017.8 [822.2; 1 225.1] | 972.5 [795.6; 1 156.6] |
| Theta | 956.2 [798.4; 1 119.5] | 988.9 [803.3; 1 174.8] | 1 012.0 [829.5; 1 203.5] | 1 050.9 [883.0; 1 221.8] |
| SARIMA | 2 130.9 [1 752.3; 2 536.8] | 1 334.5 [1 120.8; 1 554.8] | 1 336.3 [1 123.0; 1 552.3] | 1 324.2 [1 071.6; 1 582.7] |
| Chronos-Bolt-tiny | 1 101.5 [849.4; 1 374.0] | 1 038.7 [829.2; 1 255.4] | 1 030.2 [830.4; 1 239.4] | 1 029.5 [823.8; 1 245.3] |
| LSTM univariada | 1 396.3 [1 079.1; 1 761.7] | 1 218.8 [993.6; 1 468.9] | 1 266.1 [1 057.6; 1 493.3] | 1 288.2 [1 064.1; 1 530.8] |
| DualLSTM | 1 842.2 [1 487.6; 2 238.3] | 1 436.7 [1 170.2; 1 726.2] | 1 354.0 [1 123.7; 1 613.4] | 1 333.1 [1 123.2; 1 562.7] |

### Sensibilidad 7 (anomalía con el valor publicado)

Etiqueta (`resultados/brazo_M/sens7_anomalia_publicada/decision.csv`): **«el modelo»**. Detalle: Δ(máx) > 0 y significativa. Monótona decreciente: False; Δ(máx) = 359.3 t; significativa en N = máx: True; alguna Δ significativamente positiva: True; años: 2021-2022-2023-2024-2025; bootstrap: iid.

Δ(N) (`resultados/brazo_M/sens7_anomalia_publicada/delta.csv`), toneladas:

| N | ref*(N) | MAE DualLSTM | MAE ref* | Δ(N) | IC 95 % de Δ | IC excluye 0 | DM-HLN (estadístico) | p DM-HLN | n |
|---|---|---|---|---|---|---|---|---|---|
| 60 | ETS | 1 826.4 | 947.1 | 879.3 | [592.5; 1 167.9] | sí | 5.435 | <0,0001 | 60 |
| 120 | ETS | 1 445.2 | 971.3 | 473.9 | [237.3; 709.9] | sí | 3.629 | 0.0006 | 60 |
| 240 | Theta | 1 342.9 | 999.9 | 343.0 | [129.5; 551.7] | sí | 2.935 | 0.0047 | 60 |
| max | ETS | 1 325.4 | 966.0 | 359.3 | [159.1; 551.8] | sí | 3.249 | 0.0019 | 60 |

MAE por modelo y N, pool (`resultados/brazo_M/sens7_anomalia_publicada/metricas.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | 1 519.3 [1 198.3; 1 861.0] | 1 519.3 [1 198.3; 1 861.0] | 1 519.3 [1 198.3; 1 861.0] | 1 519.3 [1 198.3; 1 861.0] |
| ETS | 947.1 [776.9; 1 133.0] | 971.3 [793.8; 1 166.8] | 1 001.6 [804.6; 1 214.3] | 966.0 [792.6; 1 154.0] |
| Theta | 968.3 [802.7; 1 150.8] | 985.0 [814.0; 1 170.0] | 999.9 [826.0; 1 188.7] | 1 038.2 [864.5; 1 221.8] |
| SARIMA | 2 108.2 [1 817.1; 2 408.9] | 1 333.5 [1 124.8; 1 565.2] | 1 333.8 [1 120.0; 1 566.8] | 1 339.9 [1 096.3; 1 594.3] |
| Chronos-Bolt-tiny | 1 083.4 [863.5; 1 321.4] | 1 031.6 [836.0; 1 246.6] | 1 014.4 [825.7; 1 219.7] | 1 013.7 [823.0; 1 221.1] |
| LSTM univariada | 1 406.6 [1 150.0; 1 686.7] | 1 237.9 [1 031.6; 1 460.0] | 1 252.3 [1 063.1; 1 452.0] | 1 289.1 [1 091.5; 1 497.2] |
| DualLSTM | 1 826.4 [1 537.0; 2 136.0] | 1 445.2 [1 200.7; 1 705.9] | 1 342.9 [1 122.0; 1 579.7] | 1 325.4 [1 123.0; 1 541.2] |

Ranking y top-ranked (`resultados/brazo_M/sens7_anomalia_publicada/ranking.csv`, `top_ranked.csv`):

| Modelo | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| Seasonal Naive | pos 5; umbrales 3/4 TOP | pos 7; umbrales 4/4 TOP | pos 7; umbrales 4/4 TOP | pos 7; umbrales 4/4 TOP |
| ETS | pos 1 (primero/empate); umbrales 3/4 TOP | pos 1 (primero/empate); umbrales 3/4 TOP | pos 2 (primero/empate); umbrales 3/4 TOP | pos 1 (primero/empate); umbrales 3/4 TOP |
| Theta | pos 2 (primero/empate); umbrales 3/4 TOP | pos 2 (primero/empate); umbrales 3/4 TOP | pos 1 (primero/empate); umbrales 3/4 TOP | pos 3; umbrales 3/4 TOP |
| SARIMA | pos 7; umbrales 0/4 | pos 5; umbrales 3/4 TOP | pos 5; umbrales 3/4 TOP | pos 6; umbrales 2/4 |
| Chronos-Bolt-tiny | pos 3 (primero/empate); umbrales 0/4 | pos 3 (primero/empate); umbrales 0/4 | pos 3 (primero/empate); umbrales 0/4 | pos 2 (primero/empate); umbrales 0/4 |
| LSTM univariada | pos 4; umbrales 0/4 | pos 4; umbrales 0/4 | pos 4; umbrales 1/4 | pos 4; umbrales 0/4 |
| DualLSTM | pos 6; umbrales 0/4 | pos 6; umbrales 2/4 | pos 6; umbrales 1/4 | pos 5; umbrales 3/4 TOP |

## 7. Incidencias

Principal (`resultados/brazo_M/principal/incidencias.csv`):

Especificación ETS elegida por AICc (`incidencias.csv`, tipo `ets_seleccion`):

| Año | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| 2021 | ETS(A,N,M) | ETS(M,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2022 | ETS(A,N,A) | ETS(M,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2023 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,A,M) | ETS(A,N,M) |
| 2024 | ETS(M,N,A) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2025 | ETS(A,N,A) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |

ETS elegida que no convergió: 0 de 20 ventana×N.
Especificaciones ETS con fallo de ajuste: 0 (sobre 240 ajustes); sin convergencia: 0.

SARIMA: 20 de 20 ajustes convergieron; 0 no. Avisos distintos de «ninguno»: 0.

Chronos: 0 de 20 ventana×N con contexto truncado (contexto máximo solicitado: 315 observaciones; límite del modelo: 2048).
Fallos totales o pronósticos ETS no finitos: 0.


Sensibilidad 1 («...» = faltante) (`resultados/brazo_M/sens1_estricto/incidencias.csv`):

Especificación ETS elegida por AICc (`incidencias.csv`, tipo `ets_seleccion`):

| Año | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| 2021 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2022 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2023 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,A,M) | ETS(A,N,M) |
| 2024 | ETS(M,A,M) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2025 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |

ETS elegida que no convergió: 0 de 20 ventana×N.
Especificaciones ETS con fallo de ajuste: 0 (sobre 240 ajustes); sin convergencia: 0.

SARIMA: 20 de 20 ajustes convergieron; 0 no. Avisos distintos de «ninguno»: 0.

Chronos: 0 de 20 ventana×N con contexto truncado (contexto máximo solicitado: 315 observaciones; límite del modelo: 2048).
Fallos totales o pronósticos ETS no finitos: 0.


Sensibilidad 7 (anomalía con el valor publicado) (`resultados/brazo_M/sens7_anomalia_publicada/incidencias.csv`):

Especificación ETS elegida por AICc (`incidencias.csv`, tipo `ets_seleccion`):

| Año | N=60 | N=120 | N=240 | N=max |
|---|---|---|---|---|
| 2021 | ETS(A,N,M) | ETS(M,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2022 | ETS(A,N,A) | ETS(M,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2023 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,A,M) | ETS(A,N,M) |
| 2024 | ETS(M,N,M) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |
| 2025 | ETS(A,N,M) | ETS(A,N,M) | ETS(M,N,M) | ETS(A,N,M) |

ETS elegida que no convergió: 0 de 20 ventana×N.
Especificaciones ETS con fallo de ajuste: 0 (sobre 240 ajustes); sin convergencia: 0.

SARIMA: 20 de 20 ajustes convergieron; 0 no. Avisos distintos de «ninguno»: 0.

Chronos: 0 de 20 ventana×N con contexto truncado (contexto máximo solicitado: 315 observaciones; límite del modelo: 2048).
Fallos totales o pronósticos ETS no finitos: 0.

## 8. Desviaciones propuestas

Ninguna de las decisiones siguientes cambia un valor fijado por `docs/PROTOCOLO.md`; son interpretaciones donde el protocolo es ambiguo o aclaraciones de ejecución. Todas se tomaron el 2026-11-17 antes de lanzar la corrida completa y antes de calcular cualquier error de prueba de la corrida final. Una prueba previa con una sola ventana (Y = 2023, N = 120, escenario principal) imprimió el MAE de los cinco modelos de referencia en esa ventana; no se cambió nada después de verlo y los archivos de la prueba se borraron. Para el integrador, que decidirá cuáles pasar a `docs/DESVIACIONES.md`:

1. **Conteo de parámetros de la DualLSTM (no es una desviación del protocolo).** El encargo decía 4 289, pero `src.redes.expected_parameter_count(4, 6)` da 4 545 (LSTM rama A 1 344, rama B 1 472, dos atenciones 1 056, Dense(16) 528, Dense(8) 136, salida 9) y `tests/test_comun.py` fija `expected_parameter_count(4, 8) = 4 673` (= 4 545 + 128). Las 400 unidades del escenario principal reportan 4 545 (`resultados/brazo_M/principal/estados_redes.csv`); la fórmula del artículo es la que manda (§5). El 4 289 del encargo parece un error de transcripción.
2. **Contexto de Chronos en N = max.** §5 dice «las últimas N observaciones hasta t − 1». Para N numérico se usa exactamente eso. Para N = max se usó el tamaño de la historia máxima de la ventana (`Particion.n_hist`, 267–315 meses) como longitud fija del contexto, que termina en t − 1 y por tanto incluye validación y los meses de prueba ya observados. La alternativa (historia creciente desde 1997-10) no se ejecutó. Efecto: ninguna truncación (máximo 315 frente al límite de 2048).
3. **Selección de ETS.** Se elige la especificación de menor AICc entre las que producen AICc finito, aunque el optimizador no haya convergido (como `r2lib.seleccionar_ets`); la convergencia de la elegida y el número de especificaciones fallidas o sin convergencia se registran en `incidencias.csv`. El protocolo no dice qué hacer con las no convergidas.
4. **Estado inicial de los modelos de espacio de estado.** ETS, Theta y SARIMA se ajustan con la historia de tamaño N y su estado se recalcula desde el primer mes de esa historia con los datos reales (imputados donde falte) hasta t − 1, sin reestimar parámetros. En Theta, SARIMA (`append(refit=False)`) y ETS (`smooth(params)`) el procedimiento es el de `r2lib.py` y `lib/modelos_series.py` del artículo. Se comprobó que añadir de una vez la validación y la prueba a SARIMA da los mismos pronósticos a un paso que añadirlos uno a uno (comprobación puntual en Y = 2023, N = 120, no guardada en CSV; la reproducción independiente del auditor da una diferencia máxima de 4,7e-6 t para SARIMA en esa ventana, `reportes/auditoria/reproduccion_M.csv`).
5. **Sensibilidad 1 con la ventana 2025.** Se entrenaron y pronosticaron también las 80 unidades de red y los modelos de referencia de 2025 en la sensibilidad 1, aunque esa ventana no tiene meses evaluables y no entra en el pool. Efecto sobre resultados: ninguno (solo cómputo).
6. **Sensibilidades 4 y 5.** No tienen `pronosticos.csv` propio: reutilizan `resultados/brazo_M/principal/pronosticos.csv`.
7. **CPU.** Los modelos de referencia (un proceso de un hilo, Chronos con 2 hilos) se ejecutaron mientras corrían las redes con 2 procesos, así que hubo brevemente 3 procesos de este brazo a la vez. Efecto sobre resultados: ninguno (determinismo verificado en D4).
8. **Sensibilidad 1, 17 meses evaluables (D5).** Ya registrado: 6, 2, 5, 4, 0 meses en 2021–2025 (`resultados/brazo_M/sens1_estricto/metricas.csv`).
