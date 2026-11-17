# Experimento 2, día 3: curvas de aprendizaje. Informe integrado

Protocolo congelado: `docs/PROTOCOLO.md` (commit 9ede96c). Desviaciones: `docs/DESVIACIONES.md` (D1–D10). Informes de cada brazo: `reportes/brazo_M.md` y `reportes/brazo_D.md`. Auditoría: `reportes/auditoria/AUDITORIA_DIA3.md`.

Toda cifra de resultados sale de un CSV de `resultados/` (o de `reportes/auditoria/` para la auditoría), con la ruta citada en cada tabla. La única excepción es la estimación previa de cómputo del §1.1, que es una previsión del paso 0 y no un resultado. Los números usan punto decimal, como los CSV.

**Resultado:** los dos brazos dan **«el modelo»** en el análisis principal (D a h = 1 y M a h = 1). La conclusión conjunta es **«el modelo»**: la desventaja de la DualLSTM frente a la mejor referencia se mantiene con la historia máxima y es significativa en los dos brazos. La auditoría no encontró ningún hallazgo bloqueante.

---

## 1. Verificación inicial y tiempo de cómputo

### 1.1 Verificación (paso 0)

- `docs/PROTOCOLO.md` y `docs/hashes_protocolo.json` no cambiaron entre 9ede96c y el HEAD 4f13d8f. El orden de los commits está registrado en D1.
- `.venv\Scripts\python scripts/04_hashes_protocolo.py --verificar` → «OK: todos los hashes coinciden», al inicio y al final del día.
- Entorno de redes: `.venv311` con Python 3.11.9 y las versiones del artículo (`requirements-redes.txt`, D2). El código de la DualLSTM sale de «eee9a71 (futura v2.2-forecasting)» (D3).
- Ejecución paralela determinista, comprobada igual a la de un solo proceso (D4). La reejecución del auditor lo confirma (`reportes/auditoria/reproduccion_D.csv`, fila «unidad publicada»: diferencia 0).

**Estimación del paso 0 actualizada con las sensibilidades 6 y 7 (previsión, no resultado):**
- unas 3 400 unidades de red en D y 1 200 en M;
- unas 26 h secuenciales;
- unas 9 h de pared con procesos en paralelo, en un rango de 5 a 20 h.

La estimación partía de medir una semilla de la DualLSTM D con N = máx (59 s) y una de M (6,6 s); esas mediciones no se guardaron en CSV.

### 1.2 Tiempos reales

Fuente: `resultados/tiempos_dia3.csv`, generado por `resultados/tiempos_dia3.py` a partir de los registros de `resultados/brazo_*/logs/`.

| Brazo | Bloque | Escenario | h | Unidades | Fallos | Segundos de pared |
|---|---|---|---|---|---|---|
| D | redes | principal | 1 | 400 | 0 | 6295 |
| D | redes | principal | 7 | 400 | 0 | 4502 |
| D | redes | sens1_estricto | 1 | 400 | 0 | 6834 |
| D | redes | sens2_bolsa | 1 | 400 | 0 | 4132 |
| D | redes | sens6_desde2006 (solo N = máx) | 1 | 100 | 0 | 1189 |
| D | redes | sens7_anomalia_publicada | 1 | 400 | 0 | 3028 |
| D | estadísticos y Chronos | 5 escenarios | 1 y 7 | 100 cada uno | — | 92 a 122 por escenario (536 en total) |
| M | redes | principal | 1 | 400 | 0 | 3280.8 |
| M | redes | sens1_estricto | 1 | 400 | 0 | 4648.2 |
| M | redes | sens7_anomalia_publicada | 1 | 400 | 0 | 3137.6 |
| M | estadísticos y Chronos | 3 escenarios | 1 | — | — | 38.5 a 39.3 por escenario (116.7 en total) |

- **Redes de D:** 25 980 s (7.2 h) de pared en 2 100 unidades. Es el camino crítico, porque los dos brazos corrieron a la vez en la misma máquina.
- **Redes de M:** 11 066.6 s (3.1 h) en 1 200 unidades.
- **Unidades de D:** se ejecutaron 2 100 frente a las 3 400 estimadas. 300 de la diferencia se deben a que la sensibilidad 6 reutiliza de forma exacta las redes de la principal con N = 120, 360 y 1 000 (D7).
- **Sin recortes ni fallos:** no se recortó ninguna sensibilidad y ninguna unidad falló.
- **Estimación frente a lo real:** el tiempo real del brazo D (25 980 s de redes más 536 s de estadísticos y Chronos, unas 7.4 h) queda dentro del rango estimado. Los modelos estadísticos y Chronos fueron mucho más rápidos de lo previsto.

---

## 2. Curvas de aprendizaje

Pool de prueba 2021–2025, h = 1. El MAE está en toneladas, con IC 95 % por bootstrap pareado estratificado por ventana (B = 10 000, semilla 42). Las redes se promedian entre las 10 semillas por fecha.

### 2.1 Brazo D (n = 1 818 días evaluables)

MAE [IC 95 %] por N. Fuente: `resultados/brazo_D/principal/metricas.csv` (filas `anio = pool`).

| Modelo | N = 120 | N = 360 | N = 1000 | N = máx |
|---|---|---|---|---|
| Seasonal Naive | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] | 116.5 [109.8; 123.4] |
| ETS | 300.3 [275.1; 326.4] | 93.2 [87.7; 98.9] | 90.4 [85.3; 95.6] | 91.0 [85.9; 96.1] |
| Theta | 192.8 [182.0; 204.0] | 177.2 [164.7; 190.1] | 125.7 [119.6; 131.9] | 159.5 [152.8; 166.2] |
| SARIMA | 97.5 [91.7; 103.6] | 95.5 [89.7; 101.5] | 96.6 [90.9; 102.4] | 89.3 [83.8; 95.0] |
| Chronos-Bolt-tiny | 272.3 [264.3; 280.4] | 262.9 [254.4; 271.4] | 256.8 [247.9; 265.6] | 255.5 [246.7; 264.3] |
| LSTM univariada | 235.6 [227.8; 243.5] | 152.6 [146.0; 159.5] | 135.3 [129.4; 141.5] | 125.8 [119.9; 131.8] |
| DualLSTM | 206.9 [200.8; 213.2] | 156.8 [151.1; 162.5] | 137.4 [132.0; 142.9] | 121.6 [116.4; 127.1] |

Δ = MAE(DualLSTM) − MAE(ref*). Fuente: `resultados/brazo_D/principal/delta.csv`.

| N | ref* | Δ (t) | IC 95 % | ¿Significativa? | HLN | p DM-HLN |
|---|---|---|---|---|---|---|
| 120 | SARIMA | 109.4 | [103.7; 115.1] | sí | 36.56 | < 1e-15 |
| 360 | ETS | 63.6 | [58.1; 69.0] | sí | 22.66 | < 1e-15 |
| 1000 | ETS | 47.1 | [42.5; 51.7] | sí | 19.97 | < 1e-15 |
| máx | SARIMA | 32.3 | [27.8; 36.7] | sí | 13.79 | < 1e-15 |

El CSV da p_DM = 0 en las cuatro filas por subdesbordamiento numérico. Con esos valores de HLN y n = 1 818, p es menor que 1e-15.

- **Δ:** baja de forma estrictamente monótona con N (`monotona_decreciente = True` en `decision.csv`), pero sigue siendo positiva y significativa con la historia máxima.
- **Mejor modelo en N = máx:** SARIMA, con 89.3 t. La DualLSTM tiene 121.6 t, un 36 % más.
- **ETS con N = 120:** la especificación elegida (ETS(A,A,M)) extrapola mal en 2023 y da pronósticos negativos (§5, I2). Es un resultado legítimo del procedimiento preregistrado y no afecta a Δ(120), porque ref*(120) es SARIMA.

### 2.2 Brazo M (n = 59 meses evaluables)

MAE [IC 95 %] por N. Fuente: `resultados/brazo_M/principal/metricas.csv` (filas `anio = pool`).

| Modelo | N = 60 | N = 120 | N = 240 | N = máx |
|---|---|---|---|---|
| Seasonal Naive | 1523.0 [1196.4; 1863.0] | 1523.0 [1196.4; 1863.0] | 1523.0 [1196.4; 1863.0] | 1523.0 [1196.4; 1863.0] |
| ETS | 967.0 [797.1; 1147.0] | 976.5 [792.7; 1172.0] | 1017.8 [824.3; 1229.4] | 972.5 [794.0; 1158.5] |
| Theta | 956.2 [782.7; 1140.1] | 988.9 [811.3; 1175.8] | 1012.0 [836.7; 1201.6] | 1050.9 [875.2; 1234.8] |
| SARIMA | 2130.9 [1834.4; 2439.3] | 1334.5 [1116.8; 1558.0] | 1336.3 [1117.0; 1560.4] | 1324.2 [1079.4; 1581.1] |
| Chronos-Bolt-tiny | 1101.5 [877.4; 1342.9] | 1038.7 [841.5; 1252.8] | 1030.2 [841.2; 1232.8] | 1029.5 [838.7; 1236.5] |
| LSTM univariada | 1396.3 [1138.4; 1679.0] | 1218.8 [1003.9; 1448.1] | 1266.1 [1074.1; 1473.4] | 1288.2 [1086.0; 1505.7] |
| DualLSTM | 1842.2 [1545.6; 2159.0] | 1436.7 [1191.8; 1697.9] | 1354.0 [1133.4; 1595.1] | 1333.1 [1128.9; 1553.9] |

Δ = MAE(DualLSTM) − MAE(ref*). Fuente: `resultados/brazo_M/principal/delta.csv`.

| N | ref* | Δ (t) | IC 95 % | ¿Significativa? | HLN | p DM-HLN |
|---|---|---|---|---|---|---|
| 60 | Theta | 886.0 | [575.5; 1196.5] | sí | 5.07 | 4.4e-06 |
| 120 | ETS | 460.1 | [220.6; 696.6] | sí | 3.54 | 8.0e-04 |
| 240 | Theta | 342.1 | [126.9; 553.4] | sí | 2.94 | 4.7e-03 |
| máx | ETS | 360.6 | [159.3; 561.7] | sí | 3.26 | 1.8e-03 |

- **Δ:** baja mucho entre N = 60 y N = 240, pero no de forma estrictamente monótona: Δ(máx) = 360.6 > Δ(240) = 342.1. Con la historia máxima sigue siendo positiva y significativa.
- **Mejor modelo en N = máx:** ETS, con 972.5 t. La DualLSTM tiene 1 333.1 t.

### 2.3 «Consistently top-ranked» (complemento; no entra en la regla)

Fuente: `resultados/brazo_*/principal/top_ranked.csv`.

- **D:** SARIMA es «consistently top-ranked» en los cuatro N y ETS en N = 360 y 1000. La DualLSTM no es primera en ningún umbral y ningún N.
- **M:** con 59 meses, los IC se solapan mucho y hay muchos empates. Seasonal Naive, ETS, Theta y SARIMA son «consistently top-ranked» con N ≥ 120. La DualLSTM queda primera o empatada en 2 de 4 umbrales con N = 120 y N = máx, y no llega a «consistently top-ranked» en ningún N.

---

## 3. Regla de decisión (§1), aplicación literal

Las condiciones de la regla, en el orden en que las evalúa `src/evaluacion.py`:
1. **«el modelo»:** Δ(máx) > 0 y significativa.
2. **Subcaso «no se reproduce la desventaja»:** Δ no es significativamente positiva en ningún N. Se clasifica como «no concluyente».
3. **«los datos»:** Δ estrictamente decreciente y Δ(máx) no significativa o < 0.
4. **«no concluyente»:** cualquier otro caso.

El orden se registra en D8; no cambia ninguna etiqueta.

| Brazo | Δ(máx) (t) | IC 95 % | ¿Δ(máx) > 0 y significativa? | ¿Monótona? | Etiqueta | Fuente |
|---|---|---|---|---|---|---|
| D (h = 1) | 32.3 | [27.8; 36.7] | sí | sí | **el modelo** | `resultados/brazo_D/principal/decision.csv` |
| M (h = 1) | 360.6 | [159.3; 561.7] | sí | no | **el modelo** | `resultados/brazo_M/principal/decision.csv` |

En los dos brazos se cumple la primera condición: con la historia máxima de cada ventana, la DualLSTM sigue siendo peor que la mejor referencia y la diferencia es significativa.

**Conclusión conjunta (§1):** los dos brazos dan la misma etiqueta, así que la conclusión es **«el modelo explica la desventaja»**.

El auditor recalculó las 13 etiquetas con código propio y coinciden todas (`reportes/auditoria/recalculo_decision.csv`).

---

## 4. Sensibilidades (§9; no cambian la etiqueta)

| Sensibilidad | Brazo | Δ por N (ref*), en t | Δ(máx) IC 95 % | Etiqueta | Fuente |
|---|---|---|---|---|---|
| 1. «...» como faltante | D | 120: 86.9 (SARIMA); 360: 53.8 (ETS); 1000: 38.1 (ETS); máx: 28.7 (ETS) | [24.0; 33.5] | el modelo | `resultados/brazo_D/sens1_estricto/` |
| 1. «...» como faltante | M (n = 17, D5) | 60: 198.8 (Theta); 120: 309.6 (Theta); 240: 345.3 (ETS); máx: 205.4 (ETS) | [−293.2; 700.3] | no concluyente – no se reproduce la desventaja | `resultados/brazo_M/sens1_estricto/` |
| 2. Bolsa | D | 120: 112.4; 360: 68.2; 1000: 50.7; máx: 35.1 (todas SARIMA) | [30.2; 39.9] | el modelo | `resultados/brazo_D/sens2_bolsa/` |
| 3. h = 7 | D | 120: 176.0 (SARIMA); 360: 94.9 (ETS); 1000: 58.2 (ETS); máx: 28.1 (ETS) | [24.5; 31.6] | el modelo | `resultados/brazo_D/sens3_h7/` |
| 4. Pool 2023–2025 | D | 120: 111.5 (SARIMA); 360: 60.1 (ETS); 1000: 45.8 (ETS); máx: 18.7 (SARIMA) | [13.4; 24.0] | el modelo | `resultados/brazo_D/sens4_pool2023_2025/` |
| 4. Pool 2023–2025 | M | 60: 897.6 (Theta); 120: 459.3 (ETS); 240: 373.2 (Chronos); máx: 372.4 (ETS) | [169.7; 591.5] | el modelo | `resultados/brazo_M/sens4_pool2023_2025/` |
| 5. Bootstrap por bloques | D | iguales a la principal | [27.8; 36.7] | el modelo | `resultados/brazo_D/sens5_bloques/` |
| 5. Bootstrap por bloques | M | iguales a la principal | [180.1; 556.4] | el modelo | `resultados/brazo_M/sens5_bloques/` |
| 6. Historia desde 2006 | D | 120–1000 iguales a la principal; máx: 31.4 (SARIMA) | [27.1; 35.6] | el modelo | `resultados/brazo_D/sens6_desde2006/` |
| 7. Anomalía con el valor publicado | D | 120: 109.6 (SARIMA); 360: 63.7 (ETS); 1000: 47.3 (ETS); máx: 32.7 (SARIMA) | [28.1; 37.2] | el modelo | `resultados/brazo_D/sens7_anomalia_publicada/` |
| 7. Anomalía con el valor publicado | M | 60: 879.3 (ETS); 120: 473.9 (ETS); 240: 343.0 (Theta); máx: 359.3 (ETS) | [159.1; 551.8] | el modelo | `resultados/brazo_M/sens7_anomalia_publicada/` |

Lectura:
- **Todas menos una dan «el modelo».** La excepción es la sensibilidad 1 de M: con solo 17 meses evaluables, los IC son muy anchos y Δ no es significativa en ningún N. Es falta de potencia; no apunta a «los datos», porque las Δ puntuales siguen siendo positivas y no bajan de forma monótona.
- **D, Δ(máx):** va de 18.7 t (pool 2023–2025) a 35.1 t (bolsa), y es siempre significativa.
- **Sensibilidad 5:** las tablas «consistently top-ranked» de las dos carpetas `sens5_bloques` se calcularon con bootstrap iid. Son iguales a las de la principal (D9).

---

## 5. Hallazgos del auditor y su resolución

Fuente: `reportes/auditoria/AUDITORIA_DIA3.md`. El auditor (Opus, esfuerzo alto) revisó el código sin ver antes los informes de los brazos:
- reprodujo de forma independiente D (Y = 2024, N = 360, h = 1 y 7, más Y = 2023, N = 120) y M (Y = 2023, N = 120);
- recalculó Δ y las 13 etiquetas;
- comparó 2 652 celdas de los informes con sus CSV (`reportes/auditoria/verificacion_cifras_resumen.csv`).

**Reproducción independiente:**
- **D** (`reportes/auditoria/reproduccion_D.csv`):
  - Seasonal Naive: diferencia 0.
  - ETS (procedimiento literal, día a día): diferencia máxima 4.9e-05 t.
  - Chronos: 9.3e-05 t.
  - DualLSTM con semilla 0, reentrenada: 3.6e-05 t frente a `pronosticos.csv`, que está redondeado, y 0 frente a la unidad publicada.
- **M** (`reportes/auditoria/reproduccion_M.csv`): diferencia máxima 5e-06 t o menos en Seasonal Naive, ETS, SARIMA, Chronos y la DualLSTM con semilla 0.
- **Fugas:** el auditor no encontró ninguna.

**Bloqueantes: ninguno.** La conclusión no queda en suspenso.

| Hallazgo | Clase | Resolución |
|---|---|---|
| I1. Orden de la regla de decisión no registrado | importante | Registrado en D8. Sin efecto: ningún escenario cumple a la vez el subcaso y «los datos». |
| I2. Pronósticos negativos no declarados (D principal: 4 638 de 365 200; M: 0) | importante | Registrado en D10 y añadido a `reportes/brazo_D.md` §7. No se trunca nada a posteriori. |
| I3. Cifras sin respaldo en CSV (1,8e-12 t en M; 60–70 s y «cientos de toneladas» en D; tiempos tomados de logs y JSON) | importante | Se quitaron o se marcaron como «no guardada en CSV» las tres cifras. Se corrigieron las cabeceras de los dos informes para citar logs y JSON. Los tiempos están ahora en `resultados/tiempos_dia3.csv`. |
| m1. Cinco redondeos dobles | menor | Corregidos en `reportes/brazo_D.md` (121.63 ×3 y 107.49) y en `reportes/brazo_M.md` (1 572.4 y 1 826.4). |
| m2. p = 0 por subdesbordamiento; formato de p no homogéneo | menor | En este informe se escribe «< 1e-15». Los informes de los brazos no se cambiaron. |
| m3. Dos frases inexactas en D7 | menor | Corregidas: el límite de 2 048 es el del modelo y §5 prevé usar la cola; la reutilización de la sensibilidad 6 es exacta. Lo mismo en `reportes/brazo_D.md` §8. |
| m4. Top-ranked de la sensibilidad 5 con bootstrap iid | menor | Registrado en D9. No se recalcula a posteriori. |
| m5. Selección de ETS entre especificaciones no convergidas en D | menor | Añadido a D7. Sin efecto. |
| m6. Early stopping con objetivos de validación imputados | menor | Registrado en D10. |
| m7. Falta la fila de la sensibilidad 3 en la tabla «ETS inestable» | menor | Añadida a `reportes/brazo_D.md` (1195.7 t, `resultados/brazo_D/sens3_h7/metricas.csv`). |
| m8. Frase inexacta sobre los umbrales en la sensibilidad 4 | menor | Corregida en `reportes/brazo_D.md`. |
| m9. Restos de formato de Python | menor | Limpiados en `reportes/brazo_D.md`. |
| m10. D4 sin artefacto | menor | D4 cita ahora `reportes/auditoria/reproduccion_D.csv`. |
| m11. RMSE de las redes como media del RMSE de cada semilla | menor | Registrado en D10. |

**Nota:** las correcciones de I3, m1, m3 y m7–m9 se hicieron a mano sobre los `.md` de los brazos. Si se vuelven a ejecutar `resultados/brazo_M/scripts/05_informe.py` o `resultados/brazo_D/scripts/generar_informe.py`, se pierden.

---

## 6. Desviaciones

Todas están en `docs/DESVIACIONES.md`. Ninguna cambia una etiqueta.

| Id. | Resumen | Momento |
|---|---|---|
| D1 | Orden de los commits: los datos y scripts de los días 1–2 se versionaron después de la congelación; el protocolo no cambió. | antes de entrenar |
| D2 | `.venv311` (Python 3.11.9) para TensorFlow y Chronos; `ml_dtypes` 0.5.4. | antes de entrenar |
| D3 | Cita «eee9a71 (futura v2.2-forecasting)» en §1. | antes de entrenar |
| D4 | Ejecución paralela determinista, comprobada igual a la de un solo proceso. | antes de ver errores |
| D5 | Sensibilidad 1 de M con 17 meses evaluables, no 18: 2021-02 se excluye por la regla de §2.2. | antes de ver errores |
| D6 | Interpretaciones de ejecución de M: contexto de Chronos, selección de ETS, estado de ETS, Theta y SARIMA. | antes de ver errores |
| D7 | Interpretaciones de ejecución de D: `smooth` único, Chronos en lotes y truncado a 2 048, Theta, SARIMA, reutilización exacta en la sensibilidad 6. | antes de ver errores |
| D8 | Orden de aplicación de la regla de decisión. | código anterior; registrado tras la auditoría |
| D9 | Top-ranked de la sensibilidad 5 con bootstrap iid. | registrado tras la auditoría |
| D10 | Pronósticos negativos sin truncar, early stopping con validación imputada, RMSE de las redes. | código anterior; registrado tras la auditoría |

---

## 7. Archivos para `git add` y mensaje de commit sugerido

Al cierre del día:
- `.venv\Scripts\python scripts/04_hashes_protocolo.py --verificar` → «OK: todos los hashes coinciden»;
- `.venv311\Scripts\python -m pytest -q tests` → 20 passed.

**Datos crudos:** ningún CSV de `resultados/` ni de `reportes/auditoria/` contiene `y_true` ni valores de la serie del SISAP. `resultados/brazo_D/sens6_comprobacion.csv` solo guarda la diferencia máxima, 0.0. Se buscaron las columnas `y_true`, `y_eval` y `vol_total`.

**Propuesta de qué versionar:**
- **Sí:** el código, los informes y, por escenario, `pronosticos.csv` (consolida las unidades), `estados_redes.csv`, las tablas de análisis e `incidencias.csv`; además, scripts, logs y los CSV de comprobación.
- **Dejar fuera (decisión de YesaR):**
  - `resultados/brazo_D/redes_unidades/` (67 MB), `resultados/brazo_M/redes_unidades/` (6.4 MB) y `resultados/brazo_D/estadisticos/` (16 MB). Son las salidas por unidad, ya consolidadas en `pronosticos.csv` y `estados_redes.csv`. Lo único que no se consolida son las predicciones de validación de las redes.
  - Si se quieren dejar fuera de forma permanente, hay que añadir estas líneas a `.gitignore`:
    - `resultados/*/redes_unidades/`
    - `resultados/brazo_D/estadisticos/`

```
git add .gitignore requirements-redes.txt docs/DESVIACIONES.md src tests \
  reportes/INFORME_DIA3.md reportes/brazo_M.md reportes/brazo_D.md reportes/auditoria \
  resultados/tiempos_dia3.py resultados/tiempos_dia3.csv \
  resultados/brazo_M/scripts resultados/brazo_M/logs resultados/brazo_M/versiones.json resultados/brazo_M/semillas.json \
  resultados/brazo_M/principal resultados/brazo_M/sens1_estricto resultados/brazo_M/sens4_pool2023_2025 \
  resultados/brazo_M/sens5_bloques resultados/brazo_M/sens7_anomalia_publicada \
  resultados/brazo_D/scripts resultados/brazo_D/logs resultados/brazo_D/versiones.json resultados/brazo_D/semillas.json \
  resultados/brazo_D/equivalencias.csv resultados/brazo_D/tiempos_una_ventana.csv \
  resultados/brazo_D/sens6_comprobacion.csv resultados/brazo_D/sens6_vs_principal.csv \
  resultados/brazo_D/principal resultados/brazo_D/sens1_estricto resultados/brazo_D/sens2_bolsa \
  resultados/brazo_D/sens3_h7 resultados/brazo_D/sens4_pool2023_2025 resultados/brazo_D/sens5_bloques \
  resultados/brazo_D/sens6_desde2006 resultados/brazo_D/sens7_anomalia_publicada
```

**Mensaje de commit sugerido:**

```
Día 3 del experimento 2: curvas de aprendizaje de los brazos D y M

- src/ y tests/: código común (datos, ventanas rolling origin, redes, evaluación y análisis) y 20 pruebas
- resultados/brazo_D y brazo_M: pronósticos sin y_true, tablas de análisis, scripts y registros
  del escenario principal y de las sensibilidades 1-7 (§9)
- reportes/: informes de cada brazo, auditoría independiente e informe integrado del día 3
- docs/DESVIACIONES.md: desviaciones D1-D10
- requirements-redes.txt y .gitignore: entorno .venv311 para TensorFlow y Chronos

Etiqueta (§1): «el modelo» en D (h = 1) y en M (h = 1); conclusión conjunta «el modelo».
```
