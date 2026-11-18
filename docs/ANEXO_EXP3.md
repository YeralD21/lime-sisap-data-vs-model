# Anexo — Experimento 3: ¿ayudan las variables exógenas a la rama B con datos alineados? (ablación)

**Estado:** escrito antes de entrenar ningún modelo de este experimento. YesaR lo congela con un commit; los SHA-256 de este archivo y de las exógenas están en `docs/hashes_anexo_exp3.json`.

**Relación con el protocolo.** Es un experimento **posterior** al protocolo congelado del experimento 2 (`docs/PROTOCOLO.md`, 9ede96c), que no se modifica. Reutiliza su serie principal, su imputación, sus ventanas, su red y su evaluación. Lo que cambia se dice aquí. No altera ninguna etiqueta del experimento 2.

Cualquier desviación de este anexo se registra en `docs/DESVIACIONES.md` (entradas «EXP3-…») con fecha, motivo y efecto.

---

## 1. Pregunta

Con la misma DualLSTM y las mismas fechas (periodo común 2016–2025), ¿agregar clima, emergencias (INDECI) o noticias (sentimiento NLP) a la rama B reduce el error de prueba frente a la rama B base (calendario + precio rezagado)?

## 2. Datos

### 2.1 Objetivo

- Serie principal del protocolo (§2.1–2.2): `vol_total` del GMML con faltantes imputados con t − 7 y la anomalía 2021-02-01 como faltante. Se usa la misma imputación que el experimento 2, hecha sobre la serie completa.
- Después se **recorta a 2016-01-01 – 2025-12-31** (`src/exogenas.serie_exp3`).
- **Brazo D:** diario, h = 1.
- **Brazo M:** mensual, h = 1.
- Días y meses evaluables: los mismos del experimento 2 en 2021–2025: 1 818 días en D y 59 meses en M.

### 2.2 Exógenas

Se construyen con `scripts/07_exogenas_exp3.py` y `src/exogenas.py`. Las fuentes del repositorio del artículo (eee9a71) y de su copia privada se usan solo en lectura.

**Ponderación geográfica.** Pesos por provincia = participación en la producción de limón sutil de SISAGRI en **2016–2019** (`limon_sutil_provincia.csv` del artículo, `produccion_t`). Se calculan una sola vez y quedan fijos para todas las ventanas. Las provincias con producción 0 en 2016–2019 quedan fuera. Los pesos se guardan en `resultados/exp3/pesos_provincias_2016_2019.csv`.

Las coordenadas provinciales son las del archivo NASA mensual del artículo.

**Brazo D (diarias, `data/processed/exogenas_diarias.csv`; no se versiona):**

| Familia | Variables base por día | Construcción |
|---|---|---|
| Clima | `prec`, `tmax`, `tmin`, `rh` | NASA POWER **diario** (API `temporal/daily/point`, community AG; PRECTOTCORR, T2M_MAX, T2M_MIN, RH2M) en cada provincia con peso > 0; media ponderada por los pesos (renormalizados si falta un punto ese día). Caché en `data/raw/nasa_power_diario/`. |
| INDECI | `eventos`, `afectados` | `BD_2003-2025_EMERGENCIAS.csv` (SHA-256 `a1fe63f0…2687`, el de `data_sources.json` del artículo). Mismas reglas que `etl/02_indeci_a_panel.py` del artículo: 10 peligros climáticos; LIMA unificada; afectados = AFECTADOS + DAMNIFICADOS (= `total_afectados`). Por **fecha de la emergencia**, contando en las provincias con peso > 0, sin ponderar. Un día sin registros = 0. |
| Noticias | `sent`, `n_noticias` (+ `sin_noticias`) | Sentimiento por noticia (RoBERTuito vía pysentimiento, score = P(pos) − P(neg)), de `noticias_por_item.csv` de la copia privada del artículo. Solo se leen las columnas `fecha` y `score`; los textos no se leen ni se copian. Por día: media de los scores y número de noticias. Con 0 noticias, la media es faltante y `sin_noticias` = 1. La suma mensual de `n_noticias` reproduce la serie mensual del artículo (prueba `test_noticias_diarias_reproducen_mensual_del_articulo`). |

**Brazo M (mensuales, `data/processed/exogenas_mensuales.csv`; no se versiona):** son las exógenas mensuales del artículo (`VARS_CLIMA`, `VARS_INDECI` y `VARS_NLP` de su `lib/datos.py`), alineadas a 2016-01 – 2025-12:

| Familia | Variables | Construcción |
|---|---|---|
| Clima | T2M, T2M_MAX, WS2M, PRECTOTCORR, RH2M | NASA mensual por provincia del artículo, media ponderada con los pesos 2016–2019 |
| INDECI | num_emergencias, personas_afectadas, personas_damnificadas, total_afectados, hectareas_cultivo_perdidas, hectareas_cultivo_afectadas | panel mensual del artículo (`indeci_limpio_2016_2025.csv`), suma en las provincias con peso > 0 |
| Noticias | avg_sentiment, n_noticias | `sentimiento_mensual_2016_2025.csv` del artículo, tal cual (0 cuando no hay noticias) |

El artículo ponderaba con la superficie media del periodo de entrenamiento de cada ventana. Aquí se usa la producción de 2016–2019, fija, como pide el diseño de este experimento.

### 2.3 Rezagos: ninguna exógena del día (mes) pronosticado

La red usa un contexto de 6 filas (`lookback` = 6) que termina en t − 1. Las exógenas se anclan en el objetivo t: la última fila del contexto (t − 1) lleva:

- **D:** x(t − 1), x(t − 7) y la media de x(t − 30) … x(t − 1).
  - En sentimiento: el valor del día (0 si no hay noticias) más su indicador, en t − 1 y t − 7; y la media de 30 días ponderada por noticias (suma de scores / número de noticias), con indicador si no hubo ninguna. Son 6 columnas.
- **M:** x(t − 1), x(t − 3) y x(t − 6).

Es decir, la fila j lleva x(j), x(j − 6) y la media de x(j − 29 … j) en D, y x(j), x(j − 2) y x(j − 5) en M.

Ancho de la rama B (columnas):

| Variante | D | M |
|---|---|---|
| V0 base: calendario + precio rezagado (igual que el día 3) | 8 | 6 |
| V1 = V0 + clima | 20 | 21 |
| V2 = V0 + INDECI | 14 | 24 |
| V3 = V0 + noticias | 17 | 12 |
| V4 = V0 + clima + INDECI + noticias | 35 | 45 |

## 3. Diseño

- **Ventanas** (rolling origin, igual que el protocolo §3): años de prueba 2021–2025; validación Y − 1; entrenamiento desde 2016-01-01 hasta el 31-12 de Y − 2. La historia es la máxima disponible dentro de 2016–(Y − 2), igual para todas las variantes:
  - D: 1 461 a 2 922 días;
  - M: 48 a 96 meses.
- **Calentamiento común:**
  - D: la primera fila de contexto es la 30.ª de 2016 (2016-01-30), porque la media de 30 días la necesita.
  - M: la primera fila es la 7.ª (2016-07), por los rezagos de la rama A.
  - Se aplica a **todas** las variantes, incluida V0, para que todas tengan los mismos objetivos de entrenamiento (`margen_filas` en `src/redes.preparar`).
- **Escalado:** StandardScaler ajustado solo con las filas de la historia de entrenamiento, después del calentamiento; aplica a todas las columnas de A y B, como el día 3.
- **Red:** DualLSTM-Bahdanau del día 3 (`src/redes.py`; arquitectura de eee9a71) con los mismos hiperparámetros:
  - lookback 6, LSTM 16, atención 16, densas 16 y 8, dropout 0,20;
  - Adam 1e-3, MSE, lote 8, máximo 300 épocas, sin barajar;
  - EarlyStopping (val_loss, paciencia 15, restaura la mejor), ReduceLROnPlateau (0,5; paciencia 8; mínimo 1e-6) y ModelCheckpoint de la mejor;
  - 10 semillas (0–9).
  - Solo cambia el ancho de la rama B (el validador acepta otros anchos con `n_b_libre`).

  Parámetros por variante (`expected_parameter_count(4, B)`):

| Variante | D | M |
|---|---|---|
| V0 | 4 673 | 4 545 |
| V1 | 5 441 | 5 505 |
| V2 | 5 057 | 5 697 |
| V3 | 5 249 | 4 929 |
| V4 | 6 401 | 7 041 |

- **Unidades:** 2 brazos × 5 variantes × 5 años × 10 semillas = 500 corridas.
  - Determinismo y paralelismo como el día 3 (D4): un proceso por corrida, `enable_op_determinism`, 2 hilos intra y 1 inter.
  - Ejecución reanudable (`src/redes.correr_unidades`).
  - Cada corrida guarda su historial de pérdida por época (`estado.json`, clave `historial`).
- **Error por fecha:** el error absoluto de cada semilla se promedia entre las 10 semillas por fecha, antes del bootstrap y del DM (como DA8 del experimento 2). En las figuras de pronósticos se dibuja la media de las 10 semillas.

## 4. Referencia (solo para situar; no entra en la regla)

- El mejor modelo univariado del día 3 con N = máx en el mismo pool de prueba: SARIMA en D y ETS en M (`resultados/brazo_*/principal/delta.csv`).
- Se toman sus pronósticos ya guardados, sin reentrenar.
- Su historia (desde 1997) no es la de este experimento (desde 2016). Se informa como contexto, junto con la DualLSTM del día 3 con N = máx.

## 5. Evaluación y regla de decisión (fijada antes de entrenar)

Por brazo, se hacen 4 comparaciones con V0: V1, V2, V3 y V4. Las cifras son:

- **MAE pool 2021–2025** de cada variante sobre las fechas evaluables, con el error promediado entre semillas por fecha.
- **ΔMAE(Vk) = MAE(Vk) − MAE(V0)**, con IC 95 % por bootstrap pareado iid estratificado por año de prueba (B = 10 000, semilla 42; `src/evaluacion.remuestreos`).
- **Diebold–Mariano con corrección HLN** (pérdida absoluta, h = 1, bilateral, t(n − 1); `src/evaluacion.dm_hln`). La cola se calcula con `t.sf` para que no se desborde.
- **Corrección de Holm** sobre los 4 p-valores del brazo.

**Etiqueta de cada variante y brazo:**

| Etiqueta | Condición |
|---|---|
| **«ayuda»** | MAE(Vk) < MAE(V0), p de Holm < 0,05 **y** el IC 95 % de ΔMAE excluye 0 (límite superior < 0) |
| **«perjudica»** | MAE(Vk) > MAE(V0), p de Holm < 0,05 **y** el IC 95 % de ΔMAE excluye 0 (límite inferior > 0) |
| **«sin efecto»** | cualquier otro caso |

No hay conclusión conjunta obligatoria: se reporta cada variante en cada brazo. Se reporta además, sin que cambie la etiqueta:
- el MAE por año;
- la dispersión entre semillas;
- la posición de cada variante frente a la referencia del §4.

## 6. Evidencia sin fuga

1. **Pruebas automatizadas** (`tests/test_exogenas.py`, junto con las 20 del experimento 2):
   - cambiar las exógenas del día (mes) t y posteriores no altera la entrada de la red para t, mientras que cambiar las de t − 1 sí;
   - cambiar las exógenas posteriores al fin del entrenamiento no altera ni el escalador ni las secuencias y objetivos de entrenamiento;
   - todas las variantes tienen los mismos objetivos;
   - los pesos solo usan 2016–2019;
   - las noticias diarias reproducen la serie mensual del artículo;
   - el código del día 3 no cambia con los valores por defecto.
2. **Prueba de perturbación** (fase B, diagnóstico): una unidad por brazo (V4, Y = 2025, semilla 0). Se alteran al azar todas las exógenas, el objetivo y el precio posteriores al origen (fin de 2024) y se reentrena. El escalador, el entrenamiento, la época elegida y el pronóstico del primer paso de prueba deben ser idénticos.

## 7. Figuras previstas

Se generan en `reportes/figuras/{es,en}/exp3/`, en PNG a 300 ppp y PDF:

1. MAE por variante con IC 95 %, por brazo, con la referencia del §4 como línea.
2. ΔMAE frente a V0 con IC 95 %, con p de Holm y la etiqueta.
3. Curvas de pérdida de entrenamiento y validación por época, por variante (Y = 2025; semilla de MAE de prueba mediano entre las 10, la de rango 5 de 10 por orden de MAE), con la mejor época marcada.
4. Pronósticos de 2025 de V0 frente a V4 junto al observado. Por la serie del SISAP (D13 del experimento 2), se versiona solo el PNG.

## 8. Salidas

- **`resultados/exp3/`:**
  - pesos;
  - resumen de exógenas;
  - por brazo: `metricas.csv`, `delta.csv` (ΔMAE, IC, p, p de Holm, etiqueta), `semillas.csv`, `estados_redes.csv`, `perturbacion.csv`.
- **No se versionan:** `pronosticos.csv` (por prudencia, con el mismo criterio que D13) ni las unidades por corrida.
- **Informe:** `reportes/INFORME_EXP3.md`, marcado **PRELIMINAR (sin auditoría independiente)**.
