# PROTOCOLO — Experimento 2: ¿falla el modelo o faltan datos?

**Estado: LISTO PARA CONGELAR, todavía no congelado.** Lo congela YesaR con un commit que contenga solo este archivo. El SHA-256 de este archivo y de las series procesadas está en `docs/hashes_protocolo.json`. A partir de ese commit:

- ningún script de evaluación puede modificar este archivo;
- cualquier cambio se registra en `docs/DESVIACIONES.md`, con la fecha, el motivo y si ocurrió antes o después de ver errores de prueba.

Redactado el 2026-11-14 y ajustado el 2026-11-16 (supuesto principal sobre «...» y objetivos de los dos brazos, §2.1; días con «...», §2.3; datos no publicados, §2). Hasta la redacción no se había entrenado ningún modelo ni calculado ningún error de pronóstico. Reemplaza a `docs/PROTOCOLO_BORRADOR.md`; los cambios están listados en `reportes/INFORME_DIA2.md`, §8.

---

## 1. Pregunta, hipótesis y regla de decisión

**Pregunta.** La DualLSTM-Bahdanau del artículo (*Forecasting*, MDPI; repositorio `duallstm-lime-forecasting-peru`, release v2.2-forecasting) perdió frente a los modelos estadísticos con unos 120 meses de historia. ¿Fue por la arquitectura («modelo») o por la longitud de la historia («datos»)?

**Diseño.** Hay dos brazos y en cada uno se construye una curva de aprendizaje en la que solo cambia la longitud N de la historia de entrenamiento. Los días o meses de prueba son los mismos para todos los N.

**Cantidad comparada.** En cada brazo, horizonte principal y tamaño N:

> Δ(N) = MAE_DualLSTM(N) − MAE_ref*(N)

donde `ref*(N)` es el modelo de referencia con menor MAE agregado (pool de prueba 2021–2025) en ese mismo N, elegido entre Seasonal Naive, ETS, Theta, SARIMA y Chronos-Bolt-tiny.

- La significancia de Δ(N) se juzga con el **IC 95 % del bootstrap pareado** (§6): es significativa si el IC excluye el 0.
- Diebold-Mariano con corrección HLN se reporta como prueba secundaria.

**Regla de decisión** (fijada antes de ver resultados; se aplica por separado a cada brazo, con D a h = 1 y M a h = 1):

| Resultado | Condición |
|---|---|
| **«La cantidad de datos explica la desventaja»** | Δ baja de forma estrictamente monótona entre los tamaños de la curva (estimaciones puntuales: Δ(N₁) > Δ(N₂) > … > Δ(máx)) **y**, en N = máx, Δ deja de ser significativa o cambia de signo (Δ(máx) < 0) |
| **«El modelo explica la desventaja»** | En N = máx, Δ(máx) > 0 y es significativa |
| **«No concluyente»** | Cualquier otro patrón. Por ejemplo: Δ(máx) no significativa sin una bajada monótona, o Δ(máx) < 0 significativa sin que haya existido desventaja con N pequeño |
| *Subcaso, que se informa como «no se reproduce la desventaja»* | Si Δ(N) no es significativamente positiva en **ningún** N, la premisa no se cumple en esta serie. Se reporta así y se clasifica como «no concluyente» |

**Conclusión conjunta:**

- Si los dos brazos dan la misma etiqueta, esa es la conclusión.
- Si dan etiquetas distintas, la conclusión es «no concluyente» y se reporta brazo por brazo.
- D a h = 7, el tratamiento anterior de «...» (como faltante), la serie de bolsa sola y el resto de las sensibilidades (§9) **no cambian la etiqueta**: se reportan aparte, cada una con su propia etiqueta.

---

## 2. Datos

**Fuente:** SISAP v2.0, módulo de Mercados Mayoristas (MIDAGRI), endpoint `resumenes/filtrar`:

- mercado *Gran mercado mayorista de lima* (`15011501`);
- *Limon sutil bolsa* (`061101`) y *Limon sutil cajon* (`061102`).

Las descargas están en `reportes/manifiesto_crudos.csv` (URL, parámetros, fecha de consulta y SHA-256). Las series procesadas se generan con `scripts/03_series.py`:

- `data/processed/sutil_diario.csv`
- `data/processed/sutil_mensual.csv`

Sus hashes están en `docs/hashes_protocolo.json`. **El experimento usa exactamente esos archivos.** Si se vuelve a descargar y el hash cambia, cuenta como desviación.

`data/processed/*.csv` no se publica en el repositorio (no se encontró una licencia que permita redistribuir los datos del SISAP). Se regenera con `scripts/02_descarga_sisap.py` y `scripts/03_series.py` (ver el README) y se comprueba con `python scripts/04_hashes_protocolo.py --verificar`.

**Fecha de corte:** 2026-11-13. Los datos de 2026 no se usan para nada (igual que en el artículo, que excluyó 2026).

**Inicio de ambos brazos: 1997-10-01.** Septiembre de 1997 no tiene volumen de bolsa ni de cajón en ningún día (es el primer hueco largo de la serie), y antes de esa fecha hay solo 8 meses.

### 2.1 Supuesto principal sobre «...» y variables objetivo

**Supuesto principal: «...» en una presentación significa que ese día no ingresó limón de esa presentación (ingreso cero).**

- Si una presentación (bolsa o cajón) tiene dato y la otra tiene «...», la segunda vale **0** y `vol_total` es la observada.
- Si **ambas** tienen «...», o la fecha no aparece en la tabla del SISAP, el día es **faltante**.
- La regla es la misma en toda la historia. En los años en que dominaba el cajón, bolsa «...» = 0.

**Evidencia** (`reportes/INFORME_DIA2.md`, §3 y §4.1):

1. **El SISAP nunca registra un volumen 0.** En 30 años no hay ni un 0, ni en el total ni por procedencia; el mínimo es 0,02 t. Si un día no entra limón de una presentación, el SISAP no tiene un 0 con que registrarlo, y «...» es lo único que aparece en su lugar.
2. **Patrón por día de la semana.** En 2021–2025, el 42 % de los días con cajón en «...» son miércoles y el 21 % martes, los días de menor ingreso; entre los días con las dos presentaciones observadas, los miércoles son el 4 %. La mediana de bolsa de los días con cajón en «...» es 61 t, frente a 488 t (Mann–Whitney, p = 2,6 × 10⁻⁶⁸).
3. **Medianas parecidas dentro de cada día de la semana.** La mediana de bolsa de los días con cajón en «...» es similar a la de los días con cajón observado del mismo día de la semana (p. ej., lunes 886 frente a 798 t; miércoles 4 frente a 5,5 t). Los días con «...» no son días anómalos: son días normales sin ingreso de esa presentación, concentrados en los días de poco ingreso.
4. **El cajón es hoy pequeño e intermitente** (9–18 % del volumen en 2021–2025), y la suma mensual del propio SISAP trata «...» como cero.

**Límite.** El supuesto **no se puede comprobar físicamente** con estos datos: no hay forma de distinguir «no ingresó limón» de «no se registró». Los PDF de gob.pe muestran «:» y no calculan promedios cuando hay «...» (informe, §3.1), lo que es compatible con las dos lecturas. Por eso el tratamiento anterior («...» = faltante) se reporta como sensibilidad 1 (§9).

**Columnas** (`sutil_diario.csv`):

- `vol_bolsa`, `vol_cajon`: valores **originales** del SISAP («...» = vacío).
- `estado_bolsa`, `estado_cajon`: `observado`, `cero_supuesto` (la otra presentación tiene dato) o `faltante` (las dos en «...»).
- `vol_bolsa_regla`, `vol_cajon_regla`: valores con la regla (0 donde `cero_supuesto`).
- `vol_total`: objetivo principal de D. `vol_total_estricto`: tratamiento anterior (sensibilidad 1).

| Brazo | Objetivo | Definición |
|---|---|---|
| **D (diario)** | `vol_total` (t/día) en `sutil_diario.csv` | `vol_bolsa_regla + vol_cajon_regla`. Faltante solo si las dos presentaciones están en «...» (`falta_vol_total = 1`) |
| **M (mensual)** | `vol_total` (t/mes) en `sutil_mensual.csv` | **Suma de `vol_total` diario del mes** con la regla anterior. Los días faltantes no suman. El mes es **faltante** (`flag_faltante = 1`) si **más del 20 %** de sus días son faltantes (`dias_faltantes`), o si contiene la anomalía documentada (§2.2) |

- **Relación con la suma del SISAP:** con esta regla, el mensual es en la práctica la suma publicada por el SISAP (columna `vol_total_sisap`). En los 357 meses comparables la diferencia máxima es de 0,13 t (redondeo). Se calcula desde el diario para que la regla sea explícita y la misma en los dos brazos.
- **Días faltantes en meses no faltantes:** suman 0, igual que en el SISAP. En 2021–2025 hay 7 días así, en 6 meses (2023-03, 2023-07, 2023-08 con 2 días, 2023-09, 2024-12 y 2025-10). Esos meses quedan algo subestimados; se acepta y se declara.
- `vol_total_escalado`, `dias_observados_estricto` y `flag_faltante_estricto` quedan como columnas de la **sensibilidad 1** (promedio diario de `vol_total_estricto` en los días observados × días del mes; faltante con menos del 80 % de días observados). **No son el objetivo principal.**
- **Mes incompleto** (`mes_incompleto`, definición del día 2): queda solo como descripción.
- **Secundaria y exógena:** `precio_prom`, el precio promedio diario de la bolsa en S/ por kg. Para el brazo M se usa el precio mensual de la bolsa del SISAP.

### 2.2 Faltantes

**En entrenamiento y en las entradas de los modelos** (incluidas las entradas usadas para pronosticar en validación y prueba):

- **Brazo D:** cada valor faltante de `vol_total` (las dos presentaciones en «...», o la anomalía) o de `precio_prom` se reemplaza por el **valor del mismo día de la semana anterior** (t − 7). Se aplica en orden cronológico sobre la serie ya imputada, de modo que una racha larga repite la última semana disponible. Usa **solo información pasada**.

  La elección se justifica por la estacionalidad semanal. Entre 2016 y 2026, la mediana de `vol_total` (días con las dos presentaciones observadas) es de 869 t los lunes y 36 t los miércoles. Arrastrar el último valor observado o interpolar mezclaría días de la semana de niveles muy distintos. t − 7 es además el pronóstico del Seasonal Naive, así que la imputación no da ventaja a ningún modelo por fuera de lo que ya hace la referencia más simple.

- **Brazo M:** cada mes faltante (`flag_faltante = 1`) se reemplaza por la **suma mensual de la serie diaria de `vol_total` imputada** con la regla del brazo D. Usa solo información pasada y es coherente entre brazos.

**Anomalía documentada** (columna `anomalia_documentada = 1` en `sutil_diario.csv`): el cajón del 2021-02-01. El SISAP da 1 235 t (Chiclayo 1 100 + Sullana 135), pero el PDF oficial de gob.pe de ese día da 245 t (= 110 + 135), lo que apunta a un probable cero de más en el SISAP. Ver `config/fuentes.yaml` y el informe del día 2, §3.3.

- El valor no es «...», así que el supuesto de ingreso cero no se le aplica: en el análisis principal es un valor **desconocido** y `vol_total` de ese día es **faltante**. Se imputa en las entradas con la regla anterior y no se evalúa.
- En el brazo M, el mes 2021-02 es **faltante** (`mes_con_anomalia = 1`): se imputa en las entradas y no se evalúa.
- La decisión se tomó antes de calcular cualquier error. La sensibilidad 7 (§9) usa el valor publicado.
- No se corrige ningún otro valor: el cribado de atípicos del día 2 es solo descriptivo.

**En la evaluación:** los días o meses de prueba cuyo objetivo es faltante **no se evalúan**. Se excluyen para todos los modelos por igual, de modo que el conjunto evaluado es idéntico entre modelos y entre tamaños N. A h = 7 se evalúa el día objetivo t + 7 solo si no es faltante.

Días y meses evaluables por año de prueba (`reportes/dia2_evaluables_por_anio.csv`):

| Año | 2021 | 2022 | 2023 | 2024 | 2025 | Total |
|---|---|---|---|---|---|---|
| **D, días (principal)** | 364 | 365 | 360 | 365 | 364 | **1 818** |
| D, días con cajón = 0 supuesto | 65 | 95 | 76 | 94 | 132 | 462 |
| D, días con bolsa = 0 supuesto | 1 | 2 | 0 | 1 | 0 | 4 |
| D, días (sensibilidad 1, «...» = faltante) | 298 | 268 | 284 | 270 | 232 | 1 352 |
| D, días (sensibilidad 2, bolsa sola) | 364 | 363 | 360 | 364 | 364 | 1 815 |
| **M, meses (principal)** | 11 | 12 | 12 | 12 | 12 | **59** |
| M, meses (sensibilidad 1) | 7 | 2 | 5 | 4 | 0 | 18 |

- D excluye en 2021–2025 el 2021-02-01 (anomalía) y 7 días con las dos presentaciones en «...»: 2023-03-12, 2023-07-15, 2023-08-03 y 2023-08-04 (fuera de la tabla), 2023-09-20, 2024-12-25 (Navidad) y 2025-10-15.
- M excluye 2021-02 (anomalía).
- En la sensibilidad 1 del brazo M, el año 2025 no tiene meses evaluables: esa ventana se reporta como «sin meses evaluables» y no entra en su pool.

### 2.3 Días con «...» frente a días observados (brazo D)

Esta comparación se hizo el 2026-11-16 con el tratamiento entonces vigente («...» = faltante), antes de entrenar ningún modelo. Es la evidencia del supuesto de §2.1 (puntos 2 y 3) y queda como descripción del conjunto evaluado en la sensibilidad 1.

**Comparación.** En 2021–2025, días **evaluados** con el tratamiento anterior (`vol_total_estricto` observado y sin la anomalía) frente a los **excluidos** (el resto): distribución por día de la semana y por mes, distribución de `vol_bolsa` y Mann–Whitney bilateral para `vol_bolsa` (`reportes/dia2_excluidos_vs_evaluados.csv`).

| | Evaluados | Excluidos |
|---|---|---|
| Días | 1 352 | 474 (463 con bolsa) |
| Miércoles / martes | 4,4 % / 12,0 % | 42,4 % / 20,9 % |
| Mediana de `vol_bolsa` | 488 t | 61 t |

Mann–Whitney: U = 142 986,5, p = 2,6 × 10⁻⁶⁸.

**Regla anterior y su sustitución.** La regla escrita el 2026-11-16 decía que, con p < 0,05, D con `vol_bolsa` pasaría a ser la sensibilidad principal. Esa regla estaba pensada para el objetivo estricto, que dejaba fuera 474 días de prueba. Con el supuesto de §2.1, el objetivo principal deja fuera solo 8 días de prueba (7 con las dos presentaciones en «...», en los que no hay bolsa con la que comparar, y la anomalía), así que la regla **ya no se aplica**. La comparación se conserva como evidencia del supuesto. D con `vol_bolsa` sigue como sensibilidad (§9, sensibilidad 2).
---

## 3. Ventanas (rolling origin, como en el artículo)

- **Años de prueba:** Y ∈ {2021, 2022, 2023, 2024, 2025}, iguales en los dos brazos. Es el pool principal del artículo.
- **Pool de sensibilidad:** 2023–2025.
- Para cada Y:
  - **entrenamiento:** hasta el 31 de diciembre de Y − 2;
  - **validación:** el año Y − 1 completo;
  - **prueba:** el año Y completo.
- **Parámetros:** se estiman **una vez por ventana** con el segmento de entrenamiento (las redes usan la validación solo para el early stopping) y quedan fijos durante la validación y la prueba.
- **Pronóstico a h = 1:** para el día o mes t se usan los datos observados hasta t − 1 (imputados según §2.2 si faltan). Después se incorpora el valor real de t como estado o entrada, **sin reestimar** (equivalente a `append(refit=False)`).
- No se reajusta con la validación, igual que en el artículo.

---

## 4. Curva de aprendizaje

La historia de entrenamiento de tamaño N está formada por las **últimas N observaciones del segmento de entrenamiento**, que termina el 31 de diciembre de Y − 2 (origen de entrenamiento). La validación (Y − 1) y la prueba (Y) son las mismas para todos los N.

| Brazo | Tamaños N | N = máx por ventana |
|---|---|---|
| D | 120, 360, 1 000 días y máx | desde 1997-10-01: 8 127 días (Y = 2021) … 9 588 días (Y = 2025) |
| M | 60, 120, 240 meses y máx | desde 1997-10: 267 meses (Y = 2021) … 315 meses (Y = 2025) |

- N cuenta observaciones del calendario, incluidas las imputadas.
- Las secuencias de entrenamiento de las redes se construyen solo dentro de la historia de tamaño N, así que las primeras observaciones se consumen como rezagos.
- La MASE usa una escala **fija por ventana**: la de la historia máxima (§6). Así es comparable entre tamaños N.

---

## 5. Modelos

No se busca ningún hiperparámetro con los datos de validación o prueba, igual que en el artículo. Las únicas selecciones automáticas son las del artículo: ETS por AICc dentro del entrenamiento y el early stopping de las redes en la validación. Todos los modelos reciben la misma serie imputada.

| Modelo | Brazo D (h = 1; h = 7) | Brazo M (h = 1) |
|---|---|---|
| **Seasonal Naive** | ŷ_t = y_{t−7}; a h = 7, ŷ_{t+7} = y_t | ŷ_t = y_{t−12} |
| **ETS** | `statsmodels` ETSModel, `seasonal_periods = 7`, especificación con **menor AICc en entrenamiento** entre las 12 del artículo: error {A, M} × tendencia {N, A, Ad} × estacionalidad {A, M}; empate por orden de la rejilla. Estado actualizado con `smooth(params)` | igual, con `seasonal_periods = 12` |
| **Theta** | `statsmodels` ThetaModel (θ = 2), `period = 7`; b0, α y factores estacionales fijados en entrenamiento y estado SES actualizado con cada dato (como `r2lib.ThetaFijo` del artículo) | igual, con `period = 12` |
| **SARIMA** | SARIMA(1,0,1)(0,1,1)₇ sobre log(y) con regresores de Fourier anuales (K = 3, periodo 365,25). Orden fijado a priori, sin selección. Con N < 730 se omiten los términos de Fourier (no hay un ciclo anual completo). Opciones como en el artículo: `enforce_stationarity = False`, `enforce_invertibility = False`, `initialization = "approximate_diffuse"`, `maxiter = 500` | **SARIMA congelado del artículo para el limón sutil:** (1,1,1)(1,1,0)₁₂, `trend = "c"`, mismas opciones. Se declara que en el artículo ese orden se adoptó después de ver la prueba de 2025; aquí se aplica sin cambios a una serie nueva |
| **Chronos-Bolt-tiny** | `amazon/chronos-bolt-tiny`, snapshot `a0e552de83495b5c28c14c71c374f3e33280b340` (el del artículo), cero disparos (*zero-shot*), mediana (cuantil 0,5), CPU, float32. Contexto: las últimas N observaciones hasta t − 1; `prediction_length` = 1 o 7. **No comprobado:** la longitud máxima de contexto del modelo. Si N la supera, el modelo usa solo la cola y se reporta | igual, con `prediction_length = 1` |
| **LSTM univariada** | Una rama: entrada de ventana de 6 pasos con 1 variable (y), LSTM(16), Dense(16, ReLU), Dropout(0,2), Dense(8, ReLU), Dense(1). Entrenamiento idéntico a la DualLSTM. 10 semillas | igual |
| **DualLSTM-Bahdanau** | Misma arquitectura e hiperparámetros que el artículo (detalle abajo). 10 semillas | igual |

**DualLSTM-Bahdanau.** Se copia el código de `v2_reentrenamiento/src/models/dual_lstm_attention.py` (solo lectura); **no se copian pesos**. La única modificación es el validador de dimensiones, porque cambia el número de variables de la rama B.

- **Arquitectura:**
  - lookback 6;
  - dos ramas LSTM(16) con `return_sequences`;
  - atención de Bahdanau aditiva (16 unidades, sin sesgo) en cada rama;
  - concatenación, Dense(16, ReLU), Dropout(0,20), Dense(8, ReLU), Dense(1).
- **Entrenamiento** (`hiperparametros_congelados.json`):
  - Adam con lr 0,001, pérdida MSE, batch 8, hasta 300 épocas, `shuffle = False`;
  - EarlyStopping sobre `val_loss` con paciencia 15 y restauración de los mejores pesos;
  - ReduceLROnPlateau con paciencia 8, factor 0,5 y lr mínimo 1e-6;
  - semillas 0–9; semilla global 42.
- **Escalado:** StandardScaler ajustado solo con el entrenamiento de la ventana, como en el artículo. El pronóstico se devuelve a toneladas antes de evaluar.
- **Rama A** (4 variables, como en el artículo): y_t, y_{t−1}, y_{t−3}, y_{t−6}. En D los rezagos son días; en M, meses.
- **Rama B:**
  - D, 8 variables: calendario (sen y cos del día de la semana, sen y cos del día del año, `t_index`) y `precio_prom` rezagado 1, 3 y 6 días;
  - M, 6 variables: calendario (sen y cos del mes, `t_index`) y precio mensual rezagado 1, 3 y 6 meses.
  - No se usan clima, INDECI ni noticias.
- **Número de parámetros:** cambia respecto al artículo por la dimensión de la rama B. Se reporta y se verifica con la fórmula `expected_parameter_count` del artículo.
- **h = 7 en D:** estrategia **directa**. Se entrena un modelo con el objetivo y_{t+6} usando entradas hasta t − 1. Lo mismo para la LSTM univariada.
- **Declaración:** con lookback 6 y rezagos {1, 3, 6}, la red ve hasta 11 días atrás, lo que incluye t − 7. No se cambia el lookback porque es un hiperparámetro congelado del artículo.

**Agregación de semillas** (DA8 del artículo): el error absoluto de cada día o mes se promedia entre las 10 semillas. El bootstrap y Diebold-Mariano usan ese promedio. La dispersión entre semillas se reporta aparte.

---

## 6. Métricas y pruebas

- **MAE en toneladas (principal)**, RMSE y MASE.
- **Escala de la MASE:** MAE dentro de la muestra del Seasonal Naive (m = 7 en D, m = 12 en M) sobre la historia **máxima** de entrenamiento de cada ventana, igual para todos los N.
- **Bootstrap pareado:**
  - iid sobre días o meses, estratificado por ventana (año de prueba);
  - **B = 10 000, semilla 42**, IC 95 % por percentiles;
  - implementación como en `lib/evaluacion.py` del artículo.
  - Sensibilidad: bootstrap por bloques (7 días en D, 3 meses en M).
- **Diebold-Mariano** con corrección de Harvey-Leybourne-Newbold, pérdida absoluta, h = 1 (h = 7 en la sensibilidad de D).
- **«Consistently top-ranked»** (misma definición que el artículo):
  - un modelo cumple si es **primero o está empatado con el primero** (el IC 95 % de su diferencia de MAE con el mejor contiene 0) en **al menos 3 de los 4 umbrales** P50, P60, P75 y P90;
  - los umbrales salen de percentiles lineales de la variación relativa absoluta, calculados con la historia hasta el origen de cada ventana, con comparación estricta (>);
  - en M la variación es mes a mes (como en el artículo);
  - en D es semana a semana, |y_t − y_{t−7}| / y_{t−7}. Es una adaptación declarada: la variación día a día está dominada por el día de la semana.
  - Se reporta en cada N.

---

## 7. Qué se reporta

1. La curva de aprendizaje con el MAE de cada modelo en función de N, por brazo y horizonte.
2. Δ(N) con IC 95 %, DM-HLN y la etiqueta de la regla de decisión.
3. El ranking y «consistently top-ranked» en cada N.
4. Los resultados por año de prueba y en el pool 2023–2025.
5. La dispersión entre semillas.
6. Las incidencias de convergencia (ETS, SARIMA) y las veces que Chronos truncó el contexto.

---

## 8. Puente con el artículo (descriptivo, no cambia la etiqueta)

Correlación de `vol_total` mensual del GMML con la producción nacional de limón sutil de SISAGRI entre 2016 y 2025. La serie de SISAGRI viene de `v2_reentrenamiento/data/interim/limon_sutil_nacional_mensual.csv` del repositorio del artículo, con SHA-256 `9e8e06c6…70e31`.

- Pearson y Spearman, en niveles y en variaciones mensuales, con rezagos de 0 a 2 meses.
- Se calcula sin meses incompletos (y con todos, como control), con la suma mensual del SISAP (`vol_total_sisap`), que coincide con el objetivo principal del brazo M salvo por redondeo (§2.1).
- Resultado ya calculado el día 2: `reportes/dia2_correlacion_sisagri.csv`.

---

## 9. Sensibilidades preregistradas (no cambian la etiqueta)

1. **Tratamiento anterior de «...»: faltante en los dos brazos.**
   - D con `vol_total_estricto` (faltante si falta cualquiera de las dos presentaciones; imputado con t − 7 en las entradas): 1 352 días evaluables en 2021–2025.
   - M con `vol_total_escalado` (promedio diario de `vol_total_estricto` en los días observados × días del mes; faltante con menos del 80 % de días observados, `flag_faltante_estricto`): 18 meses evaluables, ninguno en 2025.
   - Se reporta aparte, con su propia etiqueta por brazo.
2. **D con `vol_bolsa` como objetivo** (valores originales; faltantes imputados con t − 7, como en §2.2). Días evaluables: 360–364 por año (1 815 en 2021–2025).
3. **D a h = 7.**
4. **Pool de prueba 2023–2025.**
5. **Bootstrap por bloques.**
6. **D con historia máxima desde 2006-01-01**, después del cambio a volúmenes enteros de 2005-12. Ver el informe del día 2, §5.
7. **Anomalía de 2021-02-01 con el valor publicado** por el SISAP (1 235 t de cajón): evaluada en D, y en M con 2021-02 como mes no faltante.

## 10. Fuera del alcance

- Limón dulce: no existe en el SISAP del GMML.
- Limón tahití: está en el catálogo pero no tiene datos.
- Otros mercados.
- Información de noticias (NLP), clima e INDECI.
- Datos de 2026.
- Búsqueda de hiperparámetros.
