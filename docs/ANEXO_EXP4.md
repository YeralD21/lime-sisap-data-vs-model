# Anexo — Experimento 4: ¿qué variables usa la DualLSTM para pronosticar? (SHAP + permutación)

**Estado:** escrito antes de calcular ningún valor SHAP, ninguna importancia por permutación y ningún peso de atención de los modelos del estudio. YesaR lo congela con un commit. Los SHA-256 de este archivo y de los scripts nuevos están en `docs/hashes_anexo_exp4.json`.

**Naturaleza: DESCRIPTIVO / POSTERIOR.** Explica modelos ya entrenados en los experimentos 1, 2 y 3. No entrena modelos nuevos (solo reconstruye los mismos, §2), no cambia ninguna etiqueta ni ninguna regla de decisión y no modifica el protocolo (`docs/PROTOCOLO.md`, 9ede96c), el anexo del experimento 3 (`docs/ANEXO_EXP3.md`, 512b7cc), `data/` ni los scripts de los experimentos 2 y 3.

Cualquier desviación de este anexo se registra en `docs/DESVIACIONES.md` (entradas «EXP4-…») con fecha, motivo y efecto.

> **Advertencia que se repite en el informe.** SHAP, la importancia por permutación y la atención describen **el comportamiento del modelo**: qué entradas mueven su pronóstico. No muestran causas en el mundo real ni dicen si usar una variable mejora el error. Una variable puede tener un SHAP alto y aun así empeorar el error de prueba, como ocurrió con el clima en el brazo D del experimento 3 (etiqueta «perjudica»).

---

## 1. Pregunta

En los modelos ya entrenados, ¿qué grupo de variables pesa más en el pronóstico de la DualLSTM? ¿Coincide ese peso con lo que mostraron las ablaciones del experimento 3?

**Grupos de variables:**

| Grupo | Rama | Columnas (este repositorio) | Columnas (artículo, exp. 1) |
|---|---|---|---|
| historia | A | y(j), y(j − 1), y(j − 3), y(j − 6) | `produccion_t_sutil` y sus rezagos 1, 3, 6 |
| calendario | B | D: sen/cos día de la semana, sen/cos día del año, `t_index`; M: sen/cos mes, `t_index` | `mes_sin`, `mes_cos`, `t_index` |
| precio | B | precio rezagado 1, 3, 6 | — (el artículo no usa precio) |
| clima | B | exp. 3, anexo §2.2–2.3 | NASA (5 variables × rezagos 1, 3, 6) |
| INDECI | B | exp. 3, anexo §2.2–2.3 | INDECI (5 variables × rezagos 1, 3, 6) |
| noticias | B | exp. 3, anexo §2.2–2.3 | sentimiento y número de noticias, rezagos 1, 3, 6 (solo GE) |

`t_index` (índice de tiempo) se cuenta en «calendario», porque en el experimento 2 la rama B base se describió como «calendario + precio». La asignación de cada columna a su grupo está en `src/explicabilidad.py` y se comprueba en `tests/test_exp4.py`.

## 2. Modelos a explicar

Año de prueba Y = 2025, semillas 0–9 en todos los casos.

| Caso | Origen | Brazo | Rama B | Unidades |
|---|---|---|---|---|
| `exp3_V4_D` | Exp. 3, variante V4 (todas las exógenas) | D (diario, h = 1) | 35 columnas | 10 |
| `exp3_V4_M` | Exp. 3, variante V4 | M (mensual, h = 1) | 45 columnas | 10 |
| `exp2_Nmax_D` | Exp. 2, DualLSTM, escenario principal, N = máx, h = 1 | D | 8 (calendario + precio) | 10 |
| `exp2_Nmax_M` | Exp. 2, DualLSTM, escenario principal, N = máx, h = 1 | M | 6 (calendario + precio) | 10 |
| `exp1_GC3` | Exp. 1 (artículo), GC3, limón sutil (opcional) | mensual | 33 (calendario, clima, INDECI) | 10 |
| `exp1_GE` | Exp. 1 (artículo), GE, limón sutil (opcional) | mensual | 39 (GC3 + noticias) | 10 |

En el exp. 2 se usa h = 1 en el brazo D (el protocolo también tiene h = 7) para que el horizonte sea el mismo en todos los casos.

### 2.1 Pesos guardados

Ningún caso tiene pesos guardados:

- `src/redes.entrenar_y_predecir` (exp. 2 y 3) borra `checkpoint_best.keras` después de predecir;
- `resultados/exp3/unidades/` y `D:\respaldo_dia3_unidades.zip` solo contienen `estado.json` y `predicciones.csv` (0 archivos `.keras`);
- `04_modelos/` del artículo tampoco guarda archivos `.keras`.

Por eso se reentrenan **solo** las 60 unidades de la tabla, con el código congelado y las mismas semillas (§2.2).

### 2.2 Reconstrucción sin cambiar el código congelado

- **Exp. 2 y 3:** `scripts/10_exp4.py --modelos` ejecuta cada unidad en un proceso nuevo, con las mismas variables de entorno y determinismo que el día 3 (`src.redes.VARIABLES_ENTORNO`, `enable_op_determinism`, 2 hilos intra y 1 inter). Llama a `src.redes.entrenar_y_predecir` **sin modificarla**. Lo único que cambia en ese proceso es el borrado de `checkpoint_best.keras`: se intercepta para que el archivo se conserve como `modelo.keras`. Después se guardan las secuencias de entrada (`entradas.npz`), calculadas con la misma `src.redes.preparar`. El SHA-256 de `src/redes.py` sigue siendo el de `docs/hashes_anexo_exp3.json`.
- **Exp. 1 (artículo, eee9a71, solo lectura):** `scripts/10_exp4_articulo.py` se ejecuta con el intérprete del propio artículo (`venv`). Llama a `lib.entorno.preparar_entorno`, a `lib.integridad.Integridad().verificar()` y a `lib.modelo_lstm.entrenar_y_predecir(2025, "sutil", modelo, semilla, destino)`, igual que `07_lstm.py una`. Esa función ya deja `checkpoint_best.keras` en el destino, así que no hace falta tocar nada. El destino está fuera del repositorio del artículo (`resultados/exp4/modelos/`, no versionado).

### 2.3 Comprobación de la reconstrucción

Para cada unidad se comparan los pronósticos de prueba de 2025 del modelo reconstruido con los publicados (`resultados/exp4/reproduccion.csv`). El criterio es una diferencia máxima ≤ 1e-3 t:

1. **Frente a la unidad original, a precisión completa:**
   - exp. 3: `resultados/exp3/unidades/exp3_*/V4/Y2025/s*/predicciones.csv`;
   - exp. 2: `resultados/brazo_*/redes_unidades/*_principal_1997_faltante/dual/Y2025/Nmax/h1/s*/predicciones.csv`;
   - exp. 1: `04_modelos/{gc3_lstm_sin_nlp,ge_lstm_con_nlp}/sutil/2025/seed_*/predicciones.csv` del artículo.
2. **Frente a `pronosticos.csv`** (exp. 2 y 3). Ese archivo se guardó redondeado:
   - exp. 3: `float_format="%.6g"`, así que por encima de 1 000 t el archivo solo tiene una o dos cifras decimales;
   - exp. 2: `%.8g` en D y `%.10g` en M.

   Por eso aquí una fila también se acepta si el valor reconstruido, escrito con el mismo formato, reproduce exactamente el valor guardado.

Si alguna unidad no coincide, `--modelos` se detiene y se informa; no se explica ningún modelo que no reproduzca sus pronósticos publicados.

**Resultado de la Fase A:** ver §8.

## 3. Métodos (fijados antes de calcular)

Todos se aplican a cada unidad (caso × semilla) y después se agregan entre las 10 semillas. Los valores SHAP y la permutación se expresan en toneladas: la salida de la red está en escala estandarizada y se multiplica por la desviación típica del escalador del objetivo de esa unidad.

### 3.1 SHAP

- **Explicador:** `shap.GradientExplainer` (expected gradients), versión 0.51.0, sobre el modelo Keras de dos entradas `[rama_a, rama_b]` cargado de `modelo.keras`. `tests/test_exp4.py` comprueba con una red **sin entrenar** que funciona con la capa `BahdanauAttention`.
  - Si fallara con algún modelo, se usaría `shap.KernelExplainer` sobre una función envoltorio con las entradas aplanadas (fondo resumido con k-means de 20 centros, 1 000 evaluaciones por muestra), y se registraría como desviación.
- **Fondo (background):** 200 secuencias tomadas **solo de las secuencias de entrenamiento** de la unidad (objetivos hasta el 31-12 de Y − 2), sin reemplazo, con `numpy.random.default_rng(42)`.
  - Si la unidad tiene menos de 200 secuencias de entrenamiento, se usan todas. Pasa en `exp3_V4_M` (84) y en `exp1_*` (84).
  - El fondo es el mismo para las 10 semillas de un caso.
- **Muestras explicadas:** todas las secuencias de prueba de 2025: 365 días en D y 12 meses en M y en el exp. 1.
- **Parámetros:** `nsamples = 200`, `rseed = 42`.
- **Medida por grupo (principal):** para cada muestra, la contribución del grupo es la suma con signo de los SHAP de sus columnas en las 6 posiciones (por aditividad). La importancia del grupo es la media sobre las muestras de |contribución|.
  - Como medida secundaria se guarda la media de la suma de |SHAP| de sus columnas; esta crece con el número de columnas del grupo.
- **Medida por posición en la secuencia (rezago):** la misma medida principal, pero por posición: t − 6, …, t − 1.
- **Medida por variable:** media sobre las muestras de |suma de los SHAP de la variable en las 6 posiciones|.
- **Entre semillas:** media y desviación típica de las 10 importancias por semilla.
- **Ranking por SHAP:** grupos ordenados por la media entre semillas (1 = el que más pesa).

### 3.2 Importancia por permutación

- **Qué se permuta:** las columnas de un grupo, en sus 6 posiciones, en las secuencias de prueba de 2025. Cada secuencia i toma las columnas del grupo de la secuencia π(i). El resto de columnas no cambia. Para «historia» se permuta la rama A completa.
- **Permutación en bloques:** π baraja bloques contiguos de muestras de 7 días en D y de 3 meses en M, y conserva el orden dentro de cada bloque (`src/explicabilidad.permutacion_bloques`). El último bloque puede ser más corto (365 = 52 × 7 + 1).
- **Repeticiones:** 30. El generador `numpy.random.default_rng(42)` se reinicia para cada grupo y cada semilla, así que todos los grupos y semillas usan las mismas 30 permutaciones (números aleatorios comunes).
  - En M solo hay 4 bloques (4! = 24 órdenes posibles): algunas permutaciones se repiten y alguna puede ser la identidad. Se cuentan y se informan (`repeticiones_identidad`).
- **Medida:** aumento de MAE = MAE(entradas permutadas) − MAE(entradas originales), en t, sobre las fechas de prueba evaluables del modelo:
  - exp. 2 y 3: `y_eval` no faltante, como en sus evaluaciones;
  - exp. 1: los 12 meses con el objetivo del artículo.

  Se calcula con los pronósticos de cada semilla (sin promediar entre semillas).
- **Agregación:** media de los 300 valores (10 semillas × 30 repeticiones). El intervalo 95 % se toma de los percentiles 2,5 y 97,5 de esos 300 valores, es decir, entre repeticiones y semillas. Es un intervalo de variabilidad, no un IC de la media. Se informa también la desviación típica entre las medias por semilla.
- **Ranking por permutación:** grupos ordenados por el aumento medio de MAE (1 = el que más lo aumenta).

### 3.3 Comparación con las ablaciones del experimento 3

Para `exp3_V4_D` y `exp3_V4_M` se ponen lado a lado tres cosas:

1. el ranking por SHAP;
2. el ranking por permutación;
3. el resultado de la ablación del exp. 3 (`resultados/exp3/{D,M}/delta.csv`): ΔMAE de V1 (clima), V2 (INDECI) y V3 (noticias) frente a V0, con su IC 95 %, su p de Holm y su etiqueta; y además V4 frente a V0.

**Cómo se ordena:**

- La ablación solo cubre clima, INDECI y noticias. Por eso los tres rankings también se calculan **solo entre esos tres grupos**.
- En la ablación, el rango 1 es el ΔMAE más bajo (añadir el grupo es lo más favorable o lo menos perjudicial).
- Historia, calendario y precio se marcan como «no ablacionados».

**Cómo se interpreta:**

- Las coincidencias y discrepancias se describen, no se fuerzan.
- Las dos cosas miden conceptos distintos:
  - SHAP y la permutación miden **cuánto usa** el modelo V4 ya entrenado un grupo;
  - la ablación mide **si añadirlo y reentrenar** mejora el error de prueba.

  Un grupo muy usado y con ablación «perjudica» se lee como «el modelo usa mucho el grupo, pero usarlo no mejora el error» (por ejemplo, si se ajusta a ruido del entrenamiento).
- En los casos del exp. 2 y del exp. 1 no hay ablación en este repositorio; solo se informan SHAP y permutación.

### 3.4 Atención (descriptivo)

- Pesos α de las dos capas Bahdanau (rama A y rama B) por posición de la secuencia, con query = último estado oculto, en las muestras de prueba de 2025.
- Se obtienen de las salidas de `lstm_a` y `lstm_b` y de los pesos de la capa (`src/explicabilidad.pesos_atencion`). Se comprueba que Σ α·h reproduce la salida de la capa (diferencia máxima en `resultados/exp4/cordura.csv`).
- Se informa la media sobre las muestras y la media y la desviación entre semillas. La línea de referencia es 1/6 (atención uniforme).
- **Advertencia:** los pesos de atención no son una explicación causal ni una medida fiable de importancia de las entradas (Jain y Wallace, 2019, «Attention is not Explanation»). Además, la atención pondera estados ocultos de la LSTM, que ya mezclan todas las posiciones anteriores, no las entradas crudas.

### 3.5 Comprobaciones de cordura (fase B)

En cada unidad:

- **Completitud:** f(x) − E[f(fondo)] − Σ SHAP. Expected gradients cumple la completitud solo en esperanza: con `nsamples = 200` el error es aproximado. Se informa el error absoluto medio y el máximo, en t, junto con la media de |f(x) − E[f(fondo)]|.
- **Carga del modelo:** el modelo cargado de `modelo.keras` reproduce las predicciones de prueba guardadas por su propia corrida (diferencia máxima, en t).
- **Atención:** la comprobación de §3.4.

Ningún resultado de estas comprobaciones cambia las cifras. Si el error de completitud medio superara el 10 % de la media de |f(x) − E[f(fondo)]| en algún caso, se diría en el informe y en las desviaciones.

## 4. Figuras

En `reportes/figuras/{es,en}/exp4/`, en PNG a 300 ppp y PDF:

| Figura | Contenido |
|---|---|
| `x1_shap_por_grupo` | barras de \|SHAP\| medio por grupo, con barras de error entre semillas (±1 desviación típica); un panel por caso |
| `x2_shap_por_rezago` | \|SHAP\| por grupo y posición en la secuencia (t − 6 … t − 1) |
| `x3_permutacion` | aumento de MAE por grupo, con el intervalo 95 % de §3.2 |
| `x4_shap_vs_ablacion` | rankings de SHAP, permutación y ablación lado a lado (exp. 3 V4, D y M) |
| `x5_beeswarm_V4_D` | beeswarm de SHAP del exp. 3 V4 diario por variable individual (las 20 de mayor \|SHAP\| medio). Cada punto es un día de prueba; su SHAP es la suma de las 6 posiciones, promediada entre semillas. El color es el valor escalado de la variable en t − 1, entre sus percentiles 5 y 95. Se versiona solo el PNG: el PDF vectorial guarda el valor escalado de cada punto, que en la rama A es la serie del SISAP (criterio D13). |
| `x6_atencion` | pesos de atención medios por posición, ramas A y B |

## 5. Salidas

**`resultados/exp4/` (versionado; solo agregados, ninguna fila de la serie):**

- `reproduccion.csv`, `modelos.log`, `explicacion.log`;
- `shap_grupo.csv` y `shap_grupo_semillas.csv`, `shap_rezago.csv`, `shap_variable.csv`;
- `permutacion.csv` y `permutacion_repeticiones.csv`;
- `atencion.csv`, `cordura.csv`, `comparacion_ablacion.csv`.

**`resultados/exp4/modelos/` (no versionado):** modelos, entradas escaladas, SHAP por muestra y predicciones por unidad. Las entradas de la rama A son la serie escalada.

**Informe:** `reportes/INFORME_EXP4.md`, marcado **DESCRIPTIVO / POSTERIOR (sin auditoría independiente)**. Toda cifra del informe sale de un CSV de `resultados/exp4/` (o de `resultados/exp3/*/delta.csv` en el caso de las ablaciones).

## 6. Limitaciones que el informe debe declarar

- **Variables correlacionadas:** SHAP y la permutación reparten o desplazan la importancia entre variables correlacionadas. Por ejemplo, los rezagos del precio y de la serie, o las variables de clima entre sí y con el calendario. La permutación crea combinaciones de entradas poco realistas. Por eso se agrega por grupo, pero los grupos también están correlacionados entre sí (clima con calendario).
- **Fondo:** los SHAP dependen del fondo elegido (secuencias de entrenamiento). Un fondo distinto, como el de validación, daría otros valores.
- **Muestras pequeñas en M y en el exp. 1:** 12 meses explicados, 84 secuencias de fondo y 4 bloques de permutación.
- **Sin causalidad:** SHAP, la permutación y la atención no muestran causas ni efectos en el mundo real.
- **Posterior y sin auditoría independiente:** el análisis se hace después de conocer los resultados de los experimentos 1–3.

## 7. Tiempo de cómputo estimado

- **Reconstrucción (Fase A):** 60 unidades en 6 procesos. Los casos más lentos son las 10 de `exp2_Nmax_D` (unos 150 s cada una en el día 3); el total ronda 10 min de pared.
- **SHAP:** unos 0,02 s por muestra explicada con fondo de 200 (medido con una red sin entrenar de V4 D): unos 8 s por unidad diaria y menos de 1 s por unidad mensual, más el arranque de TensorFlow, unos 10 s por proceso.
- **Permutación:** 6 grupos × 30 repeticiones × 1 predicción por unidad, del orden de segundos.
- **Total de la fase B:** menos de 15 min de pared con 6 procesos.

## 8. Resultado de la Fase A (antes de congelar)

Ejecutado el 2026-11-18 con `scripts/10_exp4.py --modelos --n-proc 6` (`resultados/exp4/modelos.log`, `resultados/exp4/reproduccion.csv`):

- **Pesos guardados encontrados:** 0 en `resultados/`, 0 en `D:\respaldo_dia3_unidades.zip` y 0 en `04_modelos/` del artículo. Se reconstruyeron las 60 unidades en 348 s de pared.
- **Reproducción frente a la unidad original:** la diferencia máxima es 0 t en las 60 unidades, en los 6 casos. Se cumple el criterio (≤ 1e-3 t).
- **Reproducción frente a `pronosticos.csv`:** todas las filas pasan el criterio de §2.3.
  - En `exp3_V4_D` y `exp3_V4_M` las diferencias absolutas máximas (0,0049 t y 0,050 t) se deben solo al redondeo `%.6g` del archivo.
  - En el exp. 2, la diferencia máxima es ≤ 5e-5 t.
- **Exp. 1 (opcional):** se pudo reconstruir sin cambiar el código del artículo, así que **se incluye**. Después de la reconstrucción, `git status` del artículo sigue limpio.
- **Entorno:** se añadieron a `.venv311` `shap==0.51.0` y sus dependencias `numba==0.68.1`, `llvmlite==0.50.0` y `slicer==0.0.8` (`requirements-exp4.txt`). Ningún otro paquete cambió de versión: `pip freeze` sin esos cuatro es idéntico a `requirements-redes.txt`.
- **Pruebas:** `pytest` da 61 pruebas aprobadas (las 20 del exp. 2, las del exp. 3 y las nuevas de `tests/test_exp4.py`).
