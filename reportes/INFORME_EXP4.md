# Informe — Experimento 4: ¿qué variables usa la DualLSTM para pronosticar? (SHAP + permutación)

> **DESCRIPTIVO / POSTERIOR (sin auditoría independiente).** El análisis explica modelos ya entrenados en los experimentos 1, 2 y 3. No cambia ninguna etiqueta ni regla de decisión. El anexo (`docs/ANEXO_EXP4.md`) se congeló en b052f2d antes de calcular.
>
> **Advertencia.** SHAP, la importancia por permutación y la atención describen **el comportamiento del modelo**: qué entradas mueven su pronóstico. No muestran causas en el mundo real ni dicen si usar una variable mejora el error. Una variable puede pesar mucho en el modelo y aun así empeorar el error de prueba, como ocurre aquí con el clima en el brazo D (§3).

Las cifras salen de `resultados/exp4/*.csv`. Los ΔMAE de la ablación salen de `resultados/exp3/{D,M}/delta.csv`, copiados en `resultados/exp4/comparacion_ablacion.csv`. Todas están en toneladas (t), Y = 2025, con 10 semillas por caso.

## 1. Qué se hizo

**Modelos.** Se explicaron 6 casos × 10 semillas:

| Caso | Modelo |
|---|---|
| `exp3_V4_D`, `exp3_V4_M` | exp. 3 con todas las exógenas |
| `exp2_Nmax_D`, `exp2_Nmax_M` | exp. 2, DualLSTM, N = máx, h = 1 |
| `exp1_GC3`, `exp1_GE` | artículo, limón sutil |

Ninguno tenía pesos guardados, así que se reconstruyeron con el código congelado. Las 60 unidades reproducen sus pronósticos publicados con diferencia 0 t (`reproduccion.csv`; anexo §8).

**Métodos** (anexo §3):

- **SHAP:** `GradientExplainer` (expected gradients). Fondo de hasta 200 secuencias de entrenamiento; en `exp3_V4_M` y el exp. 1 solo hay 84 y se usan todas.
- **Permutación por bloques:** de 7 días en D o 3 meses en M, con 30 repeticiones.
- **Atención:** pesos α de las capas Bahdanau.

**Medida principal de SHAP por grupo:** media sobre las muestras de prueba de |suma de los SHAP del grupo| (columnas y posiciones), con media y desviación típica (DE) entre semillas.

**Intervalo de la permutación:** percentiles 2,5–97,5 de los 300 valores (10 semillas × 30 repeticiones). Es un intervalo de variabilidad, no un IC de la media.

## 2. Tabla por grupo

En cada tabla, «Rango» ordena los grupos de mayor a menor importancia (1 = el que más pesa). La última columna es el ΔMAE de la ablación del exp. 3 (Vk − V0) con su IC 95 % y su etiqueta.

**Exp. 3 V4, brazo D** (MAE de las entradas originales, media entre semillas: 116,8 t)

| Grupo | Columnas | \|SHAP\| medio (t) ± DE semillas | Rango SHAP | Aumento de MAE por permutación (t) [intervalo 95 %] | Rango perm. | Ablación exp. 3 |
|---|---|---|---|---|---|---|
| historia | 4 | 255,2 ± 9,3 | 1 | 199,8 [13,3; 352,8] | 1 | — |
| calendario | 5 | 49,7 ± 18,9 | 2 | 5,5 [−0,8; 14,8] | 2 | — |
| precio | 3 | 10,2 ± 3,0 | 6 | 0,5 [−1,3; 2,3] | 5 | — |
| clima | 12 | 24,2 ± 5,8 | 3 | 0,9 [−7,1; 7,1] | 3 | V1: +7,3 [5,3; 9,2], perjudica |
| INDECI | 6 | 16,9 ± 3,7 | 5 | −0,7 [−11,7; 8,2] | 6 | V2: +5,8 [4,1; 7,4], perjudica |
| noticias | 9 | 18,9 ± 6,9 | 4 | 0,6 [−2,9; 4,3] | 4 | V3: −1,1 [−2,2; 0,1], sin efecto |

V4 frente a V0 (todas las exógenas): +6,5 [4,6; 8,5], perjudica.

**Exp. 3 V4, brazo M** (MAE original: 1 702,3 t)

| Grupo | Columnas | \|SHAP\| medio (t) ± DE semillas | Rango SHAP | Aumento de MAE por permutación (t) [intervalo 95 %] | Rango perm. | Ablación exp. 3 |
|---|---|---|---|---|---|---|
| historia | 4 | 80,3 ± 33,7 | 5 | −1,2 [−90,4; 40,6] | 3 | — |
| calendario | 3 | 82,1 ± 46,9 | 4 | −7,7 [−181,3; 55,9] | 5 | — |
| precio | 3 | 15,7 ± 9,0 | 6 | −2,5 [−29,8; 7,6] | 4 | — |
| clima | 15 | 111,8 ± 111,4 | 2 | 9,9 [−261,7; 206,5] | 1 | V1: −218,9 [−310,1; −128,5], ayuda |
| INDECI | 18 | 233,5 ± 349,9 | 1 | −10,1 [−387,2; 134,3] | 6 | V2: −10,2 [−123,9; 108,7], sin efecto |
| noticias | 6 | 105,4 ± 68,4 | 3 | 1,4 [−78,3; 62,0] | 2 | V3: +95,5 [38,7; 151,5], sin efecto |

V4 frente a V0: −66,6 [−139,1; 12,3], sin efecto.

**Exp. 2, N = máx, brazo D** (MAE original: 106,4 t)

| Grupo | Columnas | \|SHAP\| medio (t) ± DE | Rango SHAP | Aumento de MAE (t) [intervalo 95 %] | Rango perm. |
|---|---|---|---|---|---|
| historia | 4 | 198,6 ± 13,7 | 1 | 187,8 [12,9; 335,2] | 1 |
| calendario | 5 | 105,2 ± 19,1 | 2 | 23,2 [4,5; 50,0] | 2 |
| precio | 3 | 10,1 ± 2,9 | 3 | 1,2 [−1,1; 4,7] | 3 |

**Exp. 2, N = máx, brazo M** (MAE original: 755,9 t)

| Grupo | Columnas | \|SHAP\| medio (t) ± DE | Rango SHAP | Aumento de MAE (t) [intervalo 95 %] | Rango perm. |
|---|---|---|---|---|---|
| historia | 4 | 3 212,5 ± 278,8 | 1 | 3,2 [−48,3; 79,9] | 3 |
| calendario | 3 | 1 841,8 ± 299,5 | 2 | 657,1 [0,0; 1 618,1] | 1 |
| precio | 3 | 219,6 ± 112,4 | 3 | 3,6 [−76,1; 112,2] | 2 |

**Exp. 1 (artículo), GC3** (MAE original: 7 746,0 t)

| Grupo | Columnas | \|SHAP\| medio (t) ± DE | Rango SHAP | Aumento de MAE (t) [intervalo 95 %] | Rango perm. |
|---|---|---|---|---|---|
| historia | 4 | 1 070,5 ± 566,9 | 4 | 436,6 [−353,5; 1 592,7] | 2 |
| calendario | 3 | 1 771,1 ± 1 065,8 | 2 | 412,7 [−197,1; 1 324,2] | 3 |
| clima | 15 | 2 172,9 ± 1 293,3 | 1 | 1 409,8 [0,0; 4 812,5] | 1 |
| INDECI | 15 | 1 672,9 ± 962,2 | 3 | 350,1 [−1 038,2; 2 155,3] | 4 |

**Exp. 1 (artículo), GE** (MAE original: 9 426,5 t)

| Grupo | Columnas | \|SHAP\| medio (t) ± DE | Rango SHAP | Aumento de MAE (t) [intervalo 95 %] | Rango perm. |
|---|---|---|---|---|---|
| historia | 4 | 838,6 ± 571,5 | 5 | 396,4 [−176,2; 1 524,3] | 2 |
| calendario | 3 | 1 416,0 ± 1 448,8 | 2 | 84,2 [−308,8; 989,6] | 4 |
| clima | 15 | 1 312,4 ± 1 192,8 | 3 | 750,6 [−208,6; 4 351,7] | 1 |
| INDECI | 15 | 1 074,6 ± 842,0 | 4 | 75,5 [−2 420,8; 1 447,0] | 5 |
| noticias | 6 | 1 494,4 ± 1 232,7 | 1 | 111,4 [−529,9; 924,9] | 3 |

**Notas sobre la permutación en M y en el exp. 1:**

- Con 12 meses solo hay 4 bloques: 2 de las 30 permutaciones son la identidad (20 de 300 valores por grupo; `repeticiones_identidad`).
- Por eso algunos extremos del intervalo valen exactamente 0,0.

## 3. Qué grupo pesa más

**Brazo D, exp. 3 V4 y exp. 2:**

- La **historia de la serie (rama A)** domina con claridad: es la primera en SHAP y en permutación en los dos modelos.
- La sigue el **calendario**.
- El **precio** pesa poco.
- En V4, las tres familias exógenas tienen |SHAP| de 17–24 t, frente a 255 t de la historia. Al permutarlas, el MAE cambia menos de 1 t en promedio, con intervalos que incluyen 0.
- En `x5_beeswarm_V4_D`, las variables individuales con más peso son `y_l1` y `y_l0` (historia), seguidas de `t_index` (`shap_variable.csv`).
  - `t_index` tiene SHAP negativo en todos los días de prueba. En 2025 su valor escalado queda por encima de todo el rango de entrenamiento, así que el modelo está extrapolando una tendencia.

**Brazo M, exp. 3 V4:**

- No hay un grupo claramente dominante.
- Por SHAP, INDECI es el primero (233,5 t), pero con una DE entre semillas mayor que la media (349,9 t): unas semillas lo usan mucho y otras poco.
- Por permutación, ningún grupo cambia el MAE de forma clara: todos los intervalos incluyen 0.

**Brazo M, exp. 2:**

- Por SHAP domina la historia, pero por permutación domina el calendario (657,1 t frente a 3,2 t).
- Las dos medidas responden a preguntas distintas:
  - SHAP compara cada entrada con el fondo, que son secuencias de entrenamiento desde 1997, con otros niveles de la serie;
  - la permutación solo baraja meses dentro de 2025.

  Barajar la historia entre meses de 2025 cambia poco el error. Barajar el calendario, que es la posición estacional, lo cambia mucho.

**Exp. 1 (artículo):**

- En GC3, el clima es el primero por SHAP y por permutación.
- En GE, las noticias son las primeras por SHAP y el clima por permutación.
- Las DE entre semillas son del orden de la media o mayores, y los intervalos de permutación son muy anchos. Además, la completitud de SHAP en estos dos casos es aproximada (§6). Ningún orden del exp. 1 debe leerse como firme.

**Por posición en la secuencia (`x2_shap_por_rezago`):**

- En todos los casos y grupos, el |SHAP| es mayor en la posición más antigua (t − 6) y baja hacia t − 1.
- En la rama A, la fila t − 6 contiene y(t − 7) en D e y(t − 12) en M, los rezagos estacionales; pero el patrón aparece también en el calendario y en las exógenas.
- El diagnóstico posterior EXP4-3 (`diagnostico_posicion.csv`) confirma que viene de los modelos y no de SHAP: sustituir la posición t − 6 cambia el pronóstico más que sustituir t − 1 en los 6 casos.
  - Por ejemplo, en `exp3_V4_D` el cambio medio va de 342,2 t (t − 6) a 30,6 t (t − 1).
- No se interpreta más allá de esa descripción.

**Atención (`x6_atencion`, `atencion.csv`):**

- En el exp. 3 los pesos son casi uniformes: entre 0,151 y 0,180 en todas las posiciones y ramas, frente a 1/6 ≈ 0,167. En el exp. 1, entre 0,145 y 0,187.
- En el exp. 2 se separan más: rama A en D, de 0,116 a 0,236; en M, de 0,112 a 0,206, creciendo hacia t − 1.
- La atención **no** coincide con el patrón de SHAP por posición. Esto es coherente con la advertencia de Jain y Wallace (2019): los pesos de atención no son una explicación de qué entradas importan, y además ponderan estados ocultos de la LSTM, no entradas.

## 4. ¿Coinciden SHAP y la permutación con las ablaciones? (exp. 3, `x4_shap_vs_ablacion`)

**Rangos solo entre clima, INDECI y noticias** (1 = mayor uso; en la ablación, 1 = ΔMAE más favorable):

| Brazo | SHAP | Permutación | Ablación (ΔMAE frente a V0) |
|---|---|---|---|
| D | clima 1, noticias 2, INDECI 3 | clima 1, noticias 2, INDECI 3 | noticias 1 (−1,1, sin efecto), INDECI 2 (+5,8, perjudica), clima 3 (+7,3, perjudica) |
| M | INDECI 1, clima 2, noticias 3 | clima 1, noticias 2, INDECI 3 | clima 1 (−218,9, ayuda), INDECI 2 (−10,2, sin efecto), noticias 3 (+95,5, sin efecto) |

**Brazo D: no coinciden.** SHAP y la permutación están de acuerdo entre sí: el clima es la exógena que más usa V4. Pero la ablación muestra que añadir el clima es lo que más empeora el error (+7,3 t, «perjudica»).

Interpretación prudente: **el modelo usa el clima más que las otras exógenas, pero usarlo no mejora el error; lo empeora.** Es compatible con que el modelo aprenda relaciones con el clima que no se sostienen fuera del entrenamiento, aunque este análisis no lo demuestra. Hay que añadir dos cosas:

- el uso de las exógenas en V4 es pequeño frente a la historia (§3);
- al permutar el clima, el error de V4 casi no cambia (0,9 t, intervalo que incluye 0).

Que el clima empeore el error al añadirlo y reentrenar (ablación) no es lo mismo que el modelo V4 ya entrenado dependa mucho de él (permutación).

**Brazo M: coinciden en parte.** La permutación pone el clima primero, igual que la ablación (la única exógena con etiqueta «ayuda»). SHAP pone INDECI primero, pero con una dispersión entre semillas tan grande que su orden no es estable. Además, en M ninguna permutación cambia el MAE de forma clara y la completitud de SHAP es aproximada (14,7 %; §6).

Interpretación prudente: **en M la coincidencia entre el uso del clima y su beneficio es débil y no permite afirmar que el modelo mejora porque usa el clima.**

En los casos del exp. 2 y del exp. 1 no hay ablación en este repositorio, así que no se compara.

## 5. Figuras

En `reportes/figuras/{es,en}/exp4/`, generadas con `scripts/11_exp4_figuras.py` (EXP4-2):

| Figura | Contenido |
|---|---|
| `x1_shap_por_grupo` | \|SHAP\| medio por grupo ± DE entre semillas |
| `x2_shap_por_rezago` | \|SHAP\| por grupo y posición |
| `x3_permutacion` | aumento de MAE con intervalo 95 % |
| `x4_shap_vs_ablacion` | rankings de SHAP, permutación y ablación lado a lado |
| `x5_beeswarm_V4_D` | beeswarm de variables individuales |
| `x6_atencion` | pesos de atención por posición |

En x1, las barras de ±1 DE pasan por debajo de 0 en algunos casos mensuales: eso solo indica una dispersión grande entre semillas, porque |SHAP| no puede ser negativo.

## 6. Comprobaciones de cordura (`cordura.csv`)

- **Completitud de SHAP:** error medio |f(x) − E[f(fondo)] − Σ SHAP| frente a la media de |f(x) − E[f(fondo)]|:

| Caso | Error relativo |
|---|---|
| `exp2_Nmax_M` | 5,0 % |
| `exp2_Nmax_D` | 9,4 % |
| `exp3_V4_D` | 9,8 % |
| `exp1_GC3` | 13,8 % |
| `exp3_V4_M` | 14,7 % |
| `exp1_GE` | 20,1 % |

  Los tres últimos superan el umbral del 10 % del anexo (**EXP4-1**). Sus SHAP deben leerse como aproximados.
- **Carga del modelo:** el modelo cargado reproduce sus predicciones guardadas con diferencia máxima ≤ 0,0002 t en el exp. 2 y 3, y ≤ 0,0029 t en el exp. 1. En el exp. 1 se carga con la copia de la capa de atención de este repositorio, con el mismo cálculo.
- **Atención:** Σ α·h reproduce la salida de la capa (diferencia máxima < 1e-6).
- **Método:** se usó `GradientExplainer` en las 60 unidades; no hizo falta `KernelExplainer`.

## 7. Limitaciones

- **Variables correlacionadas:** SHAP y la permutación reparten la importancia entre variables correlacionadas: rezagos de la serie entre sí, clima entre sí y con el calendario, INDECI con el clima. La permutación crea combinaciones de entradas poco realistas, por ejemplo clima de un mes con calendario de otro. Agregar por grupo lo atenúa, pero los grupos también están correlacionados.
- **Fondo:** los SHAP se miden frente a secuencias de entrenamiento. Un fondo de otro periodo daría otros valores, y esto explica en parte la diferencia entre SHAP y permutación en el exp. 2 M (§3).
- **Muestras pequeñas en M y en el exp. 1:** 12 meses explicados, 84 secuencias de fondo y 4 bloques de permutación. Los intervalos son muy anchos y la completitud es aproximada.
- **Sin causalidad:** SHAP, la permutación y la atención no muestran causas en el mundo real ni dicen si una variable mejora el pronóstico.
- **Posterior y sin auditoría independiente:** el análisis se hizo después de conocer los resultados de los experimentos 1–3. El diagnóstico por posición (EXP4-3) se añadió después de ver las figuras.

## 8. Desviaciones (`docs/DESVIACIONES.md`)

| Entrada | Qué pasó | Efecto en las cifras |
|---|---|---|
| **EXP4-1** | Error de completitud de SHAP > 10 % en `exp3_V4_M`, `exp1_GC3` y `exp1_GE`. No se recalcula, porque cambiaría el método congelado. | Ninguno; esos SHAP se leen como aproximados |
| **EXP4-2** | Solape de etiquetas en x4, corregido con `scripts/11_exp4_figuras.py` sin modificar `scripts/10_exp4.py` | Ninguno |
| **EXP4-3** | Diagnóstico posterior de importancia por posición (`scripts/12_exp4_diagnostico_posicion.py`), no previsto en el anexo | Ninguno en las cifras del anexo |

**Otras notas:**

- Se añadió `shap` 0.51.0 a `.venv311` (anexo §8; `requirements-exp4.txt`).
- `.gitignore`: se versionan los PDF de x1–x4 y x6, que solo tienen agregados. El PDF de x5 no se versiona, porque guarda los valores escalados de la serie (criterio D13).

## 9. Verificación final

- `scripts/04_hashes_protocolo.py --verificar`: OK.
- `scripts/08_exp3.py --verificar`: OK.
- `scripts/10_exp4.py --verificar`: OK.
- `pytest`: 61 pruebas aprobadas.

## 10. `git add` y mensaje de commit sugerido

`resultados/exp4/modelos/` (modelos, entradas escaladas, SHAP por muestra) y los PDF de x5 quedan fuera por `.gitignore`.

```
git add .gitignore docs/DESVIACIONES.md scripts/11_exp4_figuras.py scripts/12_exp4_diagnostico_posicion.py \
  resultados/exp4/atencion.csv resultados/exp4/comparacion_ablacion.csv resultados/exp4/cordura.csv \
  resultados/exp4/diagnostico_posicion.csv resultados/exp4/explicacion.log resultados/exp4/permutacion.csv \
  resultados/exp4/permutacion_repeticiones.csv resultados/exp4/shap_grupo.csv resultados/exp4/shap_grupo_semillas.csv \
  resultados/exp4/shap_rezago.csv resultados/exp4/shap_variable.csv \
  reportes/figuras/es/exp4 reportes/figuras/en/exp4 reportes/INFORME_EXP4.md
```

Comprobación antes del commit (debe salir vacía):

```
git diff --cached --name-only | grep -E "modelos/|x5_.*\.pdf"
```

**Mensaje de commit sugerido:**

```
Experimento 4: explicabilidad de la DualLSTM (SHAP, permutación y atención; descriptivo)

- 60 unidades reconstruidas con el código congelado (exp. 1 GC3/GE, exp. 2 N = máx, exp. 3 V4; Y = 2025)
  y pronósticos idénticos a los publicados; anexo congelado en b052f2d
- D: la historia de la serie domina; en V4 el clima es la exógena más usada, aunque su ablación «perjudica»
- M y exp. 1: sin grupo dominante estable (gran dispersión entre semillas)
- Desviaciones EXP4-1 (completitud SHAP > 10 % en 3 casos), EXP4-2 (figura x4) y EXP4-3 (diagnóstico posterior)
- Figuras x1-x6 (es/en) y reportes/INFORME_EXP4.md (DESCRIPTIVO / POSTERIOR, sin auditoría independiente)
```
