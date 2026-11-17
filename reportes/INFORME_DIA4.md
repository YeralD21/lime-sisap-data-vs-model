# Experimento 2, día 4: figuras, diagnósticos y evidencia

Hoy no se reentrenó ningún modelo del análisis ni se recalculó la regla de decisión. Las etiquetas del día 3 no cambian: «el modelo» en D y en M, y conclusión conjunta «el modelo» (`reportes/INFORME_DIA3.md`). La única ejecución nueva es la prueba de perturbación (T3), un diagnóstico posterior registrado en D12.

Toda cifra de este informe sale del archivo citado. Leyendas completas en los dos idiomas: `reportes/figuras/LEYENDAS.md`.

**Scripts nuevos:**
- `scripts/05_figuras.py --idioma en|es`: lo ejecuta `.venv`, porque `.venv311` no tiene matplotlib. Es reanudable y `--forzar` rehace las salidas.
- `scripts/06_perturbacion.py`: lo ejecuta `.venv311`, porque necesita TensorFlow.

**Salidas:**
- `reportes/figuras/en/` y `reportes/figuras/es/`: 18 figuras en PDF vectorial y PNG a 300 ppp;
- `reportes/figuras/tablas/`: tablas en CSV y LaTeX en los dos idiomas;
- `reportes/figuras/datos/`: datos agregados de cada figura, sin valores de la serie.

---

## 1. Figuras y tablas

| Tarea | Archivo | Leyenda corta | Datos |
|---|---|---|---|
| T1.1 | `f01_curva_aprendizaje` | MAE frente a N (log), 7 modelos, IC 95 %, D y M | `resultados/brazo_*/principal/metricas.csv` |
| T1.2 | `f02_brecha` | Δ(N) con IC 95 %, línea en 0, * si DM p < 0,05, ref*(N) bajo cada punto | `resultados/brazo_*/principal/delta.csv` |
| T1.3 | `f03_brecha_pct` y `t1_mae_dual_ref_brecha` | Δ(N) en % del MAE de ref* | `reportes/figuras/tablas/t1_mae_dual_ref_brecha.csv` |
| T1.4 | `t1_mae_dual_ref_brecha.{csv,en.tex,es.tex}` | MAE de la DualLSTM y de ref* por N y brazo, Δ, Δ % y p | ídem |
| T2.1 | `f04_rolling_origin` | Entrenamiento, validación y prueba por año y N | `reportes/figuras/datos/f04_ventanas.csv` |
| T2.2 | `t2_pruebas.{csv,en.tex,es.tex}` | 20 pruebas: qué comprueban y resultado | `reportes/figuras/datos/pytest_junit.xml` |
| T2.3 | `t2_hashes.{csv,en.tex,es.tex}` | SHA-256 congelados frente a actuales | `docs/hashes_protocolo.json`, `reportes/figuras/tablas/t2_hashes.csv` |
| T3 | `f05_perturbacion` | Qué no cambia y qué sí al alterar el futuro | `reportes/figuras/datos/perturbacion.csv` |
| T4.2 | `f06_epoca_parada` | Mejor época y época de parada por N, 50 unidades por caja | `resultados/brazo_*/principal/estados_redes.csv`, `reportes/figuras/datos/f06_epocas.csv` |
| T4.4 | `f07_variabilidad_semillas` | MAE de las 10 semillas por N, con ref* como línea | `reportes/figuras/datos/f07_mae_por_semilla.csv` |
| T5.1 | `f08_D_2025` | D 2025: observado, DualLSTM, SARIMA, Seasonal Naive, y detalle de 8 semanas | `pronosticos.csv` + `data/processed` (no versionados) |
| T5.2 | `f09_D_2023_episodio` | D, febrero a junio de 2023 | ídem |
| T5.3 | `f10_M_2021_2025` | M 2021–2025: observado, DualLSTM, ETS, Seasonal Naive | ídem |
| T6.1 | `f11_mae_dia_semana_D` | MAE por día de la semana (D, N = máx) | `reportes/figuras/datos/f11_f12_mae_calendario.csv` |
| T6.2 | `f12_mae_mes` | MAE por mes del año (D y M, N = máx) | ídem |
| T6.3 | `f13_acf_residuos` | ACF de los residuos de la DualLSTM y de ref* (N = máx) | `reportes/figuras/datos/f13_acf_residuos.csv` |
| T6.4 | `f14_shock_vs_normal` | MAE en shock frente a normal, P50–P90 (§6) | `reportes/figuras/datos/f14_shock_vs_normal.csv` |
| T7 | `f15_dm_heatmap` y `t7_top_ranked` | p DM-HLN de la DualLSTM frente a cada referencia; tabla top-ranked (sens. 5 marcada iid) | `reportes/figuras/datos/f15_dm_pares.csv`, `resultados/brazo_*/{principal,sens5_bloques}/top_ranked.csv` |
| T8.1 | `f16_M_N120_vs_max` | Brecha en M con N = 120 (artículo) frente a N = máx | `resultados/brazo_M/principal/delta.csv` |
| T8.2 | `f17_sisap_vs_sisagri` | SISAP mensual frente a SISAGRI 2016–2025, con r | `reportes/dia2_mensual_vs_sisap.csv`, `reportes/dia2_correlacion_sisagri.csv` |
| T8.3 | `f18_tiempos` | Segundos por unidad (mediana) por N y modelo | `reportes/figuras/datos/f18_tiempos_por_unidad.csv`; por fase: `resultados/tiempos_dia3.csv` |

### Lo que dicen las figuras (descriptivo; no cambia ninguna etiqueta)

- **Brecha en porcentaje** (`t1_mae_dual_ref_brecha.csv`):
  - D: baja de 112.2 % (N = 120) a 36.2 % [31.2; 41.1] (N = máx).
  - M: baja de 92.7 % (N = 60) a 37.1 % [16.4; 57.8] (N = máx).
- **Semillas** (`f07_mae_por_semilla.csv`): ninguna de las 10 semillas de la DualLSTM llega al MAE de ref* en ningún N ni en ningún brazo.
  - D, N = máx: las semillas van de 115.1 a 126.9 t, frente a 89.3 t de SARIMA.
  - M, N = máx: van de 1 132.6 a 1 642.3 t, frente a 972.5 t de ETS (`delta.csv`).
  - En M la dispersión entre semillas de la DualLSTM es mucho mayor que la de la LSTM univariada (de 1 260.4 a 1 330.5 t con N = máx).
- **Día de la semana en D** (`f11_f12_mae_calendario.csv`): la mayor diferencia relativa es el miércoles. La DualLSTM tiene 76.3 t de MAE y SARIMA 19.4 t; el Seasonal Naive, 31.9 t. El martes, 75.2 t frente a 44.3 t. El lunes y el jueves son los días de mayor error para todos los modelos.
- **ACF de los residuos en D** (`f13_acf_residuos.csv`, banda ±0.046):
  - la DualLSTM tiene autocorrelación semanal marcada en los rezagos 7, 14, 21 y 28 (0.217, 0.350, 0.332 y 0.337); SARIMA no pasa de 0.063 en esos rezagos.
  - Es decir, la red deja sin capturar parte del patrón semanal que SARIMA sí modela.
- **ACF de los residuos en M** (banda ±0.255): el rezago 1 es 0.610 en la DualLSTM y 0.280 en ETS.
- **Shocks en D** (`f14_shock_vs_normal.csv`): todos los modelos se equivocan más en días de shock. La DualLSTM queda por encima de SARIMA tanto en días de shock como en días normales, en los cuatro umbrales.
- **Shocks en M:** los grupos de shock son pequeños (27, 19, 10 y 2 meses), así que no se interpretan.
- **DM frente a cada referencia** (`f15_dm_pares.csv`):
  - en D, la DualLSTM es significativamente mejor que Chronos en todos los N, que ETS con N = 120 (el ETS inestable del día 3) y que Theta con N = 360 y N = máx.
  - En D, con N = máx, no difiere significativamente del Seasonal Naive (HLN 1.88).
  - En M no difiere significativamente de SARIMA ni del Seasonal Naive con N ≥ 120 (|HLN| ≤ 1.38). Frente a ETS, Theta y Chronos es significativamente peor en todos los N.
- **Épocas** (`f06_epocas.csv`): la mediana de la mejor época de la DualLSTM baja con N (D: de 27 a 11; M: de 28.5 a 15). La parada llega siempre 15 épocas después de la mejor (paciencia 15). Ocurre en las 1 200 unidades de la principal: 800 de D con h = 1 y 7, y 400 de M (`estados_redes.csv`).
- **Puente con el artículo** (`f16`, `delta.csv`): en M, la brecha con N = 120 es 460.1 t (47.1 %) y con N = máx es 360.6 t (37.1 %). Las dos son significativas.
- **Hashes** (`t2_hashes.csv`): coinciden los 9 archivos congelados.
- **Pruebas** (`t2_pruebas.csv`): pasan las 20.

---

## 2. Lo que no se pudo producir y por qué

| Tarea | Estado | Motivo |
|---|---|---|
| T4.1 Curvas de pérdida train y val por época | **No producida** | El día 3 no guardó historiales por época. `estado.json` de cada unidad solo tiene `epocas`, `mejor_epoca` y `segundos`, y los checkpoints se borraban tras predecir (`src/redes.py`, `entrenar_y_predecir`). Tampoco hay pérdidas finales. No se reconstruyó reentrenando. En su lugar están `epocas` y `mejor_epoca` (f06). |
| T4.3 Brecha de generalización (val − train en la mejor época) | **No producida** | Mismo motivo: no hay pérdidas guardadas. |
| T4.1 «Semilla de mediana MAE con Y = 2025» | Sustituida | Al no haber curvas, f06 resume todas las semillas y ventanas. f07 muestra el MAE de cada semilla. |
| T6.4 «Días con cero supuesto» | No aplica | El protocolo sí define shocks (§6, P50–P90), así que se usó esa definición. |
| T8.3 Tiempos de redes por N | Con matiz | Se ejecutaron en paralelo en una máquina compartida entre los dos brazos, así que los segundos por unidad no son tiempos aislados (D4). |

**Para el futuro:** guardar `hist.history` (pérdida train y val por época) en `estado.json` costaría muy poco espacio. Requiere cambiar `src/redes.py` y reentrenar, así que queda fuera del día 4.

---

## 3. Prueba de perturbación (T3; diagnóstico posterior, D12)

Unidad por brazo: principal, Y = 2025, N = máx, h = 1, DualLSTM, semilla 0.
- **Perturbación:** todas las observaciones (y, y_eval, precio) posteriores al origen se sustituyen por valores uniformes al azar dentro del rango de la serie (semilla 20250101). El origen es el fin de la validación: 2024-12-31 en D y 2024-12 en M.
- **Ejecución:** escalado, secuencias, entrenamiento y pronóstico se repiten en subprocesos nuevos, con el mismo determinismo que el día 3.

Fuente: `reportes/figuras/datos/perturbacion.csv`.

| Comparación (perturbada frente a original) | D | M |
|---|---|---|
| Escalador (media y escala de y) | 0 | 0 |
| Secuencias y objetivos de entrenamiento | 0 | 0 |
| Secuencias y objetivos de validación | 0 | 0 |
| Entrada del primer paso de prueba | 0 | 0 |
| Épocas y mejor época | 0 | 0 |
| **Pronóstico del primer paso de prueba** | **0** | **0** |
| Pronóstico del primer paso: original frente a publicado el día 3 | 0 | 0 |
| **Valor observado del primer paso** | **828 t** | **7 915 t** |
| Entrada del último paso de prueba | 8.95 (escala estandarizada) | 4.92 (escala estandarizada) |
| Pronósticos del resto de la prueba (máx.) | 890 t | 4 854 t |

**Lectura:**
- Nada posterior al origen entra en el escalador, el entrenamiento, la selección de época ni el pronóstico del primer paso: el resultado es idéntico bit a bit.
- El valor observado de ese paso sí cambia, así que la prueba tenía efecto.
- Los pasos posteriores cambian porque sus entradas usan observaciones de la prueba anteriores a cada t. Eso es lo previsto en un pronóstico a un paso con el estado actualizado.

Figura: `f05_perturbacion`.

---

## 4. Decisiones de este día (registradas en `docs/DESVIACIONES.md`)

- **D11:** los informes de brazo se corrigieron a mano tras generarse; la fuente de verdad son los CSV.
- **D12:** la prueba de perturbación es un diagnóstico posterior.
- **D13: hay datos del SISAP dentro de `pronosticos.csv`.**
  - El pronóstico del Seasonal Naive es la serie imputada desplazada 7 días o 12 meses. Coincide con el observado en 1 818 de 1 826 fechas (D) y en 59 de 60 meses (M). El día 3 no se detectó.
  - **Decisión provisional:** `pronosticos.csv`, `base_pronosticos.csv` y `reportes/auditoria/reproduccion_*_detalle.csv` van a `.gitignore` y quedan en el respaldo. Se versionan solo las tablas agregadas.
  - **Figuras T5:** se versiona el PNG, como las figuras de la serie del día 2, y no el PDF vectorial, que guarda las coordenadas exactas.
  - **La decisión final es de YesaR.**
- **D9 sigue vigente:** la tabla top-ranked de la sensibilidad 5 no se recalcula y en `t7_top_ranked` aparece marcada como iid.

---

## 5. Cierre (T9)

- **Respaldo:** `D:\respaldo_dia3_unidades.zip`, 7 711 archivos y 30 591 667 bytes. SHA-256 `24372c39f7246dabe79ca1232cb4169ffa06d81ed3fa7bb9eb1902a6634127b2` (`reportes/figuras/datos/respaldo_dia3.csv`).
  - **Contenido:** `resultados/brazo_D/redes_unidades/`, `resultados/brazo_M/redes_unidades/`, `resultados/brazo_D/estadisticos/` y, por D13, todos los `pronosticos.csv` y `base_pronosticos.csv`.
  - Pasó `testzip`.
  - Los originales **no se borraron**: siguen en disco y quedan fuera de git por `.gitignore`.
- **`.gitignore`:** se añadieron esas rutas, los archivos con la serie (D13) y las PDF de T5. También una excepción `!reportes/figuras/*/*.pdf`, porque la regla `*.pdf` del día 1 (pensada para PDF descargados) ignoraba las figuras vectoriales.
- **Tamaño:** de los 257 archivos que quedan por versionar, ninguno supera 10 MB (y por tanto tampoco 100 MB). Suman 8.7 MB.
- **`.venv\Scripts\python scripts/04_hashes_protocolo.py --verificar`:** «OK: todos los hashes coinciden».
- **`.venv311\Scripts\python -m pytest -q tests`:** 20 passed.

---

## 6. Archivos para `git add` y mensaje de commit sugerido

Con el `.gitignore` actualizado, añadir las carpetas completas no incluye los archivos con la serie:

```
git add .gitignore requirements-redes.txt docs/DESVIACIONES.md src tests \
  scripts/05_figuras.py scripts/06_perturbacion.py \
  reportes/INFORME_DIA3.md reportes/INFORME_DIA4.md reportes/brazo_D.md reportes/brazo_M.md \
  reportes/auditoria reportes/figuras resultados
```

Comprobación antes del commit (las dos órdenes deben salir vacías):

```
git diff --cached --name-only | grep -E "pronosticos|redes_unidades|estadisticos/|_detalle\.csv|f0[89]_.*\.pdf|f10_.*\.pdf"
git diff --cached --name-only | grep -E "^(docs/PROTOCOLO\.md|docs/hashes_protocolo\.json|data/|scripts/0[1-4]_)"
```

**Mensaje de commit sugerido:**

```
Días 3 y 4 del experimento 2: curvas de aprendizaje, auditoría y figuras

Día 3
- src/ y tests/: código común (datos, ventanas rolling origin, redes, evaluación, análisis) y 20 pruebas
- resultados/brazo_D y brazo_M: tablas de análisis, estados de redes, incidencias, scripts y registros
  del escenario principal y de las sensibilidades 1-7 (sin pronosticos.csv: contienen la serie, D13)
- reportes/: informes de brazo, auditoría independiente e INFORME_DIA3
- Etiqueta (§1): «el modelo» en D y en M; conclusión conjunta «el modelo»

Día 4
- scripts/05_figuras.py: 18 figuras (en/es, PDF y PNG) y tablas CSV/LaTeX en reportes/figuras/
- scripts/06_perturbacion.py: prueba de perturbación (diagnóstico posterior, D12)
- reportes/INFORME_DIA4.md y reportes/figuras/LEYENDAS.md
- docs/DESVIACIONES.md: D1-D13
- .gitignore: salidas por unidad y archivos con la serie del SISAP fuera del repositorio
  (respaldo externo con SHA-256 en reportes/figuras/datos/respaldo_dia3.csv)
```
