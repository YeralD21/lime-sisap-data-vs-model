# Desviaciones del protocolo — Experimento 2

Registro exigido por `docs/PROTOCOLO.md` (cabecera): fecha, motivo, efecto y si ocurrió antes o después de ver errores de prueba.

> **Ubicación de este archivo.** El prompt del día 3 pedía `reportes/DESVIACIONES.md`, pero el protocolo congelado (9ede96c, cabecera) fija `docs/DESVIACIONES.md`. Gana el protocolo, así que el registro está aquí.

---

## D1 — Orden de commits: los datos y scripts de los días 1–2 se versionaron después de la congelación

- **Fecha de registro:** 2026-11-17.
- **Momento:** antes de entrenar cualquier modelo y antes de calcular cualquier error de prueba.
- **Hecho:** el commit de congelación `9ede96c` contiene solo `docs/PROTOCOLO.md` y `docs/hashes_protocolo.json`. El HEAD es `4f13d8f`, cuyo padre directo es `9ede96c`, y añade los scripts 01–04, `sisap_comun.py`, los informes de los días 1 y 2 y la configuración.
- **Verificación:**
  - `git diff 9ede96c 4f13d8f -- docs/PROTOCOLO.md docs/hashes_protocolo.json` está vacío;
  - `.venv\Scripts\python scripts/04_hashes_protocolo.py --verificar` → «OK: todos los hashes coinciden» (los hashes de los scripts congelados en `9ede96c` coinciden con los versionados en `4f13d8f`).
- **Efecto sobre el protocolo:** ninguno. Es solo el orden de los commits.

---

## D2 — Cambio de intérprete para las redes y Chronos: `.venv311` (Python 3.11.9)

- **Fecha:** 2026-11-17. **Momento:** antes de entrenar cualquier modelo del experimento.
- **Motivo:** el prompt del día 3 indicaba `.venv` (Python 3.14.3), pero TensorFlow no tiene distribución para Python 3.14 (`pip`: «No matching distribution found for tensorflow»). YesaR eligió crear `.venv311` en este repositorio.
- **Hecho:**
  - `.venv311` usa Python 3.11.9 con TensorFlow 2.21.0, Keras 3.14.0, torch 2.11.0+cpu, chronos-forecasting 2.3.2, statsmodels 0.14.6, scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.2 y scipy 1.17.1. Son las mismas versiones que el venv del artículo.
  - Versiones completas en `requirements-redes.txt` (`pip freeze`). `.venv311/` está en `.gitignore`.
  - `ml_dtypes` se fijó en 0.5.4, la del venv del artículo: la 0.7.0 que instaló `pip` no cargaba porque el Control de aplicaciones de Windows bloquea su DLL («Una directiva de Control de aplicaciones bloqueó este archivo»).
  - `scripts/04_hashes_protocolo.py --verificar` sigue ejecutándose con `.venv`.
- **Efecto sobre el protocolo:** ninguno (el protocolo no fija el intérprete).

## D3 — Versión del artículo citada en §1

- **Fecha:** 2026-11-17. **Momento:** antes de entrenar cualquier modelo.
- **Motivo:** §1 cita la release `v2.2-forecasting` del repositorio `duallstm-lime-forecasting-peru`, pero esa etiqueta todavía no existe (la única es `v2.1-forecasting`). YesaR la creará sobre `eee9a71`.
- **Hecho:** el código copiado (arquitectura de `v2_reentrenamiento/src/models/dual_lstm_attention.py`; bootstrap, ranking y DM-HLN de `experimentos_backtesting_rodante/02_scripts/lib/evaluacion.py`; referencias de ETS y Theta en `13_revision_r2/scripts/r2lib.py`) se toma del commit **eee9a71 (futura v2.2-forecasting)**, sin copiar pesos.
- **Efecto:** desviación menor de §1, solo de la cita.

## D4 — Ejecución paralela de las redes

- **Fecha:** 2026-11-17. **Momento:** antes de calcular cualquier error de prueba.
- **Motivo:** tiempo de cómputo. YesaR aprobó unos 6 procesos en paralelo, sin recortes.
- **Hecho:** cada corrida (escenario, ventana, N, horizonte, modelo, semilla) se ejecuta en un proceso nuevo (`src/redes.py`) con:
  - `PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1` y `TF_ENABLE_ONEDNN_OPTS=0`;
  - la semilla de la corrida fijada antes de entrenar (`keras.utils.set_random_seed`) y `tf.config.experimental.enable_op_determinism()`;
  - `intra_op = 2` e `inter_op = 1` hilos.
- **Comprobación:** se corrió la unidad D principal, Y = 2025, N = 1000, h = 1, DualLSTM, semilla 3, primero sola y después junto con otras 5 en 6 procesos paralelos. Las 731 predicciones (validación y prueba) son idénticas: diferencia máxima 0,0 t, 33 épocas y mejor época 18 en las dos ejecuciones. La comprobación se hizo en un directorio temporal y no dejó artefacto. La respalda la reejecución del auditor, que da diferencia 0 frente a la unidad publicada, corrida en paralelo (`reportes/auditoria/reproduccion_D.csv`).
- **Efecto sobre los resultados:** ninguno.

## D5 — Sensibilidad 1, brazo M: 17 meses evaluables en lugar de 18

- **Fecha:** 2026-11-17. **Momento:** antes de calcular cualquier error de prueba.
- **Motivo:** la tabla de §2.2 da 7 meses evaluables en 2021 (18 en total) para la sensibilidad 1 del brazo M. Esa cifra sale de `flag_faltante_estricto` de `reportes/dia2_evaluables_por_anio.csv`, que no marca 2021-02. Sin embargo, §2.2 establece que en el brazo M «el mes 2021-02 es faltante: se imputa en las entradas y no se evalúa», y la sensibilidad 1 solo cambia el tratamiento de «...» (§9), no el de la anomalía. En el brazo D, la misma tabla sí excluye la anomalía en la sensibilidad 1 (298 días en 2021).
- **Decisión:** se aplica la regla de §2.2. 2021-02 no se evalúa en la sensibilidad 1 de M, que queda con 6, 2, 5, 4 y 0 meses en 2021–2025 (**17**). La tabla de §2.2 y la regla escrita del mismo apartado se contradicen, y se aplica la regla.
- **Efecto:** un mes menos en una sensibilidad que no cambia la etiqueta. El análisis principal no cambia.

## D6 — Brazo M: interpretaciones de ejecución (propuestas por el agente del brazo M)

- **Fecha:** 2026-11-17. **Momento:** antes de lanzar la corrida completa y antes de calcular errores de prueba de la corrida final. Una prueba previa (Y = 2023, N = 120) mostró el MAE de las referencias en esa ventana; no se cambió nada después de verlo (`reportes/brazo_M.md`, §8).
- **Chronos con N = máx:** el contexto es una ventana móvil de longitud fija igual a la historia máxima de la ventana (`Particion.n_hist`, 267–315 meses), que termina en t−1. Es la lectura literal de §5 («las últimas N observaciones hasta t−1», con N = máx por ventana según §4). No se truncó ningún contexto.
- **ETS:** se elige el menor AICc entre las especificaciones con AICc finito, aunque no hayan convergido, como `r2lib.seleccionar_ets`. El protocolo no lo define. No hubo ni fallos ni falta de convergencia (`resultados/brazo_M/*/incidencias.csv`).
- **Estado de ETS, Theta y SARIMA:** se recalcula desde el inicio de la historia N con parámetros fijos, como en el artículo.
- **Sensibilidad 1:** se ejecutó también la ventana 2025, que no tiene meses evaluables. Solo costó cómputo.
- **Efecto sobre el protocolo:** ninguno. Son aclaraciones de puntos que el protocolo no especifica.
- **Nota:** la DualLSTM del brazo M tiene 4 545 parámetros (`expected_parameter_count(4, 6)`). El 4 289 del encargo era un error de cálculo del integrador, no del código.

## D7 — Brazo D: interpretaciones de ejecución (propuestas por el agente del brazo D)

- **Fecha:** 2026-11-17. **Momento:** todas se decidieron antes de calcular errores de prueba (`reportes/brazo_D.md`, §8).
- **ETS y SARIMA, «estado actualizado sin reestimar»:** se calcula con un solo `smooth(params)` por ventana sobre toda la serie, en lugar de repetirlo día a día. Se comprobó contra el procedimiento literal en 3 ventanas, 6 días y h = 1 y 7: diferencia 0 (`resultados/brazo_D/equivalencias.csv`). Antes de lanzar las corridas, esa misma comprobación detectó un desplazamiento de un día en el ancla de `simulate` del ETS, que se corrigió.
- **Chronos:** pronostica en lotes. La diferencia máxima con un contexto suelto es 1,2×10⁻⁴ t. Con N = máx, la historia (8 127–9 588 días) supera el contexto máximo de 2 048 que admite el modelo, así que el contexto se trunca a 2 048 en todos los pronósticos, como prevé §5 («usa solo la cola y se reporta»). Chronos con N = máx equivale a N = 2 048. Es coherente con el brazo M (D6), donde la historia máxima, de 267–315 meses, cabe entera.
- **Theta:** se copió la lógica de `r2lib.ThetaFijo` con periodo 7 en `resultados/brazo_D/scripts/estadisticos_lib.py`.
- **ETS:** `maxiter = 1000`, el valor por defecto de statsmodels. El protocolo no lo fija. Como en el brazo M (D6), se elige el menor AICc entre las especificaciones con AICc finito, aunque no hayan convergido. Hubo de 1 a 4 especificaciones no convergidas de 12 por escenario; todos los ajustes elegidos convergieron (`resultados/brazo_D/<escenario>/incidencias.csv`).
- **SARIMA:** el pronóstico es exp(media en escala log), es decir la mediana. En los términos de Fourier, `t` es la posición del día en la serie del escenario.
- **Sensibilidad 6:** las redes con N = 120, 360 y 1 000 se reutilizan de la principal; solo se reentrenan las de N = máx.
  - Las series de esas historias son idénticas (`resultados/brazo_D/sens6_comprobacion.csv`).
  - La reutilización es exacta. El auditor comprobó que las entradas escaladas, incluido `t_index` de la rama B, son idénticas bit a bit, y que una unidad reentrenada da diferencia 0 (`reportes/auditoria/sens6_tindex.csv`).
  - Los estadísticos y Chronos se reejecutaron con todos los N. Coinciden con la principal salvo SARIMA con N = 1 000, con una diferencia máxima de 0,16 t por la fase de los Fourier (`resultados/brazo_D/sens6_vs_principal.csv`).
- **`pronosticos.csv`:** incluye todos los días del año de prueba, no solo los evaluables. El análisis filtra con `evaluables`.
- **Efecto sobre el protocolo:** ninguno. Son aclaraciones o atajos de cómputo comprobados como equivalentes.

## D8 — Orden de aplicación de la regla de decisión (hallazgo I1 de la auditoría)

- **Fecha:** 2026-11-17. **Momento:** el código (`src/evaluacion.py`, `regla_decision`) se escribió antes de calcular errores de prueba. Se registra aquí después de verlos, a raíz de la auditoría.
- **Hecho:** la tabla de §1 no fija el orden de las filas. El código evalúa:
  1. «el modelo» (Δ(máx) > 0 y significativa);
  2. el subcaso «no se reproduce la desventaja» (Δ no es significativamente positiva en ningún N), clasificado como «no concluyente»;
  3. «los datos» (Δ estrictamente decreciente y Δ(máx) no significativa o < 0);
  4. «no concluyente».
  El orden solo importa si se cumplen a la vez el subcaso y «los datos». En ese caso el código da «no concluyente», porque §1 dice que entonces «la premisa no se cumple en esta serie».
- **Efecto:** ninguno. En los 13 análisis con etiqueta, ese caso no se da. El único con el subcaso es la sensibilidad 1 de M, cuya Δ no es monótona (309,6 > 198,8 en N = 120 frente a 60; `resultados/brazo_M/sens1_estricto/delta.csv`), así que tampoco cumpliría «los datos». El auditor recalculó las 13 etiquetas con código propio y coinciden (`reportes/auditoria/recalculo_decision.csv`).

## D9 — Sensibilidad 5: «consistently top-ranked» con bootstrap iid (hallazgo menor 4)

- **Fecha:** 2026-11-17. **Momento:** registrado después de ver los resultados, a raíz de la auditoría.
- **Hecho:** en la sensibilidad 5 (bootstrap por bloques), el IC de Δ, el IC del MAE y el ranking se calculan por bloques (7 días en D, 3 meses en M). La tabla «consistently top-ranked» (`src/evaluacion.top_ranked`) sigue usando el bootstrap iid, porque esa función no recibe el tipo de bootstrap.
- **Efecto:** ninguno en la etiqueta, porque el top-ranked no entra en la regla de §1. Las tablas `top_ranked*.csv` de `resultados/brazo_*/sens5_bloques/` son iguales a las de la principal y no deben leerse como versión por bloques. No se corrigió para no reescribir resultados después de verlos; se puede corregir en un día posterior si YesaR lo pide.

## D10 — Aclaraciones no declaradas antes (hallazgos I2 y menores 6 y 11 de la auditoría)

- **Fecha:** 2026-11-17. **Momento:** las decisiones son anteriores a ver errores de prueba (están en el código común); se registran después, a raíz de la auditoría.
- **Pronósticos negativos:** no se truncan en 0, porque el protocolo no lo prevé. En la principal de D hay 4 638 pronósticos negativos de 365 200: ETS con N = 120 llega a −4 348 t y la DualLSTM con N = 120 tiene 620 a h = 1. En M no hay ninguno (`reportes/auditoria/pronosticos_negativos.csv`).
- **Early stopping:** la pérdida de validación (año Y − 1) usa los objetivos imputados con t − 7 en los días faltantes, igual que el entrenamiento.
- **RMSE de las redes:** es la media del RMSE de cada semilla, como en el artículo. El MAE, en cambio, se calcula sobre el error promediado entre semillas por fecha (§5).
- **Efecto:** ninguno en la etiqueta.

## D11 — Informes de brazo corregidos a mano (día 4, T9)

- **Fecha:** 2026-11-17. **Momento:** después de ver los resultados, tras la auditoría del día 3.
- **Hecho:** `reportes/brazo_D.md` y `reportes/brazo_M.md` se generaron con `resultados/brazo_D/scripts/generar_informe.py` y `resultados/brazo_M/scripts/05_informe.py`, y después se corrigieron a mano para resolver los hallazgos I2, I3, m1, m3 y m7–m9 de `reportes/auditoria/AUDITORIA_DIA3.md` (detalle en `reportes/INFORME_DIA3.md` §5).
- **Fuente de verdad:** los CSV de `resultados/`. Si se vuelven a ejecutar los generadores, las correcciones se pierden y hay que rehacerlas.
- **Efecto:** ninguno en cifras ni etiquetas.

## D12 — Prueba de perturbación (día 4, T3): análisis diagnóstico posterior

- **Fecha:** 2026-11-17. **Momento:** después de ver los resultados. Es un diagnóstico descriptivo, no forma parte del protocolo y no modifica ninguna etiqueta.
- **Hecho:** `scripts/06_perturbacion.py` (con `.venv311`) toma una unidad por brazo (principal, Y = 2025, N = máx, h = 1, DualLSTM, semilla 0). Sustituye al azar (semilla 20250101) todos los valores de y, y_eval y precio posteriores al origen (2024-12-31 en D; 2024-12 en M). Vuelve a ejecutar el escalado, las secuencias, el entrenamiento y el pronóstico en subprocesos nuevos, con el mismo determinismo que el día 3.
- **Por qué reentrena:** el día 3 no guardó checkpoints (se borraban tras predecir) ni historiales de pérdida. Para obtener el pronóstico del primer paso hay que reentrenar la unidad dos veces (serie original y perturbada).
- **Resultado** (`reportes/figuras/datos/perturbacion.csv`): el escalador, las secuencias y objetivos de entrenamiento y validación, la entrada del primer paso de prueba y su pronóstico son idénticos (diferencia 0) en los dos brazos. El pronóstico original coincide además con el publicado el día 3 (diferencia 0). El valor observado del primer paso cambia (828 t en D y 7 915 t en M), igual que las entradas y pronósticos de los pasos posteriores, que sí usan observaciones posteriores al origen.

## D13 — Datos del SISAP en los archivos de resultados y en las figuras (día 4)

- **Fecha:** 2026-11-17. **Momento:** después de ver los resultados; no afecta a ningún cálculo.
- **Hallazgo:** el pronóstico del Seasonal Naive guardado en `pronosticos.csv` es la serie imputada desplazada 7 días (D) o 12 meses (M). Coincide con el valor observado en 1 818 de 1 826 fechas en D y en 59 de 60 meses en M (comprobación del día 4 con `src.datos.serie`). Por tanto, `resultados/brazo_*/*/pronosticos.csv`, `resultados/brazo_M/*/base_pronosticos.csv` y `reportes/auditoria/reproduccion_*_detalle.csv` contienen la serie del SISAP, aunque no tengan la columna `y_true`. Esto no se detectó el día 3.
- **Decisión provisional** (la decisión final es de YesaR):
  - esos archivos se añaden a `.gitignore` y quedan respaldados en `D:\respaldo_dia3_unidades.zip` (SHA-256 en `reportes/figuras/datos/respaldo_dia3.csv`);
  - se versionan las tablas agregadas (métricas, Δ, ranking, top-ranked, semillas, decisión, estados de redes, incidencias), que no permiten reconstruir la serie.
- **Figuras con la serie observada (T5: f08, f09, f10):** se versiona el PNG a 300 ppp, como las figuras de la serie del día 2 (`reportes/figuras_dia2/`), y no el PDF vectorial, que guarda las coordenadas exactas de cada punto. Los PDF quedan en disco para el artículo y la presentación.
- **Efecto:** ninguno en resultados ni etiquetas. Sin `pronosticos.csv`, el análisis del día 3 se reproduce regenerando los pronósticos con los scripts de cada brazo.

---

# Experimento 3 (anexo `docs/ANEXO_EXP3.md`, congelado en 512b7cc)

## EXP3-1 — Error en la prueba de perturbación de `scripts/08_exp3.py`

- **Fecha:** 2026-11-17. **Momento:** después de entrenar las 500 corridas y de aplicar la regla (`--analizar`). La prueba es un diagnóstico y no interviene en la regla.
- **Hecho:** `_perturbar_exog` de `scripts/08_exp3.py` asignaba valores aleatorios decimales a columnas enteras de las exógenas (por ejemplo, `eventos`), pandas lo rechazó y la prueba no corrió.
- **Corrección:** `scripts/09_exp3_perturbacion.py` reutiliza 08 **sin modificarlo** (su SHA-256 sigue coincidiendo con `docs/hashes_anexo_exp3.json`). Solo sustituye esa función por otra que convierte antes las exógenas a decimal. La perturbación, el entrenamiento y las comparaciones son las del anexo §6.2.
- **Efecto:** ninguno en las etiquetas. La prueba sale como esperaba el anexo (`resultados/exp3/{D,M}/perturbacion.csv`).

## EXP3-2 — Figuras del experimento 3 sin retoques de maquetación

- **Fecha:** 2026-11-17. **Momento:** después de ver los resultados.
- **Hecho:** en `e2_delta_vs_V0` algunas anotaciones de p se solapan con el eje, y en `e3_curvas_perdida_Y2025` las etiquetas del eje y de las dos filas se solapan. Corregirlo exige cambiar `scripts/08_exp3.py`, congelado con el anexo, así que no se tocó. La corrección de maquetación queda para una versión posterior de las figuras.
- **Efecto:** ninguno en las cifras.

---

# Experimento 4 (anexo `docs/ANEXO_EXP4.md`, congelado en b052f2d)

## EXP4-1 — Error de completitud de SHAP mayor que el 10 % en tres casos

- **Fecha:** 2026-11-18. **Momento:** fase B, después de calcular SHAP (comprobación de cordura del anexo §3.5).
- **Hecho:** el error de completitud medio |f(x) − E[f(fondo)] − Σ SHAP| supera el 10 % de la media de |f(x) − E[f(fondo)]| (umbral del anexo) en tres casos (`resultados/exp4/cordura.csv`):
  - `exp3_V4_M`: 14,7 %;
  - `exp1_GC3`: 13,8 %;
  - `exp1_GE`: 20,1 %.

  En los otros tres casos queda por debajo: `exp3_V4_D` 9,8 %, `exp2_Nmax_D` 9,4 % y `exp2_Nmax_M` 5,0 %. Expected gradients cumple la completitud solo en esperanza (`nsamples = 200`); en los casos mensuales el fondo tiene 84 secuencias y hay solo 12 muestras explicadas.
- **Decisión:** no se recalcula con otro `nsamples`, porque cambiaría el método congelado. Se declara en el informe, y los SHAP de esos tres casos se leen como aproximados, sobre todo en el orden de grupos con importancias parecidas.
- **Efecto:** ninguno en las cifras publicadas; sí en cuánta confianza merecen los rankings SHAP de esos tres casos.

## EXP4-2 — Corrección de maquetación de la figura x4

- **Fecha:** 2026-11-18. **Momento:** después de generar las figuras.
- **Hecho:** en `x4_shap_vs_ablacion` las etiquetas del eje x «Permutación» y «Ablación exp. 3» se solapaban.
- **Corrección:** `scripts/11_exp4_figuras.py` reutiliza `scripts/10_exp4.py` **sin modificarlo** (su SHA-256 sigue coincidiendo con `docs/hashes_anexo_exp4.json`) y solo parte esas etiquetas en dos líneas. Todas las figuras x1–x6 versionadas salen de ese script.
- **Efecto:** ninguno en las cifras.

## EXP4-3 — Diagnóstico posterior de la importancia por posición

- **Fecha:** 2026-11-18. **Momento:** después de ver `x2_shap_por_rezago`. No estaba previsto en el anexo.
- **Hecho:** el |SHAP| crece hacia la posición más antigua (t − 6) en todos los grupos y casos. Para saber si esto viene del modelo o del método, `scripts/12_exp4_diagnostico_posicion.py` hace una sustitución por posición en cada unidad:
  - sustituye una posición completa (ramas A y B) por la misma posición de secuencias del fondo (10 sorteos, semilla 42);
  - mide el cambio absoluto medio del pronóstico (`resultados/exp4/diagnostico_posicion.csv`).
- **Resultado:** el patrón se repite, así que viene de los modelos, no de SHAP.
- **Efecto:** ninguno en las cifras del anexo. Es descriptivo y se informa como posterior.
