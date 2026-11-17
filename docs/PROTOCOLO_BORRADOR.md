# PROTOCOLO — BORRADOR (no definitivo)

> **Estado: BORRADOR.** Este documento todavía puede cambiar. Antes de calcular **cualquier** métrica sobre los años de prueba se congelará una versión final (`docs/PROTOCOLO_CONGELADO.md`), con su SHA-256 y la fecha, y YesaR le pondrá una etiqueta git. Si algo cambia después de congelarlo, se registrará como desviación, con la fecha y el motivo.
>
> Redactado el 2026-11-14 a partir del reconocimiento del día 1 (`reportes/INFORME_DIA1.md`). Las cifras citadas salen de la muestra de ese día, no de la serie completa.

## 1. Pregunta

¿La DualLSTM-Bahdanau del artículo pierde frente a los modelos estadísticos por la arquitectura o por la longitud de la historia? Hipótesis operativa: si el problema es la cantidad de datos, la **diferencia relativa** de error (red frente al mejor modelo clásico) se reduce o se invierte a medida que crece la historia de entrenamiento N.

## 2. Datos y variable objetivo

**Fuente:** SISAP v2.0, módulo de Mercados Mayoristas, mercado *Gran mercado mayorista de lima* (código `15011501`), género *Limon* (`0611`), variedades *Limon sutil bolsa* (`061101`) y *Limon sutil cajon* (`061102`).

**Objetivo principal (propuesto):** volumen diario total de limón sutil que ingresa al GMML, en t/día:

`y_t = Volumen Total(Limon Sutil Bolsa)_t + Volumen Total(Limon Sutil Cajon)_t`

Se suman las dos presentaciones porque la muestra muestra un traslado del cajón a la bolsa: el cajón pasó de 71 960 t en 2019 a 16 486 t en 2025, mientras que la bolsa subió de 105 400 t a 166 344 t. Modelar una sola presentación mezclaría ese cambio de envase con la dinámica de la oferta.

**Objetivo secundario:** precio promedio diario del limón sutil en bolsa (S/ por kg). En la muestra viene todos los días, incluidos los días sin volumen.

**Decisiones pendientes (se cierran antes de congelar):**

1. **Celdas «...» en volumen.** Hay que decidir si significan ingreso nulo o falta de registro. Propuesta: tratarlas como 0 en el análisis principal, porque se concentran en los miércoles, donde la mediana de 2019 ya es de 5 t, y hacer una sensibilidad excluyéndolas.
2. **Inicio de la serie.** El SISAP tiene datos mensuales desde enero de 1997, pero la resolución diaria antes de 2016 **no se comprobó**. Además, hay posibles rupturas: los volúmenes mensuales llevan decimales hasta 2005 y son enteros desde 2006, y la sede del mercado pudo cambiar (no comprobado). Propuesta: serie principal desde el primer día de 2013 si se comprueba la resolución diaria; si no, desde 2016-01-01 (3 971 días hasta 2026-11-14).
3. **Transformación.** Propuesta: `log1p(y)` para ajustar y deshacer la transformación para evaluar. Las métricas se calculan siempre en la escala original (t).

**Alternativa semanal**, si la serie diaria resulta demasiado irregular: suma por semana ISO (lunes a domingo) de `y_t`. Da unas 565 semanas desde 2016 y unas 1 550 desde 1997. Con esa cantidad, la curva de aprendizaje solo llega a N ≈ 500, lo que reduce su alcance.

**Lo que ya se sabe de la serie diaria (muestra de 2019, bolsa):** tiene una estacionalidad semanal muy marcada. Las medianas por día son: lunes 416 t, martes 130 t, miércoles 5 t, jueves 351 t, viernes 378 t, sábado 351 t y domingo 276 t. También hay estacionalidad anual (cosecha del norte). Cualquier modelo tiene que capturar el ciclo de 7 días.

## 3. Diseño de la curva de aprendizaje

- **Años de prueba fijos:** 2024 y 2025 (730 días). 2026 queda como reserva para una confirmación posterior y no se usa para elegir nada.
- **Validación:** 2023 se usa para ajustar hiperparámetros de las redes y para la parada temprana. Nunca se tocan los años de prueba.
- **Tamaños de historia** N ∈ {120, 360, 1 000, 3 000, máx}, donde «máx» es toda la historia disponible antes del origen. En cada origen, el modelo se entrena con los **últimos N días** anteriores. Así se varía solo la longitud de la historia y no el período.
- **Rolling origin:** un origen por día de prueba para h = 1. Los modelos se reajustan cada 28 días; entre reajustes se actualiza el estado o el contexto con los datos observados, pero sin reestimar parámetros. El mismo calendario se aplica a todos los modelos.
- **Horizontes:** h = 1 día (principal) y h = 7 días (sensibilidad: pronóstico directo o recursivo según el modelo, y se evalúa el día t + 7).
- **Semillas:** 5 semillas por cada combinación de red y N. Se reporta la media y la dispersión.
- **Restricción con N pequeño:** con N = 120 no caben la estacionalidad anual ni el Seasonal Naive anual. En ese caso se reportan como «no aplicable», no se rellenan.

## 4. Modelos

| Modelo | Detalle propuesto |
|---|---|
| Seasonal Naive semanal | ŷ_{t+h} = y_{t+h−7k}, con el menor k que cumpla 7k ≥ h |
| Seasonal Naive anual | ŷ_t = y_{t−364} (52 semanas, para conservar el día de la semana). Solo con N ≥ 364 |
| ETS | estacionalidad m = 7 (aditiva sobre `log1p`) |
| Theta | desestacionalización con m = 7 |
| SARIMA o TBATS | SARIMA(p,d,q)(P,D,Q)₇ con términos de Fourier anuales como regresores, o TBATS con periodos {7; 365,25}. Se elige uno en validación (2023) y se congela |
| Chronos-Bolt-tiny | cero disparos (*zero-shot*), con el contexto igual a los últimos N días. **Verificar** la longitud máxima de contexto del modelo; si N la supera, se trunca y se documenta |
| DualLSTM-Bahdanau | **misma arquitectura** que el artículo (se copia el código del repositorio del artículo, que se usa solo en lectura; no se copian pesos). Se reentrena desde cero con cada N |
| LSTM univariada | línea base neuronal sin atención, con la misma ventana de entrada y el mismo presupuesto de ajuste que la DualLSTM |

Las ventanas de entrada de las redes y los hiperparámetros se eligen en validación con un presupuesto fijo, igual para cada N, y se congelan.

## 5. Métricas y comparación

- **Principal:** MASE, con escala calculada sobre la ventana de entrenamiento con Seasonal Naive m = 7. Se reporta también MAE y RMSE en t.
- **No se usa MAPE ni sMAPE como métrica principal**, porque los miércoles hay valores cercanos a 0.
- **Curva de aprendizaje:** MASE en función de N para cada modelo, además del cociente MASE(DualLSTM) / MASE(mejor modelo clásico) en función de N.
- **Pruebas:** Diebold-Mariano con corrección de Harvey et al. por pares en cada N (h = 1 y h = 7), con corrección de Holm para comparaciones múltiples.
- **Criterio de interpretación (se fija antes de ver la prueba):** se considera que «faltaban datos» si el cociente baja de forma monótona con N y, en N = máx, la DualLSTM no es significativamente peor que el mejor modelo clásico. Cualquier otro patrón se reporta tal cual.

## 6. Puente con el artículo

1. Agregar `y_t` a meses (suma) entre 2016 y 2025.
2. Correlacionarlo con la producción nacional mensual de limón de SISAGRI usada en el artículo: Pearson y Spearman en niveles y en variaciones interanuales, con rezagos de 0 a 2 meses.
3. Repetir la comparación de modelos sobre la serie mensual del GMML (~120 meses) para comprobar que con esa longitud se reproduce el resultado del artículo.

## 7. Congelamiento

Antes de evaluar sobre 2024–2025:

1. Cerrar las decisiones pendientes de la sección 2.
2. Copiar este archivo a `docs/PROTOCOLO_CONGELADO.md` sin la marca de borrador, con la fecha y el SHA-256 del archivo.
3. YesaR hace commit y tag (p. ej., `protocolo-v1`).
