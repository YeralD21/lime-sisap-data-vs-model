# Informe del día 2: serie completa, calidad de datos y protocolo para congelar

- **Fecha:** 2026-11-14
- **Scripts:**
  - `scripts/02_descarga_sisap.py`: descarga reanudable
  - `scripts/03_series.py`: series, calidad y diagnóstico
  - `scripts/04_hashes_protocolo.py`: hashes
- **Cifras de respaldo:** `reportes/dia2_diagnostico.json` y las tablas `reportes/dia2_*.csv`
- **Figuras:** `reportes/figuras_dia2/`
- **Descargas:** `reportes/manifiesto_crudos.csv` (49 archivos; los 49 SHA-256 se volvieron a verificar al final)

**Hoy no se entrenó ningún modelo ni se calculó ningún error de pronóstico.**

## 1. Resumen

1. **Los datos diarios empiezan el 1997-01-01**, que es también el primer día que acepta el formulario. Ese día solo hay precio (S/ 1,61 por kg en bolsa); el primer día con volumen es el **1997-01-02**. Una consulta de diciembre de 1996 devolvió «No existen resultados».
2. **La serie completa va de 1997-01-01 a 2026-11-13** (10 909 días de calendario):
   - 10 830 días aparecen en la tabla;
   - 10 629 tienen volumen de bolsa;
   - 9 792 tienen `vol_total` (bolsa y cajón);
   - 10 828 tienen precio.

   La serie mensual tiene 359 meses (enero de 1997 a noviembre de 2026); solo falta el volumen de septiembre de 1997.
3. **El día 2, «...» se trató como faltante, y la propia fuente lo hace en sus promedios.** Desde el ajuste del 2026-11-16, el supuesto principal es otro: «...» en una presentación = ingreso cero de esa presentación (§3.4). El SISAP nunca registra un volumen 0. Además, los PDF oficiales de gob.pe no calculan sus promedios de 7 días ni de 4 semanas cuando la ventana contiene un «...»: muestran «:». Esto ocurrió en los **25 de 25** casos en que la lectura «faltante» y la lectura «cero» dan resultados distintos.

   Matiz importante: el **mensual** del SISAP suma solo los días disponibles, y eso equivale en la práctica a contar «...» como cero. Qué pasó físicamente esos días (no entró limón o no se registró) **no está comprobado**.
4. **La ruptura de 2005-12 es solo de redondeo.** Hasta 2005-11 los volúmenes llevan decimales y desde 2005-12 son enteros. El escalón de nivel no es significativo: +5,0 %, IC 95 % −2,8 % a +13,3 %, p = 0,21. La dispersión relativa no aumenta (el coeficiente de variación diario pasa de 0,576 a 0,545).
5. **El mensual del SISAP coincide con la suma de los diarios** en los 357 meses comparables (diferencia máxima de 0,11 t).
6. **Efecto miércoles:** existe desde 1997 y en 2026 es más fuerte que nunca. La mediana del miércoles es del 0,5 % al 6 % de la de los otros días en casi todos los años; las excepciones son 1997–1999 y 2015–2018, con 8 % a 37 %.
7. **Correlación con SISAGRI** (2016–2025, sin meses incompletos): Pearson **r = 0,52** en niveles y **r = 0,27** en diferencias mensuales (Spearman 0,46 y 0,27). Es positiva y significativa, pero moderada: el ingreso al GMML no reproduce la estacionalidad marcada de la producción nacional (fig. 6).
8. **Hallazgo nuevo con consecuencias en la prueba:** el cajón del **2021-02-01** vale 1 235 t en el SISAP, pero el PDF oficial dice 245 t. Es un probable cero de más (Chiclayo 1 100 en lugar de 110). Se fija como anomalía documentada en el protocolo.
9. **Supuesto principal sobre «...»** (ajuste del 2026-11-16, §3.4): si una presentación tiene dato y la otra «...», la segunda vale 0; si las dos tienen «...», el día es faltante. Se apoya en que el SISAP nunca registra 0 y en que los «...» se concentran en los días de menor ingreso, con medianas parecidas dentro de cada día de la semana (§4.1). No se puede comprobar físicamente. El tratamiento anterior («...» = faltante) queda como sensibilidad.
10. **Días y meses evaluables en 2021–2025 con la regla nueva** (§3.4 y §6.1): D tiene **1 818 días** (antes 1 352) y M **59 meses** (antes 18; ninguno de 2025). Con las dos presentaciones en «...» hay 152 días en toda la historia y solo 7 en 2021–2025, de los cuales uno es feriado (Navidad de 2024).
11. **El mensual nuevo es la suma del SISAP** salvo por redondeo: diferencia máxima de 0,13 t en 357 meses (§6.1). Los faltantes del brazo M en 1997-10 a 2025-12 son 3 meses: 2012-10, 2016-01 y 2021-02 (anomalía).

## 2. Primer día con datos diarios (T1)

Se hizo una consulta por intervalo del 1 al 7 de marzo en cada año sondeado, para el GMML, *Limon sutil bolsa* y *Limon sutil cajon*:

| Año | Días en la tabla | Días con volumen | Días con precio |
|---|---|---|---|
| 1997 | 7 | 7 | 7 |
| 2000 | 7 | 7 | 7 |
| 2005 | 7 | 7 | 7 |
| 2010 | 7 | 7 | 7 |
| 2012 | 7 | 7 | 7 |
| 1996 (15–31 de diciembre) | «No existen resultados para los criterios elegidos.» | — | — |

Todos los años sondeados tienen datos diarios, así que no hizo falta bisección. La descarga completa por años confirma:

- **Primer día con precio:** 1997-01-01 (bolsa y cajón).
- **Primer día con volumen:** 1997-01-02 (bolsa 170,01 t; cajón 64,68 t). El 1997-01-01, feriado, tiene volumen «...».

## 3. Significado de «...»

### 3.1 Evidencia

1. **El SISAP nunca registra 0.** En 30 años de datos no hay ni un volumen 0, ni en el total ni por procedencia. El mínimo observado es 0,02 t. Las únicas celdas no numéricas son «...» (132 502 celdas en el formato largo).
2. **En el PDF de gob.pe, «...» aparece como «:».** Se descargaron los PDF del GMML de 6 fechas con «...» o su día siguiente:
   - 2024-11-20, 2024-12-26, 2025-01-07, 2025-10-15, 2025-10-16 y 2026-07-08;
   - el 2024-12-25 y el 2025-01-04 no tienen PDF publicado;
   - se suman los 3 PDF del día 1 (2021-02-01, 2023-12-29 y 2025-10-31).

   En los **11 de 11** casos en que «Ayer» u «Hoy» del PDF corresponden a un día con «...» en el SISAP, el PDF muestra «:». **El PDF tampoco muestra 0 en ninguna fila** de los PDF revisados.
3. **El PDF trata «:» como faltante al calcular promedios.** Los promedios del PDF se reproducen con el SISAP usando estas ventanas:
   - «Promedio últ. 7 días»: t − 7 … t − 1;
   - «Últ. 4 <día>»: t − 28, t − 21, t − 14, t − 7.

   Se comparó cada promedio con las dos lecturas de «...» (`reportes/dia2_control_puntos_suspensivos.csv`):

   | | Comparaciones | Coinciden con «faltante» | Coinciden con «cero» |
   |---|---|---|---|
   | Todas | 72 | 67 | 42 |
   | Discriminantes (las lecturas dan resultados distintos) | 25 | **25** | **0** |

   Cuando la ventana contiene un «...», el PDF **no** muestra el promedio con ceros: muestra «:». Por ejemplo, el 2025-10-16 en bolsa: el promedio con ceros sería 393,4 t y el PDF muestra «:».

   Las 5 comparaciones que no coinciden con ninguna lectura son todas del PDF del 2021-02-01 (§3.3).
4. **Pero el mensual del SISAP suma los días disponibles.** En los 357 meses comparables, la serie mensual es igual a la suma de los diarios sin contar los «...» (§6). En los totales mensuales, la fuente se comporta como si «...» fuera cero.

### 3.2 Conclusión

La fuente es **ambigua por construcción**. Para promedios, MIDAGRI trata «...» como «sin dato»; para totales, lo omite, que equivale a cero. Si físicamente no entró limón o no se registró **no se puede saber** con estos datos.

El día 2, siguiendo la regla acordada entonces, «...» se trató como **faltante**: `NaN` en `data/processed/sutil_diario.csv`, con las columnas `falta_*`. Es la lectura que usa la propia fuente cuando calcula estadísticas sobre la serie diaria. El 2026-11-16 se cambió el supuesto principal (§3.4); los valores originales se conservan en `vol_bolsa` y `vol_cajon`.

### 3.4 Supuesto principal desde el 2026-11-16: «...» en una presentación = ingreso cero

**Regla** (protocolo, §2.1):

- Si una presentación (bolsa o cajón) tiene dato y la otra tiene «...», la segunda vale **0** y `vol_total` es la observada (`estado_* = cero_supuesto`).
- Si las dos tienen «...», o la fecha no está en la tabla, el día es **faltante** (`estado_* = faltante`).
- Se aplica igual en toda la historia: en los años en que dominaba el cajón, bolsa «...» = 0.
- Columnas nuevas en `sutil_diario.csv`: `estado_bolsa`, `estado_cajon`, `vol_bolsa_regla`, `vol_cajon_regla`, `vol_total` (principal) y `vol_total_estricto` (tratamiento anterior). `vol_bolsa` y `vol_cajon` conservan los valores originales.

**Evidencia:** el SISAP nunca registra 0 (§3.1, punto 1); los «...» se concentran en miércoles y martes, los días de menor ingreso, y dentro de cada día de la semana las medianas de bolsa de los días con y sin cajón son parecidas (§4.1).

**Límite:** no se puede comprobar físicamente si esos días no entró limón o no se registró. Los PDF muestran «:», compatible con las dos lecturas.

**Días con las dos presentaciones en «...»** (`reportes/dia2_dias_ambas_faltantes.csv`):

- **152 días** en toda la historia (1997-01-01 a 2026-11-13). 79 son días fuera de la tabla y 73 están en la tabla con «...» en las dos presentaciones; de estos 73, 40 son miércoles.
- Por año: 31 en 1997, 26 en 2006, 22 en 2016, 15 en 2012, 12 en 2014, 11 en 2015 y menos de 8 en el resto. Ninguno entre 2018 y 2022.
- **Feriados:** se cruzaron con una lista fija de feriados nacionales de larga data (Año Nuevo, Jueves y Viernes Santo, 1 de mayo, 29 de junio, 28 y 29 de julio, 30 de agosto, 8 de octubre, 1 de noviembre, 8 y 25 de diciembre). La lista no incluye los feriados agregados desde 2022 ni los días no laborables decretados.
  - 14 caen en un feriado. 11 de ellos están fuera de la tabla, entre 2013-12 y 2016-01 (Navidad, Año Nuevo, Viernes Santo, 1 de mayo); los otros 3 son 1997-01-01, 2014-10-08 y 2024-12-25.
  - 8 caen el día antes o el día después de un feriado.
  - Los otros 130 no están junto a un feriado de la lista. **Los feriados explican solo una parte de estos días.**
- **En 2021–2025 hay 7:**

  | Fecha | Día | En la tabla | Feriado |
  |---|---|---|---|
  | 2023-03-12 | dom | sí | — |
  | 2023-07-15 | sáb | sí | — |
  | 2023-08-03 | jue | no | — |
  | 2023-08-04 | vie | no | — |
  | 2023-09-20 | mié | sí | — |
  | 2024-12-25 | mié | sí | Navidad |
  | 2025-10-15 | mié | sí | — |

**Días con una sola presentación en «...»** (cero supuesto): 965 en toda la historia, 466 en 2021–2025. En 2021–2025, 462 son de cajón y 4 de bolsa.

### 3.3 Discrepancia del 2021-02-01

| Fecha | SISAP bolsa | PDF bolsa | SISAP cajón | PDF cajón |
|---|---|---|---|---|
| 2021-01-31 («Ayer») | 115 | 114 | 59 | 58 |
| 2021-02-01 («Hoy») | 1 116 | 1 115 | **1 235** | **245** |

- Las diferencias de 1 t parecen revisiones o redondeos posteriores (no comprobado).
- La del cajón es otra cosa. El SISAP desglosa el cajón de ese día en **Chiclayo 1 100 t** y **Sullana 135 t**. Como 245 = **110** + 135, lo más probable es un cero de más en el SISAP. No se comprobó con una tercera fuente.
- Es el mayor volumen de cajón de 2020–2021: el siguiente es de 891 t y el percentil 99 de esos años es de 538 t.
- Está registrado en `config/fuentes.yaml` (`anomalias_documentadas`) y marcado en la columna `anomalia_documentada` de `sutil_diario.csv`. **No se modificó el valor publicado.**
- El protocolo lo trata como faltante en el análisis principal y con el valor publicado en una sensibilidad. Esta regla se fijó antes de calcular cualquier error.

**Cribado de atípicos** (solo descriptivo, `reportes/dia2_candidatos_atipicos.csv`):

- Criterio: valores mayores que 4 veces la mediana del mismo día de la semana en las ±4 semanas, y mayores que 100 t.
- Resultado: 94 candidatos, 56 de ellos en 2021–2025.
- 17 son miércoles con volumen normal, casi todos junto a feriados (p. ej., 2025-12-24 y 2025-12-31): el día de poco ingreso se desplaza.
- La mayoría de los demás son picos del cajón, que desde 2021 es pequeño e intermitente.
- Solo el 2021-02-01 tiene evidencia documental de error. **No se corrige ningún otro valor.**

## 4. Cobertura y huecos por año

Fuente: `reportes/dia2_huecos_por_anio.csv`. Columnas:

- **Fuera de tabla:** la fecha no aparece en la respuesta del SISAP.
- **Sin bolsa, sin cajón, sin total:** días sin volumen de cada tipo; el total falta si falta bolsa o cajón.
- **Sin total (mié):** cuántos de esos días son miércoles.
- **% cajón:** participación del cajón en el volumen del año.

| Año | Días | Fuera de tabla | Sin bolsa | Sin cajón | Sin total | Sin total (mié) | Sin precio | % cajón |
|---|---|---|---|---|---|---|---|---|
| 1997 | 365 | 0 | 35 | 35 | 39 | 13 | 0 | 59,3 |
| 1998 | 365 | 0 | 5 | 4 | 9 | 4 | 0 | 65,2 |
| 1999 | 365 | 0 | 19 | 22 | 34 | 32 | 1 | 58,7 |
| 2000 | 366 | 0 | 5 | 0 | 5 | 5 | 0 | 54,4 |
| 2001 | 365 | 0 | 3 | 0 | 3 | 1 | 0 | 49,7 |
| 2002 | 365 | 0 | 3 | 1 | 4 | 4 | 0 | 52,3 |
| 2003 | 365 | 0 | 1 | 1 | 2 | 1 | 0 | 50,5 |
| 2004 | 366 | 0 | 0 | 0 | 0 | 0 | 0 | 43,4 |
| 2005 | 365 | 3 | 12 | 4 | 13 | 4 | 4 | 56,8 |
| 2006 | 365 | 26 | 36 | 30 | 40 | 17 | 26 | 45,3 |
| 2007 | 365 | 4 | 8 | 9 | 11 | 7 | 4 | 45,0 |
| 2008 | 366 | 0 | 14 | 3 | 16 | 15 | 0 | 38,5 |
| 2009 | 365 | 0 | 6 | 2 | 8 | 8 | 0 | 37,9 |
| 2010 | 365 | 0 | 4 | 10 | 13 | 13 | 0 | 45,4 |
| 2011 | 365 | 0 | 4 | 18 | 22 | 21 | 0 | 52,9 |
| 2012 | 366 | 7 | 25 | 27 | 37 | 25 | 7 | 50,4 |
| 2013 | 365 | 1 | 13 | 27 | 35 | 32 | 1 | 37,4 |
| 2014 | 365 | 4 | 21 | 46 | 55 | 43 | 4 | 49,6 |
| 2015 | 365 | 5 | 11 | 38 | 38 | 33 | 5 | 47,5 |
| 2016 | 366 | 22 | 24 | 48 | 50 | 25 | 22 | 44,8 |
| 2017 | 365 | 5 | 5 | 29 | 29 | 22 | 5 | 39,3 |
| 2018 | 365 | 0 | 0 | 31 | 31 | 25 | 0 | 38,9 |
| 2019 | 365 | 0 | 6 | 32 | 38 | 34 | 0 | 40,6 |
| 2020 | 366 | 0 | 8 | 24 | 32 | 13 | 0 | 31,3 |
| **2021** | 365 | 0 | 1 | 65 | 66 | 39 | 0 | 17,9 |
| **2022** | 365 | 0 | 2 | 95 | 97 | 44 | 0 | 13,4 |
| **2023** | 365 | 2 | 5 | 81 | 81 | 36 | 2 | 13,7 |
| **2024** | 366 | 0 | 2 | 95 | 96 | 38 | 0 | 12,6 |
| **2025** | 365 | 0 | 1 | 133 | 133 | 44 | 0 | 9,0 |
| 2026* | 317 | 0 | 1 | 79 | 80 | 38 | 0 | 13,9 |

\* Hasta el 2026-11-13. En negrita, los años de prueba.

**Rachas de 3 o más días fuera de la tabla:**

- 2012-10-27 a 2012-10-31 (5 días);
- **2016-01-11 a 2016-01-31 (21 días)**;
- 2017-03-17 a 2017-03-21 (5 días).

En total hay 79 días fuera de la tabla. Las causas no se comprobaron.

**Meses incompletos:** más de 2 días sin volumen de bolsa que no sean miércoles. Son 1997-09, 2005-11, 2006-01, 2006-06, 2006-10, 2006-11, 2006-12, 2012-09, 2012-10, 2016-01 y 2017-03. **Ninguno cae en 2021–2025.**

Como el mensual del SISAP suma solo los días disponibles, en esos meses está subestimado. Por ejemplo, en enero de 2016 el índice de la fig. 6 baja a 33.

Desde el ajuste del 2026-11-16, `mes_incompleto` es solo descriptivo: los faltantes del brazo M se definen con `flag_faltante` (§6.1).

Las tablas de §4 a §7 usan la definición del día 2: «...» = faltante (`vol_total_estricto` en el diario, suma del SISAP en el mensual). Sus cifras no cambian con el ajuste del 2026-11-16. La columna `faltan_ambas_presentaciones` de `dia2_huecos_por_anio.csv` da los días faltantes con la regla nueva.

**Días y meses evaluables por año de prueba** (`reportes/dia2_evaluables_por_anio.csv`):

| Año | D principal (regla nueva) | D con cajón = 0 supuesto | D, «...» = faltante (sens. 1) | D, bolsa sola (sens. 2) | M principal | M, sens. 1 |
|---|---|---|---|---|---|---|
| 2021 | 364 | 65 | 298 | 364 | 11 | 7 |
| 2022 | 365 | 95 | 268 | 363 | 12 | 2 |
| 2023 | 360 | 76 | 284 | 360 | 12 | 5 |
| 2024 | 365 | 94 | 270 | 364 | 12 | 4 |
| 2025 | 364 | 132 | 232 | 364 | 12 | 0 |
| **Total** | **1 818** | 462 | 1 352 | 1 815 | **59** | 18 |

- D principal excluye el 2021-02-01 (anomalía) y los 7 días con las dos presentaciones en «...» (§3.4).
- M principal excluye 2021-02, que contiene la anomalía.
- Además hay 4 días con bolsa = 0 supuesto (cajón observado y bolsa en «...»): 1 en 2021, 2 en 2022 y 1 en 2024.

### 4.1 Días excluidos frente a evaluados en 2021–2025 (brazo D)

Ajuste del 2026-11-16. Fuente: `reportes/dia2_excluidos_vs_evaluados.csv` y `dia2_diagnostico.json` (`excluidos_vs_evaluados`). Es descriptivo: no usa ningún modelo.

- **Evaluados:** `vol_total` observado y sin la anomalía documentada (1 352 días).
- **Excluidos:** el resto (474 días; 463 tienen volumen de bolsa). Por año: 67, 97, 81, 96 y 133 (2021 a 2025; 2021 incluye el 2021-02-01).

**Día de la semana** (% de cada grupo):

| | Lun | Mar | Mié | Jue | Vie | Sáb | Dom |
|---|---|---|---|---|---|---|---|
| Evaluados | 17,9 | 12,0 | **4,4** | 16,9 | 15,9 | 17,8 | 15,1 |
| Excluidos | 4,0 | 20,9 | **42,4** | 6,5 | 9,7 | 4,4 | 12,0 |

**Mes** (% de cada grupo): los evaluados van de 7,8 % a 9,1 % por mes y los excluidos de 6,8 % (octubre) a 10,1 % (marzo). No hay un mes con concentración marcada de exclusiones.

**`vol_bolsa`** (t/día):

| | n | Media | P10 | P25 | Mediana | P75 | P90 |
|---|---|---|---|---|---|---|---|
| Evaluados | 1 352 | 478,9 | 71 | 184 | **488** | 711 | 884 |
| Excluidos | 463 | 206,1 | 3 | 5 | **61** | 264 | 705 |

**Mann–Whitney bilateral sobre `vol_bolsa`:** U = 142 986,5, **p = 2,6 × 10⁻⁶⁸**. La mediana de los excluidos es mucho menor.

**Aplicación de la regla:** con p < 0,05, la regla escrita esa mañana hacía de D con `vol_bolsa` la sensibilidad principal. Esa regla se pensó para el objetivo estricto y quedó sustituida por el supuesto de §3.4, con el que solo se excluyen 8 días de prueba (protocolo, §2.3). La bolsa sola sigue como sensibilidad 2. Esta comparación queda como evidencia del supuesto.

**Lectura** (no cambia la regla):

- La diferencia viene sobre todo de la composición por día de la semana: los excluidos son en su mayoría miércoles y martes, los días de menor ingreso.
- Dentro de cada día de la semana, las medianas de bolsa de los dos grupos son parecidas (p. ej., lunes 798 t en evaluados frente a 886 t en excluidos; miércoles 5,5 frente a 4 t).
- Como complemento, sin los miércoles: medianas de 502 t (evaluados) y 222 t (excluidos), Mann–Whitney p = 1,4 × 10⁻¹⁴. Sigue habiendo diferencia porque también pesan los martes y domingos.
- Consecuencia para la sensibilidad 1: el MAE de `vol_total_estricto` se calcula sobre un conjunto con pocos días de bajo ingreso, así que su nivel no es comparable con el del objetivo principal.

## 5. Ruptura de 2005/2006

- **Cuándo:** el último mes con decimales es **2005-11** y desde **2005-12** todos los volúmenes son enteros. Entre 1997 y 2004 tienen decimales el 98–99 % de los valores diarios, y en 2005 el 91 %.
- **Nivel:** se ajustó log(`vol_total` mensual) con constante, 11 dummies de mes, tendencia lineal y un escalón desde 2005-12, para 2000-01 a 2009-12 (120 meses), con errores HAC de Newey-West (12 rezagos). El escalón vale **+5,0 %** (coeficiente 0,049; IC 95 % −0,028 a 0,125; **p = 0,21**). **No hay un salto de nivel significativo.**
- **Varianza** (3 años antes frente a 3 años después, diario):

  | Periodo | Media (t/día) | Desv. típica (t) | Coef. de variación | Desv. típica de log |
  |---|---|---|---|---|
  | 2002-12 a 2005-11 | 223,3 | 128,7 | 0,576 | 1,279 |
  | 2005-12 a 2008-11 | 284,2 | 154,9 | 0,545 | 1,167 |

  - La prueba de Brown-Forsythe detecta varianzas absolutas distintas (p = 2,7 × 10⁻⁷), pero lo explica el crecimiento del nivel: la dispersión **relativa** (coeficiente de variación y desviación del log) es algo **menor** después.
  - La desviación típica de la variación mensual del log pasa de 0,116 a 0,131.
- **Magnitud del redondeo:** la parte decimal media antes del cambio es 0,50 t, frente a una media diaria de ~223 t (≈ 0,2 %).
- **Conclusión:** solo cambia el redondeo; no hay salto de nivel. Coincide con otras irregularidades: 26 días fuera de la tabla en 2006 y 6 meses incompletos entre 2005-11 y 2006-12. Por eso el protocolo mantiene la historia desde 1997 y preregistra la sensibilidad «historia desde 2006-01-01».

## 6. Consistencia diario frente a mensual

Fuente: `reportes/dia2_consistencia_mensual.csv`.

| Presentación | Meses comparables | Iguales (tolerancia 0,5 t) | Diferencia máxima |
|---|---|---|---|
| Bolsa | 357 | **357** | 0,11 t |
| Cajón | 357 | **357** | 0,05 t |

- Los meses comparables excluyen 1997-09 (sin volumen) y 2026-11 (incompleto).
- El total diario es igual a la suma por procedencias en 20 534 de 20 549 casos; en el resto la diferencia es de 0,02 t como máximo.
- El precio mensual de la bolsa es la media de los precios diarios: diferencia absoluta mediana de 0,007 S/ por kg y máxima de 0,22 S/ por kg.
- **El mensual del SISAP es la agregación del diario, no una fuente independiente.**

### 6.1 Objetivo mensual del brazo M frente a la suma del SISAP

Ajuste del 2026-11-16. Fuente: `reportes/dia2_mensual_vs_sisap.csv` (por mes) y `dia2_diagnostico.json` (`mensual_vs_sisap`). Es solo descriptivo.

**Objetivo principal** (`vol_total` en `sutil_mensual.csv`): suma de `vol_total` diario con la regla de §3.4. Los días faltantes no suman. El mes es faltante (`flag_faltante = 1`) si más del 20 % de sus días son faltantes (`dias_faltantes`) o si contiene la anomalía documentada (`mes_con_anomalia`). La suma publicada por el SISAP pasa a llamarse `vol_total_sisap`.

**Meses faltantes** en el rango del brazo M (1997-10 a 2025-12, 339 meses): **3**.

| Mes | Días faltantes | Motivo |
|---|---|---|
| 2012-10 | 8 de 31 | ambas presentaciones en «...» o fuera de la tabla |
| 2016-01 | 22 de 31 | ídem (racha fuera de la tabla del 11 al 31) |
| 2021-02 | 0 de 28 | contiene la anomalía documentada (cajón del 2021-02-01) |

- Por año de prueba: 1 faltante en 2021 (2021-02) y ninguno en 2022–2025. **Meses evaluables: 11, 12, 12, 12 y 12 (59).**
- Hay 7 días faltantes repartidos en 6 meses de prueba no faltantes (2023-03, 2023-07, 2023-08 con 2 días, 2023-09, 2024-12 y 2025-10). Esos días suman 0, igual que en el SISAP, así que esos meses quedan algo subestimados. El protocolo lo declara (§2.1).

**Diferencia con la suma del SISAP** (`vol_total` − `vol_total_sisap`; 357 meses comparables, 1997-01 a 2026-10):

- Diferencia absoluta máxima: **0,13 t**. Ningún mes difiere más de 0,5 t ni más del 5 %.
- Era lo esperable: con la regla nueva, la suma diaria es bolsa observada + cajón observado, que es lo que suma el SISAP (§6). La diferencia es solo de redondeo.
- 2021-02 también coincide (16 990 t), porque las dos sumas incluyen el valor anómalo de 1 235 t. Por eso ese mes es faltante en el análisis principal.

**Sensibilidad 1, tratamiento anterior** (columnas `vol_total_escalado`, `dias_observados_estricto` y `flag_faltante_estricto`): promedio diario de `vol_total_estricto` en los días observados × días del mes; faltante con menos del 80 % de días observados.

- 47 meses faltantes en 1997-10 a 2025-12. **42 de los 60 meses de prueba** son faltantes, incluidos los 12 de 2025; quedan 18 evaluables (7, 2, 5, 4 y 0).
- Frente a la suma del SISAP: mediana +6,8 % (percentiles 10 y 90: 0,0 % y +21,9 %); 202 de 357 meses difieren más del 5 %. En 2021–2025, mediana +20,4 % (rango −1,2 % a +45,6 %) y 57 de 60 meses por encima del 5 %. La sobreestimación viene de que los días sin `vol_total_estricto` son sobre todo martes y miércoles, de poco ingreso.

## 7. Diagnóstico descriptivo (sin modelos)

### 7.1 Patrón semanal

Figuras 2 y 3. Tablas: `reportes/dia2_semanal_mediana.csv` y `dia2_diagnostico.json`.

Mediana de `vol_total` (t/día) por día de la semana:

| Periodo | Lun | Mar | **Mié** | Jue | Vie | Sáb | Dom |
|---|---|---|---|---|---|---|---|
| 1997–2005 | 345 | 103 | **15** | 270 | 242 | 281 | 174 |
| 2006–2015 | 496 | 133 | **11** | 425 | 377 | 414 | 261 |
| 2016–2026 | 869 | 147 | **36** | 706 | 531 | 699 | 312 |

- **El efecto miércoles está presente en los 30 años.** Razón entre la mediana del miércoles y la de los otros días (bolsa):
  - 1997–1999: 0,13 a 0,37;
  - 2000–2014: 0,01 a 0,06;
  - **2015–2018: sube a 0,08–0,22** (no se investigó por qué);
  - desde 2019: 0,005 a 0,014, el valor más bajo de la serie.
- El martes también es un día bajo (17–30 % de la mediana del lunes según el periodo).
- **Sesgo a tener en cuenta:** las medianas de `vol_total` de los miércoles están sobreestimadas, porque los miércoles sin cajón no entran en `vol_total`. Por eso la razón se calcula con la bolsa.
- Implicación: la estacionalidad semanal domina la varianza diaria. El Seasonal Naive (t − 7) será una referencia difícil de batir a h = 1.

### 7.2 Patrón anual y tendencia

Figuras 1, 4 y 5.

- **Estacionalidad** (`vol_total` mensual sobre la media del año, mediana):
  - máximo de diciembre a marzo (índice 1,05–1,27);
  - mínimo de junio a septiembre (0,75–0,91);
  - el perfil es estable en los tres periodos; entre 1998 y 2005 el mínimo de septiembre era más profundo (0,75 frente a 0,88–0,89).
- **Tendencia:** el volumen anual pasó de 63 757 t (1998) a 182 830 t (2025), un crecimiento log-lineal de **≈ 4,3 % al año**. Hubo un máximo en 2020 (188 349 t) y un descenso en 2022–2023 (160 203 y 152 422 t).
- **Composición:** la bolsa reemplaza al cajón. El cajón era el 65 % del volumen en 1998 y es el 9 % en 2025.
- Esta información queda para interpretar los resultados; no cambia el protocolo.

### 7.3 Puente con el artículo: correlación con SISAGRI 2016–2025

Fuente: `reportes/dia2_correlacion_sisagri.csv`; fig. 6.

- SISAGRI: producción nacional mensual de limón sutil, tomada de `limon_sutil_nacional_mensual.csv` del repositorio del artículo (solo lectura; SHA-256 `9e8e06c6…6f970e31`).
- GMML: `vol_total` mensual del SISAP.
- Muestra principal: sin los meses incompletos 2016-01 y 2017-03.
- Rezago 0 = mismo mes.

| Transformación | n | Pearson r (p) | Spearman ρ (p) |
|---|---|---|---|
| Niveles | 118 | **0,520** (1,6 × 10⁻⁹) | 0,459 (1,7 × 10⁻⁷) |
| Diferencia mensual | 116 | **0,266** (0,004) | 0,268 (0,004) |
| Variación % mensual | 116 | 0,376 (3 × 10⁻⁵) | 0,283 (0,002) |

- **Con rezago** (producción de t − 1 frente al GMML de t), en niveles: r = 0,325 (p = 0,0003). Con t − 2 deja de ser significativa. En diferencias, el rezago 1 no es significativo (r = 0,053).
- **Con todos los meses** (120): niveles r = 0,505; diferencias r = 0,294.
- **Lectura:**
  - El ingreso de limón al GMML y la producción nacional se mueven juntos en el mismo mes, pero solo en parte.
  - La producción nacional oscila mucho más: índice entre ~20 y ~175 en la fig. 6, frente a ~45–130 del GMML sin contar 2016-01.
  - El GMML tiene una tendencia propia.
  - **Por lo tanto, la serie del GMML no es un sustituto de la serie del artículo.** El brazo M responde a la pregunta «modelo o datos» sobre una serie mensual distinta, aunque relacionada.

## 8. Cambios respecto al borrador del protocolo

| Tema | Borrador (día 1) | Protocolo final (día 2) | Motivo |
|---|---|---|---|
| Brazos | Solo diario (y semanal como alternativa) | **Dos brazos:** D (diario) y M (mensual desde 1997) | Decisión de YesaR (separar modelo y datos con la frecuencia del artículo) |
| Inicio de la serie | 2013 o 2016, pendiente de comprobar | **1997-10-01** | Hay datos diarios desde 1997-01-01. Septiembre de 1997 no tiene volumen |
| «...» | Propuesta: 0 | **Una sola presentación en «...» = 0; las dos en «...» = faltante** (ajuste del 2026-11-16). El tratamiento como faltante, regla del día 2, queda como sensibilidad 1 | §3.4 y §4.1. Decisión de YesaR |
| Imputación | No definida | Brazo D: **t − 7**, solo pasado. Brazo M: suma de los diarios imputados | Estacionalidad semanal (§7.1). Coherencia entre brazos |
| Objetivo mensual (ajuste del 2026-11-16) | Suma del SISAP | **Suma de los diarios con la regla de «...»**; faltante si más del 20 % de los días son faltantes o si el mes contiene la anomalía. El promedio observado × días del mes queda como sensibilidad 1 | §6.1. Decisión de YesaR |
| Días con «...» (ajuste del 2026-11-16) | — | Comparación de los días con «...» frente a los días observados, como evidencia del supuesto. La regla «bolsa sola = sensibilidad principal» queda sustituida; la bolsa sola sigue como sensibilidad 2 | §4.1. Decisión de YesaR |
| Anomalías | — | El cajón del 2021-02-01 como faltante y una sensibilidad con el valor publicado | §3.3 |
| Años de prueba | 2024–2025 (validación 2023) | **2021–2025**; entrenamiento hasta Y − 2, validación Y − 1 | Igual que el artículo |
| Tamaños N | D: 120 / 360 / 1 000 / 3 000 / máx | D: 120 / 360 / 1 000 / máx; M: 60 / 120 / 240 / máx | Decisión de YesaR |
| Horizontes | D: h = 1 y 7 | D: h = 1 (principal) y 7; M: h = 1 | Decisión de YesaR |
| Modelos | Seasonal Naive semanal y anual; SARIMA o TBATS por validación | Seasonal Naive semanal (D) y anual (M); ETS por AICc (como el artículo); Theta θ = 2; SARIMA con orden fijo (D: (1,0,1)(0,1,1)₇ + Fourier; M: el congelado del artículo); Chronos; LSTM; DualLSTM | Sin búsqueda de hiperparámetros, como el artículo |
| DualLSTM | Ventana y presupuesto por validación | **Arquitectura e hiperparámetros congelados** del artículo; rama B = calendario + precio rezagado; 10 semillas | Decisión de YesaR |
| Métrica principal | MASE | **MAE** (MASE y RMSE secundarias) | Igual que el artículo |
| Pruebas | DM con Holm | Bootstrap pareado B = 10 000, semilla 42 (principal) y DM-HLN; «consistently top-ranked» con 3 de 4 umbrales | Igual que el artículo |
| Regla de decisión | Cociente de MASE monótono | Δ MAE monótono **y** no significativo o de signo contrario en máx → «datos»; significativo en máx → «modelo»; otro patrón → «no concluyente» | Decisión de YesaR, más el subcaso «no se reproduce la desventaja» |
| Transformación | `log1p` | Sin transformación general; SARIMA en D sobre log; redes con StandardScaler como el artículo | Igual que el artículo |

**Decisiones tomadas el 2026-11-16:**

- supuesto principal: «...» en una sola presentación = 0 (§3.4);
- brazo D: objetivo `vol_total` con esa regla;
- brazo M: suma de los diarios, faltante con más del 20 % de días faltantes;
- sensibilidades: el tratamiento anterior («...» = faltante en los dos brazos) es la 1 y la bolsa sola es la 2.

**Decisión mía, a revisar antes de congelar:** el cajón anómalo del 2021-02-01 no es «...», así que no se le aplica el cero. Se trata como valor desconocido: el día es faltante en D y el mes 2021-02 es faltante en M (protocolo, §2.2). Contar solo los días con las dos presentaciones en «...» habría dejado 2021-02 con un total que incluye las 1 235 t anómalas.

## 9. Lo que no se pudo comprobar

1. **Qué ocurrió físicamente los días con «...».** No se sabe si no entró limón o si no se registró (§3.2).
2. **Cuál de los dos valores del 2021-02-01 es el correcto**, más allá de la aritmética (§3.3). Tampoco el origen de las diferencias de 1 t de enero y febrero de 2021.
3. **Las causas de los días fuera de la tabla** (p. ej., 2016-01-11 a 2016-01-31), del aumento del volumen de los miércoles en 2015–2018 y del cambio de redondeo de 2005-12.
4. **La longitud máxima de contexto de Chronos-Bolt-tiny.** El protocolo prevé que trunque y lo registre.
5. **Que la serie no se revise en el futuro.** Las revisiones de 1 t de 2021 indican que el SISAP puede cambiar valores pasados. Por eso el experimento usa los archivos con hash, no una nueva descarga.
6. **Los PDF de gob.pe anteriores a noviembre de 2024**, salvo los 3 del día 1. robots.txt prohíbe paginar la colección.
7. **La licencia de los datos.** No se encontró una licencia de reutilización de los datos del SISAP. Por eso `data/processed/*.csv` no se publica (está en `.gitignore`); se publican los scripts, el manifiesto y los hashes (ver §10 y el README).

**Peticiones de hoy:** unas 60 al SISAP y a gob.pe, con 1,5 s o más entre peticiones, el User-Agent del proyecto y sin bloqueos ni errores HTTP. Una nueva ejecución de `02_descarga_sisap.py` con la caché completa hace 2 peticiones.

## 10. Archivos para `git add` y mensajes de commit sugeridos

**Ajuste técnico previo.** Esta máquina tiene `core.autocrlf=true`. Para que los SHA-256 no dependan del sistema operativo:

- se añadió `.gitattributes` (`* text=auto eol=lf`);
- se normalizaron a LF los archivos del repositorio;
- los hashes se calculan sobre el contenido con LF, como el «método 2» del artículo.

`python scripts/04_hashes_protocolo.py --verificar` debe dar «OK» después del checkout.

**`data/processed/*.csv` no se versiona** (ajuste del 2026-11-16): está en `.gitignore` porque no se encontró una licencia que permita redistribuir los datos del SISAP. Se publican los scripts, `reportes/manifiesto_crudos.csv` y `docs/hashes_protocolo.json`, con los SHA-256 de las series; el README explica cómo regenerarlas y comprobarlas. Sí se versiona `data/processed/.gitkeep`.

### Commit 1: datos y scripts

```bash
git add .gitattributes .gitignore README.md requirements.txt config/fuentes.yaml \
  scripts/sisap_comun.py scripts/02_descarga_sisap.py scripts/03_series.py scripts/04_hashes_protocolo.py \
  scripts/01_reconocimiento_sisap.py \
  reportes/INFORME_DIA2.md reportes/manifiesto_crudos.csv reportes/dia2_sondeo.json reportes/dia2_diagnostico.json \
  reportes/dia2_huecos_por_anio.csv reportes/dia2_control_puntos_suspensivos.csv reportes/dia2_consistencia_mensual.csv \
  reportes/dia2_semanal_mediana.csv reportes/dia2_correlacion_sisagri.csv reportes/dia2_candidatos_atipicos.csv \
  reportes/dia2_excluidos_vs_evaluados.csv reportes/dia2_mensual_vs_sisap.csv \
  reportes/dia2_dias_ambas_faltantes.csv reportes/dia2_evaluables_por_anio.csv \
  reportes/figuras_dia2/ \
  reportes/INFORME_DIA1.md reportes/dia1_resumen.json reportes/dia1_manifiesto_sha256.csv \
  docs/PROTOCOLO_BORRADOR.md data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep
```

Al 2026-11-14 la rama `main` **no tiene ningún commit**: el commit del día 1 no se hizo. Por eso la lista incluye también los archivos del día 1. Si YesaR prefiere separarlo, puede hacer antes el commit del día 1 con la lista y el mensaje de `reportes/INFORME_DIA1.md`, §5; esos archivos ya tienen los cambios del día 2 (README, `fuentes.yaml` y finales de línea LF).

```
Día 2: serie diaria completa del SISAP (1997–2026), calidad de datos y diagnóstico

- Descarga reanudable con manifiesto SHA-256 (49 archivos crudos, no versionados).
- Series procesadas de limón sutil en el GMML: diaria (1997-01-01 a 2026-11-13),
  mensual y por procedencia; '...' tratado como faltante con evidencia de los
  PDF de gob.pe; anomalía documentada en el cajón del 2021-02-01.
- Diagnóstico sin modelos: huecos por año, ruptura de redondeo 2005-12,
  consistencia diario-mensual, patrón semanal y anual, correlación con SISAGRI.
- Supuesto principal: '...' en una sola presentación = ingreso cero; las dos
  en '...' = día faltante. Valores originales y estado por presentación en el
  diario; el tratamiento anterior ('...' = faltante) queda como sensibilidad.
- Mensual del brazo M: suma de los diarios con esa regla; faltante si más del
  20 % de los días son faltantes. Comparado con la suma del SISAP.
- Días con '...' frente a días observados (Mann–Whitney sobre la bolsa), días
  con las dos presentaciones en '...' y feriados, evaluables por año.
- data/processed/*.csv fuera del repositorio (sin licencia de redistribución);
  se publican los scripts, el manifiesto y los hashes.
- .gitattributes con LF para que los hashes sean estables.
```

### Commit 2: solo el protocolo (congelación)

```bash
python scripts/04_hashes_protocolo.py --verificar   # debe decir OK
git add docs/PROTOCOLO.md docs/hashes_protocolo.json
```

```
Protocolo del Experimento 2 congelado antes de ver resultados

docs/PROTOCOLO.md con SHA-256 registrado en docs/hashes_protocolo.json
junto con los de las series procesadas. A partir de este commit los
cambios se registran en docs/DESVIACIONES.md.
```

Después de este commit, YesaR puede ponerle una etiqueta (p. ej., `protocolo-v1`). Yo no hice commit, push, tag ni cambio de rama.
