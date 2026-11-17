# Informe del día 1: reconocimiento de datos diarios de limón (SISAP / MIDAGRI)

- **Fecha de consulta de todas las fuentes:** 2026-11-14
- **Código:** `scripts/01_reconocimiento_sisap.py` (parámetros en `config/fuentes.yaml`)
- **Cifras de respaldo:** `reportes/dia1_resumen.json`
- **Descargas (URL, parámetros POST, SHA-256):** `reportes/dia1_manifiesto_sha256.csv`

Hoy no se entrenó ningún modelo.

## Resumen

1. **El SISAP v2.0 (módulo de Mercados Mayoristas) es la fuente principal.** Un formulario POST automatizable devuelve una tabla HTML con datos **diarios** de *Limon Sutil Bolsa* y *Limon Sutil Cajon* en el GMML: precio máximo, promedio y mínimo en S/ por kg, y volumen en t, total y por procedencia. La consulta mensual cubre **de enero de 1997 a noviembre de 2026** (359 meses). En toda la muestra de 2016 a 2026 el precio está presente todos los días.
2. **Un año completo de datos diarios sale en una sola petición** (2019: 365 filas, 290 KB). Descargar toda la serie tomaría unas 30 peticiones: con la pausa de 1,5 s, son pocos minutos (no se midió el tiempo de respuesta del servidor).
3. **Los PDF diarios de gob.pe coinciden con el SISAP al dígito.** Se compararon volúmenes y precios de 6 días. Sirven como validación, pero no como fuente principal: hay un PDF por día hábil (no todos los días), traen precios por unidad de medida (bolsa de 45 kg, cajón de 23 kg) y la paginación de gob.pe está prohibida por robots.txt.
4. **No hay limón dulce en el SISAP del GMML.** Las variedades son *Limon sutil bolsa*, *Limon sutil cajon* y *Limon tahití*; esta última aparece en el catálogo pero no tiene datos entre 1997 y 2026. El limón dulce aparece solo en los PDF del Mercado Mayorista de Frutas N.° 2 (*Limon Dulce - Selva*). Para el MM N.° 2, el SISAP respondió «No existen resultados» a las consultas de limón.
5. **Hallazgos con consecuencias para el modelado:**
   - Los **miércoles** el volumen cae a casi cero (mediana de 5 t en 2019, frente a 276–416 t los demás días salvo el martes, con 130 t).
   - Hay un **traslado del cajón a la bolsa** (cajón: 71 960 t en 2019 y 16 486 t en 2025).
   - Hay una posible **ruptura de registro hacia 2005–2006** (volúmenes mensuales con decimales hasta 2005 y enteros desde 2006).

## 1. Tabla de fuentes

| # | Fuente | URL exacta | Formato | Cobertura comprobada | ¿Limón sutil / dulce? | ¿Automatizable? |
|---|---|---|---|---|---|---|
| 1 | **SISAP v2.0, Mercados Mayoristas** (MIDAGRI) | Portal: `http://sistemas.midagri.gob.pe/sisap/portal/` → módulo: `http://sistemas.midagri.gob.pe/sisap/portal2/mayorista/`; endpoint de consulta: `POST …/mayorista/resumenes/filtrar` | Tabla HTML en pantalla (ISO-8859-1). Hay un botón «Exportar a Excel» que reenvía la misma tabla (no probado) | Mensual: ENE-1997 a NOV-2026 (GMML). Diario: comprobado en 2016-03, todo 2019, 2023-12/2024-01, 2025-09/10 y 2026-10-15 a 2026-11-14 | Sutil: **sí** (bolsa y cajón por separado). Tahití: en catálogo, sin datos. Dulce: **no** | **Sí.** POST con sesión (cookie `autentificator`), sin login ni captcha. Parámetros en `config/fuentes.yaml` |
| 2 | Reporte de Ingreso y Precios en el GMML (gob.pe) | Colección: `https://www.gob.pe/institucion/midagri/colecciones/335-reporte-de-ingreso-y-precios-en-el-gran-mercado-mayorista-de-lima`; PDF en `https://cdn.www.gob.pe/uploads/document/file/<id>/…pdf` | PDF de 2 páginas, con texto extraíble, uno por día publicado; páginas mensuales en HTML | La colección declara «86 resultados» (páginas mensuales). En la primera hoja hay 25, de noviembre de 2024 a noviembre de 2026. La más antigua vista es febrero de 2021 (descargada); una búsqueda web mostró diciembre de 2020 | Sutil: **sí** (bolsa y cajón). Dulce: no | **Parcial.** Los PDF se descargan sin problema, pero la paginación (`?sheet=`) está **prohibida por robots.txt**, así que solo se accede a las 25 páginas mensuales más recientes. Hay subcolecciones por año (p. ej., `…/colecciones/94875-…-2025`) |
| 3 | Reporte de Ingreso y Precios en el Mercado Mayorista de Frutas N.° 2 (gob.pe) | `https://www.gob.pe/institucion/midagri/colecciones/14-reporte-de-ingreso-y-precios-en-el-mercado-mayorista-de-frutas-n-2` | PDF «BOLETÍN DIARIO» de 3 páginas | Declara «84 resultados». La primera hoja va de noviembre de 2024 a noviembre de 2026 (el último PDF es del 13/11/2026) | Sutil: no (en el PDF muestreado). Dulce: **sí** (*Limon Dulce - Selva*) | Igual que la fuente 2 |
| 4 | SISAP, módulo de Mercados Mayoristas, para el MM N.° 2 (código `15011502`) | Igual que la fuente 1 | — | El catálogo del MM N.° 2 incluye el género *Limon* (`0611`), pero tanto la consulta mensual de 1997 a 2026 como la diaria por género (01–13/11/2026) devolvieron «No existen resultados para los criterios elegidos.» | — | — |
| 5 | Boletín de Abastecimiento y Precios Mayoristas GMML y MM N.° 2 (gob.pe) | Ejemplo hallado en una búsqueda web: `https://cdn.www.gob.pe/uploads/document/file/9289092/7585650-boletin-de-abastecimiento-y-precios-mayoristas-gmml-y-mm-n-2-14-01-2026.pdf` | PDF | No comprobada (no se descargó) | No comprobado | No evaluado |
| 6 | SIEA (MIDAGRI) | `https://siea.midagri.gob.pe/portal/precios` | Portal web | La página no ofrece datos del GMML ni del MM N.° 2, ni descargas de precios mayoristas diarios. Remite a «Datos y estadísticas agrarias» (`https://siea.midagri.gob.pe/index.php/publicaciones/datos-y-estadisticas-agrarias`), que no se revisó | No comprobado | No |
| 7 | Plataforma Nacional de Datos Abiertos | `https://www.datosabiertos.gob.pe/search/type/dataset?query=mercado%20mayorista` | — | La búsqueda devuelve un solo conjunto, del Mercado Modelo de **Piura** (GORE Piura). No aparece ningún conjunto de MIDAGRI o EMMSA del GMML. Búsqueda única, no exhaustiva | — | — |
| 8 | Portal antiguo del SISAP (`index2.php`) | `http://sistemas.midagri.gob.pe/sisap/portal/index2.php` | HTML antiguo con enlaces a `minag.gob.pe/boletines/…` | No explorado | No comprobado | No evaluado |

**Notas técnicas sobre el SISAP:**

- `https://sistemas.midagri.gob.pe` **falla el handshake TLS**. OpenSSL dio `SSLV3_ALERT_HANDSHAKE_FAILURE` y SChannel `SEC_E_ALGORITHM_MISMATCH`, así que se usa **HTTP**.
- `http://sistemas.midagri.gob.pe/robots.txt` devuelve 404. No hay restricciones declaradas.
- El portal enlaza la guía de uso en `http://sistemas.midagri.gob.pe/sisap/portal2/mayorista/external_files/GUIA DE CONSULTA.pdf`. Según la guía, las consultas pueden ser por día, intervalo, semana, mes o año.
- Códigos de mercado del formulario: GMML `15011501`, MM N.° 2 de frutas `15011502`, Mercado Modelo de Frutas `15011503`, Santa Anita (productores) `15013704`, entre otros.

**Licencia:** no se encontró una licencia explícita que permita redistribuir los datos del SISAP ni los PDF de gob.pe. Por eso solo se publican el código, las URL y los hashes; los datos crudos quedan en `data/raw/`, que está en `.gitignore`.

## 2. Resultados de la muestra (T2)

### 2.1 Muestra descargada

En total se hicieron 8 consultas de datos al SISAP y se descargaron 4 PDF de gob.pe, es decir, 12 archivos de datos. Esto supera un poco el tope orientativo de unos 10; las consultas extra fueron necesarias para la cobertura mensual y para el MM N.° 2. Sumando las páginas de catálogo e índice, son 23 archivos; la lista completa con SHA-256 está en el manifiesto.

| Consulta SISAP (id) | Mercado | Periodo | Resultado |
|---|---|---|---|
| `gmml_2016_03` | GMML | 01–10/03/2016, diario | 10 días, bolsa y cajón |
| `gmml_2019_anual` | GMML | 01/01–31/12/2019, diario | 365 días |
| `gmml_2023_12` | GMML | 15/12/2023–10/01/2024, diario | 27 días |
| `gmml_2025_09_10` | GMML | 01/09–31/10/2025, diario | 61 días |
| `gmml_2026_ultimo` | GMML | 15/10–14/11/2026, diario | 31 días; el último con dato es el 14/11/2026 |
| `gmml_mensual_1997_2026` | GMML | Meses de 1997 a 2026 | 359 meses (ENE-1997 a NOV-2026) |
| `mm2_mensual_1997_2026` | MM N.° 2 | Meses de 1997 a 2026 | «No existen resultados…» |
| `mm2_2026_11_genero` | MM N.° 2 | 01–13/11/2026, género 0611 | «No existen resultados…» |

| PDF gob.pe | Fecha del reporte | Página mensual (n.º de PDF listados) |
|---|---|---|
| GMML, doc 1636162 | lunes, 01/02/2021 | febrero de 2021 (20 PDF) |
| GMML, doc 5629670 | viernes, 29/12/2023 | diciembre de 2023 (19 PDF) |
| GMML, doc 8926013 | viernes, 31/10/2025 | octubre de 2025 (23 PDF) |
| MM N.° 2, doc 10961327 | viernes, 13/11/2026 | noviembre de 2026 (5 PDF hasta el 13/11) |

Además, en el sondeo exploratorio previo al script se descargó y comparó el PDF del GMML del 01/10/2025 (`https://cdn.www.gob.pe/uploads/document/file/8759840/7244910-reporte-de-ingreso-y-precios-en-el-gran-mercado-mayorista-de-lima-01-10-2025.pdf`). No figura en el manifiesto.

### 2.2 Nombres exactos de producto

- **Catálogo del SISAP** (género `0611` «Limon», igual en el GMML y en el MM N.° 2):
  - `061101` **«Limon sutil bolsa»**
  - `061102` **«Limon sutil cajon»**
  - `061106` **«Limon tahití»**
- **Encabezado de la tabla del SISAP:** «Limon Sutil Bolsa», «Limon Sutil Cajon». *Limon tahití* no aparece en ninguna consulta, ni siquiera en la mensual de 1997 a 2026, lo que indica que no tiene datos.
- **PDF del GMML** (2021, 2023, 2025): «Limon Sutil Bolsa» (unidad «Bolsa», equivalencia 45.00 kg) y «Limon Sutil Cajon» (unidad «Cajon», 23.00 kg).
- **PDF del MM N.° 2** (13/11/2026): «Limon Dulce - Selva» (unidad «Jaba», 20.00 kg). Es el único limón de ese PDF.
- **Limón dulce en el SISAP del GMML:** no existe. En el catálogo del GMML no hay ningún producto con «dulce» bajo el género *Limon*.

### 2.3 Variables y unidades exactas

**SISAP** (título de la tabla: «Lima Metropolitana: Precio Máximo, Precio Promedio, Precio Mínimo, Volumen según mercado y producto y procedencia (sólo para volumen).»):

| Variable (literal) | Unidad (literal) |
|---|---|
| Precio Máximo | «(S/. x Kg., S/. x Lt. o S/. x Unid.)». Para el limón es S/ por kg: se comprobó dividiendo el precio del PDF entre la equivalencia en kg (§2.5) |
| Precio Promedio | igual |
| Precio Mínimo | igual |
| Volumen | «(t)». Viene una columna «Total» y una por procedencia a nivel de **provincia** (p. ej., Zarumilla, Sullana, Tumbes, Paita, Chiclayo, Lambayeque, Piura) |

- Pie de la tabla: «Fuente: GRAN MERCADO MAYORISTA DE LIMA», «Elaboración: MIDAGRI-DGESEP-DEIA-Área de Comercio».
- La celda vacía se escribe «...».
- **No hay variable de stock** en el SISAP ni en los PDF.

**PDF del GMML** (encabezado literal: «Masa de Ingreso (t)» y «Precios S/ x Unidad de medida»):

- Masa de ingreso: «Ayer», «Hoy», «Promedio Últ. 7 días», «Últ. 4 <día de la semana>» (p. ej., «Últ. 4 viernes»).
- Precio: «Unidad de medida», «Equiv en Kg», «Ayer», «Hoy», «Precio promedio últimos 7 días».
- En 2021 y 2023 hay una columna de tendencia más (valores vistos: «Baja», «Estable»), que en 2025 ya no aparece.

### 2.4 Días sin dato

En el SISAP del GMML **sí hay datos los domingos** (p. ej., 06/03/2016) y el **precio promedio aparece todos los días** en todas las muestras diarias. Los huecos solo afectan al **volumen**:

| Muestra | Bolsa: días con volumen | Cajón: días con volumen | Días sin volumen de bolsa |
|---|---|---|---|
| 2019 (365 días) | 359 | 333 (29 de los 32 huecos son miércoles) | 2019-01-01, 07-10, 07-17, 08-14, 11-20, 12-25 (4 miércoles, 1 de enero y 25 de diciembre) |
| 15/12/2023–10/01/2024 (27 días) | 27 | 23 | — |
| 01/09–31/10/2025 (61 días) | 60 | 41 | 2025-10-15 (miércoles) |
| 15/10–14/11/2026 (31 días) | 31 | 23 | — |

Volumen diario de bolsa en 2019 por día de la semana (mediana, t): lunes 416, martes 130, **miércoles 5**, jueves 351, viernes 378, sábado 351, domingo 276. En 2019 hubo 35 días con menos de 10 t de bolsa. La causa del mínimo de los miércoles **no se investigó**.

- **Serie mensual:** falta el volumen de **septiembre de 1997** en ambas presentaciones; el precio está en los 359 meses.
- **PDF de gob.pe:** no se publica un PDF todos los días. Las páginas mensuales listan 20 PDF (febrero de 2021), 19 (diciembre de 2023) y 23 (octubre de 2025). Qué días faltan no se comprobó.

### 2.5 Validación cruzada SISAP frente a PDF

Las columnas «Ayer» y «Hoy» del PDF coinciden exactamente con el volumen total diario del SISAP. El precio del PDF dividido entre la equivalencia en kg coincide con el «Precio Promedio» del SISAP:

| Día | Producto | Volumen en PDF (t) | Volumen en SISAP (t) | Precio en PDF / equiv. | Precio prom. en SISAP (S/ por kg) |
|---|---|---|---|---|---|
| 28/12/2023 | Bolsa | 408 | 408 | — | 1.08 |
| 29/12/2023 | Bolsa | 324 | 324 | 48.75 / 45 = 1.083 | 1.08 |
| 29/12/2023 | Cajón | 33 | 33 | 28.75 / 23 = 1.25 | 1.25 |
| 30/10/2025 | Bolsa | 899 | 899 | 102.50 / 45 = 2.278 | 2.28 |
| 31/10/2025 | Bolsa | 584 | 584 | 97.50 / 45 = 2.167 | 2.17 |
| 31/10/2025 | Cajón | «:» (sin dato) | «...» (sin dato) | 57.50 / 23 = 2.50 | 2.50 |
| 30/09/2025 y 01/10/2025 | Bolsa | 76 y 3 | 76 y 3 | — | — |

Además, la suma de los volúmenes diarios de 2019 es idéntica al total mensual del SISAP: 105 400 t de bolsa y 71 960 t de cajón. El precio mensual es aproximadamente la media de los precios diarios (enero de 2019, bolsa: 0.82 mensual frente a 0.828 de media diaria).

### 2.6 Cambios de formato en el tiempo

- **PDF del GMML:**
  - En 2021 y 2023 hay una columna de tendencia («Baja», «Estable») que en 2025 no aparece. Con pdfplumber, esa columna **se superpone** al precio de 7 días (salen cadenas como `58E.5n0`, que son «58.50» intercaladas con «En…»).
  - La línea de fuente cambia: en 2021 «Fuente : Gran Mercado Mayorista de Lima»; en 2023 y 2025 «Fuente : EMMSA - Gran Mercado Mayorista de Lima».
  - Las URL de 2021 llevan barras en el nombre (`…LIMA - 01/02/21.pdf`); las de 2023 y 2025 siguen el patrón `<id>-reporte-…-dd-mm-aa(aa).pdf`.
- **SISAP:**
  - Los volúmenes mensuales llevan **decimales de 1997 a 2005** (p. ej., 1997: 28 414.87 t de bolsa) y son **enteros desde 2006**.
  - Número de procedencias de la bolsa: 4 en 01–10/03/2016, 11 en todo 2019 y 40 en la serie mensual de 1997 a 2026 (incluidas «Procedencia Desconocida» y «Sin Especificacion»).
  - El formato de la tabla es el mismo en todas las muestras de 2016 a 2026.
- **Composición por presentación** (volumen anual en el GMML, t):

| Año | 1997 | 2005 | 2010 | 2015 | 2019 | 2020 | 2023 | 2025 |
|---|---|---|---|---|---|---|---|---|
| Bolsa | 28 415 | 36 830 | 59 649 | 75 050 | 105 400 | 129 407 | 131 479 | 166 344 |
| Cajón | 41 342 | 48 444 | 49 594 | 67 813 | 71 960 | 58 942 | 20 943 | 16 486 |

## 3. Factibilidad de la serie completa (T3)

| Aspecto | SISAP (fuente 1) | PDF de gob.pe (fuente 2) |
|---|---|---|
| Primer día | Mensual desde **ENE-1997**. La resolución **diaria** se comprobó solo desde 2016-03-01 (antes, no comprobado; el formulario acepta fechas desde 1997) | La página mensual más antigua vista es diciembre de 2020 (solo en una búsqueda web) y la descargada es febrero de 2021. Si los 86 resultados fueran meses consecutivos, el primero sería octubre de 2019 (**inferencia, no comprobada**) |
| Último día | **14/11/2026**, el día de la consulta (no se comprobó si el dato del día es provisional) | 13/11/2026 (MM N.° 2); noviembre de 2026 listado (GMML) |
| N.º aproximado de días | 1997-01-01 a 2026-11-14: **10 910 días de calendario** si hubiera datos diarios desde 1997. Desde 2016-01-01: **3 971 días**. Con precio todos los días y volumen de bolsa en ~98 % de los días (2019: 359 de 365) | Unos 20–23 PDF por mes, es decir, unos 1 700–2 000 PDF si la serie empieza en 2019–2020 |
| Esfuerzo | **Bajo.** Un POST por año devuelve la tabla diaria completa (2019: 290 KB). La serie entera son unas 30 peticiones y se parsea con BeautifulSoup (ya implementado en `parsear_tabla_sisap`) | **Alto.** Hay que llegar a cada página mensual sin paginar (robots.txt) y descargar unos 2 000 PDF. Con **pdfplumber**, `extract_text` y `extract_tables` sacan bien la fila del limón en 2025; en 2021 y 2023 la columna de tendencia corrompe el precio de 7 días (se arreglaría filtrando caracteres por posición). **camelot no se probó** porque pdfplumber bastó |
| Riesgos | (a) Ambigüedad de «...»: ¿ingreso cero o falta de registro? (b) Posible ruptura en 2005–2006 (decimales a enteros) y un posible cambio de sede del mercado (no comprobado). (c) Mínimo estructural de los miércoles. (d) Traslado del cajón a la bolsa. (e) HTTPS roto: el servidor es antiguo y podría desaparecer o cambiar. (f) Los datos se pueden revisar con el tiempo, por eso se guardan hashes. (g) Límite de tiempo de la consulta: el JavaScript del cliente usa 35 s; una consulta de 1 año respondió bien, y 30 años mensuales (840 KB) también | (a) PDF faltantes (no todos los días). (b) Cambios de formato (§2.6). (c) Paginación prohibida. (d) Solo cubre ~2019/2020 en adelante. (e) El precio viene por unidad de medida y hay que convertirlo a kg |

### Recomendación

- **Fuente principal: SISAP v2.0** (`resumenes/filtrar`), mercado GMML `15011501`, variedades `061101` y `061102`, descargando un año por petición desde 1997. La variable objetivo propuesta es la suma de bolsa y cajón (ver `docs/PROTOCOLO_BORRADOR.md`).
- **Validación:** muestreo aleatorio de unos 20 PDF de gob.pe repartidos entre 2021 y 2026 para comparar volumen y precio con el SISAP.
- **Limón dulce:** queda fuera del experimento principal porque no hay una serie larga automatizable. Solo aparece en los PDF del MM N.° 2.
- **Primer paso del día 2:**
  1. Una consulta diaria corta en 1997, otra en 2005 y otra en 2012, para fijar desde cuándo existe la resolución diaria.
  2. Comprobar que la suma de los días coincide con el mensual en un año anterior a 2006.
  3. Después, la descarga completa (unas 30 peticiones).

## 4. Lo que no se pudo comprobar

1. **Resolución diaria antes de 2016-03-01.** Solo se comprobó la serie mensual desde enero de 1997.
2. **Significado de «...» en el volumen** del SISAP (y de «:» en el PDF): ingreso nulo o dato faltante. No hay documentación al respecto en la guía de consulta.
3. **Por qué el volumen de los miércoles es casi nulo.** No se buscó la causa.
4. **Si el dato del día de la consulta (14/11/2026) es definitivo o provisional.**
5. **Causa de la ruptura 2005–2006** (decimales a enteros) y si cambió la sede física del mercado asociado al código `15011501`. No se verificó con ninguna fuente.
6. **Primer mes de los PDF en gob.pe.** robots.txt prohíbe paginar (`Disallow: /*?sheet=`), así que no se recorrió el índice completo. Tampoco se comprobó qué días faltan en los PDF.
7. **El botón «Exportar a Excel» del SISAP.** No se probó; por el código, reenvía la tabla HTML ya generada a un formulario aparte.
8. **El limón dulce en otros mercados del SISAP** (p. ej., el Mercado Modelo de Frutas `15011503`) o con otro nombre. Solo se revisaron el GMML y el MM N.° 2.
9. **El contenido del «Boletín de Abastecimiento y Precios Mayoristas GMML y MM N.° 2»** y de la sección «Datos y estadísticas agrarias» del SIEA.
10. **La existencia de un conjunto de datos de MIDAGRI o EMMSA en datosabiertos.gob.pe.** Se hizo una sola búsqueda, no exhaustiva.
11. **Licencia de reutilización** del SISAP y de los PDF de gob.pe. No se encontró ninguna; se asume que no se pueden redistribuir.
12. **camelot.** No se instaló ni se probó.
13. **Stock.** No es una variable del SISAP ni de los PDF muestreados. Las notas de prensa hablan de «stock» de limón, pero no se verificó de dónde sale ese dato.

## 5. Archivos para `git add` y mensaje sugerido

```bash
git add .gitignore README.md requirements.txt \
        config/fuentes.yaml \
        scripts/01_reconocimiento_sisap.py \
        docs/PROTOCOLO_BORRADOR.md \
        reportes/INFORME_DIA1.md reportes/dia1_resumen.json reportes/dia1_manifiesto_sha256.csv \
        data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep
```

No se añaden `data/raw/*`, `data/interim/*` ni `.venv/`, que ya están excluidos en `.gitignore`.

Mensaje de commit sugerido:

```
Día 1: reconocimiento de fuentes SISAP/gob.pe y esqueleto del repositorio

- SISAP v2.0 (mayoristas) identificado como fuente principal: consulta POST
  automatizable con datos diarios de limón sutil bolsa/cajón en el GMML
  (mensual ENE-1997–NOV-2026; diario comprobado 2016–2026).
- Muestra de 8 consultas SISAP y 4 PDF de gob.pe; validación cruzada
  SISAP vs PDF exacta en volumen y precio.
- Script de reconocimiento reproducible, configuración de fuentes,
  manifiesto SHA-256 de descargas y borrador de protocolo.
```
