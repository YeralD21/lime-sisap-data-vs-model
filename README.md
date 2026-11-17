# lime-sisap-data-vs-model

**Experimento 2: ¿falla el modelo o faltan datos?** Pronóstico del abastecimiento y del precio diario del limón en los mercados mayoristas de Lima con datos del SISAP (MIDAGRI).

Autor: YesaR (Fabrizio Sanchez Saravia).

## Objetivo y pregunta

En el artículo publicado en MDPI *Forecasting* (repositorio `duallstm-lime-forecasting-peru`, release v2.2-forecasting), una red DualLSTM con atención de Bahdanau quedó por debajo de SARIMA, ETS, Theta, Chronos y Seasonal Naive al pronosticar la producción **mensual** de limón (SISAGRI, ~120 meses por cultivo).

Queda una pregunta abierta: **¿la red pierde por el modelo o porque tiene pocos datos?**

Para responderla se construyen **curvas de aprendizaje** con el ingreso de limón sutil (bolsa + cajón, en t) al Gran Mercado Mayorista de Lima (GMML), registrado por el SISAP desde 1997. En cada curva solo cambia la longitud de la historia de entrenamiento:

- **Brazo D (diario):** 120, 360, 1 000 días y la historia máxima (unos 8 000–9 600 días).
- **Brazo M (mensual):** 60, 120, 240 meses y la máxima (unos 270–315 meses).

Si la red mejora más que los modelos clásicos a medida que crece la historia, el problema del artículo eran los datos. Si sigue perdiendo, el problema es el modelo. La regla de decisión está escrita en [`docs/PROTOCOLO.md`](docs/PROTOCOLO.md) antes de ver ningún resultado.

## Estado

| Día | Contenido | Estado |
|---|---|---|
| 1 | Reconocimiento de fuentes, muestra pequeña y esqueleto del repositorio | hecho, ver [`reportes/INFORME_DIA1.md`](reportes/INFORME_DIA1.md) |
| 2 | Serie completa (1997-01-01 a 2026-11-13), calidad de datos y diagnóstico descriptivo | hecho, ver [`reportes/INFORME_DIA2.md`](reportes/INFORME_DIA2.md) |
| – | Protocolo | **listo para congelar**: [`docs/PROTOCOLO.md`](docs/PROTOCOLO.md), con hashes en [`docs/hashes_protocolo.json`](docs/hashes_protocolo.json). Lo congela el commit de YesaR. El borrador del día 1 queda como historial |

## Fuentes (consultadas el 2026-11-14)

| Fuente | URL | Uso |
|---|---|---|
| SISAP v2.0, módulo de Mercados Mayoristas (MIDAGRI) | <http://sistemas.midagri.gob.pe/sisap/portal2/mayorista/> (portal: <http://sistemas.midagri.gob.pe/sisap/portal/>) | **Fuente principal.** Consulta por formulario (POST) que devuelve una tabla HTML diaria con precio máx./prom./mín. (S/ por kg) y volumen (t) por procedencia, de 1997 a hoy |
| Reporte de Ingreso y Precios en el GMML (gob.pe) | <https://www.gob.pe/institucion/midagri/colecciones/335-reporte-de-ingreso-y-precios-en-el-gran-mercado-mayorista-de-lima> | Validación cruzada (un PDF por día hábil) |
| Reporte de Ingreso y Precios en el Mercado Mayorista de Frutas N.° 2 (gob.pe) | <https://www.gob.pe/institucion/midagri/colecciones/14-reporte-de-ingreso-y-precios-en-el-mercado-mayorista-de-frutas-n-2> | Limón dulce (no está en el SISAP) |
| SISAGRI, producción nacional (artículo) | ver el repositorio del artículo | Puente mensual 2016–2025 |

Los parámetros exactos (códigos de mercado, de producto y endpoints) están en [`config/fuentes.yaml`](config/fuentes.yaml).

## Política de datos

- Los datos de terceros **no se redistribuyen**, ni crudos ni procesados: `data/raw/`, `data/interim/` y `data/processed/*.csv` están en `.gitignore`. No se encontró una licencia que permita redistribuirlos explícitamente, y las series procesadas son una transformación directa de los datos del SISAP.
- Se publican los scripts, el manifiesto de descargas y los SHA-256 de las series procesadas ([`docs/hashes_protocolo.json`](docs/hashes_protocolo.json)), para que cualquiera pueda regenerarlas y comprobar que obtiene los mismos archivos (ver «Cómo regenerar las series procesadas»).
- Se publican el código, las URL de origen y el SHA-256 de cada descarga: [`reportes/dia1_manifiesto_sha256.csv`](reportes/dia1_manifiesto_sha256.csv) del día 1 y [`reportes/manifiesto_crudos.csv`](reportes/manifiesto_crudos.csv) de la serie completa. El contenido del SISAP puede cambiar con el tiempo, así que el hash documenta lo que se descargó, no garantiza que una nueva descarga sea idéntica.
- Scraping responsable: User-Agent identificable, ≥ 1,5 s entre peticiones, reintentos con espera creciente, respeto de `robots.txt` (en gob.pe no se pagina con `?sheet=`, que robots.txt prohíbe) y caché local para no repetir peticiones.

## Cómo reproducir

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_reconocimiento_sisap.py   # día 1: reconocimiento
python scripts/02_descarga_sisap.py         # día 2: descarga completa, reanudable (~60 peticiones la primera vez)
python scripts/03_series.py                 # día 2: series procesadas, calidad y figuras (sin modelos)
python scripts/04_hashes_protocolo.py --verificar   # comprueba los hashes congelados
```

### Cómo regenerar las series procesadas

`data/processed/sutil_diario.csv`, `sutil_mensual.csv` y `sutil_diario_procedencia.csv` no están en el repositorio. Para obtenerlas:

1. `python scripts/02_descarga_sisap.py`: descarga los crudos a `data/raw/` (reanudable, con caché). La fecha de corte (2026-11-13) está en `config/fuentes.yaml`.
2. Compare los SHA-256 de lo descargado con [`reportes/manifiesto_crudos.csv`](reportes/manifiesto_crudos.csv).
3. `python scripts/03_series.py`: genera `data/processed/*.csv` y las tablas y figuras de `reportes/`.
4. `python scripts/04_hashes_protocolo.py --verificar`: compara las series con los hashes congelados. Si dice «OK», son exactamente los archivos que usa el experimento. Si falta un archivo dice «FALTA»; si difiere, dice «DIFIERE».

El SISAP puede revisar valores pasados (se vieron revisiones de 1 t en 2021), así que una descarga nueva puede no reproducir los hashes. En ese caso el protocolo lo trata como desviación (`docs/DESVIACIONES.md`).

`03_series.py` lee la producción nacional de SISAGRI del repositorio del artículo, en solo lectura. La ruta está en `config/fuentes.yaml` y se puede cambiar con la variable de entorno `SISAGRI_CSV`.

El script del día 1 hace unas 25 peticiones (pausa de 1,5 s entre ellas) y guarda caché: si un archivo crudo ya existe, no lo vuelve a pedir. Genera:

- `data/raw/sisap/*.html`, `data/raw/gobpe/*.{html,pdf}`: respuestas crudas (no versionadas).
- `data/interim/sisap_muestra_long.csv`: muestra en formato largo (no versionada).
- `reportes/dia1_resumen.json`: cifras usadas en el informe.
- `reportes/dia1_manifiesto_sha256.csv`: URL, parámetros POST y SHA-256 de cada descarga.

Nota técnica: `https://sistemas.midagri.gob.pe` falló el handshake TLS el 2026-11-14 (el servidor no tiene un algoritmo en común con OpenSSL ni con SChannel), por eso el SISAP se consulta por HTTP.

## Estructura

```
config/fuentes.yaml                  URL y parámetros de cada fuente
scripts/01_reconocimiento_sisap.py   reconocimiento del día 1 (T1–T3)
scripts/sisap_comun.py               cliente HTTP, formulario y parser del SISAP (día 2 en adelante)
scripts/02_descarga_sisap.py         descarga reanudable de la serie completa
scripts/03_series.py                 series procesadas, calidad de datos, diagnóstico y figuras
scripts/04_hashes_protocolo.py       SHA-256 del protocolo y de las series
docs/PROTOCOLO.md                    protocolo final (listo para congelar)
docs/hashes_protocolo.json           hashes para la congelación
docs/PROTOCOLO_BORRADOR.md           borrador del día 1 (historial)
reportes/INFORME_DIA{1,2}.md         informes diarios
reportes/dia1_*, reportes/dia2_*     cifras, tablas y manifiestos
reportes/figuras_dia2/               figuras descriptivas
data/processed/                      sutil_diario.csv, sutil_mensual.csv, sutil_diario_procedencia.csv (no versionados; se regeneran)
data/raw/ data/interim/              no versionados
```
