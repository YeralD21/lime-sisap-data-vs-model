"""Dia 1 - reconocimiento de fuentes de datos diarios de limon (SISAP / gob.pe).

Reproduce las tareas T1-T3 del informe reportes/INFORME_DIA1.md:
  * SISAP v2.0 (modulo mayoristas): catalogo de variedades de limon y una
    muestra pequena de consultas (tabla HTML) por mercado y periodo.
  * gob.pe: primera pagina de las colecciones de reportes diarios (PDF) y
    una muestra de PDF, extraidos con pdfplumber.

Scraping responsable: User-Agent identificable, >= 1.5 s entre peticiones,
reintentos con espera creciente, respeto de robots.txt y cache local
(si el archivo crudo ya existe no se vuelve a pedir).

Salidas:
  data/raw/sisap/*.html, data/raw/gobpe/*.{html,pdf}   (no versionados)
  data/interim/sisap_muestra_long.csv                  (no versionado)
  reportes/dia1_manifiesto_sha256.csv                  (URL + SHA-256 de cada descarga)
  reportes/dia1_resumen.json                           (cifras usadas en el informe)

Uso:  python scripts/01_reconocimiento_sisap.py
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import pdfplumber
import requests
import yaml
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config" / "fuentes.yaml").read_text(encoding="utf-8"))
RAW = RAIZ / "data" / "raw"
INTERIM = RAIZ / "data" / "interim"
REPORTES = RAIZ / "reportes"


# --------------------------------------------------------------------------
# Cliente HTTP responsable
# --------------------------------------------------------------------------
class Cliente:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s = requests.Session()
        self.s.headers["User-Agent"] = cfg["user_agent"]
        self._ultima = 0.0
        self._robots: dict[str, list[str] | None] = {}
        self.manifiesto: list[dict] = []

    # robots.txt minimo con comodines '*' (urllib.robotparser no los soporta)
    def _reglas(self, url: str) -> list[str] | None:
        p = urllib.parse.urlsplit(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            try:
                r = self._get_crudo(host + "/robots.txt")
                if r.status_code != 200:
                    self._robots[host] = None  # sin robots.txt -> sin restricciones
                else:
                    reglas, aplica = [], False
                    for linea in r.text.splitlines():
                        linea = linea.split("#")[0].strip()
                        if not linea or ":" not in linea:
                            continue
                        k, v = (x.strip() for x in linea.split(":", 1))
                        if k.lower() == "user-agent":
                            aplica = v == "*"
                        elif k.lower() == "disallow" and aplica and v:
                            reglas.append(v)
                    self._robots[host] = reglas
            except requests.RequestException:
                self._robots[host] = None
        return self._robots[host]

    def permitido(self, url: str) -> bool:
        reglas = self._reglas(url)
        if not reglas:
            return True
        p = urllib.parse.urlsplit(url)
        ruta = p.path + (("?" + p.query) if p.query else "")
        for patron in reglas:
            rx = "^" + re.escape(patron).replace(r"\*", ".*")
            if re.match(rx, ruta):
                return False
        return True

    def _esperar(self):
        falta = self.cfg["espera_min_s"] - (time.monotonic() - self._ultima)
        if falta > 0:
            time.sleep(falta)
        self._ultima = time.monotonic()

    def _get_crudo(self, url):
        self._esperar()
        return self.s.get(url, timeout=self.cfg["timeout_s"])

    def pedir(self, metodo: str, url: str, **kw) -> requests.Response:
        if not self.permitido(url):
            raise PermissionError(f"robots.txt no permite: {url}")
        espera = self.cfg["espera_reintento_s"]
        for intento in range(1, self.cfg["reintentos"] + 1):
            self._esperar()
            try:
                r = self.s.request(metodo, url, timeout=self.cfg["timeout_s"], **kw)
                if r.status_code in (401, 403):
                    raise PermissionError(f"{r.status_code} (bloqueo o login) en {url}: se detiene")
                if r.status_code < 500 and r.status_code != 429:
                    return r
                print(f"  ! {r.status_code} en {url} (intento {intento})")
            except requests.RequestException as e:
                print(f"  ! {e.__class__.__name__} en {url} (intento {intento})")
            time.sleep(espera)
            espera *= 2
        raise RuntimeError(f"fallo tras {self.cfg['reintentos']} intentos: {url}")

    def descargar(self, destino: Path, metodo: str, url: str, fuente: str, **kw) -> bytes:
        """Descarga con cache: si 'destino' existe no se vuelve a pedir."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            contenido = destino.read_bytes()
        else:
            r = self.pedir(metodo, url, **kw)
            r.raise_for_status()
            contenido = r.content
            destino.write_bytes(contenido)
        self.manifiesto.append({
            "fuente": fuente,
            "archivo": destino.relative_to(RAIZ).as_posix(),
            "metodo": metodo,
            "url": url,
            "parametros_post": kw.get("data", "") if metodo == "POST" else "",
            "bytes": len(contenido),
            "sha256": hashlib.sha256(contenido).hexdigest(),
        })
        return contenido


# --------------------------------------------------------------------------
# SISAP
# --------------------------------------------------------------------------
H_AJAX = {"Content-Type": "application/x-www-form-urlencoded", "X-Requested-With": "XMLHttpRequest"}


def _q(pares: list[tuple[str, str]]) -> str:
    # El JS original usa escape(); quote() es equivalente para estos valores.
    return "&".join(f"{k}={urllib.parse.quote(v, safe='')}" for k, v in pares)


def form_consulta(sis: dict, item: dict, productos: list[str]) -> str:
    merc = sis["mercados"][item["mercado"]]
    pares = [("mercado", merc)]
    pares += [("variables[]", v) for v in sis["variables"]]
    pares += [("procedencias[]", p) for p in sis["procedencias"]]
    hoy = dt.date.today().strftime("%d/%m/%Y")
    pares += [("fecha", item.get("hasta", hoy)), ("desde", item.get("desde", hoy)),
              ("hasta", item.get("hasta", hoy))]
    if item["periodicidad"] == "mensual":
        a0, a1 = (int(x) for x in item["anios"].split("-"))
        pares += [("anios[]", str(a)) for a in range(a0, a1 + 1)]
        pares += [("meses[]", f"{m:02d}") for m in range(1, 13)]
    pares += [("productos[]", p) for p in productos]
    pares += [("periodicidad", item["periodicidad"]), ("__ajax_carga_final", "consulta"), ("ajax", "true")]
    return _q(pares)


def parsear_tabla_sisap(texto: str) -> tuple[list[dict], dict]:
    """Convierte la tabla HTML del SISAP a filas largas.

    Encabezado (3 filas): producto (colspan) / variable (rowspan 2 o colspan
    para volumen) / procedencia (Total + provincias). Celda '...' = sin dato.
    """
    soup = BeautifulSoup(texto, "html.parser")
    meta = {"titulos": [h.get_text(" ", strip=True) for h in soup.find_all("h1")],
            "pie": [t.get_text(" ", strip=True) for t in soup.find_all("tr") if "Fuente" in t.get_text()][:1]}
    tabla = soup.find("table")
    if tabla is None:
        return [], {**meta, "error": soup.get_text(" ", strip=True)[:500]}
    enc = [tr for tr in tabla.find_all("tr") if "encabezado" in (tr.get("class") or [])]
    filas = [tr for tr in tabla.find_all("tr") if "contenido" in (tr.get("class") or [])]
    # fila 0: productos
    productos = []
    for td in enc[0].find_all("td")[1:]:
        productos += [td.get_text(strip=True)] * int(td.get("colspan", 1))
    # fila 1: variables; las de volumen tienen colspan = n procedencias
    variables, unidades = [], {}
    for td in enc[1].find_all("td"):
        nombre = td.contents[0].strip() if td.contents else ""
        nombre = html.unescape(str(nombre)).strip()
        unidad = td.find("small").get_text(strip=True) if td.find("small") else td.get_text(" ", strip=True).replace(nombre, "").strip()
        unidades[nombre] = unidad
        variables += [nombre] * int(td.get("colspan", 1))
    # fila 2: procedencias (solo para columnas de volumen)
    procs = [td.get_text(strip=True) for td in enc[2].find_all("td")] if len(enc) > 2 else []
    columnas, ip = [], 0
    for prod, var in zip(productos, variables):
        if var.lower().startswith("volumen"):
            columnas.append((prod, "Volumen", procs[ip])); ip += 1
        else:
            columnas.append((prod, var, ""))
    meta["unidades"] = unidades
    meta["productos"] = sorted(set(productos))
    largo = []
    for tr in filas:
        celdas = [td.get_text(strip=True) for td in tr.find_all("td")]
        periodo, valores = celdas[0], celdas[1:]
        for (prod, var, proc), val in zip(columnas, valores):
            largo.append({"periodo": periodo, "producto": prod, "variable": var,
                          "procedencia": proc, "valor": None if val in ("...", "") else float(val.replace(",", ""))})
    return largo, meta


def reconocer_sisap(cli: Cliente) -> dict:
    sis = CFG["sisap"]
    base = sis["base"]
    out = {"portal": sis["portal"], "base": base}
    dest = RAW / "sisap"
    portada = cli.descargar(dest / "portada_mayorista.html", "GET", base, "sisap")
    portada_txt = portada.decode("utf-8", "replace")
    out["cookie_sesion"] = sorted(cli.s.cookies.get_dict().keys())
    out["anios_formulario"] = sorted(set(re.findall(r"<option value=(\d{4})>", portada_txt)))[:1] + \
        sorted(set(re.findall(r"<option value=(\d{4})[ >]", portada_txt)))[-1:]
    out["mercados"] = re.findall(r"<option value=(\d{8})>([^<]+)</option>", portada_txt)

    # Si la portada vino de cache no hay cookie: pedir una sesion fresca.
    if not cli.s.cookies:
        cli.pedir("GET", base)

    variedades = {}
    for clave, cod in sis["mercados"].items():
        cli.descargar(dest / f"generos_{clave}.html", "POST", base + sis["endpoints"]["generos_por_mercado"],
                      "sisap", data=_q([("mercado", cod), ("ajax", "true")]), headers=H_AJAX)
        b = cli.descargar(dest / f"variedades_limon_{clave}.html", "POST",
                          base + sis["endpoints"]["variedades_por_genero"], "sisap",
                          data=_q([("expandir", f"{sis['genero_limon']}~checkBox"), ("mercado", cod), ("ajax", "true")]),
                          headers=H_AJAX)
        t = b.decode("latin-1")
        variedades[clave] = re.findall(r"value=(\d{6})[^>]*>([^<]+)</label>", t)
    out["variedades_limon"] = variedades

    productos = list(sis["variedades_limon"].keys())
    filas_todas, consultas = [], {}
    for item in sis["muestra"]:
        datos = form_consulta(sis, item, item.get("productos", productos))
        b = cli.descargar(dest / f"consulta_{item['id']}.html", "POST", base + sis["endpoints"]["consulta"],
                          "sisap", data=datos, headers=H_AJAX)
        filas, meta = parsear_tabla_sisap(b.decode("latin-1"))
        for f in filas:
            f.update({"consulta": item["id"], "mercado": item["mercado"]})
        filas_todas += filas
        consultas[item["id"]] = {**item, **meta, "n_filas_largas": len(filas), "bytes": len(b)}
        print(f"  SISAP {item['id']}: {len(filas)} valores, productos={meta.get('productos')}")
    out["consultas"] = consultas

    INTERIM.mkdir(parents=True, exist_ok=True)
    with open(INTERIM / "sisap_muestra_long.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["consulta", "mercado", "periodo", "producto", "variable", "procedencia", "valor"])
        w.writeheader(); w.writerows(filas_todas)
    out["resumen_muestra"] = resumir_sisap(filas_todas, sis["muestra"])
    return out


def resumir_sisap(filas: list[dict], muestra: list[dict]) -> dict:
    res = {}
    for item in muestra:
        fi = [f for f in filas if f["consulta"] == item["id"]]
        por_prod = {}
        for prod in sorted({f["producto"] for f in fi}):
            fp = [f for f in fi if f["producto"] == prod]
            periodos = sorted({f["periodo"] for f in fp})
            con_precio = {f["periodo"] for f in fp if f["variable"] == "Precio Promedio" and f["valor"] is not None}
            con_vol = {f["periodo"] for f in fp if f["variable"] == "Volumen" and f["procedencia"] == "Total" and f["valor"] is not None}
            d = {"n_periodos_en_tabla": len(periodos), "n_con_precio_prom": len(con_precio),
                 "n_con_volumen_total": len(con_vol),
                 "procedencias": sorted({f["procedencia"] for f in fp if f["variable"] == "Volumen" and f["procedencia"] != "Total"})}
            if item["periodicidad"] == "intervalo":
                d0 = dt.datetime.strptime(item["desde"], "%d/%m/%Y").date()
                d1 = dt.datetime.strptime(item["hasta"], "%d/%m/%Y").date()
                todos = {(d0 + dt.timedelta(i)) for i in range((d1 - d0).days + 1)}
                vol_d = {dt.datetime.strptime(p, "%d/%m/%Y").date() for p in con_vol}
                faltan = sorted(todos - vol_d)
                d["dias_calendario"] = len(todos)
                d["dias_sin_volumen"] = len(faltan)
                d["dias_sin_volumen_por_dia_semana"] = {
                    n: sum(1 for x in faltan if x.weekday() == i)
                    for i, n in enumerate(["lun", "mar", "mie", "jue", "vie", "sab", "dom"])}
                d["ejemplos_dias_sin_volumen"] = [x.isoformat() for x in faltan[:15]]
                vols = [f["valor"] for f in fp if f["variable"] == "Volumen" and f["procedencia"] == "Total" and f["valor"] is not None]
                precios = [f["valor"] for f in fp if f["variable"] == "Precio Promedio" and f["valor"] is not None]
                if vols:
                    d["volumen_total_t"] = {"min": min(vols), "max": max(vols), "media": round(sum(vols) / len(vols), 1)}
                if precios:
                    d["precio_prom_soles_kg"] = {"min": min(precios), "max": max(precios)}
            else:
                d["primer_periodo_con_volumen"] = _primer_ultimo(con_vol)[0]
                d["ultimo_periodo_con_volumen"] = _primer_ultimo(con_vol)[1]
                d["primer_periodo_con_precio"] = _primer_ultimo(con_precio)[0]
                d["ultimo_periodo_con_precio"] = _primer_ultimo(con_precio)[1]
                d["meses_faltantes_volumen"] = _meses_faltantes(con_vol)
                d["meses_faltantes_precio"] = _meses_faltantes(con_precio)
            por_prod[prod] = d
        res[item["id"]] = por_prod
    return res


MESES_ABR = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AGO": 8,
             "SEP": 9, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12}


def _clave_periodo(p: str):
    """Clave de orden para periodos mensuales del SISAP ('ABR-1997')."""
    m = re.match(r"([A-Z]{3})-(\d{4})$", p.strip().upper())
    if not m:
        return (0, 0, p)
    return (int(m.group(2)), MESES_ABR.get(m.group(1), 0), p)


def _meses_faltantes(periodos: set[str]) -> list[str]:
    claves = sorted(_clave_periodo(p)[:2] for p in periodos)
    if not claves:
        return []
    (a, m), fin, presentes, faltan = claves[0], claves[-1], set(claves), []
    while (a, m) <= fin:
        if (a, m) not in presentes:
            faltan.append(f"{a}-{m:02d}")
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return faltan


def _primer_ultimo(periodos: set[str]):
    if not periodos:
        return None, None
    o = sorted(periodos, key=_clave_periodo)
    return o[0], o[-1]


# --------------------------------------------------------------------------
# gob.pe (reportes PDF)
# --------------------------------------------------------------------------
RX_PDF = re.compile(r'https://cdn\.www\.gob\.pe/uploads/document/file/\d+/[^"?]+\.pdf(?:\?v=\d+)?')


def extraer_pdf(ruta: Path) -> dict:
    info = {"archivo": ruta.relative_to(RAIZ).as_posix()}
    with pdfplumber.open(ruta) as pdf:
        info["paginas"] = len(pdf.pages)
        textos = [(pg.extract_text() or "") for pg in pdf.pages]
        info["encabezado"] = textos[0].splitlines()[:6] if textos else []
        info["lineas_limon"] = [l for t in textos for l in t.splitlines() if re.search(r"(?i)lim[oó]n", l)]
        info["lineas_fuente"] = [l for t in textos for l in t.splitlines() if l.strip().lower().startswith("fuente")]
        # Prueba de extraccion tabular (pdfplumber.extract_tables)
        filas_tab = []
        for pg in pdf.pages:
            for tab in pg.extract_tables() or []:
                for fila in tab:
                    if fila and any(c and re.search(r"(?i)lim[oó]n", c) for c in fila):
                        filas_tab.append(fila)
        info["filas_limon_extract_tables"] = filas_tab
        info["n_caracteres_texto"] = sum(len(t) for t in textos)
    return info


def reconocer_gobpe(cli: Cliente) -> dict:
    g = CFG["gobpe"]
    dest = RAW / "gobpe"
    out = {"colecciones": {}, "pdf": []}
    for clave, url in g["colecciones"].items():
        b = cli.descargar(dest / f"coleccion_{clave}.html", "GET", url, "gobpe").decode("utf-8", "replace")
        meses = sorted(set(re.findall(r"/institucion/midagri/informes-publicaciones/(\d+-[a-z0-9-]+)", b)))
        n_res = re.search(r"(\d+)\s+resultados", b)
        hojas = sorted(set(re.findall(r"\?sheet=(\d+)", b)), key=int)
        out["colecciones"][clave] = {
            "url": url, "resultados_declarados": int(n_res.group(1)) if n_res else None,
            "hojas_paginacion": hojas,
            "paginacion_permitida_robots": cli.permitido(url + "?sheet=2"),
            "paginas_mensuales_primera_hoja": meses,
        }
    paginas = list(g["muestra_paginas_mensuales"])
    if g.get("muestra_mm2_desde_coleccion"):
        mm2 = out["colecciones"]["MM2_FRUTAS"]["paginas_mensuales_primera_hoja"]
        if mm2:
            ultimo = max(mm2, key=lambda s: int(s.split("-")[0]))
            paginas.append({"id": "mm2_ultimo_listado",
                            "url": "https://www.gob.pe/institucion/midagri/informes-publicaciones/" + ultimo})
    for p in paginas:
        b = cli.descargar(dest / f"mes_{p['id']}.html", "GET", p["url"], "gobpe").decode("utf-8", "replace")
        pdfs = list(dict.fromkeys(RX_PDF.findall(b)))
        pdfs = [u for u in pdfs if "/preview_" not in u]
        registro = {"id": p["id"], "pagina": p["url"], "n_pdf_listados": len(pdfs),
                    "primer_pdf": pdfs[0] if pdfs else None, "ultimo_pdf": pdfs[-1] if pdfs else None}
        if pdfs:
            # Nombre por id de documento: algunas URL antiguas traen '/' en el nombre (p. ej. .../01/02/21.pdf)
            doc_id = re.search(r"/document/file/(\d+)/", pdfs[0]).group(1)
            ruta = dest / f"pdf_{p['id']}_doc{doc_id}.pdf"
            cli.descargar(ruta, "GET", pdfs[0], "gobpe")
            registro.update(extraer_pdf(ruta))
        out["pdf"].append(registro)
        print(f"  gob.pe {p['id']}: {len(pdfs)} PDF listados; limon: {registro.get('lineas_limon')}")
    return out


# --------------------------------------------------------------------------
def main() -> int:
    cli = Cliente(CFG["cliente_http"])
    resumen = {"fecha_ejecucion": dt.datetime.now().isoformat(timespec="seconds"),
               "user_agent": CFG["cliente_http"]["user_agent"]}
    print("SISAP ..."); resumen["sisap"] = reconocer_sisap(cli)
    print("gob.pe ..."); resumen["gobpe"] = reconocer_gobpe(cli)
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / "dia1_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(REPORTES / "dia1_manifiesto_sha256.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cli.manifiesto[0].keys()))
        w.writeheader(); w.writerows(cli.manifiesto)
    print(f"OK: {len(cli.manifiesto)} archivos en el manifiesto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
