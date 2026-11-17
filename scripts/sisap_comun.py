"""Utilidades comunes (dia 2 en adelante): cliente HTTP responsable, formulario
del SISAP, parser de la tabla HTML y registro de descargas con SHA-256.

Cada archivo crudo `X` tiene un archivo hermano `X.meta.json` con la URL, el
metodo, los parametros POST, la fecha de consulta y el SHA-256. El manifiesto
(`reportes/manifiesto_crudos.csv`) se regenera a partir de esos archivos, de
modo que una descarga interrumpida se puede reanudar sin perder el registro.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config" / "fuentes.yaml").read_text(encoding="utf-8"))
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
REPORTES = RAIZ / "reportes"
H_AJAX = {"Content-Type": "application/x-www-form-urlencoded", "X-Requested-With": "XMLHttpRequest"}


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# --------------------------------------------------------------------------
class Cliente:
    """Cliente HTTP con pausa minima, reintentos con espera creciente y robots.txt."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or CFG["cliente_http"]
        self.s = requests.Session()
        self.s.headers["User-Agent"] = self.cfg["user_agent"]
        self._ultima = 0.0
        self._robots: dict[str, list[str] | None] = {}
        self.n_peticiones = 0

    def _esperar(self):
        falta = self.cfg["espera_min_s"] - (time.monotonic() - self._ultima)
        if falta > 0:
            time.sleep(falta)
        self._ultima = time.monotonic()

    def _reglas(self, url: str):
        p = urllib.parse.urlsplit(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            self._esperar()
            self.n_peticiones += 1
            try:
                r = self.s.get(host + "/robots.txt", timeout=self.cfg["timeout_s"])
                reglas, aplica = [], False
                if r.status_code == 200:
                    for linea in r.text.splitlines():
                        linea = linea.split("#")[0].strip()
                        if ":" not in linea:
                            continue
                        k, v = (x.strip() for x in linea.split(":", 1))
                        if k.lower() == "user-agent":
                            aplica = v == "*"
                        elif k.lower() == "disallow" and aplica and v:
                            reglas.append(v)
                self._robots[host] = reglas or None
            except requests.RequestException:
                self._robots[host] = None
        return self._robots[host]

    def permitido(self, url: str) -> bool:
        reglas = self._reglas(url)
        if not reglas:
            return True
        p = urllib.parse.urlsplit(url)
        ruta = p.path + (("?" + p.query) if p.query else "")
        return not any(re.match("^" + re.escape(r).replace(r"\*", ".*"), ruta) for r in reglas)

    def pedir(self, metodo: str, url: str, **kw) -> requests.Response:
        if not self.permitido(url):
            raise PermissionError(f"robots.txt no permite: {url}")
        espera = self.cfg["espera_reintento_s"]
        for intento in range(1, self.cfg["reintentos"] + 1):
            self._esperar()
            self.n_peticiones += 1
            try:
                r = self.s.request(metodo, url, timeout=self.cfg["timeout_s"], **kw)
                if r.status_code in (401, 403):
                    raise PermissionError(f"{r.status_code} (bloqueo o login) en {url}: se detiene")
                if r.status_code < 500 and r.status_code != 429:
                    return r
                print(f"  ! HTTP {r.status_code} en {url} (intento {intento})")
            except requests.RequestException as e:
                print(f"  ! {e.__class__.__name__} en {url} (intento {intento})")
            time.sleep(espera)
            espera *= 2
        raise RuntimeError(f"fallo tras {self.cfg['reintentos']} intentos: {url}")

    def descargar(self, destino: Path, metodo: str, url: str, fuente: str, validar=None, **kw) -> bytes:
        """Descarga reanudable: si `destino` y su .meta.json existen, no se pide de nuevo.

        `validar(bytes) -> None` puede lanzar ValueError para rechazar una respuesta
        (p. ej. HTML de error); en ese caso no se guarda nada.
        """
        meta = destino.with_name(destino.name + ".meta.json")
        if destino.exists() and meta.exists():
            return destino.read_bytes()
        destino.parent.mkdir(parents=True, exist_ok=True)
        r = self.pedir(metodo, url, **kw)
        r.raise_for_status()
        contenido = r.content
        if validar:
            validar(contenido)
        tmp = destino.with_name(destino.name + ".part")
        tmp.write_bytes(contenido)
        tmp.replace(destino)
        meta.write_text(json.dumps({
            "fuente": fuente,
            "archivo": destino.relative_to(RAIZ).as_posix(),
            "metodo": metodo,
            "url": url,
            "parametros_post": kw.get("data", "") if metodo == "POST" else "",
            "fecha_consulta": dt.datetime.now().isoformat(timespec="seconds"),
            "http_status": r.status_code,
            "bytes": len(contenido),
            "sha256": hashlib.sha256(contenido).hexdigest(),
        }, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        return contenido

    def iniciar_sesion_sisap(self):
        if not self.s.cookies:
            self.pedir("GET", CFG["sisap"]["base"])


def escribir_manifiesto(carpetas: list[Path], salida: Path) -> int:
    """Regenera el manifiesto a partir de los .meta.json y verifica cada hash."""
    filas = []
    for c in carpetas:
        for m in sorted(c.rglob("*.meta.json")):
            d = json.loads(m.read_text(encoding="utf-8"))
            ruta = RAIZ / d["archivo"]
            d["sha256_verificado"] = (ruta.exists() and sha256_archivo(ruta) == d["sha256"])
            filas.append(d)
    campos = ["fuente", "archivo", "metodo", "url", "parametros_post", "fecha_consulta",
              "http_status", "bytes", "sha256", "sha256_verificado"]
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(filas)
    return len(filas)


# --------------------------------------------------------------------------
def _q(pares) -> str:
    return "&".join(f"{k}={urllib.parse.quote(v, safe='')}" for k, v in pares)


def form_consulta(mercado: str, productos: list[str], periodicidad: str,
                  desde: str | None = None, hasta: str | None = None,
                  anios: tuple[int, int] | None = None) -> str:
    """Cuerpo POST de `resumenes/filtrar` (mismos campos que envia el JavaScript del portal)."""
    sis = CFG["sisap"]
    hoy = dt.date.today().strftime("%d/%m/%Y")
    pares = [("mercado", mercado)]
    pares += [("variables[]", v) for v in sis["variables"]]
    pares += [("procedencias[]", p) for p in sis["procedencias"]]
    pares += [("fecha", hasta or hoy), ("desde", desde or hoy), ("hasta", hasta or hoy)]
    if periodicidad == "mensual":
        pares += [("anios[]", str(a)) for a in range(anios[0], anios[1] + 1)]
        pares += [("meses[]", f"{m:02d}") for m in range(1, 13)]
    pares += [("productos[]", p) for p in productos]
    pares += [("periodicidad", periodicidad), ("__ajax_carga_final", "consulta"), ("ajax", "true")]
    return _q(pares)


SIN_RESULTADOS = "No existen resultados"


def validar_respuesta_sisap(b: bytes) -> None:
    t = b.decode("latin-1")
    if "<table" not in t and SIN_RESULTADOS not in t:
        raise ValueError("respuesta SISAP inesperada (ni tabla ni 'No existen resultados'): " + t[:200])


def parsear_tabla_sisap(texto: str) -> tuple[list[dict], dict]:
    """Tabla HTML del SISAP -> filas largas (periodo, producto, variable, procedencia, valor, crudo).

    Encabezado de 3 filas: producto (colspan) / variable (rowspan=2, o colspan
    para Volumen) / procedencia (Total + provincias). '...' = celda vacia.
    """
    soup = BeautifulSoup(texto, "html.parser")
    meta = {"titulos": [h.get_text(" ", strip=True) for h in soup.find_all("h1")]}
    tabla = soup.find("table")
    if tabla is None:
        meta["sin_tabla"] = soup.get_text(" ", strip=True)[:300]
        return [], meta
    enc = [tr for tr in tabla.find_all("tr") if "encabezado" in (tr.get("class") or [])]
    filas = [tr for tr in tabla.find_all("tr") if "contenido" in (tr.get("class") or [])]
    productos = []
    for td in enc[0].find_all("td")[1:]:
        productos += [td.get_text(strip=True)] * int(td.get("colspan", 1))
    variables = []
    for td in enc[1].find_all("td"):
        nombre = html.unescape(str(td.contents[0])).strip() if td.contents else ""
        variables += [nombre] * int(td.get("colspan", 1))
    procs = [td.get_text(strip=True) for td in enc[2].find_all("td")] if len(enc) > 2 else []
    columnas, ip = [], 0
    for prod, var in zip(productos, variables):
        if var.lower().startswith("volumen"):
            columnas.append((prod, "Volumen", procs[ip])); ip += 1
        else:
            columnas.append((prod, var, ""))
    if len(columnas) != len(productos) or ip != len(procs):
        raise ValueError("encabezado de tabla SISAP inconsistente")
    largo = []
    for tr in filas:
        celdas = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(celdas) - 1 != len(columnas):
            raise ValueError(f"fila con {len(celdas)-1} celdas, se esperaban {len(columnas)}: {celdas[:3]}")
        for (prod, var, proc), val in zip(columnas, celdas[1:]):
            largo.append({"periodo": celdas[0], "producto": prod, "variable": var, "procedencia": proc,
                          "crudo": val,
                          "valor": None if val in ("...", "") else float(val.replace(",", ""))})
    return largo, meta
