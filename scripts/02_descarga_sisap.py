"""Dia 2 - descarga de la serie completa del SISAP (y PDF de control de gob.pe).

Fases (por defecto se ejecutan todas, en este orden):
  sondeo   consultas diarias de una semana (1-7 de marzo) en los anios de
           config `descarga.sondeo_anios`; si hace falta, biseccion anual para
           acotar el primer anio con datos diarios.
  diario   una consulta por anio civil (intervalo 01/01-31/12; el ultimo anio
           termina en `descarga.fecha_corte`), desde el primer anio con datos
           diarios. Limon sutil bolsa (061101) y cajon (061102), GMML.
  mensual  serie mensual 1997-anio de corte.
  pdf      PDF diarios de gob.pe de las fechas en `descarga.control_pdf`
           (para investigar el significado de '...').

Reanudable: cada archivo crudo tiene un .meta.json; lo ya descargado no se
vuelve a pedir. Al final regenera reportes/manifiesto_crudos.csv (verificando
cada SHA-256) y reportes/dia2_sondeo.json.

Uso:  python scripts/02_descarga_sisap.py [--fase sondeo diario mensual pdf]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.parse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from sisap_comun import (CFG, H_AJAX, RAW, REPORTES, SIN_RESULTADOS, Cliente,  # noqa: E402
                         escribir_manifiesto, form_consulta, parsear_tabla_sisap,
                         validar_respuesta_sisap)

SIS = CFG["sisap"]
DES = CFG["descarga"]
URL_CONSULTA = SIS["base"] + SIS["endpoints"]["consulta"]
GMML = SIS["mercados"]["GMML"]
PRODUCTOS = DES["productos"]
D_SONDEO = RAW / "sisap_dia2" / "sondeo"
D_DIARIO = RAW / "sisap_dia2" / "diario"
D_MENSUAL = RAW / "sisap_dia2" / "mensual"
D_PDF = RAW / "gobpe_dia2"


def corte() -> dt.date:
    return dt.datetime.strptime(DES["fecha_corte"], "%Y-%m-%d").date()


def consulta_intervalo(cli: Cliente, destino, d0: dt.date, d1: dt.date) -> list[dict]:
    datos = form_consulta(GMML, PRODUCTOS, "intervalo", d0.strftime("%d/%m/%Y"), d1.strftime("%d/%m/%Y"))
    b = cli.descargar(destino, "POST", URL_CONSULTA, "sisap", validar=validar_respuesta_sisap,
                      data=datos, headers=H_AJAX)
    t = b.decode("latin-1")
    if SIN_RESULTADOS in t:
        return []
    return parsear_tabla_sisap(t)[0]


def resumen_semana(filas: list[dict]) -> dict:
    vol = sorted({f["periodo"] for f in filas if f["variable"] == "Volumen" and f["procedencia"] == "Total"
                  and f["valor"] is not None}, key=lambda p: dt.datetime.strptime(p, "%d/%m/%Y"))
    pre = sorted({f["periodo"] for f in filas if f["variable"] == "Precio Promedio" and f["valor"] is not None},
                 key=lambda p: dt.datetime.strptime(p, "%d/%m/%Y"))
    return {"n_dias_en_tabla": len({f["periodo"] for f in filas}), "dias_con_volumen": vol,
            "dias_con_precio": pre, "productos": sorted({f["producto"] for f in filas})}


def fase_sondeo(cli: Cliente) -> dict:
    res = {}

    def sondear(anio: int) -> bool:
        d0, d1 = dt.date(anio, 3, 1), dt.date(anio, 3, 7)
        filas = consulta_intervalo(cli, D_SONDEO / f"semana_{anio}_03_01-07.html", d0, d1)
        res[anio] = resumen_semana(filas)
        print(f"  sondeo {anio}: {res[anio]['n_dias_en_tabla']} dias en tabla, "
              f"{len(res[anio]['dias_con_volumen'])} con volumen")
        return res[anio]["n_dias_en_tabla"] > 0

    hay = {a: sondear(a) for a in DES["sondeo_anios"]}
    # Biseccion: entre el ultimo anio sondeado sin datos y el primero con datos.
    con = [a for a, v in hay.items() if v]
    sin = [a for a, v in hay.items() if not v and (not con or a < min(con))]
    if con and sin:
        lo, hi = max(sin), min(con)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if sondear(mid):
                hi = mid
            else:
                lo = mid
    primer = min(a for a, r in res.items() if r["n_dias_en_tabla"] > 0) if any(
        r["n_dias_en_tabla"] for r in res.values()) else None
    return {"semanas": res, "primer_anio_sondeado_con_datos_diarios": primer}


def fase_diario(cli: Cliente, primer_anio: int) -> list[str]:
    hechos = []
    fc = corte()
    # Se incluye el anio anterior al primero detectado: el sondeo mira solo marzo y los
    # datos diarios podrian empezar despues de marzo de ese anio anterior.
    for anio in range(max(1997, primer_anio - 1), fc.year + 1):
        d0, d1 = dt.date(anio, 1, 1), min(dt.date(anio, 12, 31), fc)
        destino = D_DIARIO / f"diario_{anio}.html"
        filas = consulta_intervalo(cli, destino, d0, d1)
        print(f"  diario {anio}: {len({f['periodo'] for f in filas})} dias")
        hechos.append(destino.name)
    return hechos


def fase_mensual(cli: Cliente):
    fc = corte()
    datos = form_consulta(GMML, PRODUCTOS, "mensual", anios=(1997, fc.year))
    b = cli.descargar(D_MENSUAL / f"mensual_1997_{fc.year}.html", "POST", URL_CONSULTA, "sisap",
                      validar=validar_respuesta_sisap, data=datos, headers=H_AJAX)
    filas, _ = parsear_tabla_sisap(b.decode("latin-1"))
    print(f"  mensual: {len({f['periodo'] for f in filas})} meses")


MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
            "setiembre", "octubre", "noviembre", "diciembre"]


def fase_pdf(cli: Cliente):
    """Descarga el PDF del GMML de cada fecha de control (busca su pagina mensual en la coleccion)."""
    col_url = CFG["gobpe"]["colecciones"]["GMML"]
    col = cli.descargar(D_PDF / "coleccion_GMML.html", "GET", col_url, "gobpe").decode("utf-8", "replace")
    paginas = sorted(set(re.findall(r"/institucion/midagri/informes-publicaciones/(\d+-[a-z0-9-]+)", col)))
    for f in DES["control_pdf"]:
        fecha = dt.datetime.strptime(f, "%Y-%m-%d").date()
        sufijo = f"{MESES_ES[fecha.month - 1]}-{fecha.year}"
        cand = [p for p in paginas if p.endswith(sufijo) or p.endswith(sufijo.replace("setiembre", "septiembre"))]
        if not cand:
            print(f"  pdf {f}: pagina mensual no listada en la primera hoja de la coleccion (no se pagina)")
            continue
        url_mes = "https://www.gob.pe/institucion/midagri/informes-publicaciones/" + cand[0]
        mes = cli.descargar(D_PDF / f"mes_{fecha:%Y_%m}.html", "GET", url_mes, "gobpe").decode("utf-8", "replace")
        pdfs = [u for u in dict.fromkeys(re.findall(
            r'https://cdn\.www\.gob\.pe/uploads/document/file/\d+/[^"?]+\.pdf(?:\?v=\d+)?', mes)) if "/preview_" not in u]
        patrones = [f"{fecha:%d-%m-%Y}", f"{fecha:%d-%m-%y}", f"{fecha:%d/%m/%y}"]
        elegido = [u for u in pdfs if any(p in urllib.parse.unquote(u) for p in patrones)]
        if not elegido:
            print(f"  pdf {f}: no hay PDF publicado para esa fecha en la pagina mensual")
            continue
        doc = re.search(r"/document/file/(\d+)/", elegido[0]).group(1)
        cli.descargar(D_PDF / f"pdf_gmml_{fecha:%Y-%m-%d}_doc{doc}.pdf", "GET", elegido[0], "gobpe")
        print(f"  pdf {f}: ok (doc {doc})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", nargs="*", default=["sondeo", "diario", "mensual", "pdf"])
    args = ap.parse_args()
    cli = Cliente()
    cli.iniciar_sesion_sisap()
    sondeo_json = REPORTES / "dia2_sondeo.json"
    sondeo = json.loads(sondeo_json.read_text(encoding="utf-8")) if sondeo_json.exists() else None
    if "sondeo" in args.fase:
        print("Fase sondeo"); sondeo = fase_sondeo(cli)
        sondeo_json.write_text(json.dumps(sondeo, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    if "diario" in args.fase:
        if not sondeo or not sondeo.get("primer_anio_sondeado_con_datos_diarios"):
            raise SystemExit("Primero hay que ejecutar la fase sondeo.")
        print("Fase diario"); fase_diario(cli, int(sondeo["primer_anio_sondeado_con_datos_diarios"]))
    if "mensual" in args.fase:
        print("Fase mensual"); fase_mensual(cli)
    if "pdf" in args.fase:
        print("Fase pdf"); fase_pdf(cli)
    n = escribir_manifiesto([RAW / "sisap_dia2", D_PDF], REPORTES / "manifiesto_crudos.csv")
    print(f"OK: {cli.n_peticiones} peticiones en esta ejecucion; manifiesto con {n} archivos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
