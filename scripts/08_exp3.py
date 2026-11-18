"""Experimento 3, fase B (docs/ANEXO_EXP3.md): entrenamiento, regla de decisión, perturbación y figuras.

Pasos (en este orden):
  --verificar     comprueba que el anexo está en HEAD sin cambios y que los SHA-256 coinciden con
                  docs/hashes_anexo_exp3.json (los demás pasos lo exigen)
  --entrenar      500 corridas (2 brazos x 5 variantes x 5 años x 10 semillas), paralelas y reanudables
  --analizar      MAE, ΔMAE con IC bootstrap, DM-HLN, Holm y etiqueta (anexo §5)
  --perturbacion  prueba de perturbación de exógenas y objetivo futuros (anexo §6.2; diagnóstico)
  --figuras       figuras del anexo §7 (--idioma en|es)
Uso:
  .venv311\\Scripts\\python scripts/08_exp3.py --entrenar --n-proc 6
  .venv311\\Scripts\\python scripts/08_exp3.py --analizar --perturbacion
  .venv\\Scripts\\python scripts/08_exp3.py --figuras --idioma es   (y en; matplotlib solo está en .venv)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src import config as C  # noqa: E402
from src import evaluacion as E  # noqa: E402
from src import exogenas as X3  # noqa: E402
from src.analisis import fechas_evaluables  # noqa: E402

SAL = RAIZ / "resultados" / "exp3"
UNID = SAL / "unidades"
ANEXO = RAIZ / "docs" / "ANEXO_EXP3.md"
HASHES = RAIZ / "docs" / "hashes_anexo_exp3.json"
VARS = list(X3.VARIANTES)
REF_DIA3 = {"D": "SARIMA", "M": "ETS"}


def sha256_lf(ruta: Path) -> str:
    b = Path(ruta).read_bytes()
    if Path(ruta).suffix.lower() in {".md", ".csv", ".py", ".json", ".txt"}:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def verificar() -> None:
    h = json.loads(HASHES.read_text(encoding="utf-8"))
    for ruta, info in h["archivos"].items():
        if sha256_lf(RAIZ / ruta) != info["sha256"]:
            raise SystemExit(f"SHA-256 distinto: {ruta}")
    r = subprocess.run(["git", "ls-files", "--error-unmatch", "docs/ANEXO_EXP3.md"], cwd=RAIZ, capture_output=True)
    d = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "docs/ANEXO_EXP3.md", "docs/hashes_anexo_exp3.json"],
                       cwd=RAIZ)
    if r.returncode or d.returncode:
        raise SystemExit("El anexo no está en HEAD o tiene cambios sin commit: no se entrena.")
    print("Anexo en HEAD y hashes OK.")


def unidades(brazos=("D", "M")) -> list[dict]:
    return [{"escenario": {"brazo": b, "objetivo": "principal", "inicio": C.INICIO, "anomalia": "faltante"},
             "anio": a, "N": "max", "h": 1, "modelo": "DualLSTM", "semilla": s, "variante": v}
            for b in brazos for v in VARS for a in C.ANIOS_TEST for s in C.SEMILLAS]


def entrenar(n_proc: int) -> None:
    from src.redes import correr_unidades
    SAL.mkdir(parents=True, exist_ok=True)
    log = open(SAL / "entrenamiento.log", "a", encoding="utf-8")

    def escribir(m):
        print(m)
        log.write(m + "\n")
        log.flush()
    import time
    t0 = time.time()
    res = correr_unidades(unidades(), UNID, n_proc=n_proc, log=escribir)
    fallos = sum("error" in r for r in res)
    escribir(f"TOTAL exp3: {time.time() - t0:.0f} s de pared; {len(res)} unidades; {fallos} fallos")


# ----------------------------------------------------------------------------- lectura
def leer(brazo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred, est = [], []
    for f in sorted((UNID / f"exp3_{brazo}").glob("V*/Y*/s*/estado.json")):
        e = json.loads(f.read_text(encoding="utf-8"))
        p = pd.read_csv(f.parent / "predicciones.csv")
        p = p[p.particion == "test"].assign(variante=e["variante"], semilla=e["semilla"], anio=e["anio"])
        pred.append(p[["variante", "semilla", "anio", "fecha", "y_pred"]])
        est.append({k: e[k] for k in ("variante", "anio", "semilla", "n_params", "n_b", "n_train", "n_val",
                                      "n_test", "n_hist", "epocas", "mejor_epoca", "segundos")}
                   | {"escalador_desde": e["escalador_filas"][0], "escalador_hasta": e["escalador_filas"][1]})
    if len(est) != len(VARS) * len(C.ANIOS_TEST) * len(C.SEMILLAS):
        raise SystemExit(f"Brazo {brazo}: {len(est)} corridas; faltan unidades.")
    return pd.concat(pred, ignore_index=True), pd.DataFrame(est)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    orden = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    acum = 0.0
    for k, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - k) * p[i]))
        adj[i] = acum
    return adj


def etiqueta(delta, lo, hi, p_holm) -> str:
    if delta < 0 and p_holm < 0.05 and hi < 0:
        return "ayuda"
    if delta > 0 and p_holm < 0.05 and lo > 0:
        return "perjudica"
    return "sin efecto"


def analizar() -> None:
    for b in "DM":
        df = X3.serie_exp3(b)
        pron, est = leer(b)
        ref = fechas_evaluables(df, list(C.ANIOS_TEST))
        yt = ref.set_index("fecha").y_true
        pron["y_true"] = pron.fecha.map(yt)
        d = SAL / b
        d.mkdir(parents=True, exist_ok=True)
        pron.drop(columns="y_true").to_csv(d / "pronosticos.csv", index=False, float_format="%.6g")  # no versionado
        est.to_csv(d / "estados_redes.csv", index=False)
        ev = pron.dropna(subset=["y_true"]).assign(ae=lambda x: (x.y_pred - x.y_true).abs())
        AE = np.vstack([ev[ev.variante == v].groupby("fecha").ae.mean().reindex(ref.fecha).to_numpy() for v in VARS])
        if np.isnan(AE).any():
            raise SystemExit(f"Brazo {b}: faltan pronósticos en fechas evaluables.")
        estr = ref.anio.to_numpy()
        boot = E.remuestreos(AE, estr)
        met = []
        for i, v in enumerate(VARS):
            lo, hi = E.ic(boot[i])
            met.append({"variante": v, "anio": "pool", "MAE": AE[i].mean(), "MAE_ic_lo": lo, "MAE_ic_hi": hi,
                        "n": AE.shape[1]})
            for a in C.ANIOS_TEST:
                k = estr == a
                if k.any():
                    met.append({"variante": v, "anio": a, "MAE": AE[i, k].mean(), "n": int(k.sum())})
        # referencias del día 3 (mismas fechas; solo para situar)
        p3 = pd.read_csv(RAIZ / "resultados" / f"brazo_{b}" / "principal" / "pronosticos.csv", dtype={"N": str},
                         low_memory=False)
        p3 = p3[(p3.N == "max") & (p3.h == 1)]
        for mod, rol in ((REF_DIA3[b], "referencia_dia3_ref*"), ("DualLSTM", "DualLSTM_dia3_Nmax")):
            s = p3[p3.modelo == mod].assign(ae=lambda x: (x.y_pred - x.fecha.map(yt)).abs())
            s = s.groupby("fecha").ae.mean().reindex(ref.fecha).to_numpy()     # AE promediado entre semillas (DA8)
            met.append({"variante": f"{rol} ({mod})", "anio": "pool", "MAE": float(np.mean(s)), "n": AE.shape[1]})
        pd.DataFrame(met).to_csv(d / "metricas.csv", index=False, float_format="%.6g")
        filas = []
        for k in range(1, len(VARS)):
            lo, hi = E.ic(boot[k] - boot[0])
            dm = E.dm_hln(AE[k], AE[0], 1)
            p = float(2 * stats.t.sf(abs(dm["HLN"]), df=dm["n"] - 1))
            filas.append({"variante": VARS[k], "MAE": AE[k].mean(), "MAE_V0": AE[0].mean(),
                          "delta": AE[k].mean() - AE[0].mean(), "ic_lo": lo, "ic_hi": hi, "HLN": dm["HLN"], "p": p,
                          "n": dm["n"]})
        t = pd.DataFrame(filas)
        t["p_holm"] = holm(t.p.to_numpy())
        t["etiqueta"] = [etiqueta(r.delta, r.ic_lo, r.ic_hi, r.p_holm) for r in t.itertuples()]
        t.to_csv(d / "delta.csv", index=False, float_format="%.6g")
        sem = (ev.groupby(["variante", "semilla"]).ae.mean().rename("MAE").reset_index())
        sem.to_csv(d / "semillas.csv", index=False, float_format="%.6g")
        # historial Y = 2025 (pérdidas en escala estandarizada)
        hs = []
        for f in sorted((UNID / f"exp3_{b}").glob("V*/Y2025/s*/estado.json")):
            e = json.loads(f.read_text(encoding="utf-8"))
            for ep, (lo_, vl_) in enumerate(zip(e["historial"]["loss"], e["historial"]["val_loss"]), 1):
                hs.append({"variante": e["variante"], "semilla": e["semilla"], "epoca": ep, "loss": lo_,
                           "val_loss": vl_, "mejor_epoca": e["mejor_epoca"]})
        pd.DataFrame(hs).to_csv(d / "historial_Y2025.csv", index=False, float_format="%.6g")
        print(b)
        print(t.to_string())


# ----------------------------------------------------------------------------- perturbación (diagnóstico)
def _perturbar_serie(df, origen):
    rng = np.random.default_rng(20250101)
    out = df.copy()
    n = len(out) - origen - 1
    for c in ("y", "y_eval", "precio"):
        v = out[c].to_numpy(float)
        nuevo = rng.uniform(np.nanmin(v), np.nanmax(v), n)
        if c == "y_eval":
            nuevo = np.where(np.isnan(v[origen + 1:]), np.nan, nuevo)
        out.iloc[origen + 1:, out.columns.get_loc(c)] = nuevo
    return out


def _perturbar_exog(ex, fecha_origen):
    rng = np.random.default_rng(20250102)
    out = ex.copy()
    k = out.index > fecha_origen
    for c in out.columns:
        v = out[c].to_numpy(float)
        out.loc[k, c] = rng.uniform(np.nanmin(v), np.nanmax(v), int(k.sum()))
    if "sin_noticias" in out:
        out.loc[k, "n_noticias"] = np.round(out.loc[k, "n_noticias"])
        out["sin_noticias"] = (out.n_noticias == 0).astype(int)
        out.loc[k, "sent"] = out.loc[k, "sent"].where(out.loc[k, "n_noticias"] > 0)
    return out


def _datos(brazo, perturbada):
    from src.ventanas import Ventana, particion
    df, ex = X3.serie_exp3(brazo), X3.cargar(brazo)
    part = particion(df.index, Ventana(2025), "max")
    if perturbada:
        df = _perturbar_serie(df, int(part.val[-1]))
        ex = _perturbar_exog(ex, df.index[int(part.val[-1])])
    return df, ex, part


def hijo(unidad_json, perturbada, destino):
    from src import redes as R
    u = json.loads(Path(unidad_json).read_text(encoding="utf-8"))
    df, ex, _ = _datos(u["escenario"]["brazo"], perturbada == "1")
    X3.serie_exp3 = lambda brazo, _d=df: _d
    X3.cargar = lambda brazo, _e=ex: _e
    R.entrenar_y_predecir(u, Path(destino))


def perturbacion() -> None:
    from src import redes as R
    for b in "DM":
        u = [x for x in unidades((b,)) if x["variante"] == "V4" and x["anio"] == 2025 and x["semilla"] == 0][0]
        fam = X3.VARIANTES["V4"]
        marg = X3.MARGEN_FILAS[b]
        res = {}
        for etq in ("0", "1"):
            df, ex, part = _datos(b, etq == "1")
            res[etq] = (df, R.preparar(df, b, "DualLSTM", part, 1, X3.matriz_B(df.index, b, fam, ex), marg))
        (df0, (X0, ys0, pos, ms0, _, _)), (df1, (X1, ys1, _, ms1, _, _)) = res["0"], res["1"]
        t0 = int(pos["test"][0])
        pred = {}
        with tempfile.TemporaryDirectory() as tmp:
            uj = Path(tmp) / "u.json"
            uj.write_text(json.dumps(u), encoding="utf-8")
            for etq in ("0", "1"):
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--hijo", str(uj), etq,
                                str(Path(tmp) / etq)], check=True, cwd=RAIZ, env={**os.environ, **R.VARIABLES_ENTORNO})
                pred[etq] = (pd.read_csv(Path(tmp) / etq / "predicciones.csv"),
                             json.loads((Path(tmp) / etq / "estado.json").read_text(encoding="utf-8")))
        f0 = df0.index[t0].strftime("%Y-%m-%d")

        def p1(t):
            return float(t.loc[(t.particion == "test") & (t.fecha == f0), "y_pred"].iloc[0])

        def dif(a, c):
            return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(c, float))))
        comp = {"escalador_y": dif(ms0, ms1),
                "secuencias_entrenamiento": max(dif(a, c) for a, c in zip(X0["train"], X1["train"])),
                "objetivos_entrenamiento": dif(ys0[pos["train"]], ys1[pos["train"]]),
                "secuencias_validacion": max(dif(a, c) for a, c in zip(X0["val"], X1["val"])),
                "entrada_primer_paso_prueba": max(dif(a[0], c[0]) for a, c in zip(X0["test"], X1["test"])),
                "epocas": abs(pred["1"][1]["epocas"] - pred["0"][1]["epocas"]),
                "mejor_epoca": abs(pred["1"][1]["mejor_epoca"] - pred["0"][1]["mejor_epoca"]),
                "pronostico_primer_paso_prueba": abs(p1(pred["1"][0]) - p1(pred["0"][0])),
                "observado_primer_paso_prueba": abs(float(df1.y_eval.iloc[t0] - df0.y_eval.iloc[t0])),
                "entrada_ultimo_paso_prueba": max(dif(a[-1], c[-1]) for a, c in zip(X0["test"], X1["test"]))}
        (SAL / b).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"brazo": b, "comparacion": k, "dif_max_abs": v} for k, v in comp.items()]).to_csv(
            SAL / b / "perturbacion.csv", index=False, float_format="%.6g")
        print(b, comp)


# ----------------------------------------------------------------------------- figuras (anexo §7)
COLOR_V = {"V0": "#999999", "V1": "#0072B2", "V2": "#009E73", "V3": "#CC79A7", "V4": "#D55E00"}
TXT = {"es": {"brazo_D": "Brazo D (diario, h = 1)", "brazo_M": "Brazo M (mensual, h = 1)", "mae": "MAE (t)",
              "delta": "ΔMAE frente a V0 (t)", "epoca": "Época", "perdida": "Pérdida (MSE, escala estandarizada)",
              "train": "entrenamiento", "val": "validación", "obs": "Observado", "ref": "Referencia del día 3",
              "dual3": "DualLSTM del día 3 (N = máx)", "mejor": "mejor época",
              "V": {"V0": "V0 base", "V1": "V1 +clima", "V2": "V2 +INDECI", "V3": "V3 +noticias", "V4": "V4 todas"},
              "et": {"ayuda": "ayuda", "perjudica": "perjudica", "sin efecto": "sin efecto"}},
       "en": {"brazo_D": "Arm D (daily, h = 1)", "brazo_M": "Arm M (monthly, h = 1)", "mae": "MAE (t)",
              "delta": "ΔMAE vs V0 (t)", "epoca": "Epoch", "perdida": "Loss (MSE, standardised scale)",
              "train": "training", "val": "validation", "obs": "Observed", "ref": "Day-3 reference",
              "dual3": "Day-3 DualLSTM (N = max)", "mejor": "best epoch",
              "V": {"V0": "V0 base", "V1": "V1 +climate", "V2": "V2 +INDECI", "V3": "V3 +news", "V4": "V4 all"},
              "et": {"ayuda": "helps", "perjudica": "hurts", "sin efecto": "no effect"}}}


def figuras(idioma: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = TXT[idioma]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.bbox": "tight"})
    out = RAIZ / "reportes" / "figuras" / idioma / "exp3"
    out.mkdir(parents=True, exist_ok=True)

    def guardar(fig, n):
        fig.savefig(out / f"{n}.pdf")
        fig.savefig(out / f"{n}.png", dpi=300)
        plt.close(fig)
    # 1. MAE por variante
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, b in zip(axs, "DM"):
        m = pd.read_csv(SAL / b / "metricas.csv")
        s = m[(m.anio == "pool") & m.variante.isin(VARS)].set_index("variante").loc[VARS]
        x = np.arange(len(VARS))
        ax.errorbar(x, s.MAE, yerr=[s.MAE - s.MAE_ic_lo, s.MAE_ic_hi - s.MAE], fmt="none", ecolor="k", capsize=3)
        ax.scatter(x, s.MAE, c=[COLOR_V[v] for v in VARS], s=30, zorder=3)
        r = m[m.variante.str.startswith("referencia_dia3")].iloc[0]
        dl = m[m.variante.str.startswith("DualLSTM_dia3")].iloc[0]
        ax.axhline(r.MAE, color="#009E73" if b == "D" else "#0072B2", ls="--", lw=1,
                   label=f"{L['ref']}: {REF_DIA3[b]}")
        ax.axhline(dl.MAE, color="#D55E00", ls=":", lw=1, label=L["dual3"])
        ax.set_xticks(x, [L["V"][v] for v in VARS], rotation=25, ha="right")
        ax.set_ylabel(L["mae"])
        ax.set_title(L[f"brazo_{b}"])
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    guardar(fig, "e1_mae_variantes")
    # 2. ΔMAE frente a V0
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, b in zip(axs, "DM"):
        t = pd.read_csv(SAL / b / "delta.csv")
        x = np.arange(len(t))
        ax.axhline(0, color="k", lw=0.8)
        ax.errorbar(x, t.delta, yerr=[t.delta - t.ic_lo, t.ic_hi - t.delta], fmt="none", ecolor="k", capsize=3)
        ax.scatter(x, t.delta, c=[COLOR_V[v] for v in t.variante], s=30, zorder=3)
        for xi, r in zip(x, t.itertuples()):
            ax.annotate(f"p_Holm={r.p_holm:.2g}\n{L['et'][r.etiqueta]}", (xi, r.ic_hi), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=6.5)
        ax.set_xticks(x, [L["V"][v] for v in t.variante], rotation=25, ha="right")
        ax.set_ylabel(L["delta"])
        ax.set_title(L[f"brazo_{b}"])
        ax.margins(y=0.25)
    fig.tight_layout()
    guardar(fig, "e2_delta_vs_V0")
    # 3. curvas de pérdida (Y = 2025, semilla de MAE mediano: rango 5 de 10)
    fig, axs = plt.subplots(2, 5, figsize=(9.5, 4.2), sharey="row")
    for i, b in enumerate("DM"):
        h = pd.read_csv(SAL / b / "historial_Y2025.csv")
        pr = pd.read_csv(SAL / b / "pronosticos.csv")
        df = X3.serie_exp3(b)
        ref = fechas_evaluables(df, [2025])
        pr = pr[(pr.anio == 2025) & pr.fecha.isin(set(ref.fecha))]
        pr["ae"] = (pr.y_pred - pr.fecha.map(ref.set_index("fecha").y_true)).abs()
        for j, v in enumerate(VARS):
            mae = pr[pr.variante == v].groupby("semilla").ae.mean().sort_values(kind="stable")
            sem = int(mae.index[4])
            s = h[(h.variante == v) & (h.semilla == sem)]
            ax = axs[i, j]
            ax.plot(s.epoca, s.loss, color=COLOR_V[v], lw=1, label=L["train"])
            ax.plot(s.epoca, s.val_loss, color=COLOR_V[v], lw=1, ls="--", label=L["val"])
            ax.axvline(s.mejor_epoca.iloc[0], color="0.5", lw=0.7, ls=":")
            ax.set_title(f"{b} — {L['V'][v]} (s{sem})", fontsize=8)
            ax.set_xlabel(L["epoca"])
            if j == 0:
                ax.set_ylabel(L["perdida"])
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    guardar(fig, "e3_curvas_perdida_Y2025")
    # 4. pronósticos 2025, V0 frente a V4 (contiene la serie: solo el PNG se versiona)
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0))
    for ax, b in zip(axs, "DM"):
        df = X3.serie_exp3(b)
        pr = pd.read_csv(SAL / b / "pronosticos.csv")
        pr = pr[pr.anio == 2025]
        obs = df.y_eval[df.index.year == 2025]
        ax.plot(obs.index, obs.values, color="k", lw=1, label=L["obs"], marker="o" if b == "M" else None, ms=3)
        for v in ("V0", "V4"):
            s = pr[pr.variante == v].groupby("fecha").y_pred.mean()
            ax.plot(pd.to_datetime(s.index), s.values, color=COLOR_V[v], lw=0.9, label=L["V"][v])
        ax.set_title(L[f"brazo_{b}"] + " — 2025")
        ax.set_ylabel("t")
        ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    guardar(fig, "e4_pronosticos_2025_V0_V4")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo":
        hijo(*sys.argv[2:5])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--entrenar", action="store_true")
    ap.add_argument("--analizar", action="store_true")
    ap.add_argument("--perturbacion", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    ap.add_argument("--idioma", choices=("en", "es"), default="es")
    ap.add_argument("--n-proc", type=int, default=6)
    a = ap.parse_args()
    verificar()
    if a.entrenar:
        entrenar(a.n_proc)
    if a.analizar:
        analizar()
    if a.perturbacion:
        perturbacion()
    if a.figuras:
        figuras(a.idioma)
