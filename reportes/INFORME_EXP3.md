# Experimento 3: ¿ayudan las variables exógenas a la rama B con datos alineados? (ablación)

> **PRELIMINAR (sin auditoría independiente).**

- **Anexo:** `docs/ANEXO_EXP3.md`, congelado en 512b7cc antes de entrenar.
- **Verificación antes de entrenar:** `scripts/08_exp3.py --verificar` dio «Anexo en HEAD y hashes OK». Los SHA-256 de `docs/hashes_anexo_exp3.json` coinciden con el anexo, las exógenas, los pesos y el código.
- **Pruebas antes de entrenar:** 37 passed (20 del experimento 2 y 17 de `tests/test_exogenas.py`).

Toda cifra sale del CSV citado.

---

## 1. Resultado: etiqueta por variante y brazo (anexo §5, aplicación literal)

ΔMAE = MAE(Vk) − MAE(V0), con IC 95 % por bootstrap pareado estratificado por año (B = 10 000, semilla 42). p es el de DM-HLN bilateral; p de Holm corrige sobre las 4 comparaciones del brazo. El MAE es el del pool de prueba 2021–2025, con el error promediado entre las 10 semillas por fecha.

**Brazo D** (diario, h = 1, n = 1 818 días). Fuente: `resultados/exp3/D/delta.csv` y `metricas.csv`.

| Variante | MAE (t) [IC 95 %] | ΔMAE (t) [IC 95 %] | p DM-HLN | p de Holm | Etiqueta |
|---|---|---|---|---|---|
| V0 base | 121.4 [116.1; 126.8] | — | — | — | — |
| V1 +clima | 128.7 [123.3; 134.2] | +7.29 [5.35; 9.22] | 1.8e-12 | 7.3e-12 | **perjudica** |
| V2 +INDECI | 127.2 [121.8; 132.6] | +5.77 [4.14; 7.43] | 7.4e-12 | 2.2e-11 | **perjudica** |
| V3 +noticias | 120.3 [114.9; 125.7] | −1.08 [−2.23; 0.11] | 0.073 | 0.073 | **sin efecto** |
| V4 todas | 127.9 [122.5; 133.5] | +6.53 [4.59; 8.52] | 5.0e-10 | 1.0e-09 | **perjudica** |

**Brazo M** (mensual, h = 1, n = 59 meses). Fuente: `resultados/exp3/M/delta.csv` y `metricas.csv`.

| Variante | MAE (t) [IC 95 %] | ΔMAE (t) [IC 95 %] | p DM-HLN | p de Holm | Etiqueta |
|---|---|---|---|---|---|
| V0 base | 1 746.9 [1 447.3; 2 068.2] | — | — | — | — |
| V1 +clima | 1 528.0 [1 267.7; 1 810.0] | −218.9 [−310.1; −128.5] | 0.00018 | 0.00072 | **ayuda** |
| V2 +INDECI | 1 736.7 [1 448.8; 2 045.8] | −10.2 [−123.9; 108.7] | 0.90 | 0.90 | **sin efecto** |
| V3 +noticias | 1 842.4 [1 536.3; 2 168.1] | +95.5 [38.7; 151.5] | 0.040 | 0.12 | **sin efecto** |
| V4 todas | 1 680.3 [1 403.7; 1 980.8] | −66.6 [−139.1; 12.3] | 0.12 | 0.24 | **sin efecto** |

En M, V3 tiene un IC que excluye 0 y p sin corregir < 0.05, pero con Holm p = 0.12. Por la regla del anexo es «sin efecto».

### Referencias para situar los resultados (no entran en la regla; anexo §4)

Mismas fechas de prueba. Fuente: `resultados/exp3/{D,M}/metricas.csv`.

| | D | M |
|---|---|---|
| Mejor modelo univariado del día 3 con N = máx (historia desde 1997) | SARIMA 89.3 | ETS 972.5 |
| DualLSTM del día 3 con N = máx (rama B = V0, historia desde 1997) | 121.6 | 1 333.1 |
| Mejor variante del experimento 3 (historia desde 2016) | V3 120.3 | V1 1 528.0 |

### Lectura

- **D:**
  - el clima, INDECI y todas las exógenas juntas **empeoran** la DualLSTM entre 5.8 y 7.3 t de forma significativa;
  - las noticias no cambian el error de forma significativa (−1.1 t, IC que incluye 0);
  - V0 con historia desde 2016 tiene un MAE de 121.4 t, casi igual que la DualLSTM del día 3 con historia desde 1997 (121.6 t);
  - todas las variantes quedan muy por encima de SARIMA (89.3 t).
- **M:**
  - el clima **ayuda**: −218.9 t frente a V0. Las demás variantes, sin efecto.
  - Pero con solo 2016–2025 la red mensual entrena con muy pocos ejemplos: de 36 a 84 objetivos (`resultados/exp3/M/estados_redes.csv`). La V0 de este experimento (1 746.9 t) es bastante peor que la DualLSTM del día 3 con historia desde 1997 (1 333.1 t).
  - En el 18–38 % de las corridas mensuales la mejor época de validación es la 1, es decir, la red casi sin entrenar (tabla del §3).
  - Así, la ganancia del clima se mide sobre una base débil, y ninguna variante se acerca a ETS (972.5 t).
- **En conjunto:** con datos alineados, las exógenas no ayudan en el brazo diario. En el mensual solo el clima mejora a una V0 que, con tan poca historia, entrena mal.

---

## 2. Evidencia sin fuga

1. **Pruebas automatizadas** (`tests/test_exogenas.py`, 17 pruebas). Entre otras, comprueban que:
   - las exógenas del día (mes) t y posteriores no cambian la entrada de la red para t, y las de t − 1 sí, en D y M y para V1–V4;
   - las exógenas posteriores al fin del entrenamiento no cambian ni el escalador ni el entrenamiento;
   - todas las variantes tienen los mismos objetivos;
   - los pesos solo usan 2016–2019;
   - las noticias e INDECI diarios reproducen las series mensuales del artículo.
2. **Prueba de perturbación** (anexo §6.2; diagnóstico; corrida con `scripts/09_exp3_perturbacion.py`, ver EXP3-1).
   - **Unidad:** V4, Y = 2025, semilla 0.
   - **Perturbación:** se sustituyen al azar las exógenas, el objetivo y el precio posteriores al origen (fin de 2024).
   - Fuente: `resultados/exp3/{D,M}/perturbacion.csv`.

| Comparación (perturbada frente a original) | D | M |
|---|---|---|
| Escalador, secuencias y objetivos de entrenamiento, secuencias de validación | 0 | 0 |
| Entrada del primer paso de prueba | 0 | 0 |
| Épocas y mejor época | 0 | 0 |
| **Pronóstico del primer paso de prueba** | **0** | **0** |
| Valor observado del primer paso | 827.9 t | 5 353.6 t |
| Entrada del último paso de prueba (escala estandarizada) | 20.1 | 8.06 |

---

## 3. Entrenamiento

Fuente: `resultados/exp3/{D,M}/estados_redes.csv`.

- **Tiempo:** 500 corridas en 1 795 s de pared con 6 procesos, sin fallos (`resultados/exp3/entrenamiento.log`). La suma de los tiempos por corrida es 6 138 s en D y 1 792 s en M.
- **Tamaño:**
  - D: 1 426 a 2 887 objetivos de entrenamiento (historia de 1 461 a 2 922 días).
  - M: 36 a 84 objetivos (historia de 48 a 96 meses).
- **Parámetros:** son los del anexo:
  - D: 4 673 (V0), 5 441 (V1), 5 057 (V2), 5 249 (V3) y 6 401 (V4);
  - M: 4 545, 5 505, 5 697, 4 929 y 7 041.

| Brazo | Variante | Mejor época (mediana) | % de corridas con mejor época = 1 |
|---|---|---|---|
| D | V0 / V1 / V2 / V3 / V4 | 11.5 / 9.5 / 11 / 9 / 8 | 0 / 0 / 0 / 0 / 2 |
| M | V0 / V1 / V2 / V3 / V4 | 27.5 / 26.5 / 21.5 / 25 / 22 | 34 / 18 / 30 / 38 / 30 |

**MAE por semilla** (`resultados/exp3/{D,M}/semillas.csv`, mínimo – máximo):
- D: V0 de 113.6 a 129.7 t; V3 de 112.5 a 136.7 t.
- M: V0 de 1 550.7 a 1 875.5 t; V1 de 1 333.8 a 1 630.9 t.

**MAE por año** (`metricas.csv`): en M la ventaja de V1 se concentra en 2024 (1 121.0 frente a 1 758.1 t) y 2025 (1 348.2 frente a 1 549.3 t).

---

## 4. Figuras

En `reportes/figuras/{es,en}/exp3/`, PNG a 300 ppp y PDF. Las genera `scripts/08_exp3.py --figuras` con `.venv`.

| Figura | Contenido | Datos |
|---|---|---|
| `e1_mae_variantes` | MAE por variante con IC 95 %; líneas: referencia univariada y DualLSTM del día 3 | `resultados/exp3/{D,M}/metricas.csv` |
| `e2_delta_vs_V0` | ΔMAE frente a V0 con IC 95 %, p de Holm y etiqueta | `resultados/exp3/{D,M}/delta.csv` |
| `e3_curvas_perdida_Y2025` | Pérdida de entrenamiento y validación por época, Y = 2025, semilla de MAE mediano por variante; línea punteada = mejor época | `resultados/exp3/{D,M}/historial_Y2025.csv` |
| `e4_pronosticos_2025_V0_V4` | 2025: observado, V0 y V4 (media de semillas). Contiene la serie del SISAP: solo se versiona el PNG | `pronosticos.csv` (no versionado) y `data/processed` |

Las curvas de pérdida de `e3` muestran en M el problema del §1: en V0, V2 y V3, con la semilla mediana, la pérdida de validación es mínima en la época 1 y sube después.

---

## 5. Desviaciones

Están registradas en `docs/DESVIACIONES.md`:
- **EXP3-1:** la prueba de perturbación de `scripts/08_exp3.py` fallaba por un error de tipos y se corrió con `scripts/09_exp3_perturbacion.py`, sin modificar el script congelado. Sin efecto en las etiquetas.
- **EXP3-2:** hay solapes de maquetación en `e2` y `e3` que no se corrigieron, porque exigen cambiar el script congelado. Sin efecto en las cifras.

Además: el `.gitignore` con las reglas del experimento 3 (unidades por corrida y PDF de `e4`) no entró en el commit del anexo y se incluye abajo.

**Hashes y pruebas al final:**
- `.venv\Scripts\python scripts/04_hashes_protocolo.py --verificar` → «OK: todos los hashes coinciden»;
- `.venv311\Scripts\python -m pytest -q tests` → 37 passed;
- los hashes del anexo coinciden (`scripts/08_exp3.py --verificar`).

---

## 6. `git add` y mensaje de commit sugerido

`pronosticos.csv` y `resultados/exp3/unidades/` quedan fuera por `.gitignore`.

```
git add .gitignore docs/DESVIACIONES.md scripts/09_exp3_perturbacion.py \
  resultados/exp3/D resultados/exp3/M resultados/exp3/entrenamiento.log \
  reportes/figuras/es/exp3 reportes/figuras/en/exp3 reportes/INFORME_EXP3.md
```

Comprobación antes del commit (debe salir vacía):

```
git diff --cached --name-only | grep -E "pronosticos|unidades/|e4_.*\.pdf"
```

**Mensaje de commit sugerido:**

```
Experimento 3: ablación de exógenas en la rama B (resultados preliminares)

- 500 corridas de la DualLSTM (2 brazos x 5 variantes x 5 años x 10 semillas), anexo congelado en 512b7cc
- Regla del anexo: en D, clima, INDECI y todas perjudican y noticias sin efecto;
  en M, clima ayuda y el resto sin efecto (V0 mensual débil: 36-84 objetivos de entrenamiento)
- Prueba de perturbación sin fuga en ambos brazos (scripts/09_exp3_perturbacion.py, EXP3-1)
- Figuras e1-e4 (es/en) y reportes/INFORME_EXP3.md (PRELIMINAR, sin auditoría independiente)
```
