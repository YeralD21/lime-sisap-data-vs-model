#!/bin/bash
# Cadena de corridas de redes en el orden decidido por YesaR. 4 procesos. Reanudable.
cd /d/lime-sisap-data-vs-model
PY=.venv311/Scripts/python.exe
S=resultados/brazo_D/scripts/redes_run.py
$PY $S --escenario principal --h 1 --n_proc 4
$PY $S --escenario principal --h 7 --n_proc 4
$PY $S --escenario sens1_estricto --h 1 --n_proc 4
$PY $S --escenario sens2_bolsa --h 1 --n_proc 4
$PY $S --escenario sens6_desde2006 --h 1 --N max --n_proc 4
$PY $S --escenario sens7_anomalia_publicada --h 1 --n_proc 4
echo CADENA_COMPLETA
