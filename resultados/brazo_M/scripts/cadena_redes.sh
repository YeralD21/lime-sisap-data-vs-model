#!/bin/bash
# Espera a que termine la corrida del escenario principal y lanza las redes de sens1 y sens7 (2 procesos cada vez).
cd /d/lime-sisap-data-vs-model
until grep -q "TOTAL redes" resultados/brazo_M/logs/redes_principal.log; do sleep 30; done
for e in sens1_estricto sens7_anomalia_publicada; do
  .venv311/Scripts/python.exe resultados/brazo_M/scripts/02_redes.py --escenario $e > resultados/brazo_M/logs/redes_$e.log 2>&1
done
echo cadena_fin > resultados/brazo_M/logs/cadena_fin.txt
