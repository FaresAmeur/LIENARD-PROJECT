#!/bin/bash
# Ancrage OpenTimestamps du registre — à lancer après chaque entrée
# pip install opentimestamps-client
for f in r5_log_entry*.json; do
  [ -f "$f.ots" ] || ots stamp "$f"     # crée la preuve .ots (Bitcoin, gratuit)
done
ots upgrade r5_log_entry*.json.ots 2>/dev/null   # complète les preuves en attente
ots verify r5_log_entry002_chained.json.ots      # vérification indépendante
