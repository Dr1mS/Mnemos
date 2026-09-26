#!/bin/bash
# Surveillance d'un déploiement Mnemos pendant une évaluation AML.
#
#   bash scripts/aml_monitor.sh --url https://exemple.tld
#
# La clé est lue dans .env (API_KEY) ou dans la variable d'environnement
# API_KEY. Elle n'est volontairement PAS un argument : tout argument de ligne
# de commande est visible dans la liste des processus de la machine.
#
# Options : --log <fichier> --serve-log <fichier> --tours <n> --seuil-mo <Mo>
#
# ── Deux leçons apprises pendant le run Full des 22-23/09 ────────────────────
#
# 1. LE TRAFIC RÉEL ARBITRE, PAS MES SONDES.
#    Sous charge, la machine de mesure sature son propre résolveur : curl rend
#    le code 000 en ~2 s, y compris vers 127.0.0.1. Quatre fois cette nuit-là,
#    les quatre sondes ont échoué simultanément alors que la plateforme était
#    servie normalement — jusqu'à 171 requêtes dans la fenêtre d'« incident ».
#    On ne déclare donc une panne que si les sondes échouent DEUX fois ET que
#    le compteur de requêtes servies n'a pas bougé.
#
# 2. /health NE SUFFIT PAS.
#    Le 22/09, l'étape Add est restée bloquée 15 minutes alors que /health
#    répondait 200 : le serveur était sain mais n'avait plus rien à faire, ses
#    embeddings échouant en amont. Le signal utile était ailleurs — CPU à zéro,
#    GPU à 1 %. D'où le relevé systématique du GPU et du trafic entrant.

set -u

URL=""; LOG="aml_monitor.log"; SERVE_LOG="data/aml/serve.log"
TOURS=64; SEUIL_MO=2048; LLAMACPP="http://127.0.0.1:8899"; ENV_FILE=".env"
# VRAM libre en deçà de laquelle on avertit. Utile quand un autre travail
# partage la carte : le serveur d'embeddings a deja reserve la sienne, mais un
# modele charge a cote peut saturer la carte et ralentir ou faire echouer les
# embeddings. On avertit sans rien arreter — l'arbitrage revient a l'humain.
VRAM_ALERTE_MO=1024
# Nombre de relevés bas consécutifs avant de rendre la main. Un pic isolé — un
# modèle qui se charge, un batch ponctuel — ne doit alerter personne.
VRAM_TOLERANCE=3
VRAM_BAS=0
# Idem pour la RAM système : trois creux consécutifs avant de rendre la main.
RAM_TOLERANCE=3
RAM_BAS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --serve-log) SERVE_LOG="$2"; shift 2 ;;
    --tours) TOURS="$2"; shift 2 ;;
    --seuil-mo) SEUIL_MO="$2"; shift 2 ;;
    --llamacpp) LLAMACPP="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) echo "option inconnue : $1" >&2; exit 2 ;;
  esac
done
[ -n "$URL" ] || { echo "usage : bash scripts/aml_monitor.sh --url <url>" >&2; exit 2; }

# Jamais en argument : la ligne de commande d'un processus est lisible par tous.
KEY="${API_KEY:-}"
if [ -z "$KEY" ] && [ -f "$ENV_FILE" ]; then
  KEY=$(grep '^API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '\r')
fi
[ -n "$KEY" ] || { echo "cle introuvable : definir API_KEY, ou la placer dans $ENV_FILE" >&2; exit 2; }

# 150 s : à la concurrence 16 imposée par AML, un /add peut légitimement durer
# plus de 100 s (max mesuré : 106 s). Un délai plus court fabrique de fausses
# alertes pendant les pointes.
TIMEOUT=150

prev_req=$(grep -c -E '"POST /(add|search)' "$SERVE_LOG" 2>/dev/null || echo 0)
prev_err=$(grep -c 'OllamaError' "$SERVE_LOG" 2>/dev/null || echo 0)
NREQ=0; DETAIL=""

sonde() { # url methode [corps] -> "code temps"
  if [ "$2" = "GET" ]; then
    curl -s -o /dev/null -w "%{http_code} %{time_total}" --max-time "$TIMEOUT" "$1"
  else
    curl -s -o /dev/null -w "%{http_code} %{time_total}" --max-time "$TIMEOUT" -X POST "$1" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$3"
  fi
}

memoire_libre_mo() {
  powershell.exe -NoProfile -Command \
    "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)" 2>/dev/null | tr -d '\r'
}

VRAM_LIBRE=0; RAM_LIBRE=0

check() { # 0 si les quatre points répondent 200
  local h s a l util req err nerr
  h=$(sonde "$URL/health" GET)
  s=$(sonde "$URL/search" POST '{"query":"sonde","user_id":"__monitor__","top_k":10}')
  a=$(sonde "$URL/add" POST \
      "{\"request_id\":\"mon-$(date +%s%N)\",\"user_id\":\"__monitor__\",\"session_id\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"sonde $(date +%H:%M)\"}]}")
  l=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 "$LLAMACPP/health")

  # Ressources relevées à chaque tour, pas seulement comparées à un seuil : la
  # tendance vaut mieux qu'une alarme, surtout si un autre travail partage la
  # machine.
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  VRAM_LIBRE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  RAM_LIBRE=$(memoire_libre_mo)

  req=$(grep -c -E '"POST /(add|search)' "$SERVE_LOG" 2>/dev/null || echo 0)
  err=$(grep -c 'OllamaError' "$SERVE_LOG" 2>/dev/null || echo 0)
  NREQ=$((req - prev_req)); nerr=$((err - prev_err))
  prev_req=$req; prev_err=$err

  DETAIL="health=$h search=$s add=$a llamacpp=$l gpu=${util}% vram_libre=${VRAM_LIBRE}Mo ram_libre=${RAM_LIBRE}Mo trafic=+$NREQ erreurs=+$nerr"
  for code in "${h%% *}" "${s%% *}" "${a%% *}" "$l"; do
    [ "$code" = "200" ] || return 1
  done
  return 0
}

echo "$(date '+%d/%m %H:%M')  === surveillance de $URL ===" >> "$LOG"

tour=0
while [ "$tour" -lt "$TOURS" ]; do
  tour=$((tour + 1))

  if check; then
    echo "$(date '+%d/%m %H:%M')  OK    $DETAIL" >> "$LOG"
  else
    premier="$DETAIL"
    sleep 45                       # un creux isolé ne doit pas alerter
    if check; then
      echo "$(date '+%d/%m %H:%M')  REPRIS | 1er: $premier | 2e: $DETAIL" >> "$LOG"
    elif [ "$NREQ" -gt 0 ]; then
      echo "$(date '+%d/%m %H:%M')  SONDE AVEUGLE (cote client) — la prod sert toujours, +$NREQ requetes | $DETAIL" >> "$LOG"
    else
      {
        echo "$(date '+%d/%m %H:%M')  *** PANNE — sondes KO ET plus aucune requete servie ***"
        echo "    tentative 1 : $premier"
        echo "    tentative 2 : $DETAIL"
      } >> "$LOG"
      exit 1
    fi
  fi

  # Ressources. Un journal que personne ne lit n'alerte personne : une pression
  # SOUTENUE rend la main, ce qui déclenche une notification. Un pic isolé, lui,
  # ne justifie pas de réveiller qui que ce soit.
  if [ -n "$VRAM_LIBRE" ] && [ "$VRAM_LIBRE" -lt "$VRAM_ALERTE_MO" ] 2>/dev/null; then
    VRAM_BAS=$((VRAM_BAS + 1))
    echo "$(date '+%d/%m %H:%M')  VRAM BASSE (${VRAM_BAS}/${VRAM_TOLERANCE}) : ${VRAM_LIBRE} Mo libres" >> "$LOG"
    if [ "$VRAM_BAS" -ge "$VRAM_TOLERANCE" ]; then
      echo "$(date '+%d/%m %H:%M')  *** VRAM SOUS PRESSION SOUTENUE : ${VRAM_LIBRE} Mo libres sur $VRAM_TOLERANCE releves ***" >> "$LOG"
      echo "VRAM sous pression soutenue : ${VRAM_LIBRE} Mo libres."
      echo "Un autre travail partage la carte. La production Mnemos n'est pas arretee."
      exit 4
    fi
  else
    VRAM_BAS=0
  fi

  # Même tolérance que pour la VRAM : sur une machine partagée, un calcul qui
  # charge un lot fait plonger la RAM quelques instants. S'arrêter au premier
  # creux reviendrait à ne jamais survivre au travail normal des voisins.
  if [ -n "$RAM_LIBRE" ] && [ "$RAM_LIBRE" -lt "$SEUIL_MO" ] 2>/dev/null; then
    RAM_BAS=$((RAM_BAS + 1))
    echo "$(date '+%d/%m %H:%M')  RAM BASSE (${RAM_BAS}/${RAM_TOLERANCE}) : ${RAM_LIBRE} Mo libres" >> "$LOG"
    if [ "$RAM_BAS" -ge "$RAM_TOLERANCE" ]; then
      echo "$(date '+%d/%m %H:%M')  *** RAM CRITIQUE SOUTENUE : ${RAM_LIBRE} Mo sur $RAM_TOLERANCE releves — arret ***" >> "$LOG"
      echo "RAM systeme critique et durable : ${RAM_LIBRE} Mo libres. Surveillance arretee, priorite a la machine."
      exit 3
    fi
  else
    RAM_BAS=0
  fi

  # Cadence : rapprochée au début du run, puis horaire. Pas d'attente après le
  # dernier tour — sinon le script reste bloqué un quart d'heure pour rien.
  [ "$tour" -lt "$TOURS" ] || break
  if [ "$tour" -lt 16 ]; then sleep 900; else sleep 3600; fi
done

echo "$(date '+%d/%m %H:%M')  surveillance terminee ($tour tours)" >> "$LOG"
