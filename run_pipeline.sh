#!/bin/bash
set -uo pipefail   # no -e so we still notify on failure

LOG_DIR="/Users/abhishekkumbhar/Documents/s2-data-processing/logs"
mkdir -p "$LOG_DIR"

APP_LOG="$LOG_DIR/pipeline_S2_Pipeline.log"
OUT_LOG="$LOG_DIR/pipeline.log"
ERR_LOG="$LOG_DIR/pipeline_err.log"

SLACK_URL="${SLACK_WEBHOOK_URL:-}"

START_TS="$(date -Iseconds)"
START_EPOCH="$(date +%s)"
STATUS=0

# Always send a Slack message when the script exits
notify() {
  local END_TS="$(date -Iseconds)"
  local END_EPOCH="$(date +%s)"
  local ELAPSED=$((END_EPOCH - START_EPOCH))
  local H=$((ELAPSED/3600)); local M=$(((ELAPSED%3600)/60)); local S=$((ELAPSED%60))
  local DURATION=$(printf "%02d:%02d:%02d" "$H" "$M" "$S")

  echo "[launchd] end ${END_TS} with status ${STATUS} (duration ${DURATION})" >> "$OUT_LOG"

  [[ -z "$SLACK_URL" ]] && return 0

  local STATUS_EMOJI
  if [[ "$STATUS" == "0" ]]; then STATUS_EMOJI=":white_check_mark:"; else STATUS_EMOJI=":x:"; fi

  local TAIL_APP
  TAIL_APP="$(tail -n 20 "$APP_LOG" 2>/dev/null | sed 's/"/\\"/g')"

  read -r -d '' PAYLOAD <<JSON
{
  "text": "*S2 Pipeline ${STATUS_EMOJI}* (exit ${STATUS})",
  "blocks": [
    {"type":"section","text":{"type":"mrkdwn","text":"*S2 Pipeline ${STATUS_EMOJI}* (exit ${STATUS})\n*Start:* ${START_TS}\n*End:* ${END_TS}\n*Duration:* ${DURATION}"}},
    {"type":"section","text":{"type":"mrkdwn","text":"*Last 20 lines of app log:*\\n\\n\`\`\`${TAIL_APP}\`\`\`"}}
  ]
}
JSON

  /usr/bin/curl -sS -X POST -H 'Content-type: application/json' --data "$PAYLOAD" "$SLACK_URL" >/dev/null || true
}
trap notify EXIT

echo "[launchd] start ${START_TS}" >> "$OUT_LOG"

/Users/abhishekkumbhar/miniconda3/bin/conda run -n s2-data-processing \
  python /Users/abhishekkumbhar/Documents/s2-data-processing/scripts/main_pipeline.py --automated \
  >> "$OUT_LOG" 2>> "$ERR_LOG" || STATUS=$?

exit $STATUS
