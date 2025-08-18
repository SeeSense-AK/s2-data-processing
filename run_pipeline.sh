#!/bin/bash

# S2 Data Processing Pipeline Runner for Launch Daemon
# Enhanced for bulletproof headless operation

set -uo pipefail

# Configuration
PROJECT_DIR="/Users/abhishekkumbhar/Documents/s2-data-processing"
CONDA_ENV="s2-data-processing"
LOG_DIR="$PROJECT_DIR/logs"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/main_pipeline.py"
CONDA_BASE="/Users/abhishekkumbhar/miniconda3"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Log files
APP_LOG="$LOG_DIR/pipeline_S2_Pipeline.log"
DAEMON_LOG="$LOG_DIR/daemon_runner.log"
OUT_LOG="$LOG_DIR/pipeline.log"
ERR_LOG="$LOG_DIR/pipeline_err.log"

# Timestamps
START_TS="$(date -Iseconds)"
START_EPOCH="$(date +%s)"
STATUS=0

# Function to send Slack notification
notify_slack() {
    local END_TS="$(date -Iseconds)"
    local END_EPOCH="$(date +%s)"
    local ELAPSED=$((END_EPOCH - START_EPOCH))
    local H=$((ELAPSED/3600))
    local M=$(((ELAPSED%3600)/60))
    local S=$((ELAPSED%60))
    local DURATION=$(printf "%02d:%02d:%02d" "$H" "$M" "$S")

    echo "[$(date)] 🏁 Pipeline completed with status $STATUS (duration $DURATION)" >> "$DAEMON_LOG"

    # Send Slack notification if webhook URL is set
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local STATUS_EMOJI STATUS_TEXT
        if [[ "$STATUS" == "0" ]]; then
            STATUS_EMOJI=":white_check_mark:"
            STATUS_TEXT="SUCCESS"
        else
            STATUS_EMOJI=":x:"
            STATUS_TEXT="FAILED"
        fi

        # Get last 20 lines of app log for Slack
        local TAIL_APP=""
        if [[ -f "$APP_LOG" ]]; then
            TAIL_APP="$(tail -n 20 "$APP_LOG" 2>/dev/null | sed 's/"/\\"/g' | head -c 1000)"
        fi

        # Create Slack payload
        local PAYLOAD=$(cat <<EOFSLACK
{
  "text": "*S2 Pipeline $STATUS_EMOJI $STATUS_TEXT* (Launch Daemon - exit $STATUS)",
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*S2 Pipeline $STATUS_EMOJI $STATUS_TEXT* (Launch Daemon)\\n*Start:* $START_TS\\n*End:* $END_TS\\n*Duration:* $DURATION\\n*Exit Code:* $STATUS\\n*Host:* $(hostname)"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Last 20 lines of pipeline log:*\\n\\n\`\`\`$TAIL_APP\`\`\`"
      }
    }
  ]
}
EOFSLACK
)

        # Send notification with timeout
        timeout 30 curl -sS -X POST -H 'Content-type: application/json' \
             --data "$PAYLOAD" \
             "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || \
             echo "[$(date)] ⚠️ Failed to send Slack notification" >> "$DAEMON_LOG"
    fi
}

# Set up trap to always send notification on exit
trap notify_slack EXIT

# Start logging
echo "" >> "$DAEMON_LOG"
echo "========================================" >> "$DAEMON_LOG"
echo "[$(date)] 🚀 Starting S2 Pipeline (Launch Daemon)" >> "$DAEMON_LOG"
echo "[$(date)] ⏰ Start time: $START_TS" >> "$DAEMON_LOG"
echo "[$(date)] 📁 Working directory: $PROJECT_DIR" >> "$DAEMON_LOG"
echo "[$(date)] 🐍 Conda environment: $CONDA_ENV" >> "$DAEMON_LOG"
echo "[$(date)] 👤 Running as user: $(whoami)" >> "$DAEMON_LOG"
echo "[$(date)] 🖥️ Hostname: $(hostname)" >> "$DAEMON_LOG"

# Change to project directory
cd "$PROJECT_DIR" || {
    echo "[$(date)] ❌ ERROR: Cannot change to project directory: $PROJECT_DIR" >> "$DAEMON_LOG"
    STATUS=1
    exit 1
}

# Initialize conda for daemon operation
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    echo "[$(date)] ✅ Conda profile sourced successfully" >> "$DAEMON_LOG"
else
    echo "[$(date)] ❌ ERROR: Cannot find conda profile at $CONDA_BASE/etc/profile.d/conda.sh" >> "$DAEMON_LOG"
    STATUS=1
    exit 1
fi

# Check if conda environment exists
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "[$(date)] ✅ Conda environment '$CONDA_ENV' found" >> "$DAEMON_LOG"
else
    echo "[$(date)] ❌ ERROR: Conda environment '$CONDA_ENV' not found" >> "$DAEMON_LOG"
    conda env list >> "$DAEMON_LOG"
    STATUS=1
    exit 1
fi

# Check Docker availability
if ! command -v docker >/dev/null 2>&1; then
    echo "[$(date)] ❌ ERROR: Docker command not found" >> "$DAEMON_LOG"
    STATUS=1
    exit 1
fi

# Check if OSRM containers are running
echo "[$(date)] 🐳 Checking OSRM containers..." >> "$DAEMON_LOG"
REQUIRED_CONTAINERS=("osrm-england" "osrm-finland" "osrm-ireland" "osrm-sydney" "osrm-wales")
MISSING_CONTAINERS=()

for container in "${REQUIRED_CONTAINERS[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
        echo "[$(date)] ✅ OSRM container '$container' is running" >> "$DAEMON_LOG"
    else
        MISSING_CONTAINERS+=("$container")
        echo "[$(date)] ⚠️ OSRM container '$container' is not running" >> "$DAEMON_LOG"
    fi
done

# Start missing containers
if [[ ${#MISSING_CONTAINERS[@]} -gt 0 ]]; then
    echo "[$(date)] 🔄 Attempting to start missing containers: ${MISSING_CONTAINERS[*]}" >> "$DAEMON_LOG"
    
    for container in "${MISSING_CONTAINERS[@]}"; do
        if timeout 60 docker start "$container" >> "$DAEMON_LOG" 2>&1; then
            echo "[$(date)] ✅ Started container: $container" >> "$DAEMON_LOG"
        else
            echo "[$(date)] ❌ Failed to start container: $container" >> "$DAEMON_LOG"
        fi
    done
    
    # Wait for containers to fully initialize
    echo "[$(date)] ⏳ Waiting 20 seconds for containers to initialize..." >> "$DAEMON_LOG"
    sleep 20
    
    # Verify containers are now running
    echo "[$(date)] 🔍 Final container status check:" >> "$DAEMON_LOG"
    for container in "${REQUIRED_CONTAINERS[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
            echo "[$(date)] ✅ $container is now running" >> "$DAEMON_LOG"
        else
            echo "[$(date)] ❌ $container failed to start" >> "$DAEMON_LOG"
        fi
    done
fi

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "[$(date)] ❌ ERROR: Python script not found: $PYTHON_SCRIPT" >> "$DAEMON_LOG"
    STATUS=1
    exit 1
fi

# Activate conda environment and run the pipeline
echo "[$(date)] 🐍 Activating conda environment '$CONDA_ENV'..." >> "$DAEMON_LOG"

if conda activate "$CONDA_ENV"; then
    echo "[$(date)] ✅ Conda environment activated successfully" >> "$DAEMON_LOG"
    echo "[$(date)] 🚀 Starting pipeline execution..." >> "$DAEMON_LOG"
    
    # Run the pipeline
    python "$PYTHON_SCRIPT" --automated >> "$OUT_LOG" 2>> "$ERR_LOG" || STATUS=$?
    
    echo "[$(date)] 🏁 Pipeline execution completed with status: $STATUS" >> "$DAEMON_LOG"
else
    echo "[$(date)] ❌ ERROR: Failed to activate conda environment" >> "$DAEMON_LOG"
    STATUS=1
fi

echo "[$(date)] 📊 Final exit status: $STATUS" >> "$DAEMON_LOG"
echo "========================================" >> "$DAEMON_LOG"

# Exit with the status from the pipeline
exit $STATUS
