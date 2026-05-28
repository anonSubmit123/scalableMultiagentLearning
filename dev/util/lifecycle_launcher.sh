#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  lifecycle_launcher.sh \
    --project-dir PROJECT_DIR \
    --module MODULE \
    --physical-system-id PHYSICAL_SYSTEM_ID \
    --host HOST \
    --port PORT
EOF
}

PROJECT_DIR=""
MODULE="runtime.host_lifecycle_server"
PHYSICAL_SYSTEM_ID=""
LIFECYCLE_HOST=""
PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --module)
      MODULE="$2"
      shift 2
      ;;
    --physical-system-id)
      PHYSICAL_SYSTEM_ID="$2"
      shift 2
      ;;
    --host)
      LIFECYCLE_HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PROJECT_DIR" || -z "$MODULE" || -z "$PHYSICAL_SYSTEM_ID" || -z "$LIFECYCLE_HOST" || -z "$PORT" ]]; then
  usage
  exit 2
fi

if [[ -f "$PROJECT_DIR/util/.activate_umamba.sh" ]]; then
  # shellcheck source=/dev/null
  . "$HOME/.activate_umamba.sh"
else
  echo "Missing $HOME/.activate_umamba.sh" >&2
  exit 1
fi

micromamba activate rl
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

process_pattern="python3 -m $MODULE --physical-system-id $PHYSICAL_SYSTEM_ID --host $LIFECYCLE_HOST --port $PORT"

if pgrep -f -- "$process_pattern" >/dev/null; then
  echo "Lifecycle server already running: $process_pattern"
  exit 0
fi

nohup python3 -m "$MODULE" \
  --physical-system-id "$PHYSICAL_SYSTEM_ID" \
  --host "$LIFECYCLE_HOST" \
  --port "$PORT" \
  > /dev/null 2>&1 &

echo "Started lifecycle server: $process_pattern"
