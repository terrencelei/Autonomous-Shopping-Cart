#!/usr/bin/env bash
# Launch the full cart stack — pathfinder + vision — fully detached from
# the current SSH session.  Both processes keep running after you exit
# the shell.  Logs are written to logs/<timestamp>-{pathfinder,vision}.log.
#
# Typical workflow:
#   ssh pi
#   ./start_cart.sh        # place cart at the start pose first!
#   exit                   # SSH disconnects, cart keeps going
#   ./stop_cart.sh         # later: ssh back in and stop it
#
# Use --display to keep the OpenCV preview window open (only useful if
# you're running from the Pi's own desktop, not over SSH).

set -euo pipefail
cd "$(dirname "$0")"

YOLO_FLAGS="--no-display"
for arg in "$@"; do
  case "$arg" in
    --display) YOLO_FLAGS="" ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

PATHFINDER_PID_FILE=/tmp/cart-pathfinder.pid
VISION_PID_FILE=/tmp/cart-vision.pid

# Refuse to double-start.
for f in "$PATHFINDER_PID_FILE" "$VISION_PID_FILE"; do
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
    echo "Already running (pid $(cat "$f") from $f). Run ./stop_cart.sh first." >&2
    exit 1
  fi
done

mkdir -p logs
ts=$(date +%Y%m%d-%H%M%S)
PATHFINDER_LOG="logs/${ts}-pathfinder.log"
VISION_LOG="logs/${ts}-vision.log"

echo "Starting pathfinder → $PATHFINDER_LOG"
setsid python3 -u Pathfinding_algorithm.py \
  > "$PATHFINDER_LOG" 2>&1 < /dev/null &
echo $! > "$PATHFINDER_PID_FILE"

# Give the UDP receiver and serial port a moment to come up so vision's
# first packets aren't sent into the void.
sleep 2

echo "Starting vision     → $VISION_LOG"
( cd vision && setsid python3 -u yolo_detect.py $YOLO_FLAGS \
    > "../$VISION_LOG" 2>&1 < /dev/null & echo $! > "$VISION_PID_FILE" )

echo ""
echo "Running:"
echo "  pathfinder  pid=$(cat "$PATHFINDER_PID_FILE")"
echo "  vision      pid=$(cat "$VISION_PID_FILE")"
echo ""
echo "Tail logs:    tail -f $PATHFINDER_LOG $VISION_LOG"
echo "Stop:         ./stop_cart.sh"
echo ""
echo "Safe to disconnect (exit / Ctrl-D) — both processes are detached."
