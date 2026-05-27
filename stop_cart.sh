#!/usr/bin/env bash
# Stop the pathfinder and vision processes started by start_cart.sh.
# The pathfinder's `finally` block sends a zero-velocity stop to the
# ESP32 before exiting, and the ESP32's own 500 ms watchdog will cut
# the motors regardless.

set -u
cd "$(dirname "$0")"

stop_pid_file() {
  local f="$1" name="$2"
  if [ -f "$f" ]; then
    local pid
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid $pid)"
      kill "$pid" 2>/dev/null || true
      # Give it a moment to clean up, then SIGKILL if it's still alive.
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "  ... did not exit, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
      fi
    else
      echo "$name pid file present but process $pid is gone"
    fi
    rm -f "$f"
  fi
}

stop_pid_file /tmp/cart-pathfinder.pid pathfinder
stop_pid_file /tmp/cart-vision.pid     vision

# Belt-and-suspenders in case the pid files drifted out of sync with reality.
pkill -f 'python3 .*Pathfinding_algorithm.py' 2>/dev/null || true
pkill -f 'python3 .*yolo_detect.py'           2>/dev/null || true

echo "Done."
