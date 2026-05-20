#!/bin/bash
# Monitors training for thermal throttling. When detected, alerts and surfaces Macs Fan Control.
LOG="/private/tmp/claude-501/-Users-leifamily-Desktop-Autonomous-Shopping-Cart-vision/b9cbd0e7-cc2b-4751-83af-c2514d555f45/tasks/bmjkai0q1.output"
THROTTLE_THRESHOLD=3
boosted=0

alert() {
    # Play system alert sound 3x and show notification
    for i in 1 2 3; do afplay /System/Library/Sounds/Sosumi.aiff & done
    osascript -e "display notification \"$1\" with title \"🌡 Thermal Alert\" sound name \"Sosumi\""
    # Bring Macs Fan Control to front so user can set fans
    open "/Applications/Macs Fan Control.app"
}

echo "[$(date '+%H:%M:%S')] Throttle watcher started (threshold: >${THROTTLE_THRESHOLD}s/it)"

while true; do
    sleep 10

    sit=$(tail -c 2000 "$LOG" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+s/it" | tail -1 | grep -oE "[0-9]+\.[0-9]+")

    [ -z "$sit" ] && continue

    is_throttling=$(echo "$sit > $THROTTLE_THRESHOLD" | bc -l 2>/dev/null)

    if [ "$is_throttling" = "1" ] && [ "$boosted" = "0" ]; then
        echo "[$(date '+%H:%M:%S')] Throttling detected (${sit}s/it) — alerting"
        alert "Throttling at ${sit}s/it — set fans to max in Macs Fan Control"
        boosted=1
    elif [ "$is_throttling" = "0" ] && [ "$boosted" = "1" ]; then
        echo "[$(date '+%H:%M:%S')] Throttling cleared (${sit}s/it)"
        osascript -e "display notification \"Throttling cleared (${sit}s/it) — fans can go back to auto\" with title \"✅ Training Monitor\""
        boosted=0
    fi

    if grep --text -qaE "Saved combined model|Training complete" "$LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Training complete — watcher exiting"
        osascript -e "display notification \"Training complete! combined.pt saved.\" with title \"✅ Training Done\""
        break
    fi
done
