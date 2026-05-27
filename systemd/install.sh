#!/usr/bin/env bash
# Installer / mode switcher for the cart's systemd services.
#
#   sudo ./systemd/install.sh --test   # autostart pathfinding_arc_test.py only
#   sudo ./systemd/install.sh --live   # autostart the full cart stack
#   sudo ./systemd/install.sh --none   # install units, enable nothing
#
# Always installs all three unit files; only the *enabled* set changes.
# Re-running is idempotent and safe — it stops + disables everything
# before flipping to the requested mode.
#
# Useful commands after install:
#   systemctl status cart-pathfinder cart-vision cart-arc-test
#   journalctl -fu cart-pathfinder -fu cart-vision
#   journalctl -u  cart-arc-test --since boot
#
# To uninstall completely:
#   sudo systemctl disable --now cart-pathfinder cart-vision cart-arc-test
#   sudo rm /etc/systemd/system/cart-{pathfinder,vision,arc-test}.service
#   sudo systemctl daemon-reload

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Re-running with sudo..."
  exec sudo "$0" "$@"
fi

MODE=""
for arg in "$@"; do
  case "$arg" in
    --live)  MODE=live ;;
    --test)  MODE=test ;;
    --none)  MODE=none ;;
    -h|--help)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      echo "Use --test, --live, or --none." >&2
      exit 1
      ;;
  esac
done

if [ -z "$MODE" ]; then
  echo "Specify a mode: --test (arc test only), --live (full cart), or --none." >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"

# Clear any masked state from previous runs before installing fresh files.
# (`systemctl mask` symlinks the unit to /dev/null, which prevents start.)
systemctl unmask cart-pathfinder.service cart-vision.service cart-arc-test.service 2>/dev/null || true

# Install / refresh all three unit files.
cp "$DIR/cart-pathfinder.service" /etc/systemd/system/
cp "$DIR/cart-vision.service"     /etc/systemd/system/
cp "$DIR/cart-arc-test.service"   /etc/systemd/system/
systemctl daemon-reload

# Quiet baseline — stop and disable everything before flipping mode.
systemctl stop    cart-pathfinder.service cart-vision.service cart-arc-test.service 2>/dev/null || true
systemctl disable cart-pathfinder.service cart-vision.service cart-arc-test.service 2>/dev/null || true

case "$MODE" in
  live)
    systemctl enable cart-pathfinder.service cart-vision.service
    echo ""
    echo "Autostart mode: LIVE  (pathfinder + vision on every boot)"
    echo ""
    echo "*** SAFETY: the cart begins driving the moment the Pi finishes"
    echo "    booting. Always place it on the start pose before powering on."
    echo ""
    echo "Start now:    sudo systemctl start cart-pathfinder cart-vision"
    echo "Live logs:    journalctl -fu cart-pathfinder -fu cart-vision"
    ;;
  test)
    systemctl enable cart-arc-test.service
    echo ""
    echo "Autostart mode: TEST  (pathfinding_arc_test.py runs once at boot)"
    echo ""
    echo "Writes pathfinding_arc_test*.png into the repo root each boot."
    echo "Run now:      sudo systemctl start cart-arc-test"
    echo "Result log:   journalctl -u cart-arc-test --since boot"
    ;;
  none)
    echo ""
    echo "Autostart mode: NONE  (manual launch only)"
    echo ""
    echo "Use ./start_cart.sh for the full stack, or systemctl start <unit>."
    ;;
esac
