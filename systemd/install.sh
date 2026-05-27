#!/usr/bin/env bash
# One-time installer for the cart's systemd services.
#
#   sudo ./systemd/install.sh           # install + enable autostart
#   sudo ./systemd/install.sh --no-enable   # install but don't autostart
#
# Re-running is idempotent: it just overwrites the unit files.
#
# Useful commands after install:
#   systemctl status cart-pathfinder cart-vision
#   journalctl -fu cart-pathfinder -fu cart-vision   # live logs
#   sudo systemctl start  cart-pathfinder cart-vision
#   sudo systemctl stop   cart-vision cart-pathfinder
#   sudo systemctl disable cart-pathfinder cart-vision   # remove from boot
#
# To uninstall completely:
#   sudo systemctl disable --now cart-pathfinder cart-vision
#   sudo rm /etc/systemd/system/cart-{pathfinder,vision}.service
#   sudo systemctl daemon-reload

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Re-running with sudo..."
  exec sudo "$0" "$@"
fi

ENABLE=1
for arg in "$@"; do
  case "$arg" in
    --no-enable) ENABLE=0 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$DIR/cart-pathfinder.service" /etc/systemd/system/
cp "$DIR/cart-vision.service"     /etc/systemd/system/
systemctl daemon-reload

if [ "$ENABLE" -eq 1 ]; then
  systemctl enable cart-pathfinder.service cart-vision.service
  echo ""
  echo "Installed and enabled. The cart will start on every boot."
  echo ""
  echo "*** SAFETY: the cart begins driving the moment the Pi finishes"
  echo "    booting. Always place it on the start pose before powering on,"
  echo "    or run:    sudo systemctl disable cart-pathfinder cart-vision"
else
  systemctl disable cart-pathfinder.service cart-vision.service 2>/dev/null || true
  echo ""
  echo "Installed but autostart is OFF. Start manually:"
  echo "    sudo systemctl start cart-pathfinder cart-vision"
fi

echo ""
echo "Live logs:   journalctl -fu cart-pathfinder -fu cart-vision"
echo "Status:      systemctl status cart-pathfinder cart-vision"
