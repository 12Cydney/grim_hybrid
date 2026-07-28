#!/usr/bin/env bash
# =====================================================================
# link_mt5_history.sh
# =====================================================================
# Finds the Wine MT5 "MQL5/Files/grim_hybrid_history" folder (which has
# an unpredictable <HASH> in its path) and creates a stable symlink at
# /opt/grim_hybrid/data/mt5_history so the collector can read from it.
#
# Run as: sudo bash /opt/grim_hybrid/scripts/link_mt5_history.sh
# =====================================================================

set -euo pipefail

TRADER_USER="${TRADER_USER:-trader}"
WINE_PREFIX="/home/${TRADER_USER}/.wine"
GRIM_ROOT="/opt/grim_hybrid"
LINK_TARGET="${GRIM_ROOT}/data/mt5_history"

if [ ! -d "$WINE_PREFIX" ]; then
    echo "ERROR: Wine prefix not found at $WINE_PREFIX"
    echo "Make sure Phase 1 install script completed and MT5 has been launched at least once."
    exit 1
fi

echo ">>> Searching for MT5 Files folder under Wine prefix..."
CANDIDATES=$(find "$WINE_PREFIX/drive_c/users/$TRADER_USER/AppData/Roaming/MetaQuotes/Terminal" \
             -type d -name "Files" 2>/dev/null || true)

if [ -z "$CANDIDATES" ]; then
    echo "ERROR: No MT5 Files folder found."
    echo "  Launch MT5 at least once under Wine, then re-run this script."
    exit 1
fi

echo "Candidates found:"
echo "$CANDIDATES" | while read -r p; do echo "  $p"; done

# Prefer the one that actually has grim_hybrid_history inside it
CHOSEN=""
while IFS= read -r p; do
    if [ -d "$p/grim_hybrid_history" ]; then
        CHOSEN="$p/grim_hybrid_history"
        break
    fi
done <<< "$CANDIDATES"

# Otherwise take the first Files folder and CREATE grim_hybrid_history in it
if [ -z "$CHOSEN" ]; then
    FIRST=$(echo "$CANDIDATES" | head -n 1)
    CHOSEN="$FIRST/grim_hybrid_history"
    echo ">>> Creating: $CHOSEN"
    sudo -u "$TRADER_USER" mkdir -p "$CHOSEN"
fi

echo ">>> Chosen source: $CHOSEN"

mkdir -p "${GRIM_ROOT}/data"
# Remove old symlink or directory at the target
if [ -L "$LINK_TARGET" ] || [ -e "$LINK_TARGET" ]; then
    rm -rf "$LINK_TARGET"
fi

ln -s "$CHOSEN" "$LINK_TARGET"
chown -h "$TRADER_USER":"$TRADER_USER" "$LINK_TARGET"

echo ""
echo "==============================================================="
echo " DONE."
echo " Wine MT5 dumps CSVs into:"
echo "   $CHOSEN"
echo " Which is now reachable at:"
echo "   $LINK_TARGET"
echo ""
echo " Next: run the history_dump.mq5 script inside MT5, then:"
echo "   sudo -u $TRADER_USER $GRIM_ROOT/.venv/bin/python \\"
echo "       $GRIM_ROOT/collector/collect_training_data.py \\"
echo "       --from-csv $LINK_TARGET"
echo "==============================================================="
