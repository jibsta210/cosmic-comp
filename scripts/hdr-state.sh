#!/usr/bin/env bash
# hdr-state.sh — ground-truth inspector for the live HDR / hardware-color-pipeline
# state on this system. Reads DRM properties directly so it doesn't depend on
# anything cosmic-comp logs. Run any time to confirm whether HDR is actually
# engaged on the panel side and whether the CRTC color pipeline is staged.

set -euo pipefail

CARD="${1:-/dev/dri/card1}"

if ! command -v proptest >/dev/null 2>&1; then
    echo "Need libdrm 'proptest' tool. Falling back to drm_info."
    if ! command -v drm_info >/dev/null 2>&1; then
        echo "drm_info not found either. Install drm_info or libdrm-utils." >&2
        exit 1
    fi
    drm_info "$CARD" 2>/dev/null \
        | rg -A2 'Colorspace|HDR_OUTPUT_METADATA|CTM|DEGAMMA_LUT|GAMMA_LUT' \
        | head -40
    exit 0
fi

# proptest path (preferred — shows live values per connector / crtc)
echo "===== Connectors ====="
sudo proptest -M xe -D "$CARD" 2>&1 | awk '
/^Connector / { conn=$0; show=0 }
/Colorspace|HDR_OUTPUT_METADATA|max bpc/ { print conn; print "  "$0; show=1 }
'

echo
echo "===== CRTCs (color pipeline) ====="
sudo proptest -M xe -D "$CARD" 2>&1 | awk '
/^CRTC / { crtc=$0; show=0 }
/CTM|DEGAMMA_LUT|GAMMA_LUT/ { print crtc; print "  "$0; show=1 }
'

echo
echo "===== Quick verdict ====="
if sudo proptest -M xe -D "$CARD" 2>&1 | rg -q 'Colorspace.*BT2020'; then
    echo "  ✔ Connector Colorspace == BT2020_RGB  (HDR signaling on)"
else
    echo "  ✘ Connector Colorspace != BT2020_RGB  (probably SDR)"
fi
if sudo proptest -M xe -D "$CARD" 2>&1 | rg -q 'HDR_OUTPUT_METADATA.*\b[1-9][0-9]*\b'; then
    echo "  ✔ HDR_OUTPUT_METADATA blob staged   (PQ EOTF metadata sent)"
else
    echo "  ✘ HDR_OUTPUT_METADATA == 0           (no PQ metadata)"
fi
if sudo proptest -M xe -D "$CARD" 2>&1 | rg -q 'DEGAMMA_LUT.*\b[1-9][0-9]*\b'; then
    echo "  ✔ CRTC DEGAMMA_LUT blob staged       (sRGB→linear in hardware)"
else
    echo "  ✘ CRTC DEGAMMA_LUT == 0"
fi
if sudo proptest -M xe -D "$CARD" 2>&1 | rg -q 'CTM.*\b[1-9][0-9]*\b'; then
    echo "  ✔ CRTC CTM blob staged               (gamut+ref_white in hardware)"
else
    echo "  ✘ CRTC CTM == 0"
fi
if sudo proptest -M xe -D "$CARD" 2>&1 | rg -q 'GAMMA_LUT.*\b[1-9][0-9]*\b'; then
    echo "  ✔ CRTC GAMMA_LUT blob staged         (linear→PQ in hardware)"
else
    echo "  ✘ CRTC GAMMA_LUT == 0"
fi
