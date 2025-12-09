#!/usr/bin/env bash
set -euo pipefail

# Fetch a minimal FastMRI knee single-coil subset (research-only, CC BY-NC 4.0).
#
# ⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️
#
# This data and any software using it is provided strictly for research,
# educational, and experimental purposes. Do NOT use for medical diagnosis,
# treatment decisions, or clinical patient care.
#
# This requires prior registration/acceptance of the FastMRI terms. You must
# provide a signed URL or mirror via FASTMRI_KNEE_URL (zip or tar.gz).

if [[ "${USE_REAL_DATA:-0}" != "1" ]]; then
  echo "USE_REAL_DATA not set to 1; aborting to avoid accidental download." >&2
  exit 1
fi

if [[ -z "${FASTMRI_KNEE_URL:-}" ]]; then
  cat <<'EOF' >&2
FASTMRI_KNEE_URL not provided.
Please obtain a signed download link for the knee single-coil small subset
after accepting CC BY-NC 4.0 at https://fastmri.org/ and set:
  FASTMRI_KNEE_URL="https://.../knee_singlecoil.zip"
Then re-run:
  USE_REAL_DATA=1 FASTMRI_KNEE_URL=$FASTMRI_KNEE_URL bash data/fastmri_fetch.sh
EOF
  exit 1
fi

target_root="$(cd "$(dirname "$0")" && pwd)/fastmri"
mkdir -p "$target_root"

archive="$target_root/knee_singlecoil_subset.zip"
if [[ -f "$archive" ]]; then
  echo "✓ Archive exists: $archive"
else
  echo "↓ Downloading FastMRI knee subset from provided URL..."
  curl -fL "$FASTMRI_KNEE_URL" -o "$archive"
fi

if [[ -d "$target_root/knee_singlecoil_subset" ]]; then
  echo "✓ Extracted folder exists: $target_root/knee_singlecoil_subset"
  exit 0
fi

echo "→ Extracting..."
unzip -q "$archive" -d "$target_root"
echo "Done. Files under $target_root"
