#!/usr/bin/env bash
#
# Download the pretrained baseline checkpoints into
# baselines/<method>/checkpoints/, where train.py / evaluate.py look for them.
#
# Weights are served from $CARLANOMALY_BASE_URL/models
# (default: https://data.carlanomaly.de/v1). Re-runs are idempotent:
# a file whose size already matches the server is skipped, otherwise the
# download resumes from where it left off.
#
# Usage:
#   ./download_checkpoints.sh
#
# Note: weather_prediction has no published checkpoint yet — train it locally.

set -euo pipefail

BASE_URL="${CARLANOMALY_BASE_URL:-https://data.carlanomaly.de/v1}"
MODELS_URL="${BASE_URL%/}/models"

# Repo root = the directory this script lives in (so it works from any cwd).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# method  ->  checkpoint filename
CKPTS=(
  "rgb_segmentation:deeplabv3_r50_msp.pt"
  "lidar_segmentation:pointnet_seg.pt"
  "frame_prediction:unet_ffp.pt"
)

# Print the remote Content-Length in bytes, or nothing if unavailable.
remote_size() {
  curl -fsIL --max-time 30 "$1" \
    | awk 'BEGIN{IGNORECASE=1} /^content-length:/{v=$2} END{gsub(/\r/,"",v); print v}'
}

# Local file size in bytes, portable across GNU (Linux) and BSD (macOS) stat.
local_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

for entry in "${CKPTS[@]}"; do
  method="${entry%%:*}"
  file="${entry#*:}"
  url="$MODELS_URL/$file"
  dest_dir="$ROOT/baselines/$method/checkpoints"
  dest="$dest_dir/$file"

  echo "==> $method/$file"
  mkdir -p "$dest_dir"

  if [[ -f "$dest" ]]; then
    rsize="$(remote_size "$url" || true)"
    lsize="$(local_size "$dest")"
    if [[ -n "$rsize" && "$lsize" == "$rsize" ]]; then
      echo "    already complete ($lsize bytes), skipping"
      continue
    fi
  fi

  curl -fL --progress-bar -C - "$url" -o "$dest"
done

echo "Done."
