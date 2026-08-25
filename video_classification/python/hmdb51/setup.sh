#!/usr/bin/env bash
# One-shot environment setup for a freshly rented GPU box.
# Usage: ./setup.sh [--venv]
set -euo pipefail

cd "$(dirname "$0")"

if [ "${1:-}" = "--venv" ]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "==> using venv at $(pwd)/.venv"
fi

echo "==> installing python dependencies"
pip install --upgrade pip
pip install -r requirements.txt


echo
echo "==> environment"
python3 - <<'PY'
import torch, transformers
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  cuda         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu          {p.name} ({p.total_memory / 1024**3:.1f} GB)")
import importlib.util as u
for m in ("videoprism", "vjepa2", "pe_video"):
    ok = u.find_spec(f"transformers.models.{m}") is not None
    print(f"  model {m:12s} {'available' if ok else 'MISSING - upgrade transformers'}")
# timm is required by PE Video; a decoder and libarchive by the dataset fetcher.
for m, why in (("timm", "needed by pe_video"),
               ("torchcodec", "preferred decoder"),
               ("av", "decoder fallback"),
               ("cv2", "decoder fallback"),
               ("libarchive", "needed by download_hmdb51.py")):
    ok = u.find_spec(m) is not None
    print(f"  dep   {m:12s} {'available' if ok else 'MISSING - ' + why}")
PY

echo
echo "Next:"
echo "  python download_hmdb51.py /data/hmdb51"
echo "  python smoke_test.py"
echo "  ./run_all.sh /data/hmdb51"
