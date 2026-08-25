#!/usr/bin/env bash
# End-to-end HMDB-51 frozen-probe benchmark for every configured backbone.
#
# Usage: ./run_all.sh /data/hmdb51 [split] [head]
set -euo pipefail

cd "$(dirname "$0")"

DATA_ROOT="${1:?usage: ./run_all.sh <hmdb51_root> [split] [head]}"
SPLIT="${2:-1}"
HEAD="${3:-linear}"
FEATURES_DIR="features"
RESULTS_DIR="results"

mkdir -p "$FEATURES_DIR" "$RESULTS_DIR"

echo "==> validating backbones before the long run"
python3 smoke_test.py

for config in configs/*.json; do
  name="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$config")"
  echo
  echo "############ $name ############"

  for subset in train test; do
    echo "==> extracting $subset features"
    python3 extract_features.py \
      --config "$config" \
      --data-root "$DATA_ROOT" \
      --out "$FEATURES_DIR" \
      --subset "$subset" \
      --split "$SPLIT"
  done

  echo "==> training $HEAD probe"
  python3 train_probe.py \
    --features "$FEATURES_DIR/$name" \
    --split "$SPLIT" \
    --head "$HEAD" \
    --report "$RESULTS_DIR/${name}_split${SPLIT}_${HEAD}.json"
done

echo
echo "############ HMDB-51 split $SPLIT ($HEAD probe) ############"
python3 - "$RESULTS_DIR" "$SPLIT" "$HEAD" <<'PY'
import glob, json, os, sys

results_dir, split, head = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for path in sorted(glob.glob(os.path.join(results_dir, f"*_split{split}_{head}.json"))):
    with open(path) as fh:
        rows.append(json.load(fh))

if not rows:
    print("no results found")
    sys.exit(0)

print(f"{'model':<14}{'dim':>6}{'top-1':>10}{'top-5':>10}{'test clips':>12}")
print("-" * 52)
for r in sorted(rows, key=lambda r: -r["top1"]):
    print(f"{r['model']:<14}{r['feature_dim']:>6}{r['top1']:>9.2f}%{r['top5']:>9.2f}%{r['test_clips']:>12}")
PY
