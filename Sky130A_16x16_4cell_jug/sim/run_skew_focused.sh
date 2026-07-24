#!/usr/bin/env bash
# focused skew curve: faithful lag {1,2,4,8} + one worst-case rand {8}.
# Baseline (skew=0) = 67.47% at this fast config (ep=2 ch=4000 seed=42).
cd "$(dirname "$0")"
EP=2; CH=4000; SEED=42; OUT=SKEW_FOCUSED.txt
echo "focused skew: ep=$EP ch=$CH seed=$SEED  baseline(skew=0)=67.47%  $(date +%H:%M)" > "$OUT"
for spec in "1 lag" "2 lag" "4 lag" "8 lag" "8 rand"; do
  set -- $spec; f=$1; m=$2
  r=$(python3 pcn_jug_skew.py --epochs $EP --chunk $CH --seed $SEED \
        --skew_folds $f --skew_mode $m 2>/dev/null | grep 'BEST =' | tail -1)
  printf "skew=%s %-4s : %s\n" "$f" "$m" "${r:-(no result)}" >> "$OUT"
done
echo "done $(date +%H:%M)" >> "$OUT"
