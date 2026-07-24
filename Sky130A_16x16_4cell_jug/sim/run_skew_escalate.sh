#!/usr/bin/env bash
# escalation: confirm the skew finding at a MORE-CONVERGED config (ep=3 ch=8000).
# Checks that "skew=1 free, skew=2 ~1pp, bounded" survives as accuracy climbs above the
# fast-config 67%.
cd "$(dirname "$0")"
EP=3; CH=8000; SEED=42; OUT=SKEW_ESCALATE.txt
echo "escalate skew: ep=$EP ch=$CH seed=$SEED  $(date +%H:%M)" > "$OUT"
for spec in "0 lag" "1 lag" "2 lag"; do
  set -- $spec; f=$1; m=$2
  r=$(python3 pcn_jug_skew.py --epochs $EP --chunk $CH --seed $SEED \
        --skew_folds $f --skew_mode $m 2>/dev/null | grep 'BEST =' | tail -1 \
        | sed -E 's/.*BEST = //; s/ .*//')
  printf "skew=%s %-4s : %s\n" "$f" "$m" "${r:-(no result)}" >> "$OUT"
done
echo "done $(date +%H:%M)" >> "$OUT"
