#!/usr/bin/env bash
# Regression: elaborate + run every testbench, report PASS/FAIL per TB.
# Usage:  ./run_all_tb.sh            (all)
#         ./run_all_tb.sh tb_cap     (substring filter)
#
# A TB counts as FAIL if it does not elaborate, or if its output contains a
# failure marker, or if it never prints a pass marker. Silence is a FAIL —
# a test that cannot fail is not a test.
set -u
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/pcn_tb.$$"
mkdir -p "$OUT"
filter="${1:-}"
pass=0; fail=0; failed=()

for tb in tb_*.v; do
    name="${tb%.v}"
    [[ -n "$filter" && "$name" != *"$filter"* ]] && continue
    if ! iverilog -g2012 -y . -I. -o "$OUT/$name" "$tb" > "$OUT/$name.elab" 2>&1; then
        printf '  %-28s ELAB-FAIL\n' "$name"; failed+=("$name (elaborate)"); ((fail++)); continue
    fi
    vvp "$OUT/$name" > "$OUT/$name.log" 2>&1
    log="$OUT/$name.log"
    # Failure markers first (a FAIL anywhere outranks a PASS elsewhere), excluding
    # benign zero-count phrasings: "0 mismatches", "0 checks failed",
    # "Results: 16 passed, 0 failed", "0 over TOL(2)".
    fails=$(grep -iE '\bFAIL|\bMISMATCH|\bERROR|\$fatal' "$log" \
            | grep -vE '\b0 +[A-Za-z|Δ]* *(mismatch|error|fail)' \
            | grep -vE '(passed|,) *0 +fail')
    if [[ -n "$fails" ]]; then
        printf '  %-28s FAIL\n' "$name"; failed+=("$name"); ((fail++))
    # TBs use different verdicts: "ALL PASS", "RTL-B1: PASS", "PASS (…)".
    elif grep -qE 'ALL PASS|\bPASS\b' "$log"; then
        printf '  %-28s pass\n' "$name"; ((pass++))
    else
        printf '  %-28s NO-VERDICT\n' "$name"; failed+=("$name (no verdict)"); ((fail++))
    fi
done

echo "----------------------------------------"
echo "  pass: $pass   fail: $fail"
if (( fail )); then printf '  failed: %s\n' "${failed[@]}"; echo "  logs: $OUT"; exit 1; fi
echo "  logs: $OUT"
