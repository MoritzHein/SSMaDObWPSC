#!/usr/bin/env bash
# Drives the blocking-aware selection study end to end.
#
#   run_policy_study.sh [MAXJOBS]
#
# Phase 0  policy_experiments.py            settings + cloned inputs + manifest
# Phase 1  placement.py                     ONLY for the model-free references
#                                           (620-625); the policy runs reuse a
#                                           solved placement and are never
#                                           re-solved
# Phase 2  selection_simulation.py          one pass per run, which is what
#                                           creates results/run_<N>/ and lets
#                                           resimulate_seeds.py accept the run
# Phase 3  resimulate_seeds.py              the ten paired request streams
# Phase 4  visualization_policy.py          tables and figures
#
# Runs touch only their own parameter and result files, so the work splits
# across cores by run number.
set -u
cd "$(dirname "$0")"

PY=./.optifaasvenv/Scripts/python.exe
MANIFEST=settings/policy_campaign.json
MAXJOBS=${1:-12}
LOGDIR=logs/policy
mkdir -p "$LOGDIR"

POLICY_RUNS=$(seq 500 607)
HEURISTIC_RUNS=$(seq 620 625)

echo "== phase 0: generating the campaign =="
$PY policy_experiments.py || exit 1

echo "== phase 1: solving the model-free references only =="
for r in $HEURISTIC_RUNS; do
    $PY placement.py --num_run=$r > "$LOGDIR/placement_$r.log" 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
done
wait

echo "== phases 2+3: simulating $(echo $POLICY_RUNS $HEURISTIC_RUNS | wc -w) runs on $MAXJOBS workers =="
worker() {
    local group="$1" log="$2"
    for r in ${group//,/ }; do
        $PY selection_simulation.py --num_run=$r >> "$log" 2>&1 || echo "SIMFAIL $r" >> "$log"
    done
    $PY resimulate_seeds.py --seeds=10 --manifest="$MANIFEST" --runs="$group" >> "$log" 2>&1
    echo "DONE $group" >> "$log"
}

ALL=$(echo $POLICY_RUNS $HEURISTIC_RUNS)
i=0
declare -a GROUP
for r in $ALL; do
    idx=$((i % MAXJOBS))
    GROUP[$idx]="${GROUP[$idx]:-},$r"
    i=$((i + 1))
done
for idx in $(seq 0 $((MAXJOBS - 1))); do
    g="${GROUP[$idx]:-}"
    g="${g#,}"
    [ -z "$g" ] && continue
    worker "$g" "$LOGDIR/worker_$idx.log" &
done
wait

fails=$(grep -l "SIMFAIL" "$LOGDIR"/worker_*.log 2>/dev/null | wc -l)
echo "workers finished; logs with failures: $fails"

echo "== phase 4: tables and figures =="
$PY visualization_policy.py --paper --latex
