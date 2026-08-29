#!/usr/bin/env bash
# End-to-end demo: brings up the stack, seeds the dataset, runs a small
# eval, waits for it, and prints the paired comparison. Doubles as a smoke
# test of the `pe` CLI.
#
#   ./scripts/demo.sh
#
# Override the run shape with env vars:
#   DEMO_SAMPLE=20 DEMO_REPEATS=5 ./scripts/demo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/backend"

pe() { uv run pe "$@"; }

echo "==> Bringing up the stack"
pe up --wait

echo "==> Seeding the eval dataset"
pe seed

echo "==> Starting a run (sample=${DEMO_SAMPLE:-10}, repeats=${DEMO_REPEATS:-3})"
RUN_ID="$(pe run --sample "${DEMO_SAMPLE:-10}" --repeats "${DEMO_REPEATS:-3}" --seed 42 -q)"
echo "    run id: $RUN_ID"

echo "==> Waiting for the run to finish"
pe watch "$RUN_ID" || echo "    (some calls failed — see: pe results $RUN_ID)"

echo "==> Paired comparison"
pe stats compare "$RUN_ID" || true

echo
echo "Done. Dashboard: http://localhost:${FRONTEND_PORT:-5173}/runs/$RUN_ID"
