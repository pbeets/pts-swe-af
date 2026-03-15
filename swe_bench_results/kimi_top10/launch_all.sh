#!/bin/bash
# Launch all 10 benchmark instances in parallel
# Run from: /Users/santoshkumarradha/Documents/agentfield/code/examples/af-swe

cd /Users/santoshkumarradha/Documents/agentfield/code/examples/af-swe

INSTANCES=(
  "sphinx-doc__sphinx-7590:kimi_sphinx-doc__sphinx-7590"
  "sphinx-doc__sphinx-9461:kimi_sphinx-doc__sphinx-9461"
  "django__django-11885:kimi_django__django-11885"
  "sphinx-doc__sphinx-10673:kimi_sphinx-doc__sphinx-10673"
  "sphinx-doc__sphinx-8548:kimi_sphinx-doc__sphinx-8548"
  "django__django-12155:kimi_django__django-12155"
  "django__django-12325:kimi_django__django-12325"
  "sphinx-doc__sphinx-8551:kimi_sphinx-doc__sphinx-8551"
  "sphinx-doc__sphinx-7748:kimi_sphinx-doc__sphinx-7748"
)

PIDS=()

for entry in "${INSTANCES[@]}"; do
  IFS=':' read -r instance_id repo_dir <<< "$entry"
  results_dir="./swe_bench_results/kimi_top10/${instance_id}"
  log_file="./swe_bench_results/kimi_top10/${instance_id}.log"
  mkdir -p "$results_dir"

  echo "Launching: $instance_id"
  nohup python3 -m benchmarks.swe_bench run \
    --instances "$instance_id" \
    --model "openrouter/moonshotai/kimi-k2.5" \
    --repo-path "/workspaces/staged/$repo_dir" \
    --results-dir "$results_dir" \
    --batch-size 1 --concurrency 1 \
    --timeout 3600 --poll-interval 30 \
    > "$log_file" 2>&1 &

  PIDS+=($!)
  echo "  PID: $!"
done

echo ""
echo "All ${#PIDS[@]} instances launched."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor with:"
echo "  tail -f swe_bench_results/kimi_top10/*.log"
echo "  # or check individual:"
echo "  tail -f swe_bench_results/kimi_top10/django__django-11885.log"
