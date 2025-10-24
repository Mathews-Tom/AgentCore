#!/bin/bash
# Performance validation script for Integration Layer
# Validates that performance targets are met before deployment

set -e

echo "=================================================="
echo "  Integration Layer Performance Validation"
echo "=================================================="
echo ""

# Configuration
HOST="${HOST:-http://localhost:8001}"
USERS="${USERS:-1000}"
SPAWN_RATE="${SPAWN_RATE:-100}"
RUN_TIME="${RUN_TIME:-2m}"
RESULTS_DIR="test-results/$(date +%Y%m%d-%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo "  Host: $HOST"
echo "  Users: $USERS"
echo "  Spawn Rate: $SPAWN_RATE"
echo "  Duration: $RUN_TIME"
echo "  Results: $RESULTS_DIR"
echo ""

# Check if AgentCore is running
echo "→ Checking if AgentCore is running..."
if ! curl -s "$HOST/health" > /dev/null 2>&1; then
    echo "❌ ERROR: AgentCore is not running at $HOST"
    echo "   Start server: uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001"
    exit 1
fi
echo "✓ AgentCore is running"
echo ""

# Run load test
echo "→ Running load test..."
echo "  (This will take approximately $RUN_TIME)"
echo ""

uv run locust \
    -f tests/load/integration_layer_load_test.py \
    --headless \
    --users "$USERS" \
    --spawn-rate "$SPAWN_RATE" \
    --run-time "$RUN_TIME" \
    --host "$HOST" \
    --html "$RESULTS_DIR/report.html" \
    --csv "$RESULTS_DIR/stats" \
    2>&1 | tee "$RESULTS_DIR/output.log"

echo ""
echo "✓ Load test complete"
echo ""

# Parse results
echo "→ Validating performance targets..."
echo ""

# Extract key metrics from CSV
if [ -f "$RESULTS_DIR/stats_stats.csv" ]; then
    # Parse P95 latency and RPS from stats
    STATS_FILE="$RESULTS_DIR/stats_stats.csv"

    # Get aggregated stats (last line)
    AGGREGATED=$(tail -n 1 "$STATS_FILE")

    # Extract metrics (columns: Type,Name,Request Count,Failure Count,Median,95%ile,...)
    REQUEST_COUNT=$(echo "$AGGREGATED" | cut -d',' -f3)
    FAILURE_COUNT=$(echo "$AGGREGATED" | cut -d',' -f4)
    P95_LATENCY=$(echo "$AGGREGATED" | cut -d',' -f7)

    # Calculate RPS (from output log)
    RPS=$(grep -oP 'Total requests per second:.*\K[0-9.]+' "$RESULTS_DIR/output.log" | tail -1 || echo "0")

    # Calculate success rate
    if [ "$REQUEST_COUNT" -gt 0 ]; then
        SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", (1 - $FAILURE_COUNT / $REQUEST_COUNT) * 100}")
    else
        SUCCESS_RATE="0"
    fi

    echo "Performance Metrics:"
    echo "  Total Requests:   $REQUEST_COUNT"
    echo "  Failed Requests:  $FAILURE_COUNT"
    echo "  Success Rate:     ${SUCCESS_RATE}%"
    echo "  P95 Latency:      ${P95_LATENCY}ms"
    echo "  Requests/sec:     $RPS"
    echo ""

    # Validate targets
    TARGETS_MET=true

    echo "Target Validation:"

    # Target 1: 10,000+ requests per second
    if (( $(echo "$RPS >= 10000" | bc -l) )); then
        echo "  ✅ Throughput: $RPS req/s (target: 10,000+)"
    else
        echo "  ❌ Throughput: $RPS req/s (target: 10,000+) - FAILED"
        TARGETS_MET=false
    fi

    # Target 2: <100ms P95 latency
    if (( $(echo "$P95_LATENCY < 100" | bc -l) )); then
        echo "  ✅ P95 Latency: ${P95_LATENCY}ms (target: <100ms)"
    else
        echo "  ❌ P95 Latency: ${P95_LATENCY}ms (target: <100ms) - FAILED"
        TARGETS_MET=false
    fi

    # Target 3: 99.9% success rate
    if (( $(echo "$SUCCESS_RATE >= 99.9" | bc -l) )); then
        echo "  ✅ Success Rate: ${SUCCESS_RATE}% (target: 99.9%+)"
    else
        echo "  ❌ Success Rate: ${SUCCESS_RATE}% (target: 99.9%+) - FAILED"
        TARGETS_MET=false
    fi

    echo ""

    # Final result
    if [ "$TARGETS_MET" = true ]; then
        echo "=================================================="
        echo "  ✅ ALL PERFORMANCE TARGETS MET"
        echo "=================================================="
        echo ""
        echo "Results saved to: $RESULTS_DIR"
        echo "  - HTML Report: $RESULTS_DIR/report.html"
        echo "  - CSV Stats: $RESULTS_DIR/stats_stats.csv"
        echo "  - Output Log: $RESULTS_DIR/output.log"
        echo ""
        exit 0
    else
        echo "=================================================="
        echo "  ❌ PERFORMANCE TARGETS NOT MET"
        echo "=================================================="
        echo ""
        echo "Review results at: $RESULTS_DIR/report.html"
        echo ""
        echo "Troubleshooting:"
        echo "  - Check server logs for errors"
        echo "  - Verify database connectivity"
        echo "  - Monitor server resource usage"
        echo "  - Consider horizontal scaling"
        echo ""
        exit 1
    fi
else
    echo "❌ ERROR: Stats file not found"
    echo "   Check $RESULTS_DIR/output.log for errors"
    exit 1
fi
