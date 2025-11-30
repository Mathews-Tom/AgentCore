#!/usr/bin/env bash
#
# Load Test Runner for Modular Agent Core (MOD-028)
#
# Executes load test scenarios and generates comprehensive reports.
#
# Usage:
#   ./tests/load/run_modular_load_test.sh [scenario]
#
# Scenarios:
#   quick     - Quick validation (10 users, 2 min)
#   standard  - Standard load test (50 users, 5 min)
#   full      - Full load test (100 users, 10 min) [default]
#   stress    - Stress test (150 users, 15 min)

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Test scenarios
declare -A SCENARIOS=(
    [quick]="10 5 2m"
    [standard]="50 10 5m"
    [full]="100 10 10m"
    [stress]="150 15 15m"
)

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check if locust is installed
    if ! command -v locust &> /dev/null; then
        log_error "Locust is not installed. Install it via: uv add locust"
        exit 1
    fi

    # Check if server is running
    if ! curl -s -f http://localhost:8001/api/v1/health > /dev/null 2>&1; then
        log_error "AgentCore server is not running on http://localhost:8001"
        log_info "Start server with: uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001"
        exit 1
    fi

    log_success "All dependencies satisfied"
}

create_results_dir() {
    mkdir -p "${RESULTS_DIR}"
    log_info "Results will be saved to: ${RESULTS_DIR}"
}

run_load_test() {
    local scenario=$1
    local users=$2
    local spawn_rate=$3
    local run_time=$4

    log_info "Starting ${scenario} load test..."
    log_info "  Users: ${users}"
    log_info "  Spawn Rate: ${spawn_rate}/s"
    log_info "  Duration: ${run_time}"
    echo ""

    # Generate output file names
    local csv_prefix="${RESULTS_DIR}/modular_load_${scenario}_${TIMESTAMP}"
    local html_report="${RESULTS_DIR}/modular_load_${scenario}_${TIMESTAMP}.html"

    # Run locust load test
    if uv run locust \
        -f "${SCRIPT_DIR}/test_modular_load.py" \
        --host=http://localhost:8001 \
        --users="${users}" \
        --spawn-rate="${spawn_rate}" \
        --run-time="${run_time}" \
        --headless \
        --csv="${csv_prefix}" \
        --html="${html_report}" \
        --loglevel=INFO; then

        log_success "Load test completed successfully"
        echo ""
        log_info "Results saved:"
        log_info "  - CSV: ${csv_prefix}_*.csv"
        log_info "  - HTML Report: ${html_report}"
        echo ""

        # Display quick summary
        display_summary "${csv_prefix}_stats.csv"

        return 0
    else
        log_error "Load test failed"
        return 1
    fi
}

display_summary() {
    local stats_file=$1

    if [[ -f "${stats_file}" ]]; then
        log_info "Test Summary:"
        echo ""

        # Parse CSV and display key metrics
        # Skip header line and process stats
        tail -n +2 "${stats_file}" | head -n 1 | while IFS=, read -r type name request_count failure_count median_response_time avg_response_time min_response_time max_response_time avg_content_size requests_per_s failures_per_s p50 p66 p75 p80 p90 p95 p98 p99 p999 p100; do
            echo "  Total Requests:        ${request_count}"
            echo "  Failed Requests:       ${failure_count}"
            echo "  Success Rate:          $(awk "BEGIN {printf \"%.2f%%\", (${request_count}-${failure_count})/${request_count}*100}")"
            echo "  Avg Response Time:     ${avg_response_time}ms"
            echo "  p95 Response Time:     ${p95}ms"
            echo "  p99 Response Time:     ${p99}ms"
            echo "  Requests/sec:          ${requests_per_s}"
        done
        echo ""
    fi
}

validate_nfr_targets() {
    local stats_file=$1

    log_info "Validating NFR targets..."
    echo ""

    if [[ ! -f "${stats_file}" ]]; then
        log_warning "Stats file not found: ${stats_file}"
        return 1
    fi

    # Extract metrics from CSV
    local metrics=$(tail -n +2 "${stats_file}" | head -n 1)
    local request_count=$(echo "${metrics}" | cut -d',' -f3)
    local failure_count=$(echo "${metrics}" | cut -d',' -f4)
    local p95=$(echo "${metrics}" | cut -d',' -f16)

    # Calculate success rate
    local success_rate=$(awk "BEGIN {printf \"%.2f\", (${request_count}-${failure_count})/${request_count}*100}")

    # Baseline p95 from benchmarks (1500ms)
    local baseline_p95=1500
    local target_p95=$((baseline_p95 * 3))

    echo "  NFR Target Validation:"
    echo "  ---------------------"

    # Check success rate (>95%)
    if (( $(echo "${success_rate} >= 95.0" | bc -l) )); then
        log_success "Success Rate: ${success_rate}% (Target: >95%)"
    else
        log_error "Success Rate: ${success_rate}% (Target: >95%)"
    fi

    # Check p95 latency (<3x baseline)
    if (( $(echo "${p95} < ${target_p95}" | bc -l) )); then
        log_success "p95 Latency: ${p95}ms (Target: <${target_p95}ms)"
    else
        log_error "p95 Latency: ${p95}ms (Target: <${target_p95}ms)"
    fi

    echo ""
}

main() {
    local scenario=${1:-full}

    # Validate scenario
    if [[ ! -v "SCENARIOS[${scenario}]" ]]; then
        log_error "Invalid scenario: ${scenario}"
        echo ""
        echo "Available scenarios:"
        echo "  quick     - Quick validation (10 users, 2 min)"
        echo "  standard  - Standard load test (50 users, 5 min)"
        echo "  full      - Full load test (100 users, 10 min) [default]"
        echo "  stress    - Stress test (150 users, 15 min)"
        exit 1
    fi

    # Parse scenario parameters
    IFS=' ' read -r users spawn_rate run_time <<< "${SCENARIOS[${scenario}]}"

    echo ""
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Modular Agent Core Load Test (MOD-028)"
    log_info "Scenario: ${scenario}"
    log_info "═══════════════════════════════════════════════════════════"
    echo ""

    # Pre-flight checks
    check_dependencies
    create_results_dir

    echo ""

    # Run load test
    if run_load_test "${scenario}" "${users}" "${spawn_rate}" "${run_time}"; then
        # Validate NFR targets
        local csv_prefix="${RESULTS_DIR}/modular_load_${scenario}_${TIMESTAMP}"
        validate_nfr_targets "${csv_prefix}_stats.csv"

        log_success "Load test completed successfully!"
        exit 0
    else
        log_error "Load test failed!"
        exit 1
    fi
}

# Execute main function
main "$@"
