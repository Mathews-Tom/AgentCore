# Modular Agent Metrics Integration Example

This document demonstrates how to integrate the Modular Agent Metrics with the FastAPI application to expose Prometheus metrics at `/metrics`.

## 1. Integration with Main Application

Add to `src/agentcore/a2a_protocol/main.py`:

```python
from prometheus_client import make_asgi_app
from agentcore.modular.metrics import get_metrics

# Initialize metrics on startup
@app.on_event("startup")
async def startup_metrics():
    """Initialize modular agent metrics."""
    from agentcore.modular.metrics import get_metrics
    metrics = get_metrics()
    logger.info("modular_metrics_initialized")

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## 2. Integration with Coordinator

Update `src/agentcore/modular/coordinator.py` to use metrics:

```python
from agentcore.modular.metrics import get_metrics

class ModuleCoordinator:
    def __init__(self) -> None:
        # ... existing initialization ...
        self._metrics = get_metrics()

    async def execute_with_refinement(
        self,
        query: str,
        planner: Any,
        executor: Any,
        verifier: Any,
        generator: Any,
        max_iterations: int = 5,
        timeout_seconds: float = 300.0,
        confidence_threshold: float = 0.7,
        output_format: str = "text",
        include_reasoning: bool = False,
    ) -> dict[str, Any]:
        """Execute with metrics tracking."""

        # Track overall coordination
        async with self._metrics.track_coordination() as coord_tracker:
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Planning with metrics
                async with self._metrics.track_module_execution("planner") as plan_tracker:
                    if iteration == 1:
                        plan = await planner.analyze_query(planner_query)
                    else:
                        plan = await planner.refine_plan(refinement_request)

                    plan_tracker.set_success(True)
                    # Set tokens and cost if available from LLM response
                    if hasattr(plan, 'tokens_used'):
                        plan_tracker.set_tokens(plan.tokens_used)
                    if hasattr(plan, 'cost_usd'):
                        plan_tracker.set_cost(plan.cost_usd)

                # Execution with metrics
                async with self._metrics.track_module_execution("executor") as exec_tracker:
                    execution_results = await executor.execute_plan(plan)
                    exec_tracker.set_success(True)
                    # Set metrics from execution results

                # Verification with metrics
                async with self._metrics.track_module_execution("verifier") as verify_tracker:
                    verification_result = await verifier.validate_results(verification_request)
                    verify_tracker.set_success(True)

                # Record verification result
                self._metrics.record_verification_result(
                    passed=verification_result.valid,
                    confidence=verification_result.confidence
                )
                self._metrics.record_iteration_confidence(
                    iteration=iteration,
                    confidence=verification_result.confidence
                )

                # Check if done
                if verification_result.valid and verification_result.confidence >= confidence_threshold:
                    break

            # Generation with metrics
            async with self._metrics.track_module_execution("generator") as gen_tracker:
                generated_response = await generator.synthesize_response(generation_request)
                gen_tracker.set_success(True)

            # Set coordination success
            coord_tracker.set_success(True)
            coord_tracker.set_iterations(iteration)
            # Set total cost if tracked

            # Record final iteration count
            self._metrics.record_iteration_count(iteration)

            # Return result...
```

## 3. Integration with Individual Modules

Each module can track its own internal operations:

```python
from agentcore.modular.metrics import get_metrics

class PlannerModule:
    def __init__(self):
        self._metrics = get_metrics()

    async def analyze_query(self, query: PlannerQuery) -> ExecutionPlan:
        async with self._metrics.track_module_execution("planner") as tracker:
            # Call LLM
            response = await self._llm_call(query)

            # Track tokens and cost
            tracker.set_success(True)
            tracker.set_tokens(response.usage.total_tokens)
            tracker.set_cost(self._calculate_cost(response.usage))

            return self._parse_plan(response)
```

## 4. Querying Metrics

Once integrated, metrics are available at `http://localhost:8001/metrics`.

### Example Prometheus Queries

**Module Latency (p95):**
```promql
histogram_quantile(0.95,
  sum(rate(modular_agent_module_latency_seconds_bucket[5m])) by (module, le)
)
```

**Success Rate by Module:**
```promql
sum(rate(modular_agent_module_executions_total{status="success"}[5m])) by (module)
/
sum(rate(modular_agent_module_executions_total[5m])) by (module)
```

**Error Rate by Type:**
```promql
sum(rate(modular_agent_module_errors_total[5m])) by (module, error_type)
```

**Average Iteration Count:**
```promql
avg(modular_agent_iteration_count)
```

**Token Usage by Module:**
```promql
sum(rate(modular_agent_module_tokens_total[5m])) by (module, token_type)
```

**Cost per Module (hourly):**
```promql
sum(increase(modular_agent_module_cost_usd_total[1h])) by (module)
```

## 5. Grafana Dashboard Example

Create a dashboard with panels for:

1. **Latency Panel**: Module execution time (p50, p95, p99)
2. **Success Rate Panel**: Success % by module over time
3. **Error Panel**: Errors by type and module
4. **Iteration Panel**: Distribution of iteration counts
5. **Token Panel**: Token usage by module
6. **Cost Panel**: Cost tracking by module

## 6. Alerts Example

```yaml
groups:
  - name: modular_agent_alerts
    rules:
      # High error rate
      - alert: HighModuleErrorRate
        expr: |
          sum(rate(modular_agent_module_errors_total[5m])) by (module)
          > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in {{ $labels.module }}"

      # High latency
      - alert: HighModuleLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(modular_agent_module_latency_seconds_bucket[5m])) by (module, le)
          ) > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p95 latency in {{ $labels.module }}"

      # Low verification success rate
      - alert: LowVerificationSuccessRate
        expr: |
          sum(rate(modular_agent_verification_results_total{result="passed"}[5m]))
          /
          sum(rate(modular_agent_verification_results_total[5m]))
          < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low verification success rate"
```

## 7. Testing Metrics

Run integration tests:

```bash
# Run metrics tests
uv run pytest tests/unit/modular/test_metrics.py -v
uv run pytest tests/integration/modular/test_metrics_integration.py -v

# Check metrics endpoint (with running server)
curl http://localhost:8001/metrics | grep modular_agent
```

## Notes

- Metrics are automatically exported in Prometheus format
- All metrics use labels for filtering (module, status, error_type)
- Histograms use appropriate buckets for agent execution times
- Cost tracking requires LLM response metadata (token counts, model info)
- Metrics persist across requests using the global metrics instance
