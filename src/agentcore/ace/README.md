# ACE (Agent Context Engineering) - Capability Evaluation Module

**Version:** 2.0.0 (COMPASS-Enhanced)
**Status:** Phase 5 Complete (ACE-025 to ACE-028)

## Overview

The ACE Capability Evaluation module implements **COMPASS ACE-4: Dynamic Capability Evaluation** for the AgentCore Meta-Thinker system. It provides intelligent assessment of agent capabilities against task requirements, identifying gaps and recommending optimal capability configurations.

## Features

### ✅ Implemented (Phase 5)

#### 1. **CapabilityEvaluator** (ACE-025)
Task-capability fitness evaluation with multi-dimensional scoring:
- **Fitness scoring**: 0-1 scale evaluation combining coverage, performance, and efficiency
- **Gap identification**: Automatic detection of missing or underperforming capabilities
- **Performance tracking**: History-based scoring from execution metrics
- **Caching**: Built-in LRU cache for <100ms evaluation latency

```python
from agentcore.ace.capability import CapabilityEvaluator

evaluator = CapabilityEvaluator()

fitness = await evaluator.evaluate_fitness(
    agent_id="agent-001",
    capability_id="api_client",
    capability_name="API Client",
    task_requirements=[...],
    performance_history={...}
)

print(f"Fitness: {fitness.fitness_score:.2f}")  # 0.85
print(f"Level: {fitness.fitness_level}")  # excellent/good/acceptable/poor
```

#### 2. **FitnessScorer** (ACE-026)
Multi-factor scoring with weighted components:
- **Coverage score**: Task requirement matching via semantic similarity
- **Performance score**: Success rate, error correlation, consistency
- **Efficiency score**: Time budget adherence, resource utilization
- **Trend analysis**: Historical performance trajectory detection

```python
from agentcore.ace.capability.fitness_scorer import FitnessScorer

scorer = FitnessScorer(
    coverage_weight=0.4,
    performance_weight=0.4,
    efficiency_weight=0.2
)

trend = scorer.compute_fitness_trend(historical_scores)
# Returns: {trend_direction: 1/-1/0, trend_strength: 0-1, ...}
```

#### 3. **CapabilityRecommender** (ACE-027)
Intelligent recommendation engine with risk assessment:
- **Addition recommendations**: Missing capabilities for task requirements
- **Removal recommendations**: Underperforming capabilities (fitness < 0.3)
- **Upgrade recommendations**: Moderate performers (0.3 <= fitness < 0.5)
- **Risk assessment**: Low/medium/high classification
- **Confidence scoring**: Data-driven recommendation confidence

```python
from agentcore.ace.capability.recommender import CapabilityRecommender

recommender = CapabilityRecommender(
    fitness_threshold=0.5,
    removal_threshold=0.3
)

recommendation = await recommender.recommend_capability_changes(
    agent_id="agent-001",
    current_capabilities=[...],
    fitness_scores={...},
    capability_gaps=[...]
)

print(f"Add: {recommendation.capabilities_to_add}")
print(f"Remove: {recommendation.capabilities_to_remove}")
print(f"Confidence: {recommendation.confidence:.2f}")
print(f"Risk: {recommendation.risk_level}")
```

#### 4. **Comprehensive Testing** (ACE-028)
- **Unit tests**: 95%+ coverage for all components
- **Integration tests**: End-to-end workflow validation
- **Performance tests**: Latency target verification
- **Edge cases**: Empty capabilities, no requirements, extreme values

## Architecture

```
src/agentcore/ace/
├── __init__.py                      # Package initialization
├── models/
│   ├── __init__.py
│   └── ace_models.py                # Pydantic models (CapabilityFitness, etc.)
├── capability/
│   ├── __init__.py
│   ├── evaluator.py                 # CapabilityEvaluator (ACE-025)
│   ├── fitness_scorer.py            # FitnessScorer (ACE-026)
│   └── recommender.py               # CapabilityRecommender (ACE-027)
└── database/                        # (Reserved for future DB integration)

tests/ace/
├── unit/
│   ├── test_capability_evaluator.py # Evaluator tests
│   ├── test_fitness_scorer.py       # Scorer tests
│   └── test_recommender.py          # Recommender tests
└── integration/
    └── test_capability_workflows.py # End-to-end tests
```

## Data Models

### CapabilityFitness
```python
class CapabilityFitness(BaseModel):
    capability_id: str
    capability_name: str
    agent_id: str

    fitness_score: float          # Overall 0-1 score
    coverage_score: float         # Task requirement coverage
    performance_score: float      # Execution performance

    metrics: FitnessMetrics       # Detailed metrics
    sample_size: int              # Evaluation sample size
    evaluated_at: datetime

    @property
    def is_fit(self) -> bool:     # >= 0.5
    @property
    def fitness_level(self) -> str # excellent/good/acceptable/poor
```

### CapabilityGap
```python
class CapabilityGap(BaseModel):
    required_capability: str
    capability_type: CapabilityType
    current_fitness: float | None
    required_fitness: float
    impact: float                 # 0-1 task impact
    gap_severity: str             # critical/high/medium/low
    mitigation_suggestion: str
```

### CapabilityRecommendation
```python
class CapabilityRecommendation(BaseModel):
    agent_id: str
    current_capabilities: list[str]

    capabilities_to_add: list[str]
    capabilities_to_remove: list[str]
    capabilities_to_upgrade: list[dict]

    identified_gaps: list[CapabilityGap]
    rationale: str
    confidence: float
    expected_improvement: float
    risk_level: str
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Fitness evaluation latency | <100ms (p95) | ✅ Met (with caching) |
| Recommendation generation | <500ms (p95) | ✅ Met |
| Fitness score range | 0.0 - 1.0 | ✅ Enforced |
| Test coverage | 95%+ | ✅ Achieved |

## Usage Examples

### Complete Evaluation Workflow

```python
from agentcore.ace.capability import CapabilityEvaluator
from agentcore.ace.capability.recommender import CapabilityRecommender
from agentcore.ace.models import TaskRequirement, CapabilityType

# 1. Define task requirements
requirements = [
    TaskRequirement(
        requirement_id="req-1",
        capability_type=CapabilityType.API,
        capability_name="api_client",
        required=True,
        weight=1.0
    )
]

# 2. Evaluate current capabilities
evaluator = CapabilityEvaluator()
current_caps = [
    {"id": "api_client", "name": "API Client"},
    {"id": "old_parser", "name": "Old Parser"}
]

fitness_scores = await evaluator.evaluate_all_capabilities(
    agent_id="agent-001",
    current_capabilities=current_caps,
    task_requirements=requirements
)

# 3. Identify gaps
gaps = await evaluator.identify_capability_gaps(
    agent_id="agent-001",
    current_capabilities=[c["id"] for c in current_caps],
    task_requirements=requirements,
    fitness_scores=fitness_scores
)

# 4. Generate recommendations
recommender = CapabilityRecommender()
recommendation = await recommender.recommend_capability_changes(
    agent_id="agent-001",
    current_capabilities=[c["id"] for c in current_caps],
    fitness_scores=fitness_scores,
    capability_gaps=gaps
)

# 5. Act on recommendations
if recommendation.has_critical_gaps:
    print(f"⚠️  Critical gaps: {recommendation.capabilities_to_add}")
if recommendation.confidence > 0.7:
    print(f"✅ High-confidence recommendation (risk: {recommendation.risk_level})")
```

## Testing

```bash
# Run all ACE tests
uv run pytest tests/ace/ -v

# Run unit tests only
uv run pytest tests/ace/unit/ -v

# Run integration tests
uv run pytest tests/ace/integration/ -v

# Check coverage
uv run pytest tests/ace/ --cov=src/agentcore/ace --cov-report=term-missing
```

## Configuration

The evaluator supports configuration for performance tuning:

```python
# Custom cache TTL
evaluator = CapabilityEvaluator()
evaluator._cache_ttl_seconds = 600  # 10 minutes

# Custom scoring weights
from agentcore.ace.capability.fitness_scorer import FitnessScorer

scorer = FitnessScorer(
    coverage_weight=0.5,    # Prioritize task coverage
    performance_weight=0.3,
    efficiency_weight=0.2
)

# Custom recommendation thresholds
from agentcore.ace.capability.recommender import CapabilityRecommender

recommender = CapabilityRecommender(
    fitness_threshold=0.6,      # Stricter fitness requirement
    removal_threshold=0.2,      # More aggressive removal
    confidence_threshold=0.8    # Higher confidence required
)
```

## Future Enhancements

**Phase 6: Production Readiness** (ACE-029 to ACE-033)
- [ ] ACE-029: Performance optimization (Redis caching, database integration)
- [ ] ACE-030: COMPASS validation tests (90%+ recall, 85%+ precision)
- [ ] ACE-031: Load testing (100 concurrent agents, 1000 tasks)
- [ ] ACE-032: Documentation (API docs, runbook, architecture diagrams)
- [ ] ACE-033: Production deployment (K8s manifests, monitoring, alerts)

**Potential Extensions:**
- LLM-based semantic capability matching
- Capability registry integration
- Agent Manager coordination for automatic capability updates
- Real-time capability switching
- Capability cost-benefit analysis

## COMPASS Compliance

This implementation follows **COMPASS ACE-4** specification for dynamic capability evaluation:

✅ **Task-capability matching algorithm**: Semantic similarity + coverage scoring
✅ **Fitness score computation**: Multi-factor weighted scoring (0-1 scale)
✅ **Capability gap identification**: Severity-based gap classification
✅ **Performance targets**: <100ms evaluation, <500ms recommendation

## Contributing

When extending this module:
1. Follow existing code patterns (async-first, modern typing)
2. Add comprehensive tests (95%+ coverage target)
3. Update this README with new features
4. Ensure performance targets are met
5. Use semantic commit messages

## References

- **Specification**: `docs/specs/ace-integration/spec.md`
- **Tasks**: `docs/specs/ace-integration/tasks.md`
- **Tickets**: `.sage/tickets/ACE-025.md` through `ACE-028.md`
- **COMPASS Paper**: https://arxiv.org/abs/2510.08790

---

**Status**: ✅ Phase 5 Complete (ACE-025 to ACE-028)
**Last Updated**: 2025-11-08
**Maintainer**: AgentCore Team
