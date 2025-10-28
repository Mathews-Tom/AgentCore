# DSPy Optimization Best Practices

**Version:** 1.0
**Last Updated:** 2025-10-29

This document provides comprehensive best practices for DSPy optimization within AgentCore, derived from pattern analysis, historical optimization results, and expert knowledge.

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm Selection](#algorithm-selection)
3. [Parameter Tuning](#parameter-tuning)
4. [Performance Optimization](#performance-optimization)
5. [Cost Optimization](#cost-optimization)
6. [Resource Management](#resource-management)
7. [Quality Assurance](#quality-assurance)
8. [Workflow Design](#workflow-design)
9. [Anti-Patterns to Avoid](#anti-patterns)
10. [Integration Guide](#integration-guide)

---

## Overview

The insights module provides automated pattern recognition, recommendation generation, and knowledge management for DSPy optimization. Key components:

- **Knowledge Base**: Stores historical patterns, lessons learned, and best practices
- **Recommendation Engine**: Context-aware recommendations based on targets and constraints
- **Best Practice Extractor**: Automatically extracts actionable practices from results
- **Pattern Recognizer**: Identifies successful optimization strategies

### Core Principles

1. **Evidence-Based Decisions**: All recommendations backed by historical data
2. **Context Awareness**: Consider target type, constraints, and objectives
3. **Continuous Learning**: System learns from every optimization run
4. **Confidence Tracking**: Track reliability of patterns and recommendations

---

## Algorithm Selection

### MIPROv2: Proven Reliability

**Use When:**
- Target success rate improvement >= 20%
- Time budget allows 100-200 iterations
- General-purpose optimization needed
- Proven track record in similar contexts

**Configuration:**
```python
from agentcore.dspy_optimization import optimize_target

result = await optimize_target(
    target=optimization_target,
    algorithms=["miprov2"],
    constraints=OptimizationConstraints(
        max_optimization_time=3600,
        min_improvement_threshold=0.20
    )
)
```

**Do:**
- ✅ Use as default algorithm for most optimization tasks
- ✅ Monitor convergence after 50 iterations
- ✅ Start with default parameters and tune incrementally
- ✅ Validate improvements with statistical tests

**Don't:**
- ❌ Skip baseline measurement before optimization
- ❌ Terminate early without validation
- ❌ Use for extremely time-constrained scenarios
- ❌ Ignore convergence warnings

### GEPA: Fast Convergence

**Use When:**
- Time constraints are tight (< 30 minutes)
- Quick iteration needed
- Good-enough results acceptable
- Exploratory optimization phase

**Configuration:**
```python
result = await optimize_target(
    target=optimization_target,
    algorithms=["gepa"],
    constraints=OptimizationConstraints(
        max_optimization_time=1800,
        min_improvement_threshold=0.15
    )
)
```

**Do:**
- ✅ Use early stopping for efficiency
- ✅ Set reasonable iteration limits (30-50)
- ✅ Monitor resource usage closely
- ✅ Validate results despite speed

**Don't:**
- ❌ Expect highest possible improvements
- ❌ Use for critical production optimizations
- ❌ Skip statistical significance testing
- ❌ Ignore quality metrics

### Algorithm Selection Matrix

| Scenario | Recommended Algorithm | Rationale |
|----------|----------------------|-----------|
| Production deployment | MIPROv2 | Proven reliability, high success rate |
| Rapid experimentation | GEPA | Fast convergence, good-enough results |
| High-stakes optimization | MIPROv2 + Extended time | Maximum quality and improvement |
| Resource-constrained | GEPA | Lower resource consumption |
| Complex multi-objective | MIPROv2 | Better handling of trade-offs |

---

## Parameter Tuning

### Temperature Settings

**Recommended Range:** 0.5 - 0.8

```python
optimization_details = OptimizationDetails(
    algorithm_used="miprov2",
    iterations=100,
    parameters={
        "temperature": 0.7,  # Balanced exploration/exploitation
        "top_p": 0.9,
        "max_tokens": 2048
    }
)
```

**Guidelines:**
- **Low (0.5-0.6)**: Conservative, deterministic outputs
- **Medium (0.7)**: Balanced, recommended default
- **High (0.8+)**: Creative, exploratory outputs

### Iteration Limits

**Recommended Ranges:**

| Optimization Goal | Iterations | Time Estimate |
|------------------|------------|---------------|
| Quick validation | 30-50 | 15-30 min |
| Standard optimization | 100-150 | 45-90 min |
| Thorough optimization | 200-300 | 2-4 hours |
| Exhaustive search | 500+ | 6+ hours |

### Parameter Combination Patterns

Based on historical data, these combinations show 80%+ success rates:

1. **Balanced Strategy**
   ```python
   {
       "temperature": 0.7,
       "iterations": 100,
       "early_stopping": True,
       "patience": 10
   }
   ```

2. **Aggressive Strategy**
   ```python
   {
       "temperature": 0.8,
       "iterations": 200,
       "exploration_rate": 0.3,
       "early_stopping": False
   }
   ```

3. **Conservative Strategy**
   ```python
   {
       "temperature": 0.6,
       "iterations": 150,
       "early_stopping": True,
       "patience": 5
   }
   ```

**Do:**
- ✅ Start with proven combinations
- ✅ Change one parameter at a time
- ✅ Track parameter impact with A/B tests
- ✅ Document parameter choices and rationale

**Don't:**
- ❌ Make multiple simultaneous changes
- ❌ Skip validation after tuning
- ❌ Use arbitrary parameter values
- ❌ Ignore convergence patterns

---

## Performance Optimization

### Fast Convergence Techniques

**Target:** Achieve results in < 30 minutes

1. **Early Stopping**
   ```python
   constraints = OptimizationConstraints(
       max_optimization_time=1800,
       min_improvement_threshold=0.15
   )

   parameters = {
       "early_stopping": True,
       "patience": 5,
       "min_delta": 0.01
   }
   ```

2. **Iteration Limits**
   - Set realistic limits based on time budget
   - Monitor convergence every 10 iterations
   - Stop when improvement plateaus

3. **Parallel Evaluation**
   ```python
   parameters = {
       "parallel_trials": 4,  # If resources allow
       "batch_size": "medium"
   }
   ```

### High-Impact Optimization

**Target:** Achieve 30%+ improvement

1. **Extended Search**
   ```python
   constraints = OptimizationConstraints(
       max_optimization_time=7200,  # 2 hours
       min_improvement_threshold=0.30
   )

   parameters = {
       "iterations": 250,
       "exploration_rate": 0.3,
       "early_stopping": False
   }
   ```

2. **Thorough Validation**
   - Use statistical significance testing
   - Validate on holdout dataset
   - Monitor multiple metrics simultaneously

3. **Resource Allocation**
   - Allocate sufficient compute resources
   - Use dedicated optimization environments
   - Monitor system performance continuously

**Do:**
- ✅ Set realistic time budgets
- ✅ Use early stopping for efficiency
- ✅ Monitor convergence metrics
- ✅ Validate improvements thoroughly

**Don't:**
- ❌ Sacrifice quality for speed without validation
- ❌ Over-optimize beyond practical benefits
- ❌ Ignore resource consumption
- ❌ Skip significance testing

---

## Cost Optimization

### ROI-Focused Strategies

**Goal:** Maximize improvement per resource unit

1. **Efficient Algorithms**
   - Use GEPA for cost-sensitive scenarios
   - Set iteration limits based on budget
   - Monitor resource usage real-time

2. **Smart Early Stopping**
   ```python
   parameters = {
       "early_stopping": True,
       "patience": 5,
       "min_delta": 0.02,  # Stop if improvement < 2%
       "max_iterations": 100
   }
   ```

3. **Resource Monitoring**
   ```python
   constraints = OptimizationConstraints(
       max_resource_usage=0.15,  # 15% of available resources
       max_optimization_time=3600
   )
   ```

### Cost-Performance Trade-offs

| Strategy | Cost | Performance | Use Case |
|----------|------|-------------|----------|
| Quick & Cheap | Low | Good | Exploration, validation |
| Balanced | Medium | Very Good | Standard optimization |
| Premium | High | Excellent | Production, critical systems |

**Do:**
- ✅ Set resource limits proactively
- ✅ Use early stopping for ROI
- ✅ Monitor cost metrics continuously
- ✅ Balance cost and improvement targets

**Don't:**
- ❌ Run excessive iterations for diminishing returns
- ❌ Ignore resource consumption metrics
- ❌ Optimize without clear ROI targets
- ❌ Skip cost-benefit analysis

---

## Resource Management

### Memory Management

1. **Batch Size Optimization**
   ```python
   parameters = {
       "batch_size": "small",  # For limited memory
       "gradient_accumulation_steps": 4
   }
   ```

2. **Model Caching**
   - Cache intermediate results
   - Reuse computed gradients
   - Clear cache periodically

### Compute Allocation

1. **Parallel Execution**
   ```python
   parameters = {
       "parallel_trials": min(cpu_count, 8),
       "workers": 4
   }
   ```

2. **Time Management**
   - Set realistic time budgets
   - Use timeouts for safety
   - Monitor execution time per iteration

**Do:**
- ✅ Profile resource usage before optimization
- ✅ Set appropriate resource limits
- ✅ Use parallel execution when available
- ✅ Monitor and log resource metrics

**Don't:**
- ❌ Exceed available resources
- ❌ Run without monitoring
- ❌ Ignore memory warnings
- ❌ Use unlimited timeouts

---

## Quality Assurance

### Validation Framework

1. **Statistical Significance**
   ```python
   from agentcore.dspy_optimization.analytics import ImprovementAnalyzer

   analyzer = ImprovementAnalyzer()
   validation = await analyzer.validate_improvement(
       baseline_metrics=baseline,
       optimized_metrics=optimized,
       sample_size=100,
       significance_level=0.05
   )

   if validation.is_statistically_significant:
       # Deploy optimization
   ```

2. **Holdout Validation**
   - Reserve 20% data for validation
   - Test on unseen examples
   - Compare multiple metrics

3. **A/B Testing**
   ```python
   from agentcore.dspy_optimization.testing import ABTester

   tester = ABTester()
   result = await tester.compare_versions(
       baseline_version="v1.0",
       optimized_version="v1.1",
       traffic_split=0.5,
       duration_hours=24
   )
   ```

### Quality Metrics

Track these metrics for every optimization:

- **Success Rate**: Task completion rate
- **Cost Efficiency**: Cost per successful task
- **Latency**: Response time (p50, p95, p99)
- **Quality Score**: Output quality rating
- **Statistical Significance**: p-value < 0.05

**Do:**
- ✅ Validate all improvements statistically
- ✅ Use holdout data for testing
- ✅ Track multiple quality metrics
- ✅ Run A/B tests before deployment

**Don't:**
- ❌ Deploy without validation
- ❌ Rely on single metric
- ❌ Skip significance testing
- ❌ Use training data for validation

---

## Workflow Design

### Standard Optimization Workflow

```python
from agentcore.dspy_optimization import (
    OptimizationPipeline,
    PatternRecognizer,
    KnowledgeBase,
    RecommendationEngine
)

# 1. Initialize components
pipeline = OptimizationPipeline()
knowledge_base = KnowledgeBase()
recognizer = PatternRecognizer()
engine = RecommendationEngine(recognizer, knowledge_base)

# 2. Get recommendations
context = RecommendationContext(
    target=target,
    objectives=objectives,
    constraints=constraints
)
recommendations = await engine.generate_recommendations(context)

# 3. Execute optimization
result = await pipeline.optimize(
    target=target,
    algorithms=["miprov2"],
    constraints=constraints
)

# 4. Validate results
if result.improvement_percentage >= 0.20:
    validation = await analyzer.validate_improvement(
        baseline_metrics=result.baseline_performance,
        optimized_metrics=result.optimized_performance
    )

# 5. Update knowledge base
if validation.is_statistically_significant:
    await knowledge_base.learn_from_results([result])
```

### Continuous Learning Loop

1. **Collect Data**: Track all optimization runs
2. **Analyze Patterns**: Extract successful strategies
3. **Update Knowledge**: Store lessons learned
4. **Generate Recommendations**: Use patterns for future optimizations
5. **Validate & Iterate**: Continuous improvement

**Do:**
- ✅ Follow systematic workflows
- ✅ Update knowledge base continuously
- ✅ Track optimization history
- ✅ Use recommendations for new optimizations

**Don't:**
- ❌ Skip workflow steps
- ❌ Ignore historical patterns
- ❌ Reinvent strategies each time
- ❌ Forget to update knowledge base

---

## Anti-Patterns to Avoid

### 1. Premature Optimization

**Problem:** Optimizing before establishing baseline
**Impact:** Cannot measure actual improvement
**Solution:** Always measure baseline performance first

### 2. Over-Optimization

**Problem:** Continuing optimization beyond practical benefits
**Impact:** Wasted resources, diminishing returns
**Solution:** Set clear improvement targets and stop when reached

### 3. Ignoring Statistical Significance

**Problem:** Deploying improvements without validation
**Impact:** False positives, unreliable results
**Solution:** Always validate with statistical tests

### 4. Single-Metric Optimization

**Problem:** Optimizing only one metric
**Impact:** Trade-offs and side effects ignored
**Solution:** Track multiple metrics simultaneously

### 5. No Monitoring

**Problem:** Running optimization without monitoring
**Impact:** Resource exhaustion, missed issues
**Solution:** Monitor resources and progress continuously

### 6. Parameter Soup

**Problem:** Changing multiple parameters simultaneously
**Impact:** Cannot identify what works
**Solution:** Change one parameter at a time

### 7. Ignoring Context

**Problem:** Using same strategy for all targets
**Impact:** Suboptimal results
**Solution:** Use context-aware recommendations

### 8. No Documentation

**Problem:** Not documenting optimization decisions
**Impact:** Cannot reproduce or learn from results
**Solution:** Document all parameters and rationale

---

## Integration Guide

### Basic Usage

```python
from agentcore.dspy_optimization.insights import (
    KnowledgeBase,
    RecommendationEngine,
    BestPracticeExtractor
)
from agentcore.dspy_optimization.analytics import PatternRecognizer

# Initialize components
kb = KnowledgeBase(storage_path=Path("knowledge.json"))
recognizer = PatternRecognizer()
engine = RecommendationEngine(recognizer, kb)
extractor = BestPracticeExtractor()

# Get recommendations
context = RecommendationContext(
    target=your_target,
    objectives=your_objectives,
    constraints=your_constraints
)
recommendations = await engine.generate_recommendations(context)

# Use top recommendation
if recommendations:
    top_rec = recommendations[0]
    print(f"Recommendation: {top_rec.title}")
    print(f"Confidence: {top_rec.confidence:.1%}")
    print(f"Rationale: {top_rec.rationale}")
```

### Pattern Analysis

```python
# Analyze historical results
patterns = await recognizer.analyze_patterns(
    results=optimization_results,
    min_improvement_threshold=0.20
)

# Add patterns to knowledge base
for pattern in patterns:
    await kb.add_pattern(pattern)

# Extract best practices
practices = await extractor.extract_from_patterns(
    patterns=patterns,
    results=optimization_results
)

# Generate documentation
markdown = extractor.generate_markdown_documentation(
    title="Our DSPy Best Practices"
)
Path("best_practices.md").write_text(markdown)
```

### Learning from Results

```python
# Learn automatically
new_entries = await kb.learn_from_results(
    results=optimization_results,
    min_sample_size=5
)

print(f"Learned {len(new_entries)} new insights")

# Get statistics
stats = kb.get_statistics()
print(f"Total knowledge entries: {stats['total_entries']}")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
```

### Export/Import Knowledge

```python
# Export knowledge base
await kb.export_knowledge(Path("knowledge_export.json"))

# Import to another system
new_kb = KnowledgeBase()
count = await new_kb.import_knowledge(Path("knowledge_export.json"))
print(f"Imported {count} entries")
```

---

## Appendix: Metrics Reference

### Performance Metrics

- **Success Rate**: Percentage of successful task completions
- **Avg Cost Per Task**: Average cost in tokens/API calls
- **Avg Latency**: Average response time in milliseconds
- **Quality Score**: Normalized output quality rating (0-1)

### Improvement Metrics

- **Improvement Percentage**: Overall improvement across all metrics
- **Statistical Significance**: p-value from significance test
- **Confidence Interval**: Range of expected improvement

### Pattern Metrics

- **Pattern Confidence**: Reliability of identified pattern
- **Success Rate**: Pattern success rate in historical data
- **Sample Count**: Number of observations supporting pattern

---

## Support & Resources

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Full API reference available in `/docs`
- **Examples**: Sample code in `/examples`
- **Community**: Join discussions in GitHub Discussions

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Maintained By:** AgentCore Team
