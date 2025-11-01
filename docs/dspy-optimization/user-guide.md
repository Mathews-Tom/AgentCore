# DSPy Optimization User Guide

**Version**: 1.0
**Last Updated**: 2025-11-01
**Status**: Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Optimization Algorithms](#optimization-algorithms)
6. [Advanced Features](#advanced-features)
7. [MLflow Integration](#mlflow-integration)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Introduction

### What is DSPy Optimization?

DSPy (Declarative Self-improving Python) optimization is an automated framework for improving LLM prompts and agent behavior without manual tuning. Instead of manually iterating on prompts, DSPy uses algorithms to automatically discover optimal prompts based on your evaluation metrics.

### Why Use DSPy?

**Traditional Prompt Engineering**:

```python
# Manual iteration - slow and error-prone
prompt_v1 = "Summarize this text"
prompt_v2 = "Provide a concise summary of the following text"
prompt_v3 = "Create a brief summary highlighting key points from this text"
# ... hours of trial and error
```

**DSPy Optimization**:

```python
# Automated optimization - fast and data-driven
optimizer = MIPROv2Optimizer()
optimized_prompt = optimizer.optimize(
    task="summarization",
    training_data=examples,
    metric="rouge_score"
)
# Optimal prompt discovered automatically
```

### Key Benefits

✅ **80% faster** than manual prompt engineering
✅ **Data-driven** optimization based on real performance
✅ **Reproducible** results with experiment tracking
✅ **Continuous improvement** with online learning
✅ **Cost optimization** through intelligent model selection

---

## Quick Start

### Prerequisites

```bash
# Ensure AgentCore is installed
uv sync

# Set up environment variables
export MLFLOW_TRACKING_URI="http://localhost:5000"  # Optional
export OPENAI_API_KEY="sk-..."                      # Required
export ANTHROPIC_API_KEY="sk-ant-..."               # Optional
export GOOGLE_API_KEY="AI..."                       # Optional
```

### Your First Optimization

```python
from agentcore.dspy_optimization.pipeline import DSPyOptimizationPipeline
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationObjective,
    MetricType
)

# Initialize the pipeline
pipeline = DSPyOptimizationPipeline()

# Define your optimization request
request = OptimizationRequest(
    # What you're optimizing
    target="summarization_prompt",

    # What you want to achieve
    objective=OptimizationObjective.MAXIMIZE_ACCURACY,

    # How to measure success
    evaluation_metric=MetricType.ROUGE_SCORE,

    # Algorithm to use
    algorithm="miprov2",

    # Stop conditions
    max_iterations=50,
    timeout_seconds=300,

    # Your training examples
    training_examples=[
        {
            "input": "Long text to summarize...",
            "expected_output": "Concise summary..."
        },
        # Add 10-100 examples for best results
    ]
)

# Run optimization
result = await pipeline.optimize(request)

# Use the optimized prompt
print(f"Original accuracy: {result.baseline_metrics.accuracy}")
print(f"Optimized accuracy: {result.optimized_metrics.accuracy}")
print(f"Improvement: {result.improvement_percentage}%")
print(f"Optimized prompt: {result.optimized_prompt}")
```

**Expected Output**:

```
Original accuracy: 0.65
Optimized accuracy: 0.82
Improvement: 26.15%
Optimized prompt: Analyze the following text and create a concise summary
                  highlighting the main points, key findings, and critical
                  conclusions in 2-3 sentences.
```

---

## Core Concepts

### Optimization Pipeline

The DSPy optimization pipeline consists of four stages:

```
1. Baseline Evaluation    →  Measure current performance
2. Optimization Loop      →  Try different prompts/strategies
3. Validation             →  Test on held-out data
4. Deployment             →  Save and track results
```

### Optimization Targets

**What can you optimize?**

| Target Type | Description | Example |
|-------------|-------------|---------|
| `prompt_template` | System/user prompts | "Summarize this text" → optimized version |
| `few_shot_examples` | Example selection | Choose best 5 examples from 100 |
| `reasoning_strategy` | Chain-of-thought patterns | Optimize reasoning steps |
| `tool_selection` | Which tools to use | Auto-select best tools for task |
| `model_selection` | Which LLM to use | GPT-5 vs Claude vs Gemini |

### Evaluation Metrics

**How do you measure success?**

| Metric | Use Case | Range |
|--------|----------|-------|
| `accuracy` | Classification tasks | 0.0 - 1.0 |
| `f1_score` | Balanced precision/recall | 0.0 - 1.0 |
| `rouge_score` | Text summarization | 0.0 - 1.0 |
| `bleu_score` | Translation quality | 0.0 - 1.0 |
| `latency` | Response time | seconds |
| `cost` | API costs | USD |
| `custom` | Your own metric | any |

### Optimization Algorithms

AgentCore supports three main algorithms:

1. **MIPROv2** - Multi-prompt instruction optimization (recommended)
2. **BootstrapFewShot** - Example-based optimization
3. **GEPA** - Genetic evolutionary prompt algorithm

---

## Basic Usage

### Example 1: Text Classification

```python
from agentcore.dspy_optimization.pipeline import DSPyOptimizationPipeline
from agentcore.dspy_optimization.models import *

# Classification task: sentiment analysis
pipeline = DSPyOptimizationPipeline()

request = OptimizationRequest(
    target="sentiment_classifier",
    objective=OptimizationObjective.MAXIMIZE_ACCURACY,
    evaluation_metric=MetricType.F1_SCORE,
    algorithm="miprov2",

    # Training data with labels
    training_examples=[
        {"text": "This product is amazing!", "label": "positive"},
        {"text": "Worst purchase ever.", "label": "negative"},
        {"text": "It's okay, nothing special.", "label": "neutral"},
        # Add 50-100 more examples
    ],

    # Validation data (20% of total)
    validation_examples=[
        {"text": "Great value for money", "label": "positive"},
        # Add validation examples
    ],

    # Constraints
    constraints=OptimizationConstraints(
        max_tokens=100,
        max_cost_usd=5.0,
        min_accuracy=0.75
    )
)

result = await pipeline.optimize(request)

# Save the optimized classifier
await pipeline.save_optimized_model(
    result.model_id,
    output_path="models/sentiment_classifier_v1.pkl"
)
```

### Example 2: Question Answering

```python
# QA task: extract answers from documents
request = OptimizationRequest(
    target="qa_system",
    objective=OptimizationObjective.MAXIMIZE_ACCURACY,
    evaluation_metric=MetricType.EXACT_MATCH,
    algorithm="bootstrapfewshot",

    training_examples=[
        {
            "context": "The Eiffel Tower is 330 meters tall...",
            "question": "How tall is the Eiffel Tower?",
            "answer": "330 meters"
        },
        # Add more QA pairs
    ],

    # Few-shot learning configuration
    few_shot_config={
        "num_examples": 5,
        "selection_strategy": "diverse",
        "include_reasoning": True
    }
)

result = await pipeline.optimize(request)

# Test the optimized system
test_question = {
    "context": "Paris is the capital of France...",
    "question": "What is the capital of France?"
}

answer = await pipeline.predict(result.model_id, test_question)
print(f"Answer: {answer}")
```

### Example 3: Custom Evaluation Metric

```python
from agentcore.dspy_optimization.models import CustomMetric

# Define your own evaluation metric
def custom_relevance_scorer(prediction: str, ground_truth: str) -> float:
    """Custom metric: relevance score based on keyword overlap"""
    pred_keywords = set(prediction.lower().split())
    truth_keywords = set(ground_truth.lower().split())

    if not truth_keywords:
        return 0.0

    overlap = pred_keywords & truth_keywords
    return len(overlap) / len(truth_keywords)

# Register custom metric
custom_metric = CustomMetric(
    name="relevance_score",
    evaluation_function=custom_relevance_scorer,
    higher_is_better=True,
    min_value=0.0,
    max_value=1.0
)

# Use in optimization
request = OptimizationRequest(
    target="content_generator",
    objective=OptimizationObjective.CUSTOM,
    evaluation_metric="relevance_score",
    custom_metrics=[custom_metric],
    algorithm="miprov2"
)

result = await pipeline.optimize(request)
```

---

## Optimization Algorithms

### MIPROv2 (Recommended)

**Best for**: Most tasks, especially when you need high-quality results

**How it works**:

1. Generates multiple prompt variations
2. Tests each variation on training data
3. Combines best performing elements
4. Iteratively refines until convergence

**Configuration**:

```python
request = OptimizationRequest(
    algorithm="miprov2",
    algorithm_config={
        "num_candidates": 20,        # Prompts to try per iteration
        "num_iterations": 10,         # Optimization rounds
        "temperature": 0.7,           # Creativity in generation
        "diversity_penalty": 0.1,     # Encourage diverse prompts
        "early_stopping": True,       # Stop if no improvement
        "patience": 3                 # Iterations without improvement
    }
)
```

**Pros**:

- Highest quality results
- Good for complex tasks
- Robust to noisy data

**Cons**:

- Slower than other methods
- Higher API costs
- Requires 20+ training examples

---

### BootstrapFewShot

**Best for**: Tasks with limited training data (5-20 examples)

**How it works**:

1. Uses your examples as few-shot demonstrations
2. Bootstraps additional synthetic examples
3. Selects most informative examples
4. Optimizes example ordering

**Configuration**:

```python
request = OptimizationRequest(
    algorithm="bootstrapfewshot",
    algorithm_config={
        "num_examples": 5,            # Examples to include in prompt
        "bootstrap_rounds": 3,        # Synthetic data generation rounds
        "selection_strategy": "diverse", # "diverse", "hard", "random"
        "max_bootstrapped": 50        # Max synthetic examples
    }
)
```

**Pros**:

- Works with few examples
- Fast optimization
- Lower API costs

**Cons**:

- May not reach peak performance
- Sensitive to example quality
- Limited to few-shot tasks

---

### GEPA (Genetic Evolutionary)

**Best for**: Exploring novel prompt strategies

**How it works**:

1. Creates population of prompt variants
2. Evaluates fitness (performance)
3. Breeds best prompts (crossover)
4. Introduces mutations (variations)
5. Repeats for N generations

**Configuration**:

```python
request = OptimizationRequest(
    algorithm="gepa",
    algorithm_config={
        "population_size": 30,        # Prompts per generation
        "num_generations": 15,        # Evolution rounds
        "mutation_rate": 0.2,         # Probability of mutation
        "crossover_rate": 0.7,        # Probability of breeding
        "elite_size": 5,              # Best prompts to keep
        "tournament_size": 3          # Selection tournament size
    }
)
```

**Pros**:

- Discovers creative solutions
- Good exploration of prompt space
- Can escape local optima

**Cons**:

- Unpredictable convergence
- Requires tuning parameters
- Highest computational cost

---

## Advanced Features

### Continuous Learning Pipeline

Automatically improve prompts based on production data:

```python
from agentcore.dspy_optimization.learning.pipeline import ContinuousLearningPipeline

# Set up continuous learning
learning_pipeline = ContinuousLearningPipeline(
    model_id="sentiment_classifier",
    retraining_schedule="daily",
    drift_threshold=0.05,           # Retrain if accuracy drops 5%
    min_training_samples=100        # Need 100 new examples to retrain
)

# Start monitoring production data
await learning_pipeline.start()

# In your application, log predictions
await learning_pipeline.log_prediction(
    input_data={"text": "Customer review..."},
    prediction="positive",
    ground_truth="positive",        # If available
    latency_ms=250
)

# Pipeline automatically:
# 1. Detects performance drift
# 2. Triggers retraining
# 3. Validates new model
# 4. Deploys if better
# 5. Rolls back if worse
```

### A/B Testing

Compare different prompt strategies in production:

```python
from agentcore.dspy_optimization.testing.experiment import ABTestExperiment

# Create A/B test
experiment = ABTestExperiment(
    name="prompt_comparison_v1_v2",
    variants={
        "control": "model_v1_abc123",   # Current production model
        "treatment": "model_v2_def456"  # New optimized model
    },
    traffic_split={"control": 0.8, "treatment": 0.2},  # 80/20 split
    success_metric="accuracy",
    minimum_sample_size=1000,
    significance_level=0.05
)

# Start experiment
await experiment.start()

# In your application, assign users to variants
variant = experiment.get_variant(user_id="user_123")
prediction = await pipeline.predict(variant.model_id, input_data)

# Log results
await experiment.log_result(
    user_id="user_123",
    variant=variant.name,
    outcome={"correct": True, "latency_ms": 200}
)

# Check if we have a winner
if experiment.has_significant_result():
    winner = experiment.get_winner()
    print(f"Winner: {winner.name} with {winner.improvement}% improvement")

    # Promote winner to production
    await experiment.promote_winner()
```

### GPU Acceleration

Speed up optimization with GPU:

```python
from agentcore.dspy_optimization.scalability.gpu_device import GPUConfig

# Configure GPU usage
gpu_config = GPUConfig(
    enabled=True,
    device_ids=[0, 1],              # Use GPUs 0 and 1
    memory_fraction=0.8,            # Use 80% of GPU memory
    batch_size=32                   # Process 32 samples at once
)

# Pipeline automatically uses GPU if available
pipeline = DSPyOptimizationPipeline(gpu_config=gpu_config)

# Expect 3-5x speedup on large datasets
result = await pipeline.optimize(request)  # Faster with GPU!
```

### Distributed Optimization

Scale across multiple workers:

```python
from agentcore.dspy_optimization.scalability.job_queue import JobQueue, JobConfig

# Set up job queue (uses Redis)
job_queue = JobQueue(redis_url="redis://localhost:6379")

# Submit optimization jobs
job_id = await job_queue.submit_job(
    job_type="optimization",
    config=JobConfig(
        request=request,
        priority="high",
        timeout_seconds=3600,
        retry_policy={"max_retries": 3}
    )
)

# Monitor progress
status = await job_queue.get_job_status(job_id)
print(f"Status: {status.state}, Progress: {status.progress}%")

# Get results when complete
result = await job_queue.get_job_result(job_id)
```

---

## MLflow Integration

### Experiment Tracking

All optimizations are automatically tracked in MLflow:

```python
from agentcore.dspy_optimization.tracking.mlflow_tracker import MLflowTracker, MLflowConfig

# Configure MLflow
mlflow_config = MLflowConfig(
    tracking_uri="http://mlflow:5000",
    experiment_name="sentiment_analysis_optimization",
    artifact_location="s3://bucket/mlflow-artifacts"  # Optional
)

tracker = MLflowTracker(config=mlflow_config)

# Tracking happens automatically during optimization
# But you can also track manually:
async with tracker.start_run(run_name="experiment_1"):
    # Log baseline metrics
    await tracker.log_baseline_metrics(baseline_performance)

    # Log optimization progress
    for iteration in range(num_iterations):
        metrics = await run_optimization_iteration()
        await tracker.log_optimized_metrics(metrics, step=iteration)

    # Log final model
    await tracker.log_model_artifact(optimized_model, "model")
```

### Viewing Results

Access MLflow UI to view results:

```bash
# Open MLflow UI
open http://localhost:5000

# Or use CLI
mlflow experiments list
mlflow runs list --experiment-id 1
```

### Comparing Runs

```python
# Search for best runs
best_runs = await tracker.search_runs(
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

for run in best_runs:
    print(f"Run: {run.run_id}")
    print(f"  Accuracy: {run.metrics['accuracy']}")
    print(f"  Cost: ${run.metrics['cost_usd']}")
    print(f"  Duration: {run.duration_seconds}s")

# Get best run
best_run = await tracker.get_best_run(
    experiment_name="sentiment_analysis_optimization",
    metric="accuracy"
)

# Load model from best run
model = await tracker.load_model_artifact(best_run.run_id, "model")
```

---

## Best Practices

### 1. Start with Good Training Data

**Do's**:

- ✅ Use 50-100 diverse examples
- ✅ Include edge cases and hard examples
- ✅ Balance classes in classification tasks
- ✅ Validate data quality before optimization

**Don'ts**:

- ❌ Don't use <10 examples (use BootstrapFewShot instead)
- ❌ Don't include duplicate or near-duplicate examples
- ❌ Don't use low-quality or inconsistent labels
- ❌ Don't forget to split train/validation/test sets

### 2. Choose the Right Metric

**Task → Metric Mapping**:

- Classification → `f1_score` (balanced), `accuracy` (if balanced classes)
- Summarization → `rouge_score` or `bleu_score`
- QA → `exact_match` or `f1_score`
- Ranking → `ndcg` or `map`
- Generation → custom metric (fluency + relevance)

### 3. Set Realistic Constraints

```python
# Good constraints
constraints = OptimizationConstraints(
    max_tokens=500,             # Leave room for response
    max_cost_usd=10.0,          # Set budget
    max_latency_ms=2000,        # P95 latency target
    min_accuracy=0.70,          # Don't sacrifice quality
    target_model="gpt-5-mini"   # Use cost-effective model
)
```

### 4. Monitor and Iterate

```python
# Enable performance analytics
request = OptimizationRequest(
    ...,
    enable_analytics=True,
    track_costs=True,
    log_predictions=True
)

# After optimization, review analytics
analytics = await pipeline.get_analytics(result.optimization_id)

print(f"Total cost: ${analytics.total_cost_usd}")
print(f"Cost per sample: ${analytics.cost_per_sample}")
print(f"Latency P50: {analytics.latency_p50_ms}ms")
print(f"Latency P95: {analytics.latency_p95_ms}ms")

# Iterate based on insights
if analytics.cost_per_sample > 0.01:
    # Try with smaller model
    request.constraints.target_model = "gpt-5-mini"
```

### 5. Use Versioning

```python
# Always version your models
await pipeline.save_optimized_model(
    model_id=result.model_id,
    version="1.0.0",
    metadata={
        "algorithm": "miprov2",
        "training_date": "2025-11-01",
        "accuracy": 0.85,
        "use_case": "production_sentiment_analysis"
    }
)

# Load specific version in production
model = await pipeline.load_model(
    name="sentiment_classifier",
    version="1.0.0"  # Pin to stable version
)
```

---

## Troubleshooting

### Issue: Optimization Not Improving

**Symptoms**: Metrics stuck or decreasing after several iterations

**Causes & Solutions**:

1. **Insufficient Training Data**

   ```python
   # Solution: Add more diverse examples
   if len(training_examples) < 50:
       print("Warning: Need 50+ examples for MIPROv2")
       # Use BootstrapFewShot instead
   ```

2. **Wrong Algorithm**

   ```python
   # Solution: Try different algorithm
   # Complex task → MIPROv2
   # Few examples → BootstrapFewShot
   # Need exploration → GEPA
   ```

3. **Poor Metric Choice**

   ```python
   # Solution: Verify metric aligns with goal
   # Classification with imbalanced classes? Use f1_score, not accuracy
   ```

### Issue: High API Costs

**Symptoms**: Optimization costing more than expected

**Solutions**:

```python
# 1. Use smaller model for optimization
request.constraints.target_model = "gpt-5-mini"

# 2. Reduce candidates
request.algorithm_config["num_candidates"] = 10  # Down from 20

# 3. Enable early stopping
request.algorithm_config["early_stopping"] = True
request.algorithm_config["patience"] = 2

# 4. Set cost limit
request.constraints.max_cost_usd = 5.0
```

### Issue: Optimization Too Slow

**Symptoms**: Taking hours to complete

**Solutions**:

```python
# 1. Enable GPU acceleration
gpu_config = GPUConfig(enabled=True)
pipeline = DSPyOptimizationPipeline(gpu_config=gpu_config)

# 2. Use distributed processing
job_queue = JobQueue(redis_url="redis://localhost:6379")
await job_queue.submit_job(job_type="optimization", config=request)

# 3. Reduce iterations
request.max_iterations = 25  # Down from 50

# 4. Use faster algorithm
request.algorithm = "bootstrapfewshot"  # Faster than MIPROv2
```

### Issue: MLflow Connection Error

**Symptoms**: `ConnectionError: Failed to connect to MLflow`

**Solutions**:

```bash
# 1. Check MLflow is running
curl http://localhost:5000/health

# 2. Start MLflow if needed
mlflow server --host 0.0.0.0 --port 5000 &

# 3. Verify environment variable
echo $MLFLOW_TRACKING_URI

# 4. Or disable MLflow temporarily
export DISABLE_MLFLOW=true
```

---

## API Reference

### DSPyOptimizationPipeline

Main entry point for optimization.

```python
class DSPyOptimizationPipeline:
    def __init__(
        self,
        gpu_config: GPUConfig | None = None,
        mlflow_config: MLflowConfig | None = None
    ):
        """Initialize optimization pipeline."""
        pass

    async def optimize(
        self,
        request: OptimizationRequest
    ) -> OptimizationResult:
        """Run optimization."""
        pass

    async def predict(
        self,
        model_id: str,
        input_data: dict[str, Any]
    ) -> Any:
        """Make prediction with optimized model."""
        pass

    async def save_optimized_model(
        self,
        model_id: str,
        output_path: str,
        version: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Save optimized model to disk."""
        pass
```

### OptimizationRequest

Configuration for optimization run.

```python
class OptimizationRequest(BaseModel):
    # Required fields
    target: str                              # What to optimize
    objective: OptimizationObjective         # What to achieve
    evaluation_metric: MetricType | str      # How to measure
    algorithm: str                           # Which algorithm

    # Training data
    training_examples: list[dict[str, Any]]
    validation_examples: list[dict[str, Any]] | None = None

    # Stop conditions
    max_iterations: int = 50
    timeout_seconds: int = 3600

    # Constraints
    constraints: OptimizationConstraints | None = None

    # Algorithm-specific config
    algorithm_config: dict[str, Any] = {}

    # Feature flags
    enable_analytics: bool = True
    track_costs: bool = True
    log_predictions: bool = False
```

### OptimizationResult

Results from optimization run.

```python
class OptimizationResult(BaseModel):
    # Identifiers
    optimization_id: str
    model_id: str

    # Performance
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_percentage: float

    # Artifacts
    optimized_prompt: str | None = None
    optimized_config: dict[str, Any]

    # Metadata
    algorithm_used: str
    iterations_completed: int
    duration_seconds: float
    total_cost_usd: float

    # Tracking
    mlflow_run_id: str | None = None
```

---

## Examples Repository

Find more examples in the repository:

- **[examples/dspy/text_classification.py](../../examples/dspy/text_classification.py)** - Complete classification example
- **[examples/dspy/question_answering.py](../../examples/dspy/question_answering.py)** - QA system optimization
- **[examples/dspy/summarization.py](../../examples/dspy/summarization.py)** - Text summarization
- **[examples/dspy/continuous_learning.py](../../examples/dspy/continuous_learning.py)** - Production learning
- **[examples/dspy/ab_testing.py](../../examples/dspy/ab_testing.py)** - A/B test setup

---

## Additional Resources

- **[DSPy Specification](../specs/dspy-optimization/spec.md)** - Detailed requirements
- **[DSPy Implementation Plan](../specs/dspy-optimization/plan.md)** - Architecture details
- **[Performance Benchmarks](../benchmarks/llm-performance.md)** - Performance data
- **[DSPy Framework Docs](https://github.com/stanfordnlp/dspy)** - Official DSPy docs

---

## Getting Help

**Issues or Questions?**

- GitHub Issues: <https://github.com/Mathews-Tom/AgentCore/issues>
- Discussions: <https://github.com/Mathews-Tom/AgentCore/discussions>
- Documentation: <https://github.com/Mathews-Tom/AgentCore/tree/main/docs>

---

**Happy Optimizing! 🚀**
