# Bounded Context Reasoning Engine

## Overview

Bounded context reasoning is a paradigm for managing long-form reasoning and inference in language models by maintaining a fixed-size context window throughout the reasoning process. Unlike traditional approaches where context grows linearly with reasoning depth (leading to quadratic computational costs), bounded context reasoning achieves linear scaling by periodically resetting the context while preserving essential information through compact carryover mechanisms.

**Important:** Bounded context reasoning is **one of multiple reasoning strategies** available in AgentCore. It is not mandatory and should be used when its specific characteristics (cost efficiency for long-form reasoning, constant memory usage) align with the task requirements. This document describes the bounded context approach in detail, while other strategies (Chain of Thought, ReAct, Tree of Thought) may be more appropriate for different use cases.

## Technical Description

### The Quadratic Cost Problem

Traditional reasoning approaches suffer from quadratic computational complexity:

**Problem:** Standard transformer-based reasoning

```plaintext
Step 1: [Prompt] + [Thought 1]                    → Context length: C + T₁
Step 2: [Prompt] + [Thought 1] + [Thought 2]      → Context length: C + T₁ + T₂
Step 3: [Prompt] + [Thought 1] + [Thought 2] + [Thought 3] → Context length: C + T₁ + T₂ + T₃
...
Step N: [Prompt] + [All previous thoughts]        → Context length: C + ΣTᵢ

Total compute: O(N² × C) where N is reasoning depth
```

This makes long reasoning prohibitively expensive. For example, 100 reasoning steps with 8K context results in ~800K token processing.

### Bounded Context Solution

**Chunked Generation with Context Reset:**

```plaintext
Iteration 1: [Prompt] + [Thought Chunk 1] → [Carryover 1]
Iteration 2: [Prompt] + [Carryover 1] + [Thought Chunk 2] → [Carryover 2]
Iteration 3: [Prompt] + [Carryover 2] + [Thought Chunk 3] → [Carryover 3]
...

Context length stays constant: C
Total compute: O(N × C) where N is number of chunks
```

### Key Components

**1. Chunk Size (C)**

- Fixed context window for each reasoning iteration
- Typical values: 4K-8K tokens
- Balance between reasoning capacity and compute efficiency

**2. Carryover Size (m)**

- Compressed summary of progress from previous chunk
- Typical values: C/2 to C/4 tokens
- Contains essential information to maintain reasoning continuity

**3. Iteration Budget (I)**

- Maximum number of reasoning chunks allowed
- Total reasoning capacity: C + (I-1) × (C-m)
- Example: C=8K, m=4K, I=5 → Total budget = 8K + 4×4K = 24K tokens

**4. Carryover Generation**

- Model learns to compress progress into compact form
- Maintains: key insights, current strategy, unresolved questions
- Discards: redundant information, completed steps

### Algorithm

```python
async def bounded_context_reasoning(
    query: str,
    chunk_size: int = 8192,
    carryover_size: int = 4096,
    max_iterations: int = 5
) -> str:
    """
    Perform bounded context reasoning with fixed memory.

    Args:
        query: The problem to solve
        chunk_size: Maximum tokens per iteration (C)
        carryover_size: Tokens to carry forward (m)
        max_iterations: Maximum reasoning iterations (I)

    Returns:
        Final answer after reasoning
    """
    prompt = format_prompt(query)
    carryover = ""

    for iteration in range(max_iterations):
        # Build current context
        if iteration == 0:
            context = prompt
            max_new_tokens = chunk_size
        else:
            context = prompt + "\n\nPrevious progress:\n" + carryover
            max_new_tokens = chunk_size - len(tokenize(prompt + carryover))

        # Generate reasoning chunk
        chunk = await generate(
            context=context,
            max_tokens=max_new_tokens,
            stop_on=["<answer>", "<continue>"]
        )

        # Check if answer found
        if "<answer>" in chunk:
            return extract_answer(chunk)

        # Generate carryover for next iteration
        if iteration < max_iterations - 1:
            carryover = await generate_carryover(
                context=context + chunk,
                max_tokens=carryover_size,
                instruction="Summarize the key progress and insights"
            )

    # Fallback if no answer after all iterations
    return generate_final_answer(prompt, carryover)
```

### Training Considerations

For optimal performance, models should be trained to:

1. **Generate Informative Carryovers**
   - Learn to identify key information to preserve
   - Compress reasoning state effectively
   - Maintain continuity across iterations

2. **Utilize Carryover Context**
   - Build upon previous progress
   - Avoid redundant recomputation
   - Progress toward solution incrementally

3. **Boundary Awareness**
   - Recognize when approaching context limit
   - Generate appropriate stopping points
   - Signal continuation vs completion

## Value Analysis

### Performance Benefits

**1. Linear Computational Scaling**

- Traditional: O(N²) compute complexity
- Bounded: O(N) compute complexity
- Savings: 10-100x reduction for long reasoning tasks
- Example: 50-step reasoning
  - Traditional: ~2,500 chunk-equivalents of compute
  - Bounded: ~50 chunk-equivalents of compute
  - **98% compute reduction**

**2. Constant Memory Footprint**

- Memory usage remains fixed regardless of reasoning depth
- Enables deployment on resource-constrained devices
- Prevents OOM errors during long reasoning
- Predictable resource requirements

**3. Extended Reasoning Capability**

- Can perform reasoning up to 128K+ tokens with 8K context
- Enables solving complex problems requiring extensive thought
- Maintains quality throughout long reasoning chains

**4. Cost Efficiency**

```plaintext
Traditional Approach (24K tokens):
- Context grows from 0 to 24K
- Average context: 12K tokens
- Steps: 30
- Total tokens processed: 30 × 12K = 360K tokens
- Cost at $0.50/M tokens: $0.18 per query

Bounded Context (8K fixed):
- Context stays at 8K
- Steps: 30
- Total tokens processed: 30 × 8K = 240K tokens
- Cost at $0.50/M tokens: $0.12 per query
- **33% cost reduction**
```

### Quality Metrics

Based on evaluation across mathematical reasoning tasks:

**Accuracy:**

- 8K bounded context matches 24K traditional accuracy
- 24K bounded context improves over 24K traditional by 5-10%
- Quality maintained while reducing compute

**Scalability:**

- Beyond trained budget, bounded context continues improving
- Traditional approaches plateau
- Demonstrates better generalization

## Implementation Considerations

### Technical Challenges

**1. Carryover Quality**

- Challenge: Generating effective compressed summaries
- Solution: Train models specifically for carryover generation
- Metrics: Measure information retention, reasoning continuity

**2. Boundary Management**

- Challenge: Determining optimal chunk boundaries
- Solution: Use heuristics or train model to self-manage boundaries
- Considerations: Natural stopping points, thought completeness

**3. Context Coherence**

- Challenge: Maintaining coherent reasoning across resets
- Solution: Design carryover format with explicit structure
- Example: "Current strategy: ...", "Key findings: ...", "Next steps: ..."

**4. Training Complexity**

- Challenge: Models need special training for bounded reasoning
- Solution: Use reinforcement learning with trajectory rewards
- Alternative: Supervised fine-tuning on bounded reasoning traces

### Resource Requirements

**1. Inference Infrastructure**

- Standard LLM inference servers (vLLM, SGLang)
- Context management layer for chunking
- Carryover generation pipeline

**2. Training Infrastructure (if training custom models)**

- RL training framework (e.g., PPO)
- Distributed training setup for policy optimization
- Reward model for trajectory evaluation

## Integration Strategy

### Phase 1: Basic Implementation (Weeks 1-2)

**Core Bounded Context Engine**

```python
# agentcore/reasoning/bounded_context.py

class BoundedContextEngine:
    """Engine for bounded context reasoning."""

    def __init__(
        self,
        llm_client: LLMClient,
        chunk_size: int = 8192,
        carryover_size: int = 4096,
        max_iterations: int = 5
    ):
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.carryover_size = carryover_size
        self.max_iterations = max_iterations

    async def reason(self, query: str) -> ReasoningResult:
        """Execute bounded context reasoning."""
        iterations = []
        carryover = ""

        for i in range(self.max_iterations):
            iteration = await self._execute_iteration(
                query=query,
                carryover=carryover,
                iteration_num=i
            )
            iterations.append(iteration)

            if iteration.has_answer:
                return ReasoningResult(
                    answer=iteration.answer,
                    iterations=iterations,
                    total_tokens=sum(it.tokens for it in iterations)
                )

            carryover = iteration.carryover

        # Generate final answer if no answer found
        final_answer = await self._generate_final_answer(query, carryover)
        return ReasoningResult(
            answer=final_answer,
            iterations=iterations,
            total_tokens=sum(it.tokens for it in iterations)
        )

    async def _execute_iteration(
        self,
        query: str,
        carryover: str,
        iteration_num: int
    ) -> ReasoningIteration:
        """Execute a single reasoning iteration."""
        # Build context
        if iteration_num == 0:
            context = self._format_initial_prompt(query)
        else:
            context = self._format_continuation_prompt(query, carryover)

        # Generate reasoning chunk
        response = await self.llm_client.generate(
            prompt=context,
            max_tokens=self._calculate_max_tokens(context),
            stop_sequences=["<answer>", "<continue>"]
        )

        # Check for answer
        has_answer = "<answer>" in response.text
        answer = self._extract_answer(response.text) if has_answer else None

        # Generate carryover if continuing
        new_carryover = ""
        if not has_answer and iteration_num < self.max_iterations - 1:
            new_carryover = await self._generate_carryover(
                context + response.text
            )

        return ReasoningIteration(
            iteration_num=iteration_num,
            context=context,
            response=response.text,
            has_answer=has_answer,
            answer=answer,
            carryover=new_carryover,
            tokens=response.token_count
        )

    async def _generate_carryover(self, full_context: str) -> str:
        """Generate compressed carryover for next iteration."""
        carryover_prompt = (
            f"{full_context}\n\n"
            "Summarize the key progress, insights, and next steps in a compact form.\n"
            "Keep only essential information needed to continue reasoning.\n\n"
            "Summary:"
        )

        response = await self.llm_client.generate(
            prompt=carryover_prompt,
            max_tokens=self.carryover_size
        )

        return response.text.strip()

    def _calculate_max_tokens(self, context: str) -> int:
        """Calculate max new tokens for current iteration."""
        context_tokens = len(self.llm_client.tokenize(context))
        return self.chunk_size - context_tokens
```

### Phase 2: JSON-RPC Integration (Week 3)

**Register Bounded Reasoning Method**

```python
# agentcore/a2a_protocol/services/reasoning_jsonrpc.py

from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.reasoning.bounded_context import BoundedContextEngine

@register_jsonrpc_method("reasoning.bounded_context")
async def handle_bounded_reasoning(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute bounded context reasoning.

    Params:
        query: str - The problem to solve
        chunk_size: int - Max tokens per iteration (default: 8192)
        carryover_size: int - Tokens to carry forward (default: 4096)
        max_iterations: int - Max reasoning iterations (default: 5)

    Returns:
        answer: str - Final answer
        iterations: list - Details of each iteration
        total_tokens: int - Total tokens processed
        compute_savings: float - Percentage saved vs traditional
    """
    params = request.params
    query = params["query"]

    engine = BoundedContextEngine(
        llm_client=get_llm_client(),
        chunk_size=params.get("chunk_size", 8192),
        carryover_size=params.get("carryover_size", 4096),
        max_iterations=params.get("max_iterations", 5)
    )

    result = await engine.reason(query)

    # Calculate compute savings
    traditional_cost = calculate_traditional_cost(result)
    compute_savings = 1 - (result.total_tokens / traditional_cost)

    return {
        "answer": result.answer,
        "iterations": [
            {
                "iteration": it.iteration_num,
                "tokens": it.tokens,
                "has_answer": it.has_answer
            }
            for it in result.iterations
        ],
        "total_tokens": result.total_tokens,
        "compute_savings_pct": compute_savings * 100
    }
```

### Phase 3: Agent Integration (Week 4)

**Add Bounded Reasoning to Agent Capabilities**

```python
# Update AgentCard to advertise bounded reasoning capability
agent_card = AgentCard(
    id="reasoning-agent",
    name="Bounded Reasoning Agent",
    capabilities=[
        "bounded_context_reasoning",
        "long_form_reasoning",
        "mathematical_reasoning"
    ],
    supported_methods=[
        "reasoning.bounded_context"
    ],
    ...
)
```

### Phase 4: Monitoring & Optimization (Week 5)

**Add Metrics**

- Track compute savings per query
- Monitor carryover quality (information retention)
- Measure reasoning depth vs accuracy
- Compare against traditional baselines

**Optimization**

- Tune chunk and carryover sizes per use case
- Implement adaptive iteration budgets
- Add caching for repeated carryover patterns

## Success Metrics

1. **Compute Efficiency**
   - Target: 50-90% reduction in tokens processed for >10K token reasoning
   - Measure: tokens_processed_bounded / tokens_processed_traditional

2. **Memory Efficiency**
   - Target: Constant memory usage regardless of reasoning depth
   - Measure: peak_memory_usage vs reasoning_depth (should be flat)

3. **Quality Preservation**
   - Target: Match or exceed traditional reasoning accuracy
   - Measure: accuracy_bounded / accuracy_traditional >= 1.0

4. **Extended Reasoning**
   - Target: Support 50K+ token reasoning with 8K context
   - Measure: max_successful_reasoning_depth

5. **Latency**
   - Target: <20% latency increase vs traditional (acceptable given compute savings)
   - Measure: end_to_end_latency

## Strategy Comparison: When to Use Bounded Context

Bounded context reasoning is one strategy among several. Understanding when to use it versus alternatives is critical for optimal performance.

### Reasoning Strategy Comparison Matrix

| Strategy | Best For | Compute Cost | Memory | Latency | Quality Trade-offs |
|----------|----------|--------------|--------|---------|-------------------|
| **Direct** | Simple queries, single-step answers | Very Low | O(1) | Very Low | No reasoning chain |
| **Chain of Thought (CoT)** | Step-by-step problems, math, logical reasoning | Medium | O(N) growing | Medium | Full context preserved |
| **Tree of Thought (ToT)** | Exploring alternatives, planning, search problems | High | O(N×B) branching | High | Explores multiple paths |
| **ReAct** | Tool use, interactive tasks, grounded reasoning | Medium | O(N) growing | Medium | Requires tool integration |
| **Bounded Context** | Long-form reasoning (>10K tokens), cost-sensitive tasks | Low (50-98% savings) | O(1) constant | Medium (+20%) | Carryover compression loss |

### When to Choose Bounded Context

**✅ Use Bounded Context When:**

- Reasoning requires >10K tokens (benefits become significant)
- Cost is a primary concern (deploy at scale with budget constraints)
- Memory usage must be predictable and constant
- Extended reasoning tasks (research synthesis, long-form planning)
- Quality can tolerate minor carryover compression loss

**❌ Avoid Bounded Context When:**

- Reasoning is short (<8K tokens) - overhead not worth it
- Exact recall of all prior steps is critical (formal proofs, legal reasoning)
- Latency is more important than cost (real-time applications)
- Problem requires seeing "full picture" continuously (debugging, complex code analysis)
- Model doesn't support good carryover compression

### Specific Use Case Recommendations

| Use Case | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| Mathematical proof | Chain of Thought | Needs exact step preservation |
| Research paper summarization | Bounded Context | Long documents, cost-effective |
| Code debugging | Chain of Thought | Needs full context visibility |
| Multi-document Q&A | Bounded Context | Handles >50K tokens efficiently |
| Interactive tool use | ReAct | Tool integration primary concern |
| Planning with alternatives | Tree of Thought | Explores multiple options |
| Simple factual queries | Direct | No reasoning needed |
| Extended technical analysis | Bounded Context | Long-form, cost-sensitive |

### Hybrid Approaches

**Adaptive Strategy Selection:**

Systems can dynamically choose strategies based on:
- Query length estimation (short → CoT, long → Bounded Context)
- User budget constraints (budget-limited → Bounded Context)
- Task type detection (tool use → ReAct, exploration → ToT)
- Performance requirements (latency-critical → CoT, cost-critical → Bounded Context)

**Strategy Chaining:**

Combine strategies for complex workflows:
1. Use Tree of Thought for initial exploration
2. Switch to Bounded Context for deep analysis of chosen path
3. Use ReAct for final verification with external tools

## Limitations and Considerations

### Bounded Context Specific Limitations

**1. Carryover Information Loss**
- Compressed summaries may lose nuanced details
- Not suitable for tasks requiring exact historical recall
- Quality depends on model's summarization capability

**2. Answer Detection Complexity**
- Requires models to emit stop sequences (`<answer>`, `<continue>`)
- May need fine-tuning or careful prompt engineering
- Not all models support this pattern natively

**3. Latency Overhead**
- Each iteration includes carryover generation step
- +20% latency vs traditional approaches
- Acceptable trade-off given compute savings, but not for real-time use

**4. Training Considerations**
- Models benefit from training on bounded context patterns
- Off-the-shelf models may produce lower-quality carryovers
- Best performance requires model aware of bounded context paradigm

**5. Context Fragmentation**
- Reasoning across iteration boundaries may feel disjointed
- Some problems benefit from continuous full context
- Carryover quality critical for maintaining coherence

### When Bounded Context May Underperform

- **Formal Verification:** Losing intermediate steps breaks proof validity
- **Code Generation:** Full codebase context often needed for consistency
- **Short Reasoning:** Overhead exceeds benefits for <8K token tasks
- **Real-Time Applications:** Latency overhead unacceptable
- **Exact Calculation Tasks:** Lossy compression problematic

## Conclusion

Bounded context reasoning transforms the economics of long-form agent reasoning by achieving linear computational scaling while maintaining quality. However, it is **one tool among many** in the reasoning toolkit.

**Key Takeaways:**

1. **Not a Silver Bullet:** Bounded context excels at long-form, cost-sensitive reasoning but has trade-offs (carryover loss, latency overhead)
2. **Strategy Selection Matters:** Choose the right strategy for the task - no single approach is universally optimal
3. **Optional by Design:** In AgentCore, bounded context is an optional strategy users can enable when beneficial
4. **Cost-Quality Trade-off:** Offers 50-98% compute reduction with acceptable quality loss for many (but not all) tasks
5. **Best for Scale:** Most valuable when deploying reasoning agents at scale with budget constraints

For AgentCore as a generic orchestration framework, providing multiple reasoning strategies with clear guidance on when to use each approach maximizes flexibility and ensures users can optimize for their specific requirements.
