# Reasoning Strategy Comparison Guide

This guide compares the three reasoning strategies available in the AgentCore Reasoning Framework, helping you choose the best strategy for your use case.

## Quick Reference

| Strategy | Best For | Token Efficiency | Latency | Complexity | Tool Support |
|----------|----------|------------------|---------|------------|--------------|
| **Chain of Thought** | Simple to medium problems, single-pass reasoning | Medium | Low | Low | No |
| **Bounded Context** | Large complex problems, multi-step reasoning | High | Medium | Medium | No |
| **ReAct** | Problems requiring actions/tool use, iterative exploration | Low | High | High | Yes |

## Strategy Details

### 1. Chain of Thought (CoT)

**Papers**: Wei et al., 2022 - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

**Description**: Single-pass reasoning where the LLM explicitly shows its step-by-step thinking before providing an answer.

**How it Works**:
1. Prompts LLM to "think step by step"
2. LLM generates reasoning trace in natural language
3. Final answer extracted from output

**Strengths**:
- ✅ Simple and fast for problems that fit in one context window
- ✅ Natural reasoning traces that are easy to understand
- ✅ Low latency (single LLM call)
- ✅ Good for math, logic, and analytical problems

**Limitations**:
- ❌ Limited by single context window (typically 4K-32K tokens)
- ❌ No iteration or multi-step exploration
- ❌ Cannot handle problems requiring external data/tools
- ❌ Less token-efficient than bounded context for large problems

**Configuration**:
```python
{
    "max_tokens": 4096,        # Max tokens for output
    "temperature": 0.7,        # Sampling temperature
    "show_reasoning": True     # Include reasoning trace
}
```

**Example Use Cases**:
- Mathematical word problems
- Logical puzzles
- Code explanation/generation
- Simple question answering
- Text analysis and summarization

**Performance**:
- Tokens: ~500-5000 per query
- Latency: ~1-5 seconds
- Cost: $0.01-$0.10 per query (GPT-4)

---

### 2. Bounded Context

**Papers**: Custom implementation based on iterative refinement principles

**Description**: Multi-iteration reasoning with fixed context windows and carryover compression. Maintains constant memory while processing arbitrarily large problems.

**How it Works**:
1. Splits reasoning into fixed-size chunks (default 8192 tokens)
2. Each iteration processes one chunk
3. Generates compressed carryover summary for next iteration
4. Continues until answer found or max iterations reached

**Strengths**:
- ✅ Handles arbitrarily large/complex problems
- ✅ Constant memory footprint (O(1) space complexity)
- ✅ High token efficiency through compression (30-50% savings)
- ✅ Linear time complexity (O(N) where N = chunks)
- ✅ Prevents context overflow errors

**Limitations**:
- ❌ Higher latency due to multiple LLM calls
- ❌ More complex implementation and monitoring
- ❌ Carryover compression may lose some detail
- ❌ Requires tuning (chunk size, carryover size)

**Configuration**:
```python
{
    "chunk_size": 8192,        # Tokens per iteration
    "carryover_size": 4096,    # Tokens to carry forward
    "max_iterations": 5        # Max reasoning iterations
}
```

**Example Use Cases**:
- Large document analysis
- Multi-step research tasks
- Complex code refactoring
- Long-form content generation
- Iterative problem solving

**Performance**:
- Tokens: ~10,000-50,000 per query
- Latency: ~5-30 seconds
- Cost: $0.20-$1.00 per query (GPT-4)
- Compute savings: 30-50% vs naive approach

---

### 3. ReAct (Reasoning + Acting)

**Papers**: Yao et al., 2022 - "ReAct: Synergizing Reasoning and Acting in Language Models"

**Description**: Iterative thought-action-observation loop that synergizes reasoning with the ability to take actions (e.g., search, API calls, calculations).

**How it Works**:
1. **Thought**: Reason about what to do next
2. **Action**: Execute an action (search, calculate, call API)
3. **Observation**: Receive result from action
4. Repeat until answer found

**Strengths**:
- ✅ Can interact with external tools/APIs
- ✅ Explicitly models reasoning and acting
- ✅ Good for tasks requiring up-to-date information
- ✅ Handles problems requiring verification
- ✅ Transparent reasoning trace

**Limitations**:
- ❌ High latency (multiple LLM calls + tool execution)
- ❌ Higher token usage (each iteration needs full context)
- ❌ Requires tool integration infrastructure
- ❌ Action execution can fail
- ❌ Most complex to implement and monitor

**Configuration**:
```python
{
    "max_iterations": 10,          # Max thought-action cycles
    "max_tokens_per_step": 2048,   # Tokens per step
    "temperature": 0.7,            # Sampling temperature
    "allow_tool_use": False        # Enable external tools
}
```

**Example Use Cases**:
- Research with web search
- Data analysis with calculations
- API-based information retrieval
- Multi-step verification tasks
- Interactive problem solving

**Performance**:
- Tokens: ~5,000-20,000 per query
- Latency: ~10-60 seconds (including tool calls)
- Cost: $0.10-$0.50 per query + tool costs

---

## Decision Tree

```
START: What type of problem are you solving?

├─ Simple problem, fits in one pass?
│  └─ Use: Chain of Thought
│     Example: "Calculate 25 * 37 + 19"
│
├─ Large/complex problem requiring multiple steps?
│  ├─ No external data needed?
│  │  └─ Use: Bounded Context
│  │     Example: "Analyze this 50-page document"
│  │
│  └─ Need external data (search, APIs, calculations)?
│     └─ Use: ReAct
│        Example: "Find and compare latest pricing for X"
│
└─ Interactive problem with verification needed?
   └─ Use: ReAct
      Example: "Debug this code by testing different scenarios"
```

## Performance Comparison

### Token Usage (Typical)

| Problem Size | Chain of Thought | Bounded Context | ReAct |
|--------------|------------------|-----------------|-------|
| Small (< 2K tokens) | 1,000 | 2,000 | 3,000 |
| Medium (2-10K tokens) | 5,000 | 8,000 | 10,000 |
| Large (10-50K tokens) | ❌ N/A | 25,000 | 30,000 |
| Very Large (> 50K tokens) | ❌ N/A | 50,000 | ❌ N/A |

### Latency (Typical)

| Problem Size | Chain of Thought | Bounded Context | ReAct |
|--------------|------------------|-----------------|-------|
| Small | 1-2s | 3-5s | 5-10s |
| Medium | 2-5s | 8-15s | 15-30s |
| Large | ❌ N/A | 15-30s | 30-60s |

### Cost (GPT-4, approximate)

| Problem Size | Chain of Thought | Bounded Context | ReAct |
|--------------|------------------|-----------------|-------|
| Small | $0.01 | $0.02 | $0.03 |
| Medium | $0.05 | $0.08 | $0.10 |
| Large | ❌ N/A | $0.25 | $0.30 |

*Note: Costs assume GPT-4 at $0.01/1K prompt tokens, $0.03/1K completion tokens*

## Strategy Selection Best Practices

### Use Chain of Thought when:
1. Problem fits in single context window
2. Low latency required
3. Simple to medium complexity
4. No external data needed
5. Clear answer expected

### Use Bounded Context when:
1. Large documents or datasets
2. Multi-step reasoning required
3. Token efficiency is priority
4. Consistent memory usage needed
5. No external tools required

### Use ReAct when:
1. External tools/APIs needed
2. Problem requires verification
3. Real-time data required
4. Interactive exploration helpful
5. Action execution is part of solution

## API Usage Examples

### Chain of Thought
```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "query": "What is 15% of 240?",
    "strategy": "chain_of_thought",
    "strategy_config": {
      "temperature": 0.7,
      "max_tokens": 2048
    }
  },
  "id": 1
}
```

### Bounded Context
```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "query": "Analyze this 50-page document and summarize key findings...",
    "strategy": "bounded_context",
    "strategy_config": {
      "chunk_size": 8192,
      "carryover_size": 4096,
      "max_iterations": 10
    }
  },
  "id": 1
}
```

### ReAct
```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "query": "Find the current weather in New York and recommend appropriate clothing",
    "strategy": "react",
    "strategy_config": {
      "max_iterations": 5,
      "allow_tool_use": true
    }
  },
  "id": 1
}
```

## Monitoring & Metrics

All strategies provide standardized metrics:

```python
{
  "total_tokens": 12500,
  "execution_time_ms": 15000,
  "strategy_specific": {
    # Chain of Thought:
    "temperature": 0.7,
    "finish_reason": "stop",

    # Bounded Context:
    "total_iterations": 3,
    "compute_savings_pct": 45.2,

    # ReAct:
    "total_iterations": 4,
    "answer_found_at_iteration": 3
  }
}
```

## Migration Guide

### From Chain of Thought to Bounded Context
Migrate when queries start hitting context limits or require multi-step reasoning.

```python
# Before (CoT)
{
  "strategy": "chain_of_thought",
  "strategy_config": {
    "max_tokens": 8192
  }
}

# After (Bounded Context)
{
  "strategy": "bounded_context",
  "strategy_config": {
    "chunk_size": 8192,
    "carryover_size": 4096,
    "max_iterations": 5
  }
}
```

### From Chain of Thought to ReAct
Migrate when you need external tool integration.

```python
# Before (CoT)
{
  "strategy": "chain_of_thought"
}

# After (ReAct)
{
  "strategy": "react",
  "strategy_config": {
    "allow_tool_use": true,
    "max_iterations": 10
  }
}
```

## References

1. **Chain of Thought**: Wei et al., 2022 - https://arxiv.org/abs/2201.11903
2. **ReAct**: Yao et al., 2022 - https://arxiv.org/abs/2210.03629
3. **Bounded Context**: Custom implementation - Internal documentation

## See Also

- [Reasoning Framework Architecture](./architecture.md)
- [API Reference](../api/reasoning.md)
- [Performance Tuning Guide](./performance-tuning.md)
- [Example Applications](./examples.md)
