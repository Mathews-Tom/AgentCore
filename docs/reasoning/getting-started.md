# Getting Started with Reasoning Strategies

This guide will help you get started with the Reasoning Framework in AgentCore. You'll learn how to execute reasoning tasks using different strategies and choose the right approach for your use case.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Basic Concepts](#basic-concepts)
- [Your First Reasoning Request](#your-first-reasoning-request)
- [Choosing the Right Strategy](#choosing-the-right-strategy)
- [Common Use Cases](#common-use-cases)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Quick Start

Get up and running in 5 minutes:

```python
import httpx
import asyncio

async def main():
    # Execute reasoning with automatic strategy selection
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.execute",
                "params": {
                    "auth_token": "your-jwt-token-here",
                    "query": "What is the capital of France?"
                },
                "id": 1
            }
        )

        result = response.json()["result"]
        print(f"Answer: {result['answer']}")
        print(f"Strategy used: {result['strategy_used']}")
        print(f"Tokens used: {result['metrics']['total_tokens']}")

asyncio.run(main())
```

## Prerequisites

### 1. AgentCore Installation

Ensure AgentCore is running:

```bash
# Using Docker Compose
docker compose -f docker-compose.dev.yml up

# Or directly
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Authentication Token

Obtain a JWT token with `reasoning:execute` permission:

```python
import httpx

# Example token request (actual implementation may vary)
response = httpx.post(
    "http://localhost:8001/api/v1/auth/token",
    json={
        "agent_id": "my-agent",
        "secret": "my-secret"
    }
)

token = response.json()["access_token"]
print(f"Token: {token}")
```

### 3. LLM Provider Setup

Configure your LLM provider API key:

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Basic Concepts

### What is a Reasoning Strategy?

A reasoning strategy is an approach for solving complex problems using large language models (LLMs). Different strategies have different characteristics:

| Strategy | Best For | Speed | Token Efficiency |
|----------|----------|-------|------------------|
| **Chain of Thought** | Simple problems | ⚡ Fast | 🟢 Good |
| **Bounded Context** | Large/complex problems | ⚠️ Medium | 🟢 Excellent |
| **ReAct** | Problems needing tools/data | 🐢 Slow | 🟡 Medium |

### How Strategy Selection Works

The system automatically selects a strategy based on:

1. **Your request**: Explicitly specify a strategy
2. **Agent preferences**: Agent advertises supported strategies
3. **System default**: Fallback to configured default

You can override at any level.

## Your First Reasoning Request

### Example 1: Simple Math Problem

**Use Case**: Quick calculation

**Best Strategy**: Chain of Thought

```python
import httpx
import asyncio

async def simple_calculation():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.execute",
                "params": {
                    "auth_token": "your-token",
                    "query": "Calculate 15% of $350",
                    "strategy": "chain_of_thought"
                },
                "id": 1
            }
        )

        result = response.json()["result"]
        print(f"Answer: {result['answer']}")
        print(f"Execution time: {result['metrics']['execution_time_ms']}ms")

        # Show reasoning trace
        if result.get('trace'):
            print("\nReasoning steps:")
            for step in result['trace']:
                if step['type'] == 'reasoning':
                    print(f"  {step['content'][:100]}...")

asyncio.run(simple_calculation())
```

**Expected Output:**

```
Answer: $52.50
Execution time: 1200ms

Reasoning steps:
  Step 1: Convert 15% to decimal: 0.15
  Step 2: Multiply 350 by 0.15: 350 × 0.15 = 52.50
  Final Answer: $52.50
```

### Example 2: Long Document Analysis

**Use Case**: Analyze a large document

**Best Strategy**: Bounded Context

```python
async def analyze_document():
    # Read a large document
    with open("research_paper.txt", "r") as f:
        document = f.read()

    query = f"""Analyze this research paper and provide:
1. Key findings (3-5 bullet points)
2. Methodology used
3. Limitations of the study
4. Implications for future research

Document:
{document}
"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.execute",
                "params": {
                    "auth_token": "your-token",
                    "query": query,
                    "strategy": "bounded_context",
                    "strategy_config": {
                        "chunk_size": 8192,
                        "max_iterations": 10
                    }
                },
                "id": 1
            }
        )

        result = response.json()["result"]
        print(f"Answer:\n{result['answer']}\n")

        metrics = result['metrics']['strategy_specific']
        print(f"Iterations: {metrics['total_iterations']}")
        print(f"Compute savings: {metrics['compute_savings_pct']:.1f}%")
        print(f"Total tokens: {result['metrics']['total_tokens']}")

asyncio.run(analyze_document())
```

**Expected Output:**

```
Answer:
**Key Findings:**
- Finding 1: The study demonstrated...
- Finding 2: Results showed...
- Finding 3: Analysis revealed...

**Methodology:**
The researchers employed a mixed-methods approach...

**Limitations:**
- Sample size was limited to...
- Data collection period was short...

**Implications:**
Future research should focus on...

Iterations: 4
Compute savings: 48.3%
Total tokens: 24500
```

### Example 3: Research with External Data

**Use Case**: Problem requiring external information

**Best Strategy**: ReAct

```python
async def research_query():
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.execute",
                "params": {
                    "auth_token": "your-token",
                    "query": "What is the current population of Tokyo and how does it compare to New York City?",
                    "strategy": "react",
                    "strategy_config": {
                        "max_iterations": 5
                    }
                },
                "id": 1
            }
        )

        result = response.json()["result"]
        print(f"Answer: {result['answer']}\n")

        # Show reasoning trace
        print("Reasoning process:")
        for iteration in result['trace']:
            print(f"\nIteration {iteration['iteration']}:")
            print(f"  Thought: {iteration['thought']}")
            print(f"  Action: {iteration['action']}")
            print(f"  Observation: {iteration['observation'][:100]}...")

asyncio.run(research_query())
```

**Expected Output:**

```
Answer: Tokyo has approximately 37.4 million people in its metropolitan area, while New York City has about 20.1 million. Tokyo is roughly 1.86 times larger than NYC.

Reasoning process:

Iteration 0:
  Thought: I need to search for the current population of Tokyo
  Action: Search for Tokyo population
  Observation: Simulated search result: Tokyo metropolitan area population...

Iteration 1:
  Thought: Now I need to search for NYC population
  Action: Search for New York City population
  Observation: Simulated search result: NYC metropolitan area population...

Iteration 2:
  Thought: I have both populations, now I can compare them
  Action: Answer
  Observation: Tokyo has approximately 37.4 million people...
```

## Choosing the Right Strategy

### Decision Guide

Ask yourself these questions:

```
1. Does my problem fit in one context window (< 4K tokens)?
   YES → Use Chain of Thought
   NO  → Continue to question 2

2. Do I need external data or tools?
   YES → Use ReAct
   NO  → Continue to question 3

3. Is token efficiency important? (large document, limited budget)
   YES → Use Bounded Context
   NO  → Use Chain of Thought (faster)
```

### Strategy Comparison

#### Chain of Thought ⚡

**When to use:**
- Simple math or logic problems
- Code explanation or generation
- Quick question answering
- Text analysis (small documents)

**Example queries:**
```
✅ "What is the square root of 144?"
✅ "Explain this Python function: [code]"
✅ "Summarize this paragraph: [text]"
✅ "What are the pros and cons of renewable energy?"
```

**Configuration tips:**
```python
{
    "strategy": "chain_of_thought",
    "strategy_config": {
        "temperature": 0.5,  # Lower for factual answers
        "max_tokens": 2048,  # Adjust based on expected answer length
        "show_reasoning": True  # See the thinking process
    }
}
```

#### Bounded Context 📊

**When to use:**
- Large document analysis (> 10K tokens)
- Multi-step research tasks
- Complex code refactoring
- Long-form content generation
- Budget-conscious applications

**Example queries:**
```
✅ "Analyze this 50-page research paper and summarize findings"
✅ "Review this codebase and identify security vulnerabilities"
✅ "Generate a comprehensive business plan with market analysis"
✅ "Compare and contrast these 5 technical specifications"
```

**Configuration tips:**
```python
{
    "strategy": "bounded_context",
    "strategy_config": {
        "chunk_size": 8192,  # 8K for complex reasoning
        "carryover_size": 4096,  # 50% of chunk_size is good
        "max_iterations": 10  # Increase for very large tasks
    }
}
```

#### ReAct 🔍

**When to use:**
- Problems requiring real-time data
- Tasks needing verification
- Interactive exploration
- Tool/API integration needed

**Example queries:**
```
✅ "What's the weather in Paris and recommend what to wear?"
✅ "Find the latest stock price of AAPL and analyze the trend"
✅ "Search for Python documentation on asyncio and explain"
✅ "Look up the definition of quantum entanglement and provide examples"
```

**Configuration tips:**
```python
{
    "strategy": "react",
    "strategy_config": {
        "max_iterations": 5,  # Depends on expected action count
        "allow_tool_use": True,  # Enable if you have tools configured
        "temperature": 0.7  # Standard for balanced reasoning
    }
}
```

## Common Use Cases

### Use Case 1: Customer Support Bot

**Scenario**: Answer customer questions about products

**Strategy**: Chain of Thought (fast responses)

```python
async def customer_support(question: str, product_docs: str):
    """Answer customer question using product documentation."""
    query = f"""Using the following product documentation, answer the customer's question.

Product Documentation:
{product_docs}

Customer Question: {question}

Provide a clear, helpful answer."""

    response = await reasoning_client.execute(
        query=query,
        strategy="chain_of_thought",
        config={"temperature": 0.5}  # Factual answers
    )

    return response["answer"]

# Example
answer = await customer_support(
    question="How do I reset my password?",
    product_docs="[Product documentation here]"
)
print(answer)
```

### Use Case 2: Legal Document Review

**Scenario**: Review contracts for compliance

**Strategy**: Bounded Context (handles large documents efficiently)

```python
async def review_contract(contract_text: str, requirements: list[str]):
    """Review contract for compliance with requirements."""
    query = f"""Review this contract and check compliance with these requirements:

Requirements:
{chr(10).join(f"- {req}" for req in requirements)}

Contract:
{contract_text}

For each requirement, indicate:
1. Is it met? (Yes/No/Partial)
2. Evidence from contract
3. Recommendations if not fully met"""

    response = await reasoning_client.execute(
        query=query,
        strategy="bounded_context",
        config={
            "chunk_size": 8192,
            "max_iterations": 15  # Large contracts
        }
    )

    return response["answer"]

# Example
review = await review_contract(
    contract_text="[50-page contract]",
    requirements=[
        "Termination clause with 30-day notice",
        "Data protection compliance (GDPR)",
        "Liability cap at $1M"
    ]
)
print(review)
```

### Use Case 3: Research Assistant

**Scenario**: Gather and synthesize information

**Strategy**: ReAct (needs to search and verify)

```python
async def research_topic(topic: str):
    """Research a topic and provide comprehensive summary."""
    query = f"""Research the topic: {topic}

Provide:
1. Overview and key concepts
2. Current state of the field
3. Recent developments (last 2 years)
4. Key challenges and opportunities
5. Sources and references"""

    response = await reasoning_client.execute(
        query=query,
        strategy="react",
        config={
            "max_iterations": 10,
            "allow_tool_use": True
        }
    )

    return response["answer"]

# Example
research = await research_topic("Quantum computing applications in cryptography")
print(research)
```

### Use Case 4: Code Review

**Scenario**: Review code for bugs and improvements

**Strategy**: Chain of Thought (code fits in context)

```python
async def code_review(code: str, language: str):
    """Review code for issues and suggest improvements."""
    query = f"""Review this {language} code and provide:
1. Potential bugs or errors
2. Security vulnerabilities
3. Performance issues
4. Code quality suggestions
5. Best practices violations

Code:
```{language}
{code}
```
"""

    response = await reasoning_client.execute(
        query=query,
        strategy="chain_of_thought",
        config={"temperature": 0.3}  # Precise analysis
    )

    return response["answer"]

# Example
review = await code_review(
    code="def process_data(data):\n    return [x*2 for x in data]",
    language="python"
)
print(review)
```

## Best Practices

### 1. Write Clear Queries

**Bad:**
```python
query = "analyze this"  # Too vague
```

**Good:**
```python
query = """Analyze this sales report and identify:
1. Top 3 performing products
2. Revenue trends by quarter
3. Regional performance differences
4. Recommendations for Q4

Sales Report:
[data here]
"""
```

### 2. Choose Appropriate Parameters

**For factual answers:**
```python
config = {
    "temperature": 0.3,  # More deterministic
    "max_tokens": 1024   # Concise answers
}
```

**For creative tasks:**
```python
config = {
    "temperature": 0.8,  # More creative
    "max_tokens": 4096   # Longer outputs
}
```

### 3. Handle Errors Gracefully

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10)
)
async def safe_reasoning(query: str):
    try:
        response = await client.post(url, json=request_body)
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            error = result["error"]
            if error["code"] == -32001:
                # Strategy not found - try default
                return await reasoning_with_default(query)
            else:
                raise ValueError(f"Reasoning error: {error['message']}")

        return result["result"]

    except httpx.TimeoutError:
        print("Request timed out. Try reducing query size.")
        raise
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        raise
```

### 4. Monitor Token Usage

```python
async def reasoning_with_budget(query: str, max_tokens: int = 10000):
    """Execute reasoning with token budget."""
    response = await reasoning_client.execute(query=query)

    tokens_used = response["metrics"]["total_tokens"]

    if tokens_used > max_tokens:
        print(f"Warning: Used {tokens_used} tokens (budget: {max_tokens})")

    cost = estimate_cost(tokens_used)
    print(f"Estimated cost: ${cost:.4f}")

    return response

def estimate_cost(tokens: int, model: str = "gpt-4") -> float:
    """Estimate cost based on token usage."""
    # GPT-4 pricing (example)
    cost_per_1k = 0.01  # prompt + completion average
    return (tokens / 1000) * cost_per_1k
```

### 5. Use Appropriate Timeouts

```python
# Short timeout for simple queries
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.post(url, json=cot_request)

# Long timeout for complex queries
async with httpx.AsyncClient(timeout=120.0) as client:
    response = await client.post(url, json=bounded_context_request)

# Very long timeout for ReAct with tools
async with httpx.AsyncClient(timeout=300.0) as client:
    response = await client.post(url, json=react_request)
```

## Troubleshooting

### Problem: "Authentication required: Missing JWT token"

**Solution:**
```python
# Ensure you're including the token in params
params = {
    "auth_token": "your-token-here",  # Must be present
    "query": "..."
}
```

### Problem: "Strategy not found: 'bounded_context'"

**Solution:**

Check enabled strategies in configuration:

```bash
# Check config.toml
cat config.toml | grep enabled_strategies

# Should include your desired strategy
enabled_strategies = ["bounded_context", "chain_of_thought", "react"]
```

### Problem: Request times out

**Solutions:**

1. **Increase timeout:**
```python
async with httpx.AsyncClient(timeout=120.0) as client:
    response = await client.post(...)
```

2. **Reduce query size:**
```python
# Split large queries into smaller chunks
chunks = split_text(large_document, chunk_size=5000)
results = [await reasoning_client.execute(chunk) for chunk in chunks]
combined = combine_results(results)
```

3. **Use bounded context for large queries:**
```python
# More efficient for large inputs
response = await reasoning_client.execute(
    query=large_query,
    strategy="bounded_context"
)
```

### Problem: High token usage / costs

**Solutions:**

1. **Use bounded context strategy:**
```python
# 30-50% token savings for large queries
strategy = "bounded_context"
```

2. **Reduce max_tokens:**
```python
config = {
    "max_tokens": 2048  # Lower limit
}
```

3. **Simplify prompts:**
```python
# Before: "Please analyze in great detail..."
# After: "Analyze..."
```

### Problem: Poor answer quality

**Solutions:**

1. **Adjust temperature:**
```python
config = {
    "temperature": 0.3  # More focused (factual)
    # or
    "temperature": 0.8  # More creative
}
```

2. **Provide more context:**
```python
query = f"""Context: {background_info}

Question: {question}

Requirements:
- Be specific
- Cite sources
- Explain reasoning
"""
```

3. **Try a different strategy:**
```python
# If Chain of Thought gives poor results, try Bounded Context
strategy = "bounded_context"
```

## Next Steps

### Learn More

- **[API Reference](../api/reasoning.md)**: Complete API documentation
- **[Architecture](./architecture.md)**: System design and components
- **[Strategy Comparison](./strategy-comparison.md)**: Detailed strategy comparison
- **[Monitoring](../../monitoring/README.md)**: Set up monitoring and alerts

### Advanced Topics

- **Extending the Framework**: Add custom reasoning strategies
- **Performance Tuning**: Optimize for your use case
- **Cost Optimization**: Reduce token usage and costs
- **Production Deployment**: Best practices for production

### Example Projects

Explore complete example projects:

```bash
# Clone examples repository
git clone https://github.com/AetherForge/agentcore-examples

# Customer support bot
cd examples/customer-support-bot

# Document analysis service
cd examples/document-analysis

# Research assistant
cd examples/research-assistant
```

### Join the Community

- **GitHub**: https://github.com/AetherForge/AgentCore
- **Discussions**: https://github.com/AetherForge/AgentCore/discussions
- **Documentation**: https://docs.agentcore.dev
- **Discord**: https://discord.gg/agentcore

## Summary

You've learned:

✅ How to execute reasoning requests with different strategies
✅ When to use each strategy (Chain of Thought, Bounded Context, ReAct)
✅ How to configure strategies for your use case
✅ Common patterns and best practices
✅ How to troubleshoot common issues

**Quick Reference:**

```python
# Simple problem - Chain of Thought
await reasoning.execute(query, strategy="chain_of_thought")

# Large document - Bounded Context
await reasoning.execute(query, strategy="bounded_context")

# Need tools/data - ReAct
await reasoning.execute(query, strategy="react")
```

Happy reasoning! 🚀
