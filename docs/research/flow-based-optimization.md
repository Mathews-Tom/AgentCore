# Flow-Based Agent Optimization

## Overview

Flow-based agent optimization is a training methodology that improves agent performance by optimizing decision-making within the context of multi-step workflows (flows). Unlike traditional approaches that optimize agents in isolation, flow-based optimization considers the entire execution trajectory—including intermediate steps, tool interactions, and state transitions—when computing learning signals.

This approach is particularly valuable for agentic systems where the quality of individual decisions compounds across multiple steps, and where sparse rewards at the end of a workflow must be attributed back to specific agent actions.

## Technical Description

### Core Concepts

**1. Flow Definition**
A flow represents a complete execution path through an agentic system:

```plaintext
Flow = [State₀, Action₁, State₁, Action₂, State₂, ..., Actionₙ, Stateₙ]
```

Where:

- States: System state at each step (context, memory, intermediate results)
- Actions: Agent decisions (planning, tool selection, parameter choices)
- Final state includes outcome and reward

**2. Credit Assignment Problem**
In multi-step flows, determining which actions contributed to success/failure is non-trivial:

```plaintext
Initial Query → [Plan Step 1] → [Execute Tool A] → [Verify Result] → [Plan Step 2] → [Execute Tool B] → Final Answer

If final answer is wrong, which step(s) caused the error?
- Was the initial plan flawed?
- Did Tool A return bad data?
- Was Tool B used incorrectly?
- Did verification miss an error?
```

Flow-based optimization solves this through trajectory-level rewards and gradient attribution.

**3. Group Refined Policy Optimization (GRPO)**
A reinforcement learning algorithm optimized for agentic flows:

**Key Features:**

- **Group-based sampling**: Generate multiple trajectories per query
- **Refined rewards**: Use trajectory comparisons for better signal
- **Policy gradient**: Update agent policy to favor high-reward trajectories
- **In-the-flow training**: Optimize while agent executes real workflows

### Algorithm Overview

**Step 1: Trajectory Rollout**

```python
async def generate_trajectories(
    query: str,
    agent: Agent,
    n_trajectories: int = 8
) -> list[Trajectory]:
    """Generate multiple solution trajectories."""
    trajectories = []

    for _ in range(n_trajectories):
        trajectory = Trajectory(query=query)

        state = initial_state(query)
        done = False

        while not done:
            # Agent makes decision
            action = await agent.act(state)

            # Execute action
            next_state = await execute_action(action, state)

            # Record step
            trajectory.add_step(state, action, next_state)

            # Check if done
            done = is_terminal(next_state)
            state = next_state

        # Evaluate final outcome
        reward = evaluate_outcome(trajectory)
        trajectory.reward = reward

        trajectories.append(trajectory)

    return trajectories
```

**Step 2: Reward Refinement**

```python
def refine_rewards(trajectories: list[Trajectory]) -> list[Trajectory]:
    """
    Refine rewards using group statistics.

    Instead of absolute rewards, use relative performance
    within the group to reduce variance.
    """
    # Calculate group statistics
    rewards = [t.reward for t in trajectories]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Normalize rewards
    for trajectory in trajectories:
        if std_reward > 0:
            trajectory.normalized_reward = (
                (trajectory.reward - mean_reward) / std_reward
            )
        else:
            trajectory.normalized_reward = 0

    # Compute advantages (how much better than average)
    for trajectory in trajectories:
        trajectory.advantage = trajectory.normalized_reward

    return trajectories
```

**Step 3: Policy Update**

```python
async def update_policy(
    agent: Agent,
    trajectories: list[Trajectory],
    optimizer: Optimizer
) -> dict[str, float]:
    """
    Update agent policy using policy gradient.

    Intuition: Increase probability of actions that led to
    high-reward trajectories, decrease for low-reward ones.
    """
    total_loss = 0.0

    for trajectory in trajectories:
        if trajectory.advantage <= 0:
            # Skip low-performing trajectories
            continue

        # Compute policy gradient
        loss = 0.0
        for step in trajectory.steps:
            # Log probability of action under current policy
            log_prob = await agent.compute_log_prob(
                state=step.state,
                action=step.action
            )

            # Policy gradient: log_prob * advantage
            loss -= log_prob * trajectory.advantage

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {
        "loss": total_loss / len(trajectories),
        "avg_reward": np.mean([t.reward for t in trajectories]),
        "std_reward": np.std([t.reward for t in trajectories])
    }
```

**Step 4: Iterative Training**

```python
async def train_agent_with_grpo(
    agent: Agent,
    training_queries: list[str],
    n_iterations: int = 1000,
    n_trajectories_per_query: int = 8,
    batch_size: int = 16
) -> Agent:
    """
    Train agent using Group Refined Policy Optimization.
    """
    optimizer = create_optimizer(agent.parameters())

    for iteration in range(n_iterations):
        # Sample batch of queries
        batch_queries = random.sample(training_queries, batch_size)

        # Collect trajectories
        all_trajectories = []
        for query in batch_queries:
            trajectories = await generate_trajectories(
                query=query,
                agent=agent,
                n_trajectories=n_trajectories_per_query
            )
            all_trajectories.extend(trajectories)

        # Refine rewards
        all_trajectories = refine_rewards(all_trajectories)

        # Update policy
        metrics = await update_policy(agent, all_trajectories, optimizer)

        # Log progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: {metrics}")

    return agent
```

### Advanced Techniques

**1. Value Function Baseline**
Reduce variance by subtracting a learned baseline:

```python
class ValueFunction(nn.Module):
    """Estimates expected reward for a given state."""

    def forward(self, state: State) -> float:
        """Predict expected reward from this state."""
        pass

# Use in advantage calculation
advantage = trajectory.reward - value_function(initial_state)
```

**2. Multi-step Credit Assignment**
Attribute rewards to specific steps in trajectory:

```python
def compute_step_rewards(trajectory: Trajectory) -> list[float]:
    """
    Assign credit to each step using temporal difference.

    Steps closer to success receive more credit.
    """
    n_steps = len(trajectory.steps)
    step_rewards = []

    for i, step in enumerate(trajectory.steps):
        # Discount factor: steps closer to end get more credit
        gamma = 0.99
        discount = gamma ** (n_steps - i - 1)

        step_reward = trajectory.reward * discount
        step_rewards.append(step_reward)

    return step_rewards
```

**3. Exploration vs Exploitation**
Balance trying new strategies vs using known good ones:

```python
async def act_with_exploration(
    agent: Agent,
    state: State,
    temperature: float = 0.8
) -> Action:
    """
    Sample action with temperature-based exploration.

    Higher temperature = more exploration
    Lower temperature = more exploitation
    """
    action_logits = await agent.compute_action_logits(state)

    # Apply temperature
    action_probs = softmax(action_logits / temperature)

    # Sample action
    action = sample_from_distribution(action_probs)

    return action
```

**4. Reward Shaping**
Provide intermediate rewards to accelerate learning:

```python
def compute_shaped_reward(trajectory: Trajectory) -> float:
    """
    Add intermediate rewards for desired behaviors.

    Final reward = outcome_reward + shaped_rewards
    """
    outcome_reward = evaluate_final_outcome(trajectory)

    # Reward correct tool usage
    tool_reward = sum(
        0.1 for step in trajectory.steps
        if step.action.tool_used and step.action.tool_success
    )

    # Reward verification
    verification_reward = sum(
        0.05 for step in trajectory.steps
        if step.action.type == "verify"
    )

    # Penalize excessive steps
    length_penalty = -0.01 * len(trajectory.steps)

    total_reward = (
        outcome_reward +
        tool_reward +
        verification_reward +
        length_penalty
    )

    return total_reward
```

## Value Analysis

### Performance Benefits

**1. Improved Agent Quality**

- Learns from real execution feedback
- Adapts to specific workflow patterns
- Expected improvement: +15-25% task success rate after training

**2. Better Tool Usage**

- Learns which tools are effective for which queries
- Reduces tool call errors
- Expected improvement: +20% tool usage accuracy

**3. Enhanced Planning**

- Learns effective decomposition strategies
- Adapts plans based on intermediate results
- Expected improvement: +10-15% on complex multi-step tasks

**4. Faster Convergence**

- Group-based reward refinement reduces variance
- Requires fewer samples to learn
- Expected improvement: 2-3x faster than standard RL

### Cost-Benefit Analysis

**Training Costs:**

```plaintext
Assumptions:
- 1000 training queries
- 8 trajectories per query
- Average 10 steps per trajectory
- $0.50 per 1M tokens
- 500 tokens per step

Total tokens: 1000 * 8 * 10 * 500 = 40M tokens
Training cost: 40M * $0.50/M = $20

Add infrastructure: $5
Total: $25 for full training run
```

**Inference Benefits:**

```plaintext
Trained agent vs baseline:
- 20% higher success rate
- 10% fewer steps per task
- 15% better tool usage

If handling 10K queries/month:
- Higher success saves support costs
- Fewer steps reduces compute costs
- Better tool usage reduces API costs

Monthly savings: ~$500-1000
ROI: 20-40x
```

## Implementation Considerations

### Technical Challenges

**1. Reward Design**

- Challenge: Defining good reward functions
- Solution: Use outcome-based rewards + shaping
- Considerations: Balance intermediate and final rewards

**2. Sample Efficiency**

- Challenge: RL requires many samples
- Solution: Use offline data, transfer learning
- Considerations: Cold start problem

**3. Distribution Shift**

- Challenge: Agent performance changes during training
- Solution: Regular policy evaluation, adaptive sampling
- Considerations: Monitor validation metrics

**4. Infrastructure**

- Challenge: Parallel trajectory generation at scale
- Solution: Distributed execution, async processing
- Considerations: Cost vs speed trade-offs

### Resource Requirements

**1. Compute**

- Multiple trajectory generations per query
- Parallel execution recommended
- GPU for policy updates
- Estimated: 4-8 GPUs for efficient training

**2. Storage**

- Trajectory data for analysis
- Model checkpoints
- Training logs
- Estimated: 100GB-1TB per training run

**3. Time**

- Training duration: 1-3 days for 1000 iterations
- Depends on: trajectory length, batch size, hardware

## Integration Strategy

### Phase 1: Training Infrastructure (Weeks 1-2)

**Core Training Components**

```python
# agentcore/training/
├── __init__.py
├── grpo.py              # GRPO algorithm implementation
├── trajectory.py        # Trajectory collection
├── rewards.py           # Reward computation
├── policy.py            # Policy update logic
└── trainer.py           # Training orchestration
```

**Trajectory Collection**

```python
# agentcore/training/trajectory.py

class TrajectoryCollector:
    """Collects execution trajectories from agents."""

    async def collect(
        self,
        agent: Agent,
        query: str,
        n_trajectories: int
    ) -> list[Trajectory]:
        """Generate n trajectories for given query."""
        tasks = [
            self._generate_single_trajectory(agent, query)
            for _ in range(n_trajectories)
        ]
        return await asyncio.gather(*tasks)

    async def _generate_single_trajectory(
        self,
        agent: Agent,
        query: str
    ) -> Trajectory:
        """Generate single trajectory with full state tracking."""
        trajectory = Trajectory(query=query)
        state = self._initialize_state(query)

        max_steps = 20
        for step_num in range(max_steps):
            # Agent acts
            action = await agent.act(state)

            # Record state and action
            trajectory.add_step(
                step_num=step_num,
                state=state.to_dict(),
                action=action.to_dict()
            )

            # Execute
            next_state = await self._execute_action(action, state)

            # Check termination
            if next_state.is_terminal:
                break

            state = next_state

        # Evaluate
        trajectory.reward = self._evaluate_trajectory(trajectory)

        return trajectory
```

### Phase 2: GRPO Implementation (Week 3)

**GRPO Trainer**

```python
# agentcore/training/grpo.py

class GRPOTrainer:
    """Group Refined Policy Optimization trainer."""

    def __init__(
        self,
        agent: Agent,
        reward_function: Callable,
        optimizer: Optimizer,
        config: GRPOConfig
    ):
        self.agent = agent
        self.reward_function = reward_function
        self.optimizer = optimizer
        self.config = config

    async def train_iteration(
        self,
        training_queries: list[str]
    ) -> dict[str, float]:
        """Execute one training iteration."""
        # Sample queries
        batch_queries = random.sample(
            training_queries,
            self.config.batch_size
        )

        # Collect trajectories
        trajectories = []
        for query in batch_queries:
            query_trajectories = await self.trajectory_collector.collect(
                agent=self.agent,
                query=query,
                n_trajectories=self.config.n_trajectories_per_query
            )
            trajectories.extend(query_trajectories)

        # Refine rewards
        trajectories = self._refine_rewards(trajectories)

        # Update policy
        metrics = await self._update_policy(trajectories)

        return metrics

    def _refine_rewards(
        self,
        trajectories: list[Trajectory]
    ) -> list[Trajectory]:
        """Normalize rewards using group statistics."""
        rewards = np.array([t.reward for t in trajectories])

        mean = rewards.mean()
        std = rewards.std()

        for trajectory in trajectories:
            if std > 0:
                trajectory.advantage = (trajectory.reward - mean) / std
            else:
                trajectory.advantage = 0.0

        return trajectories

    async def _update_policy(
        self,
        trajectories: list[Trajectory]
    ) -> dict[str, float]:
        """Update agent policy using policy gradient."""
        total_loss = 0.0
        n_updates = 0

        for trajectory in trajectories:
            if trajectory.advantage <= 0:
                continue

            # Compute loss for this trajectory
            loss = await self._compute_trajectory_loss(trajectory)

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)

        return {
            "loss": avg_loss,
            "n_updates": n_updates,
            "avg_reward": np.mean([t.reward for t in trajectories]),
            "max_reward": np.max([t.reward for t in trajectories])
        }
```

### Phase 3: Agent Training API (Week 4)

**Training Endpoints**

```python
@register_jsonrpc_method("training.start_grpo")
async def start_grpo_training(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Start GRPO training for an agent.

    Params:
        agent_id: str - Agent to train
        training_data: list[dict] - Training queries and expected outcomes
        config: dict - Training configuration

    Returns:
        training_job_id: str
    """
    agent_id = request.params["agent_id"]
    training_data = request.params["training_data"]
    config = GRPOConfig(**request.params.get("config", {}))

    # Create training job
    job = TrainingJob(
        job_id=str(uuid.uuid4()),
        agent_id=agent_id,
        training_data=training_data,
        config=config,
        status="running"
    )

    # Start training in background
    asyncio.create_task(run_training_job(job))

    return {"training_job_id": job.job_id}

@register_jsonrpc_method("training.get_status")
async def get_training_status(request: JsonRpcRequest) -> dict[str, Any]:
    """Get status of training job."""
    job_id = request.params["job_id"]
    job = await get_training_job(job_id)

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "metrics": job.current_metrics,
        "best_checkpoint": job.best_checkpoint
    }
```

### Phase 4: Evaluation & Monitoring (Week 5)

**Continuous Evaluation**

```python
class AgentEvaluator:
    """Evaluates agent performance during training."""

    async def evaluate(
        self,
        agent: Agent,
        eval_queries: list[str]
    ) -> dict[str, float]:
        """Run evaluation on held-out queries."""
        results = []

        for query in eval_queries:
            trajectory = await self.trajectory_collector.collect(
                agent=agent,
                query=query,
                n_trajectories=1
            )
            results.append(trajectory[0])

        metrics = {
            "success_rate": sum(t.reward > 0 for t in results) / len(results),
            "avg_reward": np.mean([t.reward for t in results]),
            "avg_steps": np.mean([len(t.steps) for t in results]),
            "tool_usage_accuracy": self._compute_tool_accuracy(results)
        }

        return metrics
```

## Success Metrics

1. **Task Success Rate**
   - Target: +20% improvement over baseline
   - Measure: success_rate_after / success_rate_before

2. **Sample Efficiency**
   - Target: Learn from <10K trajectories
   - Measure: trajectories_to_convergence

3. **Training Stability**
   - Target: Monotonic improvement in validation metrics
   - Measure: validation_performance over iterations

4. **Inference Quality**
   - Target: Trained agent matches/exceeds baseline on all metrics
   - Measure: comprehensive benchmark evaluation

5. **Cost Efficiency**
   - Target: ROI > 10x within 3 months
   - Measure: training_cost vs saved_inference_cost

## Conclusion

Flow-based agent optimization through GRPO provides a principled approach to improving agent performance by learning from real execution trajectories. By optimizing agents in the context of complete workflows rather than isolated actions, this method captures the complex dependencies and compounding effects that characterize agentic systems. The result is agents that make better decisions, use tools more effectively, and achieve higher success rates on complex multi-step tasks. While training requires upfront investment in infrastructure and compute, the long-term benefits of improved agent quality and reduced operational costs provide strong ROI.
