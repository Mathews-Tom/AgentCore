"""
Intervention Decision Making (COMPASS ACE-2 - ACE-017)

Strategic decision-making component that determines which intervention to execute
based on trigger signals and strategic context. Uses gpt-4.1 for accuracy-critical
strategic decisions with <200ms latency target.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from agentcore.ace.models.ace_models import (
    InterventionDecision,
    InterventionType,
    StrategicContext,
    TriggerSignal,
    TriggerType,
)
from agentcore.llm_gateway.client import LLMGatewayClient
from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.exceptions import LLMGatewayError
from agentcore.llm_gateway.models import LLMRequest

logger = structlog.get_logger()

# LLM configuration for decision making
DECISION_MODEL = "gpt-4.1"
DECISION_TEMPERATURE = 0.3  # Balance creativity with consistency
DECISION_MAX_TOKENS = 500  # Concise decisions

# System prompt for Meta-Thinker role
SYSTEM_PROMPT = """You are the Meta-Thinker component of the COMPASS intervention system.

Your role is to make strategic decisions about which intervention to execute when
performance issues are detected in agent task execution.

Available intervention types:
1. CONTEXT_REFRESH: Refresh agent's working memory and context playbook
   - Use when: Context is stale, low retrieval relevance, or low-confidence sections dominate
   - Impact: Improves decision quality by updating contextual information

2. REPLAN: Trigger replanning of task execution strategy
   - Use when: Performance degradation, high error rates, or velocity drops
   - Impact: Helps agent reassess approach and break down tasks differently

3. REFLECT: Trigger reflection on recent execution patterns
   - Use when: Error accumulation, repeated failures, or learning opportunity
   - Impact: Builds meta-cognitive understanding of what's working/not working

4. CAPABILITY_SWITCH: Switch to alternative agent capabilities
   - Use when: Capability mismatch, high action failure rate, or missing capabilities
   - Impact: Aligns agent capabilities with task requirements

Make your decision based on:
- Trigger type and signals (what problem was detected)
- Strategic context (what's happening in the task)
- Metric values (quantitative evidence)
- Expected impact (predicted outcome)

Respond with a JSON object containing:
{
  "intervention_type": "<CONTEXT_REFRESH|REPLAN|REFLECT|CAPABILITY_SWITCH>",
  "rationale": "<clear reasoning for this decision>",
  "confidence": <0.0-1.0 confidence score>,
  "expected_impact": "<predicted outcome description>",
  "alternative_interventions": ["<alternative 1>", "<alternative 2>"]
}

Be decisive but explain your reasoning. Consider alternatives but recommend the best option.
"""


class DecisionMaker:
    """
    Strategic decision maker for interventions (COMPASS ACE-2 - ACE-017).

    Features:
    - LLM-powered decision making using gpt-4.1 (accuracy-critical)
    - <200ms decision latency target (p95)
    - Structured prompts for each trigger type
    - Decision validation and error handling
    - Comprehensive logging with structlog

    Decision accuracy target: 85%+ (validated in ACE-020)
    """

    def __init__(
        self,
        llm_client: LLMGatewayClient | None = None,
        model: str = DECISION_MODEL,
        temperature: float = DECISION_TEMPERATURE,
        max_tokens: int = DECISION_MAX_TOKENS,
    ) -> None:
        """
        Initialize DecisionMaker.

        Args:
            llm_client: LLM Gateway client (if None, creates from env)
            model: Model to use for decisions (default: gpt-4.1)
            temperature: Sampling temperature (default: 0.3)
            max_tokens: Max tokens for response (default: 500)

        Raises:
            ValueError: If parameters are invalid
        """
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature}")
        if max_tokens < 100:
            raise ValueError(f"max_tokens must be >= 100, got {max_tokens}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM client
        if llm_client is None:
            config = LLMGatewayConfig.from_env()
            self.llm_client = LLMGatewayClient(config=config)
        else:
            self.llm_client = llm_client

        logger.info(
            "DecisionMaker initialized",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def decide_intervention(
        self,
        trigger: TriggerSignal,
        strategic_context: StrategicContext,
    ) -> InterventionDecision:
        """
        Decide which intervention to execute based on trigger and context.

        This is the main entry point for decision making. Uses LLM (gpt-4.1)
        to analyze trigger signals and strategic context, then selects the
        most appropriate intervention type.

        Args:
            trigger: Trigger signal with detected issues
            strategic_context: Strategic context from COMPASS stages

        Returns:
            InterventionDecision with selected intervention and rationale

        Raises:
            ValueError: If trigger or context is invalid
            LLMGatewayError: If LLM request fails after retries
        """
        start_time = time.perf_counter()

        logger.info(
            "Making intervention decision",
            trigger_type=trigger.trigger_type.value,
            trigger_confidence=trigger.confidence,
            context_health=strategic_context.context_health_score,
        )

        # Build user prompt
        user_prompt = self._build_user_prompt(trigger, strategic_context)

        # Call LLM
        try:
            llm_request = LLMRequest(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={
                    "component": "decision_maker",
                    "trigger_type": trigger.trigger_type.value,
                },
            )

            llm_response = await self.llm_client.complete(llm_request)

            # Extract content from response
            content = self._extract_content_from_response(llm_response)

            # Parse response
            decision = self._parse_llm_response(content)

            # Calculate latency
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Add metadata
            decision.metadata.update({
                "trigger_confidence": trigger.confidence,
                "context_health": strategic_context.context_health_score,
                "decision_latency_ms": latency_ms,
                "model": self.model,
                "llm_usage": llm_response.usage,
            })

            logger.info(
                "Intervention decision made",
                intervention_type=decision.intervention_type.value,
                confidence=decision.confidence,
                latency_ms=latency_ms,
            )

            return decision

        except LLMGatewayError as e:
            logger.error(
                "LLM decision request failed",
                error=str(e),
                trigger_type=trigger.trigger_type.value,
            )
            raise

    def _build_user_prompt(
        self,
        trigger: TriggerSignal,
        strategic_context: StrategicContext,
    ) -> str:
        """
        Build user prompt for LLM decision request.

        Creates a structured prompt with trigger details, strategic context,
        and metric values to guide the LLM decision.

        Args:
            trigger: Trigger signal
            strategic_context: Strategic context

        Returns:
            Formatted user prompt string
        """
        # Format trigger information
        trigger_info = f"""TRIGGER DETECTED:
Type: {trigger.trigger_type.value}
Confidence: {trigger.confidence:.2f}
Signals: {', '.join(trigger.signals)}
Rationale: {trigger.rationale}

Metric Values:
{self._format_metrics(trigger.metric_values)}"""

        # Format strategic context
        context_info = f"""STRATEGIC CONTEXT:
Context Health Score: {strategic_context.context_health_score:.2f}

Stage Summaries:
{self._format_list(strategic_context.relevant_stage_summaries)}

Critical Facts:
{self._format_list(strategic_context.critical_facts)}

Error Patterns:
{self._format_list(strategic_context.error_patterns) if strategic_context.error_patterns else "None detected"}

Successful Patterns:
{self._format_list(strategic_context.successful_patterns) if strategic_context.successful_patterns else "None identified"}"""

        # Combine into full prompt
        return f"""{trigger_info}

{context_info}

Based on this trigger and context, which intervention should be executed?
Provide your decision in JSON format as specified in the system prompt."""

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        """Format metric values for prompt."""
        if not metrics:
            return "No metric values available"
        return "\n".join(f"  - {key}: {value:.3f}" for key, value in metrics.items())

    def _format_list(self, items: list[str]) -> str:
        """Format list of items for prompt."""
        if not items:
            return "  None"
        return "\n".join(f"  - {item}" for item in items)

    def _extract_content_from_response(self, response: Any) -> str:
        """
        Extract content from LLM response.

        Handles both direct response objects and our LLMResponse wrapper.

        Args:
            response: LLM response object

        Returns:
            Extracted text content

        Raises:
            ValueError: If content cannot be extracted
        """
        try:
            # LLMResponse has choices list
            if hasattr(response, 'choices') and response.choices:
                first_choice = response.choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get('message', {})
                    content = message.get('content', '')
                    if content:
                        return content
                elif hasattr(first_choice, 'message'):
                    if hasattr(first_choice.message, 'content'):
                        return first_choice.message.content
                    elif isinstance(first_choice.message, dict):
                        return first_choice.message.get('content', '')

            raise ValueError("Could not extract content from response")
        except (AttributeError, KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract content from response: {e}")

    def _parse_llm_response(self, response_content: str) -> InterventionDecision:
        """
        Parse LLM response into InterventionDecision.

        Extracts JSON from response, validates structure, and creates
        InterventionDecision instance.

        Args:
            response_content: Raw LLM response content

        Returns:
            Validated InterventionDecision instance

        Raises:
            ValueError: If response is invalid or missing required fields
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()

            # Parse JSON
            data = json.loads(content)

            # Validate required fields
            required_fields = [
                "intervention_type",
                "rationale",
                "confidence",
                "expected_impact",
            ]
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Parse intervention type
            intervention_type_str = data["intervention_type"].upper()
            try:
                intervention_type = InterventionType(intervention_type_str.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid intervention_type: {intervention_type_str}. "
                    f"Must be one of: {', '.join(t.value for t in InterventionType)}"
                )

            # Create decision
            return InterventionDecision(
                intervention_type=intervention_type,
                rationale=data["rationale"],
                confidence=float(data["confidence"]),
                expected_impact=data["expected_impact"],
                alternative_interventions=data.get("alternative_interventions", []),
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON", error=str(e))
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to validate LLM response", error=str(e))
            raise ValueError(f"Invalid response structure: {e}")
