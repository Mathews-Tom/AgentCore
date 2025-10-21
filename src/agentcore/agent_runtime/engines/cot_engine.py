"""Chain-of-Thought (CoT) philosophy engine."""

import re
from typing import Any

import structlog

from ..config.settings import get_settings
from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState
from ..services.llm_service import (
    LLMConfig,
    LLMResponse,
    get_llm_service,
    initialize_llm_service,
)
from .base import PhilosophyEngine
from .cot_models import CoTExecutionContext, CoTPromptTemplate, CoTStep, CoTStepType

logger = structlog.get_logger()


class CoTEngine(PhilosophyEngine):
    """Chain-of-Thought philosophy execution engine."""

    def __init__(
        self,
        config: AgentConfig,
        use_real_llm: bool = True,
    ) -> None:
        """
        Initialize CoT engine.

        Args:
            config: Agent configuration
            use_real_llm: Whether to use real LLM (True) or simulated (False for testing)
        """
        super().__init__(config)
        self.prompt_template = CoTPromptTemplate()
        self.context: CoTExecutionContext | None = None
        self.use_real_llm = use_real_llm

        # Initialize LLM service if using real LLM
        if self.use_real_llm:
            try:
                runtime_config = get_settings()
                llm_config = LLMConfig(
                    portkey_api_key=runtime_config.portkey_api_key,
                    portkey_base_url=runtime_config.portkey_base_url,
                    default_model=runtime_config.default_llm_model,
                    fallback_models=runtime_config.llm_fallback_models,
                    default_temperature=runtime_config.llm_temperature,
                    default_max_tokens=runtime_config.llm_max_tokens,
                    timeout_seconds=runtime_config.llm_timeout_seconds,
                    max_retries=runtime_config.llm_max_retries,
                    cache_enabled=runtime_config.llm_cache_enabled,
                )
                self.llm_service = initialize_llm_service(llm_config)
            except Exception as e:
                logger.warning(
                    "cot_llm_initialization_failed",
                    error=str(e),
                    agent_id=self.agent_id,
                )
                self.use_real_llm = False
                self.llm_service = None
        else:
            self.llm_service = None

    async def initialize(self) -> None:
        """Initialize engine resources."""
        logger.info(
            "cot_engine_initialized",
            agent_id=self.agent_id,
        )

    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        logger.info(
            "cot_engine_cleanup",
            agent_id=self.agent_id,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        state: AgentExecutionState,
    ) -> dict[str, Any]:
        """
        Execute agent using Chain-of-Thought reasoning.

        Args:
            input_data: Input with 'goal', optional 'max_steps', 'verification_enabled'
            state: Current agent execution state

        Returns:
            Execution result with final conclusion and reasoning chain
        """
        # Initialize execution context
        goal = input_data.get("goal", "")
        max_steps = input_data.get("max_steps", 10)
        verification_enabled = input_data.get("verification_enabled", True)

        self.context = CoTExecutionContext(
            agent_id=self.agent_id,
            goal=goal,
            max_steps=max_steps,
            verification_enabled=verification_enabled,
        )

        logger.info(
            "cot_execution_start",
            agent_id=self.agent_id,
            goal=goal,
            max_steps=max_steps,
        )

        # Execute reasoning chain
        try:
            while not self.context.completed and self.context.current_step < max_steps:
                self.context.current_step += 1

                # Generate reasoning step
                step_content = await self._generate_step()
                self._add_step(CoTStepType.STEP, step_content)

                # Verify step if enabled
                if self.context.verification_enabled and self.context.current_step > 1:
                    verification = await self._verify_step(step_content)
                    self._add_step(CoTStepType.VERIFICATION, verification)

                    # Refine if verification suggests issues
                    if self._needs_refinement(verification):
                        refinement = await self._refine_step(step_content, verification)
                        self._add_step(CoTStepType.REFINEMENT, refinement)

                # Check for conclusion
                if self._is_conclusion(step_content):
                    conclusion = self._extract_conclusion(step_content)
                    self.context.final_conclusion = conclusion
                    self.context.completed = True
                    self._add_step(CoTStepType.CONCLUSION, conclusion)
                    break

                # Update context window
                self._update_context_window(step_content)

            # Handle incomplete execution
            if not self.context.completed:
                logger.warning(
                    "cot_max_steps_reached",
                    agent_id=self.agent_id,
                    steps=self.context.current_step,
                )
                self.context.final_conclusion = (
                    "Maximum steps reached without reaching a definitive conclusion."
                )

            logger.info(
                "cot_execution_complete",
                agent_id=self.agent_id,
                steps=self.context.current_step,
                total_steps=len(self.context.steps),
            )

            return {
                "final_conclusion": self.context.final_conclusion,
                "steps": self.context.current_step,
                "reasoning_chain": [step.model_dump() for step in self.context.steps],
                "completed": self.context.completed,
                "verification_enabled": self.context.verification_enabled,
            }

        except Exception as e:
            logger.error(
                "cot_execution_failed",
                agent_id=self.agent_id,
                error=str(e),
            )
            raise

    async def _generate_step(self) -> str:
        """
        Generate next reasoning step.

        Returns:
            Step content
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        # Format history and context
        history = self._format_history()
        context = self._format_context()

        # Create prompt
        prompt = self.prompt_template.step_prompt.format(
            goal=self.context.goal,
            history=history,
            context=context,
        )

        # Call LLM (simulated for now)
        response = await self._call_llm(
            prompt=prompt,
            system_prompt=self.prompt_template.system_prompt,
        )

        return response.content

    async def _verify_step(self, step_content: str) -> str:
        """
        Verify reasoning step validity.

        Args:
            step_content: Step content to verify

        Returns:
            Verification result
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        previous_steps = self._format_history()

        # Create verification prompt
        prompt = self.prompt_template.verification_prompt.format(
            step=step_content,
            previous_steps=previous_steps,
        )

        # Call LLM for verification
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a logical reasoning verifier. Check reasoning validity.",
            temperature=0.3,  # Lower temperature for verification
        )

        return response.content

    async def _refine_step(self, step_content: str, verification: str) -> str:
        """
        Refine reasoning step based on verification feedback.

        Args:
            step_content: Original step content
            verification: Verification feedback

        Returns:
            Refined step content
        """
        prompt = f"""Original reasoning step:
{step_content}

Verification feedback:
{verification}

Please refine the reasoning step based on the feedback."""

        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are refining reasoning steps based on verification feedback.",
        )

        return response.content

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Call LLM provider via Portkey.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature

        Returns:
            LLM response
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        # Use real LLM if configured
        if self.use_real_llm and self.llm_service:
            try:
                return await self.llm_service.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(
                    "cot_llm_call_failed",
                    error=str(e),
                    agent_id=self.agent_id,
                )
                # Fall back to simulated response on error
                logger.warning(
                    "cot_falling_back_to_simulated_llm", agent_id=self.agent_id
                )

        # Simulated LLM response for testing or fallback
        step_num = self.context.current_step

        # Simple rule-based simulation based on goal
        if step_num == 1:
            if "calculate" in self.context.goal.lower():
                content = f"Step 1: Break down the calculation in {self.context.goal}"
            elif "analyze" in self.context.goal.lower():
                content = (
                    f"Step 1: Identify key components to analyze in {self.context.goal}"
                )
            else:
                content = f"Step 1: Understand the goal - {self.context.goal}"
        elif step_num == 2:
            content = "Step 2: Apply logical reasoning to solve the problem"
        elif step_num >= 3:
            content = f"CONCLUSION: Based on the reasoning steps, I have analyzed {self.context.goal}"
        else:
            content = f"Step {step_num}: Continue reasoning process"

        return LLMResponse(
            content=content,
            model="gpt-4.1-simulated",
            tokens_used=100,
            finish_reason="stop",
            cached=False,
        )

    def _format_history(self) -> str:
        """
        Format execution history for prompt.

        Returns:
            Formatted history string
        """
        if not self.context or not self.context.steps:
            return "No previous steps"

        history_lines = []
        for step in self.context.steps:
            history_lines.append(f"{step.step_type.value.upper()}: {step.content}")

        return "\n".join(history_lines)

    def _format_context(self) -> str:
        """
        Format context window for prompt.

        Returns:
            Formatted context string
        """
        if not self.context or not self.context.context_window:
            return "No current context"

        return "\n".join(self.context.context_window[-3:])  # Last 3 items

    def _update_context_window(self, content: str) -> None:
        """
        Update context window with new content.

        Args:
            content: New content to add
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        self.context.context_window.append(content)

        # Keep window size manageable
        max_window_size = 5
        if len(self.context.context_window) > max_window_size:
            self.context.context_window = self.context.context_window[-max_window_size:]

    def _add_step(
        self,
        step_type: CoTStepType,
        content: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Add a step to execution context.

        Args:
            step_type: Type of step
            content: Step content
            confidence: Confidence score
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        step = CoTStep(
            step_number=len(self.context.steps) + 1,
            step_type=step_type,
            content=content,
            confidence=confidence,
        )
        self.context.steps.append(step)

    def _is_conclusion(self, content: str) -> bool:
        """
        Check if content contains conclusion.

        Args:
            content: Step content

        Returns:
            True if contains conclusion
        """
        conclusion_markers = [
            "CONCLUSION:",
            "FINAL ANSWER:",
            "IN CONCLUSION",
            "THEREFORE,",
        ]
        return any(marker in content.upper() for marker in conclusion_markers)

    def _extract_conclusion(self, content: str) -> str:
        """
        Extract conclusion from content.

        Args:
            content: Step content

        Returns:
            Extracted conclusion
        """
        # Try to extract after conclusion markers
        match = re.search(
            r"(?:CONCLUSION:|FINAL ANSWER:|IN CONCLUSION|THEREFORE,)\s*(.+)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return content

    def _needs_refinement(self, verification: str) -> bool:
        """
        Check if verification suggests refinement is needed.

        Args:
            verification: Verification content

        Returns:
            True if refinement needed
        """
        refinement_indicators = [
            "issue",
            "error",
            "incorrect",
            "inconsistent",
            "revise",
            "refine",
        ]
        return any(
            indicator in verification.lower() for indicator in refinement_indicators
        )
