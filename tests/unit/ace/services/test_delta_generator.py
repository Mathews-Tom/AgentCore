"""
Unit tests for DeltaGenerator service.

Tests delta generation logic using mocked LLM client.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agentcore.ace.models.ace_models import ContextPlaybook, ExecutionTrace
from agentcore.ace.services import DeltaGenerator
from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage


class TestDeltaGenerator:
    """Tests for DeltaGenerator service."""

    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        """Create mock LLM client."""
        client = MagicMock()
        client.complete = AsyncMock()
        return client

    @pytest.fixture
    def generator(self, mock_llm_client: MagicMock) -> DeltaGenerator:
        """Create DeltaGenerator instance with mocked LLM client."""
        return DeltaGenerator(mock_llm_client)

    @pytest.fixture
    def sample_playbook(self) -> ContextPlaybook:
        """Create sample playbook for testing."""
        return ContextPlaybook(
            playbook_id=uuid4(),
            agent_id="agent-001",
            context={
                "goal": "Complete tasks efficiently",
                "strategies": {
                    "planning": "Break down into steps",
                    "execution": "Follow step-by-step",
                },
                "preferences": {"temperature": 0.7, "model": "gpt-4.1"},
                "failures": ["Task timeout in step 3"],
            },
            version=1,
        )

    @pytest.fixture
    def sample_success_trace(self) -> ExecutionTrace:
        """Create sample successful execution trace."""
        return ExecutionTrace(
            trace_id=uuid4(),
            agent_id="agent-001",
            task_id="task-123",
            execution_time=5.2,
            success=True,
            output_quality=0.92,
            metadata={"steps_completed": 5, "retry_count": 0},
        )

    @pytest.fixture
    def sample_failure_trace(self) -> ExecutionTrace:
        """Create sample failed execution trace."""
        return ExecutionTrace(
            trace_id=uuid4(),
            agent_id="agent-001",
            task_id="task-456",
            execution_time=8.5,
            success=False,
            error_message="Timeout waiting for API response in step 3",
            metadata={"steps_completed": 2, "retry_count": 3},
        )

    @pytest.mark.asyncio
    async def test_generate_deltas_success_trace(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test generating deltas from successful execution trace."""
        # Arrange
        llm_response_content = """```json
[
  {
    "changes": {"preferences.temperature": 0.8},
    "confidence": 0.85,
    "reasoning": "Higher temperature improved output quality without sacrificing accuracy in successful executions."
  },
  {
    "changes": {"strategies.optimization": "Cache intermediate results"},
    "confidence": 0.75,
    "reasoning": "Execution time can be reduced by caching step results, as seen in metadata."
  }
]
```"""

        mock_llm_client.complete.return_value = LLMResponse(
            content=llm_response_content,
            usage=LLMUsage(prompt_tokens=500, completion_tokens=150, total_tokens=650),
            latency_ms=1200,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(
            sample_success_trace, sample_playbook, model="gpt-4.1-mini", max_deltas=5
        )

        # Assert
        assert len(deltas) == 2
        assert deltas[0].playbook_id == sample_playbook.playbook_id
        assert deltas[0].changes == {"preferences.temperature": 0.8}
        assert deltas[0].confidence == 0.85
        assert "Higher temperature" in deltas[0].reasoning
        assert deltas[0].applied is False

        assert deltas[1].changes == {"strategies.optimization": "Cache intermediate results"}
        assert deltas[1].confidence == 0.75

        mock_llm_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_deltas_failure_trace(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_failure_trace: ExecutionTrace,
    ) -> None:
        """Test generating deltas from failed execution trace."""
        # Arrange
        llm_response_content = """[
  {
    "changes": {"preferences.timeout": 30},
    "confidence": 0.90,
    "reasoning": "Increase timeout to prevent API response failures as seen in error message."
  },
  {
    "changes": {"strategies.retry": "Exponential backoff with max 5 retries"},
    "confidence": 0.82,
    "reasoning": "Failed after 3 retries suggests need for better retry strategy."
  }
]"""

        mock_llm_client.complete.return_value = LLMResponse(
            content=llm_response_content,
            usage=LLMUsage(prompt_tokens=520, completion_tokens=160, total_tokens=680),
            latency_ms=1350,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(
            sample_failure_trace, sample_playbook, model="gpt-4.1-mini"
        )

        # Assert
        assert len(deltas) == 2
        assert deltas[0].changes == {"preferences.timeout": 30}
        assert deltas[0].confidence == 0.90
        assert "timeout" in deltas[0].reasoning.lower()

        assert deltas[1].changes == {"strategies.retry": "Exponential backoff with max 5 retries"}
        assert deltas[1].confidence == 0.82

    @pytest.mark.asyncio
    async def test_generate_deltas_empty_response(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test handling empty LLM response."""
        # Arrange
        mock_llm_client.complete.return_value = LLMResponse(
            content="[]",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=10, total_tokens=510),
            latency_ms=800,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(sample_success_trace, sample_playbook)

        # Assert
        assert len(deltas) == 0

    @pytest.mark.asyncio
    async def test_generate_deltas_malformed_json(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test handling malformed JSON in LLM response."""
        # Arrange
        mock_llm_client.complete.return_value = LLMResponse(
            content="This is not valid JSON {malformed",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=20, total_tokens=520),
            latency_ms=900,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(sample_success_trace, sample_playbook)

        # Assert
        assert len(deltas) == 0  # Should return empty list, not raise

    @pytest.mark.asyncio
    async def test_generate_deltas_single_object_response(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test handling single object instead of array in LLM response."""
        # Arrange
        llm_response_content = """{
  "changes": {"preferences.temperature": 0.9},
  "confidence": 0.88,
  "reasoning": "Single delta suggestion for improvement."
}"""

        mock_llm_client.complete.return_value = LLMResponse(
            content=llm_response_content,
            usage=LLMUsage(prompt_tokens=500, completion_tokens=80, total_tokens=580),
            latency_ms=1000,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(sample_success_trace, sample_playbook)

        # Assert
        assert len(deltas) == 1
        assert deltas[0].changes == {"preferences.temperature": 0.9}
        assert deltas[0].confidence == 0.88

    @pytest.mark.asyncio
    async def test_generate_deltas_invalid_delta_fields(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test handling delta objects with invalid/missing fields."""
        # Arrange
        llm_response_content = """[
  {
    "changes": {"valid": "change"},
    "confidence": 0.85,
    "reasoning": "Valid delta"
  },
  {
    "changes": {},
    "confidence": "invalid",
    "reasoning": "short"
  },
  {
    "confidence": 0.7,
    "reasoning": "Missing changes field"
  }
]"""

        mock_llm_client.complete.return_value = LLMResponse(
            content=llm_response_content,
            usage=LLMUsage(prompt_tokens=500, completion_tokens=120, total_tokens=620),
            latency_ms=1100,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(sample_success_trace, sample_playbook)

        # Assert
        assert len(deltas) == 1  # Only the valid delta
        assert deltas[0].changes == {"valid": "change"}

    @pytest.mark.asyncio
    async def test_generate_deltas_with_markdown_code_blocks(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test parsing JSON from markdown code blocks."""
        # Arrange
        llm_response_content = """Here's my analysis:

```json
[
  {
    "changes": {"test": "value"},
    "confidence": 0.8,
    "reasoning": "Test reasoning from markdown"
  }
]
```

Hope this helps!"""

        mock_llm_client.complete.return_value = LLMResponse(
            content=llm_response_content,
            usage=LLMUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600),
            latency_ms=1050,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        deltas = await generator.generate_deltas(sample_success_trace, sample_playbook)

        # Assert
        assert len(deltas) == 1
        assert deltas[0].changes == {"test": "value"}

    @pytest.mark.asyncio
    async def test_generate_deltas_llm_exception(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test handling LLM client exceptions."""
        # Arrange
        mock_llm_client.complete.side_effect = Exception("LLM API error")

        # Act & Assert
        with pytest.raises(Exception, match="LLM API error"):
            await generator.generate_deltas(sample_success_trace, sample_playbook)

    @pytest.mark.asyncio
    async def test_generate_deltas_max_deltas_limit(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test max_deltas parameter is included in prompt."""
        # Arrange
        mock_llm_client.complete.return_value = LLMResponse(
            content="[]",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=10, total_tokens=510),
            latency_ms=800,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Act
        await generator.generate_deltas(
            sample_success_trace, sample_playbook, max_deltas=3
        )

        # Assert
        call_args = mock_llm_client.complete.call_args
        user_message = call_args[0][0].messages[1]["content"]
        assert "up to 3 context improvement deltas" in user_message

    @pytest.mark.asyncio
    async def test_generate_deltas_custom_model(
        self,
        generator: DeltaGenerator,
        mock_llm_client: MagicMock,
        sample_playbook: ContextPlaybook,
        sample_success_trace: ExecutionTrace,
    ) -> None:
        """Test using custom LLM model."""
        # Arrange
        mock_llm_client.complete.return_value = LLMResponse(
            content="[]",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=10, total_tokens=510),
            latency_ms=800,
            provider="openai",
            model="gpt-5-mini",
        )

        # Act
        await generator.generate_deltas(
            sample_success_trace, sample_playbook, model="gpt-5-mini"
        )

        # Assert
        call_args = mock_llm_client.complete.call_args
        assert call_args[0][0].model == "gpt-5-mini"

    def test_summarize_context_dict(self, generator: DeltaGenerator) -> None:
        """Test context summarization for dict values."""
        # Arrange
        context = {
            "strategies": {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
            }
        }

        # Act
        summary = generator._summarize_context(context)

        # Assert
        assert "strategies" in summary
        assert len(summary["strategies"]) == 3  # Only first 3 items

    def test_summarize_context_list(self, generator: DeltaGenerator) -> None:
        """Test context summarization for list values."""
        # Arrange
        context = {"failures": ["error1", "error2", "error3", "error4", "error5", "error6"]}

        # Act
        summary = generator._summarize_context(context)

        # Assert
        assert "failures" in summary
        assert len(summary["failures"]) == 5  # Only first 5 items

    def test_summarize_context_long_string(self, generator: DeltaGenerator) -> None:
        """Test context summarization for long strings."""
        # Arrange
        long_string = "x" * 300
        context = {"description": long_string}

        # Act
        summary = generator._summarize_context(context)

        # Assert
        assert "description" in summary
        assert len(summary["description"]) == 203  # 200 chars + "..."
        assert summary["description"].endswith("...")

    def test_summarize_context_simple_values(self, generator: DeltaGenerator) -> None:
        """Test context summarization for simple values."""
        # Arrange
        context = {
            "temperature": 0.7,
            "model": "gpt-4.1",
            "enabled": True,
        }

        # Act
        summary = generator._summarize_context(context)

        # Assert
        assert summary == context  # No changes for simple values
