"""
Unit tests for Context Chain utility.

Tests for multi-stage workflow orchestration with context accumulation.
"""

import pytest
from datetime import datetime
from agentcore.a2a_protocol.services.context_chain import (
    ContextChain,
    ContextStage,
    calendar_analysis_pattern,
    research_synthesis_pattern,
    multi_step_reasoning_pattern,
)


class TestContextStage:
    """Test ContextStage model."""

    def test_context_stage_creation(self):
        """Test creating a context stage."""
        stage = ContextStage(
            stage_id="test_stage",
            task_id="task_001",
            agent_id="agent_001",
            input_context={"input": "data"},
            output_context={"output": "result"},
            summary="Test summary",
            metadata={"key": "value"}
        )

        assert stage.stage_id == "test_stage"
        assert stage.task_id == "task_001"
        assert stage.agent_id == "agent_001"
        assert stage.input_context == {"input": "data"}
        assert stage.output_context == {"output": "result"}
        assert stage.summary == "Test summary"
        assert stage.metadata == {"key": "value"}
        assert isinstance(stage.timestamp, datetime)

    def test_context_stage_defaults(self):
        """Test context stage with default values."""
        stage = ContextStage(stage_id="test")

        assert stage.task_id is None
        assert stage.agent_id is None
        assert stage.input_context == {}
        assert stage.output_context == {}
        assert stage.summary is None
        assert stage.metadata == {}


class TestContextChain:
    """Test ContextChain class."""

    def test_init_empty_chain(self):
        """Test initializing empty context chain."""
        chain = ContextChain()

        assert chain.chain_id.startswith("chain_")
        assert len(chain.stages) == 0
        assert chain.accumulated_context == {}
        assert chain.input_transforms == {}

    def test_init_with_context(self):
        """Test initializing chain with initial context."""
        initial_context = {"user_query": "test query"}
        chain = ContextChain(initial_context=initial_context, chain_id="test_chain")

        assert chain.chain_id == "test_chain"
        assert chain.accumulated_context == initial_context

    def test_add_stage_basic(self):
        """Test adding a basic stage to the chain."""
        chain = ContextChain(initial_context={"initial": "data"})

        result = chain.add_stage(
            stage_id="stage1",
            output_context={"result": "value"},
            summary="First stage"
        )

        assert result is chain  # Method chaining
        assert len(chain.stages) == 1
        assert chain.stages[0].stage_id == "stage1"
        assert chain.stages[0].summary == "First stage"
        assert chain.accumulated_context == {"initial": "data", "result": "value"}

    def test_add_stage_with_metadata(self):
        """Test adding stage with all optional parameters."""
        chain = ContextChain()

        chain.add_stage(
            stage_id="stage1",
            output_context={"data": "value"},
            task_id="task_123",
            agent_id="agent_456",
            summary="Test stage",
            metadata={"duration": 1.5}
        )

        stage = chain.stages[0]
        assert stage.task_id == "task_123"
        assert stage.agent_id == "agent_456"
        assert stage.metadata == {"duration": 1.5}

    def test_add_stage_duplicate_id_raises_error(self):
        """Test that adding duplicate stage_id raises ValueError."""
        chain = ContextChain()
        chain.add_stage(stage_id="duplicate", output_context={})

        with pytest.raises(ValueError, match="Stage with id 'duplicate' already exists"):
            chain.add_stage(stage_id="duplicate", output_context={})

    def test_add_stage_with_input_transform(self):
        """Test adding stage with input transformation."""
        chain = ContextChain(initial_context={"value": 10})

        def double_value(ctx: dict[str, any]) -> dict[str, any]:
            return {"value": ctx["value"] * 2}

        chain.add_stage(
            stage_id="transform_stage",
            output_context={"doubled": True},
            input_transform=double_value
        )

        stage = chain.stages[0]
        assert stage.input_context == {"value": 20}  # Transformed
        assert "transform_stage" in chain.input_transforms

    def test_add_stage_input_transform_failure(self):
        """Test that failing input transform raises ValueError."""
        chain = ContextChain(initial_context={"value": 10})

        def failing_transform(ctx: dict[str, any]) -> dict[str, any]:
            raise RuntimeError("Transform failed")

        with pytest.raises(ValueError, match="Input transform failed"):
            chain.add_stage(
                stage_id="fail_stage",
                output_context={},
                input_transform=failing_transform
            )

    def test_context_accumulation(self):
        """Test that context accumulates across stages."""
        chain = ContextChain(initial_context={"a": 1})

        chain.add_stage("stage1", output_context={"b": 2})
        chain.add_stage("stage2", output_context={"c": 3})
        chain.add_stage("stage3", output_context={"d": 4})

        assert chain.accumulated_context == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_context_overwrite(self):
        """Test that later stages can overwrite context keys."""
        chain = ContextChain(initial_context={"value": 1})

        chain.add_stage("stage1", output_context={"value": 2})
        chain.add_stage("stage2", output_context={"value": 3})

        assert chain.accumulated_context == {"value": 3}

    def test_get_accumulated_context(self):
        """Test getting accumulated context returns a copy."""
        chain = ContextChain(initial_context={"original": "value"})
        chain.add_stage("stage1", output_context={"added": "data"})

        context = chain.get_accumulated_context()
        context["modified"] = "externally"

        # Original should not be modified
        assert "modified" not in chain.accumulated_context
        assert context == {"original": "value", "added": "data", "modified": "externally"}

    def test_get_stage(self):
        """Test retrieving a specific stage by ID."""
        chain = ContextChain()
        chain.add_stage("stage1", output_context={"a": 1})
        chain.add_stage("stage2", output_context={"b": 2})

        stage = chain.get_stage("stage1")
        assert stage is not None
        assert stage.stage_id == "stage1"

        missing_stage = chain.get_stage("nonexistent")
        assert missing_stage is None

    def test_get_lineage(self):
        """Test getting stage lineage."""
        chain = ContextChain()
        chain.add_stage("parse", output_context={})
        chain.add_stage("validate", output_context={})
        chain.add_stage("execute", output_context={})

        lineage = chain.get_lineage()
        assert lineage == ["parse", "validate", "execute"]

    def test_get_lineage_empty(self):
        """Test getting lineage from empty chain."""
        chain = ContextChain()
        assert chain.get_lineage() == []

    def test_get_task_lineage(self):
        """Test getting task lineage (excludes None values)."""
        chain = ContextChain()
        chain.add_stage("stage1", output_context={}, task_id="task_1")
        chain.add_stage("stage2", output_context={})  # No task_id
        chain.add_stage("stage3", output_context={}, task_id="task_3")

        task_lineage = chain.get_task_lineage()
        assert task_lineage == ["task_1", "task_3"]

    def test_generate_summary_empty(self):
        """Test generating summary for empty chain."""
        chain = ContextChain()
        summary = chain.generate_summary()
        assert summary == "Empty context chain"

    def test_generate_summary_with_stages(self):
        """Test generating summary with multiple stages."""
        chain = ContextChain()
        chain.add_stage("stage1", output_context={}, summary="Parse input")
        chain.add_stage("stage2", output_context={}, summary="Validate data")
        chain.add_stage("stage3", output_context={})  # No summary

        summary = chain.generate_summary()
        assert "Context Chain:" in summary
        assert "1. Parse input" in summary
        assert "2. Validate data" in summary
        assert "3. Stage stage3" in summary  # Default summary

    def test_export_for_artifact(self):
        """Test exporting chain data for TaskArtifact."""
        chain = ContextChain()
        chain.add_stage("stage1", output_context={}, task_id="task_1", summary="First")
        chain.add_stage("stage2", output_context={}, task_id="task_2", summary="Second")

        export = chain.export_for_artifact()

        assert "context_lineage" in export
        assert "context_summary" in export
        assert export["context_lineage"] == ["task_1", "task_2"]
        assert "First" in export["context_summary"]
        assert "Second" in export["context_summary"]

    def test_to_dict(self):
        """Test exporting chain to dictionary."""
        chain = ContextChain(
            initial_context={"initial": "data"},
            chain_id="test_chain"
        )
        chain.add_stage("stage1", output_context={"result": "value"}, task_id="task_1")

        data = chain.to_dict()

        assert data["chain_id"] == "test_chain"
        assert data["accumulated_context"] == {"initial": "data", "result": "value"}
        assert data["lineage"] == ["stage1"]
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage_id"] == "stage1"

    def test_from_dict(self):
        """Test creating chain from dictionary."""
        data = {
            "chain_id": "restored_chain",
            "accumulated_context": {"a": 1, "b": 2},
            "stages": [
                {
                    "stage_id": "stage1",
                    "task_id": "task_1",
                    "agent_id": None,
                    "input_context": {"a": 1},
                    "output_context": {"b": 2},
                    "summary": "Test stage",
                    "timestamp": "2024-01-01T00:00:00",
                    "metadata": {}
                }
            ]
        }

        chain = ContextChain.from_dict(data)

        assert chain.chain_id == "restored_chain"
        assert chain.accumulated_context == {"a": 1, "b": 2}
        assert len(chain.stages) == 1
        assert chain.stages[0].stage_id == "stage1"

    def test_len(self):
        """Test __len__ returns number of stages."""
        chain = ContextChain()
        assert len(chain) == 0

        chain.add_stage("stage1", output_context={})
        assert len(chain) == 1

        chain.add_stage("stage2", output_context={})
        assert len(chain) == 2

    def test_repr(self):
        """Test __repr__ returns string representation."""
        chain = ContextChain(chain_id="test_chain")
        chain.add_stage("stage1", output_context={})
        chain.add_stage("stage2", output_context={})

        repr_str = repr(chain)
        assert "ContextChain" in repr_str
        assert "test_chain" in repr_str
        assert "stages=2" in repr_str

    def test_method_chaining(self):
        """Test that add_stage supports method chaining."""
        chain = ContextChain(initial_context={"start": True})

        result = (
            chain
            .add_stage("stage1", output_context={"step1": True})
            .add_stage("stage2", output_context={"step2": True})
            .add_stage("stage3", output_context={"step3": True})
        )

        assert result is chain
        assert len(chain) == 3
        assert chain.accumulated_context == {
            "start": True,
            "step1": True,
            "step2": True,
            "step3": True
        }


class TestContextChainPatterns:
    """Test pre-configured context chain patterns."""

    def test_calendar_analysis_pattern(self):
        """Test calendar analysis pattern initialization."""
        chain = calendar_analysis_pattern("What meetings do I have today?")

        assert chain.chain_id == "calendar_analysis"
        assert chain.accumulated_context["user_query"] == "What meetings do I have today?"
        assert chain.accumulated_context["domain"] == "calendar"
        assert len(chain) == 0  # No stages yet

    def test_research_synthesis_pattern(self):
        """Test research synthesis pattern initialization."""
        chain = research_synthesis_pattern("Climate change impacts")

        assert chain.chain_id == "research_synthesis"
        assert chain.accumulated_context["topic"] == "Climate change impacts"
        assert chain.accumulated_context["sources"] == []
        assert chain.accumulated_context["findings"] == []

    def test_multi_step_reasoning_pattern(self):
        """Test multi-step reasoning pattern initialization."""
        chain = multi_step_reasoning_pattern("Solve quadratic equation")

        assert chain.chain_id == "multi_step_reasoning"
        assert chain.accumulated_context["problem"] == "Solve quadratic equation"
        assert chain.accumulated_context["steps"] == []
        assert chain.accumulated_context["solution"] is None


class TestContextChainIntegration:
    """Integration tests for context chain workflows."""

    def test_complete_workflow(self):
        """Test a complete multi-stage workflow."""
        # Simulate a calendar analysis workflow
        chain = calendar_analysis_pattern("Show my meetings for tomorrow")

        # Stage 1: Parse intent
        chain.add_stage(
            stage_id="parse_intent",
            task_id="task_001",
            agent_id="nlp_agent",
            output_context={
                "intent": "list_meetings",
                "time_range": "tomorrow",
                "parsed": True
            },
            summary="Parsed user query to extract intent and time range"
        )

        # Stage 2: Fetch calendar data
        chain.add_stage(
            stage_id="fetch_data",
            task_id="task_002",
            agent_id="calendar_agent",
            output_context={
                "meetings": [
                    {"title": "Team Standup", "time": "09:00"},
                    {"title": "Client Call", "time": "14:00"}
                ],
                "count": 2
            },
            summary="Fetched calendar data for tomorrow"
        )

        # Stage 3: Format response
        chain.add_stage(
            stage_id="format_response",
            task_id="task_003",
            agent_id="formatting_agent",
            output_context={
                "response": "You have 2 meetings tomorrow: Team Standup at 09:00, Client Call at 14:00"
            },
            summary="Formatted meetings into user-friendly response"
        )

        # Verify final state
        assert len(chain) == 3
        assert chain.get_lineage() == ["parse_intent", "fetch_data", "format_response"]
        assert chain.get_task_lineage() == ["task_001", "task_002", "task_003"]

        context = chain.get_accumulated_context()
        assert context["intent"] == "list_meetings"
        assert context["count"] == 2
        assert "response" in context

        summary = chain.generate_summary()
        assert "Parsed user query" in summary
        assert "Fetched calendar data" in summary
        assert "Formatted meetings" in summary

    def test_workflow_with_transformations(self):
        """Test workflow with input transformations between stages."""
        chain = ContextChain(initial_context={"raw_data": [1, 2, 3, 4, 5]})

        # Stage 1: Calculate sum
        chain.add_stage(
            stage_id="calculate_sum",
            output_context={"sum": 15},
            summary="Calculated sum of raw data"
        )

        # Stage 2: Calculate average (transform input to only use sum)
        def extract_sum(ctx: dict[str, any]) -> dict[str, any]:
            return {"sum": ctx["sum"], "count": len(ctx["raw_data"])}

        chain.add_stage(
            stage_id="calculate_average",
            output_context={"average": 3.0},
            input_transform=extract_sum,
            summary="Calculated average from sum"
        )

        # Verify transformations were applied
        avg_stage = chain.get_stage("calculate_average")
        assert avg_stage.input_context == {"sum": 15, "count": 5}
        assert chain.accumulated_context["average"] == 3.0

    def test_serialization_roundtrip(self):
        """Test serializing and deserializing a complete chain."""
        # Create original chain
        original = ContextChain(
            initial_context={"user": "test_user"},
            chain_id="test_chain"
        )
        original.add_stage("stage1", output_context={"result1": "value1"}, task_id="task1")
        original.add_stage("stage2", output_context={"result2": "value2"}, task_id="task2")

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = ContextChain.from_dict(data)

        # Verify restoration
        assert restored.chain_id == original.chain_id
        assert restored.accumulated_context == original.accumulated_context
        assert len(restored) == len(original)
        assert restored.get_lineage() == original.get_lineage()
        assert restored.get_task_lineage() == original.get_task_lineage()
