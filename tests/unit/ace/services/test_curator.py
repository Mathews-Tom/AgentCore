"""
Unit tests for SimpleCurator service.

Tests confidence-threshold filtering logic for delta curation.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.ace.models.ace_models import ContextDelta
from agentcore.ace.services import SimpleCurator


class TestSimpleCurator:
    """Tests for SimpleCurator service."""

    @pytest.fixture
    def curator(self) -> SimpleCurator:
        """Create SimpleCurator instance with default threshold."""
        return SimpleCurator()

    @pytest.fixture
    def custom_curator(self) -> SimpleCurator:
        """Create SimpleCurator instance with custom threshold."""
        return SimpleCurator(threshold=0.9)

    @pytest.fixture
    def sample_deltas(self) -> list[ContextDelta]:
        """Create sample deltas with varying confidence scores."""
        playbook_id = uuid4()
        return [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"strategy": "new approach 1"},
                confidence=0.95,
                reasoning="High confidence improvement based on success pattern",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"preference.temperature": 0.9},
                confidence=0.85,
                reasoning="Medium-high confidence optimization",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"goal": "optimized goal"},
                confidence=0.80,
                reasoning="Exactly at threshold boundary",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"constraint": "new constraint"},
                confidence=0.75,
                reasoning="Below threshold suggestion",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"metadata": "additional info"},
                confidence=0.60,
                reasoning="Low confidence speculative change",
            ),
        ]

    def test_curator_initialization_default_threshold(self) -> None:
        """Test SimpleCurator initializes with default threshold."""
        curator = SimpleCurator()
        assert curator.threshold == 0.8

    def test_curator_initialization_custom_threshold(self) -> None:
        """Test SimpleCurator initializes with custom threshold."""
        curator = SimpleCurator(threshold=0.75)
        assert curator.threshold == 0.75

    def test_curator_initialization_invalid_threshold_too_low(self) -> None:
        """Test SimpleCurator raises error for threshold < 0."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            SimpleCurator(threshold=-0.1)

    def test_curator_initialization_invalid_threshold_too_high(self) -> None:
        """Test SimpleCurator raises error for threshold > 1."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            SimpleCurator(threshold=1.5)

    def test_curator_initialization_boundary_threshold_zero(self) -> None:
        """Test SimpleCurator accepts threshold of 0.0."""
        curator = SimpleCurator(threshold=0.0)
        assert curator.threshold == 0.0

    def test_curator_initialization_boundary_threshold_one(self) -> None:
        """Test SimpleCurator accepts threshold of 1.0."""
        curator = SimpleCurator(threshold=1.0)
        assert curator.threshold == 1.0

    def test_filter_deltas_default_threshold(
        self, curator: SimpleCurator, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas with default threshold (0.8)."""
        approved = curator.filter_deltas(sample_deltas)

        # With threshold 0.8, should approve deltas with confidence >= 0.8
        # Expected: 0.95, 0.85, 0.80 (3 deltas)
        assert len(approved) == 3
        assert all(delta.confidence >= 0.8 for delta in approved)
        assert approved[0].confidence == 0.95
        assert approved[1].confidence == 0.85
        assert approved[2].confidence == 0.80

    def test_filter_deltas_custom_threshold(
        self, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas with custom threshold (0.9)."""
        curator = SimpleCurator(threshold=0.9)
        approved = curator.filter_deltas(sample_deltas)

        # With threshold 0.9, should approve only deltas with confidence >= 0.9
        # Expected: 0.95 (1 delta)
        assert len(approved) == 1
        assert approved[0].confidence == 0.95

    def test_filter_deltas_threshold_override(
        self, curator: SimpleCurator, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas with threshold override parameter."""
        # Curator has default threshold 0.8, but override to 0.75
        approved = curator.filter_deltas(sample_deltas, threshold=0.75)

        # With threshold 0.75, should approve deltas with confidence >= 0.75
        # Expected: 0.95, 0.85, 0.80, 0.75 (4 deltas)
        assert len(approved) == 4
        assert all(delta.confidence >= 0.75 for delta in approved)

    def test_filter_deltas_empty_list(self, curator: SimpleCurator) -> None:
        """Test filter_deltas with empty delta list."""
        approved = curator.filter_deltas([])
        assert approved == []

    def test_filter_deltas_all_above_threshold(self, curator: SimpleCurator) -> None:
        """Test filter_deltas when all deltas pass threshold."""
        playbook_id = uuid4()
        high_confidence_deltas = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key1": "value1"},
                confidence=0.95,
                reasoning="High confidence",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key2": "value2"},
                confidence=0.90,
                reasoning="High confidence",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key3": "value3"},
                confidence=0.85,
                reasoning="High confidence",
            ),
        ]

        approved = curator.filter_deltas(high_confidence_deltas)
        assert len(approved) == 3
        assert approved == high_confidence_deltas

    def test_filter_deltas_all_below_threshold(self, curator: SimpleCurator) -> None:
        """Test filter_deltas when all deltas fail threshold."""
        playbook_id = uuid4()
        low_confidence_deltas = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key1": "value1"},
                confidence=0.75,
                reasoning="Below threshold",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key2": "value2"},
                confidence=0.65,
                reasoning="Below threshold",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key3": "value3"},
                confidence=0.50,
                reasoning="Below threshold",
            ),
        ]

        approved = curator.filter_deltas(low_confidence_deltas)
        assert len(approved) == 0

    def test_filter_deltas_boundary_exactly_threshold(
        self, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas with delta confidence exactly at threshold."""
        playbook_id = uuid4()
        boundary_delta = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key": "value"},
                confidence=0.80,
                reasoning="Exactly at threshold",
            )
        ]

        approved = curator.filter_deltas(boundary_delta)
        assert len(approved) == 1
        assert approved[0].confidence == 0.80

    def test_filter_deltas_boundary_just_below_threshold(
        self, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas with delta confidence just below threshold."""
        playbook_id = uuid4()
        below_threshold_delta = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key": "value"},
                confidence=0.79,
                reasoning="Just below threshold",
            )
        ]

        approved = curator.filter_deltas(below_threshold_delta)
        assert len(approved) == 0

    def test_filter_deltas_boundary_just_above_threshold(
        self, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas with delta confidence just above threshold."""
        playbook_id = uuid4()
        above_threshold_delta = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key": "value"},
                confidence=0.81,
                reasoning="Just above threshold",
            )
        ]

        approved = curator.filter_deltas(above_threshold_delta)
        assert len(approved) == 1
        assert approved[0].confidence == 0.81

    @patch("agentcore.ace.services.curator.logger")
    def test_filter_deltas_logs_empty_list(
        self, mock_logger: MagicMock, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas logs info message for empty delta list."""
        curator.filter_deltas([])
        mock_logger.info.assert_any_call("No deltas to filter")

    @patch("agentcore.ace.services.curator.logger")
    def test_filter_deltas_logs_rejected_deltas(
        self, mock_logger: MagicMock, curator: SimpleCurator, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas logs rejected deltas with rationale."""
        curator.filter_deltas(sample_deltas)

        # Check that rejected deltas were logged
        # Expected rejections: confidence 0.75 and 0.60 (2 deltas)
        rejection_calls = [
            call for call in mock_logger.info.call_args_list
            if len(call.args) > 0 and "Delta rejected" in call.args[0]
        ]

        assert len(rejection_calls) == 2

        # Verify logging includes required fields
        for call in rejection_calls:
            kwargs = call.kwargs
            assert "delta_id" in kwargs
            assert "playbook_id" in kwargs
            assert "confidence_score" in kwargs
            assert "threshold" in kwargs
            assert "rationale" in kwargs
            assert "reasoning" in kwargs

    @patch("agentcore.ace.services.curator.logger")
    def test_filter_deltas_logs_summary(
        self, mock_logger: MagicMock, curator: SimpleCurator, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas logs summary of filtering results."""
        curator.filter_deltas(sample_deltas)

        # Find the summary log call
        summary_calls = [
            call for call in mock_logger.info.call_args_list
            if len(call.args) > 0 and "Delta filtering completed" in call.args[0]
        ]

        assert len(summary_calls) == 1
        summary_kwargs = summary_calls[0].kwargs

        assert summary_kwargs["total_deltas"] == 5
        assert summary_kwargs["approved_count"] == 3
        assert summary_kwargs["rejected_count"] == 2
        assert summary_kwargs["threshold"] == 0.8

    @patch("agentcore.ace.services.curator.logger")
    def test_filter_deltas_logs_rejection_rationale(
        self, mock_logger: MagicMock, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas logs correct rationale for rejection."""
        playbook_id = uuid4()
        rejected_delta = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key": "value"},
                confidence=0.65,
                reasoning="This is a low confidence suggestion",
            )
        ]

        curator.filter_deltas(rejected_delta)

        # Find rejection log
        rejection_calls = [
            call for call in mock_logger.info.call_args_list
            if len(call.args) > 0 and "Delta rejected" in call.args[0]
        ]

        assert len(rejection_calls) == 1
        kwargs = rejection_calls[0].kwargs

        assert kwargs["confidence_score"] == 0.65
        assert kwargs["threshold"] == 0.8
        assert "0.650 < threshold 0.800" in kwargs["rationale"]

    def test_filter_deltas_preserves_delta_order(
        self, curator: SimpleCurator, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas preserves original order of approved deltas."""
        approved = curator.filter_deltas(sample_deltas)

        # Approved should maintain order: 0.95, 0.85, 0.80
        assert approved[0].confidence == 0.95
        assert approved[1].confidence == 0.85
        assert approved[2].confidence == 0.80

    def test_filter_deltas_does_not_modify_input(
        self, curator: SimpleCurator, sample_deltas: list[ContextDelta]
    ) -> None:
        """Test filter_deltas does not modify input delta list."""
        original_count = len(sample_deltas)
        original_ids = [delta.delta_id for delta in sample_deltas]

        curator.filter_deltas(sample_deltas)

        assert len(sample_deltas) == original_count
        assert [delta.delta_id for delta in sample_deltas] == original_ids

    @patch("agentcore.ace.services.curator.logger")
    def test_filter_deltas_truncates_long_reasoning(
        self, mock_logger: MagicMock, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas truncates long reasoning in logs."""
        playbook_id = uuid4()
        long_reasoning = "A" * 150  # 150 characters
        delta_with_long_reasoning = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key": "value"},
                confidence=0.70,
                reasoning=long_reasoning,
            )
        ]

        curator.filter_deltas(delta_with_long_reasoning)

        # Find rejection log
        rejection_calls = [
            call for call in mock_logger.info.call_args_list
            if len(call.args) > 0 and "Delta rejected" in call.args[0]
        ]

        kwargs = rejection_calls[0].kwargs
        # Should truncate to 100 characters
        assert len(kwargs["reasoning"]) == 100

    @patch("agentcore.ace.services.curator.logger")
    def test_filter_deltas_keeps_short_reasoning(
        self, mock_logger: MagicMock, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas keeps short reasoning intact in logs."""
        playbook_id = uuid4()
        short_reasoning = "Short reason"
        delta_with_short_reasoning = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key": "value"},
                confidence=0.70,
                reasoning=short_reasoning,
            )
        ]

        curator.filter_deltas(delta_with_short_reasoning)

        # Find rejection log
        rejection_calls = [
            call for call in mock_logger.info.call_args_list
            if len(call.args) > 0 and "Delta rejected" in call.args[0]
        ]

        kwargs = rejection_calls[0].kwargs
        assert kwargs["reasoning"] == short_reasoning

    def test_filter_deltas_mixed_confidence_scores(
        self, curator: SimpleCurator
    ) -> None:
        """Test filter_deltas with varied confidence score distribution."""
        playbook_id = uuid4()
        mixed_deltas = [
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key1": "value1"},
                confidence=1.0,
                reasoning="Perfect confidence",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key2": "value2"},
                confidence=0.0,
                reasoning="No confidence",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key3": "value3"},
                confidence=0.5,
                reasoning="Medium confidence",
            ),
            ContextDelta(
                playbook_id=playbook_id,
                changes={"key4": "value4"},
                confidence=0.8,
                reasoning="At threshold",
            ),
        ]

        approved = curator.filter_deltas(mixed_deltas)

        # Should approve: 1.0 and 0.8
        assert len(approved) == 2
        assert approved[0].confidence == 1.0
        assert approved[1].confidence == 0.8
