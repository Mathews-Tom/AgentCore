"""Tests for knowledge base"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from agentcore.dspy_optimization.analytics.patterns import (
    OptimizationPattern,
    PatternConfidence,
    PatternType,
)
from agentcore.dspy_optimization.insights.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationResult,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)


@pytest.fixture
def knowledge_base() -> KnowledgeBase:
    """Create knowledge base for testing"""
    return KnowledgeBase()


@pytest.fixture
def sample_pattern() -> OptimizationPattern:
    """Create sample pattern"""
    return OptimizationPattern(
        pattern_type=PatternType.ALGORITHM_EFFECTIVENESS,
        pattern_key="miprov2",
        pattern_description="MIPROv2 algorithm",
        success_rate=0.85,
        sample_count=20,
        confidence=PatternConfidence.HIGH,
        avg_improvement=0.25,
        avg_iterations=100,
        avg_duration_seconds=300.0,
        best_results=["result_1", "result_2", "result_3"],
        common_parameters={"temperature": 0.7},
        metadata={"algorithm": "miprov2"},
    )


@pytest.fixture
def sample_entry() -> KnowledgeEntry:
    """Create sample knowledge entry"""
    return KnowledgeEntry(
        entry_id="test_entry_1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Test Pattern",
        description="Test pattern description",
        confidence_score=0.8,
        success_rate=0.75,
        tags=["test", "algorithm"],
    )


@pytest.mark.asyncio
async def test_add_entry(knowledge_base: KnowledgeBase, sample_entry: KnowledgeEntry) -> None:
    """Test adding knowledge entry"""
    await knowledge_base.add_entry(sample_entry)

    entry = await knowledge_base.get_entry(sample_entry.entry_id)
    assert entry is not None
    assert entry.title == sample_entry.title
    assert entry.entry_type == sample_entry.entry_type


@pytest.mark.asyncio
async def test_add_pattern(knowledge_base: KnowledgeBase, sample_pattern: OptimizationPattern) -> None:
    """Test adding pattern as knowledge entry"""
    entry = await knowledge_base.add_pattern(
        sample_pattern, evidence_ids=["result_1", "result_2"]
    )

    assert entry.entry_type == KnowledgeEntryType.PATTERN
    assert entry.confidence_score > 0
    assert len(entry.evidence) == 2
    assert "algorithm_effectiveness" in entry.tags


@pytest.mark.asyncio
async def test_add_lesson_learned(knowledge_base: KnowledgeBase) -> None:
    """Test adding lesson learned"""
    entry = await knowledge_base.add_lesson_learned(
        title="Test Lesson",
        description="Always validate improvements",
        result_ids=["r1", "r2", "r3"],
        target_type=OptimizationTargetType.AGENT,
        tags=["validation"],
    )

    assert entry.entry_type == KnowledgeEntryType.LESSON
    assert len(entry.evidence) == 3
    assert OptimizationTargetType.AGENT in entry.applicable_targets


@pytest.mark.asyncio
async def test_add_anti_pattern(knowledge_base: KnowledgeBase) -> None:
    """Test adding anti-pattern"""
    entry = await knowledge_base.add_anti_pattern(
        title="Avoid X",
        description="X leads to failures",
        failure_rate=0.6,
        result_ids=["r1", "r2"],
        tags=["failure"],
    )

    assert entry.entry_type == KnowledgeEntryType.ANTI_PATTERN
    assert entry.metadata["failure_rate"] == 0.6
    assert entry.success_rate == 0.4


@pytest.mark.asyncio
async def test_search_entries_by_type(
    knowledge_base: KnowledgeBase, sample_entry: KnowledgeEntry
) -> None:
    """Test searching entries by type"""
    await knowledge_base.add_entry(sample_entry)

    # Add another entry of different type
    lesson_entry = KnowledgeEntry(
        entry_id="lesson_1",
        entry_type=KnowledgeEntryType.LESSON,
        title="Test Lesson",
        description="Test",
        confidence_score=0.7,
    )
    await knowledge_base.add_entry(lesson_entry)

    # Search for patterns only
    results = await knowledge_base.search_entries(entry_type=KnowledgeEntryType.PATTERN)
    assert len(results) == 1
    assert results[0].entry_id == sample_entry.entry_id


@pytest.mark.asyncio
async def test_search_entries_by_target_type(knowledge_base: KnowledgeBase) -> None:
    """Test searching entries by target type"""
    # Add entries with different target types
    agent_entry = KnowledgeEntry(
        entry_id="agent_1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Agent Pattern",
        description="Test",
        confidence_score=0.8,
        applicable_targets=[OptimizationTargetType.AGENT],
    )
    await knowledge_base.add_entry(agent_entry)

    workflow_entry = KnowledgeEntry(
        entry_id="workflow_1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Workflow Pattern",
        description="Test",
        confidence_score=0.7,
        applicable_targets=[OptimizationTargetType.WORKFLOW],
    )
    await knowledge_base.add_entry(workflow_entry)

    # Search for agent-specific entries
    results = await knowledge_base.search_entries(
        target_type=OptimizationTargetType.AGENT
    )
    assert len(results) == 1
    assert results[0].entry_id == agent_entry.entry_id


@pytest.mark.asyncio
async def test_search_entries_by_tags(knowledge_base: KnowledgeBase) -> None:
    """Test searching entries by tags"""
    entry1 = KnowledgeEntry(
        entry_id="e1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="E1",
        description="Test",
        confidence_score=0.8,
        tags=["tag1", "tag2"],
    )
    await knowledge_base.add_entry(entry1)

    entry2 = KnowledgeEntry(
        entry_id="e2",
        entry_type=KnowledgeEntryType.PATTERN,
        title="E2",
        description="Test",
        confidence_score=0.7,
        tags=["tag1", "tag3"],
    )
    await knowledge_base.add_entry(entry2)

    # Search for entries with both tag1 and tag2
    results = await knowledge_base.search_entries(tags=["tag1", "tag2"])
    assert len(results) == 1
    assert results[0].entry_id == entry1.entry_id


@pytest.mark.asyncio
async def test_search_entries_min_confidence(knowledge_base: KnowledgeBase) -> None:
    """Test searching entries with minimum confidence"""
    high_conf = KnowledgeEntry(
        entry_id="high",
        entry_type=KnowledgeEntryType.PATTERN,
        title="High",
        description="Test",
        confidence_score=0.9,
    )
    await knowledge_base.add_entry(high_conf)

    low_conf = KnowledgeEntry(
        entry_id="low",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Low",
        description="Test",
        confidence_score=0.4,
    )
    await knowledge_base.add_entry(low_conf)

    # Search with min confidence 0.7
    results = await knowledge_base.search_entries(min_confidence=0.7)
    assert len(results) == 1
    assert results[0].entry_id == high_conf.entry_id


@pytest.mark.asyncio
async def test_get_best_practices(knowledge_base: KnowledgeBase) -> None:
    """Test getting best practices"""
    best_practice = KnowledgeEntry(
        entry_id="bp1",
        entry_type=KnowledgeEntryType.BEST_PRACTICE,
        title="Best Practice",
        description="Test",
        confidence_score=0.8,
    )
    await knowledge_base.add_entry(best_practice)

    pattern = KnowledgeEntry(
        entry_id="p1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Pattern",
        description="Test",
        confidence_score=0.9,
    )
    await knowledge_base.add_entry(pattern)

    # Get best practices only
    results = await knowledge_base.get_best_practices()
    assert len(results) == 1
    assert results[0].entry_type == KnowledgeEntryType.BEST_PRACTICE


@pytest.mark.asyncio
async def test_get_anti_patterns(knowledge_base: KnowledgeBase) -> None:
    """Test getting anti-patterns"""
    anti_pattern = KnowledgeEntry(
        entry_id="ap1",
        entry_type=KnowledgeEntryType.ANTI_PATTERN,
        title="Anti-Pattern",
        description="Test",
        confidence_score=0.6,
    )
    await knowledge_base.add_entry(anti_pattern)

    # Get anti-patterns
    results = await knowledge_base.get_anti_patterns()
    assert len(results) == 1
    assert results[0].entry_type == KnowledgeEntryType.ANTI_PATTERN


@pytest.mark.asyncio
async def test_record_usage(knowledge_base: KnowledgeBase, sample_entry: KnowledgeEntry) -> None:
    """Test recording usage of entry"""
    await knowledge_base.add_entry(sample_entry)

    # Record successful usage
    await knowledge_base.record_usage(sample_entry.entry_id, was_successful=True)

    entry = await knowledge_base.get_entry(sample_entry.entry_id)
    assert entry is not None
    assert entry.usage_count == 1

    # Record another usage (failed)
    await knowledge_base.record_usage(sample_entry.entry_id, was_successful=False)

    entry = await knowledge_base.get_entry(sample_entry.entry_id)
    assert entry is not None
    assert entry.usage_count == 2
    assert 0 < entry.success_rate < 1


@pytest.mark.asyncio
async def test_learn_from_results(knowledge_base: KnowledgeBase) -> None:
    """Test learning patterns from results"""
    # Create excellent results
    results = []
    for i in range(5):
        result = OptimizationResult(
            optimization_id=f"opt_{i}",
            status=OptimizationStatus.COMPLETED,
            baseline_performance=PerformanceMetrics(
                success_rate=0.5, avg_cost_per_task=1.0, avg_latency_ms=100
            ),
            optimized_performance=PerformanceMetrics(
                success_rate=0.8, avg_cost_per_task=0.7, avg_latency_ms=80
            ),
            improvement_percentage=0.35,
            optimization_details=OptimizationDetails(
                algorithm_used="miprov2", iterations=100, parameters={}
            ),
        )
        results.append(result)

    # Learn from results
    new_entries = await knowledge_base.learn_from_results(results)

    assert len(new_entries) > 0
    assert any(e.entry_type == KnowledgeEntryType.LESSON for e in new_entries)


@pytest.mark.asyncio
async def test_export_import_knowledge() -> None:
    """Test exporting and importing knowledge base"""
    kb1 = KnowledgeBase()

    # Add some entries
    entry = KnowledgeEntry(
        entry_id="e1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Test",
        description="Test entry",
        confidence_score=0.8,
    )
    await kb1.add_entry(entry)

    # Export to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "knowledge.json"
        await kb1.export_knowledge(export_path)

        # Import to new knowledge base
        kb2 = KnowledgeBase()
        count = await kb2.import_knowledge(export_path)

        assert count == 1

        # Verify entry exists
        imported = await kb2.get_entry("e1")
        assert imported is not None
        assert imported.title == "Test"


@pytest.mark.asyncio
async def test_persistent_storage() -> None:
    """Test persistent storage"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "knowledge.json"

        # Create KB with storage
        kb1 = KnowledgeBase(storage_path=storage_path)

        entry = KnowledgeEntry(
            entry_id="e1",
            entry_type=KnowledgeEntryType.PATTERN,
            title="Test",
            description="Test entry",
            confidence_score=0.8,
        )
        await kb1.add_entry(entry)

        # Storage file should exist
        assert storage_path.exists()

        # Create new KB from same storage
        kb2 = KnowledgeBase(storage_path=storage_path)

        # Entry should be loaded
        loaded = await kb2.get_entry("e1")
        assert loaded is not None
        assert loaded.title == "Test"


def test_get_statistics(knowledge_base: KnowledgeBase) -> None:
    """Test getting statistics"""
    stats = knowledge_base.get_statistics()
    assert stats["total_entries"] == 0

    # Add entries and check stats
    import asyncio

    async def add_entries() -> None:
        entry1 = KnowledgeEntry(
            entry_id="e1",
            entry_type=KnowledgeEntryType.PATTERN,
            title="E1",
            description="Test",
            confidence_score=0.8,
            success_rate=0.9,
            usage_count=5,
        )
        await knowledge_base.add_entry(entry1)

        entry2 = KnowledgeEntry(
            entry_id="e2",
            entry_type=KnowledgeEntryType.LESSON,
            title="E2",
            description="Test",
            confidence_score=0.6,
            success_rate=0.7,
            usage_count=2,
        )
        await knowledge_base.add_entry(entry2)

    asyncio.run(add_entries())

    stats = knowledge_base.get_statistics()
    assert stats["total_entries"] == 2
    assert stats["by_type"]["pattern"] == 1
    assert stats["by_type"]["lesson"] == 1
    assert stats["avg_confidence"] == 0.7
    assert stats["total_usage"] == 7


@pytest.mark.asyncio
async def test_pattern_to_confidence(knowledge_base: KnowledgeBase, sample_pattern: OptimizationPattern) -> None:
    """Test pattern to confidence conversion"""
    confidence = knowledge_base._pattern_to_confidence(sample_pattern)

    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7  # High confidence pattern


@pytest.mark.asyncio
async def test_extract_pattern_tags(knowledge_base: KnowledgeBase, sample_pattern: OptimizationPattern) -> None:
    """Test extracting tags from pattern"""
    tags = knowledge_base._extract_pattern_tags(sample_pattern)

    assert "algorithm_effectiveness" in tags
    assert "reliable" in tags  # High confidence
    assert "target-performance" in tags  # 25% improvement
