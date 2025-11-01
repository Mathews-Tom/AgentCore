"""
Knowledge base for optimization patterns and insights

Stores and retrieves historical patterns, successful strategies,
and optimization knowledge for future reference and recommendations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.analytics.patterns import OptimizationPattern
from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationTargetType,
)


class KnowledgeEntryType(str, Enum):
    """Type of knowledge entry"""

    PATTERN = "pattern"
    STRATEGY = "strategy"
    LESSON = "lesson"
    ANTI_PATTERN = "anti_pattern"
    BEST_PRACTICE = "best_practice"


class KnowledgeEntry(BaseModel):
    """Single knowledge base entry"""

    entry_id: str
    entry_type: KnowledgeEntryType
    title: str
    description: str
    context: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    applicable_targets: list[OptimizationTargetType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    usage_count: int = 0
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBase:
    """
    Knowledge base for optimization insights

    Stores and manages historical patterns, successful strategies,
    lessons learned, and best practices. Provides efficient retrieval
    and recommendation support.

    Key features:
    - Pattern storage and versioning
    - Context-based retrieval
    - Usage tracking and success metrics
    - Automatic learning from results
    - Export/import for sharing knowledge
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """
        Initialize knowledge base

        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = storage_path
        self._entries: dict[str, KnowledgeEntry] = {}
        self._indices: dict[str, dict[str, list[str]]] = {
            "type": defaultdict(list),
            "target": defaultdict(list),
            "tag": defaultdict(list),
        }

        if storage_path and storage_path.exists():
            self._load_from_disk()

    async def add_entry(self, entry: KnowledgeEntry) -> None:
        """
        Add knowledge entry to base

        Args:
            entry: Knowledge entry to add
        """
        # Store entry
        self._entries[entry.entry_id] = entry

        # Update indices
        self._indices["type"][entry.entry_type.value].append(entry.entry_id)

        for target in entry.applicable_targets:
            self._indices["target"][target.value].append(entry.entry_id)

        for tag in entry.tags:
            self._indices["tag"][tag].append(entry.entry_id)

        # Persist if storage configured
        if self.storage_path:
            await self._save_to_disk()

    async def add_pattern(
        self,
        pattern: OptimizationPattern,
        evidence_ids: list[str] | None = None,
    ) -> KnowledgeEntry:
        """
        Add optimization pattern as knowledge entry

        Args:
            pattern: Optimization pattern to add
            evidence_ids: Optional optimization result IDs as evidence

        Returns:
            Created knowledge entry
        """
        entry = KnowledgeEntry(
            entry_id=f"pattern_{pattern.pattern_key}_{int(datetime.now(UTC).timestamp())}",
            entry_type=KnowledgeEntryType.PATTERN,
            title=pattern.pattern_description,
            description=f"Pattern with {pattern.success_rate:.1%} success rate",
            context={
                "pattern_type": pattern.pattern_type.value,
                "pattern_key": pattern.pattern_key,
                "sample_count": pattern.sample_count,
                "avg_improvement": pattern.avg_improvement,
                "avg_iterations": pattern.avg_iterations,
                "common_parameters": pattern.common_parameters,
            },
            evidence=evidence_ids or pattern.best_results,
            confidence_score=self._pattern_to_confidence(pattern),
            tags=self._extract_pattern_tags(pattern),
            success_rate=pattern.success_rate,
            metadata=pattern.metadata,
        )

        await self.add_entry(entry)
        return entry

    async def add_lesson_learned(
        self,
        title: str,
        description: str,
        result_ids: list[str],
        target_type: OptimizationTargetType | None = None,
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        """
        Add lesson learned from optimization results

        Args:
            title: Lesson title
            description: Detailed description
            result_ids: Supporting result IDs
            target_type: Optional target type
            tags: Optional tags

        Returns:
            Created knowledge entry
        """
        entry = KnowledgeEntry(
            entry_id=f"lesson_{int(datetime.now(UTC).timestamp())}",
            entry_type=KnowledgeEntryType.LESSON,
            title=title,
            description=description,
            evidence=result_ids,
            confidence_score=min(len(result_ids) / 10.0, 1.0),
            applicable_targets=[target_type] if target_type else [],
            tags=tags or [],
        )

        await self.add_entry(entry)
        return entry

    async def add_anti_pattern(
        self,
        title: str,
        description: str,
        failure_rate: float,
        result_ids: list[str],
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        """
        Add anti-pattern (pattern to avoid)

        Args:
            title: Anti-pattern title
            description: Why to avoid
            failure_rate: Rate of failures
            result_ids: Supporting result IDs
            tags: Optional tags

        Returns:
            Created knowledge entry
        """
        entry = KnowledgeEntry(
            entry_id=f"antipattern_{int(datetime.now(UTC).timestamp())}",
            entry_type=KnowledgeEntryType.ANTI_PATTERN,
            title=title,
            description=description,
            evidence=result_ids,
            confidence_score=min(len(result_ids) / 10.0, 1.0),
            tags=tags or [],
            success_rate=1.0 - failure_rate,
            metadata={"failure_rate": failure_rate},
        )

        await self.add_entry(entry)
        return entry

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        """
        Get knowledge entry by ID

        Args:
            entry_id: Entry ID

        Returns:
            Knowledge entry or None
        """
        return self._entries.get(entry_id)

    async def search_entries(
        self,
        entry_type: KnowledgeEntryType | None = None,
        target_type: OptimizationTargetType | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[KnowledgeEntry]:
        """
        Search knowledge entries

        Args:
            entry_type: Filter by entry type
            target_type: Filter by target type
            tags: Filter by tags (entries must have all tags)
            min_confidence: Minimum confidence score
            limit: Maximum results

        Returns:
            List of matching entries
        """
        # Start with all entries
        candidates = set(self._entries.keys())

        # Apply type filter
        if entry_type:
            candidates &= set(self._indices["type"][entry_type.value])

        # Apply target filter
        if target_type:
            candidates &= set(self._indices["target"][target_type.value])

        # Apply tag filters
        if tags:
            for tag in tags:
                candidates &= set(self._indices["tag"][tag])

        # Get entries and filter by confidence
        results = [
            self._entries[eid]
            for eid in candidates
            if self._entries[eid].confidence_score >= min_confidence
        ]

        # Sort by confidence and usage
        results.sort(
            key=lambda e: (e.confidence_score, e.usage_count, e.success_rate),
            reverse=True,
        )

        return results[:limit]

    async def get_best_practices(
        self,
        target_type: OptimizationTargetType | None = None,
        min_confidence: float = 0.7,
    ) -> list[KnowledgeEntry]:
        """
        Get best practices

        Args:
            target_type: Optional target type filter
            min_confidence: Minimum confidence (default 0.7)

        Returns:
            List of best practice entries
        """
        return await self.search_entries(
            entry_type=KnowledgeEntryType.BEST_PRACTICE,
            target_type=target_type,
            min_confidence=min_confidence,
        )

    async def get_anti_patterns(
        self,
        target_type: OptimizationTargetType | None = None,
        min_confidence: float = 0.5,
    ) -> list[KnowledgeEntry]:
        """
        Get anti-patterns to avoid

        Args:
            target_type: Optional target type filter
            min_confidence: Minimum confidence (default 0.5)

        Returns:
            List of anti-pattern entries
        """
        return await self.search_entries(
            entry_type=KnowledgeEntryType.ANTI_PATTERN,
            target_type=target_type,
            min_confidence=min_confidence,
        )

    async def record_usage(
        self,
        entry_id: str,
        was_successful: bool,
    ) -> None:
        """
        Record usage of knowledge entry

        Args:
            entry_id: Entry ID
            was_successful: Whether usage was successful
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return

        # Update usage count
        entry.usage_count += 1

        # Update success rate
        old_total = entry.usage_count - 1
        old_successes = old_total * entry.success_rate

        new_successes = old_successes + (1.0 if was_successful else 0.0)
        entry.success_rate = new_successes / entry.usage_count

        # Update timestamp
        entry.updated_at = datetime.now(UTC)

        # Persist if storage configured
        if self.storage_path:
            await self._save_to_disk()

    async def learn_from_results(
        self,
        results: list[OptimizationResult],
        min_sample_size: int = 3,
    ) -> list[KnowledgeEntry]:
        """
        Learn patterns and lessons from optimization results

        Args:
            results: Optimization results to learn from
            min_sample_size: Minimum samples to extract knowledge

        Returns:
            List of newly created knowledge entries
        """
        if len(results) < min_sample_size:
            return []

        new_entries = []

        # Identify highly successful strategies
        excellent_results = [r for r in results if r.improvement_percentage >= 0.30]
        if len(excellent_results) >= min_sample_size:
            # Extract common algorithm
            algos = [
                r.optimization_details.algorithm_used
                for r in excellent_results
                if r.optimization_details
            ]
            if algos:
                most_common = max(set(algos), key=algos.count)
                entry = await self.add_lesson_learned(
                    title=f"High-performance algorithm: {most_common}",
                    description=f"Algorithm {most_common} achieved 30%+ improvement "
                    f"in {len(excellent_results)} cases",
                    result_ids=[r.optimization_id for r in excellent_results],
                    tags=["high-performance", most_common],
                )
                new_entries.append(entry)

        # Identify failure patterns
        failed_results = [
            r
            for r in results
            if r.status.value == "failed" or r.improvement_percentage < 0.05
        ]
        if len(failed_results) >= min_sample_size:
            # Extract common failure algorithm
            algos = [
                r.optimization_details.algorithm_used
                for r in failed_results
                if r.optimization_details
            ]
            if algos:
                most_common = max(set(algos), key=algos.count)
                failure_rate = len(
                    [
                        r
                        for r in results
                        if r.optimization_details
                        and r.optimization_details.algorithm_used == most_common
                        and r.improvement_percentage < 0.05
                    ]
                ) / len(
                    [
                        r
                        for r in results
                        if r.optimization_details
                        and r.optimization_details.algorithm_used == most_common
                    ]
                )

                if failure_rate > 0.3:
                    entry = await self.add_anti_pattern(
                        title=f"Unreliable in certain contexts: {most_common}",
                        description=f"Algorithm {most_common} shows {failure_rate:.1%} "
                        f"failure rate in observed cases",
                        failure_rate=failure_rate,
                        result_ids=[r.optimization_id for r in failed_results],
                        tags=["unreliable", most_common],
                    )
                    new_entries.append(entry)

        return new_entries

    async def export_knowledge(self, output_path: Path) -> None:
        """
        Export knowledge base to JSON file

        Args:
            output_path: Output file path
        """
        data = {
            "exported_at": datetime.now(UTC).isoformat(),
            "entry_count": len(self._entries),
            "entries": [
                entry.model_dump(mode="json") for entry in self._entries.values()
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def import_knowledge(self, input_path: Path) -> int:
        """
        Import knowledge base from JSON file

        Args:
            input_path: Input file path

        Returns:
            Number of entries imported
        """
        with open(input_path) as f:
            data = json.load(f)

        count = 0
        for entry_data in data.get("entries", []):
            entry = KnowledgeEntry(**entry_data)
            await self.add_entry(entry)
            count += 1

        return count

    def get_statistics(self) -> dict[str, Any]:
        """
        Get knowledge base statistics

        Returns:
            Statistics dictionary
        """
        total_entries = len(self._entries)
        if total_entries == 0:
            return {"total_entries": 0}

        entries = list(self._entries.values())

        return {
            "total_entries": total_entries,
            "by_type": {
                entry_type.value: len(self._indices["type"][entry_type.value])
                for entry_type in KnowledgeEntryType
            },
            "by_target": {
                target.value: len(self._indices["target"][target.value])
                for target in OptimizationTargetType
            },
            "avg_confidence": sum(e.confidence_score for e in entries) / total_entries,
            "avg_success_rate": sum(e.success_rate for e in entries) / total_entries,
            "total_usage": sum(e.usage_count for e in entries),
            "most_used_entry": max(entries, key=lambda e: e.usage_count).title
            if entries
            else None,
        }

    def _pattern_to_confidence(self, pattern: OptimizationPattern) -> float:
        """
        Convert pattern to confidence score

        Args:
            pattern: Optimization pattern

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from pattern confidence
        confidence_map = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5,
            "insufficient": 0.3,
        }
        base = confidence_map.get(pattern.confidence.value, 0.5)

        # Adjust by success rate
        return min(base * (0.5 + pattern.success_rate * 0.5), 1.0)

    def _extract_pattern_tags(self, pattern: OptimizationPattern) -> list[str]:
        """
        Extract tags from pattern

        Args:
            pattern: Optimization pattern

        Returns:
            List of tags
        """
        tags = [pattern.pattern_type.value]

        # Add confidence tag
        if pattern.confidence.value in ("high", "medium"):
            tags.append("reliable")

        # Add performance tags
        if pattern.avg_improvement >= 0.30:
            tags.append("high-performance")
        elif pattern.avg_improvement >= 0.20:
            tags.append("target-performance")

        # Add efficiency tags
        if pattern.avg_iterations < 50:
            tags.append("fast-converging")

        return tags

    async def _save_to_disk(self) -> None:
        """Save knowledge base to disk"""
        if not self.storage_path:
            return

        await self.export_knowledge(self.storage_path)

    def _load_from_disk(self) -> None:
        """Load knowledge base from disk"""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = KnowledgeEntry(**entry_data)
                self._entries[entry.entry_id] = entry

                # Rebuild indices
                self._indices["type"][entry.entry_type.value].append(entry.entry_id)

                for target in entry.applicable_targets:
                    self._indices["target"][target.value].append(entry.entry_id)

                for tag in entry.tags:
                    self._indices["tag"][tag].append(entry.entry_id)

        except Exception:
            # If loading fails, start with empty knowledge base
            pass
