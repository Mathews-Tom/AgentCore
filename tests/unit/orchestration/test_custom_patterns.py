"""
Unit Tests for Custom Pattern Framework

Tests pattern definition, validation, registration, and management.
"""

from uuid import uuid4

import pytest

from agentcore.orchestration.patterns.custom import (
    AgentRequirement,
    CoordinationConfig,
    CoordinationModel,
    PatternDefinition,
    PatternMetadata,
    PatternRegistry,
    PatternStatus,
    PatternType,
    TaskNode,
    ValidationRule,
)


class TestPatternMetadata:
    """Test PatternMetadata model."""

    def test_pattern_metadata_creation(self):
        """Test creating pattern metadata."""
        metadata = PatternMetadata(
            name="test_pattern",
            description="A test pattern",
            version="1.0.0",
            author="Test Author",
            tags=["test", "custom"],
        )

        assert metadata.name == "test_pattern"
        assert metadata.description == "A test pattern"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert len(metadata.tags) == 2
        assert metadata.created_at is not None
        assert metadata.updated_at is not None


class TestAgentRequirement:
    """Test AgentRequirement model."""

    def test_agent_requirement_creation(self):
        """Test creating agent requirement."""
        req = AgentRequirement(
            role="researcher",
            capabilities=["search", "analyze"],
            min_count=1,
            max_count=3,
            resource_requirements={"cpu": "2 cores", "memory": "4GB"},
        )

        assert req.role == "researcher"
        assert len(req.capabilities) == 2
        assert req.min_count == 1
        assert req.max_count == 3
        assert req.resource_requirements["cpu"] == "2 cores"


class TestTaskNode:
    """Test TaskNode model."""

    def test_task_node_creation(self):
        """Test creating task node."""
        task = TaskNode(
            task_id="task_1",
            agent_role="researcher",
            depends_on=["task_0"],
            parallel=False,
            timeout_seconds=600,
            max_retries=5,
        )

        assert task.task_id == "task_1"
        assert task.agent_role == "researcher"
        assert task.depends_on == ["task_0"]
        assert task.parallel is False
        assert task.timeout_seconds == 600
        assert task.max_retries == 5


class TestCoordinationConfig:
    """Test CoordinationConfig model."""

    def test_coordination_config_creation(self):
        """Test creating coordination configuration."""
        config = CoordinationConfig(
            model=CoordinationModel.HYBRID,
            event_driven_triggers=["agent.ready", "task.complete"],
            graph_based_tasks=["task_1", "task_2"],
            timeout_seconds=7200,
            max_concurrent_tasks=20,
        )

        assert config.model == CoordinationModel.HYBRID
        assert len(config.event_driven_triggers) == 2
        assert len(config.graph_based_tasks) == 2
        assert config.timeout_seconds == 7200
        assert config.max_concurrent_tasks == 20


class TestValidationRule:
    """Test ValidationRule model."""

    def test_validation_rule_creation(self):
        """Test creating validation rule."""
        rule = ValidationRule(
            rule_id="rule_1",
            rule_type="agent_capability",
            condition={"capability": "search"},
            error_message="Agent must have search capability",
        )

        assert rule.rule_id == "rule_1"
        assert rule.rule_type == "agent_capability"
        assert rule.condition["capability"] == "search"


class TestPatternDefinition:
    """Test PatternDefinition model."""

    def test_pattern_definition_creation(self):
        """Test creating a pattern definition."""
        metadata = PatternMetadata(
            name="research_pattern", description="Research workflow", version="1.0.0"
        )

        agents = {
            "researcher": AgentRequirement(
                role="researcher", capabilities=["search", "analyze"]
            )
        }

        tasks = [
            TaskNode(task_id="search", agent_role="researcher", depends_on=[]),
            TaskNode(
                task_id="analyze", agent_role="researcher", depends_on=["search"]
            ),
        ]

        coordination = CoordinationConfig(
            model=CoordinationModel.GRAPH_BASED, graph_based_tasks=["search", "analyze"]
        )

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
        )

        assert pattern.metadata.name == "research_pattern"
        assert pattern.pattern_type == PatternType.CUSTOM
        assert pattern.status == PatternStatus.DRAFT
        assert len(pattern.agents) == 1
        assert len(pattern.tasks) == 2

    def test_pattern_validation_success(self):
        """Test successful pattern validation."""
        metadata = PatternMetadata(
            name="valid_pattern", description="Valid pattern", version="1.0.0"
        )

        agents = {
            "worker": AgentRequirement(role="worker", capabilities=["execute"])
        }

        tasks = [
            TaskNode(task_id="task_1", agent_role="worker", depends_on=[]),
            TaskNode(task_id="task_2", agent_role="worker", depends_on=["task_1"]),
        ]

        coordination = CoordinationConfig(
            model=CoordinationModel.GRAPH_BASED, graph_based_tasks=["task_1", "task_2"]
        )

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
        )

        is_valid, errors = pattern.validate_pattern()

        assert is_valid is True
        assert len(errors) == 0

    def test_pattern_validation_missing_agents(self):
        """Test pattern validation fails with no agents."""
        metadata = PatternMetadata(
            name="invalid_pattern", description="Invalid pattern", version="1.0.0"
        )

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={},  # No agents
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        is_valid, errors = pattern.validate_pattern()

        assert is_valid is False
        assert any("agent requirement" in err.lower() for err in errors)

    def test_pattern_validation_missing_tasks(self):
        """Test pattern validation fails with no tasks."""
        metadata = PatternMetadata(
            name="invalid_pattern", description="Invalid pattern", version="1.0.0"
        )

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[],  # No tasks
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        is_valid, errors = pattern.validate_pattern()

        assert is_valid is False
        assert any("task" in err.lower() for err in errors)

    def test_pattern_validation_unknown_dependency(self):
        """Test pattern validation fails with unknown task dependency."""
        metadata = PatternMetadata(
            name="invalid_pattern", description="Invalid pattern", version="1.0.0"
        )

        agents = {
            "worker": AgentRequirement(role="worker", capabilities=["execute"])
        }

        tasks = [
            TaskNode(
                task_id="task_1",
                agent_role="worker",
                depends_on=["unknown_task"],  # Unknown dependency
            )
        ]

        coordination = CoordinationConfig(model=CoordinationModel.GRAPH_BASED)

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
        )

        is_valid, errors = pattern.validate_pattern()

        assert is_valid is False
        assert any("unknown task" in err.lower() for err in errors)

    def test_pattern_validation_unknown_agent_role(self):
        """Test pattern validation fails with unknown agent role."""
        metadata = PatternMetadata(
            name="invalid_pattern", description="Invalid pattern", version="1.0.0"
        )

        agents = {
            "worker": AgentRequirement(role="worker", capabilities=["execute"])
        }

        tasks = [
            TaskNode(
                task_id="task_1",
                agent_role="unknown_role",  # Unknown role
                depends_on=[],
            )
        ]

        coordination = CoordinationConfig(model=CoordinationModel.GRAPH_BASED)

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
        )

        is_valid, errors = pattern.validate_pattern()

        assert is_valid is False
        assert any("unknown agent role" in err.lower() for err in errors)

    def test_pattern_validation_circular_dependency(self):
        """Test pattern validation detects circular dependencies."""
        metadata = PatternMetadata(
            name="circular_pattern", description="Circular deps", version="1.0.0"
        )

        agents = {
            "worker": AgentRequirement(role="worker", capabilities=["execute"])
        }

        # Create circular dependency: A -> B -> C -> A
        tasks = [
            TaskNode(task_id="A", agent_role="worker", depends_on=["C"]),
            TaskNode(task_id="B", agent_role="worker", depends_on=["A"]),
            TaskNode(task_id="C", agent_role="worker", depends_on=["B"]),
        ]

        coordination = CoordinationConfig(model=CoordinationModel.GRAPH_BASED)

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
        )

        is_valid, errors = pattern.validate_pattern()

        assert is_valid is False
        assert any("circular" in err.lower() for err in errors)

    def test_pattern_to_template(self):
        """Test converting pattern to template format."""
        metadata = PatternMetadata(
            name="template_pattern", description="Template test", version="1.0.0"
        )

        agents = {
            "worker": AgentRequirement(role="worker", capabilities=["execute"])
        }

        tasks = [TaskNode(task_id="task_1", agent_role="worker", depends_on=[])]

        coordination = CoordinationConfig(model=CoordinationModel.GRAPH_BASED)

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
            template_parameters={"timeout": 3600},
        )

        template = pattern.to_template()

        assert template["name"] == "template_pattern"
        assert template["version"] == "1.0.0"
        assert template["type"] == PatternType.CUSTOM.value
        assert "worker" in template["agents"]
        assert len(template["tasks"]) == 1
        assert template["parameters"]["timeout"] == 3600


class TestPatternRegistry:
    """Test PatternRegistry."""

    def test_register_pattern_success(self):
        """Test successful pattern registration."""
        registry = PatternRegistry()

        metadata = PatternMetadata(
            name="test_pattern", description="Test", version="1.0.0"
        )

        agents = {
            "worker": AgentRequirement(role="worker", capabilities=["execute"])
        }

        tasks = [TaskNode(task_id="task_1", agent_role="worker", depends_on=[])]

        coordination = CoordinationConfig(model=CoordinationModel.GRAPH_BASED)

        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents=agents,
            tasks=tasks,
            coordination=coordination,
        )

        success, errors = registry.register_pattern(pattern)

        assert success is True
        assert len(errors) == 0
        assert pattern.status == PatternStatus.ACTIVE

    def test_register_invalid_pattern(self):
        """Test registering invalid pattern fails."""
        registry = PatternRegistry()

        metadata = PatternMetadata(
            name="invalid_pattern", description="Invalid", version="1.0.0"
        )

        # Invalid: no agents
        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={},
            tasks=[],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        success, errors = registry.register_pattern(pattern)

        assert success is False
        assert len(errors) > 0

    def test_register_duplicate_name(self):
        """Test registering pattern with duplicate name fails."""
        registry = PatternRegistry()

        # Create first pattern
        metadata1 = PatternMetadata(
            name="duplicate", description="First", version="1.0.0"
        )
        pattern1 = PatternDefinition(
            metadata=metadata1,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        # Create second pattern with same name
        metadata2 = PatternMetadata(
            name="duplicate", description="Second", version="2.0.0"
        )
        pattern2 = PatternDefinition(
            metadata=metadata2,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        registry.register_pattern(pattern1)
        success, errors = registry.register_pattern(pattern2)

        assert success is False
        assert any("already registered" in err.lower() for err in errors)

    def test_get_pattern(self):
        """Test retrieving pattern by ID."""
        registry = PatternRegistry()

        metadata = PatternMetadata(
            name="test_pattern", description="Test", version="1.0.0"
        )
        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        registry.register_pattern(pattern)

        retrieved = registry.get_pattern(pattern.pattern_id)
        assert retrieved is not None
        assert retrieved.metadata.name == "test_pattern"

    def test_get_pattern_by_name(self):
        """Test retrieving pattern by name."""
        registry = PatternRegistry()

        metadata = PatternMetadata(
            name="named_pattern", description="Test", version="1.0.0"
        )
        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        registry.register_pattern(pattern)

        retrieved = registry.get_pattern_by_name("named_pattern")
        assert retrieved is not None
        assert retrieved.pattern_id == pattern.pattern_id

    def test_list_patterns_no_filter(self):
        """Test listing all patterns."""
        registry = PatternRegistry()

        # Register multiple patterns
        for i in range(3):
            metadata = PatternMetadata(
                name=f"pattern_{i}", description="Test", version="1.0.0"
            )
            pattern = PatternDefinition(
                metadata=metadata,
                pattern_type=PatternType.CUSTOM,
                agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
                tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
                coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
            )
            registry.register_pattern(pattern)

        patterns = registry.list_patterns()
        assert len(patterns) == 3

    def test_list_patterns_by_type(self):
        """Test listing patterns filtered by type."""
        registry = PatternRegistry()

        # Register custom pattern
        metadata1 = PatternMetadata(
            name="custom_pattern", description="Custom", version="1.0.0"
        )
        pattern1 = PatternDefinition(
            metadata=metadata1,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        # Register supervisor pattern
        metadata2 = PatternMetadata(
            name="supervisor_pattern", description="Supervisor", version="1.0.0"
        )
        pattern2 = PatternDefinition(
            metadata=metadata2,
            pattern_type=PatternType.SUPERVISOR,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        registry.register_pattern(pattern1)
        registry.register_pattern(pattern2)

        custom_patterns = registry.list_patterns(pattern_type=PatternType.CUSTOM)
        assert len(custom_patterns) == 1
        assert custom_patterns[0].pattern_type == PatternType.CUSTOM

    def test_unregister_pattern(self):
        """Test unregistering (archiving) a pattern."""
        registry = PatternRegistry()

        metadata = PatternMetadata(
            name="test_pattern", description="Test", version="1.0.0"
        )
        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        registry.register_pattern(pattern)
        result = registry.unregister_pattern(pattern.pattern_id)

        assert result is True
        assert pattern.status == PatternStatus.ARCHIVED

    def test_update_statistics(self):
        """Test updating pattern execution statistics."""
        registry = PatternRegistry()

        metadata = PatternMetadata(
            name="test_pattern", description="Test", version="1.0.0"
        )
        pattern = PatternDefinition(
            metadata=metadata,
            pattern_type=PatternType.CUSTOM,
            agents={"worker": AgentRequirement(role="worker", capabilities=["execute"])},
            tasks=[TaskNode(task_id="task_1", agent_role="worker", depends_on=[])],
            coordination=CoordinationConfig(model=CoordinationModel.GRAPH_BASED),
        )

        registry.register_pattern(pattern)

        # Update with successful execution
        registry.update_statistics(pattern.pattern_id, execution_time=1.5, success=True)

        assert pattern.execution_count == 1
        assert pattern.success_count == 1
        assert pattern.failure_count == 0
        assert pattern.avg_execution_time_seconds == 1.5

        # Update with failed execution
        registry.update_statistics(pattern.pattern_id, execution_time=2.0, success=False)

        assert pattern.execution_count == 2
        assert pattern.success_count == 1
        assert pattern.failure_count == 1
        assert pattern.avg_execution_time_seconds == 1.75  # Average of 1.5 and 2.0
