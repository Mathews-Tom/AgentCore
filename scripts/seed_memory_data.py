"""Seed data script for memory system tables.

This script creates sample data for testing the memory system migration (MEM-004).

Usage:
    uv run python scripts/seed_memory_data.py
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from agentcore.a2a_protocol.database.connection import get_session, init_db, close_db


async def seed_memory_data() -> None:
    """Seed sample data into memory system tables."""

    print("Starting memory system data seeding...")

    # Initialize database connection
    await init_db()

    try:
        async with get_session() as session:
            # Generate test UUIDs
            agent_id = uuid.uuid4()
            session_id = uuid.uuid4()
            task_id = uuid.uuid4()
            stage_id = uuid.uuid4()
            memory_id_1 = uuid.uuid4()
            memory_id_2 = uuid.uuid4()
            error_id = uuid.uuid4()
            metric_id = uuid.uuid4()

            # 1. Insert sample memories
            print("Inserting sample memories...")
            await session.execute(
            text("""
                INSERT INTO memories (
                    memory_id, memory_layer, content, summary, embedding,
                    agent_id, session_id, task_id, timestamp,
                    entities, facts, keywords,
                    related_memory_ids, relevance_score, access_count,
                    stage_id, is_critical, criticality_reason
                ) VALUES (
                    :memory_id_1, 'episodic', 'User requested authentication feature',
                    'User wants JWT auth', ARRAY[0.1, 0.2, 0.3],
                    :agent_id, :session_id, :task_id, :timestamp,
                    ARRAY['user', 'authentication'], ARRAY['JWT auth required'], ARRAY['auth', 'jwt'],
                    ARRAY[]::uuid[], 0.9, 0,
                    :stage_id, true, 'Critical user requirement'
                ), (
                    :memory_id_2, 'semantic', 'JWT tokens should expire after 1 hour',
                    'Token expiration policy', ARRAY[0.4, 0.5, 0.6],
                    :agent_id, :session_id, :task_id, :timestamp,
                    ARRAY['JWT', 'token'], ARRAY['1 hour expiration'], ARRAY['security', 'ttl'],
                    ARRAY[:memory_id_1], 0.85, 2,
                    :stage_id, true, 'Security constraint'
                )
            """),
            {
                "memory_id_1": memory_id_1,
                "memory_id_2": memory_id_2,
                "agent_id": agent_id,
                "session_id": session_id,
                "task_id": task_id,
                "stage_id": stage_id,
                "timestamp": datetime.now(UTC),
            }
            )

            # 2. Insert sample stage memory
            print("Inserting sample stage memory...")
            await session.execute(
            text("""
                INSERT INTO stage_memories (
                    stage_id, task_id, agent_id, stage_type,
                    stage_summary, stage_insights, raw_memory_refs,
                    relevance_score, compression_ratio, compression_model,
                    created_at, updated_at, completed_at
                ) VALUES (
                    :stage_id, :task_id, :agent_id, 'planning',
                    'Planning phase: Identified authentication requirements and security constraints',
                    ARRAY['JWT auth selected', 'Token expiration needed'],
                    ARRAY[:memory_id_1, :memory_id_2]::uuid[],
                    0.95, 10.2, 'gpt-4.1-mini',
                    :created_at, :updated_at, :completed_at
                )
            """),
            {
                "stage_id": stage_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "memory_id_1": memory_id_1,
                "memory_id_2": memory_id_2,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "completed_at": datetime.now(UTC) + timedelta(minutes=5),
            }
            )

            # 3. Insert sample task context
            print("Inserting sample task context...")
            await session.execute(
            text("""
                INSERT INTO task_contexts (
                    task_id, agent_id, task_goal, current_stage_id,
                    task_progress_summary, critical_constraints,
                    performance_metrics, created_at, updated_at
                ) VALUES (
                    :task_id, :agent_id, 'Implement authentication system',
                    :stage_id, 'Planning completed. JWT auth approach selected.',
                    ARRAY['Use JWT', 'Token TTL: 1 hour', 'Secure storage required'],
                    :performance_metrics, :created_at, :updated_at
                )
            """),
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "stage_id": stage_id,
                "performance_metrics": '{"stages_completed": 1, "error_rate": 0.0, "progress_rate": 0.25}',
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            )

            # 4. Insert sample error record
            print("Inserting sample error record...")
            await session.execute(
            text("""
                INSERT INTO error_records (
                    error_id, task_id, stage_id, agent_id,
                    error_type, error_description, context_when_occurred,
                    recovery_action, error_severity, recorded_at
                ) VALUES (
                    :error_id, :task_id, :stage_id, :agent_id,
                    'missing_info', 'Token refresh endpoint not documented',
                    'During planning phase, reviewing API documentation',
                    'Added refresh endpoint to requirements', 0.6, :recorded_at
                )
            """),
            {
                "error_id": error_id,
                "task_id": task_id,
                "stage_id": stage_id,
                "agent_id": agent_id,
                "recorded_at": datetime.now(UTC),
            }
            )

            # 5. Insert sample compression metric
            print("Inserting sample compression metric...")
            await session.execute(
            text("""
                INSERT INTO compression_metrics (
                    metric_id, stage_id, task_id, compression_type,
                    input_tokens, output_tokens, compression_ratio,
                    critical_fact_retention_rate, coherence_score,
                    cost_usd, model_used, recorded_at
                ) VALUES (
                    :metric_id, :stage_id, :task_id, 'stage_compression',
                    2500, 245, 10.2, 0.97, 1.0,
                    0.0042, 'gpt-4.1-mini', :recorded_at
                )
            """),
            {
                "metric_id": metric_id,
                "stage_id": stage_id,
                "task_id": task_id,
                "recorded_at": datetime.now(UTC),
            }
            )

            await session.commit()

            print("\n✅ Seed data inserted successfully!")
            print(f"   - Agent ID: {agent_id}")
            print(f"   - Task ID: {task_id}")
            print(f"   - Stage ID: {stage_id}")
            print(f"   - 2 memories created")
            print(f"   - 1 stage memory created")
            print(f"   - 1 task context created")
            print(f"   - 1 error record created")
            print(f"   - 1 compression metric created")

            # Verify data
            print("\nVerifying data...")
            result = await session.execute(text("SELECT COUNT(*) FROM memories"))
            memory_count = result.scalar()
            print(f"   Memories: {memory_count}")

            result = await session.execute(text("SELECT COUNT(*) FROM stage_memories"))
            stage_count = result.scalar()
            print(f"   Stage memories: {stage_count}")

            result = await session.execute(text("SELECT COUNT(*) FROM task_contexts"))
            task_count = result.scalar()
            print(f"   Task contexts: {task_count}")

            result = await session.execute(text("SELECT COUNT(*) FROM error_records"))
            error_count = result.scalar()
            print(f"   Error records: {error_count}")

            result = await session.execute(text("SELECT COUNT(*) FROM compression_metrics"))
            metric_count = result.scalar()
            print(f"   Compression metrics: {metric_count}")

            print("\n✅ All seed data verified successfully!")
    finally:
        await close_db()


async def clean_seed_data() -> None:
    """Remove seed data (useful for testing)."""

    print("Cleaning seed data...")

    # Initialize database connection
    await init_db()

    try:
        async with get_session() as session:
            # Delete in reverse order of insertion to respect potential foreign keys
            await session.execute(text("DELETE FROM compression_metrics"))
            await session.execute(text("DELETE FROM error_records"))
            await session.execute(text("DELETE FROM task_contexts"))
            await session.execute(text("DELETE FROM stage_memories"))
            await session.execute(text("DELETE FROM memories"))

            await session.commit()

            print("✅ Seed data cleaned successfully!")
    finally:
        await close_db()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        asyncio.run(clean_seed_data())
    else:
        asyncio.run(seed_memory_data())
