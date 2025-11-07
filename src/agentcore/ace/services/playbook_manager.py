"""
Playbook Management Service

Handles context playbook lifecycle, delta application, and context compilation.
Provides high-level business logic for ACE self-supervised learning.
"""

from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database import (
    ContextDeltaDB,
    ContextPlaybookDB,
    DeltaRepository,
    PlaybookRepository,
)
from agentcore.ace.models.ace_models import (
    ContextDelta,
    ContextPlaybook,
    PlaybookCreateRequest,
    PlaybookResponse,
    PlaybookUpdateRequest,
)

logger = structlog.get_logger()


class PlaybookManager:
    """
    Context Playbook management service.

    Handles playbook CRUD operations, delta application, and context compilation.
    Core component of ACE's self-supervised learning system.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize playbook manager.

        Args:
            session: Async database session
        """
        self.session = session
        logger.info("Playbook manager initialized")

    async def create_playbook(
        self,
        request: PlaybookCreateRequest,
    ) -> PlaybookResponse:
        """
        Create a new context playbook for an agent.

        Args:
            request: Playbook creation request

        Returns:
            Created playbook response

        Raises:
            ValueError: If agent already has an active playbook
        """
        logger.info(
            "Creating playbook",
            agent_id=request.agent_id,
            context_size=len(str(request.context)),
        )

        # Check if agent already has a playbook
        existing = await PlaybookRepository.get_by_agent_id(
            self.session, request.agent_id
        )
        if existing:
            raise ValueError(
                f"Agent {request.agent_id} already has an active playbook. "
                f"Use update operation instead."
            )

        # Create playbook
        playbook_db = await PlaybookRepository.create(
            session=self.session,
            agent_id=request.agent_id,
            context=request.context,
            metadata=request.metadata or {},
        )

        await self.session.commit()

        logger.info(
            "Playbook created successfully",
            agent_id=request.agent_id,
            playbook_id=str(playbook_db.playbook_id),
            version=playbook_db.version,
        )

        return self._to_response(playbook_db)

    async def get_playbook(
        self,
        playbook_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> PlaybookResponse | None:
        """
        Get playbook by ID or agent ID.

        Args:
            playbook_id: Unique playbook identifier
            agent_id: Agent identifier (gets latest playbook)

        Returns:
            Playbook response if found, None otherwise

        Raises:
            ValueError: If neither playbook_id nor agent_id provided
        """
        if not playbook_id and not agent_id:
            raise ValueError("Either playbook_id or agent_id must be provided")

        if playbook_id:
            logger.debug("Getting playbook by ID", playbook_id=str(playbook_id))
            playbook_db = await PlaybookRepository.get_by_id(
                self.session, playbook_id
            )
        else:
            logger.debug("Getting playbook by agent ID", agent_id=agent_id)
            playbook_db = await PlaybookRepository.get_by_agent_id(
                self.session, agent_id  # type: ignore
            )

        if not playbook_db:
            logger.info("Playbook not found", playbook_id=playbook_id, agent_id=agent_id)
            return None

        return self._to_response(playbook_db)

    async def update_playbook(
        self,
        playbook_id: UUID,
        request: PlaybookUpdateRequest,
    ) -> PlaybookResponse:
        """
        Update playbook context (increments version).

        Args:
            playbook_id: Playbook identifier
            request: Update request with new context

        Returns:
            Updated playbook response

        Raises:
            ValueError: If playbook not found
        """
        logger.info(
            "Updating playbook",
            playbook_id=str(playbook_id),
            context_size=len(str(request.context)),
        )

        # Verify playbook exists
        playbook_db = await PlaybookRepository.get_by_id(self.session, playbook_id)
        if not playbook_db:
            raise ValueError(f"Playbook {playbook_id} not found")

        # Update context
        success = await PlaybookRepository.update_context(
            session=self.session,
            playbook_id=playbook_id,
            context=request.context,
        )

        if not success:
            raise ValueError(f"Failed to update playbook {playbook_id}")

        await self.session.commit()

        # Fetch updated playbook
        updated_playbook = await PlaybookRepository.get_by_id(self.session, playbook_id)

        logger.info(
            "Playbook updated successfully",
            playbook_id=str(playbook_id),
            new_version=updated_playbook.version if updated_playbook else None,
        )

        return self._to_response(updated_playbook)  # type: ignore

    async def apply_delta(
        self,
        playbook_id: UUID,
        delta_id: UUID,
    ) -> PlaybookResponse:
        """
        Apply a delta to playbook context.

        Args:
            playbook_id: Playbook identifier
            delta_id: Delta identifier to apply

        Returns:
            Updated playbook response

        Raises:
            ValueError: If playbook or delta not found, or delta already applied
        """
        logger.info(
            "Applying delta to playbook",
            playbook_id=str(playbook_id),
            delta_id=str(delta_id),
        )

        # Fetch playbook
        playbook_db = await PlaybookRepository.get_by_id(self.session, playbook_id)
        if not playbook_db:
            raise ValueError(f"Playbook {playbook_id} not found")

        # Fetch delta
        delta_db = await DeltaRepository.get_by_id(self.session, delta_id)
        if not delta_db:
            raise ValueError(f"Delta {delta_id} not found")

        # Check delta belongs to this playbook
        if delta_db.playbook_id != playbook_id:
            raise ValueError(
                f"Delta {delta_id} does not belong to playbook {playbook_id}"
            )

        # Check delta not already applied
        if delta_db.applied:
            raise ValueError(f"Delta {delta_id} already applied")

        # Apply delta changes to context
        updated_context = self._apply_delta_changes(
            playbook_db.context, delta_db.changes
        )

        # Update playbook with new context
        await PlaybookRepository.update_context(
            session=self.session,
            playbook_id=playbook_id,
            context=updated_context,
        )

        # Mark delta as applied
        await DeltaRepository.mark_applied(self.session, delta_id)

        await self.session.commit()

        # Fetch updated playbook
        updated_playbook = await PlaybookRepository.get_by_id(self.session, playbook_id)

        logger.info(
            "Delta applied successfully",
            playbook_id=str(playbook_id),
            delta_id=str(delta_id),
            new_version=updated_playbook.version if updated_playbook else None,
        )

        return self._to_response(updated_playbook)  # type: ignore

    async def compile_context(
        self,
        playbook_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> str:
        """
        Compile playbook into execution context string.

        Args:
            playbook_id: Playbook identifier
            agent_id: Agent identifier (gets latest playbook)

        Returns:
            Compiled context string for agent execution

        Raises:
            ValueError: If playbook not found or invalid parameters
        """
        logger.debug(
            "Compiling context",
            playbook_id=str(playbook_id) if playbook_id else None,
            agent_id=agent_id,
        )

        # Get playbook
        playbook = await self.get_playbook(playbook_id=playbook_id, agent_id=agent_id)
        if not playbook:
            raise ValueError("Playbook not found")

        # Compile context from playbook structure
        context_parts = []

        # Add agent metadata
        context_parts.append(f"# Agent Context (v{playbook.version})")
        context_parts.append("")

        # Add context sections
        for key, value in playbook.context.items():
            if isinstance(value, dict):
                context_parts.append(f"## {key.title()}")
                for subkey, subvalue in value.items():
                    context_parts.append(f"- {subkey}: {subvalue}")
                context_parts.append("")
            elif isinstance(value, list):
                context_parts.append(f"## {key.title()}")
                for item in value:
                    context_parts.append(f"- {item}")
                context_parts.append("")
            else:
                context_parts.append(f"## {key.title()}")
                context_parts.append(str(value))
                context_parts.append("")

        compiled_context = "\n".join(context_parts)

        logger.debug(
            "Context compiled",
            playbook_id=str(playbook.playbook_id),
            context_length=len(compiled_context),
        )

        return compiled_context

    async def list_playbooks(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> list[PlaybookResponse]:
        """
        List playbooks for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of playbooks to return

        Returns:
            List of playbook responses
        """
        logger.debug("Listing playbooks", agent_id=agent_id, limit=limit)

        playbooks_db = await PlaybookRepository.list_by_agent(
            self.session, agent_id, limit
        )

        return [self._to_response(pb) for pb in playbooks_db]

    async def delete_playbook(self, playbook_id: UUID) -> bool:
        """
        Delete a playbook.

        Args:
            playbook_id: Playbook identifier

        Returns:
            True if deleted, False if not found
        """
        logger.info("Deleting playbook", playbook_id=str(playbook_id))

        success = await PlaybookRepository.delete(self.session, playbook_id)
        if success:
            await self.session.commit()
            logger.info("Playbook deleted successfully", playbook_id=str(playbook_id))
        else:
            logger.warning("Playbook not found for deletion", playbook_id=str(playbook_id))

        return success

    def _apply_delta_changes(
        self,
        context: dict,
        changes: dict,
    ) -> dict:
        """
        Apply delta changes to context dictionary.

        Args:
            context: Current context dictionary
            changes: Delta changes to apply

        Returns:
            Updated context dictionary
        """
        # Deep copy to avoid mutation
        updated_context = dict(context)

        # Apply changes (merge strategy)
        for key, value in changes.items():
            if isinstance(value, dict) and key in updated_context:
                # Merge dictionaries
                updated_context[key] = {**updated_context[key], **value}
            else:
                # Replace value
                updated_context[key] = value

        return updated_context

    def _to_response(self, playbook_db: ContextPlaybookDB) -> PlaybookResponse:
        """
        Convert ORM model to response model.

        Args:
            playbook_db: ORM playbook instance

        Returns:
            Playbook response model
        """
        return PlaybookResponse(
            playbook_id=playbook_db.playbook_id,
            agent_id=playbook_db.agent_id,
            context=playbook_db.context,
            version=playbook_db.version,
            created_at=playbook_db.created_at,
            updated_at=playbook_db.updated_at,
            metadata=playbook_db.playbook_metadata,
        )
