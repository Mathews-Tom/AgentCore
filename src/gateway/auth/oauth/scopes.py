"""
OAuth Scope Management

Scope-based authorization system for granular permission control.
Implements scope validation, permission checking, and scope expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ScopePermission(str, Enum):
    """Standard OAuth scope permissions."""

    # User information
    PROFILE_READ = "profile:read"
    PROFILE_WRITE = "profile:write"
    EMAIL_READ = "email:read"

    # Agent operations
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    AGENT_EXECUTE = "agent:execute"
    AGENT_DELETE = "agent:delete"

    # Task operations
    TASK_READ = "task:read"
    TASK_WRITE = "task:write"
    TASK_DELETE = "task:delete"

    # Admin operations
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"

    # Offline access
    OFFLINE_ACCESS = "offline_access"  # Allows refresh token issuance

    # OpenID Connect
    OPENID = "openid"  # OpenID Connect authentication


@dataclass
class ScopeDefinition:
    """Scope definition with metadata."""

    scope: str
    description: str
    requires: list[str] | None = None  # Scopes that this scope depends on
    implies: list[str] | None = None  # Scopes that this scope grants
    restricted: bool = False  # Requires admin approval
    resource: str | None = None  # Resource this scope applies to


class ScopeManager:
    """
    OAuth scope manager.

    Handles scope validation, permission checking, and scope expansion.
    """

    # Standard scope definitions
    SCOPE_DEFINITIONS: dict[str, ScopeDefinition] = {
        # Profile scopes
        ScopePermission.PROFILE_READ.value: ScopeDefinition(
            scope=ScopePermission.PROFILE_READ.value,
            description="Read user profile information",
            resource="user",
        ),
        ScopePermission.PROFILE_WRITE.value: ScopeDefinition(
            scope=ScopePermission.PROFILE_WRITE.value,
            description="Update user profile information",
            requires=[ScopePermission.PROFILE_READ.value],
            resource="user",
        ),
        ScopePermission.EMAIL_READ.value: ScopeDefinition(
            scope=ScopePermission.EMAIL_READ.value,
            description="Read user email address",
            resource="user",
        ),
        # Agent scopes
        ScopePermission.AGENT_READ.value: ScopeDefinition(
            scope=ScopePermission.AGENT_READ.value,
            description="Read agent information and status",
            resource="agent",
        ),
        ScopePermission.AGENT_WRITE.value: ScopeDefinition(
            scope=ScopePermission.AGENT_WRITE.value,
            description="Create and update agents",
            requires=[ScopePermission.AGENT_READ.value],
            resource="agent",
        ),
        ScopePermission.AGENT_EXECUTE.value: ScopeDefinition(
            scope=ScopePermission.AGENT_EXECUTE.value,
            description="Execute agent tasks",
            requires=[ScopePermission.AGENT_READ.value],
            resource="agent",
        ),
        ScopePermission.AGENT_DELETE.value: ScopeDefinition(
            scope=ScopePermission.AGENT_DELETE.value,
            description="Delete agents",
            requires=[ScopePermission.AGENT_READ.value],
            restricted=True,
            resource="agent",
        ),
        # Task scopes
        ScopePermission.TASK_READ.value: ScopeDefinition(
            scope=ScopePermission.TASK_READ.value,
            description="Read task information and results",
            resource="task",
        ),
        ScopePermission.TASK_WRITE.value: ScopeDefinition(
            scope=ScopePermission.TASK_WRITE.value,
            description="Create and update tasks",
            requires=[ScopePermission.TASK_READ.value],
            resource="task",
        ),
        ScopePermission.TASK_DELETE.value: ScopeDefinition(
            scope=ScopePermission.TASK_DELETE.value,
            description="Delete tasks",
            requires=[ScopePermission.TASK_READ.value],
            restricted=True,
            resource="task",
        ),
        # Admin scopes
        ScopePermission.ADMIN_USERS.value: ScopeDefinition(
            scope=ScopePermission.ADMIN_USERS.value,
            description="Manage users and permissions",
            restricted=True,
            resource="admin",
        ),
        ScopePermission.ADMIN_SYSTEM.value: ScopeDefinition(
            scope=ScopePermission.ADMIN_SYSTEM.value,
            description="Manage system configuration",
            restricted=True,
            resource="admin",
        ),
        ScopePermission.ADMIN_AUDIT.value: ScopeDefinition(
            scope=ScopePermission.ADMIN_AUDIT.value,
            description="Access audit logs and reports",
            restricted=True,
            resource="admin",
        ),
        # Special scopes
        ScopePermission.OFFLINE_ACCESS.value: ScopeDefinition(
            scope=ScopePermission.OFFLINE_ACCESS.value,
            description="Maintain access when user is offline (refresh tokens)",
        ),
        ScopePermission.OPENID.value: ScopeDefinition(
            scope=ScopePermission.OPENID.value,
            description="OpenID Connect authentication",
        ),
    }

    @classmethod
    def parse_scopes(cls, scope_string: str | None) -> set[str]:
        """
        Parse space-separated scope string into set of scopes.

        Args:
            scope_string: Space-separated scope string

        Returns:
            Set of scope strings
        """
        if not scope_string:
            return set()

        return {s.strip() for s in scope_string.split() if s.strip()}

    @classmethod
    def format_scopes(cls, scopes: set[str] | list[str]) -> str:
        """
        Format scopes as space-separated string.

        Args:
            scopes: Set or list of scope strings

        Returns:
            Space-separated scope string
        """
        return " ".join(sorted(scopes))

    @classmethod
    def validate_scope(cls, scope: str) -> bool:
        """
        Validate individual scope.

        Args:
            scope: Scope string to validate

        Returns:
            True if scope is valid, False otherwise
        """
        return scope in cls.SCOPE_DEFINITIONS

    @classmethod
    def validate_scopes(
        cls, scopes: str | set[str] | list[str]
    ) -> tuple[bool, list[str]]:
        """
        Validate multiple scopes.

        Args:
            scopes: Scopes to validate (string, set, or list)

        Returns:
            Tuple of (all_valid, invalid_scopes)
        """
        if isinstance(scopes, str):
            scope_set = cls.parse_scopes(scopes)
        else:
            scope_set = set(scopes)

        invalid = [s for s in scope_set if not cls.validate_scope(s)]
        return len(invalid) == 0, invalid

    @classmethod
    def expand_scopes(cls, scopes: set[str]) -> set[str]:
        """
        Expand scopes to include dependencies and implied scopes.

        Args:
            scopes: Set of requested scopes

        Returns:
            Expanded set of scopes including dependencies
        """
        expanded = set(scopes)

        for scope in scopes:
            definition = cls.SCOPE_DEFINITIONS.get(scope)
            if not definition:
                continue

            # Add required scopes
            if definition.requires:
                expanded.update(definition.requires)

            # Add implied scopes
            if definition.implies:
                expanded.update(definition.implies)

        return expanded

    @classmethod
    def filter_restricted_scopes(
        cls,
        scopes: set[str],
        is_admin: bool = False,
    ) -> tuple[set[str], set[str]]:
        """
        Filter out restricted scopes that require admin approval.

        Args:
            scopes: Set of requested scopes
            is_admin: Whether user has admin privileges

        Returns:
            Tuple of (allowed_scopes, rejected_scopes)
        """
        allowed = set()
        rejected = set()

        for scope in scopes:
            definition = cls.SCOPE_DEFINITIONS.get(scope)

            if not definition:
                rejected.add(scope)
                continue

            if definition.restricted and not is_admin:
                rejected.add(scope)
            else:
                allowed.add(scope)

        return allowed, rejected

    @classmethod
    def check_permission(
        cls,
        granted_scopes: str | set[str] | list[str],
        required_scope: str,
    ) -> bool:
        """
        Check if granted scopes include required permission.

        Args:
            granted_scopes: Scopes granted to user/token
            required_scope: Required scope for operation

        Returns:
            True if permission is granted, False otherwise
        """
        if isinstance(granted_scopes, str):
            scope_set = cls.parse_scopes(granted_scopes)
        else:
            scope_set = set(granted_scopes)

        # Expand granted scopes
        expanded = cls.expand_scopes(scope_set)

        return required_scope in expanded

    @classmethod
    def check_any_permission(
        cls,
        granted_scopes: str | set[str] | list[str],
        required_scopes: list[str],
    ) -> bool:
        """
        Check if granted scopes include any of the required permissions.

        Args:
            granted_scopes: Scopes granted to user/token
            required_scopes: List of acceptable scopes

        Returns:
            True if any permission is granted, False otherwise
        """
        if isinstance(granted_scopes, str):
            scope_set = cls.parse_scopes(granted_scopes)
        else:
            scope_set = set(granted_scopes)

        # Expand granted scopes
        expanded = cls.expand_scopes(scope_set)

        return any(req in expanded for req in required_scopes)

    @classmethod
    def check_all_permissions(
        cls,
        granted_scopes: str | set[str] | list[str],
        required_scopes: list[str],
    ) -> bool:
        """
        Check if granted scopes include all required permissions.

        Args:
            granted_scopes: Scopes granted to user/token
            required_scopes: List of required scopes

        Returns:
            True if all permissions are granted, False otherwise
        """
        if isinstance(granted_scopes, str):
            scope_set = cls.parse_scopes(granted_scopes)
        else:
            scope_set = set(granted_scopes)

        # Expand granted scopes
        expanded = cls.expand_scopes(scope_set)

        return all(req in expanded for req in required_scopes)

    @classmethod
    def get_scope_info(cls, scope: str) -> dict[str, Any] | None:
        """
        Get detailed information about a scope.

        Args:
            scope: Scope to get information for

        Returns:
            Dictionary with scope information, or None if not found
        """
        definition = cls.SCOPE_DEFINITIONS.get(scope)

        if not definition:
            return None

        return {
            "scope": definition.scope,
            "description": definition.description,
            "requires": definition.requires or [],
            "implies": definition.implies or [],
            "restricted": definition.restricted,
            "resource": definition.resource,
        }

    @classmethod
    def get_all_scopes(cls) -> list[dict[str, Any]]:
        """
        Get information about all available scopes.

        Returns:
            List of scope information dictionaries
        """
        return [cls.get_scope_info(scope) for scope in cls.SCOPE_DEFINITIONS.keys()]

    @classmethod
    def get_scopes_for_resource(cls, resource: str) -> list[str]:
        """
        Get all scopes for a specific resource type.

        Args:
            resource: Resource type (e.g., "agent", "task", "user")

        Returns:
            List of scopes for the resource
        """
        return [
            scope
            for scope, definition in cls.SCOPE_DEFINITIONS.items()
            if definition.resource == resource
        ]
