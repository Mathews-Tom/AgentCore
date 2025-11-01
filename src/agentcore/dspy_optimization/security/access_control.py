"""
Access Control

Provides RBAC (Role-Based Access Control) for DSPy operations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class SecurityRole(str, Enum):
    """Security roles for access control"""

    ADMIN = "admin"  # Full access
    OPTIMIZER = "optimizer"  # Can run optimizations
    ANALYST = "analyst"  # Read-only access
    AUDITOR = "auditor"  # Audit and compliance access


class SecurityPermission(str, Enum):
    """Security permissions"""

    # Model operations
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DELETE = "model:delete"
    MODEL_ENCRYPT = "model:encrypt"

    # Optimization operations
    OPTIMIZATION_CREATE = "optimization:create"
    OPTIMIZATION_READ = "optimization:read"
    OPTIMIZATION_CANCEL = "optimization:cancel"

    # Data operations
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Security operations
    SECURITY_MANAGE = "security:manage"
    AUDIT_READ = "audit:read"
    AUDIT_WRITE = "audit:write"

    # System operations
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"


class AccessConfig(BaseModel):
    """Configuration for access control"""

    enable_rbac: bool = Field(default=True, description="Enable RBAC")
    default_role: SecurityRole = Field(
        default=SecurityRole.ANALYST, description="Default role for new users"
    )
    session_timeout_minutes: int = Field(
        default=60, description="Session timeout in minutes"
    )
    max_failed_attempts: int = Field(
        default=5, description="Max failed authentication attempts"
    )


class UserSession(BaseModel):
    """Active user session"""

    user_id: str
    role: SecurityRole
    permissions: set[SecurityPermission]
    created_at: datetime
    last_activity: datetime
    session_token: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AccessController:
    """
    Access control service.

    Provides RBAC with role and permission management.
    """

    # Role to permissions mapping
    ROLE_PERMISSIONS: dict[SecurityRole, set[SecurityPermission]] = {
        SecurityRole.ADMIN: {
            SecurityPermission.MODEL_READ,
            SecurityPermission.MODEL_WRITE,
            SecurityPermission.MODEL_DELETE,
            SecurityPermission.MODEL_ENCRYPT,
            SecurityPermission.OPTIMIZATION_CREATE,
            SecurityPermission.OPTIMIZATION_READ,
            SecurityPermission.OPTIMIZATION_CANCEL,
            SecurityPermission.DATA_READ,
            SecurityPermission.DATA_WRITE,
            SecurityPermission.DATA_DELETE,
            SecurityPermission.DATA_EXPORT,
            SecurityPermission.SECURITY_MANAGE,
            SecurityPermission.AUDIT_READ,
            SecurityPermission.AUDIT_WRITE,
            SecurityPermission.SYSTEM_CONFIG,
            SecurityPermission.SYSTEM_MONITOR,
        },
        SecurityRole.OPTIMIZER: {
            SecurityPermission.MODEL_READ,
            SecurityPermission.MODEL_WRITE,
            SecurityPermission.OPTIMIZATION_CREATE,
            SecurityPermission.OPTIMIZATION_READ,
            SecurityPermission.DATA_READ,
            SecurityPermission.DATA_WRITE,
            SecurityPermission.SYSTEM_MONITOR,
        },
        SecurityRole.ANALYST: {
            SecurityPermission.MODEL_READ,
            SecurityPermission.OPTIMIZATION_READ,
            SecurityPermission.DATA_READ,
            SecurityPermission.SYSTEM_MONITOR,
        },
        SecurityRole.AUDITOR: {
            SecurityPermission.MODEL_READ,
            SecurityPermission.OPTIMIZATION_READ,
            SecurityPermission.DATA_READ,
            SecurityPermission.AUDIT_READ,
            SecurityPermission.SYSTEM_MONITOR,
        },
    }

    def __init__(self, config: AccessConfig | None = None):
        self.config = config or AccessConfig()
        self.logger = structlog.get_logger()

        # Active sessions
        self._sessions: dict[str, UserSession] = {}

        # Failed authentication attempts
        self._failed_attempts: dict[str, int] = {}

        # Statistics
        self._access_stats = {
            "auth_success": 0,
            "auth_failed": 0,
            "access_granted": 0,
            "access_denied": 0,
            "sessions_created": 0,
            "sessions_expired": 0,
        }

        self.logger.info(
            "access_controller_initialized",
            rbac_enabled=self.config.enable_rbac,
            default_role=self.config.default_role,
        )

    def create_session(
        self,
        user_id: str,
        role: SecurityRole | None = None,
        session_token: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserSession:
        """
        Create user session.

        Args:
            user_id: User identifier
            role: User role (uses default if not provided)
            session_token: Session token (generated if not provided)
            metadata: Additional session metadata

        Returns:
            UserSession instance
        """
        import uuid

        role = role or self.config.default_role
        session_token = session_token or str(uuid.uuid4())

        session = UserSession(
            user_id=user_id,
            role=role,
            permissions=self.ROLE_PERMISSIONS[role],
            created_at=datetime.now(UTC),
            last_activity=datetime.now(UTC),
            session_token=session_token,
            metadata=metadata or {},
        )

        self._sessions[session_token] = session
        self._access_stats["sessions_created"] += 1

        self.logger.info(
            "session_created",
            user_id=user_id,
            role=role.value,
            session_token=session_token,
        )

        return session

    def get_session(self, session_token: str) -> UserSession | None:
        """
        Get user session.

        Args:
            session_token: Session token

        Returns:
            UserSession if valid, None otherwise
        """
        session = self._sessions.get(session_token)

        if not session:
            return None

        # Check session timeout
        if self._is_session_expired(session):
            self.expire_session(session_token)
            return None

        # Update last activity
        session.last_activity = datetime.now(UTC)
        return session

    def expire_session(self, session_token: str) -> bool:
        """
        Expire user session.

        Args:
            session_token: Session token

        Returns:
            True if session expired
        """
        if session_token in self._sessions:
            session = self._sessions.pop(session_token)
            self._access_stats["sessions_expired"] += 1

            self.logger.info(
                "session_expired",
                user_id=session.user_id,
                session_token=session_token,
            )
            return True
        return False

    def _is_session_expired(self, session: UserSession) -> bool:
        """Check if session has expired"""
        age_minutes = (datetime.now(UTC) - session.last_activity).total_seconds() / 60
        return age_minutes > self.config.session_timeout_minutes

    def check_permission(
        self, session_token: str, permission: SecurityPermission
    ) -> bool:
        """
        Check if user has permission.

        Args:
            session_token: Session token
            permission: Required permission

        Returns:
            True if user has permission
        """
        if not self.config.enable_rbac:
            return True

        session = self.get_session(session_token)

        if not session:
            self._access_stats["access_denied"] += 1
            self.logger.warning("access_denied_no_session", session_token=session_token)
            return False

        has_permission = permission in session.permissions

        if has_permission:
            self._access_stats["access_granted"] += 1
        else:
            self._access_stats["access_denied"] += 1
            self.logger.warning(
                "access_denied_no_permission",
                user_id=session.user_id,
                permission=permission.value,
                role=session.role.value,
            )

        return has_permission

    def require_permission(
        self, session_token: str, permission: SecurityPermission
    ) -> None:
        """
        Require permission or raise error.

        Args:
            session_token: Session token
            permission: Required permission

        Raises:
            PermissionError: If user lacks permission
        """
        if not self.check_permission(session_token, permission):
            raise PermissionError(
                f"Permission denied: {permission.value} is required"
            )

    def get_user_permissions(self, session_token: str) -> set[SecurityPermission]:
        """
        Get user permissions.

        Args:
            session_token: Session token

        Returns:
            Set of user permissions
        """
        session = self.get_session(session_token)
        return session.permissions if session else set()

    def assign_role(self, session_token: str, new_role: SecurityRole) -> bool:
        """
        Assign new role to user.

        Args:
            session_token: Session token
            new_role: New role to assign

        Returns:
            True if role assigned successfully
        """
        session = self._sessions.get(session_token)

        if not session:
            return False

        session.role = new_role
        session.permissions = self.ROLE_PERMISSIONS[new_role]

        self.logger.info(
            "role_assigned",
            user_id=session.user_id,
            new_role=new_role.value,
        )

        return True

    def grant_permission(
        self, session_token: str, permission: SecurityPermission
    ) -> bool:
        """
        Grant additional permission to user.

        Args:
            session_token: Session token
            permission: Permission to grant

        Returns:
            True if permission granted
        """
        session = self._sessions.get(session_token)

        if not session:
            return False

        session.permissions.add(permission)

        self.logger.info(
            "permission_granted",
            user_id=session.user_id,
            permission=permission.value,
        )

        return True

    def revoke_permission(
        self, session_token: str, permission: SecurityPermission
    ) -> bool:
        """
        Revoke permission from user.

        Args:
            session_token: Session token
            permission: Permission to revoke

        Returns:
            True if permission revoked
        """
        session = self._sessions.get(session_token)

        if not session:
            return False

        session.permissions.discard(permission)

        self.logger.info(
            "permission_revoked",
            user_id=session.user_id,
            permission=permission.value,
        )

        return True

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired_tokens = [
            token
            for token, session in self._sessions.items()
            if self._is_session_expired(session)
        ]

        for token in expired_tokens:
            self.expire_session(token)

        if expired_tokens:
            self.logger.info("expired_sessions_cleaned", count=len(expired_tokens))

        return len(expired_tokens)

    def get_active_sessions(self) -> list[UserSession]:
        """Get all active sessions"""
        return list(self._sessions.values())

    def get_access_stats(self) -> dict[str, Any]:
        """Get access control statistics"""
        return {
            **self._access_stats,
            "active_sessions": len(self._sessions),
            "access_grant_rate": (
                self._access_stats["access_granted"]
                / (self._access_stats["access_granted"] + self._access_stats["access_denied"])
                if (self._access_stats["access_granted"] + self._access_stats["access_denied"]) > 0
                else 0.0
            ),
        }
