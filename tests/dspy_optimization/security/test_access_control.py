"""
Tests for access control
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.dspy_optimization.security.access_control import (
    AccessConfig,
    AccessController,
    SecurityPermission,
    SecurityRole,
    UserSession,
)


class TestAccessConfig:
    """Tests for AccessConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = AccessConfig()
        assert config.enable_rbac is True
        assert config.default_role == SecurityRole.ANALYST
        assert config.session_timeout_minutes == 60
        assert config.max_failed_attempts == 5


class TestAccessController:
    """Tests for AccessController"""

    @pytest.fixture
    def controller(self) -> AccessController:
        """Create access controller"""
        return AccessController()

    def test_initialization(self, controller: AccessController):
        """Test controller initialization"""
        assert controller.config.enable_rbac is True
        assert len(controller._sessions) == 0

    def test_create_session(self, controller: AccessController):
        """Test session creation"""
        session = controller.create_session("user1", SecurityRole.OPTIMIZER)

        assert session.user_id == "user1"
        assert session.role == SecurityRole.OPTIMIZER
        assert len(session.permissions) > 0
        assert session.session_token is not None

    def test_create_session_default_role(self, controller: AccessController):
        """Test session creation with default role"""
        session = controller.create_session("user1")

        assert session.role == SecurityRole.ANALYST

    def test_get_session_valid(self, controller: AccessController):
        """Test getting valid session"""
        created = controller.create_session("user1", SecurityRole.ADMIN)
        retrieved = controller.get_session(created.session_token)

        assert retrieved is not None
        assert retrieved.user_id == created.user_id
        assert retrieved.role == created.role

    def test_get_session_invalid(self, controller: AccessController):
        """Test getting invalid session"""
        session = controller.get_session("invalid_token")
        assert session is None

    def test_session_timeout(self, controller: AccessController):
        """Test session timeout"""
        config = AccessConfig(session_timeout_minutes=0)
        controller = AccessController(config)

        session = controller.create_session("user1")

        # Session should expire immediately
        import time

        time.sleep(0.1)
        retrieved = controller.get_session(session.session_token)
        assert retrieved is None

    def test_expire_session(self, controller: AccessController):
        """Test explicit session expiration"""
        session = controller.create_session("user1")
        assert controller.expire_session(session.session_token) is True

        # Session should be gone
        retrieved = controller.get_session(session.session_token)
        assert retrieved is None

    def test_check_permission_admin(self, controller: AccessController):
        """Test admin has all permissions"""
        session = controller.create_session("admin_user", SecurityRole.ADMIN)

        # Admin should have all permissions
        for permission in SecurityPermission:
            assert controller.check_permission(session.session_token, permission) is True

    def test_check_permission_optimizer(self, controller: AccessController):
        """Test optimizer permissions"""
        session = controller.create_session("opt_user", SecurityRole.OPTIMIZER)

        # Should have optimization permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.OPTIMIZATION_CREATE
            )
            is True
        )
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is True
        )

        # Should not have security permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.SECURITY_MANAGE
            )
            is False
        )

    def test_check_permission_analyst(self, controller: AccessController):
        """Test analyst permissions (read-only)"""
        session = controller.create_session("analyst_user", SecurityRole.ANALYST)

        # Should have read permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_READ
            )
            is True
        )
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.OPTIMIZATION_READ
            )
            is True
        )

        # Should not have write permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is False
        )
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.OPTIMIZATION_CREATE
            )
            is False
        )

    def test_check_permission_auditor(self, controller: AccessController):
        """Test auditor permissions"""
        session = controller.create_session("auditor_user", SecurityRole.AUDITOR)

        # Should have audit read permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.AUDIT_READ
            )
            is True
        )

        # Should not have audit write permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.AUDIT_WRITE
            )
            is False
        )

    def test_check_permission_invalid_session(self, controller: AccessController):
        """Test permission check with invalid session"""
        assert (
            controller.check_permission("invalid", SecurityPermission.MODEL_READ)
            is False
        )

    def test_require_permission_success(self, controller: AccessController):
        """Test require permission succeeds"""
        session = controller.create_session("user1", SecurityRole.ADMIN)

        # Should not raise
        controller.require_permission(
            session.session_token, SecurityPermission.MODEL_READ
        )

    def test_require_permission_failure(self, controller: AccessController):
        """Test require permission raises error"""
        session = controller.create_session("user1", SecurityRole.ANALYST)

        with pytest.raises(PermissionError, match="Permission denied"):
            controller.require_permission(
                session.session_token, SecurityPermission.MODEL_DELETE
            )

    def test_get_user_permissions(self, controller: AccessController):
        """Test getting user permissions"""
        session = controller.create_session("user1", SecurityRole.OPTIMIZER)
        permissions = controller.get_user_permissions(session.session_token)

        assert isinstance(permissions, set)
        assert SecurityPermission.OPTIMIZATION_CREATE in permissions
        assert SecurityPermission.MODEL_WRITE in permissions

    def test_assign_role(self, controller: AccessController):
        """Test role assignment"""
        session = controller.create_session("user1", SecurityRole.ANALYST)

        # Initially analyst
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is False
        )

        # Assign optimizer role
        controller.assign_role(session.session_token, SecurityRole.OPTIMIZER)

        # Now has optimizer permissions
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is True
        )

    def test_grant_permission(self, controller: AccessController):
        """Test granting additional permission"""
        session = controller.create_session("user1", SecurityRole.ANALYST)

        # Initially no write permission
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is False
        )

        # Grant write permission
        controller.grant_permission(
            session.session_token, SecurityPermission.MODEL_WRITE
        )

        # Now has write permission
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is True
        )

    def test_revoke_permission(self, controller: AccessController):
        """Test revoking permission"""
        session = controller.create_session("user1", SecurityRole.OPTIMIZER)

        # Initially has write permission
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is True
        )

        # Revoke write permission
        controller.revoke_permission(
            session.session_token, SecurityPermission.MODEL_WRITE
        )

        # No longer has write permission
        assert (
            controller.check_permission(
                session.session_token, SecurityPermission.MODEL_WRITE
            )
            is False
        )

    def test_cleanup_expired_sessions(self, controller: AccessController):
        """Test cleaning up expired sessions"""
        config = AccessConfig(session_timeout_minutes=0)
        controller = AccessController(config)

        # Create sessions
        session1 = controller.create_session("user1")
        session2 = controller.create_session("user2")

        # Wait for expiration
        import time

        time.sleep(0.1)

        # Cleanup
        count = controller.cleanup_expired_sessions()

        assert count == 2
        assert len(controller._sessions) == 0

    def test_get_active_sessions(self, controller: AccessController):
        """Test getting active sessions"""
        controller.create_session("user1", SecurityRole.ADMIN)
        controller.create_session("user2", SecurityRole.OPTIMIZER)

        sessions = controller.get_active_sessions()

        assert len(sessions) == 2
        assert all(isinstance(s, UserSession) for s in sessions)

    def test_get_access_stats(self, controller: AccessController):
        """Test getting access statistics"""
        session = controller.create_session("user1", SecurityRole.OPTIMIZER)

        controller.check_permission(
            session.session_token, SecurityPermission.MODEL_READ
        )
        controller.check_permission(
            session.session_token, SecurityPermission.SECURITY_MANAGE
        )

        stats = controller.get_access_stats()

        assert stats["sessions_created"] >= 1
        assert stats["access_granted"] >= 1
        assert stats["access_denied"] >= 1
        assert "active_sessions" in stats
        assert "access_grant_rate" in stats

    def test_rbac_disabled(self):
        """Test with RBAC disabled"""
        config = AccessConfig(enable_rbac=False)
        controller = AccessController(config)

        # Should allow all permissions
        assert (
            controller.check_permission("any_token", SecurityPermission.MODEL_DELETE)
            is True
        )


class TestRolePermissions:
    """Tests for role-permission mappings"""

    def test_admin_has_all_permissions(self):
        """Test admin role has all permissions"""
        admin_perms = AccessController.ROLE_PERMISSIONS[SecurityRole.ADMIN]
        all_perms = set(SecurityPermission)

        assert admin_perms == all_perms

    def test_optimizer_has_subset(self):
        """Test optimizer has subset of permissions"""
        optimizer_perms = AccessController.ROLE_PERMISSIONS[SecurityRole.OPTIMIZER]
        admin_perms = AccessController.ROLE_PERMISSIONS[SecurityRole.ADMIN]

        assert optimizer_perms.issubset(admin_perms)
        assert SecurityPermission.OPTIMIZATION_CREATE in optimizer_perms
        assert SecurityPermission.SECURITY_MANAGE not in optimizer_perms

    def test_analyst_read_only(self):
        """Test analyst has only read permissions"""
        analyst_perms = AccessController.ROLE_PERMISSIONS[SecurityRole.ANALYST]

        # Should have read permissions
        assert SecurityPermission.MODEL_READ in analyst_perms
        assert SecurityPermission.OPTIMIZATION_READ in analyst_perms

        # Should not have write/delete permissions
        assert SecurityPermission.MODEL_WRITE not in analyst_perms
        assert SecurityPermission.MODEL_DELETE not in analyst_perms

    def test_auditor_has_audit_permissions(self):
        """Test auditor has audit permissions"""
        auditor_perms = AccessController.ROLE_PERMISSIONS[SecurityRole.AUDITOR]

        assert SecurityPermission.AUDIT_READ in auditor_perms
        assert SecurityPermission.AUDIT_WRITE not in auditor_perms


class TestAccessControlIntegration:
    """Integration tests for access control"""

    def test_session_lifecycle(self):
        """Test complete session lifecycle"""
        controller = AccessController()

        # Create session
        session = controller.create_session("user1", SecurityRole.OPTIMIZER)
        assert len(controller._sessions) == 1

        # Use session
        assert controller.check_permission(
            session.session_token, SecurityPermission.OPTIMIZATION_CREATE
        ) is True

        # Expire session
        controller.expire_session(session.session_token)
        assert len(controller._sessions) == 0

        # Session no longer valid
        assert controller.get_session(session.session_token) is None

    def test_multiple_users_different_roles(self):
        """Test multiple users with different roles"""
        controller = AccessController()

        admin_session = controller.create_session("admin", SecurityRole.ADMIN)
        opt_session = controller.create_session("optimizer", SecurityRole.OPTIMIZER)
        analyst_session = controller.create_session("analyst", SecurityRole.ANALYST)

        # Admin can do everything
        assert controller.check_permission(
            admin_session.session_token, SecurityPermission.MODEL_DELETE
        ) is True

        # Optimizer can write
        assert controller.check_permission(
            opt_session.session_token, SecurityPermission.MODEL_WRITE
        ) is True

        # Analyst can only read
        assert controller.check_permission(
            analyst_session.session_token, SecurityPermission.MODEL_READ
        ) is True
        assert controller.check_permission(
            analyst_session.session_token, SecurityPermission.MODEL_WRITE
        ) is False
