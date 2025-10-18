"""
Tests for OAuth scope management.
"""

from __future__ import annotations

import pytest

from gateway.auth.oauth.scopes import ScopeManager, ScopePermission


class TestScopeManager:
    """Test OAuth scope management."""

    def test_parse_scopes_valid(self) -> None:
        """Test parsing valid scope string."""
        scope_string = "profile:read email:read agent:read"
        scopes = ScopeManager.parse_scopes(scope_string)

        assert scopes == {"profile:read", "email:read", "agent:read"}

    def test_parse_scopes_empty(self) -> None:
        """Test parsing empty scope string."""
        assert ScopeManager.parse_scopes("") == set()
        assert ScopeManager.parse_scopes(None) == set()

    def test_parse_scopes_with_extra_spaces(self) -> None:
        """Test parsing scope string with extra spaces."""
        scope_string = "  profile:read   email:read  "
        scopes = ScopeManager.parse_scopes(scope_string)

        assert scopes == {"profile:read", "email:read"}

    def test_format_scopes(self) -> None:
        """Test formatting scopes as string."""
        scopes = {"agent:read", "profile:read", "email:read"}
        formatted = ScopeManager.format_scopes(scopes)

        # Should be space-separated and sorted
        assert formatted == "agent:read email:read profile:read"

    def test_format_scopes_from_list(self) -> None:
        """Test formatting scopes from list."""
        scopes = ["agent:read", "profile:read"]
        formatted = ScopeManager.format_scopes(scopes)

        assert "agent:read" in formatted
        assert "profile:read" in formatted

    def test_validate_scope_valid(self) -> None:
        """Test validating valid scope."""
        assert ScopeManager.validate_scope(ScopePermission.PROFILE_READ.value) is True
        assert ScopeManager.validate_scope(ScopePermission.AGENT_EXECUTE.value) is True

    def test_validate_scope_invalid(self) -> None:
        """Test validating invalid scope."""
        assert ScopeManager.validate_scope("invalid:scope") is False

    def test_validate_scopes_all_valid(self) -> None:
        """Test validating all valid scopes."""
        scopes = "profile:read email:read agent:read"
        is_valid, invalid = ScopeManager.validate_scopes(scopes)

        assert is_valid is True
        assert invalid == []

    def test_validate_scopes_some_invalid(self) -> None:
        """Test validating scopes with some invalid."""
        scopes = "profile:read invalid:scope agent:read"
        is_valid, invalid = ScopeManager.validate_scopes(scopes)

        assert is_valid is False
        assert "invalid:scope" in invalid

    def test_validate_scopes_from_set(self) -> None:
        """Test validating scopes from set."""
        scopes = {ScopePermission.PROFILE_READ.value, "invalid:scope"}
        is_valid, invalid = ScopeManager.validate_scopes(scopes)

        assert is_valid is False
        assert "invalid:scope" in invalid

    def test_expand_scopes_with_dependencies(self) -> None:
        """Test scope expansion with dependencies."""
        # profile:write requires profile:read
        scopes = {ScopePermission.PROFILE_WRITE.value}
        expanded = ScopeManager.expand_scopes(scopes)

        assert ScopePermission.PROFILE_WRITE.value in expanded
        assert ScopePermission.PROFILE_READ.value in expanded

    def test_expand_scopes_without_dependencies(self) -> None:
        """Test scope expansion without dependencies."""
        scopes = {ScopePermission.PROFILE_READ.value}
        expanded = ScopeManager.expand_scopes(scopes)

        assert expanded == scopes

    def test_expand_scopes_multiple_dependencies(self) -> None:
        """Test scope expansion with multiple scopes."""
        scopes = {
            ScopePermission.PROFILE_WRITE.value,
            ScopePermission.AGENT_WRITE.value,
        }
        expanded = ScopeManager.expand_scopes(scopes)

        assert ScopePermission.PROFILE_READ.value in expanded
        assert ScopePermission.AGENT_READ.value in expanded

    def test_filter_restricted_scopes_admin(self) -> None:
        """Test filtering restricted scopes for admin user."""
        scopes = {
            ScopePermission.PROFILE_READ.value,
            ScopePermission.ADMIN_USERS.value,  # Restricted
        }
        allowed, rejected = ScopeManager.filter_restricted_scopes(scopes, is_admin=True)

        assert ScopePermission.PROFILE_READ.value in allowed
        assert ScopePermission.ADMIN_USERS.value in allowed
        assert len(rejected) == 0

    def test_filter_restricted_scopes_non_admin(self) -> None:
        """Test filtering restricted scopes for non-admin user."""
        scopes = {
            ScopePermission.PROFILE_READ.value,
            ScopePermission.ADMIN_USERS.value,  # Restricted
        }
        allowed, rejected = ScopeManager.filter_restricted_scopes(scopes, is_admin=False)

        assert ScopePermission.PROFILE_READ.value in allowed
        assert ScopePermission.ADMIN_USERS.value not in allowed
        assert ScopePermission.ADMIN_USERS.value in rejected

    def test_check_permission_granted(self) -> None:
        """Test permission check with granted scope."""
        granted_scopes = "profile:read email:read agent:read"
        required_scope = ScopePermission.PROFILE_READ.value

        assert ScopeManager.check_permission(granted_scopes, required_scope) is True

    def test_check_permission_denied(self) -> None:
        """Test permission check with denied scope."""
        granted_scopes = "profile:read email:read"
        required_scope = ScopePermission.AGENT_WRITE.value

        assert ScopeManager.check_permission(granted_scopes, required_scope) is False

    def test_check_permission_with_dependency(self) -> None:
        """Test permission check with dependency expansion."""
        # Granted profile:write, which requires profile:read
        granted_scopes = {ScopePermission.PROFILE_WRITE.value}
        required_scope = ScopePermission.PROFILE_READ.value

        assert ScopeManager.check_permission(granted_scopes, required_scope) is True

    def test_check_any_permission_granted(self) -> None:
        """Test any permission check with at least one granted."""
        granted_scopes = "profile:read email:read"
        required_scopes = [
            ScopePermission.PROFILE_READ.value,
            ScopePermission.AGENT_WRITE.value,
        ]

        assert ScopeManager.check_any_permission(granted_scopes, required_scopes) is True

    def test_check_any_permission_denied(self) -> None:
        """Test any permission check with none granted."""
        granted_scopes = "profile:read"
        required_scopes = [
            ScopePermission.AGENT_WRITE.value,
            ScopePermission.TASK_DELETE.value,
        ]

        assert ScopeManager.check_any_permission(granted_scopes, required_scopes) is False

    def test_check_all_permissions_granted(self) -> None:
        """Test all permissions check with all granted."""
        granted_scopes = "profile:read email:read agent:read"
        required_scopes = [
            ScopePermission.PROFILE_READ.value,
            ScopePermission.EMAIL_READ.value,
        ]

        assert ScopeManager.check_all_permissions(granted_scopes, required_scopes) is True

    def test_check_all_permissions_partial(self) -> None:
        """Test all permissions check with partial grant."""
        granted_scopes = "profile:read"
        required_scopes = [
            ScopePermission.PROFILE_READ.value,
            ScopePermission.EMAIL_READ.value,
        ]

        assert ScopeManager.check_all_permissions(granted_scopes, required_scopes) is False

    def test_get_scope_info_valid(self) -> None:
        """Test getting scope information."""
        info = ScopeManager.get_scope_info(ScopePermission.PROFILE_READ.value)

        assert info is not None
        assert info["scope"] == ScopePermission.PROFILE_READ.value
        assert "description" in info
        assert "resource" in info

    def test_get_scope_info_invalid(self) -> None:
        """Test getting scope information for invalid scope."""
        info = ScopeManager.get_scope_info("invalid:scope")

        assert info is None

    def test_get_all_scopes(self) -> None:
        """Test getting all available scopes."""
        all_scopes = ScopeManager.get_all_scopes()

        assert isinstance(all_scopes, list)
        assert len(all_scopes) > 0
        assert all(s["scope"] for s in all_scopes)

    def test_get_scopes_for_resource(self) -> None:
        """Test getting scopes for specific resource."""
        agent_scopes = ScopeManager.get_scopes_for_resource("agent")

        assert ScopePermission.AGENT_READ.value in agent_scopes
        assert ScopePermission.AGENT_WRITE.value in agent_scopes
        assert ScopePermission.PROFILE_READ.value not in agent_scopes

    def test_get_scopes_for_resource_empty(self) -> None:
        """Test getting scopes for non-existent resource."""
        scopes = ScopeManager.get_scopes_for_resource("nonexistent")

        assert scopes == []

    def test_scope_permission_enum_values(self) -> None:
        """Test that all scope permission enum values are valid."""
        for permission in ScopePermission:
            assert ScopeManager.validate_scope(permission.value) is True
