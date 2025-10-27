"""Unit tests for budget enforcement."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentcore.training.utils.budget import (
    BudgetEnforcer,
    BudgetStatus,
    check_budget)


def test_budget_enforcer_initialization():
    """Test budget enforcer initialization."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    assert enforcer.max_budget_usd == Decimal("100.00")
    assert enforcer.current_cost_usd == Decimal("0.00")
    assert enforcer.warning_threshold_75 == Decimal("0.75")
    assert enforcer.warning_threshold_90 == Decimal("0.90")


def test_budget_enforcer_custom_thresholds():
    """Test budget enforcer with custom thresholds."""
    enforcer = BudgetEnforcer(
        max_budget_usd=Decimal("50.00"),
        warning_threshold_75=0.70,
        warning_threshold_90=0.85)

    assert enforcer.warning_threshold_75 == Decimal("0.70")
    assert enforcer.warning_threshold_90 == Decimal("0.85")


def test_check_budget_available_ok():
    """Test budget check when budget is available."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    # Check for small amount
    available, status = enforcer.check_budget_available(Decimal("10.00"))

    assert available is True
    assert status == BudgetStatus.OK


def test_check_budget_available_warning_75():
    """Test budget check at 75% threshold."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("50.00")

    # Adding 25 would bring to 75%
    available, status = enforcer.check_budget_available(Decimal("25.00"))

    assert available is True
    assert status == BudgetStatus.WARNING_75


def test_check_budget_available_warning_90():
    """Test budget check at 90% threshold."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("70.00")

    # Adding 20 would bring to 90%
    available, status = enforcer.check_budget_available(Decimal("20.00"))

    assert available is True
    assert status == BudgetStatus.WARNING_90


def test_check_budget_available_exceeded():
    """Test budget check when budget would be exceeded."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("90.00")

    # Adding 20 would exceed budget (110 > 100)
    available, status = enforcer.check_budget_available(Decimal("20.00"))

    assert available is False
    assert status == BudgetStatus.EXCEEDED


def test_check_budget_available_exact_limit():
    """Test budget check at exact budget limit."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("80.00")

    # Adding 20 would exactly hit budget (100 == 100)
    available, status = enforcer.check_budget_available(Decimal("20.00"))

    assert available is True
    # At 100% utilization, should trigger warning
    assert status == BudgetStatus.WARNING_90


def test_add_cost():
    """Test adding cost to budget."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    enforcer.add_cost(Decimal("25.50"))
    assert enforcer.current_cost_usd == Decimal("25.50")

    enforcer.add_cost(Decimal("10.25"))
    assert enforcer.current_cost_usd == Decimal("35.75")


def test_get_remaining_budget():
    """Test getting remaining budget."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("37.50")

    remaining = enforcer.get_remaining_budget()

    assert remaining == Decimal("62.50")


def test_get_remaining_budget_exceeded():
    """Test remaining budget when exceeded."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("120.00")

    remaining = enforcer.get_remaining_budget()

    assert remaining == Decimal("-20.00")


def test_get_utilization_percentage():
    """Test getting utilization percentage."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("75.00")

    utilization = enforcer.get_utilization_percentage()

    assert utilization == 75.0


def test_get_utilization_percentage_zero_budget():
    """Test utilization with zero budget."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("0.00"))
    enforcer.current_cost_usd = Decimal("10.00")

    utilization = enforcer.get_utilization_percentage()

    assert utilization == 0.0


def test_is_budget_exceeded():
    """Test budget exceeded check."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    # Not exceeded
    enforcer.current_cost_usd = Decimal("90.00")
    assert enforcer.is_budget_exceeded() is False

    # Exceeded
    enforcer.current_cost_usd = Decimal("110.00")
    assert enforcer.is_budget_exceeded() is True


def test_get_status_ok():
    """Test getting status when OK."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("50.00")

    status = enforcer.get_status()

    assert status["status"] == BudgetStatus.OK.value
    assert status["current_cost_usd"] == 50.0
    assert status["max_budget_usd"] == 100.0
    assert status["remaining_usd"] == 50.0
    assert status["utilization_percentage"] == 50.0
    assert status["is_exceeded"] is False


def test_get_status_warning_75():
    """Test getting status at 75% warning."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("75.00")

    status = enforcer.get_status()

    assert status["status"] == BudgetStatus.WARNING_75.value
    assert status["utilization_percentage"] == 75.0
    assert status["is_exceeded"] is False


def test_get_status_warning_90():
    """Test getting status at 90% warning."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("90.00")

    status = enforcer.get_status()

    assert status["status"] == BudgetStatus.WARNING_90.value
    assert status["utilization_percentage"] == 90.0
    assert status["is_exceeded"] is False


def test_get_status_exceeded():
    """Test getting status when exceeded."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("110.00")

    status = enforcer.get_status()

    assert status["status"] == BudgetStatus.EXCEEDED.value
    assert status["utilization_percentage"] == 110.0
    assert status["is_exceeded"] is True
    assert status["remaining_usd"] == -10.0


def test_reset():
    """Test resetting budget enforcer."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))
    enforcer.current_cost_usd = Decimal("75.00")

    enforcer.reset()

    assert enforcer.current_cost_usd == Decimal("0.00")
    assert enforcer.max_budget_usd == Decimal("100.00")  # Budget unchanged


def test_check_budget_function_ok():
    """Test standalone check_budget function when OK."""
    is_ok, message = check_budget(
        current_cost=Decimal("50.00"),
        max_budget=Decimal("100.00"),
        required_amount=Decimal("25.00"))

    assert is_ok is True
    assert "Budget OK" in message
    assert "25.00" in message


def test_check_budget_function_exceeded():
    """Test standalone check_budget function when exceeded."""
    is_ok, message = check_budget(
        current_cost=Decimal("80.00"),
        max_budget=Decimal("100.00"),
        required_amount=Decimal("30.00"))

    assert is_ok is False
    assert "Budget would be exceeded" in message
    assert "110.00" in message  # Projected cost


def test_check_budget_function_exact_limit():
    """Test standalone check_budget at exact limit."""
    is_ok, message = check_budget(
        current_cost=Decimal("75.00"),
        max_budget=Decimal("100.00"),
        required_amount=Decimal("25.00"))

    assert is_ok is True


def test_check_budget_function_no_additional_cost():
    """Test standalone check_budget with no additional cost."""
    is_ok, message = check_budget(
        current_cost=Decimal("90.00"),
        max_budget=Decimal("100.00"),
        required_amount=Decimal("0.00"))

    assert is_ok is True
    assert "10.00" in message  # Remaining


def test_progressive_cost_addition():
    """Test progressive cost additions."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    # Add costs progressively
    costs = [Decimal("20.00"), Decimal("30.00"), Decimal("25.00")]

    for cost in costs:
        available, _ = enforcer.check_budget_available(cost)
        assert available is True
        enforcer.add_cost(cost)

    # Total: 75.00
    assert enforcer.current_cost_usd == Decimal("75.00")

    # This would exceed
    available, status = enforcer.check_budget_available(Decimal("30.00"))
    assert available is False
    assert status == BudgetStatus.EXCEEDED


def test_threshold_boundaries():
    """Test exact threshold boundaries."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    # Just below 75%
    enforcer.current_cost_usd = Decimal("74.99")
    status = enforcer.get_status()
    assert status["status"] == BudgetStatus.OK.value

    # Exactly at 75%
    enforcer.current_cost_usd = Decimal("75.00")
    status = enforcer.get_status()
    assert status["status"] == BudgetStatus.WARNING_75.value

    # Just below 90%
    enforcer.current_cost_usd = Decimal("89.99")
    status = enforcer.get_status()
    assert status["status"] == BudgetStatus.WARNING_75.value

    # Exactly at 90%
    enforcer.current_cost_usd = Decimal("90.00")
    status = enforcer.get_status()
    assert status["status"] == BudgetStatus.WARNING_90.value

    # Just over 100%
    enforcer.current_cost_usd = Decimal("100.01")
    status = enforcer.get_status()
    assert status["status"] == BudgetStatus.EXCEEDED.value


def test_decimal_precision():
    """Test decimal precision handling."""
    enforcer = BudgetEnforcer(max_budget_usd=Decimal("100.00"))

    # Add costs with high precision
    enforcer.add_cost(Decimal("33.333333"))
    enforcer.add_cost(Decimal("33.333333"))
    enforcer.add_cost(Decimal("33.333334"))

    # Should be exactly 100.00
    assert enforcer.current_cost_usd == Decimal("100.000000")

    # Should not exceed
    assert enforcer.is_budget_exceeded() is False
