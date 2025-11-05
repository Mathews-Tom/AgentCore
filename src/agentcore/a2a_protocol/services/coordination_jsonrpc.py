"""Coordination Service JSON-RPC Methods.

JSON-RPC 2.0 methods for coordination signal management and agent selection.
Exposes the Ripple Effect Protocol coordination service via A2A protocol.

Methods:
- coordination.signal: Register sensitivity signal from agent
- coordination.state: Get agent coordination state
- coordination.metrics: Get coordination metrics snapshot
- coordination.predict_overload: Get overload prediction for agent
"""

from typing import Any

import structlog
from pydantic import ValidationError

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.coordination_service import coordination_service
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

logger = structlog.get_logger()


@register_jsonrpc_method("coordination.signal")
async def handle_coordination_signal(request: JsonRpcRequest) -> dict[str, Any]:
    """Register a sensitivity signal from an agent.

    Method: coordination.signal
    Params:
        - agent_id: Agent identifier (string)
        - signal_type: Signal type (LOAD, CAPACITY, QUALITY, COST, LATENCY, AVAILABILITY)
        - value: Signal value 0.0-1.0 (float)
        - ttl_seconds: Time-to-live in seconds (int, optional, default 60)
        - confidence: Signal confidence 0.0-1.0 (float, optional, default 1.0)
        - trace_id: Distributed tracing ID (string, optional)

    Returns:
        - signal_id: UUID of registered signal
        - agent_id: Agent identifier
        - status: "registered"
        - routing_score: Current routing score for agent
        - message: Success message

    Raises:
        ValueError: Invalid parameters or signal validation failure
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: agent_id, signal_type, value, and optional ttl_seconds, confidence, trace_id"
            )

        # Extract parameters
        agent_id = request.params.get("agent_id")
        signal_type_str = request.params.get("signal_type")
        value = request.params.get("value")
        ttl_seconds = request.params.get("ttl_seconds", 60)
        confidence = request.params.get("confidence", 1.0)
        trace_id = request.params.get("trace_id")

        # Validate required parameters
        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")
        if not signal_type_str:
            raise ValueError("Missing required parameter: signal_type")
        if value is None:
            raise ValueError("Missing required parameter: value")

        # Parse signal type
        try:
            signal_type = SignalType(signal_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid signal_type: {signal_type_str}. "
                f"Must be one of: {', '.join([st.value for st in SignalType])}"
            )

        # Create and validate signal
        signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=signal_type,
            value=value,
            ttl_seconds=ttl_seconds,
            confidence=confidence,
            trace_id=trace_id or request.a2a_context.trace_id if request.a2a_context else None,
        )

        # Register signal
        coordination_service.register_signal(signal)

        # Get updated routing score
        routing_score = coordination_service.compute_routing_score(agent_id)

        logger.info(
            "Coordination signal registered via JSON-RPC",
            agent_id=agent_id,
            signal_type=signal_type.value,
            value=value,
            routing_score=routing_score,
            trace_id=signal.trace_id,
            method="coordination.signal",
        )

        return {
            "signal_id": str(signal.signal_id),
            "agent_id": agent_id,
            "status": "registered",
            "routing_score": routing_score,
            "message": f"Signal {signal_type.value} registered successfully",
        }

    except ValidationError as e:
        logger.error("Signal validation failed", error=str(e))
        raise ValueError(f"Signal validation failed: {e}")
    except Exception as e:
        logger.error("Signal registration failed", error=str(e))
        raise


@register_jsonrpc_method("coordination.state")
async def handle_coordination_state(request: JsonRpcRequest) -> dict[str, Any]:
    """Get coordination state for an agent.

    Method: coordination.state
    Params:
        - agent_id: Agent identifier (string)

    Returns:
        - agent_id: Agent identifier
        - signals: Active signals (dict)
        - load_score: Load score 0.0-1.0
        - capacity_score: Capacity score 0.0-1.0
        - quality_score: Quality score 0.0-1.0
        - cost_score: Cost score 0.0-1.0
        - availability_score: Availability score 0.0-1.0
        - routing_score: Composite routing score 0.0-1.0
        - last_updated: Last update timestamp

    Raises:
        ValueError: Invalid parameters or agent not found
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id")

        agent_id = request.params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")

        # Get coordination state
        state = coordination_service.get_coordination_state(agent_id)

        if not state:
            return {
                "agent_id": agent_id,
                "status": "not_found",
                "message": "No coordination state found for agent",
            }

        # Convert signals to serializable format
        signals_data = {}
        for signal_type, signal in state.signals.items():
            signals_data[signal_type.value] = {
                "signal_id": str(signal.signal_id),
                "value": signal.value,
                "timestamp": signal.timestamp.isoformat(),
                "ttl_seconds": signal.ttl_seconds,
                "confidence": signal.confidence,
                "is_expired": signal.is_expired(),
                "decay_factor": signal.decay_factor(),
            }

        logger.debug(
            "Coordination state retrieved via JSON-RPC",
            agent_id=agent_id,
            routing_score=state.routing_score,
            method="coordination.state",
        )

        return {
            "agent_id": state.agent_id,
            "signals": signals_data,
            "load_score": state.load_score,
            "capacity_score": state.capacity_score,
            "quality_score": state.quality_score,
            "cost_score": state.cost_score,
            "availability_score": state.availability_score,
            "routing_score": state.routing_score,
            "last_updated": state.last_updated.isoformat(),
        }

    except Exception as e:
        logger.error("State retrieval failed", error=str(e))
        raise


@register_jsonrpc_method("coordination.metrics")
async def handle_coordination_metrics(request: JsonRpcRequest) -> dict[str, Any]:
    """Get coordination service metrics snapshot.

    Method: coordination.metrics
    Params: None

    Returns:
        - total_signals: Total signals registered (int)
        - signals_by_type: Signal counts by type (dict)
        - total_selections: Total agent selections (int)
        - average_selection_time_ms: Average selection latency (float)
        - average_signal_age_seconds: Average signal age (float)
        - coordination_score_avg: Average routing score (float)
        - expired_signals_cleaned: Expired signals cleaned (int)
        - agents_tracked: Number of agents tracked (int)

    Raises:
        Exception: Metrics retrieval failure
    """
    try:
        metrics = coordination_service.metrics

        # Convert signal types to strings for JSON serialization
        signals_by_type_str = {}
        for signal_type, count in metrics.signals_by_type.items():
            signals_by_type_str[signal_type.value] = count

        logger.debug(
            "Coordination metrics retrieved via JSON-RPC",
            total_signals=metrics.total_signals,
            agents_tracked=metrics.agents_tracked,
            method="coordination.metrics",
        )

        return {
            "total_signals": metrics.total_signals,
            "signals_by_type": signals_by_type_str,
            "total_selections": metrics.total_selections,
            "average_selection_time_ms": metrics.average_selection_time_ms,
            "average_signal_age_seconds": metrics.average_signal_age_seconds,
            "coordination_score_avg": metrics.coordination_score_avg,
            "expired_signals_cleaned": metrics.expired_signals_cleaned,
            "agents_tracked": metrics.agents_tracked,
        }

    except Exception as e:
        logger.error("Metrics retrieval failed", error=str(e))
        raise


@register_jsonrpc_method("coordination.predict_overload")
async def handle_coordination_predict_overload(request: JsonRpcRequest) -> dict[str, Any]:
    """Predict if agent will overload within forecast window.

    Method: coordination.predict_overload
    Params:
        - agent_id: Agent identifier (string)
        - forecast_seconds: Forecast window in seconds (int, optional, default 60)
        - threshold: Overload threshold 0.0-1.0 (float, optional, default 0.8)

    Returns:
        - agent_id: Agent identifier
        - will_overload: True if overload predicted (bool)
        - probability: Predicted load probability 0.0-1.0 (float)
        - forecast_seconds: Forecast window used (int)
        - threshold: Threshold used (float)
        - message: Human-readable prediction message

    Raises:
        ValueError: Invalid parameters
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: agent_id, and optional forecast_seconds, threshold"
            )

        agent_id = request.params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")

        forecast_seconds = request.params.get("forecast_seconds", 60)
        threshold = request.params.get("threshold", 0.8)

        # Validate numeric parameters
        if not isinstance(forecast_seconds, (int, float)) or forecast_seconds <= 0:
            raise ValueError("forecast_seconds must be a positive number")

        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")

        # Make prediction
        will_overload, probability = coordination_service.predict_overload(
            agent_id, forecast_seconds=int(forecast_seconds), threshold=float(threshold)
        )

        message = (
            f"Agent will {'likely' if will_overload else 'not'} overload within {forecast_seconds}s "
            f"(predicted load: {probability:.2f}, threshold: {threshold})"
        )

        logger.info(
            "Overload prediction via JSON-RPC",
            agent_id=agent_id,
            will_overload=will_overload,
            probability=probability,
            forecast_seconds=forecast_seconds,
            threshold=threshold,
            method="coordination.predict_overload",
        )

        return {
            "agent_id": agent_id,
            "will_overload": will_overload,
            "probability": probability,
            "forecast_seconds": forecast_seconds,
            "threshold": threshold,
            "message": message,
        }

    except Exception as e:
        logger.error("Overload prediction failed", error=str(e))
        raise
