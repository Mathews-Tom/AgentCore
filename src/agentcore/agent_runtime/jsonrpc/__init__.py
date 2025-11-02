"""JSON-RPC integration for agent runtime."""

# JSON-RPC methods are auto-registered via @register_jsonrpc_method decorator
# Import the module to trigger registration
from . import tools_jsonrpc  # noqa: F401

__all__ = ["tools_jsonrpc"]
