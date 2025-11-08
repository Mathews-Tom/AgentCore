"""
ACE Integration Layer

Runtime and MEM interfaces for COMPASS Meta-Thinker integration.
Handles bidirectional communication with Agent Runtime and strategic
context queries to MEM.
"""

from agentcore.ace.integration.mem_interface import ACEMemoryInterface
from agentcore.ace.integration.runtime_interface import RuntimeInterface

__all__ = ["RuntimeInterface", "ACEMemoryInterface"]
