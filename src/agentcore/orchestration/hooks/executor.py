"""
Hook Executor

Executes workflow hooks with timeout protection, retries, and error handling.
"""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import traceback
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from agentcore.orchestration.hooks.models import (
    HookConfig,
    HookEvent,
    HookExecution,
    HookExecutionMode,
    HookStatus,
)

logger = structlog.get_logger()


class HookExecutor:
    """
    Executes workflow hooks with timeout and error handling.

    Supports both shell commands and Python function references.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

    async def execute_hook(
        self, hook: HookConfig, event: HookEvent, is_retry: bool = False
    ) -> HookExecution:
        """
        Execute a hook with the given event data.

        Args:
            hook: Hook configuration
            event: Event that triggered the hook
            is_retry: Whether this is a retry attempt

        Returns:
            Hook execution record
        """
        # Create execution record
        execution = HookExecution(
            hook_id=hook.hook_id,
            trigger=event.trigger,
            status=HookStatus.RUNNING,
            started_at=datetime.now(UTC),
            input_data=event.data,
            is_retry=is_retry,
            workflow_id=event.workflow_id,
            task_id=event.task_id,
            session_id=event.session_id,
        )

        try:
            # Execute with timeout
            timeout_seconds = hook.timeout_ms / 1000
            output = await asyncio.wait_for(
                self._execute_hook_internal(hook, event), timeout=timeout_seconds
            )

            execution.mark_completed(output)
            self.logger.info(
                "Hook executed successfully",
                hook_id=str(hook.hook_id),
                hook_name=hook.name,
                trigger=hook.trigger.value,
                duration_ms=execution.duration_ms,
            )

        except asyncio.TimeoutError:
            execution.mark_timeout()
            self.logger.error(
                "Hook execution timed out",
                hook_id=str(hook.hook_id),
                hook_name=hook.name,
                timeout_ms=hook.timeout_ms,
            )

        except Exception as e:
            error_msg = str(e)
            error_tb = traceback.format_exc()
            execution.mark_failed(error_msg, error_tb)
            self.logger.error(
                "Hook execution failed",
                hook_id=str(hook.hook_id),
                hook_name=hook.name,
                error=error_msg,
            )

        return execution

    async def _execute_hook_internal(
        self, hook: HookConfig, event: HookEvent
    ) -> dict[str, Any]:
        """
        Internal hook execution logic.

        Args:
            hook: Hook configuration
            event: Event data

        Returns:
            Execution output
        """
        command = hook.command

        # Check if command is a Python function reference (module:function)
        if ":" in command and not command.startswith("/"):
            return await self._execute_python_function(hook, event)
        else:
            return await self._execute_shell_command(hook, event)

    async def _execute_python_function(
        self, hook: HookConfig, event: HookEvent
    ) -> dict[str, Any]:
        """
        Execute a Python function hook.

        Args:
            hook: Hook configuration
            event: Event data

        Returns:
            Function output
        """
        try:
            # Parse module and function name
            module_name, function_name = hook.command.split(":", 1)

            # Import module and get function
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)

            # Prepare arguments
            func_args = {
                "event": event,
                "hook_config": hook,
                "args": hook.args,
            }

            # Execute function (async or sync)
            if asyncio.iscoroutinefunction(func):
                result = await func(**func_args)
            else:
                # Run sync function in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: func(**func_args)
                )

            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"result": result}

            return result

        except Exception as e:
            self.logger.error(
                "Python function hook failed",
                hook_name=hook.name,
                command=hook.command,
                error=str(e),
            )
            raise

    async def _execute_shell_command(
        self, hook: HookConfig, event: HookEvent
    ) -> dict[str, Any]:
        """
        Execute a shell command hook.

        Args:
            hook: Hook configuration
            event: Event data

        Returns:
            Command output
        """
        try:
            # Build command with arguments
            cmd_parts = [hook.command] + hook.args

            # Run command
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Pass event data as JSON input
            import json

            input_data = json.dumps(event.model_dump(mode="json")).encode()

            stdout, stderr = await process.communicate(input=input_data)

            # Parse output
            output: dict[str, Any] = {
                "exit_code": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
            }

            # Try to parse stdout as JSON
            if output["stdout"]:
                try:
                    output["data"] = json.loads(output["stdout"])
                except json.JSONDecodeError:
                    # Not JSON, leave as string
                    pass

            # Check exit code
            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {output['stderr']}"
                )

            return output

        except Exception as e:
            self.logger.error(
                "Shell command hook failed",
                hook_name=hook.name,
                command=hook.command,
                error=str(e),
            )
            raise

    async def execute_with_retry(
        self, hook: HookConfig, event: HookEvent
    ) -> HookExecution:
        """
        Execute hook with retry logic.

        Args:
            hook: Hook configuration
            event: Event data

        Returns:
            Final execution record
        """
        execution = await self.execute_hook(hook, event, is_retry=False)

        # If successful or retries disabled, return
        if execution.status == HookStatus.COMPLETED or not hook.retry_enabled:
            return execution

        # Retry on failure
        retry_count = 0
        while retry_count < hook.max_retries and execution.status != HookStatus.COMPLETED:
            retry_count += 1

            # Exponential backoff
            delay_seconds = (hook.retry_delay_ms * (2 ** (retry_count - 1))) / 1000
            await asyncio.sleep(delay_seconds)

            self.logger.info(
                "Retrying hook execution",
                hook_id=str(hook.hook_id),
                hook_name=hook.name,
                retry_count=retry_count,
                max_retries=hook.max_retries,
            )

            execution = await self.execute_hook(hook, event, is_retry=True)
            execution.retry_count = retry_count

        return execution
