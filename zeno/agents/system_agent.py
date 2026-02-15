"""
ZENO System Agent - Safe System Execution

Converts PlannerAgent tasks into real OS actions.

Phase 5 Updates:
- Volume control handler (set, adjust, mute, unmute)
- Brightness control handler (set, adjust)
- User-facing error metadata for TTS

Rules (STRICT):
- Executes ONLY what PlannerAgent explicitly requests
- Performs ONE action per task
- Fails safely with clear errors
- NO LLM usage
- NO intent inference
- NO action chaining
- NO automatic retries

This agent IS executable and inherits from Agent base class.
"""

import logging
import threading
from typing import Optional

from zeno.core import Agent, Task, TaskResult, ContextSnapshot
from zeno.tools import app_control, file_system

logger = logging.getLogger(__name__)


class SystemAgentError(Exception):
    """Base exception for system agent errors"""
    pass


class SystemAgent(Agent):
    """
    System execution agent for OS-level actions.
    
    Inherits from Agent base class and implements execute() contract.
    
    Supported actions:
    - open_app: Launch applications
    - open_url: Open URLs in browser
    - create_file: Create files in workspace
    - create_directory: Create directories in workspace
    - volume_control: Control system volume (Phase 5)
    - brightness_control: Control screen brightness (Phase 5)
    """
    
    def __init__(self):
        """Initialize system agent"""
        logger.info("SystemAgent initialized")
    
    def execute(
        self,
        task: Task,
        context_snapshot: ContextSnapshot,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Execute a system task.
        
        Args:
            task: Task with action-specific payload
            context_snapshot: Read-only context (mostly unused by SystemAgent)
            interrupt_event: Signal for cooperative cancellation
            
        Returns:
            TaskResult with success/failure and details
            
        Expected task.payload format:
            {
                "action": "open_app" | "open_url" | "create_file" | "create_directory" | "volume_control" | "brightness_control",
                ... action-specific fields ...
            }
        """
        logger.info(f"SystemAgent executing task: {task.id} - {task.name}")
        
        # Check for interruption before starting
        if interrupt_event.is_set():
            logger.warning("Task interrupted before execution")
            return TaskResult(
                success=False,
                error="Task was interrupted"
            )
        
        # Validate payload
        if "action" not in task.payload:
            error_msg = "Task payload missing 'action' field"
            logger.error(error_msg)
            return TaskResult(
                success=False,
                error=error_msg
            )
        
        action = task.payload["action"]
        
        try:
            # Route to appropriate handler
            if action == "open_app":
                return self._handle_open_app(task, interrupt_event)
            elif action == "open_url":
                return self._handle_open_url(task, interrupt_event)
            elif action == "create_file":
                return self._handle_create_file(task, interrupt_event)
            elif action == "create_directory":
                return self._handle_create_directory(task, interrupt_event)
            elif action == "volume_control":
                return self._handle_volume_control(task, interrupt_event)
            elif action == "brightness_control":
                return self._handle_brightness_control(task, interrupt_event)
            else:
                error_msg = f"Unknown action: {action}"
                logger.error(error_msg)
                return TaskResult(
                    success=False,
                    error=error_msg
                )
                
        except app_control.AppNotFoundError as e:
            logger.error(f"App not found: {e}")
            return TaskResult(
                success=False,
                error=str(e)
            )
        except file_system.FileExistsError as e:
            logger.error(f"File exists: {e}")
            return TaskResult(
                success=False,
                error=str(e)
            )
        except file_system.InvalidExtensionError as e:
            logger.error(f"Invalid extension: {e}")
            return TaskResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"System action failed: {e}", exc_info=True)
            return TaskResult(
                success=False,
                error=f"System action failed: {e}"
            )
    
    def _handle_open_app(
        self,
        task: Task,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle opening an application.
        
        Expected payload:
            {
                "action": "open_app",
                "app": "vscode"
            }
        """
        if "app" not in task.payload:
            return TaskResult(
                success=False,
                error="Missing 'app' field in payload"
            )
        
        app_name = task.payload["app"]
        
        # Check interruption before execution
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            logger.info(f"Opening application: {app_name}")
            app_control.open_app(app_name)
            
            return TaskResult(
                success=True,
                data={
                    "action": "open_app",
                    "app": app_name,
                    "message": f"Opened {app_name}"
                }
            )
            
        except app_control.AppNotFoundError:
            # Re-raise to be caught by outer handler
            raise
        except Exception as e:
            logger.error(f"Failed to open app {app_name}: {e}")
            raise
    
    def _handle_open_url(
        self,
        task: Task,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle opening a URL.
        
        Expected payload:
            {
                "action": "open_url",
                "url": "https://github.com"
            }
        """
        if "url" not in task.payload:
            return TaskResult(
                success=False,
                error="Missing 'url' field in payload"
            )
        
        url = task.payload["url"]
        
        # Check interruption
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            logger.info(f"Opening URL: {url}")
            app_control.open_url(url)
            
            return TaskResult(
                success=True,
                data={
                    "action": "open_url",
                    "url": url,
                    "message": f"Opened {url}"
                }
            )
            
        except ValueError as e:
            # Invalid URL
            return TaskResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Failed to open URL {url}: {e}")
            raise
    
    def _handle_create_file(
        self,
        task: Task,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle creating a file.
        
        Expected payload:
            {
                "action": "create_file",
                "filename": "script.py",
                "content": "print('hello')",
                "subdirectory": "optional"  # optional
            }
        """
        if "filename" not in task.payload:
            return TaskResult(
                success=False,
                error="Missing 'filename' field in payload"
            )
        
        if "content" not in task.payload:
            return TaskResult(
                success=False,
                error="Missing 'content' field in payload"
            )
        
        filename = task.payload["filename"]
        content = task.payload["content"]
        subdirectory = task.payload.get("subdirectory")
        
        # Check interruption
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            logger.info(f"Creating file: {filename}")
            file_path = file_system.create_file(
                filename=filename,
                content=content,
                subdirectory=subdirectory,
                overwrite=False  # Never overwrite by default
            )
            
            return TaskResult(
                success=True,
                data={
                    "action": "create_file",
                    "filename": filename,
                    "path": str(file_path),
                    "message": f"Created {file_path}"
                }
            )
            
        except file_system.FileExistsError:
            # Re-raise to be caught by outer handler
            raise
        except file_system.InvalidExtensionError:
            # Re-raise to be caught by outer handler
            raise
        except Exception as e:
            logger.error(f"Failed to create file {filename}: {e}")
            raise
    
    def _handle_create_directory(
        self,
        task: Task,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle creating a directory.
        
        Expected payload:
            {
                "action": "create_directory",
                "dirname": "my_project",
                "parent": "optional"  # optional parent directory
            }
        """
        if "dirname" not in task.payload:
            return TaskResult(
                success=False,
                error="Missing 'dirname' field in payload"
            )
        
        dirname = task.payload["dirname"]
        parent = task.payload.get("parent")
        
        # Check interruption
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            logger.info(f"Creating directory: {dirname}")
            dir_path = file_system.create_directory(
                dirname=dirname,
                parent=parent
            )
            
            return TaskResult(
                success=True,
                data={
                    "action": "create_directory",
                    "dirname": dirname,
                    "path": str(dir_path),
                    "message": f"Created {dir_path}"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create directory {dirname}: {e}")
            raise
    
    def _handle_volume_control(
        self,
        task: Task,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle volume control (Phase 5).
        
        Expected payload:
            {
                "action": "volume_control",
                "operation": "set" | "increase" | "decrease" | "mute" | "unmute",
                "value": 50  # for "set" operation only
            }
        """
        operation = task.payload.get("operation")
        value = task.payload.get("value")
        
        if not operation:
            return TaskResult(
                success=False,
                error="Missing 'operation' field in payload"
            )
        
        # Check interruption
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            if operation == "set":
                if value is None:
                    return TaskResult(
                        success=False,
                        error="Missing 'value' field for set operation"
                    )
                app_control.set_volume(value)
                message = f"Set volume to {value} percent"
                
            elif operation == "increase":
                app_control.adjust_volume(+10)
                new_volume = app_control.get_volume()
                message = f"Increased volume to {new_volume} percent"
                
            elif operation == "decrease":
                app_control.adjust_volume(-10)
                new_volume = app_control.get_volume()
                message = f"Decreased volume to {new_volume} percent"
                
            elif operation == "mute":
                app_control.mute_volume()
                message = "Muted"
                
            elif operation == "unmute":
                app_control.unmute_volume()
                new_volume = app_control.get_volume()
                message = f"Unmuted to {new_volume} percent"
                
            else:
                return TaskResult(
                    success=False,
                    error=f"Unknown volume operation: {operation}"
                )
            
            return TaskResult(
                success=True,
                data={
                    "action": "volume_control",
                    "operation": operation,
                    "value": value,
                    "message": message
                }
            )
            
        except ValueError as e:
            # Invalid value range
            return TaskResult(
                success=False,
                error=str(e),
                metadata={"user_facing": True}
            )
        except Exception as e:
            logger.error(f"Volume control failed: {e}", exc_info=True)
            return TaskResult(
                success=False,
                error=f"Volume control failed: {e}"
            )
    
    def _handle_brightness_control(
        self,
        task: Task,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle brightness control (Phase 5).
        
        Expected payload:
            {
                "action": "brightness_control",
                "operation": "set" | "increase" | "decrease",
                "value": 80  # for "set" operation only
            }
        """
        operation = task.payload.get("operation")
        value = task.payload.get("value")
        
        if not operation:
            return TaskResult(
                success=False,
                error="Missing 'operation' field in payload"
            )
        
        # Check interruption
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            if operation == "set":
                if value is None:
                    return TaskResult(
                        success=False,
                        error="Missing 'value' field for set operation"
                    )
                app_control.set_brightness(value)
                message = f"Set brightness to {value} percent"
                
            elif operation == "increase":
                app_control.adjust_brightness(+10)
                new_brightness = app_control.get_brightness()
                message = f"Increased brightness to {new_brightness} percent"
                
            elif operation == "decrease":
                app_control.adjust_brightness(-10)
                new_brightness = app_control.get_brightness()
                message = f"Decreased brightness to {new_brightness} percent"
                
            else:
                return TaskResult(
                    success=False,
                    error=f"Unknown brightness operation: {operation}"
                )
            
            return TaskResult(
                success=True,
                data={
                    "action": "brightness_control",
                    "operation": operation,
                    "value": value,
                    "message": message
                }
            )
            
        except app_control.BrightnessNotAvailableError as e:
            # User-facing error - should be spoken
            logger.warning(f"Brightness control unavailable: {e}")
            return TaskResult(
                success=False,
                error=str(e),
                metadata={"user_facing": True}  # Signal to speak this error
            )
        except ValueError as e:
            # Invalid value range
            return TaskResult(
                success=False,
                error=str(e),
                metadata={"user_facing": True}
            )
        except Exception as e:
            # Internal error - log only
            logger.error(f"Brightness control failed: {e}", exc_info=True)
            return TaskResult(
                success=False,
                error=f"Brightness control failed: {e}"
            )