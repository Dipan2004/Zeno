"""
ZENO Context Manager - In-Process Short-Term Memory

Responsibilities:
- Manage active session state
- Track conversation history (text-based messages)
- Maintain current plan/task context
- Store execution metadata for debugging
- Provide thread-safe access to shared state
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from zeno.core.orchestrator import ContextSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Structured conversation message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message"""
        if self.role not in ('user', 'assistant', 'system'):
            raise ValueError(f"Invalid role: {self.role}")
        if not isinstance(self.content, str):
            raise TypeError("Content must be a string")


@dataclass
class ExecutionRecord:
    """Record of task execution event"""
    task_id: str
    event_type: str  # 'started', 'completed', 'failed', 'cancelled'
    timestamp: float
    agent_type: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """
    Thread-safe in-process memory for ZENO's active session.
    
    Manages:
    - Conversation history with automatic pruning
    - Current task/plan context
    - Execution metadata and debugging info
    
    Thread-safety:
    - Uses RLock for nested read operations
    - Read-heavy workload optimized
    """
    
    MAX_MESSAGES = 100
    MAX_EXECUTION_RECORDS = 500
    
    def __init__(self):
        """Initialize context manager"""
        self._lock = threading.RLock()
        
        # Conversation state
        self._messages: deque = deque(maxlen=self.MAX_MESSAGES)
        
        # Active task context
        self._active_task_id: Optional[str] = None
        self._active_plan_id: Optional[str] = None
        self._active_tasks: List[str] = []
        
        # Execution metadata
        self._execution_records: deque = deque(maxlen=self.MAX_EXECUTION_RECORDS)
        self._task_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Session metadata
        self._session_start: float = time.time()
        self._session_metadata: Dict[str, Any] = {}
        
        logger.info("ContextManager initialized")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message text content
            metadata: Optional metadata dict
            
        Returns:
            Created Message object
            
        Raises:
            ValueError: If role is invalid
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._messages.append(message)
            logger.debug(f"Added {role} message (history size: {len(self._messages)})")
        
        return message
    
    def get_messages(
        self,
        role: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get conversation messages.
        
        Args:
            role: Filter by role (optional)
            limit: Maximum number of recent messages to return
            
        Returns:
            List of messages (newest last)
        """
        with self._lock:
            messages = list(self._messages)
        
        # Filter by role if specified
        if role:
            messages = [m for m in messages if m.role == role]
        
        # Apply limit if specified
        if limit and limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history as serializable dicts.
        
        Returns:
            List of message dicts with role, content, timestamp
        """
        with self._lock:
            return [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'metadata': msg.metadata
                }
                for msg in self._messages
            ]
    
    def clear_messages(self):
        """Clear all conversation messages"""
        with self._lock:
            self._messages.clear()
            logger.info("Conversation history cleared")
    
    def set_active_task(self, task_id: Optional[str]):
        """Set the currently active task"""
        with self._lock:
            self._active_task_id = task_id
            if task_id and task_id not in self._active_tasks:
                self._active_tasks.append(task_id)
            logger.debug(f"Active task set to: {task_id}")
    
    def get_active_task(self) -> Optional[str]:
        """Get currently active task ID"""
        with self._lock:
            return self._active_task_id
    
    def set_active_plan(self, plan_id: Optional[str]):
        """Set the currently active plan"""
        with self._lock:
            self._active_plan_id = plan_id
            logger.debug(f"Active plan set to: {plan_id}")
    
    def get_active_plan(self) -> Optional[str]:
        """Get currently active plan ID"""
        with self._lock:
            return self._active_plan_id
    
    def get_active_tasks(self) -> List[str]:
        """Get list of all active task IDs"""
        with self._lock:
            return list(self._active_tasks)
    
    def clear_active_tasks(self):
        """Clear active tasks list"""
        with self._lock:
            self._active_tasks.clear()
            logger.debug("Active tasks cleared")
    
    def record_execution_event(
        self,
        task_id: str,
        event_type: str,
        agent_type: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a task execution event.
        
        Args:
            task_id: Task identifier
            event_type: Event type (started, completed, failed, cancelled)
            agent_type: Agent type that handled task
            error: Error message if applicable
            metadata: Additional event metadata
        """
        record = ExecutionRecord(
            task_id=task_id,
            event_type=event_type,
            timestamp=time.time(),
            agent_type=agent_type,
            error=error,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._execution_records.append(record)
            logger.debug(f"Recorded execution event: {task_id} - {event_type}")
    
    def get_execution_records(
        self,
        task_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExecutionRecord]:
        """
        Get execution records with optional filtering.
        
        Args:
            task_id: Filter by task ID
            event_type: Filter by event type
            limit: Maximum number of recent records
            
        Returns:
            List of execution records
        """
        with self._lock:
            records = list(self._execution_records)
        
        # Apply filters
        if task_id:
            records = [r for r in records if r.task_id == task_id]
        if event_type:
            records = [r for r in records if r.event_type == event_type]
        
        # Apply limit
        if limit and limit > 0:
            records = records[-limit:]
        
        return records
    
    def set_task_metadata(self, task_id: str, key: str, value: Any):
        """
        Store metadata for a specific task.
        
        Args:
            task_id: Task identifier
            key: Metadata key
            value: Metadata value
        """
        with self._lock:
            if task_id not in self._task_metadata:
                self._task_metadata[task_id] = {}
            self._task_metadata[task_id][key] = value
            logger.debug(f"Set task metadata: {task_id}.{key}")
    
    def get_task_metadata(
        self,
        task_id: str,
        key: Optional[str] = None
    ) -> Any:
        """
        Get metadata for a specific task.
        
        Args:
            task_id: Task identifier
            key: Specific metadata key (returns all if None)
            
        Returns:
            Metadata value or dict of all metadata
        """
        with self._lock:
            task_meta = self._task_metadata.get(task_id, {})
            if key:
                return task_meta.get(key)
            return dict(task_meta)
    
    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        Get all execution metadata for snapshot creation.
        
        Returns:
            Dict containing execution state and metadata
        """
        with self._lock:
            return {
                'active_task': self._active_task_id,
                'active_plan': self._active_plan_id,
                'active_tasks_count': len(self._active_tasks),
                'session_duration': time.time() - self._session_start,
                'total_messages': len(self._messages),
                'total_execution_records': len(self._execution_records),
                'session_metadata': dict(self._session_metadata),
                'recent_events': [
                    {
                        'task_id': r.task_id,
                        'event_type': r.event_type,
                        'timestamp': r.timestamp,
                        'agent_type': r.agent_type
                    }
                    for r in list(self._execution_records)[-10:]
                ]
            }
    
    def set_session_metadata(self, key: str, value: Any):
        """Set session-level metadata"""
        with self._lock:
            self._session_metadata[key] = value
            logger.debug(f"Set session metadata: {key}")
    
    def get_session_metadata(self, key: Optional[str] = None) -> Any:
        """
        Get session metadata.
        
        Args:
            key: Specific key (returns all if None)
            
        Returns:
            Metadata value or dict
        """
        with self._lock:
            if key:
                return self._session_metadata.get(key)
            return dict(self._session_metadata)
    
    def create_snapshot(self, task_id: str) -> 'ContextSnapshot':
        """
        Create a read-only snapshot for agent execution.
        
        Args:
            task_id: Task requesting the snapshot
            
        Returns:
            ContextSnapshot object
        """
        from .orchestrator import ContextSnapshot
        
        with self._lock:
            return ContextSnapshot(
                task_id=task_id,
                conversation_history=self.get_conversation_history(),
                execution_metadata=self.get_execution_metadata(),
                active_tasks=list(self._active_tasks),
                timestamp=time.time()
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context manager statistics.
        
        Returns:
            Dict with usage statistics
        """
        with self._lock:
            return {
                'messages_count': len(self._messages),
                'messages_capacity': self.MAX_MESSAGES,
                'execution_records_count': len(self._execution_records),
                'execution_records_capacity': self.MAX_EXECUTION_RECORDS,
                'active_tasks_count': len(self._active_tasks),
                'tracked_tasks_count': len(self._task_metadata),
                'session_duration_seconds': time.time() - self._session_start,
                'session_start': self._session_start
            }
    
    def reset(self):
        """
        Reset context manager to initial state.
        
        Warning: Clears all conversation history and metadata.
        """
        with self._lock:
            self._messages.clear()
            self._execution_records.clear()
            self._task_metadata.clear()
            self._session_metadata.clear()
            self._active_task_id = None
            self._active_plan_id = None
            self._active_tasks.clear()
            self._session_start = time.time()
            logger.warning("ContextManager reset - all state cleared")