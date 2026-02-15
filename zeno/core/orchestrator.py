"""
ZENO Core Orchestrator - Central Execution Kernel

Responsibilities:
- Task lifecycle management
- DAG-based task graph execution
- Parallel execution with dependency resolution
- Cooperative interruption
- Progress hooks and logging
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class AgentType(Enum):
    """Agent types for task routing"""
    PLANNER = "planner"
    CHAT = "chat"
    DEVELOPER = "developer"
    SYSTEM = "system"


@dataclass
class TaskResult:
    """Result of task execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """
    Immutable task definition with mutable execution state.
    
    Immutable fields:
        - id, name, description, type, payload, dependencies, can_run_parallel
    
    Mutable fields (managed by orchestrator):
        - status, result, error
    """
    id: str
    name: str
    description: str
    type: AgentType
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    can_run_parallel: bool = True
    requires_network: bool = False
    
    # Mutable execution state
    status: TaskStatus = field(default=TaskStatus.PENDING)
    result: Optional[TaskResult] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Validate task definition"""
        if not self.id:
            raise ValueError("Task ID cannot be empty")
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not isinstance(self.type, AgentType):
            raise ValueError(f"Task type must be AgentType, got {type(self.type)}")


@dataclass
class ContextSnapshot:
    """Read-only snapshot of execution context for agents"""
    task_id: str
    conversation_history: List[Dict[str, Any]]
    execution_metadata: Dict[str, Any]
    active_tasks: List[str]
    timestamp: float = field(default_factory=time.time)


class Agent(ABC):
    """
    Abstract base class for all ZENO agents.
    
    Agents are pure execution units that receive:
    - Task definition
    - Read-only context snapshot
    - Interrupt signal for cooperative cancellation
    """
    
    @abstractmethod
    def execute(
        self,
        task: Task,
        context_snapshot: ContextSnapshot,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Execute a task with given context.
        
        Args:
            task: Task to execute
            context_snapshot: Read-only context at execution time
            interrupt_event: Signal for cooperative cancellation
            
        Returns:
            TaskResult with success status and data/error
            
        Raises:
            Exception: Agent-specific errors (will be caught by orchestrator)
        """
        pass


class TaskGraph:
    """
    Directed Acyclic Graph (DAG) for task dependencies.
    
    Validates:
    - No cycles
    - All dependencies exist
    - Task IDs are unique
    """
    
    def __init__(self, tasks: List[Task]):
        """
        Initialize and validate task graph.
        
        Args:
            tasks: List of tasks to execute
            
        Raises:
            ValueError: If graph contains cycles or invalid dependencies
        """
        self.tasks: Dict[str, Task] = {t.id: t for t in tasks}
        self._validate_graph()
        
    def _validate_graph(self):
        """Validate DAG properties"""
        # Check for duplicate IDs
        if len(self.tasks) != len(set(self.tasks.keys())):
            raise ValueError("Duplicate task IDs detected")
        
        # Check all dependencies exist
        for task in self.tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(
                        f"Task {task.id} depends on non-existent task {dep_id}"
                    )
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError(f"Cycle detected in task graph involving {task_id}")
    
    def get_ready_tasks(self) -> List[Task]:
        """
        Get all tasks that are ready to execute.
        
        A task is ready if:
        - Status is PENDING
        - All dependencies are COMPLETED
        
        Returns:
            List of tasks ready for execution
        """
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check all dependencies are completed
            deps_complete = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if deps_complete:
                ready.append(task)
        
        return ready
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_dependents(self, task_id: str) -> List[Task]:
        """Get all tasks that depend on the given task"""
        dependents = []
        for task in self.tasks.values():
            if task_id in task.dependencies:
                dependents.append(task)
        return dependents


class Orchestrator:
    """
    Central execution kernel for ZENO.
    
    Responsibilities:
    - Execute task graphs with dependency resolution
    - Manage parallel execution with thread pool
    - Handle interruption and cancellation
    - Emit progress events via hooks
    - Provide graceful error handling
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        context_manager: Optional[Any] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            max_workers: Maximum parallel tasks (default: 4)
            context_manager: ContextManager instance for state tracking
        """
        self.max_workers = max_workers
        self.context_manager = context_manager
        self.agent_registry: Dict[AgentType, Agent] = {}
        
        # Execution state
        self._interrupt_event = threading.Event()
        self._active_threads: Set[threading.Thread] = set()
        self._execution_lock = threading.Lock()
        
        # Progress hooks
        self._hooks: Dict[str, List[Callable]] = {
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'task_cancelled': [],
            'plan_started': [],
            'plan_finished': []
        }
        
        logger.info(f"Orchestrator initialized with max_workers={max_workers}")
    
    def register_agent(self, agent_type: AgentType, agent: Agent):
        """Register an agent for a specific type"""
        if not isinstance(agent, Agent):
            raise TypeError(f"Agent must inherit from Agent base class")
        self.agent_registry[agent_type] = agent
        logger.info(f"Registered agent for type: {agent_type.value}")
    
    def register_hook(self, event: str, callback: Callable):
        """
        Register a progress hook callback.
        
        Args:
            event: Event name (task_started, task_completed, etc.)
            callback: Function to call when event occurs
            
        Raises:
            ValueError: If event name is invalid
        """
        if event not in self._hooks:
            raise ValueError(f"Invalid event: {event}")
        self._hooks[event].append(callback)
        logger.debug(f"Registered hook for event: {event}")
    
    def _emit_event(self, event: str, *args, **kwargs):
        """Emit event to all registered hooks"""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook callback failed for {event}: {e}")
    
    def execute_plan(self, tasks: List[Task]) -> Dict[str, TaskResult]:
        """
        Execute a task graph (plan).
        
        Args:
            tasks: List of tasks forming a DAG
            
        Returns:
            Dict mapping task_id to TaskResult
            
        Raises:
            ValueError: If graph is invalid
            RuntimeError: If execution fails critically
        """
        logger.info(f"Starting plan execution with {len(tasks)} tasks")
        self._emit_event('plan_started', tasks)
        
        # Reset interrupt flag
        self._interrupt_event.clear()
        
        try:
            # Build and validate task graph
            graph = TaskGraph(tasks)
            
            # Execute until all tasks are terminal
            results = self._execute_graph(graph)
            
            logger.info(f"Plan execution completed")
            self._emit_event('plan_finished', results)
            
            return results
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            self._emit_event('plan_finished', {})
            raise RuntimeError(f"Plan execution failed: {e}") from e
    
    def _execute_graph(self, graph: TaskGraph) -> Dict[str, TaskResult]:
        """Execute task graph with dependency resolution"""
        results = {}
        
        while True:
            # Check for interruption
            if self._interrupt_event.is_set():
                logger.warning("Execution interrupted by user")
                self._cancel_pending_tasks(graph)
                break
            
            # Get tasks ready to execute
            ready_tasks = graph.get_ready_tasks()
            
            if not ready_tasks:
                # Check if we're done or stuck
                pending = [t for t in graph.tasks.values() 
                          if t.status == TaskStatus.PENDING]
                if not pending:
                    break  # All tasks processed
                else:
                    # Shouldn't happen with valid DAG, but safety check
                    logger.error(f"Deadlock detected: {len(pending)} tasks pending but none ready")
                    for task in pending:
                        self._cancel_task(task, "Deadlock detected")
                    break
            
            # Execute ready tasks (respecting max_workers)
            self._execute_ready_tasks(ready_tasks, graph, results)
        
        return results
    
    def _execute_ready_tasks(
        self,
        ready_tasks: List[Task],
        graph: TaskGraph,
        results: Dict[str, TaskResult]
    ):
        """Execute ready tasks with parallelism control"""
        # Separate parallel and sequential tasks
        parallel_tasks = [t for t in ready_tasks if t.can_run_parallel]
        sequential_tasks = [t for t in ready_tasks if not t.can_run_parallel]
        
        # Execute parallel tasks up to max_workers
        threads = []
        for task in parallel_tasks[:self.max_workers]:
            thread = threading.Thread(
                target=self._execute_task_thread,
                args=(task, graph, results),
                daemon=False
            )
            thread.start()
            threads.append(thread)
            
            with self._execution_lock:
                self._active_threads.add(thread)
        
        # Wait for parallel tasks to complete
        for thread in threads:
            thread.join()
            with self._execution_lock:
                self._active_threads.discard(thread)
        
        # Execute sequential tasks one by one
        for task in sequential_tasks:
            if self._interrupt_event.is_set():
                break
            self._execute_task(task, graph, results)
    
    def _execute_task_thread(
        self,
        task: Task,
        graph: TaskGraph,
        results: Dict[str, TaskResult]
    ):
        """Thread wrapper for task execution"""
        try:
            self._execute_task(task, graph, results)
        except Exception as e:
            logger.error(f"Thread execution failed for {task.id}: {e}", exc_info=True)
    
    def _execute_task(
        self,
        task: Task,
        graph: TaskGraph,
        results: Dict[str, TaskResult]
    ):
        """Execute a single task"""
        # Mark as running
        task.status = TaskStatus.RUNNING
        logger.info(f"Task started: {task.id} ({task.name})")
        self._emit_event('task_started', task)
        
        try:
            # Get agent for task type
            agent = self.agent_registry.get(task.type)
            if not agent:
                raise RuntimeError(f"No agent registered for type: {task.type.value}")
            
            # Create context snapshot
            context_snapshot = self._create_context_snapshot(task)
            
            # Execute task
            result = agent.execute(task, context_snapshot, self._interrupt_event)
            
            # Check if interrupted during execution
            if self._interrupt_event.is_set():
                task.status = TaskStatus.INTERRUPTED
                task.error = "Execution interrupted"
                logger.warning(f"Task interrupted: {task.id}")
                self._emit_event('task_cancelled', task)
                self._cancel_dependents(task, graph)
                return
            
            # Handle result
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result
                results[task.id] = result
                logger.info(f"Task completed: {task.id}")
                self._emit_event('task_completed', task)
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error or "Unknown error"
                task.result = result
                results[task.id] = result
                logger.error(f"Task failed: {task.id} - {task.error}")
                self._emit_event('task_failed', task, task.error)
                self._cancel_dependents(task, graph)
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task execution error: {task.id} - {e}", exc_info=True)
            self._emit_event('task_failed', task, str(e))
            self._cancel_dependents(task, graph)
    
    def _cancel_task(self, task: Task, reason: str):
        """Cancel a task"""
        task.status = TaskStatus.CANCELLED
        task.error = reason
        logger.info(f"Task cancelled: {task.id} - {reason}")
        self._emit_event('task_cancelled', task)
    
    def _cancel_dependents(self, task: Task, graph: TaskGraph):
        """Cancel all tasks that depend on the given task"""
        dependents = graph.get_dependents(task.id)
        for dependent in dependents:
            if dependent.status == TaskStatus.PENDING:
                self._cancel_task(
                    dependent,
                    f"Dependency failed: {task.id}"
                )
                # Recursively cancel dependents
                self._cancel_dependents(dependent, graph)
    
    def _cancel_pending_tasks(self, graph: TaskGraph):
        """Cancel all pending tasks"""
        for task in graph.tasks.values():
            if task.status == TaskStatus.PENDING:
                self._cancel_task(task, "Plan interrupted")
    
    def _create_context_snapshot(self, task: Task) -> ContextSnapshot:
        """Create read-only context snapshot for agent"""
        if self.context_manager:
            return self.context_manager.create_snapshot(task.id)
        else:
            # Fallback for testing without context manager
            return ContextSnapshot(
                task_id=task.id,
                conversation_history=[],
                execution_metadata={},
                active_tasks=[]
            )
    
    def interrupt(self):
        """
        Signal cooperative interruption.
        
        Sets interrupt flag that agents should check periodically.
        Does not force-kill threads.
        """
        logger.warning("Interrupt signal received")
        self._interrupt_event.set()
    
    def is_interrupted(self) -> bool:
        """Check if interrupt signal is set"""
        return self._interrupt_event.is_set()
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all active threads to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all threads completed, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self._execution_lock:
                active = list(self._active_threads)
            
            if not active:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            # Wait for threads with small timeout to allow checking
            for thread in active:
                thread.join(timeout=0.1)
            
            time.sleep(0.1)