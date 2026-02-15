"""
ZENO Mode Manager - Offline Task Routing Logic

Responsibilities:
- Route tasks to appropriate agent types
- Enforce offline-only constraints
- Provide deterministic routing decisions
- NO LLM calls, NO execution, NO business logic
"""

import logging
from typing import Optional, Dict, Any

from zeno.core.orchestrator import AgentType, Task

logger = logging.getLogger(__name__)


class RoutingError(Exception):
    """Raised when task cannot be routed"""
    pass


class ModeManager:
    """
    Deterministic offline routing for ZENO tasks.
    
    Routes tasks based on:
    - Task.type (primary)
    - Task metadata (secondary)
    - Offline-only validation
    
    Does NOT:
    - Call LLMs
    - Execute tasks
    - Perform NLP or content analysis
    """
    
    def __init__(self, enforce_offline: bool = True):
        """
        Initialize mode manager.
        
        Args:
            enforce_offline: Whether to reject tasks requiring network
        """
        self.enforce_offline = enforce_offline
        self._routing_rules: Dict[AgentType, Dict[str, Any]] = {}
        self._initialize_default_rules()
        
        logger.info(f"ModeManager initialized (enforce_offline={enforce_offline})")
    
    def _initialize_default_rules(self):
        """Initialize default routing rules"""
        # These rules can be extended in future phases
        self._routing_rules = {
            AgentType.PLANNER: {
                'description': 'Multi-step planning and task decomposition',
                'requires_offline': True
            },
            AgentType.CHAT: {
                'description': 'Conversational responses',
                'requires_offline': True
            },
            AgentType.DEVELOPER: {
                'description': 'Code generation and analysis',
                'requires_offline': True
            },
            AgentType.SYSTEM: {
                'description': 'System operations and control',
                'requires_offline': True
            }
        }
    
    def route(self, task: Task) -> AgentType:
        """
        Route a task to the appropriate agent type.
        
        Args:
            task: Task to route
            
        Returns:
            AgentType for handling the task
            
        Raises:
            RoutingError: If task cannot be routed
        """
        # Validate task
        if not task:
            raise RoutingError("Cannot route null task")
        
        if not task.id:
            raise RoutingError("Task must have an ID")
        
        # Enforce offline-only constraint
        if self.enforce_offline and task.requires_network:
            logger.error(f"Task {task.id} requires network but offline mode enforced")
            raise RoutingError(
                f"Task {task.id} requires network access but ZENO is in offline-only mode"
            )
        
        # Primary routing: use task.type directly
        if not isinstance(task.type, AgentType):
            raise RoutingError(
                f"Task {task.id} has invalid type: {task.type}. "
                f"Must be an AgentType enum."
            )
        
        agent_type = task.type
        
        # Validate agent type is known
        if agent_type not in self._routing_rules:
            raise RoutingError(
                f"Unknown agent type: {agent_type}. "
                f"Known types: {list(self._routing_rules.keys())}"
            )
        
        # Verify offline compatibility
        rule = self._routing_rules[agent_type]
        if rule.get('requires_offline') and task.requires_network:
            raise RoutingError(
                f"Agent type {agent_type.value} requires offline mode "
                f"but task {task.id} needs network"
            )
        
        logger.info(f"Routed task {task.id} to {agent_type.value}")
        return agent_type
    
    def validate_task(self, task: Task) -> bool:
        """
        Validate if a task can be routed successfully.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task can be routed, False otherwise
        """
        try:
            self.route(task)
            return True
        except RoutingError as e:
            logger.warning(f"Task validation failed: {e}")
            return False
    
    def get_agent_info(self, agent_type: AgentType) -> Dict[str, Any]:
        """
        Get information about an agent type.
        
        Args:
            agent_type: Agent type to query
            
        Returns:
            Dict with agent type metadata
        """
        if agent_type not in self._routing_rules:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return {
            'type': agent_type.value,
            **self._routing_rules[agent_type]
        }
    
    def get_available_agents(self) -> list[AgentType]:
        """
        Get list of available agent types.
        
        Returns:
            List of AgentType enums
        """
        return list(self._routing_rules.keys())
    
    def set_offline_mode(self, enabled: bool):
        """
        Enable or disable offline-only enforcement.
        
        Args:
            enabled: True to enforce offline-only
        """
        self.enforce_offline = enabled
        logger.info(f"Offline mode enforcement set to: {enabled}")
    
    def is_offline_mode(self) -> bool:
        """Check if offline-only mode is enforced"""
        return self.enforce_offline
    
    def can_handle_network_tasks(self) -> bool:
        """
        Check if manager can route network-requiring tasks.
        
        Returns:
            True if network tasks are allowed
        """
        return not self.enforce_offline
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics and configuration.
        
        Returns:
            Dict with routing stats
        """
        return {
            'enforce_offline': self.enforce_offline,
            'available_agent_types': [at.value for at in self.get_available_agents()],
            'routing_rules_count': len(self._routing_rules)
        }