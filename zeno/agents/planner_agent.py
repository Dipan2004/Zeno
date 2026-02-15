"""
ZENO Planner Agent - Task Planning & Intent Analysis

Responsibilities:
- Parse user commands and identify intents
- Create structured task graphs with dependencies
- Assign appropriate agent types to tasks
- Generate human-readable execution plans
- Validate task graphs before returning

This agent does NOT execute tasks - it only plans them.
"""

import json
import logging
import uuid
from typing import Tuple, List, Dict, Any, Optional

from zeno.core import TaskGraph, Task, AgentType, ContextSnapshot
from zeno.llm import LocalLLM, QWEN_3B_INSTRUCT, OllamaError

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Raised when planning fails or produces invalid output"""
    pass


class PlannerAgent:
    """
    Planning agent that converts user intent into executable task graphs.
    
    NOT an executable agent - this is a planning service that produces
    TaskGraphs for the Orchestrator to execute.
    
    Uses Mistral 7B via LocalLLM for intent analysis and task generation.
    """
    
    # System prompt defining ZENO's planning capabilities
    SYSTEM_PROMPT = """You are ZENO's task planning system. You are offline-only and cannot access the internet or cloud services.

Your job: Convert user commands into structured task plans with clear, reassuring explanations.

AVAILABLE TASK TYPES:
- SYSTEM: Opening applications, URLs, system operations
- DEVELOPER: Code generation, file creation, technical tasks
- CHAT: Conversational responses, explanations

TASK PAYLOAD FORMATS (use exact structure):

For SYSTEM tasks (opening apps):
  {
    "action": "open_app",
    "app": "vscode"  # Known apps: vscode, vs code, code
  }

For SYSTEM tasks (opening URLs):
  {
    "action": "open_url",
    "url": "https://github.com"  # Must be http:// or https://
  }

For SYSTEM tasks (creating files - usually via DEVELOPER):
  {
    "action": "create_file",
    "filename": "script.py",
    "content": "..."  # Actual content
  }
  
For DEVELOPER tasks (code generation):
  {
    "intent": "generate_code",
    "description": "Write a function to reverse a string",
    "language": "python"  # or javascript, java, etc.
  }

For DEVELOPER tasks (simple file creation - when user just wants a file created):
  {
    "intent": "create_file",
    "filename": "hello.c",
    "description": "optional description"  # optional
  }
  
For CHAT tasks:
  {
    "intent": "respond",
    "message": "User's question or statement"
  }

AVAILABLE SYSTEM CAPABILITIES:
- Open VS Code (vscode, vs code, code)
- Open URLs in browser (must be http:// or https://)
- Generate code (python, javascript, java, etc.)
- Create files in workspace (~/zeno_workspace)

RULES:
1. You are OFFLINE - no network calls, no cloud APIs
2. Be explicit about what each task does
3. Identify dependencies between tasks
4. Mark tasks as parallel-safe when appropriate
5. Use clear, human-readable task names
6. If user intent is unclear, return error in explanation

EXPLANATION STYLE:
- Keep explanations brief but reassuring
- Simple tasks: One confident sentence
- Multi-step tasks: Brief overview of steps
- Use phrases like "I'll help you...", "This should only take a moment"
- Stay professional and calm, not emotional
- Example: "I'll help you open VS Code. It should only take a moment."

OUTPUT FORMAT (JSON only):
{
  "tasks": [
    {
      "name": "Human readable task name",
      "description": "Clear description",
      "type": "SYSTEM|DEVELOPER|CHAT",
      "payload": { use exact format above },
      "dependencies": [],
      "can_run_parallel": true|false
    }
  ],
  "explanation": "Brief, reassuring explanation of what ZENO will do"
}

EXAMPLES:

User: "Open VS Code"
{
  "tasks": [
    {
      "name": "Open Visual Studio Code",
      "description": "Launch VS Code editor",
      "type": "SYSTEM",
      "payload": {"action": "open_app", "app": "vscode"},
      "dependencies": [],
      "can_run_parallel": false
    }
  ],
  "explanation": "I'll help you open VS Code. It should only take a moment."
}

User: "Open GitHub"
{
  "tasks": [
    {
      "name": "Open GitHub website",
      "description": "Open GitHub in browser",
      "type": "SYSTEM",
      "payload": {"action": "open_url", "url": "https://github.com"},
      "dependencies": [],
      "can_run_parallel": false
    }
  ],
  "explanation": "I'll open GitHub in your browser."
}

User: "Write a Python function to reverse a string"
{
  "tasks": [
    {
      "name": "Generate string reversal function",
      "description": "Create Python function to reverse strings",
      "type": "DEVELOPER",
      "payload": {
        "intent": "generate_code",
        "description": "Write a Python function to reverse a string",
        "language": "python"
      },
      "dependencies": [],
      "can_run_parallel": false
    }
  ],
  "explanation": "I'll write a Python function that reverses a string and save it to your workspace."
}

User: "Open VS Code and write a sorting function"
{
  "tasks": [
    {
      "name": "Open Visual Studio Code",
      "description": "Launch VS Code",
      "type": "SYSTEM",
      "payload": {"action": "open_app", "app": "vscode"},
      "dependencies": [],
      "can_run_parallel": true
    },
    {
      "name": "Generate sorting function",
      "description": "Create sorting function code",
      "type": "DEVELOPER",
      "payload": {
        "intent": "generate_code",
        "description": "Write a sorting function",
        "language": "python"
      },
      "dependencies": [],
      "can_run_parallel": true
    }
  ],
  "explanation": "I'll open VS Code and generate a sorting function for you. Both will happen at the same time."
}

Now process the user's command."""
    
    def __init__(self, llm_client: LocalLLM):
        """
        Initialize planner agent.
        
        Args:
            llm_client: LocalLLM instance for inference
        """
        self.llm = llm_client
        self._task_counter = 0
        logger.info("PlannerAgent initialized")
    
    def plan(
        self,
        user_input: str,
        context: ContextSnapshot
    ) -> Tuple[TaskGraph, str]:
        """
        Create execution plan from user input.
        
        Args:
            user_input: User command or question
            context: Current conversation context (read-only)
            
        Returns:
            Tuple of (TaskGraph, explanation)
            - TaskGraph: Validated DAG ready for execution
            - explanation: Human-readable plan description
            
        Raises:
            PlanningError: If planning fails or produces invalid output
            
        Example:
            >>> planner = PlannerAgent(llm_client)
            >>> graph, explanation = planner.plan(
            ...     "Open VS Code",
            ...     context_snapshot
            ... )
            >>> print(explanation)
            "I'll open Visual Studio Code for you."
        """
        if not user_input or not user_input.strip():
            raise PlanningError("Cannot plan with empty user input")
        
        logger.info(f"Planning for input: '{user_input[:50]}...'")
        
        try:
            # Build prompt with context
            prompt = self._build_planning_prompt(user_input, context)
            
            # Get structured plan from LLM
            logger.debug("Requesting plan from LLM...")
            response = self.llm.generate(
                prompt=prompt,
                model=QWEN_3B_INSTRUCT,
                temperature=0.3,  # Lower for more deterministic planning
                max_tokens=1200,  # Reduced for smaller model efficiency
                timeout=120
            )
            
            # Parse JSON response
            plan_data = self._parse_llm_response(response.text)
            
            # Build and validate task graph
            tasks = self._build_tasks(plan_data["tasks"])
            explanation = plan_data["explanation"]
            
            # Create and validate TaskGraph
            task_graph = TaskGraph(tasks)
            
            logger.info(
                f"Plan created: {len(tasks)} tasks, "
                f"explanation: '{explanation[:50]}...'"
            )
            
            return task_graph, explanation
            
        except OllamaError as e:
            logger.error(f"LLM error during planning: {e}")
            raise PlanningError(f"LLM service error: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise PlanningError(f"LLM returned invalid JSON: {e}") from e
        except KeyError as e:
            logger.error(f"Missing required field in plan: {e}")
            raise PlanningError(f"Invalid plan structure: missing {e}") from e
        except ValueError as e:
            logger.error(f"Invalid task graph: {e}")
            raise PlanningError(f"Invalid task graph: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected planning error: {e}", exc_info=True)
            raise PlanningError(f"Planning failed: {e}") from e
    
    def _build_planning_prompt(
        self,
        user_input: str,
        context: ContextSnapshot
    ) -> str:
        """
        Build complete prompt for LLM including context.
        
        Args:
            user_input: User's command
            context: Conversation context
            
        Returns:
            Complete prompt string
        """
        prompt_parts = [self.SYSTEM_PROMPT]
        
        # Add recent conversation history for context (last 3-5 messages)
        history = context.conversation_history[-5:]
        if history:
            prompt_parts.append("\n\nRECENT CONVERSATION:")
            for msg in history:
                role = msg['role'].upper()
                content = msg['content'][:100]  # Truncate long messages
                prompt_parts.append(f"{role}: {content}")
        
        # Add current user input
        prompt_parts.append(f"\n\nUSER COMMAND: {user_input}")
        prompt_parts.append("\n\nYour response (JSON only):")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM JSON response.
        
        Args:
            response_text: Raw text from LLM
            
        Returns:
            Parsed plan data dict
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
            PlanningError: If response structure is invalid
        """
        # Clean response (remove markdown code blocks if present)
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Parse JSON
        try:
            plan_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                plan_data = json.loads(cleaned[start:end])
            else:
                raise
        
        # Validate required fields
        if "tasks" not in plan_data:
            raise PlanningError("LLM response missing 'tasks' field")
        if "explanation" not in plan_data:
            raise PlanningError("LLM response missing 'explanation' field")
        
        if not isinstance(plan_data["tasks"], list):
            raise PlanningError("'tasks' must be a list")
        
        if not plan_data["tasks"]:
            raise PlanningError("Plan must contain at least one task")
        
        return plan_data
    
    def _build_tasks(self, task_data_list: List[Dict[str, Any]]) -> List[Task]:
        """
        Build Task objects from parsed plan data.
        
        Args:
            task_data_list: List of task dictionaries from LLM
            
        Returns:
            List of validated Task objects
            
        Raises:
            PlanningError: If task data is invalid
        """
        tasks = []
        task_id_map = {}  # Map temp IDs to generated IDs
        
        for idx, task_data in enumerate(task_data_list):
            # Generate deterministic task ID
            self._task_counter += 1
            task_id = f"task-{self._task_counter}"
            
            # Validate required fields
            self._validate_task_data(task_data, idx)
            
            # Parse agent type
            agent_type = self._parse_agent_type(task_data["type"])
            
            # Parse dependencies (map from temp IDs to real IDs)
            dependencies = []
            for dep_idx in task_data.get("dependencies", []):
                if isinstance(dep_idx, int):
                    if dep_idx >= len(task_data_list):
                        raise PlanningError(
                            f"Task {idx} depends on invalid task index {dep_idx}"
                        )
                    dependencies.append(task_id_map.get(dep_idx, f"task-{dep_idx + 1}"))
                elif isinstance(dep_idx, str):
                    dependencies.append(dep_idx)
            
            # Create task
            task = Task(
                id=task_id,
                name=task_data["name"],
                description=task_data.get("description", task_data["name"]),
                type=agent_type,
                payload=task_data["payload"],
                dependencies=dependencies,
                can_run_parallel=task_data.get("can_run_parallel", True),
                requires_network=False  # Always offline
            )
            
            tasks.append(task)
            task_id_map[idx] = task_id
        
        return tasks
    
    def _validate_task_data(self, task_data: Dict[str, Any], index: int):
        """
        Validate task data structure.
        
        Args:
            task_data: Task dictionary from LLM
            index: Task index for error reporting
            
        Raises:
            PlanningError: If validation fails
        """
        required_fields = ["name", "type", "payload"]
        
        for field in required_fields:
            if field not in task_data:
                raise PlanningError(
                    f"Task {index} missing required field: '{field}'"
                )
        
        # Validate payload is a dict
        if not isinstance(task_data["payload"], dict):
            raise PlanningError(
                f"Task {index} payload must be a dictionary"
            )
        
        # Validate payload structure based on task type
        payload = task_data["payload"]
        task_type = task_data["type"].upper()
        
        # SYSTEM tasks use "action" field
        if task_type == "SYSTEM":
            if "action" not in payload:
                raise PlanningError(
                    f"Task {index} (SYSTEM) payload must contain 'action' field"
                )
        # DEVELOPER and CHAT tasks use "intent" field
        elif task_type in ["DEVELOPER", "CHAT"]:
            if "intent" not in payload:
                raise PlanningError(
                    f"Task {index} ({task_type}) payload must contain 'intent' field"
                )
        else:
            # Unknown task type - allow it but warn
            logger.warning(f"Task {index} has unknown type: {task_type}")
    
    def _parse_agent_type(self, type_str: str) -> AgentType:
        """
        Parse agent type string to AgentType enum.
        
        Args:
            type_str: Agent type as string
            
        Returns:
            AgentType enum
            
        Raises:
            PlanningError: If type is invalid
        """
        type_map = {
            "SYSTEM": AgentType.SYSTEM,
            "DEVELOPER": AgentType.DEVELOPER,
            "CHAT": AgentType.CHAT,
            "PLANNER": AgentType.PLANNER,
        }
        
        type_upper = type_str.upper()
        if type_upper not in type_map:
            raise PlanningError(
                f"Invalid agent type: '{type_str}'. "
                f"Must be one of: {list(type_map.keys())}"
            )
        
        return type_map[type_upper]
    
    def reset_counter(self):
        """Reset task counter (useful for testing)"""
        self._task_counter = 0
        logger.debug("Task counter reset")