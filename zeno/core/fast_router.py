"""
ZENO Fast Router - Enhanced with Code Generation + OS Control Detection

Detects system commands, code generation, AND OS control tasks BEFORE calling PlannerAgent (LLM).

Performance Goal: < 50ms, ZERO LLM tokens for obvious tasks

Phase 5 Updates:
- Volume control pattern detection
- Brightness control pattern detection
- Correct priority order to avoid conflicts

Architectural Rules:
- Pure classifier only
- NO execution logic
- NO system tool calls
- Returns Task or None

Supported patterns:
- "write <code> in <file>" → DeveloperAgent
- "set volume to 50" → SystemAgent (NEW!)
- "increase brightness" → SystemAgent (NEW!)
- "open <target>" → SystemAgent
"""

import logging
import re
from typing import Optional
from zeno.core import Task, AgentType

logger = logging.getLogger(__name__)


# ============================================================================
# CLASSIFICATION KNOWLEDGE
# ============================================================================

# Known app aliases for fast-path classification
KNOWN_APPS = {
    # Dev Tools
    "vscode", "vs code", "visual studio code", "code",
    
    # Browsers
    "edge", "microsoft edge",
    "chrome", "google chrome",
    "firefox", "mozilla firefox",
    
    # Windows System Apps
    "explorer", "file explorer", "files",
    "notepad",
    "calculator", "calc",
    "paint", "mspaint",
    "cmd", "command prompt",
    "powershell", "pwsh",
    "terminal", "windows terminal", "wt",
    "task manager", "taskmgr",
    "control panel", "control",
    "settings", "windows settings",
    
    # Office (if installed)
    "word", "microsoft word",
    "excel", "microsoft excel",
    "powerpoint", "microsoft powerpoint", "ppt",
    
    # Dev Tools
    "git", "git bash",
    "python", "python3",
    "node", "nodejs",
    
    # Utilities
    "notepad++", "notepad plus plus", "npp",
}

# Known site keywords
KNOWN_SITES = {
    "github", "google", "youtube", "stackoverflow",
    "twitter", "reddit",
}


# ============================================================================
# COMMAND PATTERNS (PRIORITY ORDER MATTERS!)
# ============================================================================

# Code generation patterns (HIGHEST PRIORITY - Phase 4)
CODE_PATTERNS = [
    # "write <code_description> in <filename>"
    re.compile(r"^(?:write|create|generate)\s+(?:a\s+|an\s+)?(.+?)\s+(?:in|to|into)\s+(.+\.py)$", re.IGNORECASE),
    
    # "write <filename> with <code_description>"
    re.compile(r"^(?:write|create|generate)\s+(.+\.py)\s+(?:with|containing|for)\s+(.+)$", re.IGNORECASE),
    
    # "in <filename> write <code_description>"
    re.compile(r"^in\s+(.+\.py)\s+(?:write|create|add)\s+(?:a\s+|an\s+)?(.+)$", re.IGNORECASE),
]

# Volume control patterns (SECOND PRIORITY - Phase 5)
VOLUME_PATTERNS = [
    # "set volume to 50" or "volume 50"
    re.compile(r"^(?:set\s+)?volume\s+(?:to\s+)?(\d+)%?$", re.IGNORECASE),
    
    # "increase volume" or "decrease volume"
    re.compile(r"^(increase|raise|up|decrease|lower|down)\s+volume$", re.IGNORECASE),
    
    # "mute" or "unmute"
    re.compile(r"^(mute|unmute)$", re.IGNORECASE),
]

# Brightness control patterns (THIRD PRIORITY - Phase 5)
BRIGHTNESS_PATTERNS = [
    # "set brightness to 80" or "brightness 80"
    re.compile(r"^(?:set\s+)?brightness\s+(?:to\s+)?(\d+)%?$", re.IGNORECASE),
    
    # "increase brightness" or "decrease brightness"
    re.compile(r"^(increase|raise|up|decrease|lower|down)\s+brightness$", re.IGNORECASE),
]

# System command patterns (FOURTH PRIORITY - Phase 4)
OPEN_PATTERNS = [
    re.compile(r"^open\s+(.+)$", re.IGNORECASE),
    re.compile(r"^launch\s+(.+)$", re.IGNORECASE),
    re.compile(r"^start\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?open\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:can\s+you\s+)?open\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:could\s+you\s+)?open\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?launch\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:can\s+you\s+)?launch\s+(.+)$", re.IGNORECASE),
]


class FastRouter:
    """
    Pure rule-based classifier for system commands, code generation, and OS control.
    
    Phase 5 Enhanced: Now detects volume/brightness control.
    
    Bypasses LLM for obvious system actions and code requests.
    Does NOT execute - only classifies and creates Tasks.
    """
    
    def __init__(self):
        """Initialize fast router"""
        self._task_counter = 0
        logger.info("FastRouter initialized (Phase 5: with OS control)")
    
    def try_route(self, user_input: str) -> Optional[Task]:
        """
        Attempt to classify user input as a fast-path task.
        
        Classification order (LOCKED):
        1. Code generation (highest priority)
        2. Volume control
        3. Brightness control
        4. App/URL open
        5. PlannerAgent fallback (return None)
        
        Args:
            user_input: Raw user command
            
        Returns:
            Task object if classified, None if no match
            
        Note:
            - Returns None for unknown targets (falls back to LLM)
            - Does NOT validate if apps exist or are installed
            - Pure classification only
        """
        if not user_input or not user_input.strip():
            return None
        
        # Normalize input
        normalized = user_input.strip()
        
        # PRIORITY 1: Code generation patterns (prevents "write volume control" → mute)
        code_task = self._try_code_pattern(normalized)
        if code_task:
            return code_task
        
        # PRIORITY 2: Volume control patterns
        volume_task = self._try_volume_pattern(normalized)
        if volume_task:
            return volume_task
        
        # PRIORITY 3: Brightness control patterns
        brightness_task = self._try_brightness_pattern(normalized)
        if brightness_task:
            return brightness_task
        
        # PRIORITY 4: System command patterns (open/launch/start)
        target = self._extract_target(normalized)
        if target:
            return self._classify_and_create_task(target)
        
        # No match - fall back to LLM
        return None
    
    def _try_code_pattern(self, user_input: str) -> Optional[Task]:
        """
        Try to match code generation patterns (Phase 4).
        
        Examples:
        - "write bfs in route.py"
        - "create sorting algorithm in sort.py"
        - "in test.py write factorial function"
        
        Args:
            user_input: Normalized user input
            
        Returns:
            Task for DeveloperAgent or None
        """
        for pattern in CODE_PATTERNS:
            match = pattern.match(user_input)
            if match:
                groups = match.groups()
                
                # Different patterns have different group orders
                if groups[0].endswith('.py'):
                    # Pattern 2 or 3: filename is first
                    filename = groups[0]
                    code_description = groups[1] if len(groups) > 1 else "code"
                else:
                    # Pattern 1: description is first
                    code_description = groups[0]
                    filename = groups[1] if len(groups) > 1 else "output.py"
                
                logger.info(f"✅ Code pattern matched: {code_description} → {filename}")
                return self._create_code_task(code_description, filename)
        
        return None
    
    def _create_code_task(self, code_description: str, filename: str) -> Task:
        """
        Create task with payload matching DeveloperAgent expectations.
        
        Args:
            code_description: What to generate (e.g., "bfs algorithm")
            filename: Target file (e.g., "route.py")
            
        Returns:
            Task ready for DeveloperAgent
        """
        self._task_counter += 1
        
        # Extract language from filename
        language = "python"  # Default
        if filename.endswith('.js'):
            language = "javascript"
        elif filename.endswith('.java'):
            language = "java"
        elif filename.endswith('.cpp'):
            language = "cpp"
        elif filename.endswith('.c'):
            language = "c"
        
        return Task(
            id=f"fast-code-{self._task_counter}",
            name=f"Generate {code_description}",
            description=f"Write {code_description} in {filename}",
            type=AgentType.DEVELOPER,
            payload={
                "intent": "generate_code",
                "description": f"Write {code_description}",
                "language": language,
                "filename": filename
            }
        )
    
    def _try_volume_pattern(self, user_input: str) -> Optional[Task]:
        """
        Try to match volume control patterns (Phase 5).
        
        Examples:
        - "set volume to 50"
        - "increase volume"
        - "mute"
        
        Args:
            user_input: Normalized user input
            
        Returns:
            Task for SystemAgent or None
        """
        for pattern in VOLUME_PATTERNS:
            match = pattern.match(user_input)
            if match:
                group = match.group(1).lower()
                
                # Determine operation and value
                if group.isdigit():
                    # "set volume to 50"
                    operation = "set"
                    value = int(group)
                    if not 0 <= value <= 100:
                        continue  # Invalid value, try next pattern
                elif group in ("increase", "raise", "up"):
                    operation = "increase"
                    value = None
                elif group in ("decrease", "lower", "down"):
                    operation = "decrease"
                    value = None
                elif group == "mute":
                    operation = "mute"
                    value = None
                elif group == "unmute":
                    operation = "unmute"
                    value = None
                else:
                    continue
                
                # Create task
                self._task_counter += 1
                logger.info(f"✅ Volume pattern matched: {operation}")
                
                return Task(
                    id=f"fast-volume-{self._task_counter}",
                    name=f"Volume Control: {operation}",
                    description=f"Control system volume: {operation}",
                    type=AgentType.SYSTEM,
                    payload={
                        "action": "volume_control",
                        "operation": operation,
                        "value": value
                    }
                )
        
        return None
    
    def _try_brightness_pattern(self, user_input: str) -> Optional[Task]:
        """
        Try to match brightness control patterns (Phase 5).
        
        Examples:
        - "set brightness to 80"
        - "increase brightness"
        
        Args:
            user_input: Normalized user input
            
        Returns:
            Task for SystemAgent or None
        """
        for pattern in BRIGHTNESS_PATTERNS:
            match = pattern.match(user_input)
            if match:
                group = match.group(1).lower()
                
                # Determine operation and value
                if group.isdigit():
                    # "set brightness to 80"
                    operation = "set"
                    value = int(group)
                    if not 0 <= value <= 100:
                        continue  # Invalid value, try next pattern
                elif group in ("increase", "raise", "up"):
                    operation = "increase"
                    value = None
                elif group in ("decrease", "lower", "down"):
                    operation = "decrease"
                    value = None
                else:
                    continue
                
                # Create task
                self._task_counter += 1
                logger.info(f"✅ Brightness pattern matched: {operation}")
                
                return Task(
                    id=f"fast-brightness-{self._task_counter}",
                    name=f"Brightness Control: {operation}",
                    description=f"Control screen brightness: {operation}",
                    type=AgentType.SYSTEM,
                    payload={
                        "action": "brightness_control",
                        "operation": operation,
                        "value": value
                    }
                )
        
        return None
    
    def _extract_target(self, user_input: str) -> Optional[str]:
        """
        Extract target from user input using system command patterns.
        
        Args:
            user_input: Normalized user input
            
        Returns:
            Target string or None
        """
        for pattern in OPEN_PATTERNS:
            match = pattern.match(user_input)
            if match:
                target = match.group(1).strip()
                logger.debug(f"Pattern matched: '{target}'")
                return target
        
        return None
    
    def _classify_and_create_task(self, target: str) -> Optional[Task]:
        """
        Classify target and create appropriate Task.
        
        Classification logic (syntactic only):
        1. Check if it looks like a URL (http://, https://, www.)
        2. Check if it's a known site keyword
        3. Check if it's a known app alias
        4. Otherwise return None (fall back to LLM)
        
        Args:
            target: Extracted target string
            
        Returns:
            Task object or None
        """
        target_lower = target.lower()
        
        # Syntactic URL detection
        if self._looks_like_url(target):
            logger.info(f"Classified as URL: {target}")
            return self._create_url_task(target)
        
        # Known site keyword check
        if target_lower in KNOWN_SITES:
            logger.info(f"Classified as site keyword: {target}")
            return self._create_url_task(target)
        
        # Known app alias check
        if target_lower in KNOWN_APPS:
            logger.info(f"Classified as known app: {target}")
            return self._create_app_task(target)
        
        # Unknown target - fall back to LLM
        logger.debug(f"Unknown target '{target}' - returning None for LLM fallback")
        return None
    
    def _looks_like_url(self, target: str) -> bool:
        """Syntactic check if target looks like a URL"""
        target_lower = target.lower()
        return (
            target_lower.startswith("http://") or
            target_lower.startswith("https://") or
            target_lower.startswith("www.")
        )
    
    def _create_app_task(self, target: str) -> Task:
        """Create system task for app opening"""
        self._task_counter += 1
        
        return Task(
            id=f"fast-app-{self._task_counter}",
            name=f"Open {target}",
            description=f"Launch {target}",
            type=AgentType.SYSTEM,
            payload={
                "action": "open_app",
                "app": target
            }
        )
    
    def _create_url_task(self, target: str) -> Task:
        """Create system task for URL opening"""
        self._task_counter += 1
        
        return Task(
            id=f"fast-url-{self._task_counter}",
            name=f"Open {target}",
            description=f"Open {target} in browser",
            type=AgentType.SYSTEM,
            payload={
                "action": "open_url",
                "url": target
            },
            requires_network=False
        )
    
    def reset_counter(self):
        """Reset task counter (for testing)"""
        self._task_counter = 0
    
    def get_supported_commands(self) -> list[str]:
        """Get list of supported command patterns"""
        return [
            # Code generation
            "write <description> in <file.py>",
            "create <description> in <file.py>",
            "in <file.py> write <description>",
            # OS control (Phase 5)
            "set volume to <0-100>",
            "increase volume",
            "decrease volume",
            "mute",
            "unmute",
            "set brightness to <0-100>",
            "increase brightness",
            "decrease brightness",
            # System commands
            "open <app>",
            "launch <app>",
            "start <app>",
            "open <url>",
        ]
    
    def get_known_apps(self) -> set[str]:
        """Get set of known app aliases (for info only)"""
        return KNOWN_APPS.copy()
    
    def get_known_sites(self) -> set[str]:
        """Get set of known site keywords (for info only)"""
        return KNOWN_SITES.copy()