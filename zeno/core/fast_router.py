"""
ZENO Fast Router - Enhanced with Code Generation + OS Control + File Creation Detection

Detects system commands, code generation, file creation AND OS control tasks
BEFORE calling PlannerAgent (LLM).

Performance Goal: < 50ms, ZERO LLM tokens for obvious tasks

Phase 5 Updates:
- Volume control pattern detection
- Brightness control pattern detection
- Correct priority order to avoid conflicts

Phase 6 Updates:
- File creation patterns (create a cpp file, make a python file, new js file, etc.)
- Extended CODE_PATTERNS to cover .cpp / .c / .js / .java / .rs / .go in addition to .py

Phase 6.5 Updates:
- Chat intent detection (looks_like_chat) - bypasses planner for conversational queries

Architectural Rules:
- Pure classifier only
- NO execution logic
- NO system tool calls
- Returns Task or None

Supported patterns (priority order):
0. File creation  → DeveloperAgent  (NEW Phase 6)
1. Code generation → DeveloperAgent
2. Volume control → SystemAgent
3. Brightness     → SystemAgent
4. App / URL open → SystemAgent
5. Chat detection → ChatAgent (NEW Phase 6.5)
6. PlannerAgent fallback (return None)
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
# CHAT INTENT DETECTION (HYBRID GATE)
# ============================================================================

# Task verbs that indicate non-chat intents
TASK_VERBS = {
    "create", "build", "generate", "make", "write", "code", "open", "launch",
    "start", "run", "execute", "compile", "install", "download", "upload",
    "delete", "remove", "move", "copy", "rename", "edit", "modify", "update",
    "search", "find", "list", "show", "display", "print", "save", "load",
    "set", "get", "increase", "decrease", "mute", "unmute", "close", "kill",
}


def looks_like_chat(text: str) -> bool:
    """
    Lightweight heuristic to detect conversational chat vs task requests.
    
    NO LLM calls. Pure pattern matching.
    
    Returns True if input looks like casual chat that should go to ChatAgent.
    Returns False if input looks like it needs planning/execution.
    
    Heuristics:
    - Short inputs (≤4 words) without task verbs → likely chat
    - Question patterns without task verbs → likely chat
    - Conversational greetings/responses → chat
    - Contains task verbs → NOT chat
    - File/code-related keywords → NOT chat
    
    Examples returning True (chat):
        "hello", "hi there", "what's up", "how are you",
        "tell me about arrays", "explain bfs", "what is python"
    
    Examples returning False (task):
        "create python file", "write bfs code", "open chrome",
        "generate backend architecture"
    """
    if not text or not text.strip():
        return False
    
    normalized = text.strip().lower()
    words = normalized.split()
    word_count = len(words)
    
    # Check for task verbs (strong indicator of non-chat intent)
    if any(verb in words for verb in TASK_VERBS):
        return False
    
    # Check for file/code indicators (non-chat)
    code_indicators = ["file", ".py", ".js", ".cpp", ".c", ".java", ".go", ".rs"]
    if any(indicator in normalized for indicator in code_indicators):
        return False
    
    # Short inputs without task verbs are likely chat
    if word_count <= 4:
        return True
    
    # Question patterns (even if longer) are likely chat if no task verbs
    question_starters = ["what", "why", "how", "when", "where", "who", "which", "can you explain", "tell me", "do you"]
    if any(normalized.startswith(starter) for starter in question_starters):
        return True
    
    # Common greetings/conversational phrases
    greetings = ["hello", "hi", "hey", "yo", "sup", "good morning", "good evening"]
    if any(greeting in normalized for greeting in greetings):
        return True
    
    # If we reach here with short text, lean toward chat
    if word_count <= 6:
        return True
    
    # Longer text without clear indicators → fall through to planner (safer)
    return False


# ============================================================================
# COMMAND PATTERNS (PRIORITY ORDER MATTERS!)
# ============================================================================

# Language → extension mapping (used by both file and code patterns)
LANGUAGE_MAP = {
    "python": ".py",
    "py":     ".py",
    "javascript": ".js",
    "js":     ".js",
    "java":   ".java",
    "cpp":    ".cpp",
    "c++":    ".cpp",
    "c":      ".c",
    "rust":   ".rs",
    "rs":     ".rs",
    "go":     ".go",
    "golang": ".go",
    "html":   ".html",
    "text":   ".txt",
    "txt":    ".txt",
}

# Extension → language name (for payload)
EXT_TO_LANG = {v: k for k, v in LANGUAGE_MAP.items()}
EXT_TO_LANG.update({
    ".py":   "python",
    ".js":   "javascript",
    ".cpp":  "cpp",
    ".c":    "c",
    ".rs":   "rust",
    ".go":   "go",
    ".java": "java",
})

# Phase 6: File creation patterns (HIGHEST PRIORITY — no code, just create the file)
# Matches: "create a cpp file", "make a python file", "new js file", "generate file"
FILE_PATTERNS = [
    # "create a cpp file" / "make a python file" / "generate a js file" / "build a c file"
    re.compile(
        r"^(?:create|make|generate|build)\s+(?:a\s+|an\s+)?(\w+)\s+file$",
        re.IGNORECASE,
    ),
    # "new python file" / "new cpp file"
    re.compile(
        r"^new\s+(\w+)\s+file$",
        re.IGNORECASE,
    ),
    # "create file" / "make file" / "generate file" (language-less → default .py)
    re.compile(
        r"^(?:create|make|generate|build)\s+(?:a\s+|an\s+)?file$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^new\s+file$",
        re.IGNORECASE,
    ),
]

# Code generation patterns (SECOND PRIORITY - Phase 4, extended to all extensions)
CODE_PATTERNS = [
    # "write <code_description> in <filename>" — any extension
    re.compile(
        r"^(?:write|create|generate)\s+(?:a\s+|an\s+)?(.+?)\s+(?:in|to|into)\s+(.+\.\w+)$",
        re.IGNORECASE,
    ),
    # "write <filename> with <code_description>" — any extension
    re.compile(
        r"^(?:write|create|generate)\s+(.+\.\w+)\s+(?:with|containing|for)\s+(.+)$",
        re.IGNORECASE,
    ),
    # "in <filename> write <code_description>" — any extension
    re.compile(
        r"^in\s+(.+\.\w+)\s+(?:write|create|add)\s+(?:a\s+|an\s+)?(.+)$",
        re.IGNORECASE,
    ),
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
    Phase 6.5 Enhanced: Now detects chat intent.
    
    Bypasses LLM for obvious system actions and code requests.
    Does NOT execute - only classifies and creates Tasks.
    """
    
    def __init__(self):
        """Initialize fast router"""
        self._task_counter = 0
        logger.info("FastRouter initialized (Phase 6.5: file creation + OS control + chat gate)")
    
    def try_route(self, user_input: str) -> Optional[Task]:
        """
        Attempt to classify user input as a fast-path task.
        
        Classification order (LOCKED):
        0. File creation patterns (Phase 6)
        1. Code generation
        2. Volume control
        3. Brightness control
        4. App/URL open
        5. PlannerAgent fallback (return None)
        
        Args:
            user_input: Raw user command
            
        Returns:
            Task object if classified, None if no match
        """
        if not user_input or not user_input.strip():
            return None
        
        # Normalize input: strip whitespace and trailing punctuation
        normalized = user_input.strip().rstrip('.!?;:,')
        
        # PRIORITY 0: File creation patterns (Phase 6)
        file_task = self._try_file_pattern(normalized)
        if file_task:
            return file_task
        
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
        
        # No match - fall back to chat gate or planner
        return None
    
    def _try_file_pattern(self, user_input: str) -> Optional[Task]:
        """
        Phase 6: Match file-creation patterns (no code generation, just create the file).

        Examples:
        - "create a cpp file"   → DeveloperAgent create_file, hello.cpp
        - "make a python file"  → DeveloperAgent create_file, hello.py
        - "new js file"         → DeveloperAgent create_file, hello.js
        - "create a file"       → DeveloperAgent create_file, hello.py (default)
        - "new file"            → DeveloperAgent create_file, hello.py (default)

        Args:
            user_input: Normalized user input

        Returns:
            Task for DeveloperAgent (intent=create_file) or None
        """
        # Language-less patterns (last two in FILE_PATTERNS)
        # Check index 2 and 3 first so we know if there's no language token
        for pattern in FILE_PATTERNS[2:]:    # "create a file", "new file"
            if pattern.match(user_input):
                # Default to Python
                self._task_counter += 1
                logger.info("✅ File pattern matched (no language): default python")
                return Task(
                    id=f"fast-file-{self._task_counter}",
                    name="Create new file",
                    description="Create a new Python file",
                    type=AgentType.DEVELOPER,
                    payload={
                        "intent": "create_file",
                        "filename": "new_file.py",
                        "description": "New Python file",
                    },
                )

        # Language-bearing patterns (first two in FILE_PATTERNS)
        for pattern in FILE_PATTERNS[:2]:    # "create a <lang> file", "new <lang> file"
            match = pattern.match(user_input)
            if match:
                lang_token = match.group(1).lower()
                extension  = LANGUAGE_MAP.get(lang_token)

                if extension is None:
                    # Unknown language token — let LLM handle it
                    logger.debug(
                        "File pattern matched but unknown language token '%s' — falling through",
                        lang_token,
                    )
                    return None

                language = EXT_TO_LANG.get(extension, lang_token)
                filename = f"new_file{extension}"

                self._task_counter += 1
                logger.info(
                    "✅ File pattern matched: %s → %s", lang_token, filename
                )
                return Task(
                    id=f"fast-file-{self._task_counter}",
                    name=f"Create new {language} file",
                    description=f"Create a new {language} file",
                    type=AgentType.DEVELOPER,
                    payload={
                        "intent": "create_file",
                        "filename": filename,
                        "description": f"New {language} file",
                    },
                )

        return None

    def _try_code_pattern(self, user_input: str) -> Optional[Task]:
        """
        Try to match code generation patterns (Phase 4, extended in Phase 6).

        Examples:
        - "write bfs in route.py"
        - "create sorting algorithm in sort.cpp"
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

                # Detect which group holds the filename (has a dot extension)
                def has_extension(s: str) -> bool:
                    return bool(re.search(r'\.\w+$', s))

                if has_extension(groups[0]):
                    filename        = groups[0]
                    code_description = groups[1] if len(groups) > 1 else "code"
                else:
                    code_description = groups[0]
                    filename        = groups[1] if len(groups) > 1 else "output.py"

                logger.info(f"✅ Code pattern matched: {code_description} → {filename}")
                return self._create_code_task(code_description, filename)

        return None
    
    def _create_code_task(self, code_description: str, filename: str) -> Task:
        """
        Create task with payload matching DeveloperAgent expectations.
        Handles all extensions, not just .py.
        """
        self._task_counter += 1

        # Extract extension and map to language name
        import os as _os
        ext      = _os.path.splitext(filename)[1].lower()
        language = EXT_TO_LANG.get(ext, "python")   # default python

        return Task(
            id=f"fast-code-{self._task_counter}",
            name=f"Generate {code_description}",
            description=f"Write {code_description} in {filename}",
            type=AgentType.DEVELOPER,
            payload={
                "intent": "generate_code",
                "description": f"Write {code_description}",
                "language": language,
                "filename": filename,
            },
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
            # File creation (Phase 6)
            "create a <language> file",
            "make a <language> file",
            "new <language> file",
            "create a file",
            # Code generation
            "write <description> in <file>",
            "create <description> in <file>",
            "in <file> write <description>",
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