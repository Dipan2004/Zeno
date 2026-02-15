"""
ZENO Developer Agent - Code Generation & File Writing

Responsibilities:
- Generate code using LLM
- Write code to disk in workspace
- Confirm file location to user
- Support follow-up requests (optimize, convert, etc.)

Does NOT:
- Execute code (deferred to future phase)
- Modify existing files without explicit request
- Make assumptions about file names or locations

This agent IS executable and inherits from Agent base class.
"""

import logging
import threading
import re
from typing import Optional

from zeno.core import Agent, Task, TaskResult, ContextSnapshot
from zeno.llm import LocalLLM, QWEN_3B_INSTRUCT, OllamaError
from zeno.tools import file_system

logger = logging.getLogger(__name__)


class DeveloperAgentError(Exception):
    """Base exception for developer agent errors"""
    pass


class DeveloperAgent(Agent):
    """
    Code generation agent with file writing capability.
    
    Inherits from Agent base class and implements execute() contract.
    Uses Qwen 2.5 3B Instruct for code generation.
    """
    
    # System prompt for code generation
    SYSTEM_PROMPT = """You are a code generation assistant. Generate clean, working code based on user requests.

RULES:
- Write complete, runnable code
- Include helpful comments
- Follow language best practices
- Keep code concise but clear
- Do NOT include explanations outside the code
- Do NOT use markdown code blocks (just return the code)

When generating code:
1. Understand the requirement
2. Choose appropriate data structures and algorithms
3. Write clean, readable code
4. Add brief inline comments where helpful

You are OFFLINE and cannot access external libraries not in standard library."""
    
    def __init__(self, llm_client: LocalLLM):
        """
        Initialize developer agent.
        
        Args:
            llm_client: LocalLLM instance for code generation
        """
        self.llm = llm_client
        logger.info("DeveloperAgent initialized")
    
    def execute(
        self,
        task: Task,
        context_snapshot: ContextSnapshot,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Execute a developer task (code generation).
        
        Args:
            task: Task with code generation requirements
            context_snapshot: Read-only conversation context
            interrupt_event: Signal for cooperative cancellation
            
        Returns:
            TaskResult with generated code and file path
            
        Expected task.payload format:
            {
                "intent": "generate_code",
                "description": "Write a function to reverse a string",
                "language": "python",
                "filename": "optional_override.py"  # optional
            }
        """
        logger.info(f"DeveloperAgent executing task: {task.id} - {task.name}")
        
        # Check for interruption
        if interrupt_event.is_set():
            logger.warning("Task interrupted before execution")
            return TaskResult(
                success=False,
                error="Task was interrupted"
            )
        
        # Validate payload
        if "intent" not in task.payload:
            return TaskResult(
                success=False,
                error="Task payload missing 'intent' field"
            )
        
        intent = task.payload["intent"]
        
        if intent == "generate_code":
            return self._handle_generate_code(task, context_snapshot, interrupt_event)
        elif intent == "create_file":
            return self._handle_create_file(task, context_snapshot, interrupt_event)
        else:
            return TaskResult(
                success=False,
                error=f"Unknown developer intent: {intent}"
            )
    
    def _handle_generate_code(
        self,
        task: Task,
        context_snapshot: ContextSnapshot,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle code generation request.
        
        Generates code and writes it to a file in workspace.
        """
        # Extract requirements
        description = task.payload.get("description", "")
        language = task.payload.get("language", "python").lower()
        custom_filename = task.payload.get("filename")
        
        if not description:
            return TaskResult(
                success=False,
                error="Missing 'description' field in payload"
            )
        
        # Check interruption before LLM call
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            # Generate code using LLM
            logger.info(f"Generating {language} code: {description[:50]}...")
            code = self._generate_code_with_llm(
                description=description,
                language=language,
                context_snapshot=context_snapshot
            )
            
            # Check interruption after LLM call
            if interrupt_event.is_set():
                return TaskResult(success=False, error="Interrupted")
            
            # Determine filename
            if custom_filename:
                filename = custom_filename
            else:
                filename = self._generate_filename(description, language)
            
            # Write to file
            logger.info(f"Writing code to file: {filename}")
            file_path = file_system.create_file(
                filename=filename,
                content=code,
                overwrite=False
            )
            
            logger.info(f"Code written successfully: {file_path}")
            
            return TaskResult(
                success=True,
                data={
                    "code": code,
                    "filename": filename,
                    "path": str(file_path),
                    "language": language,
                    "message": f"I've saved the {language} code to: {file_path}"
                },
                metadata={
                    "code_length": len(code),
                    "language": language
                }
            )
            
        except OllamaError as e:
            logger.error(f"LLM error during code generation: {e}")
            return TaskResult(
                success=False,
                error=f"LLM service error: {e}"
            )
        except file_system.FileExistsError as e:
            logger.error(f"File already exists: {e}")
            return TaskResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            return TaskResult(
                success=False,
                error=f"Code generation failed: {e}"
            )
    
    def _handle_create_file(
        self,
        task: Task,
        context_snapshot: ContextSnapshot,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Handle simple file creation without LLM code generation.
        
        Creates a file with optional content. If no content is provided,
        creates an empty file or a minimal template based on language.
        
        Expected payload:
            {
                "intent": "create_file",
                "filename": "hello.c",
                "content": "optional content",  # optional
                "description": "optional description"  # optional, used for template
            }
        """
        filename = task.payload.get("filename")
        content = task.payload.get("content", "")
        description = task.payload.get("description", "")
        
        if not filename:
            return TaskResult(
                success=False,
                error="Missing 'filename' field in payload"
            )
        
        # Check interruption
        if interrupt_event.is_set():
            return TaskResult(success=False, error="Interrupted")
        
        try:
            # If no content provided, generate a minimal template based on file extension
            if not content:
                content = self._get_minimal_template(filename, description)
            
            # Write to file
            logger.info(f"Creating file: {filename}")
            file_path = file_system.create_file(
                filename=filename,
                content=content,
                overwrite=False
            )
            
            logger.info(f"File created successfully: {file_path}")
            
            return TaskResult(
                success=True,
                data={
                    "filename": filename,
                    "path": str(file_path),
                    "message": f"I've created the file: {file_path}"
                },
                metadata={
                    "content_length": len(content)
                }
            )
            
        except file_system.FileExistsError as e:
            logger.error(f"File already exists: {e}")
            return TaskResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"File creation failed: {e}", exc_info=True)
            return TaskResult(
                success=False,
                error=f"File creation failed: {e}"
            )
    
    def _get_minimal_template(self, filename: str, description: str = "") -> str:
        """
        Get a minimal code template based on file extension.
        
        Args:
            filename: The filename with extension
            description: Optional description for comment
            
        Returns:
            Minimal template content
        """
        import os
        ext = os.path.splitext(filename)[1].lower()
        
        comment = f"// {description}" if description else f"// {filename}"
        
        templates = {
            ".c": f'{comment}\n\n#include <stdio.h>\n\nint main() {{\n    printf("Hello, World!\\n");\n    return 0;\n}}\n',
            ".cpp": f'{comment}\n\n#include <iostream>\n\nint main() {{\n    std::cout << "Hello, World!" << std::endl;\n    return 0;\n}}\n',
            ".py": f'# {description or filename}\n\nprint("Hello, World!")\n',
            ".js": f'{comment}\n\nconsole.log("Hello, World!");\n',
            ".java": f'{comment}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        System.out.println("Hello, World!");\n    }}\n}}\n',
            ".rs": f'{comment}\n\nfn main() {{\n    println!("Hello, World!");\n}}\n',
            ".go": f'// {description or filename}\npackage main\n\nimport "fmt"\n\nfunc main() {{\n    fmt.Println("Hello, World!")\n}}\n',
            ".html": f'<!DOCTYPE html>\n<html>\n<head>\n    <title>{description or filename}</title>\n</head>\n<body>\n    <h1>Hello, World!</h1>\n</body>\n</html>\n',
            ".txt": f'{description or filename}\n',
        }
        
        return templates.get(ext, f"// {filename}\n")

    def _generate_code_with_llm(
        self,
        description: str,
        language: str,
        context_snapshot: ContextSnapshot
    ) -> str:
        """
        Generate code using LLM.
        
        Args:
            description: What the code should do
            language: Programming language
            context_snapshot: Conversation context
            
        Returns:
            Generated code as string
        """
        # Build prompt
        prompt = self._build_code_prompt(description, language, context_snapshot)
        
        # Generate with LLM
        response = self.llm.generate(
            prompt=prompt,
            model=QWEN_3B_INSTRUCT,
            temperature=0.3,  # Lower for more deterministic code
            max_tokens=1000,
            timeout=50
        )
        
        # Clean response (remove markdown if present)
        code = self._clean_code_response(response.text, language)
        
        return code
    
    def _build_code_prompt(
        self,
        description: str,
        language: str,
        context_snapshot: ContextSnapshot
    ) -> str:
        """Build prompt for code generation"""
        prompt_parts = [self.SYSTEM_PROMPT]
        
        # Add recent conversation for context (last 3 messages)
        history = context_snapshot.conversation_history[-3:]
        if history:
            prompt_parts.append("\n\nRECENT CONTEXT:")
            for msg in history:
                role = msg['role']
                content = msg['content'][:100]
                prompt_parts.append(f"{role.upper()}: {content}")
        
        # Add current request
        prompt_parts.append(f"\n\nUSER REQUEST: {description}")
        prompt_parts.append(f"\nLANGUAGE: {language}")
        prompt_parts.append(f"\n\nGenerate the {language} code (code only, no explanations):")
        
        return "\n".join(prompt_parts)
    
    def _clean_code_response(self, response: str, language: str) -> str:
        """
        Clean LLM response to extract just the code.
        
        Removes markdown code blocks, extra explanations, etc.
        """
        cleaned = response.strip()
        
        # Remove markdown code blocks
        # Pattern: ```language\ncode\n```
        code_block_pattern = rf"```{language}?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, cleaned, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Use the first code block found
            cleaned = matches[0].strip()
        else:
            # Try generic code block
            generic_pattern = r"```\s*\n(.*?)\n```"
            matches = re.findall(generic_pattern, cleaned, re.DOTALL)
            if matches:
                cleaned = matches[0].strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here is the code:",
            "Here's the code:",
            "Code:",
            f"{language.capitalize()} code:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def _generate_filename(self, description: str, language: str) -> str:
        """
        Generate a reasonable filename from description.
        
        Args:
            description: Code description
            language: Programming language
            
        Returns:
            Filename like "reverse_string.py"
        """
        # Get file extension
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c",
            "rust": ".rs",
            "go": ".go"
        }
        ext = extensions.get(language, ".txt")
        
        # Extract key words from description
        # Remove common words and clean
        words = description.lower().split()
        stop_words = {"a", "an", "the", "to", "write", "create", "generate", "code", "function", "that", "for"}
        meaningful_words = [w for w in words if w not in stop_words and w.isalnum()]
        
        # Take first 2-3 words
        name_words = meaningful_words[:3]
        
        if not name_words:
            # Fallback
            name = "solution"
        else:
            name = "_".join(name_words)
        
        # Clean filename
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        
        return f"{name}{ext}"