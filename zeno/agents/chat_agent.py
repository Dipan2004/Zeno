"""
ZENO Chat Agent - Conversational Interface

Responsibilities:
- Generate natural, friendly conversational responses
- Explain plans and execution status
- Answer follow-up questions
- Maintain ZENO's personality and tone
- Detect and coordinate reminder requests (NO autonomous creation)

This agent IS executable and fits the Agent interface contract.
"""

import logging
import threading
from typing import Optional , Any
from datetime import datetime

from zeno.core import Agent, Task, TaskResult, ContextSnapshot, AgentType
from zeno.llm import LocalLLM, QWEN_3B_INSTRUCT, LLAMA_1B_INSTRUCT, OllamaError

logger = logging.getLogger(__name__)


class ChatError(Exception):
    """Raised when chat generation fails"""
    pass


class ChatAgent(Agent):
    """
    Conversational agent for natural language interaction.
    
    Inherits from Agent base class and implements execute() contract.
    Uses Qwen 2.5 3B Instruct for generating contextual, friendly responses.
    
    Also handles reminder request detection and coordination (but does NOT
    create reminders autonomously - only after explicit confirmation).
    """
    
    # Keywords that might indicate reminder request
    REMINDER_KEYWORDS = ["remind me", "remember this", "don't forget", "reminder"]
    
    # System prompt defining ZENO's personality and emotional intelligence
    SYSTEM_PROMPT = """You are ZENO, a helpful local AI assistant. You run entirely offline on the user's machine.

PERSONALITY:
- Warm and empathetic
- Calm and reassuring
- Clear and concise
- Honest about limitations
- Supportive without being overbearing

CORE CAPABILITIES:
- You can open applications
- You can generate code
- You can help with tasks
- You are OFFLINE - no internet access, no cloud services

CORE LIMITATIONS:
- You cannot access the internet
- You cannot send emails or messages
- You cannot guarantee task success (you plan, but execution may fail)
- You have no memory between sessions (currently)
- You are an AI and don't experience emotions yourself

EMOTIONAL INTELLIGENCE GUIDELINES:

When user expresses sadness, stress, or feeling overwhelmed:
- Acknowledge their feelings without dismissing them
- Validate that it's okay to feel this way
- Offer gentle support or ask how you can help
- Example: "That sounds really heavy. It's completely okay to feel like this sometimes. If you want, we can talk it through, or I can help with something to lighten the load."

When user expresses confusion or frustration:
- Stay patient and calm
- Break things down clearly
- Offer alternatives if something isn't working
- Example: "I can see this is frustrating. Let me try to explain it differently. We can take this step by step."

When user seems excited or positive:
- Match their energy appropriately
- Stay supportive and encouraging
- Example: "That's great! I'm here to help make it happen."

ANTI-PATTERNS TO AVOID:
- Don't use toxic positivity ("Just be happy!" or "Everything will be fine!")
- Don't pretend to feel emotions yourself ("I'm sad to hear that" - you don't feel sadness)
- Don't be overly verbose or therapeutic (you're an assistant, not a counselor)
- Don't be robotic or cold
- Don't minimize user's feelings

RESPONSE STYLE:
- Keep responses concise but warm (2-4 sentences usually)
- Use natural, conversational language
- Be helpful and actionable
- If you don't know something, say so honestly
- Never make up information

EXAMPLES:

User: "I'm feeling really stressed about work today"
ZENO: "That sounds tough. Sometimes it helps to break things down into smaller pieces. Would you like to talk through what's overwhelming you, or is there a specific task I can help with to ease the load?"

User: "I don't understand how this works"
ZENO: "No problem - let me explain it in a simpler way. What part is unclear? I can walk through it step by step."

User: "Can you help me write some Python code?"
ZENO: "Absolutely! I can help you with Python. What would you like to create?"

CONTEXT AWARENESS:
- You can see recent conversation history
- You may reference plans you've created
- You can explain what you're doing

Remember: You are a LOCAL assistant running on the user's computer. Be genuinely helpful, warm, and grounded."""
    
    def __init__(self, llm_client: LocalLLM, reminder_agent: Optional[Any] = None):
        """
        Initialize chat agent.
        
        Args:
            llm_client: LocalLLM instance for inference
            reminder_agent: Optional ReminderAgent for reminder coordination
        """
        self.llm = llm_client
        self.reminder_agent = reminder_agent  # Optional - can be set later
        logger.info("ChatAgent initialized")
    
    def execute(
        self,
        task: Task,
        context_snapshot: ContextSnapshot,
        interrupt_event: threading.Event
    ) -> TaskResult:
        """
        Execute chat task to generate conversational response.
        
        Args:
            task: Task with payload containing message to respond to
            context_snapshot: Read-only conversation context
            interrupt_event: Signal for cooperative cancellation
            
        Returns:
            TaskResult with response text in data["response"]
            
        Raises:
            ChatError: If response generation fails
            
        Expected task.payload format:
            {
                "message": "User's message or question",
                "context": "Optional additional context"
            }
        """
        logger.info(f"ChatAgent executing task: {task.id}")
        
        # Validate payload
        if "message" not in task.payload:
            error_msg = "Chat task payload must contain 'message' field"
            logger.error(error_msg)
            return TaskResult(
                success=False,
                error=error_msg
            )
        
        user_message = task.payload["message"]
        additional_context = task.payload.get("context", "")
        
        # Check if this might be a reminder request
        # Only do keyword detection - NO autonomous creation
        is_potential_reminder = self._detect_reminder_request(user_message)
        
        if is_potential_reminder and self.reminder_agent:
            logger.info("Detected potential reminder request - will need confirmation")
            # This will be handled by asking for confirmation
            # The actual reminder creation happens elsewhere after "yes"
        
        try:
            # Build prompt — lightweight for short inputs, full for longer ones
            word_count = len(user_message.split())
            if word_count <= 6 and not additional_context:
                prompt = (
                    "You are ZENO, a friendly AI assistant.\n"
                    f"User: {user_message}\n"
                    "Assistant:"
                )
                chat_model = LLAMA_1B_INSTRUCT
                logger.info(f"Using lightweight prompt ({len(prompt)} chars) with {chat_model}")
            else:
                prompt = self._build_chat_prompt(
                    user_message=user_message,
                    context_snapshot=context_snapshot,
                    additional_context=additional_context
                )
                chat_model = LLAMA_1B_INSTRUCT
                logger.info(f"Using full prompt ({len(prompt)} chars) with {chat_model}")
            
            # Check for interruption before LLM call
            if interrupt_event.is_set():
                logger.warning("Chat task interrupted before generation")
                return TaskResult(
                    success=False,
                    error="Task interrupted"
                )
            
            # Generate response
            logger.debug("Generating chat response...")
            response = self.llm.generate(
                prompt=prompt,
                model=chat_model,
                temperature=0.7,  # Higher for more natural, empathetic conversation
                max_tokens=300,
                timeout=60
            )
            
            # Check for interruption after LLM call
            if interrupt_event.is_set():
                logger.warning("Chat task interrupted after generation")
                return TaskResult(
                    success=False,
                    error="Task interrupted"
                )
            
            # Validate response
            response_text = response.text.strip()
            if not response_text:
                logger.error("LLM returned empty response")
                return TaskResult(
                    success=False,
                    error="Generated empty response"
                )
            
            # Clean response (remove any system artifacts)
            cleaned_response = self._clean_response(response_text)
            
            logger.info(
                f"Chat response generated: {len(cleaned_response)} chars, "
                f"duration: {response.duration_ms:.0f}ms"
            )
            
            return TaskResult(
                success=True,
                data={"response": cleaned_response},
                metadata={
                    "model": response.model,
                    "duration_ms": response.duration_ms,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens
                }
            )
            
        except OllamaError as e:
            logger.error(f"LLM error during chat: {e}")
            return TaskResult(
                success=False,
                error=f"LLM service error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in chat agent: {e}", exc_info=True)
            return TaskResult(
                success=False,
                error=f"Chat generation failed: {str(e)}"
            )
    
    def _build_chat_prompt(
        self,
        user_message: str,
        context_snapshot: ContextSnapshot,
        additional_context: str = ""
    ) -> str:
        """
        Build complete prompt including conversation history.
        
        Args:
            user_message: Current user message
            context_snapshot: Conversation context
            additional_context: Optional extra context (e.g., plan explanation)
            
        Returns:
            Complete prompt string
        """
        prompt_parts = [self.SYSTEM_PROMPT]
        
        # Add recent conversation history (last 5-7 messages)
        history = context_snapshot.conversation_history[-7:]
        if history:
            prompt_parts.append("\n\nCONVERSATION HISTORY:")
            for msg in history:
                role = msg['role']
                content = msg['content']
                
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"ZENO: {content}")
                elif role == 'system':
                    prompt_parts.append(f"[System: {content}]")
        
        # Add additional context if provided (e.g., current plan)
        if additional_context:
            prompt_parts.append(f"\n\nCURRENT CONTEXT:\n{additional_context}")
        
        # Add current user message
        prompt_parts.append(f"\n\nUser: {user_message}")
        prompt_parts.append("\nZENO:")
        
        return "\n".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response to remove artifacts.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned response text
        """
        cleaned = response.strip()
        
        # Remove common prefixes that LLM might add
        prefixes_to_remove = [
            "ZENO:",
            "Assistant:",
            "Response:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove markdown code block markers if present
        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 2:
                cleaned = "\n".join(lines[1:-1])
        
        return cleaned
    
    def _detect_reminder_request(self, message: str) -> bool:
        """
        Simple keyword-based detection of reminder requests.
        
        Args:
            message: User message
            
        Returns:
            True if message might be a reminder request
        """
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.REMINDER_KEYWORDS)
    
    def parse_reminder_time(self, user_input: str, context_snapshot: ContextSnapshot) -> Optional[datetime]:
        """
        Use LLM to parse time expression into datetime.
        
        This is called when user confirms they want a reminder.
        
        Args:
            user_input: Original user message with time reference
            context_snapshot: Conversation context
            
        Returns:
            Parsed datetime, or None if parsing fails
        """
        prompt = f"""Extract the due time from this reminder request. Today is {datetime.now().strftime('%A, %B %d, %Y')}.

User request: "{user_input}"

Return ONLY a datetime in ISO format (YYYY-MM-DDTHH:MM:SS), nothing else.
If time is not specified, use 09:00 (9 AM).
If only relative time given (like "tomorrow", "Friday"), use 09:00.

Examples:
"Remind me tomorrow" → {datetime.now().replace(hour=9, minute=0, second=0).isoformat()}
"Remind me Friday at 3pm" → [appropriate Friday date]T15:00:00
"Remind me in 2 days" → [date 2 days from now]T09:00:00

Respond with ONLY the ISO datetime, no explanation."""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                model=QWEN_3B_INSTRUCT,
                temperature=0.1,  # Very low for deterministic time parsing
                max_tokens=50,
                timeout=60
            )
            
            # Parse the ISO datetime from response
            time_str = response.text.strip()
            # Remove any quotes or extra text
            time_str = time_str.replace('"', '').replace("'", '').split()[0]
            
            parsed_time = datetime.fromisoformat(time_str)
            logger.info(f"Parsed reminder time: {parsed_time}")
            return parsed_time
            
        except Exception as e:
            logger.error(f"Failed to parse reminder time: {e}")
            return None
    
    def set_reminder_agent(self, reminder_agent):
        """
        Set the reminder agent for coordination.
        
        Args:
            reminder_agent: ReminderAgent instance
        """
        self.reminder_agent = reminder_agent
        logger.info("ReminderAgent linked to ChatAgent")
    
    def respond(
        self,
        message: str,
        context_snapshot: ContextSnapshot,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Convenience method for simple chat responses (non-task mode).
        
        This is a helper for testing or direct usage outside the orchestrator.
        
        Args:
            message: User message
            context_snapshot: Conversation context
            additional_context: Optional additional context
            
        Returns:
            Response text
            
        Raises:
            ChatError: If generation fails
        """
        # Create a temporary task
        task = Task(
            id="chat-direct",
            name="Direct Chat",
            description="Direct chat response",
            type=AgentType.CHAT,  # Need to import AgentType
            payload={
                "message": message,
                "context": additional_context or ""
            }
        )
        
        # Create a dummy interrupt event
        interrupt = threading.Event()
        
        # Execute
        result = self.execute(task, context_snapshot, interrupt)
        
        if not result.success:
            raise ChatError(f"Chat failed: {result.error}")
        
        return result.data.get("response", "")