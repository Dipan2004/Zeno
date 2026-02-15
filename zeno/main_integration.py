"""
ZENO Phase 5 - Main Integration Example

This file shows how to integrate all Phase 5 components:
- VoiceInputManager (STT)
- VoiceOutputManager (TTS)
- Enhanced FastRouter
- Enhanced SystemAgent
- Volume/Brightness control

Add this code to your existing main.py or create a new entry point.
"""

import logging
import keyboard
from pathlib import Path

# Assuming your ZENO structure:
# from zeno.voice.voice_input import VoiceInputManager
# from zeno.voice.voice_output import VoiceOutputManager
# from zeno.routing.fast_router import FastRouter
# from zeno.core.orchestrator import Orchestrator
# from zeno.agents.system_agent import SystemAgent
# from zeno.agents.planner_agent import PlannerAgent
# etc.

# For this example, using relative imports
from voice_input import VoiceInputManager
from voice_output import VoiceOutputManager

logger = logging.getLogger(__name__)


class ZENOPhase5:
    """
    ZENO Phase 5 - Voice-Enabled Assistant
    
    Integrates:
    - Push-to-talk voice input
    - Voice output (TTS)
    - OS control (volume/brightness)
    - Existing text pipeline
    """
    
    def __init__(self, vosk_model_path: Path):
        """
        Initialize ZENO Phase 5.
        
        Args:
            vosk_model_path: Path to Vosk model directory
        """
        logger.info("Initializing ZENO Phase 5...")
        
        # Initialize existing components (Phase 4)
        # self.orchestrator = Orchestrator(...)
        # self.fast_router = FastRouter()
        # self.planner_agent = PlannerAgent(...)
        # self.system_agent = SystemAgent()
        # etc.
        
        # Initialize voice output (Step 2)
        self.voice_output = VoiceOutputManager(rate=175)
        
        # Initialize voice input (Step 3)
        self.voice_input = VoiceInputManager(
            model_path=vosk_model_path,
            callback=self._on_voice_input
        )
        
        # Register hotkey (Ctrl+Space)
        self._register_hotkey()
        
        logger.info("ZENO Phase 5 initialized")
    
    def _register_hotkey(self):
        """Register Ctrl+Space hotkey for push-to-talk"""
        try:
            keyboard.add_hotkey('ctrl+space', self._on_hotkey_pressed)
            logger.info("Registered Ctrl+Space hotkey")
        except Exception as e:
            logger.error(f"Failed to register hotkey: {e}", exc_info=True)
            logger.warning("Voice input will not work without hotkey")
    
    def _on_hotkey_pressed(self):
        """
        Called when user presses Ctrl+Space.
        
        Starts voice listening.
        """
        logger.info("Push-to-talk activated (Ctrl+Space)")
        
        if not self.voice_input.is_listening():
            self.voice_input.start_listening()
        else:
            logger.warning("Already listening")
    
    def _on_voice_input(self, text: Optional[str]):
        """
        Callback for voice input transcription.
        
        Args:
            text: Transcribed text (None if error occurred)
        """
        if text is None:
            # Error occurred in STT
            logger.error("Voice input error")
            self.voice_output.speak(
                "Sorry, I couldn't hear you clearly.",
                priority=True
            )
            return
        
        logger.info(f"Voice input received: {text}")
        
        # Feed to existing text processing pipeline
        self.process_user_input(text)
    
    def process_user_input(self, user_input: str):
        """
        Process user input (text or voice).
        
        This is your existing input processing function with TTS added.
        
        Args:
            user_input: User's command (from typing or voice)
        """
        logger.info(f"Processing input: {user_input}")
        
        # Step 1: Try FastRouter (zero LLM)
        task = self.fast_router.try_route(user_input)
        
        if task is None:
            # Step 2: Fall back to PlannerAgent (LLM)
            logger.info("FastRouter returned None - using PlannerAgent")
            # tasks = self.planner_agent.plan(user_input)
            # ... existing planner logic ...
            
            # For this example, just acknowledge
            self.voice_output.speak("I'm working on that.")
            return
        
        # Step 3: Execute task via Orchestrator
        logger.info(f"Executing task: {task.name}")
        
        try:
            # This is your existing orchestrator execution
            # results = self.orchestrator.execute_plan([task])
            # result = results.get(task.id)
            
            # For this example, simulate execution
            result = self._simulate_task_execution(task)
            
            # Step 4: Speak result (Phase 5 enhancement)
            if result.success:
                message = self._format_response_for_speech(result)
                self.voice_output.speak(message)
            else:
                # Check if error should be spoken
                if result.metadata.get("user_facing"):
                    # User-facing error - speak it
                    self.voice_output.speak(result.error, priority=True)
                else:
                    # Internal error - just log
                    logger.error(f"Task failed: {result.error}")
        
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            self.voice_output.speak("Sorry, something went wrong.", priority=True)
    
    def _simulate_task_execution(self, task):
        """
        Simulate task execution for example.
        
        In real code, this is done by Orchestrator + SystemAgent.
        """
        from dataclasses import dataclass
        from typing import Any, Dict, Optional
        
        @dataclass
        class TaskResult:
            success: bool
            data: Dict[str, Any] = None
            error: Optional[str] = None
            metadata: Dict[str, Any] = None
        
        # Simulate success
        return TaskResult(
            success=True,
            data=task.payload,
            metadata={}
        )
    
    def _format_response_for_speech(self, result) -> str:
        """
        Convert TaskResult to speakable text.
        
        Args:
            result: TaskResult from task execution
            
        Returns:
            Human-friendly text for TTS
        """
        if not result.data:
            return "Done"
        
        # Check for pre-formatted message
        message = result.data.get("message")
        if message:
            return message
        
        # Format based on action type
        action = result.data.get("action")
        
        if action == "open_app":
            app = result.data.get("app", "application")
            return f"Opened {app}"
        
        elif action == "open_url":
            url = result.data.get("url", "website")
            return f"Opened {url}"
        
        elif action == "volume_control":
            operation = result.data.get("operation")
            value = result.data.get("value")
            
            if operation == "set":
                return f"Set volume to {value} percent"
            elif operation == "increase":
                return "Increased volume"
            elif operation == "decrease":
                return "Decreased volume"
            elif operation == "mute":
                return "Muted"
            elif operation == "unmute":
                return "Unmuted"
        
        elif action == "brightness_control":
            operation = result.data.get("operation")
            value = result.data.get("value")
            
            if operation == "set":
                return f"Set brightness to {value} percent"
            elif operation == "increase":
                return "Increased brightness"
            elif operation == "decrease":
                return "Decreased brightness"
        
        elif action == "create_file":
            filename = result.data.get("filename", "file")
            return f"Created {filename}"
        
        # Default fallback
        return "Done"
    
    def shutdown(self):
        """Clean shutdown of ZENO Phase 5"""
        logger.info("Shutting down ZENO Phase 5...")
        
        # Shutdown voice components
        self.voice_input.shutdown()
        self.voice_output.shutdown()
        
        # Unregister hotkey
        try:
            keyboard.remove_hotkey('ctrl+space')
        except:
            pass
        
        logger.info("ZENO Phase 5 shutdown complete")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for ZENO Phase 5 CLI.
    
    Supports both text and voice input.
    """
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Path to Vosk model
    # Download from: https://alphacephei.com/vosk/models
    vosk_model_path = Path("models/vosk-model-small-en-us-0.15")
    
    if not vosk_model_path.exists():
        logger.error(f"Vosk model not found at {vosk_model_path}")
        logger.error("Download from: https://alphacephei.com/vosk/models")
        sys.exit(1)
    
    # Initialize ZENO
    zeno = ZENOPhase5(vosk_model_path)
    
    print("=" * 60)
    print("ZENO Phase 5 - Voice-Enabled AI Assistant")
    print("=" * 60)
    print()
    print("Voice Input: Press Ctrl+Space and speak")
    print("Text Input: Type your command and press Enter")
    print()
    print("Commands:")
    print("  - open <app>         : Open application")
    print("  - set volume to 50   : Set volume")
    print("  - increase brightness: Increase brightness")
    print("  - write bfs in route.py: Generate code")
    print("  - quit               : Exit ZENO")
    print()
    print("=" * 60)
    print()
    
    try:
        # Main loop
        while True:
            try:
                # Get text input
                user_input = input("ZENO> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                # Process input
                zeno.process_user_input(user_input)
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"Error: {e}")
    
    finally:
        # Clean shutdown
        zeno.shutdown()


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
TEXT COMMANDS (Step 1):
  ZENO> set volume to 80
  [Volume changes to 80%]
  [TTS: "Set volume to 80 percent"]

  ZENO> increase brightness
  [Brightness increases by 10%]
  [TTS: "Increased brightness"]

VOICE COMMANDS (Step 3):
  1. Press Ctrl+Space
  2. Say "open chrome"
  3. Chrome opens
  4. TTS: "Opened Google Chrome"

  1. Press Ctrl+Space
  2. Say "set volume to fifty"
  3. Volume changes to 50%
  4. TTS: "Set volume to 50 percent"

VOICE + CODE GENERATION:
  1. Press Ctrl+Space
  2. Say "write BFS algorithm in route.py"
  3. File created
  4. TTS: "Created route.py"
"""