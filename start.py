"""
ZENO Start - Main Entry Point with Full Phase 6 Support

This integrates:
- Phase 1: Core Runtime (Orchestrator, TaskGraph)
- Phase 2: Local LLM (Ollama)
- Phase 3: Intelligence (PlannerAgent, ChatAgent)
- Phase 3.5: Passive Reminders
- Phase 4: System Execution (SystemAgent, DeveloperAgent)
- Phase 5: Voice I/O (Push-to-Talk STT, TTS) + OS Control (Volume, Brightness)
- Phase 6: Controlled Autonomy (AutonomyController) + Hybrid LLM (Kimi K2.5 + Ollama)
- Phase 6.5: Chat Gate (lightweight intent detection, bypasses planner for chat)

Phase 6.5 additions:
- looks_like_chat() heuristic bypasses PlannerAgent for conversational queries
- Saves Kimi API calls for simple questions like "hello", "what is bfs"
- ChatAgent uses LocalLLM directly (no Kimi, fast responses)
- All Phase 1-6 behavior preserved exactly
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

from zeno.core import ContextManager, Orchestrator, AgentType, Task, FastRouter
from zeno.core.fast_router import looks_like_chat
from zeno.llm import LocalLLM, QWEN_3B_INSTRUCT
from zeno.agents import (
    ChatAgent,
    PlannerAgent,
    SystemAgent,
    DeveloperAgent
)
from zeno.agents.reminder_agent import ReminderAgent
from zeno.memory import ReminderStore
from zeno.tools import file_system

# Phase 6: Controlled Autonomy + Hybrid LLM
from zeno.core.autonomy_controller import AutonomyController
from zeno.llm.hybrid_llm import HybridLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



_VOICE_AVAILABLE  = False
_HOTKEY_AVAILABLE = False

try:
    from zeno.voice.voice_output import VoiceOutputManager
    from zeno.voice.voice_input  import VoiceInputManager
    _VOICE_AVAILABLE = True
    logger.info("Voice components available")
except ImportError:
    logger.warning(
        "Voice components not found — run: pip install faster-whisper pyaudio pyttsx3"
    )

try:
    import keyboard
    _HOTKEY_AVAILABLE = True
except ImportError:
    logger.warning("keyboard library not found — run: pip install keyboard")




def check_and_surface_reminders(
    reminder_agent: "ReminderAgent",
    chat_agent: "ChatAgent",
) -> bool:
    """
    Check for due reminders and surface them BEFORE processing new input.
    PASSIVE — only runs on ZENO activation, never autonomously.
    """
    try:
        due_reminders = reminder_agent.get_due_reminders()
        if not due_reminders:
            return False

        print("\n" + "=" * 70)
        print("⏰ REMINDERS")
        print("=" * 70)
        print("\nBefore we continue, you have reminders due:\n")

        for reminder in due_reminders:
            print(reminder_agent.format_reminder_for_user(reminder))
            print()
            action = input("  [D]ismiss or [K]eep for later? ").strip().lower()
            if action == "d":
                reminder_agent.mark_dismissed(reminder.id)
                print("  ✓ Dismissed")
            else:
                print("  ✓ Kept for later")
            print()

        print("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.error(f"Error checking reminders: {e}", exc_info=True)
        return False




def format_result_for_speech(result) -> Optional[str]:
    """
    Convert a TaskResult into a short, speakable sentence.
    Returns None when there is nothing useful to say.
    """
    if not result or not result.success or not result.data:
        return None

    # Prefer pre-formatted message set by the agent
    msg = result.data.get("message")
    if msg:
        return msg

    action = result.data.get("action")

    if action == "open_app":
        return f"Opened {result.data.get('app', 'application')}"

    if action == "open_url":
        url = result.data.get("url", "website")
        speakable = url.replace("https://", "").replace("http://", "").rstrip("/")
        return f"Opened {speakable}"

    if action == "volume_control":
        op  = result.data.get("operation")
        val = result.data.get("value")
        return {
            "set":      f"Volume set to {val} percent",
            "increase": "Volume increased",
            "decrease": "Volume decreased",
            "mute":     "Muted",
            "unmute":   "Unmuted",
        }.get(op)

    if action == "brightness_control":
        op  = result.data.get("operation")
        val = result.data.get("value")
        return {
            "set":      f"Brightness set to {val} percent",
            "increase": "Brightness increased",
            "decrease": "Brightness decreased",
        }.get(op)

    if action == "create_file":
        return f"Created {result.data.get('filename', 'file')}"

    if action == "create_directory":
        return f"Created folder {result.data.get('dirname', '')}"

    return None




def main():
    """
    Main ZENO entry point — Phase 4 logic preserved exactly,
    Phase 5 voice layered on top as an optional I/O channel,
    Phase 6 autonomy + hybrid LLM,
    Phase 6.5 chat gate for fast conversational responses.
    """
    print("=" * 70)
    print("ZENO - Local AI Assistant")
    print("Phase 6.5: Chat Gate + Controlled Autonomy + Hybrid LLM")
    print("=" * 70)
    print()

    
    try:
        logger.info("Initializing ZENO...")

        llm         = LocalLLM()
        ctx         = ContextManager()
        orch        = Orchestrator(context_manager=ctx, max_workers=2)
        fast_router = FastRouter()

        reminder_store = ReminderStore()
        reminder_agent = ReminderAgent(reminder_store)

        # Phase 6: HybridLLM wraps LocalLLM — ChatAgent stays on LocalLLM directly
        hybrid_llm = HybridLLM(llm)

        chat_agent      = ChatAgent(llm, reminder_agent=reminder_agent)   # unchanged
        planner         = PlannerAgent(hybrid_llm)    # Phase 6: Kimi for planning
        system_agent    = SystemAgent()
        developer_agent = DeveloperAgent(hybrid_llm)  # Phase 6: Kimi for code gen

        # Phase 6: AutonomyController wraps slow-path execution
        autonomy = AutonomyController(orch, planner)

        orch.register_agent(AgentType.CHAT,      chat_agent)
        orch.register_agent(AgentType.SYSTEM,    system_agent)
        orch.register_agent(AgentType.DEVELOPER, developer_agent)

        workspace = file_system.get_workspace_path()
        logger.info(f"Workspace: {workspace}")
        logger.info("ZENO core initialized successfully")

        print("✓ ZENO is ready")
        print(f"✓ Workspace: {workspace}")

    except Exception as e:
        logger.error(f"Failed to initialize ZENO: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running  (ollama serve)")
        print(f"  2. Model is pulled   (ollama pull {QWEN_3B_INSTRUCT})")
        return 1

    
    _voice_queue: Queue = Queue()

    
    voice_output: Optional["VoiceOutputManager"] = None
    voice_input:  Optional["VoiceInputManager"]  = None

    if _VOICE_AVAILABLE:
        try:
            voice_output = VoiceOutputManager(rate=175)
            print("✓ Voice output ready (TTS)")
        except Exception as e:
            logger.warning(f"TTS init failed: {e}")
            voice_output = None

        # Voice input callback — called from the STT monitor thread
        # Uses _voice_queue.put() which is thread-safe
        def _on_voice_transcription(text: Optional[str]):
            if text:
                print(f"\n[Voice] {text}")
                _voice_queue.put(text)          # thread-safe put
            else:
                logger.warning("STT returned None (worker error)")
                if voice_output:
                    voice_output.speak(
                        "Sorry, I couldn't hear that clearly.", priority=True
                    )

        try:
            voice_input = VoiceInputManager(
                callback=_on_voice_transcription,
                engine="whisper",
            )
            print("✓ Voice input ready (Whisper STT)")
        except Exception as e:
            logger.warning(f"STT init failed: {e}")
            voice_input = None

    
    if _HOTKEY_AVAILABLE and voice_input is not None:
        def _on_hotkey():
            if not voice_input.is_listening():
                print("\n[Voice] Listening... (speak now)")
                voice_input.start_listening()

        try:
            keyboard.add_hotkey("ctrl+space", _on_hotkey)
            print("✓ Push-to-talk ready (Ctrl+Space)")
        except Exception as e:
            logger.warning(f"Hotkey registration failed: {e}")

    
    print()
    print(f"  Voice output : {'enabled' if voice_output else 'disabled (pip install pyttsx3)'}")
    print(f"  Voice input  : {'enabled' if voice_input  else 'disabled (pip install faster-whisper pyaudio)'}")
    print()

    
    check_and_surface_reminders(reminder_agent, chat_agent)

    
    print("Commands:")
    print("  - Type your message  (e.g., 'open vscode', 'write a python function')")
    print("  - 'set volume to 70' / 'increase brightness'  (Phase 5 OS control)")
    print("  - 'reminders'        - List all reminders")
    print("  - 'workspace'        - Show workspace info")
    if voice_input:
        print("  - Ctrl+Space         - Push-to-talk (can be used repeatedly)")
    print("  - 'quit'             - Exit ZENO")
    print()

    

    def process_input(user_input: str):
        user_input = user_input.strip()
        if not user_input:
            return

        # Special commands (unchanged from Phase 4)
        if user_input.lower() in ["quit", "exit", "q"]:
            raise SystemExit(0)

        if user_input.lower() == "reminders":
            reminders = reminder_agent.list_reminders()
            if reminders:
                print(f"\n📋 You have {len(reminders)} reminders:\n")
                for r in reminders:
                    icon = "⏳" if r.status.value == "pending" else "✓"
                    print(f"  {icon} {r.title}  (due: {r.due_at.strftime('%Y-%m-%d %H:%M')})")
                print()
            else:
                print("\n📭 No reminders\n")
            return

        if user_input.lower() == "workspace":
            stats = file_system.get_workspace_stats()
            print(f"\n📁 Workspace: {stats['path']}")
            print(f"   Files: {stats['total_files']}")
            print(f"   Size:  {stats['total_size_mb']} MB\n")
            if stats["total_files"] > 0:
                files = file_system.list_workspace_files()
                print("   Recent files:")
                for f in files[-5:]:
                    print(f"   - {f.name}")
            print()
            return

        ctx.add_message("user", user_input)

        # Reminder detection (unchanged from Phase 4)
        if chat_agent._detect_reminder_request(user_input):
            print("\nZENO: Do you want me to remember this and remind you? (yes/no)")
            confirmation = input("You: ").strip().lower()
            if confirmation in ["yes", "y"]:
                context_snapshot = ctx.create_snapshot("reminder-creation")
                due_time = chat_agent.parse_reminder_time(user_input, context_snapshot)
                if due_time:
                    reminder_agent.create_reminder(
                        title=user_input,
                        due_at=due_time,
                        session_id="main-session",
                    )
                    print(f"\nZENO: ✓ I'll remind you: {user_input}")
                    print(f"       Due: {due_time.strftime('%A, %B %d at %I:%M %p')}\n")
                    if voice_output:
                        voice_output.speak(
                            f"Reminder set for {due_time.strftime('%B %d at %I %p')}"
                        )
                else:
                    print("\nZENO: I couldn't understand when you want to be reminded.\n")
            else:
                print("\nZENO: Okay, I won't create a reminder.\n")
            return

        # FAST PATH (Phase 4 + Phase 5 patterns)
        fast_task = fast_router.try_route(user_input)

        if fast_task:
            print(f"\nZENO: {fast_task.name}...\n")
            logger.info(f"Fast route: {fast_task.name}")

            try:
                results = orch.execute_plan([fast_task])
                result  = results.get(fast_task.id)

                if result and result.success:
                    print("✓ Done!\n")
                    if result.data and "message" in result.data:
                        print(f"  {result.data['message']}\n")
                    if voice_output:
                        speech = format_result_for_speech(result)
                        if speech:
                            voice_output.speak(speech)
                else:
                    error_msg = result.error if result else "Unknown error"
                    print(f"⚠  {error_msg}\n")
                    if voice_output and result and result.metadata.get("user_facing"):
                        voice_output.speak(error_msg, priority=True)

            except Exception as e:
                print(f"Error: {e}\n")
                logger.error(f"Fast route execution failed: {e}", exc_info=True)

            return

        # CHAT GATE — Phase 6.5: lightweight conversational intent detection
        # Bypasses PlannerAgent (and Kimi) for simple chat
        if looks_like_chat(user_input):
            logger.info("Chat gate: routing to ChatAgent (no planning needed)")
            context_snapshot = ctx.create_snapshot("chat")
            
            try:
                # ChatAgent uses LocalLLM directly (not HybridLLM)
                response = chat_agent.respond(user_input, context_snapshot)
                print(f"\nZENO: {response}\n")
                
                if voice_output:
                    voice_output.speak(response)
                
                ctx.add_message("assistant", response)
                return
                
            except Exception as e:
                logger.error(f"Chat failed: {e}", exc_info=True)
                print(f"Error: {e}\n")
                return

        # SLOW PATH — LLM planner → AutonomyController (Phase 6)
        print("\nZENO: Let me plan that...\n")
        context_snapshot = ctx.create_snapshot("planning")

        try:
            task_graph, explanation = planner.plan(user_input, context_snapshot)

            print(f"Plan: {explanation}\n")
            print("Executing...\n")

            # Phase 6: AutonomyController handles multi-step continuation
            # FastRouter tasks bypass this path entirely (no change there)
            results = autonomy.execute(task_graph, context_snapshot, user_input)

            success_count = sum(1 for r in results.values() if r.success)
            total_count   = len(results)

            if success_count == total_count:
                print("✓ All tasks completed successfully!\n")
                for task_id, result in results.items():
                    if result.data:
                        if "message" in result.data:
                            print(f"  {result.data['message']}")
                        if "path" in result.data:
                            print(f"  📄 File: {result.data['path']}")
                print()
                if voice_output and explanation:
                    short = explanation if len(explanation) <= 120 else explanation[:117] + "..."
                    voice_output.speak(short)
            else:
                print(f"⚠  {success_count}/{total_count} tasks completed\n")
                for task_id, result in results.items():
                    if not result.success:
                        # task_graph only has the first step's tasks; for follow-up
                        # steps the task object is not in task_graph, so guard safely
                        task_obj = task_graph.tasks.get(task_id)
                        name = task_obj.name if task_obj else task_id
                        print(f"  ✗ {name}: {result.error}")
                print()

            ctx.add_message("assistant", explanation)

        except Exception as e:
            print(f"Planning error: {e}\n")
            logger.error(f"Planning failed: {e}", exc_info=True)

    
    input_queue: Queue = Queue()
    input_ready_event = threading.Event()

    def _input_thread_worker():
        """Background thread that reads typed input non-blockingly"""
        while True:
            try:
                line = input("You: ").strip()
                input_queue.put(line)
                input_ready_event.set()
            except EOFError:
                # stdin closed (e.g., redirected, piped, or script ended)
                break
            except Exception as e:
                logger.debug(f"Input thread error: {e}")
                break

    # Start input thread as daemon (will die with main process)
    input_thread = threading.Thread(target=_input_thread_worker, daemon=True)
    input_thread.start()

    
    while True:
        try:
            
            try:
                voice_text = _voice_queue.get_nowait()   # thread-safe, non-blocking
                logger.info(f"Processing voice input: {voice_text}")
                process_input(voice_text)
                input_ready_event.clear()  # reset in case input was pending
                continue
            except Empty:
                pass  # nothing in queue, check typed input next

            
            try:
                # Wait up to 0.1 seconds for typed input
                # This allows voice queue to be checked ~10x per second
                user_input = input_queue.get(timeout=0.1)
                input_ready_event.clear()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break

                process_input(user_input)

            except Empty:
                # No typed input yet - loop back to check voice queue again
                pass

        except SystemExit:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            print(f"\nError: {e}\n")

    
    if voice_input is not None:
        try:
            voice_input.shutdown()
        except Exception:
            pass

    if voice_output is not None:
        try:
            voice_output.shutdown()
        except Exception:
            pass

    if _HOTKEY_AVAILABLE:
        try:
            keyboard.remove_all_hotkeys()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    exit(main())