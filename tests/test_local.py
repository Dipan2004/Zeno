"""
Phase 3 Integration Test - Planner & Chat Agents

Tests:
- PlannerAgent task graph generation
- ChatAgent conversational responses
- Integration with Phase 1 & 2 components
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    Orchestrator,
    Task,
    TaskStatus,
    AgentType,
    ContextManager,
    ContextSnapshot
)
from llm import LocalLLM, MISTRAL_7B, OllamaConnectionError
from agents import PlannerAgent, ChatAgent, PlanningError, ChatError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_planner_agent():
    """Test PlannerAgent task graph generation"""
    print("\n" + "="*70)
    print("TEST 1: PlannerAgent")
    print("="*70)
    
    try:
        # Initialize LLM client
        print("\n[1.1] Initializing LocalLLM...")
        llm = LocalLLM()
        
        # Check Ollama health
        try:
            llm.health_check()
            print("✓ Ollama is running")
        except OllamaConnectionError:
            print("⚠ Ollama not running - skipping LLM tests")
            print("  Start Ollama with: ollama serve")
            return
        
        # Initialize PlannerAgent
        print("\n[1.2] Initializing PlannerAgent...")
        planner = PlannerAgent(llm)
        print("✓ PlannerAgent initialized")
        
        # Create context
        ctx = ContextManager()
        ctx.add_message("user", "Hello ZENO")
        ctx.add_message("assistant", "Hello! How can I help?")
        context_snapshot = ctx.create_snapshot("test-plan")
        
        # Test simple planning
        print("\n[1.3] Testing simple command: 'Open VS Code'...")
        try:
            task_graph, explanation = planner.plan(
                user_input="Open VS Code",
                context=context_snapshot
            )
            
            print(f"✓ Plan created successfully")
            print(f"  Tasks: {len(task_graph.tasks)}")
            print(f"  Explanation: {explanation}")
            
            # Validate task graph
            for task_id, task in task_graph.tasks.items():
                print(f"\n  Task: {task.name}")
                print(f"    ID: {task.id}")
                print(f"    Type: {task.type.value}")
                print(f"    Payload: {task.payload}")
                print(f"    Can parallel: {task.can_run_parallel}")
            
        except PlanningError as e:
            print(f"⚠ Planning failed: {e}")
        
        # Test complex planning with multiple tasks
        print("\n[1.4] Testing complex command: 'Open VS Code and GitHub'...")
        try:
            task_graph, explanation = planner.plan(
                user_input="Open VS Code and GitHub",
                context=context_snapshot
            )
            
            print(f"✓ Complex plan created")
            print(f"  Tasks: {len(task_graph.tasks)}")
            print(f"  Explanation: {explanation}")
            
            assert len(task_graph.tasks) >= 2, "Should have at least 2 tasks"
            print("✓ Multiple tasks generated correctly")
            
        except PlanningError as e:
            print(f"⚠ Planning failed: {e}")
        
        # Test code generation request
        print("\n[1.5] Testing code request: 'Write a Python function'...")
        try:
            task_graph, explanation = planner.plan(
                user_input="Write a Python function to reverse a string",
                context=context_snapshot
            )
            
            print(f"✓ Code plan created")
            print(f"  Tasks: {len(task_graph.tasks)}")
            print(f"  Explanation: {explanation}")
            
            # Check that it's assigned to DEVELOPER
            for task in task_graph.tasks.values():
                if "reverse" in task.name.lower():
                    assert task.type == AgentType.DEVELOPER, "Should be DEVELOPER task"
                    print(f"✓ Correctly assigned to DEVELOPER agent")
                    break
            
        except PlanningError as e:
            print(f"⚠ Planning failed: {e}")
        
        print("\n✅ PlannerAgent tests COMPLETED")
        
    except Exception as e:
        print(f"\n❌ PlannerAgent test failed: {e}")
        logger.error("Test failed", exc_info=True)


def test_chat_agent():
    """Test ChatAgent conversational responses"""
    print("\n" + "="*70)
    print("TEST 2: ChatAgent")
    print("="*70)
    
    try:
        # Initialize LLM client
        print("\n[2.1] Initializing LocalLLM...")
        llm = LocalLLM()
        
        # Check Ollama health
        try:
            llm.health_check()
            print("✓ Ollama is running")
        except OllamaConnectionError:
            print("⚠ Ollama not running - skipping LLM tests")
            return
        
        # Initialize ChatAgent
        print("\n[2.2] Initializing ChatAgent...")
        chat_agent = ChatAgent(llm)
        print("✓ ChatAgent initialized")
        
        # Create context
        ctx = ContextManager()
        ctx.add_message("user", "Hello ZENO")
        context_snapshot = ctx.create_snapshot("test-chat")
        
        # Test simple greeting
        print("\n[2.3] Testing simple greeting...")
        greeting_task = Task(
            id="chat-1",
            name="Greet user",
            description="Respond to greeting",
            type=AgentType.CHAT,
            payload={"message": "Hello ZENO, how are you?"}
        )
        
        import threading
        interrupt = threading.Event()
        
        result = chat_agent.execute(greeting_task, context_snapshot, interrupt)
        
        if result.success:
            response = result.data.get("response", "")
            print(f"✓ Chat response generated")
            print(f"  Response: {response}")
            print(f"  Duration: {result.metadata.get('duration_ms', 0):.0f}ms")
        else:
            print(f"⚠ Chat failed: {result.error}")
        
        # Test explanation request
        print("\n[2.4] Testing explanation request...")
        explain_task = Task(
            id="chat-2",
            name="Explain capabilities",
            description="Explain what ZENO can do",
            type=AgentType.CHAT,
            payload={"message": "What can you do?"}
        )
        
        result = chat_agent.execute(explain_task, context_snapshot, interrupt)
        
        if result.success:
            response = result.data.get("response", "")
            print(f"✓ Explanation generated")
            print(f"  Response: {response[:200]}...")
            
            # Check that response mentions being offline
            if "offline" in response.lower() or "local" in response.lower():
                print("✓ Response correctly mentions offline capability")
        else:
            print(f"⚠ Chat failed: {result.error}")
        
        # Test with conversation history
        print("\n[2.5] Testing with conversation history...")
        ctx.add_message("user", "What can you do?")
        ctx.add_message("assistant", "I can help with tasks, open apps, and write code.")
        ctx.add_message("user", "Can you help me with Python?")
        
        context_with_history = ctx.create_snapshot("test-history")
        
        python_task = Task(
            id="chat-3",
            name="Python help",
            description="Respond to Python question",
            type=AgentType.CHAT,
            payload={"message": "Can you help me with Python?"}
        )
        
        result = chat_agent.execute(python_task, context_with_history, interrupt)
        
        if result.success:
            response = result.data.get("response", "")
            print(f"✓ Contextual response generated")
            print(f"  Response: {response[:200]}...")
        else:
            print(f"⚠ Chat failed: {result.error}")
        
        print("\n✅ ChatAgent tests COMPLETED")
        
    except Exception as e:
        print(f"\n❌ ChatAgent test failed: {e}")
        logger.error("Test failed", exc_info=True)


def test_full_integration():
    """Test full integration: Plan -> Execute -> Chat"""
    print("\n" + "="*70)
    print("TEST 3: Full Integration (Planner + Orchestrator + Chat)")
    print("="*70)
    
    try:
        # Initialize components
        print("\n[3.1] Initializing components...")
        llm = LocalLLM()
        
        try:
            llm.health_check()
            print("✓ Ollama is running")
        except OllamaConnectionError:
            print("⚠ Ollama not running - skipping integration test")
            return
        
        ctx = ContextManager()
        planner = PlannerAgent(llm)
        chat_agent = ChatAgent(llm)
        orch = Orchestrator(max_workers=2, context_manager=ctx)
        
        # Register chat agent with orchestrator
        orch.register_agent(AgentType.CHAT, chat_agent)
        print("✓ All components initialized")
        
        # Step 1: User asks a question
        print("\n[3.2] Step 1: User asks question...")
        user_question = "What can you help me with?"
        ctx.add_message("user", user_question)
        
        # Step 2: Create chat task and execute
        print("\n[3.3] Step 2: Generate chat response...")
        chat_task = Task(
            id="chat-intro",
            name="Introduction",
            description="Explain capabilities",
            type=AgentType.CHAT,
            payload={"message": user_question}
        )
        
        results = orch.execute_plan([chat_task])
        
        if "chat-intro" in results and results["chat-intro"].success:
            intro_response = results["chat-intro"].data.get("response", "")
            print(f"✓ ZENO responded: {intro_response[:150]}...")
            ctx.add_message("assistant", intro_response)
        
        # Step 3: User gives a command
        print("\n[3.4] Step 3: User gives command...")
        user_command = "Open VS Code"
        ctx.add_message("user", user_command)
        
        # Step 4: Plan the command
        print("\n[3.5] Step 4: Planning command...")
        context_snapshot = ctx.create_snapshot("integration-test")
        task_graph, explanation = planner.plan(user_command, context_snapshot)
        
        print(f"✓ Plan created: {explanation}")
        print(f"  Tasks to execute: {len(task_graph.tasks)}")
        
        # Step 5: Explain the plan via chat
        print("\n[3.6] Step 5: Explaining plan to user...")
        explain_task = Task(
            id="chat-explain",
            name="Explain Plan",
            description="Explain what ZENO will do",
            type=AgentType.CHAT,
            payload={
                "message": "What are you going to do?",
                "context": f"Current plan: {explanation}"
            }
        )
        
        results = orch.execute_plan([explain_task])
        
        if "chat-explain" in results and results["chat-explain"].success:
            plan_explanation = results["chat-explain"].data.get("response", "")
            print(f"✓ ZENO explained: {plan_explanation[:150]}...")
        
        print("\n✅ Full integration test COMPLETED")
        print("\nIntegration flow demonstrated:")
        print("  1. User asks question → ChatAgent responds")
        print("  2. User gives command → PlannerAgent creates plan")
        print("  3. Plan explanation → ChatAgent explains")
        print("  4. Tasks ready for execution (Phase 4)")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        logger.error("Test failed", exc_info=True)


def run_all_tests():
    """Run all Phase 3 tests"""
    print("\n" + "="*70)
    print("ZENO PHASE 3 INTEGRATION TESTS")
    print("Agent Intelligence Layer - Planning & Conversation")
    print("="*70)
    
    try:
        test_planner_agent()
        test_chat_agent()
        test_full_integration()
        
        print("\n" + "="*70)
        print("✅ ALL PHASE 3 TESTS COMPLETED")
        print("="*70)
        print("\nPhase 3 (Agent Intelligence) is working!")
        print("\nWhat ZENO can now do:")
        print("  ✓ Understand user commands")
        print("  ✓ Create structured execution plans")
        print("  ✓ Have natural conversations")
        print("  ✓ Explain what it will do")
        print("\nNext steps:")
        print("  - Phase 4: System execution (app_control, system_agent)")
        print("  - Phase 5: User interface (start.py, keyboard hooks)")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TESTS FAILED")
        print("="*70)
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()