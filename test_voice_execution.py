#!/usr/bin/env python3
"""
Quick test to verify voice command execution pipeline.
Simulates voice input and checks if it routes and executes correctly.
"""

import sys
import logging
from pathlib import Path

# Setup Python path
sys.path.insert(0, str(Path(__file__).parent))

from zeno.core import FastRouter, Orchestrator, ContextManager, AgentType
from zeno.agents import SystemAgent

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_voice_commands():
    """Test voice command routing and execution"""
    
    print("=" * 70)
    print("ZENO Voice Command Execution Test")
    print("=" * 70)
    print()
    
    # Initialize components
    fast_router = FastRouter()
    ctx = ContextManager()
    orch = Orchestrator(context_manager=ctx, max_workers=2)
    sys_agent = SystemAgent()
    
    orch.register_agent(AgentType.SYSTEM, sys_agent)
    
    # Test commands
    test_commands = [
        "Open chrome",
        "increase volume",
        "reduce brightness",
        "set volume to 50",
        "open vscode",
        "create a python file",
    ]
    
    print("Testing FastRouter classification:")
    print("-" * 70)
    
    for cmd in test_commands:
        print(f"\n[INPUT] {cmd}")
        
        # Step 1: Route
        task = fast_router.try_route(cmd)
        
        if task:
            print(f"[ROUTE] ✓ Task created")
            print(f"        ID: {task.id}")
            print(f"        Name: {task.name}")
            print(f"        Agent: {task.type}")
            print(f"        Payload: {task.payload}")
            
            # Step 2: Execute
            if sys_agent.agent_type == task.type or task.type == AgentType.SYSTEM:
                print(f"[EXEC]  Attempting execution...")
                import threading
                interrupt_event = threading.Event()
                
                result = sys_agent.execute(task, ctx.create_snapshot("test"), interrupt_event)
                
                if result.success:
                    print(f"[SUCCESS] {result.data.get('message', 'Done')}")
                else:
                    print(f"[FAILED] {result.error}")
            else:
                print(f"[WARN]  Task type {task.type} - would route to different agent")
        else:
            print(f"[ROUTE] ✗ FastRouter returned None - would use LLM")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_voice_commands()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
