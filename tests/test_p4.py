"""
Phase 4 Integration Tests - System Execution & Control

Tests:
- SystemAgent execution
- DeveloperAgent code generation
- App control (mocked)
- File system operations
- Full integration flow
"""

import sys
import logging
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

print("="*70)
print("Starting ZENO Phase 4 Tests...")
print("="*70)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n[IMPORT] Importing ZENO modules...")
try:
    from zeno.core import (
        Orchestrator,
        Task,
        TaskStatus,
        AgentType,
        ContextManager
    )
    from zeno.llm import LocalLLM, QWEN_3B_INSTRUCT
    from zeno.agents import SystemAgent, DeveloperAgent
    from zeno.tools import app_control, file_system
    print("✓ All imports successful!\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_file_system():
    """Test file system operations"""
    print("\n" + "="*70)
    print("TEST 1: File System Operations")
    print("="*70)
    
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override BASE_DIR for testing
        original_base = file_system.BASE_DIR
        file_system.BASE_DIR = Path(tmpdir) / "test_workspace"
        
        try:
            print("\n[1.1] Testing workspace creation...")
            file_system.ensure_workspace_exists()
            assert file_system.BASE_DIR.exists()
            print("✓ Workspace created")
            
            print("\n[1.2] Testing file creation...")
            content = "print('Hello, ZENO!')"
            file_path = file_system.create_file("test.py", content)
            assert file_path.exists()
            assert file_path.read_text() == content
            print(f"✓ File created: {file_path.name}")
            
            print("\n[1.3] Testing file exists check...")
            assert file_system.file_exists("test.py")
            assert not file_system.file_exists("nonexistent.py")
            print("✓ File existence check works")
            
            print("\n[1.4] Testing overwrite protection...")
            try:
                file_system.create_file("test.py", "new content", overwrite=False)
                assert False, "Should have raised FileExistsError"
            except file_system.FileExistsError:
                print("✓ Overwrite protection works")
            
            print("\n[1.5] Testing extension validation...")
            try:
                file_system.create_file("bad.exe", "content")
                assert False, "Should have raised InvalidExtensionError"
            except file_system.InvalidExtensionError:
                print("✓ Extension validation works")
            
            print("\n[1.6] Testing directory creation...")
            dir_path = file_system.create_directory("my_project")
            assert dir_path.exists()
            assert dir_path.is_dir()
            print("✓ Directory created")
            
            print("\n[1.7] Testing workspace stats...")
            stats = file_system.get_workspace_stats()
            assert stats['exists']
            assert stats['total_files'] >= 1
            print(f"✓ Stats: {stats['total_files']} files")
            
        finally:
            # Restore original BASE_DIR
            file_system.BASE_DIR = original_base
    
    print("\n✅ File system tests PASSED")


def test_app_control_mocked():
    """Test app control with mocked subprocess"""
    print("\n" + "="*70)
    print("TEST 2: App Control (Mocked)")
    print("="*70)
    
    print("\n[2.1] Testing platform detection...")
    platform = app_control.get_platform()
    assert platform in ["Windows", "Darwin", "Linux"]
    print(f"✓ Detected platform: {platform}")
    
    print("\n[2.2] Testing URL detection...")
    assert app_control.is_url("https://github.com")
    assert app_control.is_url("http://example.com")
    assert app_control.is_url("www.google.com")
    assert not app_control.is_url("vscode")
    print("✓ URL detection works")
    
    print("\n[2.3] Testing available apps list...")
    apps = app_control.get_available_apps()
    assert "vscode" in apps
    print(f"✓ Available apps: {', '.join(apps)}")
    
    print("\n[2.4] Testing app launch (mocked)...")
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.return_value = Mock()
        
        success = app_control.open_app("vscode")
        assert success
        assert mock_popen.called
        print("✓ App launch succeeded (mocked)")
    
    print("\n[2.5] Testing unknown app rejection...")
    try:
        app_control.open_app("unknownapp12345")
        assert False, "Should have raised AppNotFoundError"
    except app_control.AppNotFoundError as e:
        assert "unknownapp12345" in str(e)
        print("✓ Unknown app correctly rejected")
    
    print("\n[2.6] Testing URL opening (mocked)...")
    with patch('webbrowser.open') as mock_browser:
        mock_browser.return_value = True
        
        success = app_control.open_url("https://github.com")
        assert success
        assert mock_browser.called
        print("✓ URL opening succeeded (mocked)")
    
    print("\n[2.7] Testing invalid URL rejection...")
    try:
        app_control.open_url("ftp://invalid.com")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "http" in str(e).lower()
        print("✓ Invalid URL correctly rejected")
    
    print("\n✅ App control tests PASSED")


def test_system_agent():
    """Test SystemAgent execution"""
    print("\n" + "="*70)
    print("TEST 3: SystemAgent")
    print("="*70)
    
    system_agent = SystemAgent()
    ctx = ContextManager()
    interrupt = threading.Event()
    
    print("\n[3.1] Testing open_app task (mocked)...")
    with patch('zeno.tools.app_control.open_app') as mock_open:
        mock_open.return_value = True
        
        task = Task(
            id="test-1",
            name="Open VS Code",
            description="Test task",
            type=AgentType.SYSTEM,
            payload={"action": "open_app", "app": "vscode"}
        )
        
        context = ctx.create_snapshot("test")
        result = system_agent.execute(task, context, interrupt)
        
        assert result.success
        assert mock_open.called
        print("✓ open_app task executed")
    
    print("\n[3.2] Testing open_url task (mocked)...")
    with patch('zeno.tools.app_control.open_url') as mock_url:
        mock_url.return_value = True
        
        task = Task(
            id="test-2",
            name="Open GitHub",
            description="Test task",
            type=AgentType.SYSTEM,
            payload={"action": "open_url", "url": "https://github.com"}
        )
        
        context = ctx.create_snapshot("test")
        result = system_agent.execute(task, context, interrupt)
        
        assert result.success
        assert mock_url.called
        print("✓ open_url task executed")
    
    print("\n[3.3] Testing create_file task...")
    with tempfile.TemporaryDirectory() as tmpdir:
        original_base = file_system.BASE_DIR
        file_system.BASE_DIR = Path(tmpdir) / "test_workspace"
        
        try:
            task = Task(
                id="test-3",
                name="Create File",
                description="Test file creation",
                type=AgentType.SYSTEM,
                payload={
                    "action": "create_file",
                    "filename": "test.txt",
                    "content": "Hello from test"
                }
            )
            
            context = ctx.create_snapshot("test")
            result = system_agent.execute(task, context, interrupt)
            
            assert result.success
            assert "path" in result.data
            print(f"✓ create_file task executed: {result.data['filename']}")
            
        finally:
            file_system.BASE_DIR = original_base
    
    print("\n[3.4] Testing error handling (unknown action)...")
    task = Task(
        id="test-4",
        name="Unknown Action",
        description="Test error",
        type=AgentType.SYSTEM,
        payload={"action": "unknown_action"}
    )
    
    context = ctx.create_snapshot("test")
    result = system_agent.execute(task, context, interrupt)
    
    assert not result.success
    assert "Unknown action" in result.error
    print("✓ Error handling works")
    
    print("\n✅ SystemAgent tests PASSED")


def test_developer_agent():
    """Test DeveloperAgent with mocked LLM"""
    print("\n" + "="*70)
    print("TEST 4: DeveloperAgent (Mocked LLM)")
    print("="*70)
    
    # Create mock LLM
    mock_llm = Mock(spec=LocalLLM)
    
    print("\n[4.1] Testing code generation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        original_base = file_system.BASE_DIR
        file_system.BASE_DIR = Path(tmpdir) / "test_workspace"
        
        try:
            # Mock LLM response
            mock_response = Mock()
            mock_response.text = """def reverse_string(s):
    return s[::-1]"""
            mock_llm.generate.return_value = mock_response
            
            developer_agent = DeveloperAgent(mock_llm)
            
            task = Task(
                id="dev-1",
                name="Generate Code",
                description="Create reverse function",
                type=AgentType.DEVELOPER,
                payload={
                    "intent": "generate_code",
                    "description": "Write a function to reverse a string",
                    "language": "python"
                }
            )
            
            ctx = ContextManager()
            context = ctx.create_snapshot("test")
            interrupt = threading.Event()
            
            result = developer_agent.execute(task, context, interrupt)
            
            assert result.success
            assert "code" in result.data
            assert "path" in result.data
            assert mock_llm.generate.called
            print(f"✓ Code generated and saved: {result.data['filename']}")
            
            # Verify file was created
            file_path = Path(result.data['path'])
            assert file_path.exists()
            print(f"✓ File exists: {file_path.name}")
            
        finally:
            file_system.BASE_DIR = original_base
    
    print("\n[4.2] Testing filename generation...")
    developer_agent = DeveloperAgent(mock_llm)
    filename = developer_agent._generate_filename("reverse a string", "python")
    assert filename.endswith(".py")
    assert "_" in filename or filename == "solution.py"
    print(f"✓ Filename generated: {filename}")
    
    print("\n✅ DeveloperAgent tests PASSED")


def test_full_integration():
    """Test full integration with Orchestrator"""
    print("\n" + "="*70)
    print("TEST 5: Full Integration")
    print("="*70)
    
    print("\n[5.1] Setting up components...")
    ctx = ContextManager()
    orch = Orchestrator(context_manager=ctx)
    
    # Register agents
    system_agent = SystemAgent()
    orch.register_agent(AgentType.SYSTEM, system_agent)
    print("✓ SystemAgent registered")
    
    # Mock LLM for DeveloperAgent
    mock_llm = Mock(spec=LocalLLM)
    mock_response = Mock()
    mock_response.text = "def hello():\n    print('Hello!')"
    mock_llm.generate.return_value = mock_response
    
    developer_agent = DeveloperAgent(mock_llm)
    orch.register_agent(AgentType.DEVELOPER, developer_agent)
    print("✓ DeveloperAgent registered")
    
    print("\n[5.2] Testing system task execution...")
    with patch('zeno.tools.app_control.open_app') as mock_app:
        mock_app.return_value = True
        
        task = Task(
            id="int-1",
            name="Open VS Code",
            description="Integration test",
            type=AgentType.SYSTEM,
            payload={"action": "open_app", "app": "vscode"}
        )
        
        results = orch.execute_plan([task])
        
        assert "int-1" in results
        assert results["int-1"].success
        assert task.status == TaskStatus.COMPLETED
        print("✓ System task executed via Orchestrator")
    
    print("\n[5.3] Testing developer task execution...")
    with tempfile.TemporaryDirectory() as tmpdir:
        original_base = file_system.BASE_DIR
        file_system.BASE_DIR = Path(tmpdir) / "test_workspace"
        
        try:
            task = Task(
                id="int-2",
                name="Generate Code",
                description="Integration test",
                type=AgentType.DEVELOPER,
                payload={
                    "intent": "generate_code",
                    "description": "Write hello function",
                    "language": "python"
                }
            )
            
            results = orch.execute_plan([task])
            
            assert "int-2" in results
            assert results["int-2"].success
            assert task.status == TaskStatus.COMPLETED
            print("✓ Developer task executed via Orchestrator")
            
        finally:
            file_system.BASE_DIR = original_base
    
    print("\n✅ Full integration tests PASSED")


def run_all_tests():
    """Run all Phase 4 tests"""
    print("\n" + "="*70)
    print("ZENO PHASE 4 TESTS - SYSTEM EXECUTION & CONTROL")
    print("="*70)
    
    test_passed = False
    
    try:
        test_file_system()
        test_app_control_mocked()
        test_system_agent()
        test_developer_agent()
        test_full_integration()
        
        print("\n" + "="*70)
        print("✅ ALL PHASE 4 TESTS PASSED")
        print("="*70)
        print("\nPhase 4 is working correctly!")
        print("\nWhat ZENO can now do:")
        print("  ✓ Open applications (VS Code, etc.)")
        print("  ✓ Open URLs in browser")
        print("  ✓ Generate code with LLM")
        print("  ✓ Write files to workspace")
        print("  ✓ Create directories")
        print("  ✓ Execute plans via Orchestrator")
        print("\nNext steps:")
        print("  1. Test with real Ollama: python start.py")
        print("  2. Try: 'open vscode'")
        print("  3. Try: 'write a python function'")
        
        test_passed = True
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED - ASSERTION ERROR")
        print("="*70)
        logger.error(f"Assertion failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED - EXCEPTION")
        print("="*70)
        logger.error(f"Test failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    
    return test_passed


if __name__ == "__main__":
    print(">>> TEST FILE STARTED <<<\n")
    
    test_result = False
    
    try:
        test_result = run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*70)
        if test_result:
            print("Exit code: 0 (SUCCESS)")
        else:
            print("Exit code: 1 (FAILURE)")
        print("="*70)
        print("\n[Press Enter to exit]")
        input()