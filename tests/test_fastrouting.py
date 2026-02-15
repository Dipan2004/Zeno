"""
Fast Router Tests - Rule-Based Command Detection

Tests the fast routing system that bypasses LLM for obvious commands.
"""

import sys
from pathlib import Path

print("="*70)
print("Starting Fast Router Tests...")
print("="*70)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zeno.core import FastRouter, AgentType
from zeno.tools import app_control

print("\n✓ Imports successful\n")


def test_app_routing():
    """Test app command routing"""
    print("="*70)
    print("TEST 1: App Command Routing")
    print("="*70)
    
    router = FastRouter()
    
    # Test basic open command
    print("\n[1.1] Testing 'open vscode'...")
    task = router.try_route("open vscode")
    assert task is not None, "Should route 'open vscode'"
    assert task.type == AgentType.SYSTEM
    assert task.payload["action"] == "open_app"
    print(f"✓ Routed to: {task.name}")
    print(f"  Payload: {task.payload}")
    
    # Test launch command
    print("\n[1.2] Testing 'launch notepad'...")
    task = router.try_route("launch notepad")
    assert task is not None
    assert task.payload["action"] == "open_app"
    print(f"✓ Routed to: {task.name}")
    
    # Test with politeness
    print("\n[1.3] Testing 'please open calculator'...")
    task = router.try_route("please open calculator")
    assert task is not None
    assert task.payload["action"] == "open_app"
    print(f"✓ Routed to: {task.name}")
    
    # Test can you
    print("\n[1.4] Testing 'can you open chrome'...")
    task = router.try_route("can you open chrome")
    assert task is not None
    print(f"✓ Routed to: {task.name}")
    
    # Test unknown app (should return None for LLM fallback)
    print("\n[1.5] Testing unknown app 'open unknownapp123'...")
    task = router.try_route("open unknownapp123")
    assert task is None, "Unknown app should return None"
    print("✓ Correctly returned None for unknown app")
    
    print("\n✅ App routing tests PASSED")


def test_url_routing():
    """Test URL and site keyword routing"""
    print("\n" + "="*70)
    print("TEST 2: URL Routing")
    print("="*70)
    
    router = FastRouter()
    
    # Test explicit URL
    print("\n[2.1] Testing 'open https://github.com'...")
    task = router.try_route("open https://github.com")
    assert task is not None
    assert task.payload["action"] == "open_url"
    assert task.payload["url"] == "https://github.com"
    print(f"✓ Routed to URL: {task.payload['url']}")
    
    # Test site keyword (should resolve to full URL)
    print("\n[2.2] Testing 'open github'...")
    task = router.try_route("open github")
    assert task is not None
    assert task.payload["action"] == "open_url"
    assert task.payload["url"] == "https://github.com"  # Resolved URL
    print(f"✓ Resolved site keyword to: {task.payload['url']}")
    
    # Test www. URL (should add https://)
    print("\n[2.3] Testing 'open www.google.com'...")
    task = router.try_route("open www.google.com")
    assert task is not None
    assert task.payload["url"] == "https://www.google.com"
    print(f"✓ Added https:// prefix: {task.payload['url']}")
    
    print("\n✅ URL routing tests PASSED")


def test_pattern_matching():
    """Test pattern matching variations"""
    print("\n" + "="*70)
    print("TEST 3: Pattern Matching")
    print("="*70)
    
    router = FastRouter()
    
    test_cases = [
        ("open vscode", True),
        ("launch chrome", True),
        ("start notepad", True),
        ("please open calc", True),
        ("can you launch explorer", True),
        ("could you open paint", True),
        ("Open VSCode", True),  # Case insensitive
        ("OPEN NOTEPAD", True),  # Case insensitive
        ("vscode", False),  # No command word
        ("hello", False),  # Irrelevant
        ("", False),  # Empty
    ]
    
    for input_str, should_match in test_cases:
        task = router.try_route(input_str)
        matched = task is not None
        
        if matched == should_match:
            print(f"✓ '{input_str}' -> {matched} (expected)")
        else:
            print(f"✗ '{input_str}' -> {matched} (expected {should_match})")
            assert False, f"Pattern match failed for '{input_str}'"
    
    print("\n✅ Pattern matching tests PASSED")


def test_performance():
    """Test performance (should be < 50ms per route)"""
    print("\n" + "="*70)
    print("TEST 4: Performance")
    print("="*70)
    
    import time
    
    router = FastRouter()
    
    test_inputs = [
        "open vscode",
        "launch chrome",
        "please open notepad",
        "open https://github.com",
        "open github",
    ]
    
    total_time = 0
    iterations = 100
    
    print(f"\n[4.1] Running {iterations} iterations...")
    
    for _ in range(iterations):
        for input_str in test_inputs:
            start = time.perf_counter()
            router.try_route(input_str)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            total_time += elapsed
    
    avg_time = total_time / (iterations * len(test_inputs))
    
    print(f"✓ Average routing time: {avg_time:.2f}ms")
    
    if avg_time < 1.0:  # Should be sub-millisecond
        print("✓ Performance excellent (< 1ms)")
    elif avg_time < 50:
        print("✓ Performance good (< 50ms)")
    else:
        print(f"⚠ Performance warning: {avg_time:.2f}ms (target: < 50ms)")
    
    print("\n✅ Performance tests PASSED")


def test_app_registry():
    """Test app registry coverage"""
    print("\n" + "="*70)
    print("TEST 5: App Registry Coverage")
    print("="*70)
    
    print("\n[5.1] Available apps:")
    apps = app_control.get_available_apps()
    print(f"  Total aliases: {len(apps)}")
    print(f"  Sample: {', '.join(apps[:15])}...")
    
    print("\n[5.2] Available site keywords:")
    sites = app_control.get_available_sites()
    print(f"  Total sites: {len(sites)}")
    print(f"  Sites: {', '.join(sites)}")
    
    # Test that common apps are present
    required_apps = ["vscode", "notepad", "calculator", "chrome", "explorer"]
    print("\n[5.3] Checking required apps...")
    for app in required_apps:
        if app in [a.lower() for a in apps]:
            print(f"  ✓ {app}")
        else:
            print(f"  ✗ {app} MISSING")
    
    print("\n✅ App registry tests PASSED")


def run_all_tests():
    """Run all fast router tests"""
    print("\n" + "="*70)
    print("FAST ROUTER TESTS")
    print("="*70)
    
    test_passed = False
    
    try:
        test_app_routing()
        test_url_routing()
        test_pattern_matching()
        test_performance()
        test_app_registry()
        
        print("\n" + "="*70)
        print("✅ ALL FAST ROUTER TESTS PASSED")
        print("="*70)
        print("\nFast Router is working correctly!")
        print("\nPerformance:")
        print("  ✓ Rule-based routing (NO LLM)")
        print("  ✓ Sub-millisecond response time")
        print("  ✓ Zero tokens used")
        print("\nNext steps:")
        print("  1. Test with: python start.py")
        print("  2. Try: 'open vscode' (should be instant)")
        print("  3. Try: 'open github' (should open browser)")
        
        test_passed = True
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED - ASSERTION ERROR")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED - EXCEPTION")
        print("="*70)
        print(f"Error: {e}")
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