"""
Tests for ZENO Reminder System

Tests the passive reminder functionality including:
- Reminder creation and storage
- Due reminder detection
- Status transitions
- Time-based behavior (with injected datetime)
"""

import sys
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

print("="*70)
print("Starting ZENO Reminder System Tests...")
print("="*70)

# Add parent to path



print("\n[IMPORT] Importing zeno.memory...")
try:
    from zeno.memory import Reminder, ReminderStatus, create_reminder, ReminderStore
    print("✓ zeno.memory imported successfully")
except Exception as e:
    print(f"✗ Failed to import zeno.memory: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[IMPORT] Importing zeno.agents.reminder_agent...")
try:
    from zeno.agents.reminder_agent import ReminderAgent
    print("✓ zeno.agents.reminder_agent imported successfully")
except Exception as e:
    print(f"✗ Failed to import ReminderAgent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n✓ All imports successful!\n")


def test_reminder_models():
    """Test reminder data models"""
    print("\n" + "="*70)
    print("TEST 1: Reminder Models")
    print("="*70)
    
    # Test reminder creation
    print("\n[1.1] Testing reminder creation...")
    now = datetime.now()
    future = now + timedelta(days=1)
    
    reminder = create_reminder(
        title="Test reminder",
        due_at=future,
        session_id="test-session",
        description="This is a test"
    )
    
    assert reminder.title == "Test reminder"
    assert reminder.status == ReminderStatus.PENDING
    assert reminder.description == "This is a test"
    print(f"✓ Reminder created: {reminder.id}")
    
    # Test is_due logic
    print("\n[1.2] Testing due detection...")
    
    # Future reminder should not be due
    assert not reminder.is_due(now), "Future reminder should not be due"
    print("✓ Future reminder correctly not due")
    
    # Past reminder should be due
    past = now - timedelta(hours=1)
    past_reminder = create_reminder(
        title="Overdue reminder",
        due_at=past,
        session_id="test-session"
    )
    assert past_reminder.is_due(now), "Past reminder should be due"
    print("✓ Past reminder correctly due")
    
    # Exact time should be due
    exact_reminder = create_reminder(
        title="Exact time reminder",
        due_at=now,
        session_id="test-session"
    )
    assert exact_reminder.is_due(now), "Exact time reminder should be due"
    print("✓ Exact time reminder correctly due")
    
    # Test serialization
    print("\n[1.3] Testing serialization...")
    reminder_dict = reminder.to_dict()
    restored = Reminder.from_dict(reminder_dict)
    
    assert restored.id == reminder.id
    assert restored.title == reminder.title
    assert restored.status == reminder.status
    print("✓ Serialization/deserialization works")
    
    print("\n✅ Reminder models test PASSED")


def test_reminder_store():
    """Test reminder storage"""
    print("\n" + "="*70)
    print("TEST 2: Reminder Store")
    print("="*70)
    
    # Use temporary file for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_reminders.json"
        store = ReminderStore(storage_path=storage_path)
        
        print(f"\n[2.1] Testing with storage: {storage_path}")
        
        # Test adding reminders
        print("\n[2.2] Testing add reminder...")
        now = datetime.now()
        reminder1 = create_reminder(
            title="First reminder",
            due_at=now + timedelta(days=1),
            session_id="test"
        )
        
        success = store.add_reminder(reminder1)
        assert success, "Should successfully add reminder"
        print("✓ Reminder added")
        
        # Test retrieving reminder
        print("\n[2.3] Testing get reminder...")
        retrieved = store.get_reminder(reminder1.id)
        assert retrieved is not None
        assert retrieved.id == reminder1.id
        assert retrieved.title == reminder1.title
        print("✓ Reminder retrieved correctly")
        
        # Test listing reminders
        print("\n[2.4] Testing list reminders...")
        all_reminders = store.get_all_reminders()
        assert len(all_reminders) == 1
        print(f"✓ Found {len(all_reminders)} reminder(s)")
        
        # Test adding multiple reminders
        reminder2 = create_reminder(
            title="Second reminder",
            due_at=now + timedelta(days=2),
            session_id="test"
        )
        store.add_reminder(reminder2)
        
        all_reminders = store.get_all_reminders()
        assert len(all_reminders) == 2
        print(f"✓ Total reminders: {len(all_reminders)}")
        
        # Test filtering by status
        print("\n[2.5] Testing filter by status...")
        pending = store.get_all_reminders(status=ReminderStatus.PENDING)
        assert len(pending) == 2
        print(f"✓ Pending reminders: {len(pending)}")
        
        # Test marking as delivered
        print("\n[2.6] Testing mark delivered...")
        success = store.mark_delivered(reminder1.id)
        assert success
        
        delivered = store.get_all_reminders(status=ReminderStatus.DELIVERED)
        assert len(delivered) == 1
        print("✓ Reminder marked as delivered")
        
        # Test marking as dismissed
        print("\n[2.7] Testing mark dismissed...")
        success = store.mark_dismissed(reminder2.id)
        assert success
        
        dismissed = store.get_all_reminders(status=ReminderStatus.DISMISSED)
        assert len(dismissed) == 1
        print("✓ Reminder marked as dismissed")
        
        # Test due reminders
        print("\n[2.8] Testing get due reminders...")
        due_reminder = create_reminder(
            title="Due now",
            due_at=now - timedelta(hours=1),
            session_id="test"
        )
        store.add_reminder(due_reminder)
        
        due_reminders = store.get_due_reminders(now)
        assert len(due_reminders) == 1
        assert due_reminders[0].id == due_reminder.id
        print(f"✓ Found {len(due_reminders)} due reminder(s)")
        
        # Test persistence
        print("\n[2.9] Testing persistence...")
        store2 = ReminderStore(storage_path=storage_path)
        reloaded = store2.get_all_reminders()
        assert len(reloaded) == 3
        print("✓ Reminders persisted across store instances")
        
        # Test stats
        print("\n[2.10] Testing stats...")
        stats = store.get_stats()
        print(f"✓ Stats: {stats}")
        assert stats['total'] == 3
        assert stats['pending'] == 1
        assert stats['delivered'] == 1
        assert stats['dismissed'] == 1
    
    print("\n✅ Reminder store test PASSED")


def test_reminder_agent():
    """Test reminder agent"""
    print("\n" + "="*70)
    print("TEST 3: Reminder Agent")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_reminders.json"
        store = ReminderStore(storage_path=storage_path)
        agent = ReminderAgent(store)
        
        print("\n[3.1] Testing create reminder...")
        now = datetime.now()
        future = now + timedelta(days=1)
        
        reminder = agent.create_reminder(
            title="Agent test reminder",
            due_at=future,
            session_id="test-session",
            description="Created via agent"
        )
        
        assert reminder.id is not None
        assert reminder.title == "Agent test reminder"
        print("✓ Reminder created via agent")
        
        # Test get due reminders with injected time
        print("\n[3.2] Testing get due reminders (time injection)...")
        
        # Create past reminder
        past_reminder = agent.create_reminder(
            title="Past reminder",
            due_at=now - timedelta(hours=1),
            session_id="test"
        )
        
        # Check with current time
        due = agent.get_due_reminders(now=now)
        assert len(due) == 1
        assert due[0].id == past_reminder.id
        print(f"✓ Found {len(due)} due reminder(s)")
        
        # Check with past time (should find nothing)
        past_time = now - timedelta(days=1)
        due_past = agent.get_due_reminders(now=past_time)
        assert len(due_past) == 0
        print("✓ Correctly found 0 reminders for past time check")
        
        # Test list reminders
        print("\n[3.3] Testing list reminders...")
        all_reminders = agent.list_reminders()
        assert len(all_reminders) == 2
        print(f"✓ Listed {len(all_reminders)} reminders")
        
        pending_only = agent.list_reminders(status=ReminderStatus.PENDING)
        assert len(pending_only) == 2
        print(f"✓ Listed {len(pending_only)} pending reminders")
        
        # Test mark delivered
        print("\n[3.4] Testing mark delivered...")
        success = agent.mark_delivered(reminder.id)
        assert success
        
        delivered = agent.list_reminders(status=ReminderStatus.DELIVERED)
        assert len(delivered) == 1
        print("✓ Marked as delivered")
        
        # Test mark dismissed
        print("\n[3.5] Testing mark dismissed...")
        success = agent.mark_dismissed(past_reminder.id)
        assert success
        
        dismissed = agent.list_reminders(status=ReminderStatus.DISMISSED)
        assert len(dismissed) == 1
        print("✓ Marked as dismissed")
        
        # Test formatting
        print("\n[3.6] Testing reminder formatting...")
        test_reminder = agent.create_reminder(
            title="Meeting with team",
            due_at=now,
            session_id="test",
            description="Discuss Q1 goals"
        )
        
        formatted = agent.format_reminder_for_user(test_reminder)
        assert "You asked me to remind you" in formatted
        assert "Meeting with team" in formatted
        print("✓ Reminder formatted correctly")
        print(f"   Format: {formatted[:80]}...")
        
        # Test stats
        print("\n[3.7] Testing stats...")
        stats = agent.get_stats()
        print(f"✓ Stats: {stats}")
        assert stats['total'] == 3
    
    print("\n✅ Reminder agent test PASSED")


def test_time_edge_cases():
    """Test edge cases with time handling"""
    print("\n" + "="*70)
    print("TEST 4: Time Edge Cases")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_reminders.json"
        store = ReminderStore(storage_path=storage_path)
        agent = ReminderAgent(store)
        
        now = datetime.now()
        
        # Test exact match
        print("\n[4.1] Testing exact time match...")
        exact_reminder = agent.create_reminder(
            title="Exact time",
            due_at=now,
            session_id="test"
        )
        due = agent.get_due_reminders(now=now)
        assert len(due) == 1
        print("✓ Exact time is considered due")
        
        # Test 1 second before (not due)
        print("\n[4.2] Testing 1 second before due time...")
        one_sec_before = now - timedelta(seconds=1)
        due = agent.get_due_reminders(now=one_sec_before)
        assert len(due) == 0
        print("✓ 1 second before is not due")
        
        # Test 1 second after (is due)
        print("\n[4.3] Testing 1 second after due time...")
        one_sec_after = now + timedelta(seconds=1)
        due = agent.get_due_reminders(now=one_sec_after)
        assert len(due) == 1
        print("✓ 1 second after is due")
        
        # Test very old reminder
        print("\n[4.4] Testing very old reminder...")
        ancient = agent.create_reminder(
            title="Ancient reminder",
            due_at=now - timedelta(days=365),
            session_id="test"
        )
        due = agent.get_due_reminders(now=now)
        assert len(due) == 2  # exact_reminder + ancient
        print("✓ Old reminders are still detected as due")
    
    print("\n✅ Time edge cases test PASSED")


def run_all_tests():
    """Run all reminder system tests"""
    print("\n" + "="*70)
    print("ZENO REMINDER SYSTEM TESTS")
    print("="*70)
    
    test_passed = False
    
    try:
        test_reminder_models()
        test_reminder_store()
        test_reminder_agent()
        test_time_edge_cases()
        
        print("\n" + "="*70)
        print("✅ ALL REMINDER TESTS PASSED")
        print("="*70)
        print("\nReminder system is working correctly!")
        print("\nKey features validated:")
        print("  ✓ Reminder creation and storage")
        print("  ✓ Due detection with time injection")
        print("  ✓ Status transitions (pending → delivered → dismissed)")
        print("  ✓ Persistence across store instances")
        print("  ✓ Edge case time handling")
        print("\nNext steps:")
        print("  1. Pull Qwen model: ollama pull qwen2.5:3b-instruct-q4_0")
        print("  2. Test with start.py")
        print("  3. Try reminder creation flow")
        
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
        # This ALWAYS runs, even if something crashes
        print("\n" + "="*70)
        if test_result:
            print("Exit code: 0 (SUCCESS)")
        else:
            print("Exit code: 1 (FAILURE)")
        print("="*70)
        print("\n[Press Enter to exit]")
        input()