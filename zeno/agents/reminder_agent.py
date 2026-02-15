"""
ZENO Reminder Agent - Pure Utility Service

Responsibilities:
- Create reminders (after user confirmation)
- Query reminders
- Update reminder status
- NO autonomous behavior
- NO LLM usage
- NO intelligence

This is NOT an Orchestrator Agent - it's a data service.
"""

import logging
from datetime import datetime
from typing import List, Optional

from zeno.memory.reminder_models import Reminder, ReminderStatus, create_reminder
from zeno.memory.reminder_store import ReminderStore

logger = logging.getLogger(__name__)


class ReminderAgent:
    """
    Pure utility service for reminder management.
    
    Design principles:
    - No inference or decision-making
    - No LLM calls
    - No autonomous triggers
    - User-driven operations only
    - Dumb by design
    
    Does NOT inherit from Agent - it's a service, not an executable agent.
    """
    
    def __init__(self, store: ReminderStore):
        """
        Initialize reminder agent.
        
        Args:
            store: ReminderStore instance for persistence
        """
        self.store = store
        logger.info("ReminderAgent initialized")
    
    def create_reminder(
        self,
        title: str,
        due_at: datetime,
        session_id: str,
        description: Optional[str] = None
    ) -> Reminder:
        """
        Create a new reminder.
        
        IMPORTANT: This should only be called AFTER explicit user confirmation.
        No automatic reminder creation.
        
        Args:
            title: Short, human-readable title
            due_at: When to remind (naive datetime, local time)
            session_id: Session that created this reminder
            description: Optional longer description
            
        Returns:
            Created Reminder object
            
        Raises:
            ValueError: If parameters are invalid
            ReminderStoreError: If storage fails
        """
        if not title or not title.strip():
            raise ValueError("Reminder title cannot be empty")
        
        if not isinstance(due_at, datetime):
            raise TypeError("due_at must be datetime")
        
        # Create reminder
        reminder = create_reminder(
            title=title.strip(),
            due_at=due_at,
            session_id=session_id,
            description=description.strip() if description else None
        )
        
        # Persist
        success = self.store.add_reminder(reminder)
        if not success:
            logger.error(f"Failed to store reminder: {reminder.id}")
            raise RuntimeError("Failed to store reminder")
        
        logger.info(f"Created reminder: {reminder.title} (due: {reminder.due_at})")
        return reminder
    
    def get_due_reminders(self, now: Optional[datetime] = None) -> List[Reminder]:
        """
        Get all reminders that are due.
        
        Args:
            now: Current datetime (default: datetime.now())
                 Injected for testability
            
        Returns:
            List of due reminders (status=PENDING and due_at <= now)
        """
        if now is None:
            now = datetime.now()
        
        due_reminders = self.store.get_due_reminders(now)
        
        logger.info(f"Found {len(due_reminders)} due reminders")
        return due_reminders
    
    def list_reminders(
        self,
        status: Optional[ReminderStatus] = None
    ) -> List[Reminder]:
        """
        List reminders, optionally filtered by status.
        
        Args:
            status: Filter by status (None = all)
            
        Returns:
            List of reminders
        """
        reminders = self.store.get_all_reminders(status=status)
        
        logger.debug(f"Listed {len(reminders)} reminders (status={status})")
        return reminders
    
    def mark_delivered(self, reminder_id: str) -> bool:
        """
        Mark a reminder as delivered.
        
        Called after reminder has been shown to user.
        
        Args:
            reminder_id: ID of reminder to mark
            
        Returns:
            True if successful
        """
        success = self.store.mark_delivered(reminder_id)
        
        if success:
            logger.info(f"Marked reminder {reminder_id} as delivered")
        else:
            logger.warning(f"Failed to mark reminder {reminder_id} as delivered")
        
        return success
    
    def mark_dismissed(self, reminder_id: str) -> bool:
        """
        Mark a reminder as dismissed.
        
        Called when user explicitly dismisses a reminder.
        
        Args:
            reminder_id: ID of reminder to mark
            
        Returns:
            True if successful
        """
        success = self.store.mark_dismissed(reminder_id)
        
        if success:
            logger.info(f"Marked reminder {reminder_id} as dismissed")
        else:
            logger.warning(f"Failed to mark reminder {reminder_id} as dismissed")
        
        return success
    
    def get_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """
        Get a specific reminder by ID.
        
        Args:
            reminder_id: Reminder ID
            
        Returns:
            Reminder if found, None otherwise
        """
        return self.store.get_reminder(reminder_id)
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """
        Permanently delete a reminder.
        
        Args:
            reminder_id: ID of reminder to delete
            
        Returns:
            True if deleted
        """
        success = self.store.delete_reminder(reminder_id)
        
        if success:
            logger.info(f"Deleted reminder {reminder_id}")
        else:
            logger.warning(f"Failed to delete reminder {reminder_id}")
        
        return success
    
    def get_stats(self) -> dict:
        """
        Get reminder statistics.
        
        Returns:
            Dict with counts by status
        """
        return self.store.get_stats()
    
    def format_reminder_for_user(self, reminder: Reminder) -> str:
        """
        Format reminder for display to user.
        
        Uses required language: "You asked me to remind you..."
        
        Args:
            reminder: Reminder to format
            
        Returns:
            Formatted string for display
        """
        # Required phrasing - never infer or decide
        text = f"You asked me to remind you: {reminder.title}"
        
        if reminder.description:
            text += f"\n  Details: {reminder.description}"
        
        # Show when it was due (helpful context)
        now = datetime.now()
        if reminder.due_at.date() == now.date():
            time_str = "today"
        elif reminder.due_at < now:
            days_ago = (now - reminder.due_at).days
            if days_ago == 1:
                time_str = "yesterday"
            else:
                time_str = f"{days_ago} days ago"
        else:
            time_str = f"on {reminder.due_at.strftime('%B %d')}"
        
        text += f"\n  (Due {time_str})"
        
        return text