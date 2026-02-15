"""
ZENO Reminder Store - Persistent JSON Storage

Handles reading and writing reminders to ~/.zeno/reminders.json

Design:
- Simple JSON file storage
- Graceful corruption recovery
- No concurrent write handling (not needed in current architecture)
- Transparent and auditable by user
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .reminder_models import Reminder, ReminderStatus

logger = logging.getLogger(__name__)


class ReminderStoreError(Exception):
    """Base exception for reminder storage errors"""
    pass


class ReminderStore:
    """
    File-based reminder storage using JSON.
    
    Storage location: ~/.zeno/reminders.json
    
    Philosophy:
    - User can inspect/edit file directly
    - Corruption is handled gracefully
    - No hidden state or caching
    """
    
    DEFAULT_STORAGE_DIR = Path.home() / ".zeno"
    DEFAULT_STORAGE_FILE = "reminders.json"
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize reminder store.
        
        Args:
            storage_path: Custom storage file path (default: ~/.zeno/reminders.json)
        """
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = self.DEFAULT_STORAGE_DIR / self.DEFAULT_STORAGE_FILE
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not self.storage_path.exists():
            self._initialize_storage()
        
        logger.info(f"ReminderStore initialized: {self.storage_path}")
    
    def _initialize_storage(self):
        """Create empty storage file"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({"reminders": []}, f, indent=2)
            logger.info("Initialized empty reminder storage")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise ReminderStoreError(f"Cannot initialize storage: {e}") from e
    
    def _load_reminders(self) -> List[Reminder]:
        """
        Load all reminders from storage.
        
        Returns:
            List of Reminder objects
            
        Raises:
            ReminderStoreError: If storage cannot be read
        """
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            reminders = []
            for reminder_dict in data.get('reminders', []):
                try:
                    reminder = Reminder.from_dict(reminder_dict)
                    reminders.append(reminder)
                except Exception as e:
                    logger.warning(f"Skipping invalid reminder: {e}")
                    # Continue loading valid reminders
            
            return reminders
            
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON in reminder storage: {e}")
            self._backup_and_reset()
            return []
        except FileNotFoundError:
            logger.warning("Storage file not found, initializing")
            self._initialize_storage()
            return []
        except Exception as e:
            logger.error(f"Failed to load reminders: {e}", exc_info=True)
            raise ReminderStoreError(f"Cannot load reminders: {e}") from e
    
    def _save_reminders(self, reminders: List[Reminder]):
        """
        Save all reminders to storage.
        
        Args:
            reminders: List of Reminder objects to save
            
        Raises:
            ReminderStoreError: If storage cannot be written
        """
        try:
            data = {
                "reminders": [r.to_dict() for r in reminders]
            }
            
            # Write atomically (write to temp, then rename)
            temp_path = self.storage_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Replace old file with new
            temp_path.replace(self.storage_path)
            
            logger.debug(f"Saved {len(reminders)} reminders")
            
        except Exception as e:
            logger.error(f"Failed to save reminders: {e}", exc_info=True)
            raise ReminderStoreError(f"Cannot save reminders: {e}") from e
    
    def _backup_and_reset(self):
        """
        Backup corrupted file and create fresh storage.
        
        Called when JSON is corrupted or unreadable.
        """
        backup_path = self.storage_path.with_suffix('.json.bak')
        
        try:
            if self.storage_path.exists():
                # Rename corrupted file
                self.storage_path.rename(backup_path)
                logger.warning(f"Backed up corrupted storage to: {backup_path}")
            
            # Create fresh storage
            self._initialize_storage()
            logger.info("Created fresh reminder storage")
            
        except Exception as e:
            logger.error(f"Failed to backup and reset: {e}", exc_info=True)
            # Last resort: just create new file
            self._initialize_storage()
    
    def add_reminder(self, reminder: Reminder) -> bool:
        """
        Add a new reminder to storage.
        
        Args:
            reminder: Reminder to add
            
        Returns:
            True if successful
            
        Raises:
            ReminderStoreError: If operation fails
        """
        reminders = self._load_reminders()
        
        # Check for duplicate ID (shouldn't happen with UUID, but be safe)
        if any(r.id == reminder.id for r in reminders):
            logger.warning(f"Reminder with ID {reminder.id} already exists")
            return False
        
        reminders.append(reminder)
        self._save_reminders(reminders)
        
        logger.info(f"Added reminder: {reminder.id} - {reminder.title}")
        return True
    
    def get_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """
        Get a specific reminder by ID.
        
        Args:
            reminder_id: Reminder ID to find
            
        Returns:
            Reminder if found, None otherwise
        """
        reminders = self._load_reminders()
        for reminder in reminders:
            if reminder.id == reminder_id:
                return reminder
        return None
    
    def get_all_reminders(self, status: Optional[ReminderStatus] = None) -> List[Reminder]:
        """
        Get all reminders, optionally filtered by status.
        
        Args:
            status: Filter by this status (None = all reminders)
            
        Returns:
            List of reminders
        """
        reminders = self._load_reminders()
        
        if status:
            reminders = [r for r in reminders if r.status == status]
        
        return reminders
    
    def get_due_reminders(self, now: datetime) -> List[Reminder]:
        """
        Get all reminders that are due now.
        
        Args:
            now: Current datetime to check against
            
        Returns:
            List of due reminders (status=PENDING and due_at <= now)
        """
        reminders = self._load_reminders()
        due = [r for r in reminders if r.is_due(now)]
        
        logger.info(f"Found {len(due)} due reminders")
        return due
    
    def update_reminder(self, reminder: Reminder) -> bool:
        """
        Update an existing reminder.
        
        Args:
            reminder: Updated reminder object
            
        Returns:
            True if successful, False if not found
            
        Raises:
            ReminderStoreError: If operation fails
        """
        reminders = self._load_reminders()
        
        # Find and replace
        for i, r in enumerate(reminders):
            if r.id == reminder.id:
                reminders[i] = reminder
                self._save_reminders(reminders)
                logger.info(f"Updated reminder: {reminder.id}")
                return True
        
        logger.warning(f"Reminder {reminder.id} not found for update")
        return False
    
    def mark_delivered(self, reminder_id: str) -> bool:
        """
        Mark reminder as delivered.
        
        Args:
            reminder_id: ID of reminder to mark
            
        Returns:
            True if successful
        """
        reminder = self.get_reminder(reminder_id)
        if not reminder:
            logger.warning(f"Cannot mark delivered: reminder {reminder_id} not found")
            return False
        
        reminder.status = ReminderStatus.DELIVERED
        return self.update_reminder(reminder)
    
    def mark_dismissed(self, reminder_id: str) -> bool:
        """
        Mark reminder as dismissed.
        
        Args:
            reminder_id: ID of reminder to mark
            
        Returns:
            True if successful
        """
        reminder = self.get_reminder(reminder_id)
        if not reminder:
            logger.warning(f"Cannot mark dismissed: reminder {reminder_id} not found")
            return False
        
        reminder.status = ReminderStatus.DISMISSED
        return self.update_reminder(reminder)
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """
        Delete a reminder permanently.
        
        Args:
            reminder_id: ID of reminder to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            ReminderStoreError: If operation fails
        """
        reminders = self._load_reminders()
        
        # Filter out the reminder to delete
        original_count = len(reminders)
        reminders = [r for r in reminders if r.id != reminder_id]
        
        if len(reminders) == original_count:
            logger.warning(f"Reminder {reminder_id} not found for deletion")
            return False
        
        self._save_reminders(reminders)
        logger.info(f"Deleted reminder: {reminder_id}")
        return True
    
    def get_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dict with reminder counts by status
        """
        reminders = self._load_reminders()
        
        stats = {
            'total': len(reminders),
            'pending': 0,
            'delivered': 0,
            'dismissed': 0
        }
        
        for reminder in reminders:
            if reminder.status == ReminderStatus.PENDING:
                stats['pending'] += 1
            elif reminder.status == ReminderStatus.DELIVERED:
                stats['delivered'] += 1
            elif reminder.status == ReminderStatus.DISMISSED:
                stats['dismissed'] += 1
        
        return stats