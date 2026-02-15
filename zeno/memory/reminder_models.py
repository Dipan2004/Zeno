"""
ZENO Reminder Models

Data structures for the passive reminder system.

Philosophy:
- User-driven memory only
- No autonomous behavior
- No inference or intelligence
- Pure data representation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class ReminderStatus(Enum):
    """Reminder lifecycle states"""
    PENDING = "pending"        # Not yet shown to user
    DELIVERED = "delivered"    # Shown to user at least once
    DISMISSED = "dismissed"    # User explicitly dismissed


@dataclass
class Reminder:
    """
    A single reminder created at user's explicit request.
    
    Design principles:
    - Minimal fields only
    - No behavioral metadata
    - No priority or importance
    - No emotional context
    - User controls everything
    """
    id: str
    title: str
    due_at: datetime  # Naive datetime in system local time
    created_at: datetime
    status: ReminderStatus
    source_session_id: str
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate reminder data"""
        if not self.id:
            raise ValueError("Reminder ID cannot be empty")
        if not self.title:
            raise ValueError("Reminder title cannot be empty")
        if not isinstance(self.due_at, datetime):
            raise TypeError("due_at must be datetime")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be datetime")
        if not isinstance(self.status, ReminderStatus):
            raise TypeError("status must be ReminderStatus")
    
    def is_due(self, now: datetime) -> bool:
        """
        Check if reminder is due.
        
        Logic: due_at <= now (exact or overdue)
        No windows, no fuzziness.
        
        Args:
            now: Current datetime to check against
            
        Returns:
            True if reminder should be shown
        """
        # Normalize both datetimes to naive (strip timezone info) for comparison
        # This fixes the "can't compare offset-naive and offset-aware datetimes" error
        due_at_naive = self.due_at.replace(tzinfo=None) if self.due_at.tzinfo else self.due_at
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        
        return self.status == ReminderStatus.PENDING and due_at_naive <= now_naive
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'due_at': self.due_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'source_session_id': self.source_session_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Reminder':
        """Create Reminder from dict"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data.get('description'),
            due_at=datetime.fromisoformat(data['due_at']),
            created_at=datetime.fromisoformat(data['created_at']),
            status=ReminderStatus(data['status']),
            source_session_id=data['source_session_id']
        )


def create_reminder(
    title: str,
    due_at: datetime,
    session_id: str,
    description: Optional[str] = None
) -> Reminder:
    """
    Factory function to create a new reminder.
    
    Args:
        title: Short human-readable title
        due_at: When to remind (naive datetime, local time)
        session_id: Session that created this reminder
        description: Optional longer description
        
    Returns:
        New Reminder in PENDING status
    """
    return Reminder(
        id=str(uuid.uuid4()),
        title=title,
        description=description,
        due_at=due_at,
        created_at=datetime.now(),
        status=ReminderStatus.PENDING,
        source_session_id=session_id
    )