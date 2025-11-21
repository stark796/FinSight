# utils/progress.py

from typing import Dict, Optional
from datetime import datetime
import threading

# Thread-safe progress store
_progress_store: Dict[str, Dict] = {}
_lock = threading.Lock()


def set_progress(upload_id: str, total: int, completed: int, stage: str = "embedding") -> None:
    """Set progress for an upload."""
    with _lock:
        _progress_store[upload_id] = {
            "total": total,
            "completed": completed,
            "stage": stage,
            "percentage": (completed / total * 100) if total > 0 else 0,
            "updated_at": datetime.now().isoformat()
        }


def get_progress(upload_id: str) -> Optional[Dict]:
    """Get progress for an upload."""
    with _lock:
        return _progress_store.get(upload_id)


def clear_progress(upload_id: str) -> None:
    """Clear progress after completion."""
    with _lock:
        _progress_store.pop(upload_id, None)

