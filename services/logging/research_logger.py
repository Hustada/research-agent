from typing import Dict, Any, List
from queue import Queue
import json
import time

class ResearchLogger:
    def __init__(self):
        self._logs: Dict[str, Queue] = {}
        
    def create_session(self, session_id: str) -> None:
        """Create a new logging session."""
        self._logs[session_id] = Queue()
        
    def log_step(self, session_id: str, step: str, details: Any = None) -> None:
        """Log a research step with optional details."""
        if session_id not in self._logs:
            return
            
        log_entry = {
            'timestamp': time.time(),
            'step': step,
            'details': details
        }
        self._logs[session_id].put(log_entry)
        
    def get_logs(self, session_id: str) -> List[Dict]:
        """Get all logs for a session and clear the queue."""
        if session_id not in self._logs:
            return []
            
        logs = []
        while not self._logs[session_id].empty():
            logs.append(self._logs[session_id].get())
        return logs
        
    def clear_session(self, session_id: str) -> None:
        """Clear a logging session."""
        if session_id in self._logs:
            del self._logs[session_id]

# Global logger instance
research_logger = ResearchLogger()
