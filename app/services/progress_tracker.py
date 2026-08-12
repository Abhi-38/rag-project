import time
from typing import Any, Dict, Optional


class IngestionProgressTracker:
    """Thread-safe manager for tracking document ingestion stages, percentages, and ETA."""

    def __init__(self):
        self._state: Dict[str, Any] = {
            "status": "idle",  # "idle", "processing", "completed", "error"
            "stage": "Ready",
            "file_name": "",
            "current_step": 0,
            "total_steps": 3,
            "current_chunk": 0,
            "total_chunks": 0,
            "percentage": 0.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": 0.0,
            "error_message": None,
        }
        self._start_time: Optional[float] = None

    def start(self, file_name: str):
        """Initializes ingestion progress for a new file."""
        self._start_time = time.time()
        self._state = {
            "status": "processing",
            "stage": "Parsing document into semantic sections...",
            "file_name": file_name,
            "current_step": 1,
            "total_steps": 3,
            "current_chunk": 0,
            "total_chunks": 0,
            "percentage": 5.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": 0.0,
            "error_message": None,
        }

    def update_stage(self, stage: str, current_step: int, percentage: float):
        """Updates current execution stage and percentage."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        self._state["stage"] = stage
        self._state["current_step"] = current_step
        self._state["percentage"] = min(99.0, max(self._state["percentage"], round(percentage, 1)))
        self._state["elapsed_seconds"] = round(elapsed, 1)

    def update_embedding_progress(self, current_chunk: int, total_chunks: int):
        """Updates embedding generation progress and calculates real-time ETA."""
        if not self._start_time or total_chunks == 0:
            return

        elapsed = time.time() - self._start_time
        # Progress range for step 2 (embedding) is 20% to 90%
        embedding_pct = 20.0 + (current_chunk / total_chunks) * 70.0
        percentage = min(90.0, round(embedding_pct, 1))

        # Calculate estimated time remaining (ETA)
        chunks_processed = max(1, current_chunk)
        rate = chunks_processed / max(0.1, elapsed)
        remaining_chunks = total_chunks - current_chunk
        eta = remaining_chunks / rate if rate > 0 else 0.0

        self._state["stage"] = f"Generating vector embeddings ({current_chunk}/{total_chunks} chunks)..."
        self._state["current_step"] = 2
        self._state["current_chunk"] = current_chunk
        self._state["total_chunks"] = total_chunks
        self._state["percentage"] = percentage
        self._state["elapsed_seconds"] = round(elapsed, 1)
        self._state["estimated_remaining_seconds"] = max(0.0, round(eta, 1))

    def complete(self, total_parent: int, total_child: int):
        """Marks ingestion process as 100% completed."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        self._state = {
            "status": "completed",
            "stage": f"Ingestion complete! Indexed {total_parent} parent & {total_child} child chunks.",
            "file_name": self._state["file_name"],
            "current_step": 3,
            "total_steps": 3,
            "current_chunk": total_child,
            "total_chunks": total_child,
            "percentage": 100.0,
            "elapsed_seconds": round(elapsed, 1),
            "estimated_remaining_seconds": 0.0,
            "error_message": None,
        }

    def error(self, error_msg: str):
        """Sets error state."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        self._state["status"] = "error"
        self._state["stage"] = "Ingestion failed"
        self._state["error_message"] = error_msg
        self._state["elapsed_seconds"] = round(elapsed, 1)
        self._state["estimated_remaining_seconds"] = 0.0

    def get_progress(self) -> Dict[str, Any]:
        """Returns snapshot of current ingestion state."""
        return self._state.copy()


# Global thread-safe progress tracker singleton
progress_tracker = IngestionProgressTracker()
