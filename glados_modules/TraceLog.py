"""Pipeline trace ID generation and logging.

Assigns a short trace_id to each detection frame at the vision stage,
then propagates it through tracking and servo commands. A separate
traces.jsonl file records the full pipeline path with timestamps at
each stage, enabling latency analysis and cross-system correlation.

Usage:
    tracer = TraceLog("./logs")
    trace_id = TraceLog.new_id()
    tracer.start_trace(trace_id, camera="camera_head", ts_vision=time.time())
    tracer.update_trace(trace_id, ts_track_start=time.time(), world_lr=105.2)
    tracer.end_trace(trace_id, ts_servo_rx=time.time())
"""

import json
import time
import os
from uuid import uuid4
from threading import Lock


class TraceLog:
    """Append-only JSONL writer for pipeline trace records."""

    def __init__(self, log_dir: str = "./logs") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._filepath = os.path.join(log_dir, "traces.jsonl")
        self._file = open(self._filepath, 'a')
        self._lock = Lock()
        self._active: dict = {}  # trace_id -> dict of accumulated fields

    @staticmethod
    def new_id() -> str:
        """Generate a short 8-character hex trace ID."""
        return uuid4().hex[:8]

    def start_trace(self, trace_id: str, **fields) -> None:
        """Begin a new trace with initial fields."""
        with self._lock:
            self._active[trace_id] = {"trace_id": trace_id, **fields}

    def update_trace(self, trace_id: str, **fields) -> None:
        """Add fields to an active trace."""
        with self._lock:
            if trace_id in self._active:
                self._active[trace_id].update(fields)

    def end_trace(self, trace_id: str, **fields) -> None:
        """Finalize a trace: add final fields, compute latency, write to file."""
        with self._lock:
            if trace_id not in self._active:
                return
            record = self._active.pop(trace_id)
            record.update(fields)

        # Compute end-to-end latency if we have vision timestamp
        ts_vision = record.get("ts_vision")
        if ts_vision:
            record["latency_ms"] = round((time.time() - ts_vision) * 1000, 1)

        record["ts_complete"] = time.time()

        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def write_trace(self, record: dict) -> None:
        """Write a complete trace record directly (for one-shot logging)."""
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def close(self) -> None:
        """Flush and close the trace file."""
        with self._lock:
            # Write out any incomplete traces
            for trace_id, record in self._active.items():
                record["incomplete"] = True
                self._file.write(json.dumps(record) + "\n")
            self._active.clear()
            self._file.flush()
            self._file.close()

    @property
    def filepath(self) -> str:
        return self._filepath
