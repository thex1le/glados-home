"""Tests for pipeline trace logging."""

import pytest
import json
import os
import tempfile
import time

from glados_modules.TraceLog import TraceLog


class TestTraceLog:
    """Test trace ID generation and JSONL writing."""

    def test_new_id_is_8_hex_chars(self):
        tid = TraceLog.new_id()
        assert len(tid) == 8
        int(tid, 16)  # should not raise

    def test_unique_ids(self):
        ids = {TraceLog.new_id() for _ in range(1000)}
        assert len(ids) == 1000  # all unique

    def test_start_and_end_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = TraceLog(log_dir=tmpdir)
            tid = TraceLog.new_id()
            tracer.start_trace(tid, camera="camera_head", ts_vision=1000.0)
            tracer.end_trace(tid, ts_servo_rx=1000.05)
            tracer.close()

            filepath = os.path.join(tmpdir, "traces.jsonl")
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["trace_id"] == tid
            assert record["camera"] == "camera_head"
            assert record["ts_vision"] == 1000.0
            assert record["ts_servo_rx"] == 1000.05
            assert "latency_ms" in record
            assert "ts_complete" in record

    def test_update_trace_adds_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = TraceLog(log_dir=tmpdir)
            tid = TraceLog.new_id()
            tracer.start_trace(tid, camera="head")
            tracer.update_trace(tid, world_lr=105.2, world_ud=88.4)
            tracer.end_trace(tid)
            tracer.close()

            filepath = os.path.join(tmpdir, "traces.jsonl")
            with open(filepath) as f:
                record = json.loads(f.readline())
            assert record["world_lr"] == 105.2
            assert record["world_ud"] == 88.4

    def test_end_unknown_trace_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = TraceLog(log_dir=tmpdir)
            tracer.end_trace("nonexistent")  # should not crash
            tracer.close()
            filepath = os.path.join(tmpdir, "traces.jsonl")
            with open(filepath) as f:
                assert f.read().strip() == ""

    def test_close_writes_incomplete_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = TraceLog(log_dir=tmpdir)
            tracer.start_trace("abc", camera="head")
            # Don't end it -- close should write it as incomplete
            tracer.close()
            filepath = os.path.join(tmpdir, "traces.jsonl")
            with open(filepath) as f:
                record = json.loads(f.readline())
            assert record["incomplete"] is True
