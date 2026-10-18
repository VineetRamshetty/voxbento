import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

import pytest

# Add floor-bot directory to sys.path so we can import bot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "floor-bot")))

import bot


class TestWaitForProc(unittest.TestCase):
    def test_wait_for_proc_already_exited(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = 0
        bot._wait_for_proc(proc, "test-room")
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_wait_for_proc_clean_exit(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.return_value = 0

        bot._wait_for_proc(proc, "test-room")
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        proc.kill.assert_not_called()

    def test_wait_for_proc_sigkill_fallback(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        # First wait times out, second wait succeeds after kill
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=5), 0]

        bot._wait_for_proc(proc, "test-room")
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 2)
        proc.wait.assert_called_with(timeout=3)

    def test_wait_for_proc_sigkill_timeout_tolerated(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        # Both waits time out
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=5),
            subprocess.TimeoutExpired(cmd="test", timeout=3),
        ]

        # Should not raise exception
        bot._wait_for_proc(proc, "test-room")
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


class TestSubprocessManager(unittest.TestCase):
    def setUp(self):
        self.mgr = bot.SubprocessManager()

    def test_stop_room_async_not_in_rooms(self):
        # Should be a safe no-op
        self.mgr.stop_room_async("nonexistent-1")
        self.assertEqual(self.mgr.stopping_threads, {})

    def test_stop_room_async_eager_removal_and_background_teardown(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        release_wait = threading.Event()

        def slow_wait(*args, **kwargs):
            release_wait.wait(timeout=2.0)
            return 0

        proc.wait.side_effect = slow_wait

        key = "event-1"
        self.mgr.rooms[key] = proc
        self.mgr.room_states[key] = "in_meeting"
        self.mgr.room_failures[key] = "some_error"
        self.mgr.room_meta[key] = {"event_slug": "event", "room_id": 1}

        self.mgr.stop_room_async(key)

        # Immediately after stop_room_async returns:
        # Key is removed from rooms, meta, failures, and marked stopping
        self.assertNotIn(key, self.mgr.rooms)
        self.assertNotIn(key, self.mgr.room_meta)
        self.assertNotIn(key, self.mgr.room_failures)
        self.assertEqual(self.mgr.room_states[key], "stopping")
        self.assertIn(key, self.mgr.stopping_threads)

        # Allow teardown to finish
        release_wait.set()
        self.mgr.wait_for_all_stops(timeout=2.0)

        # After teardown finishes:
        self.assertEqual(self.mgr.room_states[key], "stopped")
        self.assertNotIn(key, self.mgr.stopping_threads)

    def test_stop_room_sync_waits_for_stopping_thread(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        release_wait = threading.Event()

        def slow_wait(*args, **kwargs):
            release_wait.wait(timeout=2.0)
            return 0

        proc.wait.side_effect = slow_wait

        key = "event-1"
        self.mgr.rooms[key] = proc
        self.mgr.stop_room_async(key)

        # Ensure background thread is running
        self.assertIn(key, self.mgr.stopping_threads)

        # Unblock and call synchronous stop_room
        release_wait.set()
        self.mgr.stop_room(key)
        self.assertEqual(self.mgr.room_states[key], "stopped")
        self.assertNotIn(key, self.mgr.stopping_threads)


class TestRequestHandlerStopEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Bind to port 0 for an ephemeral OS-assigned port
        cls.server = HTTPServer(("127.0.0.1", 0), bot.RequestHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_post_stop_missing_params(self):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/stop",
            data=json.dumps({"event_slug": "event"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_post_stop_non_blocking_returns_202(self):
        import urllib.request

        # Setup mock proc that blocks for 2 seconds on wait
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        block_event = threading.Event()

        def slow_wait(*args, **kwargs):
            block_event.wait(timeout=2.0)
            return 0

        proc.wait.side_effect = slow_wait

        key = "testevent-42"
        bot.manager.rooms[key] = proc
        bot.manager.room_states[key] = "in_meeting"
        bot.manager.room_meta[key] = {"event_slug": "testevent", "room_id": 42}

        try:
            start_time = time.monotonic()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/stop",
                data=json.dumps({"event_slug": "testevent", "room_id": 42}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                self.assertEqual(resp.status, 202)
                body = json.loads(resp.read().decode())
                self.assertEqual(body.get("status"), "stopping")
                self.assertEqual(body.get("event_slug"), "testevent")

            elapsed = time.monotonic() - start_time
            # Must return well under the 2-second block time (typically < 0.1s)
            self.assertLess(elapsed, 0.5)

            # GET /status immediately afterwards should not be blocked and key should be gone from active_rooms
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/status", timeout=1.0) as resp:
                self.assertEqual(resp.status, 200)
                status_data = json.loads(resp.read().decode())
                self.assertNotIn(key, status_data.get("active_rooms", {}))
        finally:
            block_event.set()
            bot.manager.wait_for_all_stops(timeout=2.0)
