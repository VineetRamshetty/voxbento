"""Tests for InMemoryRateLimiter and periodic cleanup (ISSUE-6).

Covers:
- Sliding window rate limiting allow / block behavior
- Independent tracking across different keys
- Window expiry and counter reset
- InMemoryRateLimiter.cleanup() purges expired keys and retains active keys
- _rate_limit_cleanup_loop periodically executes cleanup on global instances
- Lifespan context manager in fastapi_app starts and terminates the cleanup task
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from fastapi_app import _rate_limit_cleanup_loop, app, lifespan
from portal.rate_limit import InMemoryRateLimiter, auth_rate_limiter, token_rate_limiter

pytestmark = pytest.mark.anyio


@pytest.fixture
def limiter():
    return InMemoryRateLimiter(max_requests=3, window_seconds=60)


@pytest.fixture(autouse=True)
def _reset_global_rate_limiters():
    """Ensure global limiters have empty stores before and after each test."""
    auth_rate_limiter._store.clear()
    token_rate_limiter._store.clear()
    yield
    auth_rate_limiter._store.clear()
    token_rate_limiter._store.clear()


class TestInMemoryRateLimiterBasic:
    async def test_allows_up_to_max_requests(self, limiter: InMemoryRateLimiter):
        assert await limiter.is_rate_limited("client1") is False
        assert await limiter.is_rate_limited("client1") is False
        assert await limiter.is_rate_limited("client1") is False

    async def test_blocks_on_exceeding_max_requests(self, limiter: InMemoryRateLimiter):
        for _ in range(3):
            assert await limiter.is_rate_limited("client1") is False
        # 4th request must be rate limited
        assert await limiter.is_rate_limited("client1") is True

    async def test_different_keys_tracked_independently(self, limiter: InMemoryRateLimiter):
        for _ in range(3):
            assert await limiter.is_rate_limited("client1") is False
        assert await limiter.is_rate_limited("client1") is True
        # client2 has independent counter
        assert await limiter.is_rate_limited("client2") is False

    async def test_window_expiry_resets_count(self, limiter: InMemoryRateLimiter):
        for _ in range(3):
            await limiter.is_rate_limited("client1")
        assert await limiter.is_rate_limited("client1") is True

        # Simulate expiration of the sliding window
        limiter._store["client1"] = (3, time.time() - 1)
        assert await limiter.is_rate_limited("client1") is False


class TestInMemoryRateLimiterCleanup:
    async def test_cleanup_removes_expired_entries(self, limiter: InMemoryRateLimiter):
        await limiter.is_rate_limited("user1")
        await limiter.is_rate_limited("user2")
        now = time.time()
        limiter._store["user1"] = (1, now - 10)
        limiter._store["user2"] = (2, now - 5)

        assert len(limiter._store) == 2
        await limiter.cleanup()
        assert len(limiter._store) == 0
        assert "user1" not in limiter._store
        assert "user2" not in limiter._store

    async def test_cleanup_keeps_active_entries(self, limiter: InMemoryRateLimiter):
        await limiter.is_rate_limited("expired_user")
        await limiter.is_rate_limited("active_user")
        now = time.time()
        limiter._store["expired_user"] = (1, now - 10)
        limiter._store["active_user"] = (2, now + 100)

        assert len(limiter._store) == 2
        await limiter.cleanup()
        assert len(limiter._store) == 1
        assert "expired_user" not in limiter._store
        assert "active_user" in limiter._store
        assert limiter._store["active_user"] == (2, now + 100)

    async def test_cleanup_on_empty_store_is_a_no_op(self, limiter: InMemoryRateLimiter):
        assert len(limiter._store) == 0
        await limiter.cleanup()
        assert len(limiter._store) == 0

    async def test_cleanup_does_not_affect_rate_limit_logic(self, limiter: InMemoryRateLimiter):
        await limiter.is_rate_limited("user")
        now = time.time()
        # Keep entry active with count 1
        limiter._store["user"] = (1, now + 100)
        await limiter.cleanup()

        # Next request allowed (count 2)
        assert await limiter.is_rate_limited("user") is False
        # Next request allowed (count 3, max_requests reached)
        assert await limiter.is_rate_limited("user") is False
        # Exceeded max_requests
        assert await limiter.is_rate_limited("user") is True


class TestRateLimitCleanupLoop:
    async def test_cleanup_loop_purges_global_limiters(self):
        now = time.time()
        auth_rate_limiter._store["ip_expired"] = (1, now - 10)
        auth_rate_limiter._store["ip_active"] = (1, now + 60)
        token_rate_limiter._store["token_expired"] = (1, now - 10)
        token_rate_limiter._store["token_active"] = (1, now + 60)

        task = asyncio.create_task(_rate_limit_cleanup_loop(interval_seconds=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert "ip_expired" not in auth_rate_limiter._store
        assert "ip_active" in auth_rate_limiter._store
        assert "token_expired" not in token_rate_limiter._store
        assert "token_active" in token_rate_limiter._store

    async def test_cleanup_loop_cancels_cleanly(self):
        task = asyncio.create_task(_rate_limit_cleanup_loop(interval_seconds=100))
        await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()


class TestLifespanRateLimitCleanup:
    async def test_lifespan_starts_and_stops_cleanup_loop(self):
        async with lifespan(app):
            pass
