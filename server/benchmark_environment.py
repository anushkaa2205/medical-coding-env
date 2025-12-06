# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark Environment Implementation.

A test environment for benchmarking infrastructure and concurrency.
Actions specify how many seconds to wait, allowing testing of parallel execution.
"""

import asyncio
import hashlib
import os
import socket
import time
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import BenchmarkAction, BenchmarkObservation


def _get_host_url() -> str:
    """Get the host URL for this server."""
    hostname = socket.gethostname()
    port = os.environ.get("PORT", "8000")
    # Try to get the actual IP if possible
    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        ip = "127.0.0.1"
    return f"http://{ip}:{port}"


class BenchmarkEnvironment(Environment):
    """
    A benchmark environment for testing concurrency and infrastructure.

    Actions specify a number of seconds to wait (sleep), which is useful for
    testing parallel execution and concurrency limits. The environment returns
    identity information (host_url, pid, session_hash) to help verify which
    server instance handled the request.

    Example:
        >>> env = BenchmarkEnvironment()
        >>> obs = env.reset()
        >>> print(obs.host_url)  # "http://192.168.1.1:8000"
        >>> print(obs.pid)  # 12345
        >>> print(obs.session_hash)  # "a1b2c3d4..."
        >>>
        >>> obs = env.step(BenchmarkAction(wait_seconds=2.0))
        >>> print(obs.waited_seconds)  # 2.0
    """

    def __init__(self):
        """Initialize the benchmark environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session_hash = hashlib.sha256(
            f"{uuid4()}-{time.time()}-{os.getpid()}".encode()
        ).hexdigest()[:16]
        self._pid = os.getpid()
        self._host_url = _get_host_url()

    def _make_observation(
        self, waited_seconds: float = 0.0, done: bool = False, reward: float = 0.0
    ) -> BenchmarkObservation:
        """Create an observation with current server identity."""
        return BenchmarkObservation(
            host_url=self._host_url,
            pid=self._pid,
            session_hash=self._session_hash,
            waited_seconds=waited_seconds,
            timestamp=time.time(),
            done=done,
            reward=reward,
        )

    def reset(self) -> BenchmarkObservation:
        """
        Reset the environment.

        Returns:
            BenchmarkObservation with server identity info
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._make_observation(waited_seconds=0.0, done=False, reward=0.0)

    def step(self, action: BenchmarkAction) -> BenchmarkObservation:  # type: ignore[override]
        """
        Execute a step by waiting for the specified seconds.

        Args:
            action: BenchmarkAction containing wait_seconds

        Returns:
            BenchmarkObservation with server identity and timing info
        """
        self._state.step_count += 1

        wait_time = max(0.0, action.wait_seconds)

        # Synchronous sleep - for async version, use step_async
        if wait_time > 0:
            time.sleep(wait_time)

        # Reward based on wait time (inverse - faster is better)
        reward = 1.0 / (1.0 + wait_time)

        return self._make_observation(
            waited_seconds=wait_time,
            done=False,
            reward=reward,
        )

    async def step_async(self, action: BenchmarkAction) -> BenchmarkObservation:
        """
        Async version of step - uses asyncio.sleep for better concurrency.

        Args:
            action: BenchmarkAction containing wait_seconds

        Returns:
            BenchmarkObservation with server identity and timing info
        """
        self._state.step_count += 1

        wait_time = max(0.0, action.wait_seconds)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        reward = 1.0 / (1.0 + wait_time)

        return self._make_observation(
            waited_seconds=wait_time,
            done=False,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
