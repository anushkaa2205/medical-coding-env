# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark Environment HTTP Client.

This module provides the client for connecting to a Benchmark Environment server
over HTTP. Useful for testing concurrency and infrastructure.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core.http_env_client import HTTPEnvClient

from .models import BenchmarkAction, BenchmarkObservation


class BenchmarkEnv(HTTPEnvClient[BenchmarkAction, BenchmarkObservation]):
    """
    HTTP client for the Benchmark Environment.

    This client connects to a BenchmarkEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = BenchmarkEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.host_url)
        >>> print(result.observation.pid)
        >>> print(result.observation.session_hash)
        >>>
        >>> # Test concurrency by waiting
        >>> result = client.step(BenchmarkAction(wait_seconds=2.0))
        >>> print(result.observation.waited_seconds)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = BenchmarkEnv.from_docker_image("benchmark-env:latest")
        >>> result = client.reset()
        >>> result = client.step(BenchmarkAction(wait_seconds=1.0))
    """

    def _step_payload(self, action: BenchmarkAction) -> Dict:
        """
        Convert BenchmarkAction to JSON payload for step request.

        Args:
            action: BenchmarkAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "wait_seconds": action.wait_seconds,
        }

    def _parse_result(self, payload: Dict) -> StepResult[BenchmarkObservation]:
        """
        Parse server response into StepResult[BenchmarkObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with BenchmarkObservation
        """
        obs_data = payload.get("observation", {})
        observation = BenchmarkObservation(
            host_url=obs_data.get("host_url", ""),
            pid=obs_data.get("pid", 0),
            session_hash=obs_data.get("session_hash", ""),
            waited_seconds=obs_data.get("waited_seconds", 0.0),
            timestamp=obs_data.get("timestamp", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
