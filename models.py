# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Benchmark Environment.

The benchmark environment is designed for testing concurrency and infrastructure.
Actions specify a wait time in seconds, allowing testing of parallel execution.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class BenchmarkAction(Action):
    """Action for the Benchmark environment - specifies seconds to wait."""

    wait_seconds: float = Field(default=0.0, ge=0.0, description="Seconds to wait/sleep")


class BenchmarkObservation(Observation):
    """Observation from the Benchmark environment with server identity info."""

    # Server identity
    host_url: str = Field(default="", description="URL of the server that handled the request")
    pid: int = Field(default=0, description="Process ID of the server")
    session_hash: str = Field(default="", description="Unique hash identifying this server session")

    # Timing info
    waited_seconds: float = Field(default=0.0, description="Actual seconds waited")
    timestamp: float = Field(default=0.0, description="Unix timestamp when observation was created")
