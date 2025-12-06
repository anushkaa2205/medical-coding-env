---
title: Benchmark Environment Server
emoji: 🕹️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Benchmark Environment

A test environment for benchmarking infrastructure and concurrency. Actions specify how many seconds to wait (sleep), making it ideal for testing parallel execution and server scaling. Returns server identity information to verify which instance handled each request.

## Quick Start

The simplest way to use the Benchmark environment is through the `BenchmarkEnv` class:

```python
from benchmark import BenchmarkAction, BenchmarkEnv

try:
    # Create environment from Docker image
    benchmarkenv = BenchmarkEnv.from_docker_image("benchmark-env:latest")

    # Reset - get server identity
    result = benchmarkenv.reset()
    print(f"Host URL: {result.observation.host_url}")
    print(f"PID: {result.observation.pid}")
    print(f"Session Hash: {result.observation.session_hash}")

    # Test concurrency with different wait times
    wait_times = [0.5, 1.0, 2.0]

    for seconds in wait_times:
        result = benchmarkenv.step(BenchmarkAction(wait_seconds=seconds))
        print(f"Waited: {result.observation.waited_seconds}s")
        print(f"  → Timestamp: {result.observation.timestamp}")
        print(f"  → Reward: {result.reward}")
        print(f"  → Server PID: {result.observation.pid}")

finally:
    # Always clean up
    benchmarkenv.close()
```

That's it! The `BenchmarkEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Testing Concurrency

The benchmark environment is designed to test concurrent execution:

```python
import asyncio
from benchmark import BenchmarkAction, BenchmarkEnv

async def parallel_requests():
    # Connect to multiple servers or same server
    clients = [
        BenchmarkEnv(base_url="http://localhost:8000"),
        BenchmarkEnv(base_url="http://localhost:8001"),
        BenchmarkEnv(base_url="http://localhost:8002"),
    ]

    # Reset all clients
    for client in clients:
        result = client.reset()
        print(f"Server {result.observation.session_hash}: PID {result.observation.pid}")

    # Send concurrent requests with different wait times
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, client in enumerate(clients):
            future = executor.submit(
                client.step,
                BenchmarkAction(wait_seconds=i + 1)
            )
            futures.append((client, future))

        for client, future in futures:
            result = future.result()
            print(f"Server {result.observation.session_hash} waited {result.observation.waited_seconds}s")

    # Clean up
    for client in clients:
        client.close()
```

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t benchmark-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring

## Environment Details

### Action
**BenchmarkAction**: Contains a single field
- `wait_seconds` (float) - Seconds to wait/sleep before returning (default: 0.0)

### Observation
**BenchmarkObservation**: Contains server identity and timing information
- `host_url` (str) - The URL of the server that handled the request
- `pid` (int) - Process ID of the server
- `session_hash` (str) - Unique 16-character hash identifying this server session
- `waited_seconds` (float) - Actual seconds waited
- `timestamp` (float) - Unix timestamp when observation was created
- `reward` (float) - Reward based on wait time
- `done` (bool) - Always False for benchmark environment
- `metadata` (dict) - Additional info

### Reward
The reward is calculated as: `1.0 / (1.0 + wait_seconds)`
- 0 seconds → reward: 1.0
- 1 second → reward: 0.5
- 2 seconds → reward: 0.33
- Encourages faster responses

## Advanced Usage

### Connecting to an Existing Server

If you already have a Benchmark environment server running, you can connect directly:

```python
from benchmark import BenchmarkEnv, BenchmarkAction

# Connect to existing server
benchmarkenv = BenchmarkEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = benchmarkenv.reset()
print(f"Connected to server: {result.observation.host_url}")
print(f"Session: {result.observation.session_hash}")

result = benchmarkenv.step(BenchmarkAction(wait_seconds=1.5))
print(f"Waited {result.observation.waited_seconds}s")
```

Note: When connecting to an existing server, `benchmarkenv.close()` will NOT stop the server.

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/benchmark_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Server identity is returned correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
benchmark/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # BenchmarkEnv client implementation
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── benchmark_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```
