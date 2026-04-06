import os
from fastapi import FastAPI

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'pip install openenv'"
    ) from e

# --- THE FIX IS HERE ---
from models import Action, Observation  # Use our new names
from .environment import MedicalCodingEnv

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "100"))

# Create the app
app = create_app(
    MedicalCodingEnv, 
    Action,           # Changed from BenchmarkAction
    Observation,      # Changed from BenchmarkObservation
    env_name="medical-coding-env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)