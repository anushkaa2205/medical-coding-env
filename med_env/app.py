import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from openenv.core.env_server.http_server import create_app
from med_env.environment import MedicalCodingEnv
from models import Action, Observation

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "100"))

app = create_app(
    MedicalCodingEnv,
    Action,
    Observation,
    env_name="medical-coding-env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()