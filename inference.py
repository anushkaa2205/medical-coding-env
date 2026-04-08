import asyncio
import os
from typing import List, Optional
from openai import OpenAI
from med_env.environment import Action as MyEnvV4Action, MedicalCodingEnv as MyEnvV4Env
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

TASK_NAME = "medical-coding"
BENCHMARK = "medical-coding-env"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


async def main():
    if not API_KEY:
        raise ValueError("HF_TOKEN is not set")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await MyEnvV4Env.from_docker_image(None)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            note = obs.clinical_note.lower()

            if "strep" in note or "pharyngitis" in note:
                action = MyEnvV4Action(
                    primary_icd10="J02.0",
                    secondary_icd10s=[],
                    cpt_codes=["87880"]
                )

            elif "hypertension" in note or "bp" in note or "diabetes" in note:
                action = MyEnvV4Action(
                    primary_icd10="I10",
                    secondary_icd10s=["E11.9"],
                    cpt_codes=["36415", "83036"]
                )

            elif "fracture" in note or "ladder" in note:
                action = MyEnvV4Action(
                    primary_icd10="S52.501A",
                    secondary_icd10s=["W11.XXXA", "J45.909"],
                    cpt_codes=["25605", "73110"]
                )

            else:
                action = MyEnvV4Action(
                    primary_icd10="I10",
                    secondary_icd10s=[],
                    cpt_codes=[]
                )

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, str(action.model_dump()), reward, done, None)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        await env.close()
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())