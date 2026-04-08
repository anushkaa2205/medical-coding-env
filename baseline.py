import asyncio
from med_env.environment import MedicalCodingEnv, Action

async def run_mock_baseline():
    print("🚀 Starting MOCK Baseline (No API Key Required)...\n")

    env = MedicalCodingEnv()
    result = await env.reset()

    obs = result.observation
    total_score = 0.0
    done = False

    # Perfect answers
    answers = [
        Action(primary_icd10="J02.0", secondary_icd10s=[], cpt_codes=["87880"]),
        Action(primary_icd10="I10", secondary_icd10s=["E11.9"], cpt_codes=["36415", "83036"]),
        Action(primary_icd10="S52.501A", secondary_icd10s=["W11.XXXA", "J45.909"], cpt_codes=["25605", "73110"])
    ]

    task_idx = 0

    while not done:
        print(f"📝 Processing Note: {obs.clinical_note}")

        action = answers[task_idx]
        print(f"🤖 Mock Agent submitted: {action.primary_icd10}")

        result = await env.step(action)

        obs = result.observation
        reward = result.reward
        done = result.done

        total_score += reward

        if obs:
            print(f"⚖️  Grader Result: {obs.feedback}\n")

        task_idx += 1

    print(f"🏁 EVALUATION COMPLETE. Final Score: {total_score:.2f} / 3.00")


if __name__ == "__main__":
    asyncio.run(run_mock_baseline())