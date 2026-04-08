import asyncio
from med_env.environment import MedicalCodingEnv, Action

async def run_mock_baseline():
    print("Starting MOCK Baseline")

    env = MedicalCodingEnv()
    result = await env.reset()

    obs = result.observation
    total_score = 0.0
    done = False
    step_count = 0

    while not done:
        print(f"Processing Note: {obs.clinical_note}")

        note = obs.clinical_note.lower()

        if "strep" in note or "pharyngitis" in note:
            action = Action(primary_icd10="J02.0", secondary_icd10s=[], cpt_codes=["87880"])

        elif "hypertension" in note or "bp" in note or "diabetes" in note:
            action = Action(primary_icd10="I10", secondary_icd10s=["E11.9"], cpt_codes=["36415", "83036"])

        elif "fracture" in note or "ladder" in note:
            action = Action(primary_icd10="S52.501A", secondary_icd10s=["W11.XXXA", "J45.909"], cpt_codes=["25605", "73110"])

        else:
            break

        print(f"Submitted: {action.primary_icd10}")

        result = await env.step(action)

        obs = result.observation
        reward = result.reward
        done = result.done

        total_score += reward

        if obs:
            print(f"Grader Result: {obs.feedback}")

        step_count += 1

        if step_count > 5:
            break

    print(f"Final Score: {total_score:.2f} / 3.00")


if __name__ == "__main__":
    asyncio.run(run_mock_baseline())