import json
from server.environment import MedicalCodingEnv
from models import Action

def run_mock_baseline():
    print("🚀 Starting MOCK Baseline (No API Key Required)...\n")
    env = MedicalCodingEnv()
    obs = env.reset()
    total_score = 0.0
    done = False
    
    # These are the "perfect" answers the AI would eventually find
    answers = [
        Action(primary_icd10="J02.0", secondary_icd10s=[], cpt_codes=["87880"]),
        Action(primary_icd10="I10", secondary_icd10s=["E11.9"], cpt_codes=["36415", "83036"]),
        Action(primary_icd10="S52.501A", secondary_icd10s=["W11.XXXA", "J45.909"], cpt_codes=["25605", "73110"])
    ]
    
    task_idx = 0
    while not done:
        print(f"📝 Processing Note: {obs.clinical_note}")
        
        # Use the hardcoded answer instead of calling OpenAI
        action = answers[task_idx]
        print(f"🤖 Mock Agent submitted: {action.primary_icd10}")
        
        obs, reward, done, info = env.step(action)
        total_score += reward
        print(f"⚖️  Grader Result: {obs.feedback}\n")
        
        task_idx += 1

    print(f"🏁 EVALUATION COMPLETE. Final Score: {total_score:.2f} / 3.00")

if __name__ == "__main__":
    run_mock_baseline()