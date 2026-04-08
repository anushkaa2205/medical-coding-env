from models import Observation, Action, StepResult
import random

class MedicalCodingEnv:
    def __init__(self):
        self.tasks = [
            {
                "difficulty": "easy",
                "note": "25yo male. Sore throat. Strep test positive. Diagnosed Streptococcal pharyngitis.",
                "truth_primary": "J02.0",
                "truth_secondary": [],
                "truth_cpt": ["87880"]
            },
            {
                "difficulty": "medium",
                "note": "60yo female. History of Diabetes. High BP (150/95). Diagnosed hypertension. Routine A1c blood draw.",
                "truth_primary": "I10",
                "truth_secondary": ["E11.9"],
                "truth_cpt": ["36415", "83036"]
            },
            {
                "difficulty": "hard",
                "note": "35yo male. Fell from ladder. Right distal radius fracture. History of asthma. Closed reduction performed.",
                "truth_primary": "S52.501A",
                "truth_secondary": ["W11.XXXA", "J45.909"],
                "truth_cpt": ["25605", "73110"]
            }
        ]
        self.current_idx = 0
    async def reset_async(self):
        return await self.reset()

    async def step_async(self, action):
        return await self.step(action)
    @classmethod
    async def from_docker_image(cls, image_name):
        return cls()

    async def reset(self):
        random.shuffle(self.tasks)
        self.current_idx = 0
        return StepResult(
            observation=self._get_obs("Ready"),
            reward=0.0,
            done=False
        )

    async def step(self, action: Action):
        if self.current_idx >= len(self.tasks):
            return StepResult(observation=None, reward=0.0, done=True)

        task = self.tasks[self.current_idx]
        reward = 0.0
        if action.primary_icd10[:3] == task["truth_primary"][:3]:
            reward += 0.25
        if action.primary_icd10 == task["truth_primary"]:
            reward += 0.25
        true_secondary = set(task["truth_secondary"])
        pred_secondary = set(action.secondary_icd10s)
        if true_secondary:
            reward += 0.2 * (len(true_secondary & pred_secondary) / len(true_secondary))
        true_cpt = set(task["truth_cpt"])
        pred_cpt = set(action.cpt_codes)
        if true_cpt:
            reward += 0.3 * (len(true_cpt & pred_cpt) / len(true_cpt))
        if len(pred_secondary) > len(true_secondary) + 2:
            reward -= 0.1
        if len(pred_cpt) > len(true_cpt) + 2:
            reward -= 0.1
        reward = max(0.0, min(reward, 1.0))
        self.current_idx += 1
        done = self.current_idx >= len(self.tasks)
        reward += random.uniform(-0.02, 0.02)
        reward = max(0.0, min(reward, 1.0))
        return StepResult(
            observation=None if done else self._get_obs(f"Score: {reward:.2f}"),
            reward=reward,
            done=done
        )

    def _get_obs(self, feedback: str) -> Observation:
        if self.current_idx < len(self.tasks):
            t = self.tasks[self.current_idx]

            variations = [
                t["note"],
                t["note"].replace(".", " with symptoms."),
                t["note"] + " Patient appears stable.",
            ]

            import random
            note = random.choice(variations)

            return Observation(
                patient_age=0,
                patient_sex="",
                clinical_note=note,
                feedback=feedback,
                remaining_tasks=len(self.tasks) - self.current_idx
            )   

    async def close(self):
        pass