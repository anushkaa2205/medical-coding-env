from models import Observation, Action
from typing import Tuple, Dict, Any

class MedicalCodingEnv:
    def __init__(self):
        self.tasks = [
            {"difficulty": "easy", "note": "25yo male. Sore throat. Strep test positive. Diagnosed Streptococcal pharyngitis.", "truth_primary": "J02.0", "truth_secondary": [], "truth_cpt": ["87880"]},
            {"difficulty": "medium", "note": "60yo female. History of Diabetes. High BP (150/95). Diagnosed hypertension. Routine A1c blood draw.", "truth_primary": "I10", "truth_secondary": ["E11.9"], "truth_cpt": ["36415", "83036"]},
            {"difficulty": "hard", "note": "35yo male. Fell from ladder. Right distal radius fracture. History of asthma. Closed reduction performed.", "truth_primary": "S52.501A", "truth_secondary": ["W11.XXXA", "J45.909"], "truth_cpt": ["25605", "73110"]}
        ]
        self.current_idx = 0

    def reset(self) -> Observation:
        self.current_idx = 0
        return self._get_obs("Ready")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.current_idx >= len(self.tasks):
            return self._get_obs("Done"), 0.0, True, {}
        
        task = self.tasks[self.current_idx]
        reward = 0.5 if action.primary_icd10 == task["truth_primary"] else 0.0
        reward += 0.5 if set(action.cpt_codes) == set(task["truth_cpt"]) else 0.0
        
        self.current_idx += 1
        return self._get_obs(f"Current Reward: {reward}"), reward, self.current_idx >= len(self.tasks), {}

    def _get_obs(self, feedback: str) -> Observation:
        if self.current_idx < len(self.tasks):
            t = self.tasks[self.current_idx]
            return Observation(patient_age=0, patient_sex="", clinical_note=t["note"], feedback=feedback, remaining_tasks=len(self.tasks)-self.current_idx)
        return Observation(patient_age=0, patient_sex="", clinical_note="DONE", feedback=feedback, remaining_tasks=0)