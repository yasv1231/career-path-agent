# memory/memory_manager.py

import json
import os


class MemoryManager:
    """
    MemoryManager:
    - Stores and retrieves the last user profile in a JSON file.
    - This simulates long-term memory for the agent.
    """

    def __init__(self, path: str = "user_memory.json"):
        self.path = path

    def load_last_profile(self):
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_profile(self, profile: dict):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Memory] Failed to save profile: {e}")
