# agents/evaluator_agent.py

class EvaluatorAgent:
    """
    EvaluatorAgent:
    - Evaluates whether the recommended career matches the user's interest and skills.
    - For now it only prints a simple evaluation message.
    """

    def evaluate(self, profile: dict, career: str) -> str:
        interest = profile.get("interest", "").lower()
        skills = {s.lower() for s in profile.get("skills", [])}

        remarks = []

        if "not sure" in interest or interest.strip() == "":
            remarks.append("You were not sure about your career. This recommendation is based on your skills and favorite subjects.")

        if "data" in career.lower() and ({"java", "ds"} & skills):
            remarks.append("You already know Java and Data Structures – great foundation. You can start adding Python and SQL for data roles.")

        if not remarks:
            remarks.append("The recommended path looks reasonable based on the information you provided.")

        remarks.append("You can refine your profile later and the agent will adjust the recommendations.")

        return "\n".join(f"- {r}" for r in remarks)
