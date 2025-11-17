# agents/roadmap_agent.py

class RoadmapAgent:
    """
    RoadmapAgent:
    - Builds a weekly learning roadmap based on target career and hours per week.
    """

    def build_roadmap(self, career: str, hours_str: str) -> str:
        try:
            hours = int(hours_str)
        except ValueError:
            hours = 5  # default

        if hours < 5:
            level = "Lite"
        elif hours <= 10:
            level = "Moderate"
        else:
            level = "Intensive"

        roadmap = f"""
Suggested Weekly Plan ({level} – ~{hours} hrs/week):

Week 1–2:
  - Learn Python basics (syntax, loops, functions)
  - Practice small coding problems daily

Week 3–4:
  - Learn basic Statistics & Data concepts
  - Start using libraries like Pandas, NumPy

Week 5–6:
  - Work on 1–2 mini projects related to {career}
  - Example: Sales Analysis, Student Performance Analysis

Week 7–8:
  - Create a GitHub profile and upload your projects
  - Prepare a simple resume highlighting {career} skills
"""
        return roadmap
