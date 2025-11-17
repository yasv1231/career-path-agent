# agents/career_agent.py

class CareerAgent:
    """
    CareerAgent:
    - Takes the student profile
    - Infers the most suitable career path
    """

    def infer_career(self, profile: dict) -> str:
        skills = {s.lower() for s in profile.get("skills", [])}
        favorites = {f.lower() for f in profile.get("favorites", [])}
        interest = profile.get("interest", "").lower()

        # If user already has some idea
        if "data" in interest and "analyst" in interest:
            return "Data Analyst"
        if "science" in interest or "scientist" in interest:
            return "Data Scientist"
        if "ml" in interest or "machine learning" in interest:
            return "ML Engineer"

        # If user is not sure, infer from skills & favorites
        if {"python", "sql", "excel"} & skills:
            return "Data Analyst"
        if {"python", "ml", "machine learning"} & skills:
            return "Data Scientist"
        if {"deep learning", "computer vision", "nlp"} & skills:
            return "ML Engineer"

        if {"data structures", "algorithms"} & favorites:
            return "Software Engineer (with strong foundations in Data/AI if you wish)"

        if {"maths", "statistics"} & favorites:
            return "Data Analyst or Data Scientist"

        return "Data/AI related role (need more info from you 🙂)"
