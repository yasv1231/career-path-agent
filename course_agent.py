# agents/course_agent.py

class CourseAgent:
    """
    CourseAgent:
    - Based on the target career, suggest a list of courses/topics.
    """

    def get_courses(self, career: str) -> list:
        c = career.lower()

        if "data analyst" in c:
            return [
                "Excel for Data Analysis",
                "SQL for Data Analysis",
                "Python with Pandas & NumPy",
                "Data Visualization with Power BI or Tableau",
                "Statistics for Data Analysis"
            ]

        if "data scientist" in c:
            return [
                "Python Programming",
                "Statistics & Probability",
                "Machine Learning (Supervised & Unsupervised)",
                "SQL for Data Science",
                "Data Visualization",
                "Deep Learning (optional)"
            ]

        if "ml engineer" in c:
            return [
                "Python & OOP",
                "Machine Learning Algorithms",
                "Deep Learning (CNNs, RNNs, Transformers)",
                "MLOps basics (deployment, monitoring)",
                "Cloud platforms (GCP/AWS basics)"
            ]

        # Fallback generic path
        return [
            "Python Basics",
            "Problem Solving & Logic Building",
            "Intro to Data Analysis",
            "Basic Statistics"
        ]
