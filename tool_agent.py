# agents/tool_agent.py
import webbrowser

class ToolAgent:
    """
    ToolAgent:
    - Recommends FREE courses as clickable links
    - Instead of API calls (for now), uses curated filtered no-cost resources
    """

    FREE_COURSE_DB = {
        "data analyst": [
            ("Google Data Analytics Certificate (Free Audit)", "https://www.coursera.org/professional-certificates/google-data-analytics"),
            ("Excel for Data Analysis – FreeCodeCamp", "https://www.freecodecamp.org/learn/data-analysis-with-excel/"),
            ("SQL Full Course – YouTube", "https://youtu.be/9Pzj7Aj25lw"),
            ("Python Pandas – FreeCodeCamp", "https://www.youtube.com/watch?v=vmEHCJofslg"),
            ("Power BI Full Course – YouTube", "https://youtu.be/9O2kT-1ZV2k"),
        ],
        "data scientist": [
            ("Python for Data Science – Kaggle", "https://www.kaggle.com/learn/python"),
            ("Machine Learning – Andrew Ng (Free Audit)", "https://www.coursera.org/learn/machine-learning"),
            ("Statistics for Data Science – YouTube", "https://youtu.be/xxpc-HPKN28"),
            ("Data Visualization – Kaggle", "https://www.kaggle.com/learn/data-visualization"),
        ],
        "ml engineer": [
            ("Deep Learning – FreeCodeCamp", "https://youtu.be/3BOoqH6x7vI"),
            ("TensorFlow Course – YouTube", "https://youtu.be/tPYj3fFJGjk"),
            ("Intro to Machine Learning – Kaggle", "https://www.kaggle.com/learn/intro-to-machine-learning"),
        ]
    }

    def get_free_courses(self, career):
        for key in self.FREE_COURSE_DB:
            if key in career.lower():
                return self.FREE_COURSE_DB[key]
        return []
