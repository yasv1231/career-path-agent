# main.py
# Multi-Agent Career Path Recommendation System (Console Version)
# This version uses:
#  - CareerAgent
#  - CourseAgent
#  - RoadmapAgent
#  - EvaluatorAgent
#  - MemoryManager (simple long-term memory)
from agents.tool_agent import ToolAgent
from agents.career_agent import CareerAgent
from agents.course_agent import CourseAgent
from agents.roadmap_agent import RoadmapAgent
from agents.evaluator_agent import EvaluatorAgent
from memory.memory_manager import MemoryManager

def ask_user_profile():
    print("👋 Welcome to the AI Career Path Agent!\n")

    education = input("1) What is your current education/branch? (e.g., B.Tech CSE, AI&DS): ")
    favorites = input("2) What are your favorite subjects or topics? (comma separated): ")
    skills = input("3) What technical skills do you know? (e.g., Python, SQL, Excel): ")
    interest = input("4) What career are you interested in? (e.g., Data Analyst, Data Scientist, ML Engineer, Not sure): ")
    hours = input("5) How many hours per week can you spend on learning? (number): ")

    profile = {
        "education": education.strip(),
        "favorites": [x.strip().lower() for x in favorites.split(",") if x.strip()],
        "skills": [x.strip().lower() for x in skills.split(",") if x.strip()],
        "interest": interest.strip().lower(),
        "hours_per_week": hours.strip()
    }
    return profile


def main():
    memory = MemoryManager()
    last_profile = memory.load_last_profile()

    if last_profile:
        print("👋 Welcome back! I found your last saved profile.\n")
        print(f"Last education: {last_profile.get('education', 'N/A')}")
        print(f"Last interest : {last_profile.get('interest', 'N/A')}")
        use_last = input("\nDo you want to reuse this profile? (y/n): ").strip().lower()
        if use_last == "y":
            profile = last_profile
        else:
            profile = ask_user_profile()
    else:
        profile = ask_user_profile()

    # Save latest profile to memory
    memory.save_profile(profile)

    # Initialize agents
    career_agent = CareerAgent()
    course_agent = CourseAgent()
    roadmap_agent = RoadmapAgent()
    evaluator_agent = EvaluatorAgent()
    tool_agent = ToolAgent()

    print("\n🔍 Analyzing your profile with multiple agents...\n")

    # Multi-agent flow
    career = career_agent.infer_career(profile)
    courses = course_agent.get_courses(career)
    roadmap = roadmap_agent.build_roadmap(career, profile["hours_per_week"])
    evaluation = evaluator_agent.evaluate(profile, career)

    # Output
    print(f"🎯 Recommended Career Path for you: {career}\n")

    print("📚 Suggested Courses / Topics to Learn:")
    for c in courses:
        print(f"  - {c}")
    print("\n🌐 FREE Courses you can start:")
    free_courses = tool_agent.get_free_courses(career)
    for name, url in free_courses:
       print(f"  - {name} → {url}")


    print(roadmap)

    print("📝 Evaluation & Guidance:")
    print(evaluation)

    print("\n✅ Next Step: Start with the suggested courses and follow the roadmap step by step.")
    print("   You can run this agent again later — it will remember your profile and adjust.\n")
    print("(This is the Multi-Agent + Memory console version. In the capstone writeup,")
    print(" we will describe how to extend this with tools and deployment according to Google’s instructions.)")


if __name__ == "__main__":
    main()
