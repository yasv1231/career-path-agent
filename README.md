🌟 AI Career Path Recommendation Agent

🧠 Multi-Agent Career Guidance System using Python

Google AI Agents Intensive — Capstone Project

🚀 Problem:

Students often struggle to choose the right career because of confusion like:
->Which career suits my skills?
->What courses should I learn?
->What learning roadmap should I follow?
->How do I start, and what comes next?

💡 Solution:

This AI agent collects student details and provides:
✔ Best-fit career recommendation
✔ Required learning topics
✔ FREE high-quality courses with direct links
✔ 8-week personalized learning roadmap
✔ Smart memory — remembers past user profile

🧩 System Architecture:

User Input
    ↓
Career Agent        → Predicts best career
Course Agent        → Recommends learning topics
Tool Agent          → Fetches FREE course links
Roadmap Agent       → Builds 8-week learning plan
Evaluator Agent     → Explains why career matches the profile
Memory Manager      → Stores and retrieves user profile

🔥 Key Features (Google Score Requirements):

| Feature             | Status                                         |
| ------------------- | ---------------------------------------------- |
| Multi-Agent System  | ✔ Career + Course + Tool + Roadmap + Evaluator |
| Tools               | ✔ Free course link collector                   |
| Memory              | ✔ Stores and reuses user profile               |
| State Management    | ✔ Detects past session and continues           |
| Real-World Use Case | ✔ Student career guidance                      |
| CLI Prototype       | ✔ Fully working                                |


📂 Project Structure (root):

career_path_agent/
│ main.py
│ langgraph_flow.py
│ workflow/
│ career_agent.py
│ course_agent.py
│ tool_agent.py
│ roadmap_agent.py
│ evaluator_agent.py
│ memory_manager.py
│ rag_agent.py
│ rag_store.py
│ langsmith_integration.py
│ README.md
│ requirements.txt
│ input.txt
│ user_memory.json
│ mcp.config.json
│ visual_test.py
│ _local_langgraph.py
│
├─ skills/
│   ├─ extraction-skill/
│   ├─ memory-skill/
│   ├─ routing-skill/
│   └─ validation-skill/
│
└─ data/
    └─ rag_corpus.json

(Note: many agent modules live in the repository root for simplicity.)

🛠️ Run Locally:

Requirements
Python ≥ 3.8

Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

Execution:

```bash
python main.py
```

🖥️ Sample Output:

🎯 Recommended Career Path for you: Data Analyst
📚 Suggested Topics to Learn:
 - Excel for Data Analysis
 - SQL for Data Analysis
 - Python with Pandas & NumPy

🌐 FREE Courses to Start:
 - Google Data Analytics Certificate → Coursera
 - Excel for Data Analysis → FreeCodeCamp
 - SQL Full Course → YouTube

🗺️ Personalized 8-Week Roadmap

📝 Evaluation: Reason why this career matches your profile

🎯 Why This Project Matters:

This project gives students:
->a clear direction for their future
->course resources to start immediately
->step-by-step roadmap
->long-term guidance using memory

🔮 Future Enhancements:

->Streamlit Web UI
->Gemini / LLM conversational agent
->Live API search for jobs & courses
->Deployment using Cloud Run / Agent Engine

👩‍💻 Author:

Developed for: Google AI Agents Intensive (Capstone)
👤 Participant: Asapu Priyanjali Satya Sri

⭐ If you like this project, please give the repository a star!
🌟 AI Career Path Recommendation Agent

🧠 Multi-Agent Career Guidance System using Python

Google AI Agents Intensive — Capstone Project

🚀 Problem:

Students often struggle to choose the right career because of confusion like:
->Which career suits my skills?
->What courses should I learn?
->What learning roadmap should I follow?
->How do I start, and what comes next?

💡 Solution:

This AI agent collects student details and provides:
✔ Best-fit career recommendation
✔ Required learning topics
✔ FREE high-quality courses with direct links
✔ 8-week personalized learning roadmap
✔ Smart memory — remembers past user profile

🧩 System Architecture:

User Input
    ↓
Career Agent        → Predicts best career
Course Agent        → Recommends learning topics
Tool Agent          → Fetches FREE course links
Roadmap Agent       → Builds 8-week learning plan
Evaluator Agent     → Explains why career matches the profile
Memory Manager      → Stores and retrieves user profile

🔥 Key Features (Google Score Requirements):

| Feature             | Status                                         |
| ------------------- | ---------------------------------------------- |
| Multi-Agent System  | ✔ Career + Course + Tool + Roadmap + Evaluator |
| Tools               | ✔ Free course link collector                   |
| Memory              | ✔ Stores and reuses user profile               |
| State Management    | ✔ Detects past session and continues           |
| Real-World Use Case | ✔ Student career guidance                      |
| CLI Prototype       | ✔ Fully working                                |


📂 Project Structure:

career_path_agent/
│ main.py
│
├─ agents/
│   ├─ career_agent.py
│   ├─ course_agent.py
│   ├─ tool_agent.py
│   ├─ roadmap_agent.py
│   └─ evaluator_agent.py
│
└─ memory/
    └─ memory_manager.py

🛠️ Run Locally:

Requirements
Python ≥ 3.8

Execution:

python main.py

🖥️ Sample Output:

🎯 Recommended Career Path for you: Data Analyst
📚 Suggested Topics to Learn:
 - Excel for Data Analysis
 - SQL for Data Analysis
 - Python with Pandas & NumPy

🌐 FREE Courses to Start:
 - Google Data Analytics Certificate → Coursera
 - Excel for Data Analysis → FreeCodeCamp
 - SQL Full Course → YouTube

🗺️ Personalized 8-Week Roadmap

📝 Evaluation: Reason why this career matches your profile

🎯 Why This Project Matters:

This project gives students:
->a clear direction for their future
->course resources to start immediately
->step-by-step roadmap
->long-term guidance using memory

🔮 Future Enhancements:

->Streamlit Web UI
->Gemini / LLM conversational agent
->Live API search for jobs & courses
->Deployment using Cloud Run / Agent Engine

👩‍💻 Author:

Developed for: Google AI Agents Intensive (Capstone)
👤 Participant: Asapu Priyanjali Satya Sri

⭐ If you like this project, please give the repository a star!
