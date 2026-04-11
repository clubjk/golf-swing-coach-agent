def run_golf_crew(video_path: str, user_goal: str = "general improvement"):
    """Main function to run the full agent crew"""
    from crewai import Agent, Task, Crew
    from tools.swing_analyzer_tool import GolfSwingAnalyzerTool
    from langchain_openai import ChatOpenAI

    # Create LLM instance
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # ==================== AGENTS ====================

    vision_analyst = Agent(
        role="Vision Analyst",
        goal="Accurately extract biomechanical metrics from iPhone golf swing videos using MediaPipe",
        backstory="You are an expert in sports computer vision. You process videos carefully and return clean structured data.",
        tools=[GolfSwingAnalyzerTool()],
        llm=llm,
        verbose=True
    )

    biomechanics_critic = Agent(
        role="Golf Biomechanics Expert",
        goal="Analyze the pose metrics and identify the top 2-3 swing flaws",
        backstory="You have deep knowledge of golf swing mechanics and common amateur faults.",
        llm=llm,
        verbose=True
    )

    head_coach = Agent(
        role="PGA Teaching Professional",
        goal="Turn technical analysis into friendly, actionable coaching advice with specific drills",
        backstory="You are a patient and motivating golf coach who helps players improve quickly.",
        llm=llm,
        verbose=True
    )

    # ==================== TASKS ====================

    task1 = Task(
        description="Analyze the golf swing video at this path: {video_path}. User goal: {user_goal}. Return structured metrics and the path to the annotated video.",
        expected_output="JSON containing phases, key angles, annotated_video_path, and summary.",
        agent=vision_analyst
    )

    task2 = Task(
        description="Using the metrics from Task 1 and the user's goal '{user_goal}', identify the top 2-3 most important swing flaws with explanations.",
        expected_output="Bullet list of flaws with supporting metrics.",
        agent=biomechanics_critic,
        context=[task1]
    )

    task3 = Task(
        description="Create a friendly, encouraging coaching report. Include strengths, main flaws, 1-2 specific drills with instructions, and recording tips for next time. Reference the annotated video.",
        expected_output="Well-formatted markdown coaching report.",
        agent=head_coach,
        context=[task1, task2]
    )

    # ==================== CREW ====================

    golf_crew = Crew(
        agents=[vision_analyst, biomechanics_critic, head_coach],
        tasks=[task1, task2, task3],
        process="sequential",
        verbose=True
    )

    result = golf_crew.kickoff(inputs={"video_path": video_path, "user_goal": user_goal})
    return result