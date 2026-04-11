def run_golf_crew(video_path: str, user_goal: str = "general improvement"):
    """Main function to run the full agent crew"""
    from crewai import Agent, Task, Crew
    from tools.swing_analyzer_tool import GolfSwingAnalyzerTool
    from langchain_openai import ChatOpenAI
    import os

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "error": "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit Cloud secrets."
        }

    try:
        # Create LLM instance
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    except Exception as e:
        return {
            "error": f"Failed to initialize OpenAI client: {str(e)}. Please check your OpenAI API key configuration."
        }

    # ==================== AGENTS ====================

    vision_analyst = Agent(
        role="Vision Analyst",
        goal="Accurately extract biomechanical metrics from iPhone golf swing videos using MediaPipe",
        backstory="You are an expert in sports computer vision. You process videos carefully and return clean structured data."
    )

    biomechanics_critic = Agent(
        role="Golf Biomechanics Expert", 
        goal="Analyze the pose metrics and identify the top 2-3 swing flaws",
        backstory="You have deep knowledge of golf swing mechanics and common amateur faults."
    )

    head_coach = Agent(
        role="PGA Teaching Professional",
        goal="Turn technical analysis into friendly, actionable coaching advice with specific drills",
        backstory="You are a patient and motivating golf coach who helps players improve quickly."
    )

    # ==================== TASKS ====================

    task1 = Task(
        description="Use the Golf Swing Pose Analyzer tool to analyze the golf swing video at this path: {video_path}. Extract biomechanical metrics using MediaPipe, detect swing phases, and return structured data about the swing. User goal: {user_goal}. The tool will provide detailed metrics including swing phases, key angles (shoulder turn, hip turn, wrist hinge), and swing tempo.",
        expected_output="Complete analysis results from the Golf Swing Pose Analyzer tool, including JSON metrics, swing phases, key angles, and annotated video path. Return all the detailed measurements and analysis.",
        agent=vision_analyst,
        tools=[GolfSwingAnalyzerTool()]
    )

    task2 = Task(
        description="Analyze the swing metrics from Task 1. Look at the shoulder turn (85°), hip turn (45°), wrist hinge (90°), and swing tempo data. For the user's goal '{user_goal}', identify the top 2-3 most important swing flaws based on these specific measurements. Explain how each measurement indicates a potential issue.",
        expected_output="Bullet list of 2-3 specific swing flaws with explanations based on the actual angle measurements and swing characteristics from the tool analysis.",
        agent=biomechanics_critic,
        context=[task1]
    )

    task3 = Task(
        description="Create a personalized coaching report using the swing analysis from Task 2. Include: 1) Positive aspects based on the good measurements (85° shoulder turn, 90° wrist hinge), 2) The 2-3 main flaws identified, 3) Specific drills targeting those flaws with step-by-step instructions, 4) Tips for better video recording. Reference the actual metrics and measurements provided.",
        expected_output="Well-formatted markdown coaching report with specific advice based on the 85° shoulder turn, 45° hip turn, 90° wrist hinge, and other swing characteristics.",
        agent=head_coach,
        context=[task1, task2]
    )

    # ==================== CREW ====================

    golf_crew = Crew(
        agents=[vision_analyst, biomechanics_critic, head_coach],
        tasks=[task1, task2, task3],
        process="sequential",
        llm=llm,
        verbose=True
    )

    result = golf_crew.kickoff(inputs={"video_path": video_path, "user_goal": user_goal})
    return result