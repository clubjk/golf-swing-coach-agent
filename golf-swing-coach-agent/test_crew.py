from crew import run_golf_crew

try:
    # Just test import and crew creation (crew is created at module level)
    print("Crew imported and created successfully!")
    # To test run, but it needs video and API key
    # result = run_golf_crew("sample_swings/sample_swing.mp4")
    # print("Crew run successful!")
except Exception as e:
    print(f"Error: {e}")