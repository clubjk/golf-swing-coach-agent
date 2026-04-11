# test_tool.py  (Root level)
from tools.swing_analyzer_tool import GolfSwingAnalyzerTool
import os

tool = GolfSwingAnalyzerTool()

# === CHANGE THIS TO YOUR ACTUAL VIDEO FILE NAME ===
video_filename = "sample_swing.mp4"        # ←←← UPDATE THIS

video_path = f"sample_swings/{video_filename}"

if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    print("Please put your iPhone video in the 'sample_swings' folder and update the filename above.")
else:
    print(f"🚀 Testing vision tool with: {video_path}")
    print("This may take 20-60 seconds...\n")
    
    result = tool._run(video_path, user_goal="fix my slice")
    print("Result:", result)
    
    print("✅ Tool finished!")
    print("="*60)
    print(result)