import streamlit as st
import os
from crew import run_golf_crew
from pathlib import Path

st.set_page_config(page_title="Agentic Golf Swing Coach", layout="wide", page_icon="⛳")

st.title("⛳ Agentic Golf Swing Coach")
st.markdown("**by @clubjk** — Agentic AI Course Assignment")
st.caption("Record on iPhone → Upload → Get critique from Vision Analyst + Biomechanics Expert + PGA Coach")

uploaded_file = st.file_uploader(
    "Upload your iPhone golf swing video (.mov or .mp4)",
    help="For best performance, keep videos under 10MB. If your video is larger, compress it using: ffmpeg -i input.mp4 -vf scale=640:-1 -c:v libx264 -crf 28 -preset fast output.mp4"
)

user_goal = st.text_input(
    "Your improvement goal (optional)",
    value="fix my slice",
    help="Example: fix slice, increase driver distance, stop early extension, better hip rotation"
)

if uploaded_file and st.button("🚀 Analyze Swing with Agent Crew"):
    # Check file type
    if not uploaded_file.name.lower().endswith(('.mp4', '.mov')):
        st.error("Please upload a .mp4 or .mov video file.")
        st.stop()
    
    with st.spinner("Running full agent crew... (Vision → Biomechanics → Coaching). This may take 45-90 seconds."):
        # Save uploaded file
        os.makedirs("/tmp/sample_swings", exist_ok=True)
        video_path = f"/tmp/sample_swings/{uploaded_file.name}"
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run the CrewAI
        result = run_golf_crew(video_path, user_goal)

        # Check if there was an error
        if isinstance(result, dict) and "error" in result:
            st.error(f"❌ Analysis Failed: {result['error']}")
            st.stop()

    st.success("✅ Analysis Complete!")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🏌️ Coaching Report")
        report = result.raw_output if hasattr(result, "raw_output") else str(result)
        st.markdown(report)

    with col2:
        st.subheader("📹 Annotated Video")
        # Try to show annotated video if path is available
        if hasattr(result, 'output') or 'annotated_video_path' in str(result):
            st.info("Annotated video saved to /tmp/outputs/ — check terminal for exact path")
        else:
            st.info("Annotated video will be available once the tool is fully integrated.")

    st.info("💡 **Tip**: For best results, record **face-on** or **down-the-line** with good lighting and full body visible.")

st.markdown("---")
st.caption("Built with MediaPipe + CrewAI • Running on Apple Silicon MacBook Pro")