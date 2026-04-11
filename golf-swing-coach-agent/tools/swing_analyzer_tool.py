from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import mediapipe as mp
import os
from typing import Dict, Any
from utils.pose_utils import calculate_angle, detect_swing_phases

class SwingAnalyzerInput(BaseModel):
    video_path: str = Field(..., description="Full path to the iOS golf swing video")
    user_goal: str = Field("general improvement", description="User's goal (fix slice, more distance, etc.)")

class GolfSwingAnalyzerTool(BaseTool):
    name: str = "Golf Swing Pose Analyzer"
    description: str = "Uses MediaPipe to analyze golf swing video from iPhone. Returns angles and annotated video."
    args_schema: type[BaseModel] = SwingAnalyzerInput

    def _run(self, video_path: str, user_goal: str = "general improvement") -> Dict[str, Any]:
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(3))
        height = int(cap.get(4))

        print(f"🍎 Processing video on Apple Silicon: {width}x{height} @ {fps} fps")

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

        os.makedirs("/tmp/outputs", exist_ok=True)
        output_path = f"/tmp/outputs/annotated_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_data = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                # Draw skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                wrist_hinge = calculate_angle(
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                )

                frames_data.append({"frame": frame_num, "wrist_hinge": wrist_hinge})

            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()

        phases = detect_swing_phases(frames_data)

        return {
            "status": "success",
            "metrics": {
                "phases": phases,
                "annotated_video_path": output_path,
                "frame_count": frame_num
            },
            "annotated_video_path": output_path,
            "message": f"Analyzed {frame_num} frames. Goal: {user_goal}"
        }