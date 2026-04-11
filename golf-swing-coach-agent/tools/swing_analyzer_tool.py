from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
from typing import Dict, Any

class SwingAnalyzerInput(BaseModel):
    video_path: str = Field(..., description="Full path to the iOS golf swing video")
    user_goal: str = Field("general improvement", description="User's goal (fix slice, more distance, etc.)")

class GolfSwingAnalyzerTool(BaseTool):
    name: str = "Golf Swing Pose Analyzer"
    description: str = "Uses MediaPipe to analyze golf swing video from iPhone. Returns angles and annotated video."
    args_schema: type[BaseModel] = SwingAnalyzerInput

    def _run(self, video_path: str, user_goal: str = "general improvement") -> Dict[str, Any]:
        print(f"🔍 Starting video analysis for: {video_path}")
        print(f"🎯 User goal: {user_goal}")
        
        try:
            import cv2
            import mediapipe as mp
            from utils.pose_utils import calculate_angle, detect_swing_phases
            print("✅ Imports successful")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return {"error": f"Failed to import required libraries: {e}"}
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return {"error": f"Video not found: {video_path}"}

        print(f"📹 Found video file: {video_path}")
        
        try:
            # Set headless backend for OpenCV
            cv2.setUseOptimized(True)
            if hasattr(cv2, 'ocl'):
                cv2.ocl.setUseOpenCL(False)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("⚠️  Primary VideoCapture failed, trying with FFMPEG")
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print("❌ Video capture failed to open")
                return {"error": "Could not open video file - unsupported format or corrupted file"}

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(3))
            height = int(cap.get(4))
            print(f"📐 Video specs: {width}x{height} @ {fps} fps")

            # Initialize MediaPipe with headless settings
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                model_complexity=1, 
                min_detection_confidence=0.5,
                static_image_mode=False
            )
            print("🤖 MediaPipe pose model initialized")

            os.makedirs("/tmp/outputs", exist_ok=True)
            output_path = f"/tmp/outputs/annotated_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4"
            print(f"💾 Output path: {output_path}")

            # Use a more compatible codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                print("⚠️  H.264 failed, trying MP4V")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frames_data = []
            frame_num = 0
            processed_frames = 0

            while cap.isOpened() and frame_num < 300:  # Limit to 300 frames for testing
                ret, frame = cap.read()
                if not ret:
                    print(f"📋 Finished reading {frame_num} frames")
                    break

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)

                    if results.pose_landmarks:
                        # Draw skeleton
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        landmarks = results.pose_landmarks.landmark
                        try:
                            wrist_hinge = calculate_angle(
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                            )
                            frames_data.append({"frame": frame_num, "wrist_hinge": wrist_hinge})
                            processed_frames += 1
                        except Exception as e:
                            print(f"⚠️  Error calculating angle: {e}")

                    out.write(frame)
                    frame_num += 1
                except Exception as e:
                    print(f"⚠️  Error processing frame {frame_num}: {e}")
                    frame_num += 1
                    continue

            cap.release()
            out.release()
            pose.close()
            print(f"✅ Processed {processed_frames} frames with pose data out of {frame_num} total frames")

            phases = detect_swing_phases(frames_data)

            result = {
                "status": "success",
                "metrics": {
                    "phases": phases,
                    "annotated_video_path": output_path,
                    "frame_count": frame_num,
                    "processed_frames": processed_frames
                },
                "annotated_video_path": output_path,
                "message": f"Successfully analyzed {frame_num} frames, {processed_frames} with pose data. Goal: {user_goal}"
            }
            print(f"🎉 Analysis complete: {result['message']}")
            return result
            
        except Exception as e:
            print(f"💥 Unexpected error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed due to technical issues: {str(e)}. This may be due to missing system libraries in the cloud environment."}