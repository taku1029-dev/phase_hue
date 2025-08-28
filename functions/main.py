from typing import List
from firebase_functions import firestore_fn, https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app, firestore
import google.cloud.firestore

# Mediapipe modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Deal with images using opencv
import cv2
import tempfile
import os
import numpy as np

# Select mediapipe model_path
model_path = "dev/phase_hue/prod-env/pose_landmarker_heavy.task"
# Mediapipe Config
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# OpenCV Config


# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-function
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=10)

initialize_app()

@https_fn.on_request()
def upload_video(req: https_fn.Request) -> https_fn.Response:
    if req.method != 'POST':
        return https_fn.Response("POSTのみ対応", status=405)

    video_file = req.files.get('video')

    if video_file is None:
        return https_fn.Response("Video file is missing...", status=400)

    video_content = video_file.read()
    filename = video_file.filename

    poseList = process_video(video_content)


    return https_fn.Response("アップロード完了")

def process_video(video_binary):
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_binary)
            tmp_file.flush()
            tmp_file_path = tmp_file.name

        # Check File siez
        file_size = os.path.getsize(tmp_file_path)
        print(f"一時ファイルサイズ: {file_size} バイト")

        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            raise RuntimeError("Couldn't open the video file")

        poseList = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            print(f"Processing {frame_count}th frame...")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Each frame timestamp of 30fps video
            frame_timestamp_ms = 33 * frame_count
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            poseList.append(pose_landmarker_result)

        return poseList


# Draw landmark
# def draw_landmarks_on_image(rgb_image, detection_result):
#   pose_landmarks_list = detection_result.pose_landmarks
#   annotated_image = np.copy(rgb_image)
#
#   # Loop through the detected poses to visualize.
#   for idx in range(len(pose_landmarks_list)):
#     pose_landmarks = pose_landmarks_list[idx]
#
#     # Draw the pose landmarks.
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     pose_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       pose_landmarks_proto,
#       solutions.pose.POSE_CONNECTIONS,
#       solutions.drawing_styles.get_default_pose_landmarks_style())
#   return annotated_image
