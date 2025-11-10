# ...existing code...
import cv2
import mediapipe as mp
import numpy as np
import logging

# Configure logging (reduces Mediapipe info; some absl warnings may still appear)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Drawing and pose utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle calculation helper
def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)  # First
    b = np.array(b, dtype=float)  # Mid
    c = np.array(c, dtype=float)  # End

    # Calculate the angle (in degrees) at point b between lines ba and bc
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Access webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and find pose landmarks
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width = image.shape[:2]

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get pixel coordinates for left shoulder, elbow, wrist & right counterparts
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height
            ]
            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height
            ]
            left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height
            ]

            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height
            ]

            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height
            ]

            left_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height
            ]

            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height
            ]
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height
            ]
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height
            ]

            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height
            ]

            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height
            ]

            right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height
            ]


            # Calculate angle at the elbow
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            # Calculate angle at the shoulder
            left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_wrist)
            right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_wrist)
            # Calculate angle at the hip
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            # Calculate angle at the hip
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Draw the pose landmarks (use normalized landmarks stored in results)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Display angle near the elbow
            left_elbow_pos = (int(left_elbow[0]), int(left_elbow[1]))
            cv2.putText(
                image,
                f"{int(left_elbow_angle)} deg",
                (left_elbow_pos[0] + 5, left_elbow_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

             # Display angle near the shoulder
            left_shoulder_pos = (int(left_shoulder[0]), int(left_shoulder[1]))
            cv2.putText(
                image,
                f"{int(left_shoulder_angle)} deg",
                (left_shoulder_pos[0] + 5, left_shoulder_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Display angle near the Hip
            left_hip_pos = (int(left_hip[0]), int(left_hip[1]))
            cv2.putText(
                image,
                f"{int(left_hip_angle)} deg",
                (left_hip_pos[0] + 5, left_hip_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Display angle near the knee
            left_knee_pos = (int(left_knee[0]), int(left_knee[1]))
            cv2.putText(
                image,
                f"{int(left_knee_angle)} deg",
                (left_knee_pos[0] + 5, left_knee_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            right_elbow_pos = (int(right_elbow[0]), int(right_elbow[1]))
            cv2.putText(
                image,
                f"{int(right_elbow_angle)} deg",
                (right_elbow_pos[0] + 5, right_elbow_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # Display angle near the shoulder
            right_shoulder_pos = (int(right_shoulder[0]), int(right_shoulder[1]))
            cv2.putText(
                image,
                f"{int(right_shoulder_angle)} deg",
                (right_shoulder_pos[0] + 5, right_shoulder_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

             # Display angle near the Hip
            right_hip_pos = (int(right_hip[0]), int(right_hip[1]))
            cv2.putText(
                image,
                f"{int(right_hip_angle)} deg",
                (right_hip_pos[0] + 5, right_hip_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

             # Display angle near the Knee
            right_knee_pos = (int(right_knee[0]), int(right_knee[1]))
            cv2.putText(
                image,
                f"{int(right_knee_angle)} deg",
                (right_knee_pos[0] + 5, right_knee_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Also print angle to console
            print(f"Left elbow angle: {left_elbow_angle:.2f} degrees")
            print(f"Right elbow angle: {right_elbow_angle:.2f} degrees")
             # Also print shoulder angle to console
            print(f"Left elbow angle: {left_shoulder_angle:.2f} degrees")
            print(f"Right elbow angle: {right_shoulder_angle:.2f} degrees")
            # Also print hip angle to console
            print(f"Left hip angle: {left_hip_angle:.2f} degrees")
            print(f"Right hip angle: {right_hip_angle:.2f} degrees")
            # Also print hip angle to console
            print(f"Left knee angle: {left_knee_angle:.2f} degrees")
            print(f"Right knee angle: {right_knee_angle:.2f} degrees")

        cv2.imshow('Dance Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()