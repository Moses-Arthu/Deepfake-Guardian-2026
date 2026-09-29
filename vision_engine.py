import cv2
import mediapipe as mp
import numpy as np
import os

def calculate_ear(eye_points):
    """EAR = (||p2 - p6|| + ||p3 - p5||) / (2||p1 - p4||)"""
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4) if p1_p4 > 0 else 0

def process_video_vision(video_path, output_path="output_vision.mp4", progress_callback=None):
    import urllib.request
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception:
            pass

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    NOSE_TIP = 1

    EAR_THRESHOLD = 0.20
    CONSECUTIVE_FRAMES = 2

    blink_count = 0
    frame_counter = 0
    ear_values = []
    nose_positions = []
    jitter_values = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
            progress_callback(min(frame_idx / total_frames, 1.0))
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.face_landmarks:
            fl = results.face_landmarks[0]
            h, w, _ = frame.shape
            
            # --- EAR ---
            left_pts = np.array([[fl[p].x * w, fl[p].y * h] for p in LEFT_EYE])
            right_pts = np.array([[fl[p].x * w, fl[p].y * h] for p in RIGHT_EYE])
            ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0
            ear_values.append(ear)
            
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                frame_counter = 0
            
            # --- Jitter (visualize) ---
            nose_pos = (int(fl[NOSE_TIP].x * w), int(fl[NOSE_TIP].y * h))
            if nose_positions and nose_positions[-1] is not None:
                prev = nose_positions[-1]
                dist = np.sqrt((nose_pos[0]-prev[0])**2 + (nose_pos[1]-prev[1])**2)
                jitter_values.append(dist)
                # Draw jitter line
                cv2.line(frame, prev, nose_pos, (0, 0, 255), 2)
            nose_positions.append(nose_pos)

            # Draw eyes
            cv2.polylines(frame, [left_pts.astype(int)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_pts.astype(int)], True, (0, 255, 0), 1)
            
            # UI Text
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"BLINKS: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if jitter_values:
                cv2.putText(frame, f"JITTER: {np.mean(jitter_values):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            ear_values.append(0.0)
            nose_positions.append(None)
                
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    duration_min = frame_idx / (fps * 60.0) if fps > 0 else 0
    bpm = blink_count / duration_min if duration_min > 0 else 0
    non_zero_ears = [e for e in ear_values if e > 0.05]
    ear_std = float(np.std(non_zero_ears)) if len(non_zero_ears) > 10 else 0.05
    
    # Simple suspicion score for visual feedback (ML engine will override)
    v_conf = 0.0
    if bpm < 10 or bpm > 40: v_conf += 0.4
    if ear_std < 0.015: v_conf += 0.4

    return {
        "blinks_per_minute": bpm,
        "blink_count": blink_count,
        "ear_values": ear_values,
        "ear_std_dev": ear_std,
        "confidence_score": min(v_conf, 1.0),
        "is_deepfake": v_conf > 0.6,
        "fps": fps,
        "output_video": output_path
    }
