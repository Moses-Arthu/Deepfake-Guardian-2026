import cv2
import mediapipe as mp
import numpy as np
import librosa
from moviepy import VideoFileClip
import os

def extract_audio(video_path, audio_path="temp_audio.wav"):
    try:
        clip = VideoFileClip(video_path)
        if clip.audio:
            clip.audio.write_audiofile(audio_path, logger=None)
            clip.close()
            return True
        clip.close()
        return False
    except Exception:
        return False

def find_bilabial_timestamps(audio_path):
    """Detects M, B, P phonemes via spectral energy spikes."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        bilabial_times = []
        last_t = -1.0
        # Improved plosive detection: sharp rise from low energy
        for i in range(2, len(rms)-2):
            if rms[i] > rms[i-1] * 3.0 and rms[i-1] < 0.04:
                if times[i] - last_t >= 0.2:
                    bilabial_times.append(times[i])
                    last_t = times[i]
        return bilabial_times
    except Exception:
        return []

def process_video_audio_sync(video_path, output_path="output_audio_sync.mp4", progress_callback=None):
    audio_path = "temp_audio_sync.wav"
    has_audio = extract_audio(video_path, audio_path)
    
    phoneme_times = find_bilabial_timestamps(audio_path) if has_audio else []
    
    import urllib.request
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        try: urllib.request.urlretrieve(url, model_path)
        except: pass
            
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

    anomalies = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
            progress_callback(min(frame_idx / total_frames, 1.0))
            
        curr_t = frame_idx / fps
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        lip_open = False
        if results.face_landmarks:
            fl = results.face_landmarks[0]
            h, w, _ = frame.shape
            
            p13 = np.array([fl[13].x * w, fl[13].y * h])
            p14 = np.array([fl[14].x * w, fl[14].y * h])
            chin = np.array([fl[152].x * w, fl[152].y * h])
            forehead = np.array([fl[10].x * w, fl[10].y * h])
            face_h = np.linalg.norm(chin - forehead)
            lip_dist = np.linalg.norm(p13 - p14) / face_h if face_h > 0 else 0
            
            if lip_dist > 0.05: lip_open = True
            
            # Visualize lips
            p13_draw = (int(fl[13].x * w), int(fl[13].y * h))
            p14_draw = (int(fl[14].x * w), int(fl[14].y * h))
            cv2.circle(frame, p13_draw, 2, (255, 255, 0), -1)
            cv2.circle(frame, p14_draw, 2, (255, 255, 0), -1)
            cv2.line(frame, p13_draw, p14_draw, (255, 255, 0), 1)

            # Check for sync anomaly
            is_anomaly = False
            for pt in phoneme_times:
                if abs(curr_t - pt) < 0.15: # 150ms window
                    if lip_open:
                        is_anomaly = True
                        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 5)
                        cv2.putText(frame, "LIP-SYNC ANOMALY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if curr_t not in [a["time"] for a in anomalies]:
                            anomalies.append({"time": curr_t, "lip_dist": lip_dist})
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    if os.path.exists(audio_path): os.remove(audio_path)
    
    count = len(anomalies)
    conf = 0.0
    if count > 5: conf = 0.4
    if count > 12: conf = 0.8
    
    return {
        "anomalies_count": count,
        "anomalies": anomalies,
        "confidence_score": conf,
        "is_deepfake": conf > 0.6,
        "output_video": output_path
    }
