import streamlit as st
import os
import sys
import json

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision_engine import process_video_vision
from audio_engine import process_video_audio_sync
from report_generator import generate_report
from classifier import predict_video, load_model
from feature_extractor import FEATURE_NAMES
import yt_dlp

def download_video_from_url(url, output_path, progress_bar):
    ydl_opts = {
        # Prefer a single file that contains both video and audio to avoid needing ffmpeg for merging
        'format': 'best[ext=mp4]/best', 
        'outtmpl': output_path.replace('.mp4', ''),
        'overwrites': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    # Progress hook for yt-dlp
    def progress_hook(d):
        if d['status'] == 'downloading':
            p = d.get('_percent_str', '0%').replace('%','')
            try:
                progress_bar.progress(float(p)/100, text=f"Downloading video... {p}%")
            except:
                pass
        elif d['status'] == 'finished':
            progress_bar.progress(1.0, text="Download complete! Finalizing...")

    ydl_opts['progress_hooks'] = [progress_hook]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Ensure it ends with .mp4 as requested by outtmpl if yt-dlp didn't rename it
        actual_file = ydl.prepare_filename(info)
        if not actual_file.endswith('.mp4') and os.path.exists(actual_file):
             # This can happen if format merge fails or similar
             pass 
        return info.get('title', 'URL Video')

st.set_page_config(page_title="Deepfake Guardian 2026", layout="wide")

# Custom Dark Theme + Styling
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-container {
        background-color: #161b22;
        border-right: 4px solid #58a6ff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #1f6feb;
        color: white;
    }
    .feature-card {
        background-color: #161b22;
        padding: 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Deepfake Guardian 2026")
st.markdown("### Forensic Synthetic Media Analysis Framework")
st.markdown("Upload a video to analyze using **ML-powered multi-signal forensic detection** with DCT frequency analysis, LBP texture analysis, optical flow, and biological markers.")

# --- Sidebar: Model Status & Calibration ---
with st.sidebar:
    st.markdown("## ⚙️ Model Control")

    model, meta = load_model()
    if model is not None and meta is not None:
        accuracy = meta.get("accuracy", 0)
        color = "🟢" if accuracy >= 0.9 else "🟡" if accuracy >= 0.7 else "🔴"
        st.success(f"{color} Model Loaded")
        st.metric("Training Accuracy", f"{accuracy*100:.1f}%")
        st.caption(f"Trained on {meta.get('num_real', '?')} real + {meta.get('num_fake', '?')} fake videos")
        st.caption(f"F1 Score: {meta.get('f1_score', 0)*100:.1f}%")
    else:
        st.warning("⚠️ No trained model found. Using heuristic fallback.")
        st.caption("Click 'Calibrate Model' to train on your dataset.")

    st.markdown("---")

    if st.button("🔬 CALIBRATE MODEL", use_container_width=True):
        st.info("Starting calibration on dataset...")
        with st.spinner("Extracting features and training model... This may take several minutes."):
            try:
                from calibration_engine import run_calibration
                metrics = run_calibration(force_reextract=False)
                if metrics:
                    st.success(f"✅ Calibration complete! Accuracy: {metrics['accuracy']*100:.1f}%")
                    st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                    st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Calibration failed. Check dataset directory.")
            except Exception as e:
                st.error(f"Calibration error: {e}")

    if st.button("🔄 FORCE RE-EXTRACT", use_container_width=True):
        st.info("Re-extracting all features from scratch...")
        with st.spinner("This will take longer as every video is reprocessed..."):
            try:
                from calibration_engine import run_calibration
                metrics = run_calibration(force_reextract=True)
                if metrics:
                    st.success(f"✅ Done! Accuracy: {metrics['accuracy']*100:.1f}%")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📊 Detection Signals")
    st.markdown("""
    - 👁️ **EAR Blink Analysis** — Lyu-Siwei Method
    - 🔬 **DCT Frequency Scan** — GAN artifact detection
    - 🧱 **LBP Texture Analysis** — Skin micro-texture
    - 🌊 **Optical Flow** — Temporal coherence
    - 📍 **Landmark Jitter** — Face stability
    - 👄 **Lip-Sync Analysis** — Phoneme-Viseme check
    - 🎵 **Audio Spectral** — MFCC/Centroid analysis
    """)

# --- Main Upload Area ---
tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Video URL"])

temp_path = "temp_target.mp4"
video_name = ""
ready_to_analyze = False

with tab1:
    uploaded_file = st.file_uploader("Upload Target Video File (MP4)", type=["mp4"])
    if uploaded_file is not None:
        video_name = uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"File '{video_name}' uploaded successfully.")
        ready_to_analyze = True

with tab2:
    video_url = st.text_input("Paste Video URL (YouTube, Vimeo, or Direct Link)")
    if video_url:
        if st.button("📥 FETCH VIDEO"):
            try:
                dl_bar = st.progress(0, text="Initializing download...")
                video_name = download_video_from_url(video_url, temp_path, dl_bar)
                st.session_state['url_video_name'] = video_name
                st.session_state['url_ready'] = True
                st.success(f"Video '{video_name}' fetched successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to fetch video: {e}")

if st.session_state.get('url_ready') and not ready_to_analyze:
    video_name = st.session_state.get('url_video_name', 'URL Video')
    st.info(f"Ready to analyze: **{video_name}**")
    ready_to_analyze = True

if ready_to_analyze:
    st.success("Target video initialized. System engines ready.")

    if st.button("🚀 RUN FORENSIC ANALYSIS"):
        # ─── Step 1: ML Classification ───
        st.info("🧠 Running ML-powered multi-signal analysis...")

        ml_bar = st.progress(0, text="Extracting forensic features — 0%")
        def ml_cb(p):
            ml_bar.progress(p, text=f"Extracting forensic features — {int(p*100)}%")

        ml_result = predict_video(temp_path, progress_callback=ml_cb)
        ml_bar.progress(1.0, text="Feature extraction complete!")

        # ─── Step 2: Visual Analysis (for video overlay) ───
        vision_bar = st.progress(0, text="Generating EAR overlay video — 0%")
        def vision_cb(p):
            vision_bar.progress(p, text=f"Generating EAR overlay video — {int(p*100)}%")

        vision_results = process_video_vision(temp_path, "output_vision.mp4", vision_cb)
        vision_bar.progress(1.0, text="Vision overlay complete!")

        # ─── Step 3: Audio-Visual Sync (for video overlay) ───
        audio_bar = st.progress(0, text="Generating Lip-Sync overlay video — 0%")
        def audio_cb(p):
            audio_bar.progress(p, text=f"Generating Lip-Sync overlay video — {int(p*100)}%")

        audio_results = process_video_audio_sync(temp_path, "output_audio.mp4", audio_cb)
        audio_bar.progress(1.0, text="Audio-Visual overlay complete!")

        # ─── Step 4: Generate Report ───
        with st.spinner("Generating Forensic Report..."):
            pdf_path = generate_report(vision_results, audio_results, video_name, ml_result=ml_result)

        # ═══════════════════════════════════════════════════
        #                    RESULTS DISPLAY
        # ═══════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🔎 Analysis Results")

        # ─── ML VERDICT (Primary) ───
        prediction = ml_result["prediction"]
        confidence = ml_result["confidence"]
        method = ml_result["method"]
        fake_prob = ml_result["fake_probability"]
        real_prob = ml_result["real_probability"]

        st.markdown("### 🧠 ML Classifier Verdict")

        verdict_col1, verdict_col2, verdict_col3 = st.columns(3)
        with verdict_col1:
            st.metric("Prediction", prediction)
        with verdict_col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with verdict_col3:
            model_acc = ml_result.get("model_accuracy", 0)
            st.metric("Model Accuracy", f"{model_acc*100:.1f}%" if model_acc > 0 else "Heuristic")

        # Probability bars
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.markdown(f"**Authentic Probability:** {real_prob*100:.1f}%")
            st.progress(real_prob)
        with prob_col2:
            st.markdown(f"**Synthetic Probability:** {fake_prob*100:.1f}%")
            st.progress(fake_prob)

        # Final Verdict Banner
        if prediction == "AUTHENTIC":
            st.success("# ✅ LIKELY AUTHENTIC MEDIA ✅")
            st.caption(f"The ML model ({method}) classifies this media as genuine with {confidence*100:.1f}% confidence.")
        else:
            st.error("# 🚨 SYNTHETIC MEDIA DETECTED 🚨")
            st.caption(f"The ML model ({method}) classifies this media as AI-generated with {confidence*100:.1f}% confidence.")

        st.markdown("---")

        # ─── Feature Breakdown ───
        if ml_result.get("contributions"):
            st.markdown("### 📊 Feature Importance Breakdown")
            st.caption("Which forensic signals contributed most to the verdict:")

            contrib = ml_result["contributions"]
            sorted_contrib = sorted(contrib.items(), key=lambda x: x[1]["importance"], reverse=True)

            # Display top features
            feature_labels = {
                "blinks_per_minute": "👁️ Blink Rate (blinks/min)",
                "ear_mean": "👁️ Average Eye Openness",
                "ear_std": "👁️ Eye Variance (EAR σ)",
                "ear_entropy": "👁️ Blink Pattern Complexity",
                "landmark_jitter": "📍 Facial Landmark Jitter",
                "face_consistency": "👤 Face Detection Consistency",
                "lip_distance_mean": "👄 Avg Lip Distance",
                "lip_distance_std": "👄 Lip Movement Variance",
                "lbp_uniformity_mean": "🧱 Skin Texture Uniformity (LBP)",
                "lbp_uniformity_std": "🧱 Texture Consistency (LBP σ)",
                "dct_high_freq_ratio": "🔬 High-Freq DCT Energy",
                "dct_mid_freq_ratio": "🔬 Mid-Freq DCT Energy",
                "optical_flow_mean": "🌊 Optical Flow (mean)",
                "optical_flow_std": "🌊 Optical Flow Variance",
                "spectral_centroid_std": "🎵 Spectral Centroid Variance",
                "zcr_mean": "🎵 Zero Crossing Rate",
                "mfcc_variance": "🎵 MFCC Variance",
                "spectral_rolloff_std": "🎵 Spectral Rolloff Variance",
            }

            for name, info in sorted_contrib[:10]:
                label = feature_labels.get(name, name)
                imp = info["importance"]
                val = info["value"]
                st.markdown(f"""<div class="feature-card">
                    <b>#{info['rank']} {label}</b><br>
                    Importance: <b>{imp*100:.1f}%</b> | Value: <code>{val:.4f}</code>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ─── Individual Engine Results ───
        st.markdown("### 🔬 Individual Engine Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👁️ Biological Blink Rate")
            bpm = vision_results["blinks_per_minute"]
            v_conf = vision_results["confidence_score"]
            ear_std = vision_results.get("ear_std_dev", 0)

            if v_conf < 0.3:
                st.success(f"**Blinks per minute:** {bpm:.1f} (Normal)")
            elif v_conf < 0.6:
                st.warning(f"**Blinks per minute:** {bpm:.1f} (Borderline)")
            else:
                st.error(f"**Blinks per minute:** {bpm:.1f} (Anomalous)")

            st.metric("Vision Suspicion Score", f"{v_conf*100:.0f}%")
            st.caption(f"EAR Variance: {ear_std:.4f}")
            st.video("output_vision.mp4")

        with col2:
            st.markdown("#### 👄 Phoneme-Viseme Sync")
            anomalies = audio_results["anomalies_count"]
            a_conf = audio_results.get("confidence_score", 0.0)

            if a_conf < 0.3:
                st.success(f"**Desync Anomalies:** {anomalies} (Normal)")
            elif a_conf < 0.6:
                st.warning(f"**Desync Anomalies:** {anomalies} (Borderline)")
            else:
                st.error(f"**Desync Anomalies:** {anomalies} (Anomalous)")

            st.metric("Audio Suspicion Score", f"{a_conf*100:.0f}%")
            st.video("output_audio.mp4")

        # ─── Raw Feature Values (expandable) ───
        with st.expander("📋 Raw Feature Values"):
            features = ml_result.get("features", {})
            feature_data = {k: f"{v:.6f}" if isinstance(v, float) else str(v)
                          for k, v in features.items()
                          if k not in ["ear_values", "total_frames"]}
            st.json(feature_data)

        # ─── Download Report ───
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📄 Download Full Forensic PDF Report",
            data=pdf_bytes,
            file_name="Deepfake_Guardian_Report.pdf",
            mime="application/pdf"
        )
