# Deepfake Guardian 2026 - Guide for Innovation Fair Judges

Welcome to the Deepfake Guardian 2026 exhibit! This document explains the underlying science and mathematical principles behind our biological and multimodal inconsistency analysis engines.

## 1. The Blink Rate Analyzer (The 'Lyu-Siwei' Method)

Generative AI models, such as GANs and Diffusion models, often struggle with physiological nuances—primarily the frequency and natural physics of human blinks. Our Vision Engine uses the **Eye Aspect Ratio (EAR)** to track blinks mathematically frame by frame.

### The Mathematics of EAR
For each eye, Mediapipe extracts 6 specific 2D facial landmarks ($p_1$ through $p_6$).

The Eye Aspect Ratio (EAR) formula is:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}$$

- **Numerator:** Computes the distance between the vertical eye landmarks.
- **Denominator:** Computes the distance between horizontal eye landmarks.
- When the eye is open, the EAR stays roughly constant. When the eye closes during a blink, the numerator drops rapidly to nearly zero, causing the EAR to plummet.

**Detection Logic:** A normal human blinks between 10 and 30 times per minute. If the EAR values indicate a blink rate outside this biological baseline, the video is flagged as anomalous.

## 2. Phoneme-Viseme Sync (The 'Lip-Lock' Method)

Perfectly synchronizing generated audio with the visual shape of the mouth (**viseme**) is extremely difficult for AI in 2026.

We focus on bilabial phonemes: "M", "B", and "P".
- **Phoneme Analysis:** The audio engine (using Librosa) identifies bursts of acoustic energy that signify bilabial plosives.
- **Viseme Analysis:** The vision engine concurrently measures the distance between the upper and lower inner lips.
- **Detection Logic:** To produce an "M", "B", or "P", a person MUST close their lips. If the audio produces a bilabial phoneme but the visual data shows the lips are not fully closed (a common AI lip-sync error), a manipulation flag is triggered.

## Why this approach is robust:
- **Technical Depth:** It relies on hard mathematics and unalterable human physiology rather than black-box AI detection algorithms, which can be fooled.
- **Explainability:** When a deepfake is flagged, the system produces a forensic PDF explaining *exactly* why (e.g., "The AI failed to close the lips on the 'B' sound").
