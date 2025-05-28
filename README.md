# Unspoken_Words: Real - Time ASL Recognition with Hand Pose Feedback

This project implements a real-time American Sign Language (ASL) recognition system using YOLOv8 for hand detection, a seondary classifier for ASL alphabet recognition, and MediaPipe for hand pose tracking and comparision with reference poses

# Features
- Real-time webcam-based ASL hand sign detection
- Dual-mode operation:
  - Free mode: Predict any detected sign
  - Guided mode: Practice a specific letter with feedback
- Gaussian blur for reducing background noise
- Visual pose overlay comparing user's hand to the ideal reference
- Buffered prediction with confidence smoothing to reduce flickering

# Requirements
- Python 3.10
- Webcam
- Optional: CUDA-compatible GPU

# Dependencies:
Intsall required libraries with:
```bash
pip install opencv-python numpy mediapipe ultralytics