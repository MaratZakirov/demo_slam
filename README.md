### demo_slam

A minimal Python pipeline for **3D point cloud generation** from video frames using optical flow, Structure from Motion (SfM), and stereo rectification.

### 🚀 Live 3D Interactive Demo

Interact with the reconstructed 3D scene directly in your browser without downloading any files:

👉 **[VIEW INTERACTIVE 3D POINT CLOUD](https://maratzakirov.github.io/demo_slam/)**

*Controls: Left Click to rotate, Right Click to pan, Scroll to zoom.*

### 🛠️ Features

* **Feature Tracking:** Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK) instead of global SIFT features.
* **Camera Pose Estimation:** Essential matrix parsing with RANSAC filtering.
* **Stereo Rectification:** Reprojects stereo frame pairs onto a common plane.
* **Disparity Mapping:** Custom vectorized subpixel NumPy SAD (Sum of Absolute Differences) algorithm alongside OpenCV BM/SGBM options.
* **3D Output:** Fast matrix-based export to .ply format.

### 💻 Quick Start

### 1. Install Dependencies

bash

pip install opencv-python numpy matplotlib

Use code with caution.

### 2. Run the Pipeline

Ensure your source video is placed at sample/kitchen.mp4, then execute:

python main.py

The script will process the video stream and export the final 3D point cloud directly to **myscene.ply**.
To view it you may use online tool like https://swyvl.io/tools/point-cloud-viewer/